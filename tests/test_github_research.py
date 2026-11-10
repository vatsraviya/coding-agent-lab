"""Tests for the optional GitHub-research extension. Never contacts GitHub.

The adapter is exercised with a mocked HTTP opener and the agent with a fake
research port, so no network call is ever made. These tests prove that disabled
and early research make no call, that query and request/result budgets are
enforced, that private results are discarded, that a token never leaks into the
record, and that an allowed result keeps its GitHub source URL in the visible
observation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adapters.research.github_research import (
    MAX_QUERY_LENGTH,
    MAX_SNIPPET,
    GitHubResearch,
)
from core.models import TestResult
from core.repair_agent import RepairAgent, parse_model_reply


# --- Test doubles -----------------------------------------------------------

class _Response:
    """A minimal context-manager HTTP response wrapping a JSON payload."""

    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


class FakeOpener:
    """A stand-in for urlopen that records calls and returns a fixed payload."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0

    def __call__(self, request: object, timeout: float | None = None) -> _Response:
        self.calls += 1
        return _Response(self.payload)


class FakeResearch:
    """A fake ResearchPort that counts calls and returns a canned record."""

    def __init__(self, record: dict | None = None) -> None:
        self.calls = 0
        self.record = record or {
            "query": "Python median even length",
            "retrieved_at": "2026-01-01T00:00:00+00:00",
            "sources": [
                {
                    "repository": "octo/stats",
                    "path": "stats/median.py",
                    "url": "https://github.com/octo/stats/blob/main/stats/median.py",
                    "snippet": "def median(values): ...",
                }
            ],
        }

    def search_github(self, query: str) -> dict:
        self.calls += 1
        return dict(self.record)


class FakeTools:
    """In-memory tools, never touching Docker or evals/."""

    def __init__(self, files: dict[str, str], test_results: list[TestResult]) -> None:
        self.workspace = Path("fake-workspace")
        self.files = dict(files)
        self.test_results = list(test_results)
        self.changed_files: list[str] = []

    def read_file(self, path: str) -> str:
        return self.files[path]

    def edit_file(self, path: str, content: str) -> None:
        self.files[path] = content
        if path not in self.changed_files:
            self.changed_files.append(path)

    def run_tests(self) -> TestResult:
        return self.test_results.pop(0)


class ScriptedClient:
    """Return pre-written replies in order (local copy; no network)."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self._index = 0

    def chat(self, messages: list[dict[str, str]]) -> str:
        reply = self._replies[self._index]
        self._index += 1
        return reply


def _act(tool: str, **fields: str) -> str:
    return json.dumps({"tool": tool, **fields})


def _failing() -> TestResult:
    return TestResult(passed=False, output="failed", returncode=1)


def _passing() -> TestResult:
    return TestResult(passed=True, output="passed", returncode=0)


def _source(item: dict) -> dict:
    """One raw GitHub code-search item with a text-match fragment."""

    base = {
        "repository": {"full_name": "octo/stats", "private": False},
        "path": "stats/median.py",
        "html_url": "https://github.com/octo/stats/blob/main/stats/median.py",
        "text_matches": [{"fragment": "def median(values): ..."}],
    }
    base.update(item)
    return base


# --- Adapter: query validation ---------------------------------------------

@pytest.mark.parametrize(
    "bad",
    ["", "   ", "line one\nline two", "line one\r\nline two", "x" * (MAX_QUERY_LENGTH + 1)],
)
def test_rejects_bad_queries_without_any_request(bad: str) -> None:
    opener = FakeOpener({"items": [_source({})]})
    research = GitHubResearch(token="secret", opener=opener)
    with pytest.raises(ValueError):
        research.search_github(bad)
    assert opener.calls == 0


# --- Adapter: request budget ------------------------------------------------

def test_request_budget_is_three() -> None:
    opener = FakeOpener({"items": [_source({})]})
    research = GitHubResearch(token="secret", opener=opener)
    for _ in range(3):
        research.search_github("median even length")
    with pytest.raises(ValueError):
        research.search_github("median even length")
    assert opener.calls == 3


# --- Adapter: result budget + fields ---------------------------------------

def test_result_budget_is_three_and_fields_are_minimal() -> None:
    items = [_source({"path": f"f{i}.py"}) for i in range(5)]
    opener = FakeOpener({"items": items})
    research = GitHubResearch(token="secret", opener=opener)

    record = research.search_github("median even length")

    assert set(record) == {"query", "retrieved_at", "sources"}
    assert len(record["sources"]) == 3  # capped at three
    for source in record["sources"]:
        assert set(source) == {"repository", "path", "url", "snippet"}
    # The next request's result budget is now exhausted.
    with pytest.raises(ValueError):
        research.search_github("median even length")


def test_snippet_is_capped_and_token_never_leaks() -> None:
    long_fragment = "x" * (MAX_SNIPPET + 100)
    opener = FakeOpener({"items": [_source({"text_matches": [{"fragment": long_fragment}]})]})
    research = GitHubResearch(token="super-secret-token", opener=opener)

    record = research.search_github("median even length")

    assert len(record["sources"][0]["snippet"]) == MAX_SNIPPET
    assert "super-secret-token" not in json.dumps(record)


def test_private_results_are_discarded() -> None:
    items = [
        _source({"repository": {"full_name": "octo/secret", "private": True}}),
        _source({"repository": {"full_name": "octo/public", "private": False}}),
    ]
    opener = FakeOpener({"items": items})
    research = GitHubResearch(token="secret", opener=opener)

    record = research.search_github("median even length")

    repositories = [source["repository"] for source in record["sources"]]
    assert repositories == ["octo/public"]


# --- Parser gating ----------------------------------------------------------

def test_parser_rejects_search_when_research_disabled() -> None:
    reply = _act("search_github", query="median even length", summary="x")
    with pytest.raises(Exception):
        parse_model_reply(reply, allow_research=False)


def test_parser_accepts_search_when_research_enabled() -> None:
    reply = _act("search_github", query="median even length", summary="x")
    action = parse_model_reply(reply, allow_research=True)
    assert action.tool == "search_github"
    assert action.arguments == {"query": "median even length"}


# --- Agent: disabled and early research make no call ------------------------

def test_disabled_research_makes_no_call() -> None:
    research = FakeResearch()
    tools = FakeTools({"m.py": "x = 1\n"}, [_failing()])
    client = ScriptedClient([_act("search_github", query="median", summary="x")])

    # research=None: the parser never accepts search_github.
    result = RepairAgent(client, tools, research=None).run("median", ["m.py"])

    assert result.passed is False
    assert research.calls == 0
    assert result.trace[-1]["reason"].startswith("invalid reply")


def test_early_search_is_blocked_with_no_call() -> None:
    research = FakeResearch()
    tools = FakeTools({"m.py": "x = 1\n"}, [_failing()])
    client = ScriptedClient(
        [
            _act("search_github", query="median even length", summary="too early"),
            "not-json",  # end the run after the policy block
        ]
    )

    result = RepairAgent(client, tools, research=research).run("median", ["m.py"])

    assert research.calls == 0
    observations = [
        entry["observation"]
        for entry in result.trace
        if entry["event"] == "tool_result"
    ]
    assert any(text.startswith("policy-error") for text in observations)


# --- Agent: an allowed result keeps its source URL in the observation -------

def test_allowed_search_preserves_source_url_in_observation() -> None:
    research = FakeResearch()
    url = research.record["sources"][0]["url"]
    tools = FakeTools(
        {"statistics_tools.py": "def median(v):\n    return sorted(v)[len(v)//2]\n"},
        [_failing(), _passing()],
    )
    client = ScriptedClient(
        [
            _act("read_file", path="statistics_tools.py", summary="Read source"),
            _act("run_tests", summary="Reproduce"),
            _act("search_github", query="Python median even length", summary="Find ref"),
            _act("edit_file", path="statistics_tools.py", content="def median(v): ...\n"),
            _act("run_tests", summary="Verify"),
        ]
    )

    result = RepairAgent(client, tools, research=research).run(
        "median", ["statistics_tools.py"]
    )

    assert research.calls == 1
    search_results = [
        entry
        for entry in result.trace
        if entry["event"] == "tool_result" and entry.get("tool") == "search_github"
    ]
    assert len(search_results) == 1
    assert url in search_results[0]["observation"]
    # Research never finishes a repair; only the later passing test does.
    assert result.passed is True
