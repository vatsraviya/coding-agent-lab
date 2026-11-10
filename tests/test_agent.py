"""Repair-loop tests using ScriptedClient and fake tools, never Docker or evals."""

from __future__ import annotations

import json
from pathlib import Path

from adapters.models.scripted import ScriptedClient
from core.models import RepairResult, TestResult
from core.repair_agent import DECISION_BUDGET, RepairAgent


class FakeTools:
    """In-memory tools that record calls without touching Docker or evals/."""

    def __init__(
        self,
        files: dict[str, str] | None = None,
        test_results: list[TestResult] | None = None,
    ) -> None:
        self.workspace = Path("fake-workspace")
        self.files = dict(files or {"notes.py": "x = 1\n"})
        self.test_results = list(test_results or [])
        self.changed_files: list[str] = []
        self.read_calls: list[str] = []
        self.edit_calls: list[tuple[str, str]] = []
        self.test_calls = 0

    def read_file(self, path: str) -> str:
        self.read_calls.append(path)
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    def edit_file(self, path: str, content: str) -> None:
        self.edit_calls.append((path, content))
        self.files[path] = content
        if path not in self.changed_files:
            self.changed_files.append(path)

    def run_tests(self) -> TestResult:
        self.test_calls += 1
        return self.test_results.pop(0)


def _action(tool: str, **fields: str) -> str:
    return json.dumps({"tool": tool, **fields})


def _failing() -> TestResult:
    return TestResult(passed=False, output="failed", returncode=1)


def _passing() -> TestResult:
    return TestResult(passed=True, output="passed", returncode=0)


def _events(result: RepairResult) -> list[str]:
    return [entry["event"] for entry in result.trace]


def test_successful_read_failing_test_edit_passing_test() -> None:
    tools = FakeTools(
        files={"calculator.py": "def add(a, b):\n    return a - b\n"},
        test_results=[_failing(), _passing()],
    )
    client = ScriptedClient(
        [
            _action("read_file", path="calculator.py", summary="Read the source"),
            _action("run_tests", summary="Reproduce the failure"),
            _action("edit_file", path="calculator.py", content="def add(a, b):\n    return a + b\n"),
            _action("run_tests", summary="Verify the repair"),
        ]
    )

    result = RepairAgent(client, tools).run("arithmetic", ["calculator.py"])

    assert result.passed is True
    assert tools.read_calls == ["calculator.py"]
    assert tools.edit_calls[0][0] == "calculator.py"
    assert tools.test_calls == 2
    assert result.changed_files == ["calculator.py"]
    assert result.last_test_result == _passing()
    assert _events(result)[0] == "repair_started"
    assert _events(result)[-1] == "repair_finished"
    assert "model_response" in _events(result)
    assert "tool_result" in _events(result)
    summaries = [entry.get("summary") for entry in result.trace if entry["event"] == "model_response"]
    assert "Read the source" in summaries
    assert any("reasoning" not in entry for entry in result.trace)


def test_invalid_json_calls_no_tool() -> None:
    tools = FakeTools(test_results=[_failing()])
    client = ScriptedClient(["this is not json"])

    result = RepairAgent(client, tools).run("arithmetic", ["notes.py"])

    assert isinstance(result, RepairResult)
    assert result.passed is False
    assert tools.read_calls == []
    assert tools.edit_calls == []
    assert tools.test_calls == 0
    assert result.trace[-1]["reason"].startswith("invalid reply")


def test_edit_rejected_before_failing_test() -> None:
    tools = FakeTools(test_results=[_failing()])
    client = ScriptedClient(
        [
            _action("edit_file", path="notes.py", content="x = 2\n"),
            "not-json",
        ]
    )

    result = RepairAgent(client, tools).run("arithmetic", ["notes.py"])

    assert result.passed is False
    assert tools.edit_calls == []
    observations = [
        entry["observation"]
        for entry in result.trace
        if entry["event"] == "tool_result"
    ]
    assert any(text.startswith("policy-error") for text in observations)


def test_budget_exhaustion() -> None:
    tools = FakeTools(files={"notes.py": "x = 1\n"})
    client = ScriptedClient(
        [_action("read_file", path="notes.py") for _ in range(DECISION_BUDGET)]
    )

    result = RepairAgent(client, tools).run("arithmetic", ["notes.py"])

    assert result.passed is False
    assert len(tools.read_calls) == DECISION_BUDGET
    assert result.trace[-1]["reason"] == "decision budget exhausted"


def test_passing_baseline_is_not_a_repair() -> None:
    tools = FakeTools(test_results=[_passing()])
    client = ScriptedClient([_action("run_tests")])

    result = RepairAgent(client, tools).run("arithmetic", ["notes.py"])

    assert result.passed is False
    assert tools.test_calls == 1
    assert tools.edit_calls == []
    assert "no reproducible bug" in result.trace[-1]["reason"]
    assert result.last_test_result == _passing()


def test_client_error_still_returns_repair_result() -> None:
    tools = FakeTools()
    client = ScriptedClient([])

    result = RepairAgent(client, tools).run("arithmetic", ["notes.py"])

    assert isinstance(result, RepairResult)
    assert result.passed is False
    assert result.trace[-1]["reason"].startswith("client error")
    assert tools.read_calls == []
    assert tools.test_calls == 0
