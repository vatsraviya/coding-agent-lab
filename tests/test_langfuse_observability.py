"""Tests for the optional Langfuse observability extension. No network calls.

A fake Langfuse client records the sequence of SDK calls, so these tests prove
event order, that the root is closed and flushed on repair_finished, and that
recorder failures are best-effort (they never raise or change a repair). No
test ever contacts Langfuse Cloud, and no credentials are read.
"""

from __future__ import annotations

import json
from pathlib import Path

from adapters.observability.langfuse import LangfuseTrajectoryRecorder
from core.models import TestResult
from core.repair_agent import RepairAgent


# --- Fake Langfuse client ---------------------------------------------------

class FakeObservation:
    """A fake span/observation that logs its lifecycle to a shared list."""

    def __init__(self, log: list[str], name: str) -> None:
        self._log = log
        self._name = name

    def start_span(self, name: str, metadata: dict | None = None, input: dict | None = None):
        self._log.append(f"start:{name}")
        return FakeObservation(self._log, name)

    def update(self, output: dict | None = None) -> None:
        self._log.append(f"update:{self._name}")

    def end(self) -> None:
        self._log.append(f"end:{self._name}")


class FakeClient:
    """A fake Langfuse client; never contacts the network."""

    def __init__(self) -> None:
        self.log: list[str] = []

    def start_span(self, name: str, metadata: dict | None = None, input: dict | None = None):
        self.log.append(f"start:{name}")
        return FakeObservation(self.log, name)

    def flush(self) -> None:
        self.log.append("flush")


class BrokenClient:
    """A fake client whose every call raises, to prove best-effort handling."""

    def start_span(self, *args, **kwargs):
        raise RuntimeError("langfuse down")

    def flush(self) -> None:
        raise RuntimeError("langfuse down")


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
    """Return pre-written replies in order (local; no network)."""

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
    return TestResult(passed=False, output="boom\nE assert", returncode=1)


def _passing() -> TestResult:
    return TestResult(passed=True, output="1 passed", returncode=0)


# --- Event order and flush --------------------------------------------------

def test_event_order_root_then_children_then_flush() -> None:
    client = FakeClient()
    recorder = LangfuseTrajectoryRecorder(client=client)

    recorder.record("repair_started", {"case": "arithmetic"})
    recorder.record("model_response", {"valid": True, "tool": "read_file", "summary": "r"})
    recorder.record("tool_result", {"tool": "read_file", "observation": "secret source", "summary": "r"})
    recorder.record("repair_finished", {"passed": True, "reason": "done"})

    assert client.log == [
        "start:repair-agent",       # root opened first
        "start:model_response",     # child opened
        "update:model_response",
        "end:model_response",       # child closed
        "start:tool_result",
        "update:tool_result",
        "end:tool_result",
        "update:repair-agent",      # root updated with outcome
        "end:repair-agent",         # root closed
        "flush",                    # flushed exactly once, at the end
    ]
    assert client.log.count("flush") == 1


def test_no_source_or_test_output_is_sent() -> None:
    client = FakeClient()
    recorder = LangfuseTrajectoryRecorder(client=client)
    # read_file observation holds source content; it must never be logged.
    recorder.record("repair_started", {"case": "c"})
    recorder.record("tool_result", {"tool": "read_file", "observation": "TOP SECRET SOURCE"})
    recorder.record("repair_finished", {"passed": False, "reason": "x"})
    assert "TOP SECRET SOURCE" not in json.dumps(client.log)


# --- Best-effort failure handling ------------------------------------------

def test_recorder_swallows_client_failures() -> None:
    recorder = LangfuseTrajectoryRecorder(client=BrokenClient())
    # None of these must raise.
    recorder.record("repair_started", {"case": "c"})
    recorder.record("model_response", {"valid": True, "tool": "run_tests"})
    recorder.record("tool_result", {"tool": "run_tests", "observation": "passed=False\nreturncode=1\n..."})
    recorder.record("repair_finished", {"passed": False, "reason": "x"})


def test_broken_recorder_does_not_change_repair_result() -> None:
    recorder = LangfuseTrajectoryRecorder(client=BrokenClient())
    tools = FakeTools(
        {"calculator.py": "def add(a, b):\n    return a - b\n"},
        [_failing(), _passing()],
    )
    client = ScriptedClient(
        [
            _act("read_file", path="calculator.py", summary="Read"),
            _act("run_tests", summary="Reproduce"),
            _act("edit_file", path="calculator.py", content="def add(a, b):\n    return a + b\n"),
            _act("run_tests", summary="Verify"),
        ]
    )

    result = RepairAgent(client, tools, recorder=recorder).run("arithmetic", ["calculator.py"])

    # The repair still succeeds and the stored trace is intact.
    assert result.passed is True
    assert result.changed_files == ["calculator.py"]
    assert result.trace[0]["event"] == "repair_started"
    assert result.trace[-1]["event"] == "repair_finished"


def test_full_run_flushes_once_and_reports_changed_file() -> None:
    client = FakeClient()
    recorder = LangfuseTrajectoryRecorder(client=client)
    tools = FakeTools(
        {"calculator.py": "def add(a, b):\n    return a - b\n"},
        [_failing(), _passing()],
    )
    scripted = ScriptedClient(
        [
            _act("read_file", path="calculator.py", summary="Read"),
            _act("run_tests", summary="Reproduce"),
            _act("edit_file", path="calculator.py", content="def add(a, b):\n    return a + b\n"),
            _act("run_tests", summary="Verify"),
        ]
    )

    result = RepairAgent(scripted, tools, recorder=recorder).run("arithmetic", ["calculator.py"])

    assert result.passed is True
    assert client.log.count("flush") == 1
    assert client.log[0] == "start:repair-agent"
    assert client.log[-1] == "flush"
    # The changed relative file name is tracked internally for the root payload.
    assert recorder._changed == ["calculator.py"]
