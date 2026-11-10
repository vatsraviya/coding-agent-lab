"""Bounded repair loop: one model decision, one policy check, one safe tool."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.models import ModelAction, RepairResult, TestResult
from core.ports import ChatClient, RepairToolsPort, TraceRecorder
from core.prompts import case_request, system_instruction, tool_result_message

DECISION_BUDGET = 8
POLICY_ERROR = (
    "policy-error: an edit is not allowed until this repair has one successful "
    "read_file observation and one failing run_tests observation."
)

_ALLOWED_FIELDS = {
    "read_file": frozenset({"tool", "path", "summary"}),
    "run_tests": frozenset({"tool", "summary"}),
    "edit_file": frozenset({"tool", "path", "content", "summary"}),
}
_REQUIRED_FIELDS = {
    "read_file": frozenset({"tool", "path"}),
    "run_tests": frozenset({"tool"}),
    "edit_file": frozenset({"tool", "path", "content"}),
}


class InvalidReply(ValueError):
    """Raised when a model reply is not exactly one permitted JSON action."""


class RepairAgent:
    """Traffic controller that validates one action per decision and stops safely."""

    def __init__(
        self,
        client: ChatClient,
        tools: RepairToolsPort,
        recorder: TraceRecorder | None = None,
    ) -> None:
        self._client = client
        self._tools = tools
        self._recorder = recorder

    def _emit(self, trace: list[dict[str, Any]], entry: dict[str, Any]) -> None:
        """Append one event to the stored trace and notify any live recorder.

        The stored trace is unchanged; the optional recorder only observes, so
        repair logic and outcomes stay identical whether or not one is present.
        """

        trace.append(entry)
        if self._recorder is not None:
            payload = {key: value for key, value in entry.items() if key != "event"}
            self._recorder.record(entry["event"], payload)

    def run(self, case_name: str, relative_files: list[str] | None = None) -> RepairResult:
        """Run one bounded repair attempt and return a result in every outcome."""

        files = list(relative_files or [])
        messages = [
            {"role": "system", "content": system_instruction()},
            {"role": "user", "content": case_request(case_name, files)},
        ]
        trace: list[dict[str, Any]] = []
        self._emit(trace, {"event": "repair_started", "case": case_name})
        seen_read = False
        seen_failing_test = False
        seen_edit = False
        last_test: TestResult | None = None

        for _decision in range(DECISION_BUDGET):
            try:
                raw = self._client.chat(messages)
            except Exception as exc:
                return self._finish(
                    trace,
                    passed=False,
                    reason=f"client error: {exc}",
                    last_test=last_test,
                )

            try:
                action = parse_model_reply(raw)
            except InvalidReply as exc:
                self._emit(
                    trace,
                    {
                        "event": "model_response",
                        "valid": False,
                        "summary": None,
                        "error": str(exc),
                    },
                )
                return self._finish(
                    trace,
                    passed=False,
                    reason=f"invalid reply: {exc}",
                    last_test=last_test,
                )

            self._emit(
                trace,
                {
                    "event": "model_response",
                    "valid": True,
                    "tool": action.tool,
                    "summary": action.summary,
                },
            )
            messages.append({"role": "assistant", "content": raw})

            if action.tool == "edit_file" and not (seen_read and seen_failing_test):
                observation = POLICY_ERROR
                self._emit(
                    trace,
                    {
                        "event": "tool_result",
                        "tool": action.tool,
                        "observation": observation,
                    },
                )
                messages.append(
                    {"role": "user", "content": tool_result_message(observation)}
                )
                continue

            try:
                observation, last_test, seen_read, seen_failing_test, seen_edit = (
                    self._execute(
                        action,
                        last_test=last_test,
                        seen_read=seen_read,
                        seen_failing_test=seen_failing_test,
                        seen_edit=seen_edit,
                    )
                )
            except Exception as exc:
                observation = f"tool error: {exc}"
                self._emit(
                    trace,
                    {
                        "event": "tool_result",
                        "tool": action.tool,
                        "observation": observation,
                    },
                )
                messages.append(
                    {"role": "user", "content": tool_result_message(observation)}
                )
                continue

            self._emit(
                trace,
                {
                    "event": "tool_result",
                    "tool": action.tool,
                    "observation": observation,
                    "summary": action.summary,
                },
            )
            messages.append({"role": "user", "content": tool_result_message(observation)})

            if action.tool == "run_tests" and last_test is not None:
                if last_test.passed and seen_edit:
                    return self._finish(
                        trace,
                        passed=True,
                        reason="tests passed after an evidence-ordered edit",
                        last_test=last_test,
                    )
                if last_test.passed and not seen_edit:
                    return self._finish(
                        trace,
                        passed=False,
                        reason="no reproducible bug: tests passed before any edit",
                        last_test=last_test,
                    )

        return self._finish(
            trace,
            passed=False,
            reason="decision budget exhausted",
            last_test=last_test,
        )

    def _execute(
        self,
        action: ModelAction,
        *,
        last_test: TestResult | None,
        seen_read: bool,
        seen_failing_test: bool,
        seen_edit: bool,
    ) -> tuple[str, TestResult | None, bool, bool, bool]:
        if action.tool == "read_file":
            content = self._tools.read_file(str(action.arguments["path"]))
            return content, last_test, True, seen_failing_test, seen_edit
        if action.tool == "run_tests":
            result = self._tools.run_tests()
            observation = (
                f"passed={result.passed}\n"
                f"returncode={result.returncode}\n"
                f"{result.output}"
            )
            failing = seen_failing_test or not result.passed
            return observation, result, seen_read, failing, seen_edit
        content = str(action.arguments["content"])
        path = str(action.arguments["path"])
        self._tools.edit_file(path, content)
        return f"edited {path}", last_test, seen_read, seen_failing_test, True

    def _finish(
        self,
        trace: list[dict[str, Any]],
        *,
        passed: bool,
        reason: str,
        last_test: TestResult | None,
    ) -> RepairResult:
        self._emit(trace, {"event": "repair_finished", "passed": passed, "reason": reason})
        workspace = getattr(self._tools, "workspace", None)
        changed_files = list(getattr(self._tools, "changed_files", []))
        return RepairResult(
            passed=passed,
            trace=trace,
            changed_files=changed_files,
            workspace=workspace if isinstance(workspace, Path) else None,
            last_test_result=last_test,
        )


def parse_model_reply(raw: str) -> ModelAction:
    """Parse a raw model reply as exactly one permitted JSON action object."""

    text = raw.strip()
    if not text:
        raise InvalidReply("empty reply")
    if "```" in text:
        raise InvalidReply("Markdown fences are not allowed")
    try:
        parsed = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise InvalidReply(f"malformed JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise InvalidReply("reply must be a single JSON object")

    tool = parsed.get("tool")
    if tool not in _ALLOWED_FIELDS:
        raise InvalidReply(f"unknown tool: {tool!r}")
    keys = frozenset(parsed)
    missing = _REQUIRED_FIELDS[tool] - keys
    if missing:
        raise InvalidReply(f"missing fields for {tool}: {sorted(missing)}")
    extra = keys - _ALLOWED_FIELDS[tool]
    if extra:
        raise InvalidReply(f"fields not allowed for {tool}: {sorted(extra)}")

    if "path" in parsed and not isinstance(parsed["path"], str):
        raise InvalidReply("path must be a string")
    if "content" in parsed and not isinstance(parsed["content"], str):
        raise InvalidReply("content must be a string")
    summary = parsed.get("summary")
    if summary is not None and not isinstance(summary, str):
        raise InvalidReply("summary must be a string")

    arguments = {key: value for key, value in parsed.items() if key not in {"tool", "summary"}}
    return ModelAction(tool=tool, arguments=arguments, summary=summary)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise InvalidReply(f"duplicate key: {key}")
        seen.add(key)
        result[key] = value
    return result
