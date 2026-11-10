"""Record a repair trajectory to Langfuse Cloud as a small, public payload.

``LangfuseTrajectoryRecorder`` implements the ``TraceRecorder`` contract from
``core/ports.py`` (core never imports Langfuse). For one repair it opens a root
agent observation, adds one closed child observation per model and tool event,
then closes and flushes the root when the repair finishes.

Only a deliberately small, public payload is sent: the case name, a step
number, the requested tool name, the public summary, pass/fail status, return
code, changed relative file names, and the final outcome. Credentials,
environment variables, raw prompts, raw model text, source or replacement file
content, and full test output are never sent.

Every Langfuse call is best-effort: any failure is caught and logged here, so a
failed recording can never change the repair result or stop a repair.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_ENV_VARS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL")


def missing_langfuse_env() -> list[str]:
    """Return the names of any required Langfuse environment variables unset."""

    return [name for name in _ENV_VARS if not os.environ.get(name)]


def _build_client() -> Any:
    """Construct a real Langfuse client from environment credentials."""

    from langfuse import Langfuse  # Imported lazily; core never imports Langfuse.

    return Langfuse(
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
        host=os.environ.get("LANGFUSE_BASE_URL"),
    )


class LangfuseTrajectoryRecorder:
    """Best-effort TraceRecorder that mirrors a repair to Langfuse Cloud."""

    def __init__(self, client: Any | None = None) -> None:
        # The client is built from env credentials unless one is injected
        # (tests inject a fake client, so no network call is ever made).
        self._client = client if client is not None else _build_client()
        self._root: Any | None = None
        self._step = 0
        self._changed: list[str] = []

    def record(self, event: str, payload: dict[str, Any]) -> None:
        """Dispatch one public event to Langfuse, swallowing any failure."""

        try:
            if event == "repair_started":
                self._on_started(payload)
            elif event == "model_response":
                self._on_model(payload)
            elif event == "tool_result":
                self._on_tool(payload)
            elif event == "repair_finished":
                self._on_finished(payload)
        except Exception as exc:  # Best-effort: never break the repair.
            logger.warning("Langfuse recording failed for %s: %s", event, exc)

    # --- event handlers -----------------------------------------------------

    def _on_started(self, payload: dict[str, Any]) -> None:
        """Open the root agent observation for this repair."""

        self._root = self._client.start_span(
            name="repair-agent",
            input={"case": payload.get("case")},
            metadata={"type": "agent"},
        )

    def _on_model(self, payload: dict[str, Any]) -> None:
        """Add a closed child observation for one model decision."""

        self._step += 1
        self._child(
            name="model_response",
            metadata={"step": self._step},
            output={
                "tool": payload.get("tool"),
                "summary": payload.get("summary"),
                "valid": payload.get("valid"),
            },
        )

    def _on_tool(self, payload: dict[str, Any]) -> None:
        """Add a closed child observation for one tool result."""

        self._step += 1
        tool = payload.get("tool")
        output: dict[str, Any] = {"tool": tool, "summary": payload.get("summary")}

        observation = str(payload.get("observation", ""))
        if tool == "run_tests":
            status, return_code = _parse_test_observation(observation)
            output["status"] = status
            output["return_code"] = return_code
        elif tool == "edit_file":
            changed = _changed_file(observation)
            if changed and changed not in self._changed:
                self._changed.append(changed)

        self._child(name="tool_result", metadata={"step": self._step}, output=output)

    def _on_finished(self, payload: dict[str, Any]) -> None:
        """Close the root observation with the final outcome and flush."""

        if self._root is not None:
            self._root.update(
                output={
                    "passed": payload.get("passed"),
                    "outcome": payload.get("reason"),
                    "changed_files": list(self._changed),
                }
            )
            self._root.end()
        self._client.flush()

    # --- helpers ------------------------------------------------------------

    def _child(self, *, name: str, metadata: dict[str, Any], output: dict[str, Any]) -> None:
        """Open and immediately close one child observation under the root."""

        if self._root is None:
            return
        child = self._root.start_span(name=name, metadata=metadata)
        child.update(output=output)
        child.end()


def _parse_test_observation(observation: str) -> tuple[str, int | None]:
    """Read only pass/fail and return code from a run_tests observation.

    The full test output is never read into the payload.
    """

    status = "unknown"
    return_code: int | None = None
    for line in observation.splitlines():
        if line.startswith("passed="):
            status = "pass" if line.strip() == "passed=True" else "fail"
        elif line.startswith("returncode="):
            try:
                return_code = int(line.split("=", 1)[1].strip())
            except ValueError:
                return_code = None
            break
    return status, return_code


def _changed_file(observation: str) -> str:
    """Extract the edited relative file name from an 'edited <path>' result."""

    prefix = "edited "
    if observation.startswith(prefix):
        return observation[len(prefix):].strip()
    return ""
