"""Terminal entry point: start one repair and show it as it happens.

Usage:
    python main.py repair evals/arithmetic --provider scripted

This module only parses options and presents results. All wiring to concrete
tools and clients lives in ``bootstrap.py``; the repair rules live in ``core/``.
A live ``TerminalTrace`` recorder prints labeled START, ACTION, RESULT, and
FINISHED steps as the agent emits them, instead of raw JSON. It only observes:
the stored trace, repair logic, and safety rules are unchanged, and every
evaluation still runs inside the existing Docker sandbox, never on the host.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from bootstrap import build_service


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the selected repair, and return an exit status."""

    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Run a bounded, sandboxed repair on one eval project.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    repair_parser = subparsers.add_parser(
        "repair", help="Repair one selected project under evals/."
    )
    repair_parser.add_argument(
        "project",
        help="Path to the eval project, e.g. evals/arithmetic.",
    )
    repair_parser.add_argument(
        "--provider",
        default="scripted",
        choices=["scripted", "ollama"],
        help="Model provider to drive the repair (default: scripted).",
    )
    repair_parser.add_argument(
        "--model",
        default=None,
        help="Model name for a live provider (default: qwen2.5-coder:7b or $OLLAMA_MODEL).",
    )
    repair_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for a live provider (default: 0.0).",
    )

    subparsers.add_parser(
        "ui", help="Launch the local Gradio page for scripted repairs."
    )

    args = parser.parse_args(argv)

    if args.command == "ui":
        from adapters.gradio_ui import create_demo

        create_demo().launch()
        return 0

    recorder = TerminalTrace()
    try:
        service = build_service(
            args.provider,
            model=args.model,
            temperature=args.temperature,
            recorder=recorder,
        )
        result = service.repair(args.project)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    _print_changed_files(result.changed_files)
    _print_test_result(result.last_test_result)

    return 0 if result.passed else 1


class TerminalTrace:
    """Print labeled repair steps live as the agent emits each event.

    Implements the ``TraceRecorder`` contract. It only displays events; it never
    changes the stored trace or the repair outcome.
    """

    def record(self, event: str, payload: dict[str, Any]) -> None:
        """Render one event as a labeled, human-friendly terminal step."""

        if event == "repair_started":
            self._line("START", f"case={payload.get('case')}")
        elif event == "model_response":
            if payload.get("valid"):
                summary = payload.get("summary")
                detail = f"{payload.get('tool')}"
                if summary:
                    detail += f" — {summary}"
                self._line("ACTION", detail)
            else:
                self._line("ACTION", f"invalid reply: {payload.get('error')}")
        elif event == "tool_result":
            summary = payload.get("summary")
            detail = f"{payload.get('tool')}"
            if summary:
                detail += f" — {summary}"
            self._line("RESULT", detail)
            self._indent(payload.get("observation", ""))
        elif event == "repair_finished":
            verdict = "passed" if payload.get("passed") else "failed"
            self._line(
                "FINISHED", f"{verdict} — {payload.get('reason')}"
            )

    @staticmethod
    def _line(label: str, detail: str) -> None:
        """Print one labeled step, flushed so live runs appear immediately."""

        print(f"{label:<9}{detail}", flush=True)

    @staticmethod
    def _indent(observation: Any) -> None:
        """Print a (possibly multiline) tool observation, indented."""

        for line in str(observation).splitlines() or [""]:
            print(f"    {line}", flush=True)


def _print_changed_files(changed_files: list[str]) -> None:
    """Print the workspace-relative files the repair changed."""

    print("\n== Changed files ==")
    if not changed_files:
        print("(none)")
        return
    for path in changed_files:
        print(f"- {path}")


def _print_test_result(test_result) -> None:
    """Print the final sandboxed test outcome, if any tests ran."""

    print("== Final sandboxed test result ==")
    if test_result is None:
        print("(no test run recorded)")
        return
    print(f"passed={test_result.passed}")
    print(f"returncode={test_result.returncode}")
    print(test_result.output.rstrip("\n"))


if __name__ == "__main__":
    raise SystemExit(main())
