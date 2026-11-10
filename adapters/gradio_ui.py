"""A small Gradio presentation layer over the existing RepairService.

This module only shows input controls and displays results. It never builds
prompts, never calls Docker, and never makes repair-policy decisions: those
stay in ``core/`` and reach here only through the injected ``RepairService``.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr

from core.models import RepairResult

# The only projects the page may offer: the five supplied evals, nothing else.
CASES = [
    "arithmetic",
    "balanced_brackets",
    "median",
    "multi_function",
    "unique_words",
]


def create_demo(service: Any | None = None) -> gr.Blocks:
    """Build the Gradio page wired to a scripted RepairService by default."""

    if service is None:
        # Lazy import keeps this adapter importable without the assembly point.
        from bootstrap import build_service

        service = build_service("scripted")

    with gr.Blocks(title="Coding Agent Lab - Scripted Repair") as demo:
        gr.Markdown(
            "# Scripted repair\n"
            "Pick one supplied eval project and start a bounded, sandboxed "
            "repair. The page shows the trace, the changed files, and the "
            "final test output from the existing repair service."
        )
        with gr.Row():
            case = gr.Dropdown(
                choices=CASES,
                value=CASES[0],
                label="Eval project",
            )
            start = gr.Button("Start repair", variant="primary")

        summary = gr.Textbox(label="Repair summary", interactive=False)
        trace = gr.Textbox(label="Visible trace", lines=16, interactive=False)
        changed = gr.Code(label="Changed files and text")
        test_output = gr.Textbox(
            label="Final sandboxed test output", lines=12, interactive=False
        )
        download = gr.File(label="Download repaired workspace (ZIP)")

        start.click(
            fn=lambda selected: _run_repair(service, selected),
            inputs=case,
            outputs=[summary, trace, changed, test_output, download],
        )

    return demo


def _run_repair(service: Any, case_name: str) -> tuple[str, str, str, str, str | None]:
    """Run one repair for a selected case and format every visible result."""

    result = service.repair(f"evals/{case_name}")
    return (
        _format_summary(result),
        _format_trace(result.trace),
        _format_changed_files(result),
        _format_test_output(result),
        _zip_workspace(result),
    )


def _format_summary(result: RepairResult) -> str:
    """Give a one-line pass/fail summary with the finishing reason."""

    reason = ""
    if result.trace:
        reason = result.trace[-1].get("reason", "")
    verdict = "PASSED" if result.passed else "FAILED"
    return f"{verdict} - {reason}" if reason else verdict


def _format_trace(trace: list[dict[str, Any]]) -> str:
    """Render the visible trace events in the order they occurred."""

    lines: list[str] = []
    for entry in trace:
        event = entry.get("event", "event")
        if event == "repair_started":
            lines.append(f"[started]  case={entry.get('case')}")
        elif event == "model_response":
            if entry.get("valid"):
                lines.append(
                    f"[model]    tool={entry.get('tool')} "
                    f"summary={entry.get('summary')}"
                )
            else:
                lines.append(f"[model]    invalid reply: {entry.get('error')}")
        elif event == "tool_result":
            summary = entry.get("summary")
            label = f" summary={summary}" if summary else ""
            lines.append(f"[tool]     {entry.get('tool')}{label}")
        elif event == "repair_finished":
            lines.append(
                f"[finished] passed={entry.get('passed')} "
                f"reason={entry.get('reason')}"
            )
        else:
            lines.append(f"[event]    {entry}")
    return "\n".join(lines)


def _format_changed_files(result: RepairResult) -> str:
    """Show each changed relative file with its repaired text from the copy."""

    if not result.changed_files:
        return "(no files changed)"
    workspace = result.workspace
    sections: list[str] = []
    for relative in result.changed_files:
        header = f"# {relative}"
        if workspace is None:
            sections.append(f"{header}\n(workspace unavailable)")
            continue
        try:
            text = (Path(workspace) / relative).read_text(encoding="utf-8")
        except OSError as exc:
            text = f"(could not read file: {exc})"
        sections.append(f"{header}\n{text}")
    return "\n\n".join(sections)


def _format_test_output(result: RepairResult) -> str:
    """Show the final sandboxed test outcome, if any tests ran."""

    test_result = result.last_test_result
    if test_result is None:
        return "(no test run recorded)"
    return (
        f"passed={test_result.passed}\n"
        f"returncode={test_result.returncode}\n"
        f"{test_result.output.rstrip()}"
    )


def _zip_workspace(result: RepairResult) -> str | None:
    """Package the repaired disposable workspace into a ZIP for download."""

    workspace = result.workspace
    if workspace is None or not Path(workspace).is_dir():
        return None
    output_dir = Path(tempfile.mkdtemp(prefix="repair-zip-"))
    base = output_dir / f"{Path(workspace).name}-repaired"
    archive = shutil.make_archive(str(base), "zip", root_dir=str(workspace))
    return archive
