"""Text sent to a model: system rules, case request, and tool observations."""

from __future__ import annotations


def system_instruction(research_enabled: bool = False) -> str:
    """Tell the model the evidence order and the one-JSON-object reply format.

    When ``research_enabled`` is true, one extra action is offered. Research is
    optional evidence that must come after a local read and a failing test; it
    is never an authority and never a substitute for a passing sandboxed test.
    """

    base = (
        "You repair one local Python project. Follow this evidence order: "
        "inspect local code, reproduce a failing test before editing, make a "
        "focused edit, then verify the repair with tests.\n"
        "Reply with exactly one JSON object and nothing else. No Markdown, "
        "no code fences, and no prose.\n"
        "The optional summary field is a short public label for the trace. "
        "It is not private reasoning.\n"
        "Allowed actions:\n"
        '{"tool":"read_file","path":"test_calculator.py","summary":"Read the test"}\n'
        '{"tool":"run_tests","summary":"Reproduce the failure"}\n'
        '{"tool":"edit_file","path":"calculator.py","content":"..."}'
    )
    if not research_enabled:
        return base
    return (
        base
        + "\n"
        '{"tool":"search_github","query":"Python median even length","summary":"Find a public reference"}\n'
        "search_github is optional evidence, not an authority. It is allowed "
        "only after a successful read_file and a failing run_tests, and it "
        "never proves a fix: only a later passing sandboxed test can."
    )


def case_request(case_name: str, relative_files: list[str]) -> str:
    """Ask for the next action on a selected case and its relative files."""

    files = "\n".join(f"- {path}" for path in relative_files) or "- (no files listed)"
    return (
        f"Repair the evaluation case `{case_name}`.\n"
        "Workspace-relative files:\n"
        f"{files}\n"
        "Choose one next allowed action as a single JSON object."
    )


def tool_result_message(observation: str) -> str:
    """Format a tool observation for the next model turn."""

    return f"Tool result:\n{observation}"
