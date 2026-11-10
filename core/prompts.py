"""Text sent to a model: system rules, case request, and tool observations."""

from __future__ import annotations


def system_instruction() -> str:
    """Tell the model the evidence order and the one-JSON-object reply format."""

    return (
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
