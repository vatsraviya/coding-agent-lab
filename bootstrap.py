"""The single assembly point: wire core rules to concrete adapters.

This is the only module that imports from both ``core/`` and ``adapters/``.
It builds the runner-backed safe tools, the selected chat client, the
``RepairAgent``, and the ``RepairService``, then connects them. The scripted
answer cards live here as data, kept separate from the repair loop itself.
"""

from __future__ import annotations

import json
from pathlib import Path

from adapters.models.scripted import ScriptedClient
from adapters.sandbox.repair_tools import RepairTools
from adapters.sandbox.runner import create_workspace
from adapters.sandbox.runner import run_tests as docker_run_tests
from core.ports import TraceRecorder
from core.repair_agent import RepairAgent
from core.repair_service import RepairService

_PROJECT_ROOT = Path(__file__).resolve().parent
_EVALS_ROOT = _PROJECT_ROOT / "evals"


# --- Corrected source files, one single-bug fix per case --------------------
# Each is the COMPLETE corrected file the scripted edit writes, not a diff.

_CALCULATOR = '''def add(left, right):
    """Return the sum of two numbers."""
    return left + right
'''

_BRACKETS = '''def is_balanced(text):
    """Return whether parentheses, brackets, and braces are properly nested."""
    opening = "([{"
    closing = ")]}"
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []

    for character in text:
        if character in opening:
            stack.append(character)
        elif character in closing:
            if not stack or stack.pop() != pairs[character]:
                return False

    return not stack
'''

_STATISTICS_TOOLS = '''def median(values):
    """Return the median value without changing the input sequence."""
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 0:
        return (ordered[middle - 1] + ordered[middle]) / 2
    return ordered[middle]
'''

_TEXT_TOOLS = '''def normalize_name(name):
    """Trim surrounding whitespace and lowercase a supplied name."""
    return name.strip().lower()


def greeting(name):
    """Create a greeting from a normalized name."""
    return f"Hello, {normalize_name(name).capitalize()}!"


def initials(name):
    """Return uppercase initials for the whitespace-separated parts of a name."""
    return "".join(part[0].upper() for part in name.split())
'''

_WORD_TOOLS = '''def unique_words(text):
    """Return lowercase words once, in the order first seen."""
    return list(dict.fromkeys(text.lower().split()))
'''


def _card(**fields: str) -> str:
    """Render one answer card as a single compact JSON object string."""

    return json.dumps(fields)


def _sequence(test_file: str, source_file: str, corrected: str) -> list[str]:
    """Build the evidence-ordered read -> fail -> edit -> pass answer cards."""

    return [
        _card(
            tool="read_file",
            path=test_file,
            summary=f"Read {test_file} to learn the expected behavior",
        ),
        _card(tool="run_tests", summary="Reproduce the failing test"),
        _card(tool="edit_file", path=source_file, content=corrected),
        _card(tool="run_tests", summary="Verify the repair passes"),
    ]


# Scripted sequences for every supplied case. One client per repair draws the
# sequence named by the case, so cases never share or exhaust each other's cards.
SCRIPTED_SEQUENCES: dict[str, list[str]] = {
    "arithmetic": _sequence("test_calculator.py", "calculator.py", _CALCULATOR),
    "balanced_brackets": _sequence("test_brackets.py", "brackets.py", _BRACKETS),
    "median": _sequence(
        "test_statistics_tools.py", "statistics_tools.py", _STATISTICS_TOOLS
    ),
    "multi_function": _sequence(
        "test_text_tools.py", "text_tools.py", _TEXT_TOOLS
    ),
    "unique_words": _sequence("test_word_tools.py", "word_tools.py", _WORD_TOOLS),
}


def build_service(
    provider: str,
    model: str | None = None,
    temperature: float = 0.0,
    recorder: TraceRecorder | None = None,
) -> RepairService:
    """Assemble a RepairService for the chosen model provider.

    An optional ``recorder`` is passed to each RepairAgent so a terminal can
    display live events. It only observes; repair logic stays unchanged.
    """

    if provider == "scripted":
        def make_agent(case_name: str, workspace: Path) -> RepairAgent:
            """Build runner-backed tools and a scripted agent for one copy."""

            try:
                sequence = SCRIPTED_SEQUENCES[case_name]
            except KeyError as exc:
                raise ValueError(
                    f"No scripted answer cards for case: {case_name}"
                ) from exc

            client = ScriptedClient(sequence)
            tools = RepairTools(workspace, runner=docker_run_tests)
            return RepairAgent(client, tools, recorder=recorder)
    elif provider == "ollama":
        def make_agent(case_name: str, workspace: Path) -> RepairAgent:
            """Build runner-backed tools and a live Ollama agent for one copy."""

            del case_name  # The live model is not keyed by case.
            from adapters.models.ollama import OllamaClient

            client = OllamaClient(model=model, temperature=temperature)
            tools = RepairTools(workspace, runner=docker_run_tests)
            return RepairAgent(client, tools, recorder=recorder)
    else:
        raise ValueError(f"Unsupported provider: {provider!r}")

    return RepairService(
        project_root=_PROJECT_ROOT,
        evals_root=_EVALS_ROOT,
        create_workspace=create_workspace,
        make_agent=make_agent,
    )
