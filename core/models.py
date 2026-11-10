"""Shared result objects used by the repair workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TestResult:
    """Outcome of one sandboxed test run."""

    __test__ = False

    passed: bool
    output: str
    returncode: int


@dataclass
class ModelAction:
    """One permitted tool request parsed from a model reply."""

    tool: str
    arguments: dict[str, Any]
    summary: str | None = None


@dataclass
class RepairResult:
    """Final outcome of one bounded repair attempt."""

    passed: bool
    trace: list[dict[str, Any]] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    workspace: Path | None = None
    last_test_result: TestResult | None = None
