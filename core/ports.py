"""Contracts the repair loop relies on without choosing implementations."""

from __future__ import annotations

from typing import Any, Protocol

from core.models import TestResult


class ChatClient(Protocol):
    """A model adapter that returns the next reply as text."""

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Return one model reply for the given conversation."""


class RepairToolsPort(Protocol):
    """Safe file and test operations inside a disposable workspace.

    Task 2 provides a concrete ``RepairTools`` class that satisfies this
    contract.
    """

    def read_file(self, path: str) -> str:
        """Return UTF-8 text from a workspace-relative path."""

    def edit_file(self, path: str, content: str) -> None:
        """Replace a workspace-relative file with UTF-8 content."""

    def run_tests(self) -> TestResult:
        """Run sandboxed tests against the current workspace."""


class TraceRecorder(Protocol):
    """Optional observer for public repair events."""

    def record(self, event: str, payload: dict[str, Any]) -> None:
        """Record one public event without changing the repair outcome."""


class ResearchPort(Protocol):
    """Optional, bounded external evidence source, separate from file tools.

    This is deliberately not part of ``RepairToolsPort``: research is optional
    read-only evidence, never a sandbox file or test capability.
    """

    def search_github(self, query: str) -> dict[str, Any]:
        """Return a small, JSON-compatible evidence record for one query."""
