"""Safe read, edit, and test operations inside one disposable workspace."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from core.models import TestResult

Runner = Callable[[Path], TestResult]


def safe_path(workspace: str | Path, requested_path: str) -> Path:
    """Resolve a workspace-relative path, rejecting absolute and escaping paths."""

    if not requested_path or requested_path.strip() == "":
        raise ValueError("A relative workspace path is required.")
    requested = Path(requested_path)
    if requested.is_absolute():
        raise ValueError(f"Absolute paths are not allowed: {requested_path}")
    if ".." in requested.parts:
        raise ValueError(f"Parent-directory paths are not allowed: {requested_path}")

    root = Path(workspace).resolve()
    candidate = (root / requested).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Path escapes the workspace: {requested_path}")
    return candidate


class RepairTools:
    """Host-side file tools plus Docker-backed tests for one workspace copy."""

    def __init__(
        self,
        workspace: str | Path,
        runner: Runner | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self._runner = runner
        self.changed_files: list[str] = []

    def read_file(self, path: str) -> str:
        """Return UTF-8 text from a file inside the workspace."""

        target = safe_path(self.workspace, path)
        return target.read_text(encoding="utf-8")

    def edit_file(self, path: str, content: str) -> None:
        """Replace a workspace file and record its relative path once."""

        target = safe_path(self.workspace, path)
        target.write_text(content, encoding="utf-8")
        relative = target.relative_to(self.workspace).as_posix()
        if relative not in self.changed_files:
            self.changed_files.append(relative)

    def run_tests(self) -> TestResult:
        """Delegate pytest to the Docker-backed runner; never run host pytest."""

        runner = self._runner if self._runner is not None else _default_run_tests
        return runner(self.workspace)


def _default_run_tests(workspace: Path) -> TestResult:
    from adapters.sandbox.runner import run_tests as docker_run_tests

    return docker_run_tests(workspace)
