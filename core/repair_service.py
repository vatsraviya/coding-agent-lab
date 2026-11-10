"""Select one allowed eval project, copy it, and run the repair agent on it.

This module holds the project-selection policy only. It never parses the
command line and never issues Docker commands. The disposable copy and the
agent are supplied by ``bootstrap.py`` so this file depends on nothing in
``adapters/``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from core.models import RepairResult
from core.repair_agent import RepairAgent

# A factory that turns one disposable workspace copy into a ready repair agent.
MakeAgent = Callable[[str, Path], RepairAgent]
# A factory that copies a selected project into a fresh disposable workspace.
CreateWorkspace = Callable[[Path], Path]


class RepairService:
    """Validate a selected eval project, copy it, and run one repair attempt."""

    def __init__(
        self,
        project_root: str | Path,
        evals_root: str | Path,
        create_workspace: CreateWorkspace,
        make_agent: MakeAgent,
    ) -> None:
        self._project_root = Path(project_root).resolve()
        self._evals_root = Path(evals_root).resolve()
        self._create_workspace = create_workspace
        self._make_agent = make_agent

    def repair(self, selected: str | Path) -> RepairResult:
        """Repair one allowed project and return its result in every outcome."""

        case_dir = self._resolve_case(selected)
        case_name = case_dir.name
        workspace = Path(self._create_workspace(case_dir)).resolve()
        relative_files = self._relative_files(workspace)
        agent = self._make_agent(case_name, workspace)
        return agent.run(case_name, relative_files)

    def _resolve_case(self, selected: str | Path) -> Path:
        """Accept only a direct, existing project folder under ``evals/``."""

        requested = Path(selected)
        if requested.is_absolute():
            raise ValueError(f"Absolute paths are not allowed: {selected}")
        if ".." in requested.parts:
            raise ValueError(f"Parent-directory paths are not allowed: {selected}")

        candidate = (self._project_root / requested).resolve()
        if candidate.parent != self._evals_root:
            raise ValueError(
                f"Project must be a direct child of evals/: {selected}"
            )
        if not candidate.is_dir():
            raise ValueError(f"Unknown project under evals/: {selected}")
        return candidate

    @staticmethod
    def _relative_files(workspace: Path) -> list[str]:
        """List the copied project's files as sorted workspace-relative paths."""

        root = Path(workspace).resolve()
        return [
            path.relative_to(root).as_posix()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        ]
