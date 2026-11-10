"""Focused checks for workspace path safety and Docker-backed test delegation."""

from __future__ import annotations

from pathlib import Path

import pytest

from adapters.sandbox.repair_tools import RepairTools, safe_path
from core.models import TestResult


def _tools(workspace: Path, runner=None) -> RepairTools:
    return RepairTools(workspace, runner=runner or (lambda _path: _unused_result()))


def _unused_result() -> TestResult:
    return TestResult(passed=False, output="unused", returncode=1)


def test_read_file_returns_workspace_contents(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("hello workspace\n", encoding="utf-8")
    tools = _tools(tmp_path)

    assert tools.read_file("notes.txt") == "hello workspace\n"


def test_safe_path_rejects_absolute_paths(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    absolute = str(tmp_path / "notes.txt")

    with pytest.raises(ValueError, match="Absolute"):
        safe_path(tmp_path, absolute)
    with pytest.raises(ValueError, match="Absolute"):
        tools.read_file(absolute)


def test_safe_path_rejects_parent_directory_paths(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    secret = tmp_path.parent / "secret.txt"
    secret.write_text("do not read\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Parent-directory"):
        safe_path(tmp_path, "../secret.txt")
    with pytest.raises(ValueError, match="Parent-directory"):
        tools.read_file("../secret.txt")
    with pytest.raises(ValueError, match="Parent-directory"):
        tools.edit_file("../secret.txt", "nope\n")


def test_edit_file_saves_content_and_records_path_once(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text("value = 1\n", encoding="utf-8")
    tools = _tools(tmp_path)

    tools.edit_file("module.py", "value = 2\n")
    tools.edit_file("module.py", "value = 3\n")

    assert (tmp_path / "module.py").read_text(encoding="utf-8") == "value = 3\n"
    assert tools.changed_files == ["module.py"]


def test_run_tests_delegates_to_runner_not_host_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = TestResult(passed=True, output="from-runner", returncode=0)
    seen: list[Path] = []

    def fake_runner(workspace: Path) -> TestResult:
        seen.append(workspace)
        return sentinel

    def forbid_host_process(args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError(f"host process launched: {args}")

    monkeypatch.setattr("subprocess.run", forbid_host_process)
    tools = RepairTools(tmp_path, runner=fake_runner)

    assert tools.run_tests() == sentinel
    assert seen == [tmp_path.resolve()]
