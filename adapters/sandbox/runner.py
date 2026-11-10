"""Copy an evaluation case and run its tests in a short-lived Docker sandbox."""

from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path

from core.models import TestResult

IMAGE_NAME = "coding-agent-lab-runner"
CONTAINER_WORKSPACE = "/workspace"
WORKSPACE_ROOT = Path.home() / ".cache" / "coding-agent-lab"


def create_workspace(source: str | Path) -> Path:
    """Copy ``evals/<case>`` into a new disposable folder under the home directory."""

    source_path = Path(source).resolve()
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    destination = WORKSPACE_ROOT / f"ws-{uuid.uuid4().hex}"
    shutil.copytree(source_path, destination)
    return destination


def run_tests(workspace: str | Path) -> TestResult:
    """Run pytest in Docker with networking disabled and a read-only workspace mount."""

    workspace_path = Path(workspace).resolve()
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-v",
        f"{workspace_path}:{CONTAINER_WORKSPACE}:ro",
        "-w",
        CONTAINER_WORKSPACE,
        IMAGE_NAME,
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True)
    except OSError as exc:
        return TestResult(
            passed=False,
            output=f"Docker runner error: {exc}",
            returncode=127,
        )

    output = f"{completed.stdout}{completed.stderr}"
    return TestResult(
        passed=completed.returncode == 0,
        output=output,
        returncode=completed.returncode,
    )
