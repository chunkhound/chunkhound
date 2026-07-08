"""Shared git repository helpers for test fixtures.

Provides reusable git operations used across contract test modules.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def run(command: list[str], cwd: Path) -> None:
    """Run a command in a repo directory, raising on failure."""
    subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)


def create_repo(repo_dir: Path) -> None:
    """Initialize a new git repo with test author identity."""
    run(["git", "init"], repo_dir)
    run(["git", "config", "user.name", "ChunkHound Tests"], repo_dir)
    run(["git", "config", "user.email", "tests@chunkhound.invalid"], repo_dir)


def commit_all(repo_dir: Path, message: str) -> str:
    """Stage all changes and commit, returning the commit hash."""
    run(["git", "add", "-A"], repo_dir)
    run(["git", "commit", "-m", message], repo_dir)
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()
