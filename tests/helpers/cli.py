"""Run the CLI module with the test interpreter."""

import os
import subprocess
import sys
from pathlib import Path

from tests.utils import get_safe_subprocess_env


def cli_command(*arguments: str) -> list[str]:
    # Reuse pytest's interpreter; uv discovery must not depend on the child cwd.
    return [sys.executable, "-m", "chunkhound.api.cli.main", *arguments]


def run_cli(
    arguments: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 30,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return run_command(
        cli_command(*arguments),
        cwd=cwd or Path.cwd(),
        timeout=timeout,
        env={"CHUNKHOUND_NO_RICH": "1", **(env or {})},
    )


def run_command(
    arguments: list[str],
    *,
    cwd: Path,
    timeout: int = 30,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        cwd=cwd.resolve(),
        env=get_safe_subprocess_env({**os.environ, **(env or {})}),
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def init_git_repository(path: Path) -> None:
    """Give discovery tests a real repository, not a checkout-dependent marker."""
    path.mkdir(parents=True, exist_ok=True)
    run_command(["git", "init"], cwd=path, timeout=20).check_returncode()
