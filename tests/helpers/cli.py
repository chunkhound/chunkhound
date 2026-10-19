"""CLI entry-point and parser test helpers.

Subprocesses prove the real entry point starts. Parser work (help, version,
argument errors) runs in-process: it needs no interpreter spawn and is not
subject to cold-start latency.
"""

import argparse
import contextlib
import io
import os
import subprocess
import sys
from pathlib import Path

from tests.utils import get_safe_subprocess_env

# Hang guards, not latency assertions. A cold child imports the full CLI
# graph, including the native extension and DuckDB.
CLI_TIMEOUT_SECONDS = 30.0
CLI_STARTUP_TIMEOUT_SECONDS = 120.0


def cli_command(*arguments: str) -> list[str]:
    # Reuse pytest's interpreter; uv discovery must not depend on the child cwd.
    return [sys.executable, "-m", "chunkhound.api.cli.main", *arguments]


def run_cli(
    arguments: list[str],
    *,
    cwd: Path | None = None,
    timeout: float = CLI_TIMEOUT_SECONDS,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return run_command(
        cli_command(*arguments),
        cwd=cwd or Path.cwd(),
        timeout=timeout,
        env={"CHUNKHOUND_NO_RICH": "1", **(env or {})},
    )


def run_cli_in_process(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    """Parse ``arguments`` with the production parser, in this process.

    Returns the same result shape as run_cli, so assertions stay identical.
    """
    from chunkhound.api.cli.main import create_parser

    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = 0
    try:
        with (
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            create_parser().parse_args(arguments)
    except SystemExit as exc:
        if exc.code is None:
            exit_code = 0
        elif isinstance(exc.code, int):
            exit_code = exc.code
        else:
            exit_code = 1
    return subprocess.CompletedProcess(
        arguments, exit_code, stdout.getvalue(), stderr.getvalue()
    )


def registered_cli_commands() -> tuple[str, ...]:
    """Every subcommand the production parser registers.

    Derived, never duplicated: a new subcommand is covered automatically.
    """
    from chunkhound.api.cli.main import create_parser

    parser = create_parser()
    subparsers = next(
        action
        for action in parser._actions  # noqa: SLF001 - argparse has no public API
        if isinstance(action, argparse._SubParsersAction)
    )
    return tuple(subparsers.choices)


def run_command(
    arguments: list[str],
    *,
    cwd: Path,
    timeout: float = CLI_TIMEOUT_SECONDS,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        cwd=cwd.resolve(),
        env=get_safe_subprocess_env({**os.environ, **(env or {})}),
        stdin=subprocess.DEVNULL,  # no inherited stdin; tests are non-interactive
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def init_git_repository(path: Path) -> None:
    """Give discovery tests a real repository, not a checkout-dependent marker."""
    path.mkdir(parents=True, exist_ok=True)
    run_command(["git", "init"], cwd=path, timeout=20).check_returncode()
