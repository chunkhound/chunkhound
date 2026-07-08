"""CLI smoke tests for `chunkhound index simulate`.

This verifies the help output and a minimal functional dry-run listing.
These tests will fail until the subcommand is implemented.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def _run(
    cmd: list[str], cwd: Path | None = None, timeout: int = 15
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", *cmd],
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def test_index_simulate_help() -> None:
    proc = _run(["chunkhound", "index", "--help"], timeout=20)
    assert proc.returncode == 0, proc.stderr
    assert "simulate" in proc.stdout.lower()


@pytest.mark.asyncio
async def test_index_simulate_lists_files(tmp_path: Path) -> None:
    # Create tiny project with a couple files
    (tmp_path / "src").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "a.py").write_text("print('hi')\n")
    (tmp_path / "README.md").write_text("# demo\n")

    # Use defaults; expect at least python file included
    proc = _run(["chunkhound", "index", "--simulate", str(tmp_path)], timeout=30)
    assert proc.returncode == 0, proc.stderr

    out = proc.stdout.strip().splitlines()
    # Output is a sorted list of relative paths; ensure a.py appears
    assert any(line.endswith("src/a.py") for line in out), (
        f"Unexpected simulate output: {out!r}"
    )
