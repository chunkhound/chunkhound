from __future__ import annotations

import subprocess


def _run(cmd: list[str], timeout: int = 20) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", *cmd], text=True, capture_output=True, timeout=timeout
    )


def test_index_help_mentions_check_ignores() -> None:
    proc = _run(["chunkhound", "index", "--help"])
    assert proc.returncode == 0, proc.stderr
    # Ensure the help lists the new flag
    assert "check-ignores" in proc.stdout
