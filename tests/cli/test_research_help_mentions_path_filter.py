from __future__ import annotations

import subprocess


def _run(cmd: list[str], timeout: int = 20) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", *cmd], text=True, capture_output=True, timeout=timeout
    )


def test_research_help_mentions_path_filter() -> None:
    proc = _run(["chunkhound", "research", "--help"])
    assert proc.returncode == 0, proc.stderr
    assert "--path-filter" in proc.stdout

