from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _run(
    cmd: list[str], cwd: Path | None = None, timeout: int = 30
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", *cmd],
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def test_diagnose_reports_mismatch_for_default_excludes(tmp_path: Path) -> None:
    root = tmp_path
    # Create a git repo (no .gitignore rules)
    (root / ".git").mkdir()
    # File under a default-excluded directory; Git won't ignore it, CH will
    (root / "other" / "build").mkdir(parents=True, exist_ok=True)
    (root / "other" / "build" / "tool.txt").write_text("x")

    proc = _run(
        ["chunkhound", "index", "--check-ignores", str(root), "--vs", "git", "--json"],
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout or "{}")
    mismatches = data.get("mismatches", [])
    # Expect at least one mismatch (CH excludes, Git doesn't)
    assert any(
        m.get("path") == "other/build/tool.txt"
        and m.get("ch") is True
        and m.get("git") is False
        for m in mismatches
    ), f"Unexpected mismatches: {mismatches}"
