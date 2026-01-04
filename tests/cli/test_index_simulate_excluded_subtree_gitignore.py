from __future__ import annotations

import subprocess
from pathlib import Path


def _run(
    cmd: list[str], cwd: Path | None = None, timeout: int = 20
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", *cmd],
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def test_simulate_skips_gitignore_in_excluded_subtrees(tmp_path: Path) -> None:
    root = tmp_path

    # node_modules is excluded by default config excludes
    (root / "node_modules").mkdir()
    (root / "node_modules" / ".gitignore").write_text("*.md\n")
    (root / "node_modules" / "in_mod.txt").write_text("x")
    (root / "README.md").write_text("x")

    proc = _run(["chunkhound", "index", "--simulate", str(root)])
    assert proc.returncode == 0, proc.stderr
    out = set(proc.stdout.strip().splitlines())

    # node_modules content should not appear at all
    assert "node_modules/in_mod.txt" not in out
    # README.md should still appear (root not excluded)
    assert "README.md" in out
