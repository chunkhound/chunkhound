from __future__ import annotations

from pathlib import Path

from tests.helpers.cli import run_cli


def test_simulate_skips_gitignore_in_excluded_subtrees(tmp_path: Path) -> None:
    root = tmp_path

    # node_modules is excluded by default config excludes
    (root / "node_modules").mkdir()
    (root / "node_modules" / ".gitignore").write_text("*.md\n")
    (root / "node_modules" / "in_mod.txt").write_text("x")
    (root / "README.md").write_text("x")

    proc = run_cli(["index", "--simulate", str(root)], cwd=root)
    assert proc.returncode == 0, proc.stderr
    out = set(proc.stdout.strip().splitlines())

    # node_modules content should not appear at all
    assert "node_modules/in_mod.txt" not in out
    # README.md should still appear (root not excluded)
    assert "README.md" in out
