"""Ensure `chunkhound index --simulate` works before any DB is created.

This verifies that simulate uses an in-memory database when the default
database directory does not exist yet (fresh workspace), and that it
does not fail with DuckDB file errors nor creates .chunkhound/ on disk.
"""

from __future__ import annotations

from pathlib import Path

from tests.helpers.cli import run_cli


def test_simulate_uses_in_memory_db_on_fresh_workspace(tmp_path: Path) -> None:
    # Fresh workspace: no .chunkhound directory present
    assert not (tmp_path / ".chunkhound").exists()

    # Create a couple of files to be discovered
    (tmp_path / "pkg").mkdir(parents=True, exist_ok=True)
    (tmp_path / "pkg" / "mod.py").write_text("print('ok')\n")
    (tmp_path / "README.md").write_text("# readme\n")

    # Run simulate from the fresh workspace
    proc = run_cli(
        ["index", "--simulate", str(tmp_path), "--show-sizes", "--sort", "size"],
        cwd=tmp_path,
    )

    # Must not fail due to nonexistent DB path
    assert proc.returncode == 0, f"simulate failed: {proc.stderr}"

    # Should not create on-disk DB directories as we use ':memory:'
    assert not (tmp_path / ".chunkhound").exists(), (
        "simulate should not create .chunkhound on disk"
    )

    # Output should include our Python file path
    out_lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    assert any(ln.endswith("pkg/mod.py") for ln in out_lines), (
        f"Unexpected simulate output: {out_lines!r}"
    )
