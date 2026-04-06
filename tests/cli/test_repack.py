"""CLI integration tests for `chunkhound repack`."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def _run(cmd: list[str], cwd: Path | None = None, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["uv", "run", *cmd], cwd=str(cwd) if cwd else None, text=True, capture_output=True, timeout=timeout)


def test_repack_help() -> None:
    proc = _run(["chunkhound", "repack", "--help"])
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout.lower()
    assert "compact" in out or "repack" in out
    assert "dry-run" in out or "dry_run" in out


def test_repack_dry_run(tmp_path: Path) -> None:
    # Create a tiny file to index
    src = tmp_path / "hello.py"
    src.write_text("print('hello')\n")

    # Index into a DuckDB (no embeddings needed for repack test)
    proc = _run(["chunkhound", "index", "--no-embeddings", str(tmp_path)], cwd=tmp_path, timeout=60)
    assert proc.returncode == 0, proc.stderr

    db_dir = tmp_path / ".chunkhound" / "db"
    assert db_dir.exists(), f"Expected DB dir at {db_dir}"

    # Record size before repack
    size_before = sum(f.stat().st_size for f in db_dir.rglob("*") if f.is_file())

    # Dry-run repack — pass --db with the db subdirectory
    proc = _run(["chunkhound", "repack", "--dry-run", "--db", str(db_dir)], timeout=60)
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout.lower()
    assert "waste" in out or "reclaimable" in out

    # DB size should be unchanged
    size_after = sum(f.stat().st_size for f in db_dir.rglob("*") if f.is_file())
    assert size_after == size_before


def test_repack_non_duckdb_error() -> None:
    proc = _run(["chunkhound", "repack", "--database-provider", "lancedb"])
    assert proc.returncode != 0
    combined = (proc.stdout + proc.stderr).lower()
    # May fail with "only supported for duckdb" or "database not found" depending on config
    assert "only supported for duckdb" in combined or "not found" in combined
