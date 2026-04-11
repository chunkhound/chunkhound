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
    # Provider-type check must run before database-existence check.
    assert "only supported for duckdb" in combined


def test_repack_db_flag_anchors_config_without_positional(tmp_path: Path) -> None:
    """`chunkhound repack --db <abs>` from an unrelated CWD must succeed.

    main.py remaps args.path to the --db file's parent when no positional
    path is given, so Config does not pick up a conflicting CWD-local
    .chunkhound.json (e.g. provider=lancedb).
    """
    import json

    # 1. Index a real DuckDB project at tmp_path/dbproj
    dbproj = tmp_path / "dbproj"
    dbproj.mkdir()
    (dbproj / "hello.py").write_text("print('hello')\n")
    proc = _run(
        ["chunkhound", "index", "--no-embeddings", str(dbproj)],
        cwd=dbproj,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr

    # 2. Create a conflicting CWD-local config in an unrelated directory
    conflict_cwd = tmp_path / "conflict_cwd"
    conflict_cwd.mkdir()
    (conflict_cwd / ".chunkhound.json").write_text(
        json.dumps({"database": {"provider": "lancedb"}})
    )

    # 3. Run `chunkhound repack --db <dbproj-db> --dry-run` from the conflict
    #    dir WITHOUT the positional. main.py must remap args.path to the
    #    --db parent, bypassing the lancedb config in CWD.
    db_path = dbproj / ".chunkhound" / "db"
    proc = _run(
        ["chunkhound", "repack", "--db", str(db_path), "--dry-run"],
        cwd=conflict_cwd,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    combined = (proc.stdout + proc.stderr).lower()
    assert "only supported for duckdb" not in combined


def test_repack_positional_path_anchors_config(tmp_path: Path) -> None:
    """Positional project path must anchor config discovery on the target project.

    Without a positional arg, Config falls back to Path.cwd() and merges any
    CWD-local .chunkhound.json — so running repack from a directory with a
    conflicting lancedb config would falsely reject a valid DuckDB target.
    """
    import json

    # 1. Create a DuckDB project at tmp_path/dbproj
    dbproj = tmp_path / "dbproj"
    dbproj.mkdir()
    (dbproj / "hello.py").write_text("print('hello')\n")
    proc = _run(
        ["chunkhound", "index", "--no-embeddings", str(dbproj)],
        cwd=dbproj,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr

    # 2. Create a second dir with a conflicting CWD-local .chunkhound.json
    conflict_cwd = tmp_path / "conflict_cwd"
    conflict_cwd.mkdir()
    (conflict_cwd / ".chunkhound.json").write_text(
        json.dumps({"database": {"provider": "lancedb"}})
    )

    # 3. From the conflict dir, run `chunkhound repack <dbproj> --dry-run`.
    #    Without the positional arg fix, Config would pick up the lancedb
    #    config from CWD and reject the DuckDB target.
    proc = _run(
        ["chunkhound", "repack", str(dbproj), "--dry-run"],
        cwd=conflict_cwd,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    combined = (proc.stdout + proc.stderr).lower()
    assert "only supported for duckdb" not in combined
    assert "waste" in combined or "reclaimable" in combined
