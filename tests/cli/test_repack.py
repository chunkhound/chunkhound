"""CLI integration tests for `chunkhound repack`."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _run(
    cmd: list[str],
    cwd: Path | None = None,
    timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", *cmd],
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


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
    proc = _run(
        ["chunkhound", "index", "--no-embeddings", str(tmp_path)],
        cwd=tmp_path,
        timeout=60,
    )
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


def test_repack_backup_creates_bak_file(tmp_path: Path) -> None:
    """`chunkhound repack --backup <project>` must leave a .bak file next to the DB."""
    src = tmp_path / "hello.py"
    src.write_text("def hello():\n    return 'world'\n")

    proc = _run(
        ["chunkhound", "index", "--no-embeddings", str(tmp_path)],
        cwd=tmp_path,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr

    db_dir = tmp_path / ".chunkhound" / "db"
    db_file = db_dir / "chunks.db"
    # repack.py uses db_path.with_suffix(db_path.suffix + ".bak")
    # which yields chunks.db.bak next to chunks.db.
    backup_path = db_file.with_suffix(db_file.suffix + ".bak")

    proc = _run(
        ["chunkhound", "repack", "--backup", str(tmp_path)],
        cwd=tmp_path,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert backup_path.exists(), (
        f"--backup must create {backup_path}; got files: "
        f"{sorted(p.name for p in db_dir.iterdir())}"
    )


def test_repack_preserves_searchable_data_and_reports_results(tmp_path: Path) -> None:
    """Non-dry-run repack must preserve searchability and report before/after stats."""
    marker = "repack_survivor_marker"
    src = tmp_path / "survivor.py"
    src.write_text(
        f"def {marker}():\n"
        f"    return '{marker}'\n"
    )

    proc = _run(
        ["chunkhound", "index", "--no-embeddings", str(tmp_path)],
        cwd=tmp_path,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr

    before = _run(
        ["chunkhound", "search", marker, str(tmp_path), "--regex"],
        cwd=tmp_path,
        timeout=60,
    )
    assert before.returncode == 0, before.stderr
    assert marker in before.stdout

    repack = _run(
        ["chunkhound", "repack", str(tmp_path)],
        cwd=tmp_path,
        timeout=60,
    )
    assert repack.returncode == 0, repack.stderr
    output = repack.stdout
    assert "Before:" in output
    assert "After:" in output
    assert "Saved:" in output or "No size change" in output

    after = _run(
        ["chunkhound", "search", marker, str(tmp_path), "--regex"],
        cwd=tmp_path,
        timeout=60,
    )
    assert after.returncode == 0, after.stderr
    assert marker in after.stdout


def test_repack_non_duckdb_error() -> None:
    proc = _run(["chunkhound", "repack", "--database-provider", "lancedb"])
    assert proc.returncode != 0
    combined = (proc.stdout + proc.stderr).lower()
    # Provider-type check must remain the user-facing failure contract here.
    assert "only supported for duckdb" in combined


def test_repack_db_flag_anchors_config_without_positional(tmp_path: Path) -> None:
    """`chunkhound repack --db <abs>` from an unrelated CWD must succeed.

    The inferred target root must come from the target DB path rather than the
    invoking CWD, so an unrelated local config cannot hijack provider selection.
    """

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


def test_repack_db_flag_loads_target_project_config(tmp_path: Path) -> None:
    """`repack --db` must prefer the target project's config over the invoking CWD."""
    dbproj = tmp_path / "dbproj"
    dbproj.mkdir()
    (dbproj / "hello.py").write_text("print('hello')\n")
    proc = _run(
        ["chunkhound", "index", "--no-embeddings", str(dbproj)],
        cwd=dbproj,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr

    (dbproj / ".chunkhound.json").write_text(
        json.dumps(
            {
                "database": {
                    "provider": "lancedb",
                }
            }
        )
    )

    conflict_cwd = tmp_path / "conflict_cwd"
    conflict_cwd.mkdir()
    (conflict_cwd / ".chunkhound.json").write_text(
        json.dumps(
            {
                "database": {
                    "provider": "duckdb",
                }
            }
        )
    )

    db_path = dbproj / ".chunkhound" / "db"
    proc = _run(
        ["chunkhound", "repack", "--db", str(db_path), "--dry-run"],
        cwd=conflict_cwd,
        timeout=60,
    )
    assert proc.returncode != 0
    combined = (proc.stdout + proc.stderr).lower()
    assert "only supported for duckdb" in combined


def test_repack_db_flag_custom_db_uses_git_root_config(tmp_path: Path) -> None:
    """Custom DB locations must still anchor config discovery at the repo root."""
    dbproj = tmp_path / "dbproj"
    dbproj.mkdir()
    (dbproj / ".git").mkdir()
    (dbproj / "hello.py").write_text("print('hello')\n")
    proc = _run(
        ["chunkhound", "index", "--no-embeddings", str(dbproj)],
        cwd=dbproj,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr

    custom_db = dbproj / "var" / "chunkhound-db"
    custom_db.parent.mkdir(parents=True)
    (dbproj / ".chunkhound" / "db").rename(custom_db)
    (dbproj / ".chunkhound.json").write_text(
        json.dumps(
            {
                "database": {
                    "provider": "lancedb",
                }
            }
        )
    )

    conflict_cwd = tmp_path / "conflict_cwd"
    conflict_cwd.mkdir()
    (conflict_cwd / ".chunkhound.json").write_text(
        json.dumps({"database": {"provider": "duckdb"}})
    )

    proc = _run(
        ["chunkhound", "repack", "--db", str(custom_db), "--dry-run"],
        cwd=conflict_cwd,
        timeout=60,
    )
    assert proc.returncode != 0
    combined = (proc.stdout + proc.stderr).lower()
    assert "only supported for duckdb" in combined


def test_repack_positional_path_anchors_config(tmp_path: Path) -> None:
    """Positional project path must anchor config discovery on the target project.

    Without a positional arg, Config falls back to Path.cwd() and merges any
    CWD-local .chunkhound.json — so running repack from a directory with a
    conflicting lancedb config would falsely reject a valid DuckDB target.
    """
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
