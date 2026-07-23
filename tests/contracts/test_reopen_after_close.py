"""Contract test — the Rust pipeline's DB must survive a process restart.

Every real invocation of `chunkhound index` (or an MCP server restart) opens a
*brand new* DuckDB connection to a DB that a prior process already wrote and
fully closed. `test_incremental_updates.py` never catches a break here because
it re-indexes within the same pytest process — DuckDB's process-level cache
of the DB path can paper over an on-disk deserialization bug that only shows
up in a genuinely fresh process.

This test MUST FAIL until the Rust pipeline can reopen a DB it previously
wrote and closed, from a separate process, without error.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "pipeline"


def _rust_pipeline_available() -> bool:
    try:
        from chunkhound_native import IndexingPipeline
    except ImportError:
        return False
    return IndexingPipeline is not None


def _index_once(project_dir: Path, db_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "chunkhound.api.cli.main",
            "index",
            str(project_dir),
            "--db",
            str(db_dir),
            "--no-embeddings",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "CHUNKHOUND_USE_RUST": "1", "CHUNKHOUND_NO_RICH": "1"},
    )


@pytest.mark.skipif(
    not _rust_pipeline_available(),
    reason="chunkhound_native.IndexingPipeline not built",
)
def test_rust_pipeline_db_survives_process_restart(tmp_path: Path) -> None:
    """Index a directory, then re-index it in a fresh process — must not crash.

    Reproduces via the plain CLI (no custom harness):
      1. `chunkhound index <dir> --db <db>` — fresh DB, succeeds.
      2. `chunkhound index <dir> --db <db>` again, in a new process — the
         Rust pipeline persists an HNSW/vss catalog entry on write that a
         fresh `duckdb.connect()` currently cannot deserialize, crashing
         with `SerializationException: Failed to deserialize: field id
         mismatch, expected: 100, got: 0` before any query even runs.
    """
    project_dir = tmp_path / "fixture"
    shutil.copytree(FIXTURE_DIR, project_dir)
    db_dir = tmp_path / "db"

    first = _index_once(project_dir, db_dir)
    assert first.returncode == 0, (
        f"initial index failed:\nstdout={first.stdout}\nstderr={first.stderr}"
    )

    second = _index_once(project_dir, db_dir)
    assert second.returncode == 0, (
        "re-indexing in a fresh process crashed — the Rust pipeline cannot "
        "reopen its own DB after a process restart:\n"
        f"stdout={second.stdout}\nstderr={second.stderr}"
    )
