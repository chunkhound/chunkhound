"""Contract test — upsert idempotency and update-in-place semantics.

Migrated from the deleted ``tests/test_rust_db_writer.py``'s ``TestUpsert``
class (the old direct-writer test suite for the since-removed
``RustDbWriter``). Re-implemented against the full pipeline (diff detection
-> parse -> embed -> write) rather than a low-level backend call, since the
Python-visible contract is "reindexing a directory doesn't duplicate rows
and correctly updates changed files in place" — not any specific writer-API
call sequence, which no longer exists as a standalone concept in the new
pipeline (see ``tests/contracts/pipeline_harness.py``: ``IndexingPipeline``
exposes a single ``run()`` call, with no separate open/write-batch API for
Python to drive directly).

Note: the old suite's ``TestHnswBoundary`` (below/at-50-embeddings gating
a per-batch HNSW drop/recreate cycle) is intentionally NOT migrated here —
that concept doesn't exist in the new architecture. The new pipeline drops
all HNSW indexes once before the whole run and rebuilds them once after
(see ``src/pipeline/pipeline.rs``), regardless of embedding count, so there
is no threshold-gated behavior left to test.
"""

import shutil
from pathlib import Path

import duckdb
import pytest

from tests.contracts.pipeline_harness import index_with_rust

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "pipeline"


@pytest.fixture
def fixture_dir() -> Path:
    return FIXTURE_DIR


def _table_state(db_dir: Path) -> dict:
    """Snapshot file ids by path plus row counts for files/chunks/embeddings_*."""
    db_file = db_dir / "chunks.db"
    conn = duckdb.connect(str(db_file))
    try:
        file_ids = dict(conn.execute("SELECT path, id FROM files").fetchall())
        files_count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        chunks_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        emb_tables = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_name LIKE 'embeddings_%'"
        ).fetchall()
        embeddings_count = sum(
            conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            for (t,) in emb_tables
        )
        return {
            "file_ids": file_ids,
            "files_count": files_count,
            "chunks_count": chunks_count,
            "embeddings_count": embeddings_count,
        }
    finally:
        conn.close()


def _chunk_ids_for_path(db_dir: Path, rel_path: str) -> set[int]:
    db_file = db_dir / "chunks.db"
    conn = duckdb.connect(str(db_file))
    try:
        rows = conn.execute(
            """
            SELECT c.id FROM chunks c JOIN files f ON f.id = c.file_id
            WHERE f.path = ?
            """,
            [rel_path],
        ).fetchall()
    finally:
        conn.close()
    return {int(r[0]) for r in rows}


class TestUpsertNoDuplication:
    """Reindexing must upsert existing rows in place, never duplicate them."""

    def test_unchanged_rerun_is_a_true_no_op(self, fixture_dir: Path, tmp_path: Path):
        """Re-running on an untouched, already-indexed directory must not
        change row counts or file ids anywhere in the database.
        """
        work_dir = tmp_path / "fixtures"
        shutil.copytree(fixture_dir, work_dir)
        db_dir = tmp_path / "db"
        db_dir.mkdir()

        index_with_rust(work_dir, db_dir, skip_embeddings=False)
        before = _table_state(db_dir)
        assert before["files_count"] > 0, "expected the fixture to produce file rows"
        assert before["chunks_count"] > 0, "expected the fixture to produce chunk rows"

        index_with_rust(work_dir, db_dir, skip_embeddings=False, incremental=True)
        after = _table_state(db_dir)

        assert after["files_count"] == before["files_count"], (
            "files table grew on a no-op rerun — rows were duplicated"
        )
        assert after["chunks_count"] == before["chunks_count"], (
            "chunks table grew on a no-op rerun — rows were duplicated"
        )
        assert after["embeddings_count"] == before["embeddings_count"], (
            "embeddings table grew on a no-op rerun — rows were duplicated"
        )
        assert after["file_ids"] == before["file_ids"], (
            "file ids changed on a no-op rerun — rows were deleted and "
            "reinserted instead of left alone"
        )

    def test_modified_file_replaces_chunks_not_appends(
        self, fixture_dir: Path, tmp_path: Path
    ):
        """Editing one file and reindexing must replace its chunk rows in
        place (same file id, old chunk ids gone) — not append new chunks
        alongside the stale ones.
        """
        work_dir = tmp_path / "fixtures"
        shutil.copytree(fixture_dir, work_dir)
        db_dir = tmp_path / "db"
        db_dir.mkdir()

        index_with_rust(work_dir, db_dir, skip_embeddings=False)
        before = _table_state(db_dir)
        old_chunk_ids = _chunk_ids_for_path(db_dir, "main.py")
        assert old_chunk_ids, "expected main.py to have chunks before modification"

        main_py = work_dir / "main.py"
        main_py.write_text(
            main_py.read_text()
            + "\n\ndef upsert_test_func():\n    return 'added for upsert test'\n"
        )

        index_with_rust(work_dir, db_dir, skip_embeddings=False, incremental=True)
        after = _table_state(db_dir)
        new_chunk_ids = _chunk_ids_for_path(db_dir, "main.py")

        assert after["files_count"] == before["files_count"], (
            "files table grew — the modified file was inserted as a new "
            "row instead of updating the existing one"
        )
        assert after["file_ids"]["main.py"] == before["file_ids"]["main.py"], (
            "main.py's file id changed — it was deleted and reinserted "
            "instead of updated in place"
        )
        assert new_chunk_ids, "expected main.py to still have chunks after modification"
        assert not (old_chunk_ids & new_chunk_ids), (
            "old chunk rows for main.py are still present after re-indexing "
            "— chunks were appended instead of replaced"
        )
