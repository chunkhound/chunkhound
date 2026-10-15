"""Force-reindex crash window: dirty file rows must be rewritten incrementally.

If the process dies after ``pre_delete_for_upsert`` commits and before the
insert transaction, chunks are gone and the file row is marked dirty
(NULL ``modified_time`` and ``content_hash``). The next incremental index
must treat that as "changed" and restore chunks — not skip because disk
mtime still matches a stale non-NULL DB mtime.
"""

import shutil
from pathlib import Path

import duckdb

from tests.contracts.pipeline_harness import (
    assert_chunk_multiset_identical,
    collect_chunk_tuples_from_duckdb,
    index_with_rust,
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "pipeline"


def test_incremental_restores_chunks_after_dirty_pre_delete(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "fixtures"
    shutil.copytree(FIXTURE_DIR, work_dir)
    db_dir = tmp_path / "db"

    first = index_with_rust(work_dir, db_dir, skip_embeddings=True, incremental=False)
    assert first.chunks_written > 0, "baseline force-reindex should produce chunks"
    chunks_before = collect_chunk_tuples_from_duckdb(db_dir)
    assert chunks_before, "baseline DB must contain chunks"

    db_file = db_dir / "chunks.db"
    conn = duckdb.connect(str(db_file))
    try:
        conn.execute("DELETE FROM chunks")
        conn.execute("UPDATE files SET modified_time = NULL, content_hash = NULL")
        conn.commit()
        remaining = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        assert remaining is not None and remaining[0] == 0
    finally:
        conn.close()

    index_with_rust(work_dir, db_dir, skip_embeddings=True, incremental=True)
    chunks_after = collect_chunk_tuples_from_duckdb(db_dir)
    assert_chunk_multiset_identical(
        chunks_before,
        chunks_after,
        label_a="pre-crash baseline",
        label_b="post-recovery",
    )
