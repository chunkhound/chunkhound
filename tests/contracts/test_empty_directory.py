"""Contract test: indexing an empty directory.

Covers the design's `test_empty_directory` contract:
1. A fresh, empty directory against an empty DB produces a report of 0 files,
   0 chunks, with no crash.
2. A directory that previously had files, now emptied, cleans up all
   orphaned files/chunks/embeddings when re-indexed incrementally.

`IndexingPipeline::run()` used to return early on an empty file list without
ever touching the DB, so scenario 2 never actually deleted anything. Fixed in
`src/pipeline/pipeline.rs` — an empty file list still flows through the
incremental diff + streaming pipeline so pending deletes get flushed.
"""

import duckdb
import pytest
import tempfile
from pathlib import Path


FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "pipeline"


def _rust_config(project_root: Path, db_dir: Path) -> dict:
    return {
        "project_root": str(project_root.resolve()),
        "db_path": str(db_dir.resolve()),
        "db_batch_size": 100,
        "compaction_threshold": 0.60,
        "compaction_batch_threshold": 10,
        "compaction_min_size_mb": 10,
        "parse_batch_size": 200,
        "parse_thread_pool_size": 4,
        "embed_batch_size": 200,
        "force_reindex": False,
        "mtime_epsilon_seconds": 0.01,
        "skip_cleanup": False,
        "skip_embeddings": True,
        "per_file_timeout_secs": 3.0,
        "per_file_timeout_min_size_kb": 128,
        "detect_embedded_sql": True,
        "config_file_size_threshold_kb": 20,
        "embedding_provider": "",
        "embedding_model": "",
    }


def _table_counts(db_dir: Path) -> dict:
    """Row counts for files/chunks/any embeddings_* table."""
    db_file = db_dir / "chunks.db"
    if not db_file.exists():
        return {"files": 0, "chunks": 0, "embeddings": 0}

    conn = duckdb.connect(str(db_file))
    try:
        files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'embeddings_%'"
        ).fetchall()
        embeddings = 0
        for (table_name,) in tables:
            embeddings += conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        return {"files": files, "chunks": chunks, "embeddings": embeddings}
    finally:
        conn.close()


def _get_rust_pipeline():
    try:
        from chunkhound_native import IndexingPipeline  # type: ignore[import-untyped]
    except ImportError:
        raise NotImplementedError(
            "Rust IndexingPipeline is not yet available in chunkhound_native."
        ) from None
    return IndexingPipeline


class TestEmptyDirectory:
    """Indexing an empty directory must not crash and must clean up orphans."""

    @pytest.mark.asyncio
    async def test_fresh_empty_directory(self):
        """Empty dir, empty DB -> 0 files, 0 chunks, no crash."""
        IndexingPipeline = _get_rust_pipeline()
        from chunkhound.pipeline_bridge import parse_batch_callback

        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            tempfile.TemporaryDirectory() as tmp_db,
        ):
            empty_dir = Path(tmp_dir)
            db_dir = Path(tmp_db) / "db"

            pipeline = IndexingPipeline(_rust_config(empty_dir, db_dir))
            report = pipeline.run(
                files=[],
                parse_batch_callback=parse_batch_callback,
                embed_batch_callback=None,
                progress_callback=None,
                incremental=True,
            )

            assert report.files_processed == 0
            assert report.chunks_written == 0

    @pytest.mark.asyncio
    async def test_reindex_after_all_files_removed_cleans_up_orphans(self):
        """Fixture indexed normally, then re-run with files=[] (incremental)
        -> all files/chunks/embeddings are deleted from the DB.
        """
        IndexingPipeline = _get_rust_pipeline()
        from chunkhound.pipeline_bridge import parse_batch_callback

        with tempfile.TemporaryDirectory() as tmp_db:
            db_dir = Path(tmp_db) / "db"
            db_dir.mkdir(parents=True, exist_ok=True)

            files = sorted(FIXTURE_DIR.resolve().glob("*"))
            file_paths = [str(f) for f in files if f.is_file()]

            pipeline = IndexingPipeline(_rust_config(FIXTURE_DIR, db_dir))
            first_report = pipeline.run(
                files=file_paths,
                parse_batch_callback=parse_batch_callback,
                embed_batch_callback=None,
                progress_callback=None,
                incremental=False,
            )
            assert first_report.chunks_written > 0

            before = _table_counts(db_dir)
            assert before["files"] > 0
            assert before["chunks"] > 0

            # Re-run with an empty file list, incremental=True — simulates the
            # fixture directory having been emptied since the last index.
            second_report = pipeline.run(
                files=[],
                parse_batch_callback=parse_batch_callback,
                embed_batch_callback=None,
                progress_callback=None,
                incremental=True,
            )
            assert second_report.files_processed == 0
            assert second_report.chunks_written == 0

            after = _table_counts(db_dir)
            assert after["files"] == 0, (
                f"Expected orphaned files cleaned up, found {after['files']}"
            )
            assert after["chunks"] == 0, (
                f"Expected orphaned chunks cleaned up, found {after['chunks']}"
            )
            assert after["embeddings"] == 0
