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

import tempfile
from pathlib import Path

import pytest

from tests.contracts.pipeline_harness import collect_table_counts, default_rust_config

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "pipeline"


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

            pipeline = IndexingPipeline(default_rust_config(empty_dir, db_dir))
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

            pipeline = IndexingPipeline(default_rust_config(FIXTURE_DIR, db_dir))
            first_report = pipeline.run(
                files=file_paths,
                parse_batch_callback=parse_batch_callback,
                embed_batch_callback=None,
                progress_callback=None,
                incremental=False,
            )
            assert first_report.chunks_written > 0

            before = collect_table_counts(db_dir)
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

            after = collect_table_counts(db_dir)
            assert after["files"] == 0, (
                f"Expected orphaned files cleaned up, found {after['files']}"
            )
            assert after["chunks"] == 0, (
                f"Expected orphaned chunks cleaned up, found {after['chunks']}"
            )
            assert after["embeddings"] == 0
