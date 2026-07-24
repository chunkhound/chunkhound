"""Contract test: DB write failure surfaces cleanly (simple fail-fast case).

Design's `test_db_write_failure` contract wants: DB write fails -> pipeline
aborts, previously committed batches preserved. Reading
`src/db/duckdb_backend.rs::write_batch_incremental` shows each streamed batch
already runs in its own `BEGIN` / `COMMIT`+`CHECKPOINT` transaction with
`ROLLBACK` on error — so "previously committed batches survive a later
failure" already holds by inspection; there's no separate rollback logic to
add.

What this test covers instead is the simple, already-testable case: a DB path
that can never be opened at all (its parent path component is a regular file,
not a directory, so `std::fs::create_dir_all` fails before any pipeline
thread spawns). This proves the pipeline surfaces the failure as a clean
`RuntimeError` rather than hanging or panicking.

A test that injects a failure *between* successfully-committed batches (to
directly exercise the per-batch rollback path above) is deferred — it needs a
fault-injection seam in `DbBackend` that doesn't exist yet.
"""

import pytest
import tempfile
from pathlib import Path


FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "pipeline"


def _rust_config(project_root: Path, db_dir: Path) -> dict:
    return {
        "project_root": str(project_root.resolve()),
        "db_path": str(db_dir),
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


class TestDbWriteFailure:
    """DB path that can never be created must fail cleanly, not hang/crash."""

    @pytest.mark.asyncio
    async def test_uncreatable_db_path_raises_cleanly(self):
        try:
            from chunkhound_native import IndexingPipeline  # type: ignore[import-untyped]
        except ImportError:
            pytest.fail(
                "Rust IndexingPipeline is not yet available in chunkhound_native."
            )
        from chunkhound.pipeline_bridge import parse_batch_callback

        with tempfile.TemporaryDirectory() as tmp_root:
            # A regular file standing in the middle of the db_path — every
            # attempt to `create_dir_all` a path through it fails.
            blocker_file = Path(tmp_root) / "not_a_directory"
            blocker_file.write_text("this is a file, not a directory")
            db_dir = blocker_file / "db"

            files = sorted(FIXTURE_DIR.resolve().glob("*"))
            file_paths = [str(f) for f in files if f.is_file()]

            pipeline = IndexingPipeline(_rust_config(FIXTURE_DIR, db_dir))

            with pytest.raises(RuntimeError):
                pipeline.run(
                    files=file_paths,
                    parse_batch_callback=parse_batch_callback,
                    embed_batch_callback=None,
                    progress_callback=None,
                    incremental=False,
                )

            # The failure happens before any thread spawns — the blocker file
            # itself must be untouched (still a plain file, not corrupted).
            assert blocker_file.is_file()
            assert not db_dir.exists()
