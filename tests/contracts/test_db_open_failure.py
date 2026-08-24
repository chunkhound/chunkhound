"""Contract test: an unopenable DB path fails cleanly (pre-flight, not mid-run).

Covers the simple, already-testable case: a DB path that can never be opened
at all (its parent path component is a regular file, not a directory, so
`std::fs::create_dir_all` fails before any pipeline thread spawns). This
proves the pipeline surfaces the failure as a clean `RuntimeError` rather
than hanging or panicking.

This does NOT cover a DB write failing mid-run, partway through a real
indexing pass -- that would need to inject a failure *between*
successfully-committed batches, which needs a fault-injection seam in
`DbBackend` that doesn't exist yet (no `db_write_callback`-style hook, and
`create_backend()` hardcodes the concrete backend with no injection point).
Tracked as future work, not implemented here.

For what it's worth, reading `src/db/duckdb_backend.rs::write_batch_incremental`
shows each streamed batch already runs in its own `BEGIN` / `COMMIT`+`CHECKPOINT`
transaction with `ROLLBACK` on error -- so "previously committed batches
survive a later failure" already holds by inspection; there's no separate
rollback logic needed once that fault-injection seam exists.
"""

import tempfile
from pathlib import Path

import pytest

from tests.contracts.pipeline_harness import default_rust_config

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "pipeline"


class TestDbOpenFailure:
    """DB path that can never be created must fail cleanly, not hang/crash."""

    @pytest.mark.asyncio
    async def test_uncreatable_db_path_raises_cleanly(self):
        from chunkhound.pipeline_bridge import parse_batch_callback
        from chunkhound_native import IndexingPipeline  # type: ignore[import-untyped]

        with tempfile.TemporaryDirectory() as tmp_root:
            # A regular file standing in the middle of the db_path — every
            # attempt to `create_dir_all` a path through it fails.
            blocker_file = Path(tmp_root) / "not_a_directory"
            blocker_file.write_text("this is a file, not a directory")
            db_dir = blocker_file / "db"

            files = sorted(FIXTURE_DIR.resolve().glob("*"))
            file_entries = [(str(f), f.name) for f in files if f.is_file()]

            pipeline = IndexingPipeline(default_rust_config(db_dir))

            with pytest.raises(RuntimeError):
                pipeline.run(
                    files=file_entries,
                    parse_batch_callback=parse_batch_callback,
                    embed_batch_callback=None,
                    progress_callback=None,
                    incremental=False,
                )

            # The failure happens before any thread spawns — the blocker file
            # itself must be untouched (still a plain file, not corrupted).
            assert blocker_file.is_file()
            assert not db_dir.exists()
