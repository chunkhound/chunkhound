"""Contract test: a single file's parse exception must not abort the run.

Design's `test_parse_error_per_file` contract: one file raising during parsing
gets recorded as an error and the rest of the batch/run continues — it must
not take down the whole pipeline.

Before the fix, `chunkhound/pipeline_bridge.py`'s `_parse_one_file()` (the
`ProcessPoolExecutor` worker entry point) had no try/except, so an exception
from `parser.parse_file()` propagated out of `pool.map()` for the *whole
batch call*. In `src/pipeline/pipeline.rs`, that surfaced as a whole-batch
`Err` from `parse_one_batch()`, which the parse thread stored into the shared
fatal `error` mutex and aborted the entire run — `IndexingPipeline.run()`
raised a `RuntimeError` instead of returning a report.

Two tests, deliberately avoiding `ProcessPoolExecutor`/multiprocessing
entirely:

- `test_one_bad_file_does_not_abort_the_run` exercises the Rust-side contract
  end-to-end (item[2] error extraction -> `ParsedFile.error` ->
  `PipelineReport.errors`, no abort) via a test-local `parse_batch_callback`
  that calls the real `parse_file_callback` for good files and injects one
  error tuple for the bad file directly — this is the actual 3-tuple
  contract `_parse_one_file` implements, without going through a subprocess
  pool.
- `test_parse_one_file_catches_exceptions` calls the real
  `chunkhound.pipeline_bridge._parse_one_file` directly (in-process, no pool)
  to prove that exact function — which is what gets dispatched to
  `ProcessPoolExecutor` workers in production — never propagates an
  exception.

Both deliberately avoid going through `ProcessPoolExecutor.map()`: the
Python indexing path calls `multiprocessing.set_start_method("spawn")`
globally and permanently for the process (see
`IndexingCoordinator._ensure_mp_start_method`), so once any earlier test in
the same session exercises that path, monkeypatching a production module
attribute would silently stop working for any later fork-reliant test —
spawned workers re-import modules fresh and never see the patch. Testing at
these two boundaries instead is deterministic regardless of test order or
which multiprocessing start method is active.
"""

from pathlib import Path

import duckdb
import pytest


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


def _db_counts(db_dir: Path) -> dict:
    db_file = db_dir / "chunks.db"
    if not db_file.exists():
        return {"files": 0, "chunks": 0}
    conn = duckdb.connect(str(db_file))
    try:
        files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {"files": files, "chunks": chunks}
    finally:
        conn.close()


class TestParseErrorPerFile:
    """A single file's parse exception must be recorded, not fatal."""

    @pytest.mark.asyncio
    async def test_one_bad_file_does_not_abort_the_run(self, tmp_path):
        """End-to-end: the Rust pipeline must not abort, and must surface
        the bad file's error in PipelineReport.errors, when a batch callback
        returns an error tuple for one file — exactly the 3-tuple contract
        `_parse_one_file` implements in production.
        """
        try:
            from chunkhound_native import IndexingPipeline  # type: ignore[import-untyped]
        except ImportError:
            pytest.fail(
                "Rust IndexingPipeline is not yet available in chunkhound_native."
            )
        from chunkhound.pipeline_bridge import parse_file_callback

        good_a = tmp_path / "good_a.py"
        good_a.write_text("def a():\n    return 1\n")
        good_b = tmp_path / "good_b.py"
        good_b.write_text("def b():\n    return 2\n")
        bad_file = tmp_path / "broken.py"
        bad_file.write_text("def broken():\n    return 3\n")

        def batch_callback(file_paths, detect_embedded_sql=True):
            results = []
            for p in file_paths:
                if p == str(bad_file):
                    results.append(("", [], "simulated parser crash"))
                else:
                    lang, chunks = parse_file_callback(
                        p, detect_embedded_sql=detect_embedded_sql
                    )
                    results.append((lang, chunks, None))
            return results

        db_dir = tmp_path / "db"
        db_dir.mkdir(parents=True, exist_ok=True)

        file_paths = [str(good_a), str(good_b), str(bad_file)]

        pipeline = IndexingPipeline(_rust_config(tmp_path, db_dir))

        # Must not raise — the per-file error is collected, not fatal.
        report = pipeline.run(
            files=file_paths,
            parse_batch_callback=batch_callback,
            embed_batch_callback=None,
            progress_callback=None,
            incremental=False,
        )

        assert report.chunks_written > 0, "Chunks from the good files should still land"
        assert report.errors, (
            "The bad file's error should surface in PipelineReport.errors, "
            "not be silently dropped"
        )
        assert any("broken.py" in e for e in report.errors), (
            f"Expected an error mentioning broken.py, got: {report.errors}"
        )

        counts = _db_counts(db_dir)
        assert counts["chunks"] > 0

    def test_parse_one_file_catches_exceptions(self, monkeypatch, tmp_path):
        """The actual ProcessPoolExecutor worker entry point must never
        propagate an exception — called directly here (no pool) so the
        result is deterministic regardless of the process's multiprocessing
        start method.
        """
        import chunkhound.pipeline_bridge as pipeline_bridge

        bad_file = tmp_path / "broken.py"
        bad_file.write_text("def broken():\n    return 1\n")

        def _raising_parse_file_callback(file_path, detect_embedded_sql=True):
            raise RuntimeError("simulated parser crash")

        monkeypatch.setattr(
            pipeline_bridge, "parse_file_callback", _raising_parse_file_callback
        )

        lang, chunks, error = pipeline_bridge._parse_one_file((str(bad_file), True))

        assert chunks == []
        assert error is not None
        assert "simulated parser crash" in error
