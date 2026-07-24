"""Contract test: the parse worker pool must persist across batches/runs.

Gap this guards against: `parse_batch_callback()` is called once per Rust
parse batch (default 200 files/batch). Before this fix, it opened a fresh
`ProcessPoolExecutor` in a `with` block on every call and tore it down at
the end of that same call — spawning and killing a whole set of worker
processes ~50 times over a 10K-file index instead of once. That defeats the
whole point of using persistent worker processes (see design doc §10, "Rayon
threads vs ProcessPoolExecutor?" — process spin-up cost should be paid once,
not per batch).

The fix makes the pool a lazy, process-wide singleton (`_get_parse_pool()`),
created on first use and reused for every subsequent call — both within one
`IndexingPipeline.run()` and across multiple runs in the same process.

Also formalizes item 15 (callback contracts): the design originally imagined
Rust parallelizing per-file via rayon; the implementation instead made the
callback batch-shaped, with Python doing the intra-batch fan-out via
`ProcessPoolExecutor`. `test_pipeline_output_identical_regardless_of_batch_size`
pins down that this batch-shaped contract produces identical results no
matter how a run is chopped into batches — protecting against the shared
pool silently leaking state across batches.
"""

import os
from pathlib import Path

import pytest

from tests.contracts.pipeline_harness import (
    assert_identical,
    collect_chunk_tuples_from_duckdb,
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "pipeline"


def _rust_config(project_root: Path, db_dir: Path, *, parse_batch_size: int) -> dict:
    return {
        "project_root": str(project_root.resolve()),
        "db_path": str(db_dir.resolve()),
        "db_batch_size": 100,
        "compaction_threshold": 0.60,
        "compaction_batch_threshold": 10,
        "compaction_min_size_mb": 10,
        "parse_batch_size": parse_batch_size,
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


def _get_pid_probe(_args) -> int:
    """Module-level so ProcessPoolExecutor can pickle it."""
    return os.getpid()


class TestParsePoolReused:
    """The parse worker pool must be created once and reused, not per batch."""

    def test_parse_pool_constructed_once_across_calls(self, monkeypatch, tmp_path):
        """Two parse_batch_callback() calls must construct the pool only once.

        Uses a counting fake instead of a real ProcessPoolExecutor — fast,
        deterministic, and avoids the spawn/monkeypatch-freshness trap
        documented in test_parse_error_per_file.py (spawned workers re-import
        modules fresh and never see a patch made in the parent process).
        """
        import chunkhound.pipeline_bridge as pipeline_bridge

        construct_count = 0

        class _CountingPool:
            def __init__(self, *args, **kwargs):
                nonlocal construct_count
                construct_count += 1

            def map(self, fn, args_iterable):
                return [fn(a) for a in args_iterable]

            def shutdown(self, *args, **kwargs):
                pass

        monkeypatch.setattr(pipeline_bridge, "_parse_pool", None)
        monkeypatch.setattr(pipeline_bridge, "ProcessPoolExecutor", _CountingPool)

        file_a = tmp_path / "a.py"
        file_a.write_text("def a():\n    return 1\n")
        file_b = tmp_path / "b.py"
        file_b.write_text("def b():\n    return 2\n")

        pipeline_bridge.parse_batch_callback([str(file_a)])
        pipeline_bridge.parse_batch_callback([str(file_b)])

        assert construct_count == 1, (
            f"expected the pool to be constructed once and reused, "
            f"got {construct_count} constructions"
        )

    def test_parse_pool_reuses_worker_processes(self, monkeypatch):
        """The same worker process must serve calls across separate
        _get_parse_pool() lookups — proves actual OS-level reuse, not just
        object identity.

        Passes max_workers=1 so both submissions must land on the same
        single worker, making pid equality deterministic.
        """
        import chunkhound.pipeline_bridge as pipeline_bridge

        monkeypatch.setattr(pipeline_bridge, "_parse_pool", None)

        pool1 = pipeline_bridge._get_parse_pool(1)
        pid_a = pool1.submit(_get_pid_probe, None).result()

        pool2 = pipeline_bridge._get_parse_pool(1)
        pid_b = pool2.submit(_get_pid_probe, None).result()

        assert pool1 is pool2, "expected the same pool instance to be returned"
        assert pid_a == pid_b, (
            "expected the same worker process to serve both calls "
            f"(got pids {pid_a} and {pid_b})"
        )

    def test_pipeline_output_identical_regardless_of_batch_size(self, tmp_path):
        """Splitting one run into many small batches (all hitting the shared
        parse pool) must produce identical output to a single big batch.

        Pins down the item-15 batch-shaped callback contract: results must
        not depend on how a run happens to be chopped into
        parse_batch_callback() calls.
        """
        try:
            from chunkhound_native import (  # type: ignore[import-untyped]
                IndexingPipeline,
            )
        except ImportError:
            pytest.fail(
                "Rust IndexingPipeline is not yet available in chunkhound_native."
            )
        from chunkhound.pipeline_bridge import parse_batch_callback

        files = sorted(f for f in FIXTURE_DIR.glob("*") if f.is_file())
        file_paths = [str(f) for f in files]
        assert len(file_paths) >= 3, (
            "fixture must have enough files to force multiple batches"
        )

        db_single_batch = tmp_path / "db_single_batch"
        db_single_batch.mkdir()
        pipeline_single = IndexingPipeline(
            _rust_config(FIXTURE_DIR, db_single_batch, parse_batch_size=200)
        )
        report_single = pipeline_single.run(
            files=file_paths,
            parse_batch_callback=parse_batch_callback,
            embed_batch_callback=None,
            progress_callback=None,
            incremental=False,
        )

        db_many_batches = tmp_path / "db_many_batches"
        db_many_batches.mkdir()
        pipeline_many = IndexingPipeline(
            _rust_config(FIXTURE_DIR, db_many_batches, parse_batch_size=2)
        )
        report_many = pipeline_many.run(
            files=file_paths,
            parse_batch_callback=parse_batch_callback,
            embed_batch_callback=None,
            progress_callback=None,
            incremental=False,
        )

        assert report_single.chunks_written == report_many.chunks_written
        assert report_single.files_processed == report_many.files_processed

        from tests.contracts.pipeline_harness import IndexResult

        result_single = IndexResult(
            files_processed=report_single.files_processed,
            chunks_written=report_single.chunks_written,
            embeddings_generated=report_single.embeddings_generated,
            chunk_tuples=collect_chunk_tuples_from_duckdb(db_single_batch),
        )
        result_many = IndexResult(
            files_processed=report_many.files_processed,
            chunks_written=report_many.chunks_written,
            embeddings_generated=report_many.embeddings_generated,
            chunk_tuples=collect_chunk_tuples_from_duckdb(db_many_batches),
        )
        assert_identical(result_single, result_many)
