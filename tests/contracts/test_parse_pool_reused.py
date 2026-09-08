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

from tests.contracts.pipeline_harness import (
    assert_identical,
    collect_chunk_tuples_from_duckdb,
    default_rust_config,
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "pipeline"


def _get_pid_probe(_args) -> int:
    """Module-level so ProcessPoolExecutor can pickle it."""
    return os.getpid()


class TestParsePoolReused:
    """The parse worker pool must be created once and reused, not per batch."""

    def test_parse_pool_reuses_worker_processes(self, monkeypatch):
        """The same worker process must serve calls across separate
        _get_parse_pool() lookups.

        `_get_parse_pool()` is the narrowest seam that exposes this: the
        externally observable contract ("no per-batch process respawn cost")
        has no other surface, since parse_batch_callback() doesn't return
        anything pool-related. Asserts on worker PID equality — an actual
        OS-level effect a caller would feel as wall-clock overhead if it
        regressed — not on pool object identity.

        Passes max_workers=1 so both submissions must land on the same
        single worker, making pid equality deterministic.
        """
        import chunkhound.pipeline_bridge as pipeline_bridge

        monkeypatch.setattr(pipeline_bridge, "_parse_pool", None)

        pool1 = pipeline_bridge._get_parse_pool(1)
        pid_a = pool1.submit(_get_pid_probe, None).result()

        pool2 = pipeline_bridge._get_parse_pool(1)
        pid_b = pool2.submit(_get_pid_probe, None).result()

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
        from chunkhound.pipeline_bridge import parse_batch_callback
        from chunkhound_native import IndexingPipeline  # type: ignore[import-untyped]

        files = sorted(f for f in FIXTURE_DIR.glob("*") if f.is_file())
        file_entries = [(str(f), f.name) for f in files]
        assert len(file_entries) >= 3, (
            "fixture must have enough files to force multiple batches"
        )

        db_single_batch = tmp_path / "db_single_batch"
        db_single_batch.mkdir()
        pipeline_single = IndexingPipeline(
            default_rust_config(db_single_batch, parse_batch_size=200)
        )
        report_single = pipeline_single.run(
            files=file_entries,
            parse_batch_callback=parse_batch_callback,
            embed_batch_callback=None,
            progress_callback=None,
            incremental=False,
        )

        db_many_batches = tmp_path / "db_many_batches"
        db_many_batches.mkdir()
        pipeline_many = IndexingPipeline(
            default_rust_config(db_many_batches, parse_batch_size=2)
        )
        report_many = pipeline_many.run(
            files=file_entries,
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
