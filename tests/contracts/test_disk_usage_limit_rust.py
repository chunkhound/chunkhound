"""Contract test: the Rust pipeline enforces disk_usage_limit_mb mid-run.

Mirrors `tests/unit/test_disk_usage_limit.py`'s Python-path contract
(`test_store_parsed_results_disk_limit_exceeded`) for the Rust pipeline:
`disk_usage_limit_mb=0.0` deterministically trips on the very first batch
(any non-empty DB file already has size >= 0.0), so no rows get written and
`PipelineReport.disk_limit_exceeded` is set — as data, never a raised
exception.
"""

import tempfile
from pathlib import Path

import pytest

from tests.contracts.pipeline_harness import index_with_rust

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "pipeline"


def _write_many_fixture_files(project_dir: Path, count: int) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (project_dir / f"mod_{i}.py").write_text(f"def fn_{i}():\n    return {i}\n")


class TestDiskUsageLimitRust:
    @pytest.mark.asyncio
    async def test_zero_limit_stops_all_writes_without_raising(self):
        """disk_usage_limit_mb=0.0 must stop writes and report the trip as data."""
        with tempfile.TemporaryDirectory() as tmp:
            db_dir = Path(tmp) / "db"

            # No exception should propagate out of index_with_rust()/pipeline.run() —
            # Python's own DiskUsageLimitExceededError contract is a returned
            # status, never a raised exception, and the Rust mirror must match.
            result = index_with_rust(
                FIXTURE_DIR, db_dir, skip_embeddings=True, disk_usage_limit_mb=0.0
            )

            assert result.disk_limit_exceeded is True
            assert result.disk_limit_current_mb is not None
            assert result.disk_limit_max_mb == 0.0
            assert result.chunks_written == 0, (
                "no chunks should be written once the very first pre-write "
                "check trips"
            )

    @pytest.mark.asyncio
    async def test_no_limit_configured_writes_normally(self):
        """disk_usage_limit_mb=None (the default) must not affect a normal run."""
        with tempfile.TemporaryDirectory() as tmp:
            db_dir = Path(tmp) / "db"

            result = index_with_rust(FIXTURE_DIR, db_dir, skip_embeddings=True)

            assert result.disk_limit_exceeded is False
            assert result.disk_limit_current_mb is None
            assert result.disk_limit_max_mb is None
            assert result.chunks_written > 0

    @pytest.mark.asyncio
    async def test_disk_limit_trip_skips_compaction(self):
        """A disk-limit trip must never trigger compaction (which would
        transiently roughly double disk usage via its EXPORT/IMPORT rewrite —
        the opposite of what a disk-limit trip should do). Forces compaction
        eligibility (compaction_threshold=0.0, compaction_min_size_mb=0 --
        the same forcing technique as test_compaction_before_index.py) at the
        same time as disk_usage_limit_mb=0.0, so without the fix
        "write-compact" would fire; with it, only "write-index" (the cheap
        HNSW-only rebuild) must fire instead.
        """
        with tempfile.TemporaryDirectory() as tmp:
            db_dir = Path(tmp) / "db"

            phases_seen: list[str] = []

            def progress_callback(phase: str, current: int, total: int) -> None:
                phases_seen.append(phase)

            result = index_with_rust(
                FIXTURE_DIR,
                db_dir,
                skip_embeddings=True,
                disk_usage_limit_mb=0.0,
                compaction_threshold=0.0,
                compaction_min_size_mb=0,
                progress_callback=progress_callback,
            )

            assert result.disk_limit_exceeded is True
            assert "write-compact" not in phases_seen, (
                "compaction must not run after a disk-limit trip -- it would "
                "transiently roughly double disk usage right after "
                "determining disk is already over budget"
            )
            assert "write-index" in phases_seen, (
                "the cheap HNSW-only rebuild must still run in compaction's place"
            )

    @pytest.mark.asyncio
    async def test_disk_limit_trip_drains_in_flight_batches_without_hanging(self):
        """A trip must cleanly cancel every batch already queued upstream,
        not just the trivial single-batch case the other tests above cover.

        The standard 5-file fixture with the default parse_batch_size=200
        always collapses into exactly one batch, so store's `drop(store_rx)`
        on trip has nothing queued to cancel -- the cooperative-shutdown
        cascade (embed's `store_tx.send()` failing, which drops `parse_rx`,
        which fails parse's `parse_tx.send()`) never actually exercises
        multiple in-flight batches contending on the bounded channels. Here,
        many files with parse_batch_size=1 force dozens of concurrent
        batches; disk_usage_limit_mb=0.0 trips before the store thread's very
        first receive (backend.open() already leaves a non-empty chunks.db).
        If the shutdown cascade were broken -- e.g. the drop happening too
        late, or not at all -- parse/embed would block forever on a full
        store_tx (capacity 8) with nothing draining it, and this test would
        hang instead of completing.
        """
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "project"
            _write_many_fixture_files(project_dir, count=50)
            db_dir = Path(tmp) / "db"

            result = index_with_rust(
                project_dir,
                db_dir,
                skip_embeddings=True,
                parse_batch_size=1,
                disk_usage_limit_mb=0.0,
            )

            assert result.disk_limit_exceeded is True
            assert result.chunks_written == 0, (
                "the trip fires before the store thread's first receive, so "
                "none of the ~50 in-flight batches should land"
            )
