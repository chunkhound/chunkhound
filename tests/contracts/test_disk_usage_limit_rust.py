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
