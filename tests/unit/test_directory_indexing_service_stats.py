"""Contract: the reported embeddings_generated count must never drop below
what the Rust pipeline actually embedded, even when an unrelated per-file
error triggers the missing-embeddings backfill pass.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from chunkhound.services.directory_indexing_service import DirectoryIndexingService


class _FakeCoordinator:
    """Minimal coordinator stub covering only what process_directory() touches
    for a Rust-pipeline run with unrelated file errors."""

    def __init__(self, process_result: dict, backfill_result: dict):
        self._process_result = process_result
        self.resolve_rust_pipeline_decision = lambda log_reason=True: True
        self.process_directory = AsyncMock(return_value=process_result)
        self.generate_missing_embeddings = AsyncMock(return_value=backfill_result)
        self.compact_database_with_metrics = AsyncMock(
            return_value={"status": "skipped"}
        )
        self.clear_compaction_skip = MagicMock()


class _FakeConfig:
    class _Indexing:
        include: list[str] = []
        exclude: list[str] = []
        config_file_size_threshold_kb = 512

    indexing = _Indexing()


@pytest.mark.asyncio
async def test_unrelated_rust_errors_do_not_zero_out_embeddings_count():
    """Rust embedded everything inline (embeddings_generated=N) but reported
    321 unrelated file errors (e.g. permission/parse failures). That triggers
    the backfill pass, which finds nothing missing and returns generated=0.
    The final stat must still reflect Rust's real count, not be clobbered."""
    process_result = {
        "status": "success",
        "pipeline": "rust",
        "files_processed": 260_967,
        "total_chunks": 3_677_386,
        "embeddings_generated": 3_677_386,
        "errors": 321,
        "skipped": 0,
        "skipped_due_to_timeout": [],
        "skipped_unchanged": 0,
        "skipped_filtered": 0,
    }
    backfill_result = {"status": "up_to_date", "generated": 0}

    coordinator = _FakeCoordinator(process_result, backfill_result)
    service = DirectoryIndexingService(coordinator, _FakeConfig())

    stats = await service.process_directory(Path("/does/not/matter"))

    assert stats.embeddings_generated == 3_677_386
    coordinator.generate_missing_embeddings.assert_awaited_once()
    # Backfill found nothing to generate -- the compaction skip (Rust already
    # compacted this run) must NOT be invalidated, or every successful Rust
    # run would pay for a redundant real compaction.
    coordinator.clear_compaction_skip.assert_not_called()


@pytest.mark.asyncio
async def test_backfill_additions_are_added_not_overwritten():
    """When the backfill pass genuinely finds and embeds a few stragglers,
    those should add to Rust's count, not replace it."""
    process_result = {
        "status": "success",
        "pipeline": "rust",
        "files_processed": 100,
        "total_chunks": 1_000,
        "embeddings_generated": 990,
        "errors": 3,
        "skipped": 0,
        "skipped_due_to_timeout": [],
        "skipped_unchanged": 0,
        "skipped_filtered": 0,
    }
    backfill_result = {"status": "success", "generated": 10}

    coordinator = _FakeCoordinator(process_result, backfill_result)
    service = DirectoryIndexingService(coordinator, _FakeConfig())

    stats = await service.process_directory(Path("/does/not/matter"))

    assert stats.embeddings_generated == 1_000
    # Backfill actually wrote new embeddings after Rust's own compaction ran
    # -- those rows were never compacted, so the skip must be cleared so the
    # second compaction boundary runs for real. Regression test for PR #380
    # review finding #4 (stale _skip_compaction silently no-ops here).
    coordinator.clear_compaction_skip.assert_called_once()
