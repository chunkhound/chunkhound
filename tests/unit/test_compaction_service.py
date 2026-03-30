"""Tests for CompactionService.

Tests compaction logic including storage statistics, blocking/background modes,
atomic swap, and graceful shutdown. Uses real components where possible,
only mocking external dependencies like disk operations.
"""

import asyncio
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chunkhound.core.config.config import Config
from chunkhound.core.config.database_config import DatabaseConfig
from chunkhound.core.models import Chunk, File
from chunkhound.core.types.common import ChunkType, Language
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.compaction_service import CompactionService


@pytest.fixture
def config_with_compaction(tmp_path: Path) -> Config:
    """Create a config with compaction enabled."""
    db_config = DatabaseConfig(
        path=tmp_path / ".chunkhound",
        compaction_enabled=True,
        compaction_threshold=0.5,
        compaction_min_size_mb=1,  # Low for testing
    )
    return Config(database=db_config, target_dir=tmp_path)


@pytest.fixture
def config_compaction_disabled(tmp_path: Path) -> Config:
    """Create a config with compaction disabled."""
    db_config = DatabaseConfig(
        path=tmp_path / ".chunkhound",
        compaction_enabled=False,
    )
    return Config(database=db_config, target_dir=tmp_path)


@pytest.fixture
def mock_provider() -> MagicMock:
    """Create a mock DuckDB provider with configurable should_compact behavior."""
    provider = MagicMock()
    # Default: high fragmentation (should compact)
    provider.should_compact.return_value = (
        True,
        {
            "_raw_fragmentation_ratio": 0.6,
            "free_blocks": 1000,
            "block_size": 262144,  # ~256KB per block
            "total_blocks": 2000,
            "used_blocks": 1000,
            "row_waste_ratio": 0.0,
        },
    )
    provider.disconnect = MagicMock()
    provider.connect = MagicMock()
    return provider


class TestStorageStats:
    """Test storage statistics and compaction decision logic."""

    def test_should_compact_true_above_threshold(
        self, tmp_path: Path, config_with_compaction: Config, mock_provider: MagicMock
    ):
        """Compaction triggers when fragmentation >= threshold."""
        db_path = tmp_path / "test.duckdb"
        db_path.touch()

        service = CompactionService(db_path, config_with_compaction)

        # Provider returns fragmentation above threshold (0.6 > 0.5)
        mock_provider.should_compact.return_value = (
            True,
            {
                "_raw_fragmentation_ratio": 0.6,
                "free_blocks": 1000,
                "block_size": 262144,
                "total_blocks": 2000,
                "used_blocks": 1000,
            },
        )

        should, stats = service.check_should_compact(mock_provider)
        assert should is True
        assert stats["_raw_fragmentation_ratio"] == 0.6

    def test_should_compact_false_below_threshold(
        self, tmp_path: Path, config_with_compaction: Config, mock_provider: MagicMock
    ):
        """Compaction skipped when fragmentation < threshold."""
        db_path = tmp_path / "test.duckdb"
        db_path.touch()

        service = CompactionService(db_path, config_with_compaction)

        # Provider returns fragmentation below threshold (0.3 < 0.5)
        mock_provider.should_compact.return_value = (
            False,
            {
                "_raw_fragmentation_ratio": 0.3,
                "free_blocks": 500,
                "block_size": 262144,
                "total_blocks": 2000,
                "used_blocks": 1500,
            },
        )

        should, stats = service.check_should_compact(mock_provider)
        assert should is False

    def test_should_compact_respects_min_size(
        self, tmp_path: Path, mock_provider: MagicMock
    ):
        """Compaction skipped when reclaimable < min_size_mb."""
        db_path = tmp_path / "test.duckdb"
        db_path.touch()

        # Config with high min_size_mb (100MB)
        db_config = DatabaseConfig(
            path=tmp_path / ".chunkhound",
            compaction_enabled=True,
            compaction_threshold=0.5,
            compaction_min_size_mb=100,  # High threshold
        )
        config = Config(database=db_config, target_dir=tmp_path)

        service = CompactionService(db_path, config)

        # Provider returns high fragmentation but low reclaimable space
        # 100 blocks * 262144 bytes = ~26MB (less than 100MB min)
        mock_provider.should_compact.return_value = (
            True,
            {
                "_raw_fragmentation_ratio": 0.6,
                "free_blocks": 100,
                "block_size": 262144,
                "total_blocks": 200,
                "used_blocks": 100,
            },
        )

        should, stats = service.check_should_compact(mock_provider)
        assert should is False

    # Note: Disk space check is now delegated to provider.optimize()
    # and tested in provider tests, not service tests

    def test_should_compact_false_when_disabled(
        self,
        tmp_path: Path,
        config_compaction_disabled: Config,
        mock_provider: MagicMock,
    ):
        """Compaction skipped when disabled in config."""
        db_path = tmp_path / "test.duckdb"
        db_path.touch()

        service = CompactionService(db_path, config_compaction_disabled)

        should, stats = service.check_should_compact(mock_provider)
        assert should is False
        assert stats == {}

    def test_should_compact_false_when_already_in_progress(
        self, tmp_path: Path, config_with_compaction: Config, mock_provider: MagicMock
    ):
        """Compaction skipped when another compaction is running."""
        db_path = tmp_path / "test.duckdb"
        db_path.touch()

        service = CompactionService(db_path, config_with_compaction)
        service._compaction_in_progress = True

        should, stats = service.check_should_compact(mock_provider)
        assert should is False
        assert stats == {}


class TestBlockingCompaction:
    """Test blocking compaction mode (CLI)."""

    @pytest.mark.asyncio
    async def test_blocking_compaction_waits_for_completion(
        self, tmp_path: Path, config_with_compaction: Config, mock_provider: MagicMock
    ):
        """compact_blocking() returns only after compaction finishes."""
        db_path = tmp_path / "test.duckdb"
        db_path.write_bytes(b"x" * 1024)

        service = CompactionService(db_path, config_with_compaction)

        # Track compaction execution
        compaction_started = asyncio.Event()
        compaction_finished = asyncio.Event()

        async def mock_do_compaction(provider):
            compaction_started.set()
            await asyncio.sleep(0.05)  # Simulate work
            compaction_finished.set()
            return True

        with patch.object(service, "_do_compaction", mock_do_compaction):
            result = await service.compact_blocking(mock_provider)

            # Should have completed
            assert compaction_finished.is_set()
            assert result is True

    @pytest.mark.asyncio
    async def test_blocking_compaction_no_concurrent_calls(
        self, tmp_path: Path, config_with_compaction: Config, mock_provider: MagicMock
    ):
        """Second compaction request rejected while one is running."""
        db_path = tmp_path / "test.duckdb"
        db_path.write_bytes(b"x" * 1024)

        service = CompactionService(db_path, config_with_compaction)

        # Simulate an in-progress compaction
        service._compaction_in_progress = True

        # This should be rejected immediately
        result = await service.compact_blocking(mock_provider)
        assert result is False

    @pytest.mark.asyncio
    async def test_blocking_compaction_returns_false_on_cancellation(
        self, tmp_path: Path, config_with_compaction: Config, mock_provider: MagicMock
    ):
        """compact_blocking() propagates False when _do_compaction is cancelled."""
        db_path = tmp_path / "test.duckdb"
        db_path.write_bytes(b"x" * 1024)

        service = CompactionService(db_path, config_with_compaction)

        async def cancelled_compaction(provider):
            return False  # Cancelled, not an error

        with patch.object(service, "_do_compaction", cancelled_compaction):
            result = await service.compact_blocking(mock_provider)
            assert result is False

    @pytest.mark.asyncio
    async def test_blocking_compaction_returns_false_when_not_needed(
        self, tmp_path: Path, config_with_compaction: Config, mock_provider: MagicMock
    ):
        """compact_blocking() returns False when compaction not warranted."""
        db_path = tmp_path / "test.duckdb"
        db_path.touch()

        service = CompactionService(db_path, config_with_compaction)

        # Provider says no compaction needed
        mock_provider.should_compact.return_value = (
            False,
            {"_raw_fragmentation_ratio": 0.1, "free_blocks": 10, "block_size": 262144},
        )

        result = await service.compact_blocking(mock_provider)
        assert result is False


class TestBackgroundCompaction:
    """Test background compaction mode (MCP)."""

    @pytest.mark.asyncio
    async def test_background_compaction_returns_immediately(
        self, tmp_path: Path, config_with_compaction: Config, mock_provider: MagicMock
    ):
        """compact_background() returns before compaction finishes."""
        db_path = tmp_path / "test.duckdb"
        db_path.write_bytes(b"x" * 1024)

        service = CompactionService(db_path, config_with_compaction)

        compaction_started = asyncio.Event()
        compaction_can_finish = asyncio.Event()

        async def slow_compaction(provider):
            compaction_started.set()
            await compaction_can_finish.wait()

        with patch.object(service, "_do_compaction", slow_compaction):
            # Start background compaction
            result = await service.compact_background(mock_provider)

            # Should return True immediately
            assert result is True

            # Wait for compaction to start
            await asyncio.wait_for(compaction_started.wait(), timeout=1.0)

            # Compaction should still be in progress
            assert service.is_compacting

            # Allow compaction to finish
            compaction_can_finish.set()

            # Wait for task to complete
            if service._compaction_task:
                await service._compaction_task

    @pytest.mark.asyncio
    async def test_callback_invoked_after_success(
        self, tmp_path: Path, config_with_compaction: Config, mock_provider: MagicMock
    ):
        """on_complete callback called after successful compaction."""
        db_path = tmp_path / "test.duckdb"
        db_path.write_bytes(b"x" * 1024)

        service = CompactionService(db_path, config_with_compaction)

        callback_called = asyncio.Event()

        async def on_complete():
            callback_called.set()

        async def mock_do_compaction(provider):
            return True  # Success

        with patch.object(service, "_do_compaction", mock_do_compaction):
            await service.compact_background(mock_provider, on_complete=on_complete)

            # Wait for task to complete
            if service._compaction_task:
                await service._compaction_task

            # Callback should have been called
            assert callback_called.is_set()

    @pytest.mark.asyncio
    async def test_callback_not_invoked_on_failure(
        self, tmp_path: Path, config_with_compaction: Config, mock_provider: MagicMock
    ):
        """on_complete callback NOT called if compaction fails."""
        db_path = tmp_path / "test.duckdb"
        db_path.write_bytes(b"x" * 1024)

        service = CompactionService(db_path, config_with_compaction)

        callback_called = asyncio.Event()

        async def on_complete():
            callback_called.set()

        async def failing_compaction(provider):
            raise RuntimeError("Compaction failed")

        with patch.object(service, "_do_compaction", failing_compaction):
            await service.compact_background(mock_provider, on_complete=on_complete)

            # Wait for task to complete
            if service._compaction_task:
                try:
                    await service._compaction_task
                except Exception:
                    pass

            # Callback should NOT have been called
            assert not callback_called.is_set()

    @pytest.mark.asyncio
    async def test_callback_not_invoked_on_cancellation(
        self, tmp_path: Path, config_with_compaction: Config, mock_provider: MagicMock
    ):
        """on_complete callback NOT called if compaction is cancelled (returns False)."""
        db_path = tmp_path / "test.duckdb"
        db_path.write_bytes(b"x" * 1024)

        service = CompactionService(db_path, config_with_compaction)

        callback_called = asyncio.Event()

        async def on_complete():
            callback_called.set()

        async def cancelled_compaction(provider):
            return False  # Cancelled, not an error

        with patch.object(service, "_do_compaction", cancelled_compaction):
            await service.compact_background(mock_provider, on_complete=on_complete)

            if service._compaction_task:
                await service._compaction_task

            # Callback should NOT have been called for a cancelled compaction
            assert not callback_called.is_set()


class TestShutdown:
    """Test graceful shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_cancels_in_progress_compaction(
        self, tmp_path: Path, config_with_compaction: Config, mock_provider: MagicMock
    ):
        """shutdown() cancels running background compaction."""
        db_path = tmp_path / "test.duckdb"
        db_path.write_bytes(b"x" * 1024)

        service = CompactionService(db_path, config_with_compaction)

        compaction_started = asyncio.Event()
        compaction_cancelled = asyncio.Event()

        async def cancellable_compaction(provider):
            compaction_started.set()
            try:
                await asyncio.sleep(10)  # Long running
            except asyncio.CancelledError:
                compaction_cancelled.set()
                raise

        with patch.object(service, "_do_compaction", cancellable_compaction):
            # Start background compaction
            await service.compact_background(mock_provider)

            # Wait for it to start
            await asyncio.wait_for(compaction_started.wait(), timeout=1.0)

            # Shutdown should cancel the compaction
            await service.shutdown(timeout=1.0)

            # Compaction should have been cancelled
            assert compaction_cancelled.is_set()

    @pytest.mark.asyncio
    async def test_shutdown_cleans_up_resources(
        self, tmp_path: Path, config_with_compaction: Config, mock_provider: MagicMock
    ):
        """shutdown() resets state flags."""
        db_path = tmp_path / "test.duckdb"
        db_path.write_bytes(b"x" * 1024)

        service = CompactionService(db_path, config_with_compaction)

        # Simulate in-progress state
        service._compaction_in_progress = True

        await service.shutdown()

        # State should be reset
        assert not service.is_compacting
        assert service._compaction_task is None

    @pytest.mark.asyncio
    async def test_shutdown_idempotent(
        self, tmp_path: Path, config_with_compaction: Config
    ):
        """shutdown() can be called multiple times safely."""
        db_path = tmp_path / "test.duckdb"
        db_path.touch()

        service = CompactionService(db_path, config_with_compaction)

        # Multiple shutdowns should not raise
        await service.shutdown()
        await service.shutdown()
        await service.shutdown()

        assert not service.is_compacting

    @pytest.mark.asyncio
    async def test_shutdown_waits_for_thread_completion(
        self, tmp_path: Path, config_with_compaction: Config, mock_provider: MagicMock
    ):
        """shutdown() waits for the actual compaction thread to finish,
        not just the asyncio task cancellation."""
        db_path = tmp_path / "test.duckdb"
        db_path.write_bytes(b"x" * 1024)

        service = CompactionService(db_path, config_with_compaction)

        thread_started = threading.Event()
        thread_can_finish = threading.Event()

        def blocking_optimize(cancel_check=None):
            thread_started.set()
            thread_can_finish.wait(timeout=10.0)
            return True

        mock_provider.optimize = blocking_optimize

        await service.compact_background(mock_provider)

        # Yield to event loop so the background task starts running
        await asyncio.sleep(0)

        # Wait for the thread to actually start (blocks, but bounded)
        assert thread_started.wait(timeout=5.0), "Thread did not start"

        # Thread-done event should NOT be set (thread is blocked)
        assert not service.compaction_thread_done.is_set()

        # Allow thread to finish
        thread_can_finish.set()

        await service.shutdown(timeout=5.0)
        assert service.compaction_thread_done.is_set()

    @pytest.mark.asyncio
    async def test_compaction_thread_done_set_on_normal_completion(
        self, tmp_path: Path, config_with_compaction: Config, mock_provider: MagicMock
    ):
        """compaction_thread_done is set after successful compaction."""
        db_path = tmp_path / "test.duckdb"
        db_path.write_bytes(b"x" * 1024)

        service = CompactionService(db_path, config_with_compaction)

        # Initially set (no thread running)
        assert service.compaction_thread_done.is_set()

        mock_provider.optimize = MagicMock(return_value=True)

        await service.compact_background(mock_provider)
        if service._compaction_task:
            await service._compaction_task

        # Should be set again after completion
        assert service.compaction_thread_done.is_set()


class TestIsCompacting:
    """Test is_compacting property."""

    def test_is_compacting_false_initially(
        self, tmp_path: Path, config_with_compaction: Config
    ):
        """is_compacting is False when no compaction is running."""
        db_path = tmp_path / "test.duckdb"
        db_path.touch()

        service = CompactionService(db_path, config_with_compaction)
        assert service.is_compacting is False

    @pytest.mark.asyncio
    async def test_is_compacting_true_during_operation(
        self, tmp_path: Path, config_with_compaction: Config, mock_provider: MagicMock
    ):
        """is_compacting is True while compaction runs."""
        db_path = tmp_path / "test.duckdb"
        db_path.write_bytes(b"x" * 1024)

        service = CompactionService(db_path, config_with_compaction)

        compaction_started = asyncio.Event()
        compaction_can_finish = asyncio.Event()

        async def controlled_compaction(provider):
            compaction_started.set()
            await compaction_can_finish.wait()

        with patch.object(service, "_do_compaction", controlled_compaction):
            await service.compact_background(mock_provider)
            await asyncio.wait_for(compaction_started.wait(), timeout=1.0)

            # Should be compacting now
            assert service.is_compacting is True

            compaction_can_finish.set()
            if service._compaction_task:
                await service._compaction_task

            # Should not be compacting after
            assert service.is_compacting is False


@pytest.fixture
def provider_with_fragmentation(tmp_path: Path):
    """Create a DuckDB provider with data, then delete to create free blocks."""
    db_path = tmp_path / "test.duckdb"
    provider = DuckDBProvider(str(db_path), base_directory=tmp_path)
    provider.connect()

    # Insert multiple files and chunks to create meaningful data
    file_ids = []
    for i in range(10):
        test_file = File(
            path=f"test_{i}.py",
            mtime=1234567890.0,
            language=Language.PYTHON,
            size_bytes=1000,
        )
        file_id = provider.insert_file(test_file)
        file_ids.append(file_id)

        for j in range(50):
            chunk = Chunk(
                file_id=file_id,
                code=f"def func_{i}_{j}(): pass  # padding " + "x" * 500,
                start_line=j * 10 + 1,
                end_line=j * 10 + 6,
                chunk_type=ChunkType.FUNCTION,
                symbol=f"func_{i}_{j}",
                language=Language.PYTHON,
            )
            provider.insert_chunk(chunk)

    # Force checkpoint to ensure data is written to disk
    provider.optimize_tables()

    # Delete most files to create free blocks (keep last 2)
    for i in range(len(file_ids) - 2):
        provider.delete_file_completely(f"test_{i}.py")

    # Checkpoint again to finalize deletions
    provider.optimize_tables()

    yield provider, db_path

    if provider.is_connected:
        provider.disconnect()


class TestDuckDBProviderOptimize:
    """Test actual DuckDB compaction via DuckDBProvider.optimize()."""

    def test_optimize_reduces_size_after_deletions(
        self, provider_with_fragmentation: tuple
    ):
        """Verify optimize() reclaims space from deleted rows."""
        provider, db_path = provider_with_fragmentation

        # Get size before compaction
        size_before = db_path.stat().st_size

        # Verify there are free blocks to reclaim
        stats = provider.get_storage_stats()
        free_blocks = stats.get("free_blocks", 0)
        assert free_blocks > 0, "Test setup failed: no free blocks to reclaim"

        # Run compaction
        result = provider.optimize()
        assert result is True

        # Get size after compaction
        size_after = db_path.stat().st_size

        # Size should be reduced (or at minimum not grow significantly)
        # Note: Small databases may not show significant reduction
        assert size_after <= size_before, (
            f"Database grew after compaction: {size_before} -> {size_after}"
        )

    def test_optimize_preserves_data_integrity(
        self, provider_with_fragmentation: tuple
    ):
        """Verify data is still queryable after compaction."""
        provider, db_path = provider_with_fragmentation

        # Get data before compaction (last 2 files: test_8.py, test_9.py)
        chunks_before = provider.get_all_chunks_with_metadata()
        file8_before = provider.get_file_by_path("test_8.py")
        file9_before = provider.get_file_by_path("test_9.py")

        assert file8_before is not None, "test_8.py should exist before compaction"
        assert file9_before is not None, "test_9.py should exist before compaction"

        # Run compaction
        result = provider.optimize()
        assert result is True

        # Verify data is preserved
        chunks_after = provider.get_all_chunks_with_metadata()
        file8_after = provider.get_file_by_path("test_8.py")
        file9_after = provider.get_file_by_path("test_9.py")

        assert file8_after is not None, "test_8.py missing after compaction"
        assert file9_after is not None, "test_9.py missing after compaction"
        assert len(chunks_after) == len(chunks_before), (
            f"Chunk count mismatch: {len(chunks_before)} -> {len(chunks_after)}"
        )

    def test_optimize_cleans_up_lock_file(
        self, provider_with_fragmentation: tuple
    ):
        """Verify lock file is removed after compaction."""
        from chunkhound.providers.database.duckdb.connection_manager import (
            get_compaction_lock_path,
        )

        provider, db_path = provider_with_fragmentation
        lock_path = get_compaction_lock_path(db_path)

        # Run compaction
        provider.optimize()

        # Lock file should not exist after completion
        assert not lock_path.exists(), f"Lock file not cleaned up: {lock_path}"


class TestRowWasteRatio:
    """Test row_waste_ratio computation and its role in compaction decisions."""

    def test_row_waste_nonzero_after_deletions(
        self, provider_with_fragmentation: tuple
    ):
        """Deleting 80% of rows should produce measurable row waste."""
        provider, _db_path = provider_with_fragmentation
        stats = provider.get_storage_stats()
        assert stats["row_waste_ratio"] > 0, (
            f"Expected row_waste_ratio > 0 after deleting 80% of data, "
            f"got {stats['row_waste_ratio']}"
        )

    def test_row_waste_near_zero_on_clean_db(self, tmp_path: Path):
        """A freshly-populated database should have negligible row waste."""
        db_path = tmp_path / "clean.duckdb"
        provider = DuckDBProvider(str(db_path), base_directory=tmp_path)
        provider.connect()
        try:
            for i in range(5):
                test_file = File(
                    path=f"clean_{i}.py",
                    mtime=1234567890.0,
                    language=Language.PYTHON,
                    size_bytes=1000,
                )
                file_id = provider.insert_file(test_file)
                for j in range(20):
                    chunk = Chunk(
                        file_id=file_id,
                        code=f"def f_{i}_{j}(): pass  # " + "x" * 200,
                        start_line=j * 5 + 1,
                        end_line=j * 5 + 4,
                        chunk_type=ChunkType.FUNCTION,
                        symbol=f"f_{i}_{j}",
                        language=Language.PYTHON,
                    )
                    provider.insert_chunk(chunk)
            provider.optimize_tables()

            stats = provider.get_storage_stats()
            assert stats["row_waste_ratio"] < 0.05, (
                f"Expected near-zero row_waste_ratio on clean DB, "
                f"got {stats['row_waste_ratio']}"
            )
        finally:
            provider.disconnect()

    def test_should_compact_triggered_by_row_waste_alone(
        self, provider_with_fragmentation: tuple
    ):
        """should_compact should return True when row_waste exceeds threshold,
        even if free_blocks ratio is low."""
        provider, _db_path = provider_with_fragmentation

        # After deletions + checkpoint, row_waste should be high (~0.8)
        # while free_ratio is moderate (~0.37). Use a threshold between
        # the two so only row_waste exceeds it.
        threshold = 0.4
        should, stats = provider.should_compact(threshold=threshold)
        total = max(stats.get("total_blocks", 1), 1)
        free_ratio = stats.get("free_blocks", 0) / total
        assert free_ratio < threshold, (
            f"Test precondition: free_ratio should be below threshold, got {free_ratio}"
        )
        assert should is True, (
            f"Expected should_compact=True with row_waste={stats.get('row_waste_ratio')}, "
            f"free_ratio={free_ratio}"
        )


class TestCompactionServiceEndToEnd:
    """End-to-end: CompactionService.compact_blocking() -> provider.optimize()."""

    @pytest.mark.asyncio
    async def test_compact_blocking_end_to_end(
        self, tmp_path: Path, provider_with_fragmentation: tuple
    ):
        """Full path through CompactionService to real provider.optimize()."""
        provider, db_path = provider_with_fragmentation

        config = Config(
            database=DatabaseConfig(
                path=tmp_path / ".chunkhound",
                compaction_enabled=True,
                compaction_threshold=0.01,  # Very low to ensure trigger
                compaction_min_size_mb=0,  # No minimum for test DBs
            ),
            target_dir=tmp_path,
        )
        service = CompactionService(db_path=db_path, config=config)

        # Capture pre-compaction state
        chunks_before = provider.get_all_chunks_with_metadata()
        size_before = db_path.stat().st_size

        result = await service.compact_blocking(provider)
        assert result is True

        # Verify data integrity
        chunks_after = provider.get_all_chunks_with_metadata()
        assert len(chunks_after) == len(chunks_before), (
            f"Chunk count mismatch: {len(chunks_before)} -> {len(chunks_after)}"
        )

        # Verify remaining files survived
        file8 = provider.get_file_by_path("test_8.py")
        file9 = provider.get_file_by_path("test_9.py")
        assert file8 is not None, "test_8.py missing after compaction"
        assert file9 is not None, "test_9.py missing after compaction"

        # Verify size reduction
        size_after = db_path.stat().st_size
        assert size_after <= size_before, (
            f"Database grew after compaction: {size_before} -> {size_after}"
        )
