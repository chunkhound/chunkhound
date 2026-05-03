"""Tests for CompactionService.

Tests compaction logic including storage statistics, blocking/background modes,
atomic swap, and graceful shutdown. Uses real components where possible,
only mocking external dependencies like disk operations.
"""

import asyncio
import io
import os
import shutil
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from loguru import logger

from chunkhound.core.config.config import Config
from chunkhound.core.config.database_config import DatabaseConfig
from chunkhound.core.exceptions import CompactionError
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
            "effective_waste": 0.5,
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
        # Verify config threshold was passed to provider
        mock_provider.should_compact.assert_called_once_with(threshold=0.5)

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
        # Verify config threshold was passed to provider
        mock_provider.should_compact.assert_called_once_with(threshold=0.5)

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
        service._compaction_thread_done.clear()  # Simulate active thread

        should, stats = service.check_should_compact(mock_provider)
        assert should is False
        assert stats == {}

    def test_try_start_compaction_concurrent_only_one_wins(
        self, tmp_path: Path, config_with_compaction: Config, mock_provider: MagicMock
    ):
        """Two concurrent _try_start_compaction calls — only one succeeds."""
        db_path = tmp_path / "test.duckdb"
        db_path.touch()

        service = CompactionService(db_path, config_with_compaction)

        barrier = threading.Barrier(2)
        results: list[bool] = []

        def try_start():
            barrier.wait()
            ok, _ = service._try_start_compaction(mock_provider)
            results.append(ok)

        t1 = threading.Thread(target=try_start)
        t2 = threading.Thread(target=try_start)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results.count(True) == 1, "Exactly one caller should win"
        assert results.count(False) == 1, "Other caller should be rejected"

    def test_try_start_compaction_skips_on_compaction_error(
        self, tmp_path: Path, config_with_compaction: Config, mock_provider: MagicMock
    ):
        """should_compact raising CompactionError -> gracefully return False."""
        db_path = tmp_path / "test.duckdb"
        db_path.touch()

        mock_provider.should_compact.side_effect = CompactionError(
            "connection gated", operation="connection"
        )

        service = CompactionService(db_path, config_with_compaction)
        ok, stats = service._try_start_compaction(mock_provider)

        assert ok is False
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
        service._compaction_thread_done.clear()  # Simulate active thread

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
            assert service.last_error is None

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
            assert isinstance(service.last_error, RuntimeError)

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

    @pytest.mark.asyncio
    async def test_callback_failure_surfaces_via_last_error(
        self, tmp_path: Path, config_with_compaction: Config, mock_provider: MagicMock
    ):
        """Successful compaction + failing callback → last_error captures callback error."""
        db_path = tmp_path / "test.duckdb"
        db_path.write_bytes(b"x" * 1024)

        service = CompactionService(db_path, config_with_compaction)

        async def failing_callback():
            raise RuntimeError("Callback failed")

        async def successful_compaction(provider):
            return True

        with patch.object(service, "_do_compaction", successful_compaction):
            await service.compact_background(
                mock_provider, on_complete=failing_callback
            )

            if service._compaction_task:
                try:
                    await service._compaction_task
                except Exception:
                    pass

            assert isinstance(service.last_error, RuntimeError)
            assert "Callback failed" in str(service.last_error)


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
    async def test_cancelled_task_preserves_flag_while_thread_alive(
        self, tmp_path: Path, config_with_compaction: Config, mock_provider: MagicMock
    ):
        """When CancelledError propagates but the compaction thread is still
        running, _compaction_in_progress must remain True to prevent a
        concurrent compaction from starting."""
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
        await asyncio.sleep(0)
        assert thread_started.wait(timeout=5.0), "Thread did not start"

        # Cancel the asyncio task while thread is still running
        task = service._compaction_task
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

        # Thread is alive → flag must stay True
        assert not service.compaction_thread_done.is_set()
        assert service.is_compacting, (
            "_compaction_in_progress was reset despite thread still running"
        )

        # Cleanup
        thread_can_finish.set()
        await service.shutdown(timeout=5.0)

    @pytest.mark.asyncio
    async def test_cancelled_blocking_preserves_flag_while_thread_alive(
        self, tmp_path: Path, config_with_compaction: Config, mock_provider: MagicMock
    ):
        """When compact_blocking's task is cancelled but the compaction thread
        is still running, _compaction_in_progress must remain True."""
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

        task = asyncio.create_task(service.compact_blocking(mock_provider))
        await asyncio.sleep(0)
        assert thread_started.wait(timeout=5.0), "Thread did not start"

        # Cancel the asyncio task while thread is still running
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

        # Thread is alive → flag must stay True
        assert not service.compaction_thread_done.is_set()
        assert service.is_compacting, (
            "_compaction_in_progress was reset despite thread still running"
        )

        # Cleanup
        thread_can_finish.set()
        await service.shutdown(timeout=5.0)

    @pytest.mark.asyncio
    async def test_self_heals_flag_after_cancelled_task_and_thread_finish(
        self, tmp_path: Path, config_with_compaction: Config, mock_provider: MagicMock
    ):
        """After task cancellation + thread completion, check_should_compact
        self-heals the stuck _compaction_in_progress flag."""
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
        await asyncio.sleep(0)
        assert thread_started.wait(timeout=5.0)

        # Cancel the asyncio task while thread is still running
        task = service._compaction_task
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

        assert service.is_compacting, "Flag should stay True while thread alive"

        # Let thread finish — _compaction_thread_done gets set
        thread_can_finish.set()
        assert service.compaction_thread_done.wait(timeout=5.0)

        # Flag is still stuck True (no one reset it after cancellation)
        assert service.is_compacting

        # check_should_compact should self-heal
        should, _ = service.check_should_compact(mock_provider)
        assert not service.is_compacting, (
            "check_should_compact should have reset the stuck flag"
        )

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

    @pytest.mark.asyncio
    async def test_shutdown_timeout_keeps_flag_when_thread_alive(
        self, tmp_path: Path, config_with_compaction: Config, mock_provider: MagicMock
    ):
        """When shutdown times out with thread still alive, is_compacting stays True."""
        db_path = tmp_path / "test.duckdb"
        db_path.write_bytes(b"x" * 1024)

        service = CompactionService(db_path, config_with_compaction)

        thread_started = threading.Event()
        thread_can_finish = threading.Event()

        def blocking_optimize(cancel_check=None):
            thread_started.set()
            thread_can_finish.wait(timeout=30.0)
            return True

        mock_provider.optimize = blocking_optimize

        await service.compact_background(mock_provider)
        await asyncio.sleep(0)
        assert thread_started.wait(timeout=5.0), "Thread did not start"

        # Shutdown with very short timeout — thread will still be alive
        await service.shutdown(timeout=0.5)

        # Thread is still alive → flag must stay True
        try:
            assert not service.compaction_thread_done.is_set()
            assert service.is_compacting, (
                "_compaction_in_progress was reset despite thread still running"
            )
        finally:
            thread_can_finish.set()
            assert service.compaction_thread_done.wait(timeout=5.0), (
                "Cleanup: thread did not exit"
            )

    @pytest.mark.asyncio
    async def test_shutdown_completes_when_to_thread_fails_before_start(
        self, tmp_path: Path, config_with_compaction: Config, mock_provider: MagicMock
    ):
        """shutdown() doesn't deadlock when asyncio.to_thread raises before spawning thread."""
        db_path = tmp_path / "test.duckdb"
        db_path.write_bytes(b"x" * 1024)

        service = CompactionService(db_path, config_with_compaction)

        with patch(
            "asyncio.to_thread", side_effect=RuntimeError("no running event loop")
        ):
            with pytest.raises(RuntimeError):
                await service.compact_blocking(mock_provider)

        # The thread_entered guard should have restored _compaction_thread_done
        assert service.compaction_thread_done.is_set()

        # shutdown must complete without hanging
        await asyncio.wait_for(service.shutdown(timeout=2.0), timeout=3.0)


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

    def test_optimize_cleans_up_lock_file(self, provider_with_fragmentation: tuple):
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

    def test_optimize_succeeds_when_old_db_unlink_fails(
        self, provider_with_fragmentation: tuple
    ):
        """Compaction succeeds even if removing the old DB file fails."""
        provider, db_path = provider_with_fragmentation
        old_db_path = db_path.with_suffix(".duckdb.old")
        original_unlink = Path.unlink

        def selective_unlink(self_path, *args, **kwargs):
            if str(self_path).endswith(".duckdb.old"):
                raise PermissionError("simulated permission denied")
            return original_unlink(self_path, *args, **kwargs)

        with patch.object(Path, "unlink", selective_unlink):
            result = provider.optimize()

        assert result is True
        assert provider.is_connected, "provider should be reconnected"
        assert old_db_path.exists(), "old DB should remain when unlink fails"

    def test_optimize_does_not_delete_foreign_lock(
        self, provider_with_fragmentation: tuple
    ):
        """Pre-existing lock file is not deleted when optimize() fails to acquire it."""
        from chunkhound.providers.database.duckdb.connection_manager import (
            get_compaction_lock_path,
        )

        provider, db_path = provider_with_fragmentation
        lock_path = get_compaction_lock_path(db_path)

        # Simulate another process holding the lock
        lock_path.write_text("99999:1700000000")

        with pytest.raises(CompactionError, match="lock"):
            provider.optimize()

        assert lock_path.exists(), "Foreign lock file was incorrectly deleted"

    def test_post_compaction_reconnect_fails_for_terminal_cleanup_provider(
        self, provider_with_fragmentation: tuple
    ):
        """Reconnect helper must surface terminal cleanup as CompactionError."""
        provider, _db_path = provider_with_fragmentation

        provider.mark_terminal_after_stuck_cleanup()
        assert not provider.is_accepting_connections

        with patch.object(
            provider,
            "connect",
            side_effect=AssertionError("connect() must be skipped"),
        ):
            with pytest.raises(
                CompactionError, match="terminal stuck-cleanup state"
            ) as exc_info:
                provider._restore_connection_after_compaction()

        assert exc_info.value.operation == "compaction"
        assert exc_info.value.recoverable is False
        assert not provider.is_accepting_connections


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


class TestCompactionServiceEligibilityReal:
    """CompactionService gating exercised against a real DuckDB provider.

    The bulk of the eligibility suite uses a MagicMock provider. These tests
    validate that threshold/min-size config values actually propagate through
    the real provider.should_compact() path.
    """

    def _make_service(
        self,
        tmp_path: Path,
        db_path: Path,
        *,
        threshold: float,
        min_size_mb: int,
    ) -> CompactionService:
        config = Config(
            database=DatabaseConfig(
                path=tmp_path / ".chunkhound",
                compaction_enabled=True,
                compaction_threshold=threshold,
                compaction_min_size_mb=min_size_mb,
            ),
            target_dir=tmp_path,
        )
        return CompactionService(db_path=db_path, config=config)

    def test_should_compact_gate_respects_threshold_real_duckdb(
        self, tmp_path: Path, provider_with_fragmentation: tuple
    ):
        """Config threshold is propagated through to provider.should_compact()."""
        provider, db_path = provider_with_fragmentation

        # Low threshold: fragmented DB must exceed it.
        service_low = self._make_service(
            tmp_path, db_path, threshold=0.1, min_size_mb=0
        )
        should_low, stats_low = service_low.check_should_compact(provider)
        assert should_low is True, (
            f"Expected compaction to fire at threshold=0.1, stats={stats_low}"
        )

        # Near-impossible threshold: nothing can exceed it.
        service_high = self._make_service(
            tmp_path, db_path, threshold=0.99, min_size_mb=0
        )
        should_high, _ = service_high.check_should_compact(provider)
        assert should_high is False, (
            "Expected compaction to be gated off at threshold=0.99"
        )

    def test_should_compact_gate_respects_min_size_real_duckdb(
        self, tmp_path: Path, provider_with_fragmentation: tuple
    ):
        """Reclaimable-byte gate is applied against real provider stats."""
        provider, db_path = provider_with_fragmentation

        # min_size=0 ⇒ any reclaimable space is enough.
        service_zero = self._make_service(
            tmp_path, db_path, threshold=0.01, min_size_mb=0
        )
        should_zero, _ = service_zero.check_should_compact(provider)
        assert should_zero is True, "Expected compaction to fire with min_size_mb=0"

        # min_size=1000 MB ⇒ the small fixture DB cannot exceed it.
        service_huge = self._make_service(
            tmp_path, db_path, threshold=0.01, min_size_mb=1000
        )
        should_huge, _ = service_huge.check_should_compact(provider)
        assert should_huge is False, (
            "Expected compaction to be gated off when reclaimable bytes "
            "fall below the min-size threshold"
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


class TestBackgroundCompactionE2E:
    """End-to-end: compact_background() with a real DuckDB provider."""

    @pytest.mark.asyncio
    async def test_compact_background_end_to_end(
        self, tmp_path: Path, provider_with_fragmentation: tuple
    ):
        """compact_background() completes with real provider and preserves data."""
        provider, db_path = provider_with_fragmentation

        config = Config(
            database=DatabaseConfig(
                path=tmp_path / ".chunkhound",
                compaction_enabled=True,
                compaction_threshold=0.01,
                compaction_min_size_mb=0,
            ),
            target_dir=tmp_path,
        )
        service = CompactionService(db_path=db_path, config=config)

        chunks_before = provider.get_all_chunks_with_metadata()

        callback_called = asyncio.Event()

        async def on_complete():
            callback_called.set()

        result = await service.compact_background(provider, on_complete=on_complete)
        assert result is True

        # Wait for the background task to finish
        if service._compaction_task:
            await asyncio.wait_for(service._compaction_task, timeout=30.0)

        assert callback_called.is_set(), "on_complete callback was not invoked"

        # Verify data integrity
        chunks_after = provider.get_all_chunks_with_metadata()
        assert len(chunks_after) == len(chunks_before), (
            f"Chunk count mismatch: {len(chunks_before)} -> {len(chunks_after)}"
        )
        assert provider.get_file_by_path("test_8.py") is not None
        assert provider.get_file_by_path("test_9.py") is not None


class TestCompactionErrorRecovery:
    """Test error recovery during compaction."""

    def test_optimize_recovers_on_export_failure(
        self, provider_with_fragmentation: tuple
    ):
        """Provider recovers when _export_database_for_compaction raises."""
        provider, db_path = provider_with_fragmentation

        chunks_before = provider.get_all_chunks_with_metadata()

        with patch.object(
            provider,
            "_export_database_for_compaction",
            side_effect=RuntimeError("export crashed"),
        ):
            with pytest.raises(CompactionError, match="export crashed") as exc_info:
                provider.optimize()

            assert exc_info.value.operation == "compaction"

        # Provider should have reconnected
        assert provider.is_connected is True

        # Data should be intact
        chunks_after = provider.get_all_chunks_with_metadata()
        assert len(chunks_after) == len(chunks_before), (
            f"Chunk count mismatch: {len(chunks_before)} -> {len(chunks_after)}"
        )
        assert provider.get_file_by_path("test_8.py") is not None
        assert provider.get_file_by_path("test_9.py") is not None

        # No compact temp file should exist (export failed before creating it)
        assert not db_path.with_suffix(".compact.duckdb").exists()

    def test_optimize_recovers_on_import_failure(
        self, provider_with_fragmentation: tuple
    ):
        """Provider recovers when _import_database_for_compaction raises."""
        provider, db_path = provider_with_fragmentation

        chunks_before = provider.get_all_chunks_with_metadata()

        with patch.object(
            provider,
            "_import_database_for_compaction",
            side_effect=RuntimeError("import crashed"),
        ):
            with pytest.raises(CompactionError, match="import crashed") as exc_info:
                provider.optimize()

            assert exc_info.value.operation == "compaction"

        # Provider should have reconnected
        assert provider.is_connected is True

        # Data should be intact
        chunks_after = provider.get_all_chunks_with_metadata()
        assert len(chunks_after) == len(chunks_before), (
            f"Chunk count mismatch: {len(chunks_before)} -> {len(chunks_after)}"
        )
        assert provider.get_file_by_path("test_8.py") is not None
        assert provider.get_file_by_path("test_9.py") is not None

        # Temp files should be cleaned up
        assert not db_path.with_suffix(".compact.duckdb").exists()

    def test_optimize_opens_gate_when_import_and_reconnect_both_fail(
        self, provider_with_fragmentation: tuple
    ):
        """Connection gate is opened even when both import and reconnect fail."""
        provider, db_path = provider_with_fragmentation

        with (
            patch.object(
                provider,
                "_import_database_for_compaction",
                side_effect=RuntimeError("import crashed"),
            ),
            patch.object(
                provider,
                "connect",
                side_effect=RuntimeError("reconnect failed"),
            ),
        ):
            with pytest.raises(CompactionError):
                provider.optimize()

        # Gate must be open so MCP requests don't hang permanently.
        assert provider.is_accepting_connections, (
            "connection gate must be open after double failure"
        )
        # Provider couldn't reconnect
        assert provider.is_connected is False

    def test_optimize_raises_unrecoverable_when_both_db_files_missing(
        self, provider_with_fragmentation: tuple
    ):
        """CompactionError raised when swap fails and neither db nor backup exist."""
        provider, db_path = provider_with_fragmentation
        old_db_path = db_path.with_suffix(".duckdb.old")

        real_replace = os.replace

        def destructive_replace(src, dst):
            if Path(dst) == old_db_path:
                # First swap step: db_path -> old_db_path — let it succeed
                real_replace(src, dst)
            elif Path(dst) == db_path:
                # Second swap step: compact_db -> db_path — simulate corruption
                old_db_path.unlink(missing_ok=True)
                raise OSError("filesystem corruption during swap")
            else:
                real_replace(src, dst)

        with patch(
            "chunkhound.providers.database.duckdb_provider.os.replace",
            side_effect=destructive_replace,
        ):
            with pytest.raises(
                CompactionError, match="Manual recovery required"
            ) as exc_info:
                provider.optimize()

            assert exc_info.value.operation == "compaction"

        assert not db_path.exists()
        assert not old_db_path.exists()

    def test_optimize_rejects_insufficient_disk_space(
        self, provider_with_fragmentation: tuple
    ):
        """CompactionError raised when disk space is insufficient."""
        provider, db_path = provider_with_fragmentation

        # Return very low free space (1 byte)
        fake_usage = shutil.disk_usage(db_path.parent)._replace(free=1)
        with patch(
            "chunkhound.providers.database.duckdb_provider.shutil.disk_usage",
            return_value=fake_usage,
        ):
            with pytest.raises(
                CompactionError, match="Insufficient disk space"
            ) as exc_info:
                provider.optimize()

            assert exc_info.value.operation == "preflight"

        # Provider stays connected (preflight runs before soft_disconnect)
        assert provider.is_connected is True

    def test_optimize_handles_disk_stat_oserror(
        self, provider_with_fragmentation: tuple
    ):
        """CompactionError raised when disk usage check fails with OSError."""
        provider, _db_path = provider_with_fragmentation

        with patch(
            "chunkhound.providers.database.duckdb_provider.shutil.disk_usage",
            side_effect=OSError("device error"),
        ):
            with pytest.raises(
                CompactionError, match="Cannot verify disk space"
            ) as exc_info:
                provider.optimize()

            assert exc_info.value.operation == "preflight"

        assert provider.is_connected is True


class TestPostSwapCancelCheck:
    """Test the cancel_check branch after the atomic file swap."""

    def test_optimize_skips_reconnect_on_post_swap_cancel(
        self, provider_with_fragmentation: tuple
    ):
        """When shutdown fires after swap succeeds, reconnect is skipped
        and the backup file is cleaned up."""
        provider, db_path = provider_with_fragmentation
        old_db_path = db_path.with_suffix(".duckdb.old")

        # Track when the atomic swap completes so cancel fires at the right
        # semantic point, not at a fragile hard-coded call count.
        swap_complete = False
        original_replace = os.replace

        def tracking_replace(src, dst):
            nonlocal swap_complete
            result = original_replace(src, dst)
            if str(src).endswith(".compact.duckdb"):
                swap_complete = True
            return result

        with patch("os.replace", side_effect=tracking_replace):
            result = provider.optimize(cancel_check=lambda: swap_complete)

        assert result is True
        assert not provider.is_connected, "reconnect should have been skipped"
        assert not old_db_path.exists(), "backup should have been cleaned up"
        assert provider.is_accepting_connections, "gate should be restored"

    def test_optimize_preserves_backup_when_probe_fails_on_cancel(
        self, provider_with_fragmentation: tuple
    ):
        """When probe fails after swap on cancel path, backup is preserved."""
        provider, db_path = provider_with_fragmentation
        old_db_path = db_path.with_suffix(".duckdb.old")

        swap_complete = False
        original_replace = os.replace

        def tracking_replace(src, dst):
            nonlocal swap_complete
            result = original_replace(src, dst)
            if str(src).endswith(".compact.duckdb"):
                swap_complete = True
            return result

        log_buf = io.StringIO()
        sink_id = logger.add(log_buf, level="WARNING")
        try:
            with (
                patch("os.replace", side_effect=tracking_replace),
                patch.object(
                    provider._connection_manager, "_probe_db_valid", return_value=False
                ),
            ):
                result = provider.optimize(cancel_check=lambda: swap_complete)
        finally:
            logger.remove(sink_id)

        assert result is False
        assert not provider.is_connected, "reconnect should have been skipped"
        assert old_db_path.exists(), "backup should be preserved when probe fails"
        assert provider.is_accepting_connections, "gate should be restored"
        assert "failed integrity probe on cancel path" in log_buf.getvalue()


class TestConnectionGateDuringCompaction:
    """Test that _connection_allowed gate blocks operations during compaction."""

    def test_operations_blocked_when_gate_closed(
        self, provider_with_fragmentation: tuple
    ):
        """Provider operations raise CompactionError when connection gate is closed."""
        provider, _db_path = provider_with_fragmentation

        # Clear cached executor connection so the gate check fires
        provider.soft_disconnect()
        provider._connection_allowed.clear()

        try:
            with pytest.raises(
                CompactionError, match="compaction in progress"
            ) as exc_info:
                provider.get_file_by_path("test_8.py")

            assert exc_info.value.operation == "connection"
        finally:
            # Restore gate and reconnect
            provider._connection_allowed.set()
            provider.connect()

        # Verify operations resume after gate recovery
        result = provider.get_file_by_path("test_8.py")
        assert result is not None


class TestConcurrentAccessDuringCompaction:
    """Test that concurrent reads are blocked while compaction is in progress."""

    def test_queued_read_finishes_before_disconnect_and_later_reads_are_blocked(
        self, provider_with_fragmentation: tuple
    ):
        """A queued read drains, new reads are blocked during compaction, then resume."""
        provider, _db_path = provider_with_fragmentation

        queued_read_started = threading.Event()
        allow_queued_read = threading.Event()
        export_paused = threading.Event()
        allow_export = threading.Event()
        queued_read_result: dict[str, object] = {}
        compaction_result: dict[str, object] = {}

        original_read = provider._executor_get_file_by_path
        original_export = provider._export_database_for_compaction

        def blocking_read(conn, state, path, as_model):
            queued_read_started.set()
            assert allow_queued_read.wait(timeout=10), "Queued read never released"
            return original_read(conn, state, path, as_model)

        def pausing_export(db_p: Path, export_dir: Path) -> None:
            export_paused.set()
            assert allow_export.wait(timeout=10), "Export phase never released"
            original_export(db_p, export_dir)

        def run_queued_read() -> None:
            try:
                queued_read_result["value"] = provider.get_file_by_path("test_8.py")
            except Exception as exc:  # pragma: no cover - failure path asserted below
                queued_read_result["error"] = exc

        def run_compaction() -> None:
            try:
                compaction_result["value"] = provider.optimize()
            except Exception as exc:  # pragma: no cover - failure path asserted below
                compaction_result["error"] = exc

        with (
            patch.object(
                provider, "_executor_get_file_by_path", side_effect=blocking_read
            ),
            patch.object(
                provider, "_export_database_for_compaction", side_effect=pausing_export
            ),
        ):
            read_thread: threading.Thread | None = None
            compact_thread: threading.Thread | None = None
            try:
                read_thread = threading.Thread(target=run_queued_read)
                read_thread.start()
                assert queued_read_started.wait(timeout=10), "Queued read never started"

                compact_thread = threading.Thread(target=run_compaction)
                compact_thread.start()

                assert not export_paused.is_set(), (
                    "Compaction reached the blocked phase before the queued read drained"
                )

                allow_queued_read.set()
                read_thread.join(timeout=10)
                assert not read_thread.is_alive(), "Queued read thread did not finish"
                assert "error" not in queued_read_result, queued_read_result.get(
                    "error"
                )
                assert queued_read_result["value"] is not None

                assert export_paused.wait(timeout=10), "Export phase never started"

                with pytest.raises(
                    CompactionError, match="compaction in progress"
                ) as exc_info:
                    provider.get_file_by_path("test_8.py")
                assert exc_info.value.operation == "connection"
            finally:
                allow_queued_read.set()
                allow_export.set()
                if read_thread is not None:
                    read_thread.join(timeout=10)
                    assert not read_thread.is_alive(), (
                        "Queued read thread did not finish during cleanup"
                    )
                if compact_thread is not None:
                    compact_thread.join(timeout=10)
                    assert not compact_thread.is_alive(), (
                        "Compaction thread did not finish during cleanup"
                    )

            assert "error" not in compaction_result, compaction_result.get("error")
            assert compaction_result["value"] is True

        result = provider.get_file_by_path("test_8.py")
        assert result is not None

    def test_reads_blocked_during_export_then_resume(
        self, provider_with_fragmentation: tuple
    ):
        """Reads raise CompactionError during active compaction, then resume after."""
        provider, db_path = provider_with_fragmentation

        export_paused = threading.Event()
        export_proceed = threading.Event()
        read_result: dict = {}

        original_export = provider._export_database_for_compaction

        def pausing_export(db_p: Path, export_dir: Path) -> None:
            # Signal that export phase has started (gate closed, disconnected)
            export_paused.set()
            # Wait for the test to attempt a read before continuing
            export_proceed.wait(timeout=10)
            original_export(db_p, export_dir)

        def attempt_read() -> None:
            try:
                provider.get_file_by_path("test_8.py")
                read_result["error"] = None
            except CompactionError as e:
                read_result["error"] = e

        with patch.object(
            provider, "_export_database_for_compaction", side_effect=pausing_export
        ):
            # Start compaction in a background thread
            compact_thread = threading.Thread(
                target=provider.optimize, kwargs={"cancel_check": None}
            )
            compact_thread.start()

            # Wait until export phase is reached (gate is closed)
            assert export_paused.wait(timeout=10), "Export phase never started"

            # Attempt a read from a separate thread — should be blocked
            read_thread = threading.Thread(target=attempt_read)
            read_thread.start()
            read_thread.join(timeout=5)

            # Verify the read was rejected
            assert "error" in read_result, "Read thread didn't complete"
            assert isinstance(read_result["error"], CompactionError)
            assert read_result["error"].operation == "connection"

            # Let compaction finish
            export_proceed.set()
            compact_thread.join(timeout=10)

        # After compaction, reads should work again
        result = provider.get_file_by_path("test_8.py")
        assert result is not None


class TestEmbeddingVectorSurvival:
    """Test that embedding vectors survive EXPORT/IMPORT compaction.

    Tests the DuckDB EXPORT/IMPORT primitive directly rather than through
    DuckDBProvider.optimize() because CHECKPOINT with VSS-loaded tables
    is prohibitively slow (~60s) in small test databases.

    The remaining optimize() logic (lock acquisition, soft_disconnect,
    atomic swap, reconnect) is exercised by TestDuckDBProviderOptimize
    without embeddings. Combined coverage is sufficient: this class
    validates data fidelity of the EXPORT/IMPORT round-trip, while
    TestDuckDBProviderOptimize validates the orchestration around it.
    """

    def test_optimize_preserves_embedding_vectors(self, tmp_path: Path):
        """Embedding vectors survive the EXPORT/IMPORT compaction round-trip.

        Tests the EXPORT DATABASE / IMPORT DATABASE cycle directly because
        DuckDB's CHECKPOINT with VSS-loaded embedding tables is prohibitively
        slow in small test databases (>60s), making the full optimize() path
        impractical for CI.
        """
        import random

        import duckdb

        db_path = tmp_path / "emb.duckdb"

        # Create database with embeddings using direct DuckDB connection
        conn = duckdb.connect(str(db_path))
        try:
            conn.execute("LOAD vss")
        except duckdb.Error:
            conn.close()
            pytest.skip("VSS extension not available")
        try:
            conn.execute("SET hnsw_enable_experimental_persistence = true")
        except duckdb.Error:
            pass  # Flag unavailable in DuckDB <0.10; embeddings still survive without it

        conn.execute("""
            CREATE TABLE files (
                id INTEGER PRIMARY KEY,
                path VARCHAR
            )
        """)
        conn.execute("""
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY,
                file_id INTEGER,
                code VARCHAR
            )
        """)
        conn.execute("""
            CREATE TABLE embeddings_128 (
                chunk_id INTEGER,
                embedding FLOAT[128],
                provider VARCHAR,
                model VARCHAR
            )
        """)

        conn.execute("INSERT INTO files VALUES (1, 'test.py')")
        rng = random.Random(42)
        test_vectors = {}
        for i in range(1, 6):
            conn.execute(
                "INSERT INTO chunks VALUES (?, 1, ?)",
                [i, f"def func_{i}(): pass"],
            )
            vec = [rng.uniform(-1, 1) for _ in range(128)]
            test_vectors[i] = vec
            conn.execute(
                "INSERT INTO embeddings_128 VALUES (?, ?::FLOAT[128], 'prov', 'mod')",
                [i, vec],
            )
        conn.execute("CHECKPOINT")
        conn.close()

        # Run EXPORT/IMPORT cycle (the core of compaction)
        export_dir = tmp_path / "export"
        new_db = tmp_path / "compact.duckdb"

        # Export
        conn = duckdb.connect(str(db_path), read_only=True)
        conn.execute("LOAD vss")
        conn.execute(f"EXPORT DATABASE '{export_dir.as_posix()}' (FORMAT PARQUET)")
        conn.close()

        # Import
        conn = duckdb.connect(str(new_db))
        conn.execute("LOAD vss")
        try:
            conn.execute("SET hnsw_enable_experimental_persistence = true")
        except duckdb.Error:
            pass  # Flag unavailable in DuckDB <0.10; embeddings still survive without it
        conn.execute(f"IMPORT DATABASE '{export_dir.as_posix()}'")
        conn.execute("CHECKPOINT")

        # Verify embeddings survived
        for chunk_id, expected_vec in test_vectors.items():
            result = conn.execute(
                "SELECT embedding FROM embeddings_128 WHERE chunk_id = ?",
                [chunk_id],
            ).fetchone()
            assert result is not None, (
                f"Embedding for chunk {chunk_id} missing after compaction"
            )
            actual_vec = list(result[0])
            assert len(actual_vec) == 128
            for j, (a, b) in enumerate(zip(expected_vec, actual_vec)):
                assert abs(a - b) < 1e-6, (
                    f"Vector mismatch at chunk {chunk_id}, dim {j}: {a} vs {b}"
                )

        # Also verify non-embedding data survived
        file_count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert file_count == 1
        assert chunk_count == 5

        conn.close()
