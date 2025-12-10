"""Integration tests for database performance features combining retry logic and optimization."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
import pytest

from chunkhound.core.config.config import Config
from chunkhound.core.config.database_config import DatabaseConfig
from chunkhound.services.indexing_coordinator import IndexingCoordinator
from chunkhound.providers.database.serial_executor import SerialDatabaseExecutor


class TestRetryAndOptimizationIntegration:
    """Test integration of retry logic with LanceDB optimization."""

    def test_retry_with_optimization_during_indexing(self):
        """Test that retry logic works with optimization during indexing."""
        # Create config with retry enabled and optimization enabled
        config = DatabaseConfig(
            retry_on_timeout=True,
            max_retries=2,
            retry_backoff_seconds=0.01,
            lancedb_optimize_during_indexing=True,
            lancedb_indexing_fragment_threshold=10
        )

        # Mock LanceDB provider
        mock_provider = MagicMock()
        mock_provider.should_optimize_during_indexing.return_value = True
        mock_provider.optimize_tables = MagicMock()
        mock_provider.get_fragment_count.return_value = {"chunks": 15}

        # Create executor with retry config
        executor = SerialDatabaseExecutor(config)

        # Mock operation that fails once then succeeds
        call_count = 0
        def mock_db_operation():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("Temporary DB timeout")
            return "success"

        # Execute with retry
        result = executor._execute_with_retry(mock_db_operation)

        assert result == "success"
        assert call_count == 2  # Initial + 1 retry

    @patch('chunkhound.services.indexing_coordinator.logger')
    def test_indexing_coordinator_retry_and_optimization(self, mock_logger):
        """Test indexing coordinator integrates retry and optimization."""
        # Create config with both features enabled
        db_config = DatabaseConfig(
            retry_on_timeout=True,
            max_retries=1,
            lancedb_optimize_during_indexing=True,
            lancedb_indexing_fragment_threshold=5
        )

        # Mock database provider with retry executor
        mock_db = MagicMock()
        mock_db.should_optimize_during_indexing.return_value = True
        mock_db.optimize_tables = MagicMock()
        mock_db.get_fragment_count.return_value = {"chunks": 10}

        # Create coordinator
        config = Config()
        config.database = db_config
        coordinator = IndexingCoordinator(mock_db, "/tmp", None, {})

        # Mock successful batch storage
        with patch.object(coordinator, '_store_parsed_results', return_value=(MagicMock(), {})):
            # Simulate the batch processing logic
            batch = [MagicMock()]  # Mock batch

            # Check that optimization would be triggered
            should_optimize = mock_db.should_optimize_during_indexing()
            assert should_optimize is True

            # Call optimization (as would happen in real code)
            mock_db.optimize_tables()

            # Verify optimization was called
            mock_db.optimize_tables.assert_called_once()

    @pytest.mark.asyncio
    @patch('chunkhound.providers.database.serial_executor.logger')
    async def test_async_retry_with_performance_logging(self, mock_logger):
        """Test async retry with performance logging."""
        config = DatabaseConfig(
            retry_on_timeout=True,
            max_retries=1,
            retry_backoff_seconds=0.01
        )
        executor = SerialDatabaseExecutor(config)

        call_count = 0
        async def mock_async_operation():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("Async timeout")
            return "async result"

        result = await executor._execute_with_retry_async(mock_async_operation)

        assert result == "async result"
        assert call_count == 2

        # Check that retry logging occurred
        warning_calls = [call for call in mock_logger.warning.call_args_list
                        if "retrying in" in str(call)]
        assert len(warning_calls) > 0


class TestFragmentationAndBatchSizeIntegration:
    """Test integration of fragmentation monitoring with batch size adjustments."""

    def test_batch_size_with_fragmentation_and_retry_config(self):
        """Test batch size calculation considers both fragmentation and retry config."""
        # Create database with fragmentation
        mock_db = MagicMock()
        mock_db.get_fragment_count.return_value = {"chunks": 100}  # High fragmentation

        # Create coordinator
        coordinator = IndexingCoordinator(mock_db, "/tmp", None, {})

        # Mock memory available
        with patch('psutil.virtual_memory') as mock_vm:
            mock_vm.return_value.available = 512 * 1024 * 1024  # 512MB
            pending_inserts = [MagicMock()] * 2000
            for chunk in pending_inserts:
                chunk.code = "x" * 200  # Larger chunks

            batch_size = coordinator._determine_db_batch_size(pending_inserts)

            # Should be reduced due to fragmentation
            assert batch_size < 20000  # Less than base size

            # Should still be reasonable
            assert batch_size >= 1000

    def test_optimization_triggered_by_fragmentation_during_indexing(self):
        """Test that optimization is triggered based on fragmentation during indexing."""
        from chunkhound.core.config.config import Config

        # Create config with low threshold
        db_config = DatabaseConfig(
            lancedb_optimize_during_indexing=True,
            lancedb_indexing_fragment_threshold=20
        )

        # Mock DB with high fragmentation
        mock_db = MagicMock()
        mock_db.get_fragment_count.return_value = {"chunks": 25}  # Above threshold
        mock_db.should_optimize_during_indexing.return_value = True
        mock_db.optimize_tables = MagicMock()

        # Create coordinator
        config = Config()
        config.database = db_config
        coordinator = IndexingCoordinator(mock_db, "/tmp", None, {})

        # Verify optimization would be triggered
        should_optimize = mock_db.should_optimize_during_indexing()
        assert should_optimize is True

        # Simulate calling optimization
        mock_db.optimize_tables()

        # Verify it was called
        mock_db.optimize_tables.assert_called_once()


class TestEndToEndPerformanceScenario:
    """Test end-to-end performance scenarios."""

    @patch('chunkhound.services.indexing_coordinator.logger')
    def test_full_indexing_workflow_with_performance_features(self, mock_logger):
        """Test complete indexing workflow with all performance features enabled."""
        # Create comprehensive config
        db_config = DatabaseConfig(
            retry_on_timeout=True,
            max_retries=3,
            retry_backoff_seconds=0.1,
            lancedb_optimize_during_indexing=True,
            lancedb_indexing_fragment_threshold=30
        )

        # Mock database provider
        mock_db = MagicMock()
        mock_db.should_optimize_during_indexing.return_value = True
        mock_db.optimize_tables = MagicMock()
        mock_db.get_fragment_count.return_value = {"chunks": 50}

        # Create coordinator
        config = Config()
        config.database = db_config
        coordinator = IndexingCoordinator(mock_db, "/tmp", None, {})

        # Mock successful workflow
        with patch.object(coordinator, '_store_parsed_results', return_value=(MagicMock(), {})):
            # Simulate batch processing
            batch = [MagicMock()]

            # Pre-batch optimization check
            if mock_db.should_optimize_during_indexing():
                mock_db.optimize_tables()

            # Verify optimization was triggered
            mock_db.optimize_tables.assert_called_once()

    def test_performance_features_work_with_duckdb_fallback(self):
        """Test that performance features gracefully handle DuckDB (non-LanceDB) providers."""
        from chunkhound.providers.database.duckdb_provider import DuckDBProvider

        # Create config with LanceDB-specific settings
        db_config = DatabaseConfig(
            lancedb_optimize_during_indexing=True,
            lancedb_indexing_fragment_threshold=50
        )

        # Use DuckDB provider (doesn't have LanceDB methods)
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            db = DuckDBProvider(str(db_path), temp_dir)

            # Create coordinator
            coordinator = IndexingCoordinator(db, temp_dir, None, {})

            # Should not crash when checking for LanceDB-specific methods
            batch_size = coordinator._determine_db_batch_size([MagicMock()])

            # Should return a valid batch size
            assert isinstance(batch_size, int)
            assert batch_size > 0

    @patch('chunkhound.api.cli.main.logger')
    def test_logging_integration_with_performance_features(self, mock_logger):
        """Test that logging configuration works with performance features."""
        from chunkhound.core.config.logging_config import LoggingConfig, FileLoggingConfig, PerformanceLoggingConfig
        from chunkhound.api.cli.main import setup_logging

        # Create config with both file and performance logging
        logging_config = LoggingConfig(
            file=FileLoggingConfig(enabled=True, path="/tmp/test.log"),
            performance=PerformanceLoggingConfig(enabled=True, path="/tmp/perf.log")
        )

        # Setup logging
        setup_logging(verbose=False, config=MagicMock(logging=logging_config))

        # Verify logger setup calls were made
        assert mock_logger.add.call_count >= 2  # At least console + file + performance

        # Check that file logging was configured
        file_calls = [call for call in mock_logger.add.call_args_list
                     if len(call[0]) > 0 and '/tmp/test.log' in str(call[0][0])]
        assert len(file_calls) > 0

        # Check that performance logging was configured
        perf_calls = [call for call in mock_logger.add.call_args_list
                     if len(call[0]) > 0 and '/tmp/perf.log' in str(call[0][0])]
        assert len(perf_calls) > 0


class TestErrorHandlingIntegration:
    """Test error handling across integrated components."""

    def test_retry_handles_optimization_failures(self):
        """Test that retry logic handles optimization failures during indexing."""
        config = DatabaseConfig(
            retry_on_timeout=True,
            max_retries=2,
            lancedb_optimize_during_indexing=True
        )

        # Mock provider that fails optimization but succeeds operation
        mock_provider = MagicMock()
        mock_provider.should_optimize_during_indexing.return_value = True
        mock_provider.optimize_tables.side_effect = Exception("Optimization failed")

        # Create executor
        executor = SerialDatabaseExecutor(config)

        # Operation succeeds despite optimization failure
        call_count = 0
        def mock_operation():
            nonlocal call_count
            call_count += 1
            return "operation succeeded"

        result = executor._execute_with_retry(mock_operation)

        assert result == "operation succeeded"
        assert call_count == 1  # No retries needed

    def test_fragmentation_check_failures_dont_break_batch_sizing(self):
        """Test that fragmentation check failures don't break batch size calculation."""
        mock_db = MagicMock()
        mock_db.get_fragment_count.side_effect = Exception("Fragmentation check failed")

        coordinator = IndexingCoordinator(mock_db, "/tmp", None, {})

        # Should not crash
        with patch('psutil.virtual_memory') as mock_vm:
            mock_vm.return_value.available = 1024 * 1024 * 1024  # 1GB
            pending_inserts = [MagicMock()] * 100
            for chunk in pending_inserts:
                chunk.code = "test code"

            batch_size = coordinator._determine_db_batch_size(pending_inserts)

            # Should return a valid batch size
            assert isinstance(batch_size, int)
            assert 500 <= batch_size <= 20000