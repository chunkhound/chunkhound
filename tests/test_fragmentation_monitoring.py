"""Tests for fragmentation monitoring and batch size adjustments."""

from unittest.mock import MagicMock, patch, call
import pytest
from pathlib import Path

from chunkhound.core.config.database_config import DatabaseConfig
from chunkhound.providers.database.lancedb_provider import LanceDBProvider
from chunkhound.services.indexing_coordinator import IndexingCoordinator


class TestFragmentCountRetrieval:
    """Test get_fragment_count method with different data structures."""

    def test_get_fragment_count_dict_with_fragment_stats(self):
        """Test get_fragment_count with dict containing fragment_stats."""
        with patch.object(LanceDBProvider, '_create_connection', return_value=MagicMock()):
            from pathlib import Path
            provider = LanceDBProvider(":memory:", Path("/tmp"))

            # Mock table.stats() to return dict with fragment_stats
            mock_table = MagicMock()
            mock_table.stats.return_value = {
                "fragment_stats": {"num_fragments": 42}
            }
            provider._chunks_table = mock_table

            result = provider.get_fragment_count()
            assert result["chunks"] == 42

    def test_get_fragment_count_fragment_stats_with_attribute(self):
        """Test get_fragment_count with fragment_stats having num_fragments attribute."""
        with patch.object(LanceDBProvider, '_create_connection', return_value=MagicMock()):
            from pathlib import Path
            provider = LanceDBProvider(":memory:", Path("/tmp"))

            # Mock fragment_stats object with num_fragments attribute
            mock_fragment_stats = MagicMock()
            mock_fragment_stats.num_fragments = 25

            mock_table = MagicMock()
            mock_table.stats.return_value = {
                "fragment_stats": mock_fragment_stats
            }
            provider._chunks_table = mock_table

            result = provider.get_fragment_count()
            assert result["chunks"] == 25

    def test_get_fragment_count_fragment_stats_dict(self):
        """Test get_fragment_count with fragment_stats as dict."""
        with patch.object(LanceDBProvider, '_create_connection', return_value=MagicMock()):
            from pathlib import Path
            provider = LanceDBProvider(":memory:", Path("/tmp"))

            mock_table = MagicMock()
            mock_table.stats.return_value = {
                "fragment_stats": {"num_fragments": 15}
            }
            provider._chunks_table = mock_table

            result = provider.get_fragment_count()
            assert result["chunks"] == 15

    def test_get_fragment_count_no_fragment_stats(self):
        """Test get_fragment_count when stats dict lacks fragment_stats."""
        with patch.object(LanceDBProvider, '_create_connection', return_value=MagicMock()):
            from pathlib import Path
            provider = LanceDBProvider(":memory:", Path("/tmp"))

            mock_table = MagicMock()
            mock_table.stats.return_value = {"other_stats": "value"}
            provider._chunks_table = mock_table

            result = provider.get_fragment_count()
            assert result["chunks"] == 0

    def test_get_fragment_count_no_chunks_table(self):
        """Test get_fragment_count when chunks table is None."""
        with patch.object(LanceDBProvider, '_create_connection', return_value=MagicMock()):
            from pathlib import Path
            provider = LanceDBProvider(":memory:", Path("/tmp"))
            provider._chunks_table = None

            result = provider.get_fragment_count()
            assert "chunks" not in result or result.get("chunks", 0) == 0

    def test_get_fragment_count_exception_handling(self):
        """Test get_fragment_count handles exceptions gracefully."""
        with patch.object(LanceDBProvider, '_create_connection', return_value=MagicMock()):
            from pathlib import Path
            provider = LanceDBProvider(":memory:", Path("/tmp"))

            mock_table = MagicMock()
            mock_table.stats.side_effect = Exception("Stats error")
            provider._chunks_table = mock_table

            result = provider.get_fragment_count()
            assert result["chunks"] == 0

    def test_get_fragment_count_files_table(self):
        """Test get_fragment_count for files table."""
        with patch.object(LanceDBProvider, '_create_connection', return_value=MagicMock()):
            from pathlib import Path
            provider = LanceDBProvider(":memory:", Path("/tmp"))

            # Mock files table
            mock_files_table = MagicMock()
            mock_files_table.stats.return_value = {
                "fragment_stats": {"num_fragments": 8}
            }
            provider._files_table = mock_files_table

            result = provider.get_fragment_count()
            assert result["files"] == 8

    def test_get_fragment_count_both_tables(self):
        """Test get_fragment_count for both chunks and files tables."""
        with patch.object(LanceDBProvider, '_create_connection', return_value=MagicMock()):
            from pathlib import Path
            provider = LanceDBProvider(":memory:", Path("/tmp"))

            # Mock chunks table
            mock_chunks_table = MagicMock()
            mock_chunks_table.stats.return_value = {
                "fragment_stats": {"num_fragments": 30}
            }
            provider._chunks_table = mock_chunks_table

            # Mock files table
            mock_files_table = MagicMock()
            mock_files_table.stats.return_value = {
                "fragment_stats": {"num_fragments": 12}
            }
            provider._files_table = mock_files_table

            result = provider.get_fragment_count()
            assert result["chunks"] == 30
            assert result["files"] == 12


class TestBatchSizeAdjustment:
    """Test dynamic batch size adjustment based on fragmentation."""

    def test_determine_db_batch_size_no_fragmentation_adjustment(self):
        """Test batch size determination without fragmentation adjustment."""
        coordinator = IndexingCoordinator(None, Path("/tmp"), None, {})

        # Mock memory calculation by patching psutil before it's imported
        with patch.dict('sys.modules', {'psutil': MagicMock()}):
            with patch('psutil.virtual_memory') as mock_vm:
                mock_vm.return_value.available = 1024*1024*1024  # 1GB

                pending_inserts = [MagicMock()] * 1000  # 1000 chunks
                for chunk in pending_inserts:
                    chunk.code = "x" * 100  # 100 bytes each

                batch_size = coordinator._determine_db_batch_size(pending_inserts)
                # Should be within bounds and not adjusted for fragmentation
                assert 500 <= batch_size <= 20000

    def test_determine_db_batch_size_fragmentation_reduction_low(self):
        """Test batch size reduction with low fragmentation."""
        mock_db = MagicMock()
        mock_db.get_fragment_count.return_value = {"chunks": 150}
        coordinator = IndexingCoordinator(mock_db, Path("/tmp"), None, {})

        with patch.dict('sys.modules', {'psutil': MagicMock()}):
            with patch('psutil.virtual_memory') as mock_vm:
                mock_vm.return_value.available = 1024*1024*1024

                pending_inserts = [MagicMock()] * 1000
                for chunk in pending_inserts:
                    chunk.code = "x" * 100

                batch_size = coordinator._determine_db_batch_size(pending_inserts)
                # Should be slightly reduced (1.5x reduction for 150 fragments)
                assert batch_size < 20000  # Less than max

    def test_determine_db_batch_size_fragmentation_reduction_high(self):
        """Test batch size reduction with high fragmentation."""
        mock_db = MagicMock()
        mock_db.get_fragment_count.return_value = {"chunks": 600}
        coordinator = IndexingCoordinator(mock_db, Path("/tmp"), None, {})

        with patch.dict('sys.modules', {'psutil': MagicMock()}):
            with patch('psutil.virtual_memory') as mock_vm:
                mock_vm.return_value.available = 1024*1024*1024

                pending_inserts = [MagicMock()] * 1000
                for chunk in pending_inserts:
                    chunk.code = "x" * 100

                batch_size = coordinator._determine_db_batch_size(pending_inserts)
                # Should be significantly reduced (2x reduction for 600 fragments)
                assert batch_size <= 10000  # Much smaller

    def test_determine_db_batch_size_fragmentation_extreme(self):
        """Test batch size reduction with extreme fragmentation."""
        mock_db = MagicMock()
        mock_db.get_fragment_count.return_value = {"chunks": 2000}
        coordinator = IndexingCoordinator(mock_db, Path("/tmp"), None, {})

        with patch.dict('sys.modules', {'psutil': MagicMock()}):
            with patch('psutil.virtual_memory') as mock_vm:
                mock_vm.return_value.available = 1024*1024*1024

                pending_inserts = [MagicMock()] * 1000
                for chunk in pending_inserts:
                    chunk.code = "x" * 100

                batch_size = coordinator._determine_db_batch_size(pending_inserts)
                # Should be minimum allowed (2x reduction for 2000 fragments)
                assert batch_size <= 10000  # Much smaller

    def test_determine_db_batch_size_fragmentation_check_fails(self):
        """Test batch size when fragmentation check fails."""
        mock_db = MagicMock()
        mock_db.get_fragment_count.side_effect = Exception("DB error")
        coordinator = IndexingCoordinator(mock_db, Path("/tmp"), None, {})

        with patch.dict('sys.modules', {'psutil': MagicMock()}):
            with patch('psutil.virtual_memory') as mock_vm:
                mock_vm.return_value.available = 1024*1024*1024

                pending_inserts = [MagicMock()] * 1000
                for chunk in pending_inserts:
                    chunk.code = "x" * 100

                batch_size = coordinator._determine_db_batch_size(pending_inserts)
                # Should fall back to base calculation
                assert 500 <= batch_size <= 20000

    def test_determine_db_batch_size_no_db_provider(self):
        """Test batch size when no DB provider available."""
        coordinator = IndexingCoordinator(None, Path("/tmp"), None, {})

        with patch.dict('sys.modules', {'psutil': MagicMock()}):
            with patch('psutil.virtual_memory') as mock_vm:
                mock_vm.return_value.available = 1024*1024*1024

                pending_inserts = [MagicMock()] * 1000
                for chunk in pending_inserts:
                    chunk.code = "x" * 100

                batch_size = coordinator._determine_db_batch_size(pending_inserts)
                # Should use base calculation
                assert 500 <= batch_size <= 20000

    def test_determine_db_batch_size_psutil_unavailable(self):
        """Test batch size when psutil is unavailable."""
        coordinator = IndexingCoordinator(None, Path("/tmp"), None, {})

        # Mock psutil import failure - os.sysconf doesn't exist on Windows, so patch the fallback
        with patch.dict('sys.modules', {'psutil': None}):
            with patch('os.sysconf', return_value=1024*1024*1024, create=True):
                pending_inserts = [MagicMock()] * 1000
                for chunk in pending_inserts:
                    chunk.code = "x" * 100

                batch_size = coordinator._determine_db_batch_size(pending_inserts)
                # Should still work with fallback
                assert isinstance(batch_size, int)
                assert batch_size > 0


class TestPerformanceProfiling:
    """Test performance profiling in database operations."""

    @patch('chunkhound.providers.database.serial_executor.logger')
    def test_serial_executor_operation_profiling(self, mock_logger):
        """Test that serial executor profiles operation duration."""
        from chunkhound.providers.database.serial_executor import SerialDatabaseExecutor

        config = DatabaseConfig()
        executor = SerialDatabaseExecutor(config)

        # Mock provider and operation
        mock_provider = MagicMock()
        mock_provider._executor_test_operation = MagicMock(return_value="result")

        # Mock the connection and state functions
        with patch('chunkhound.providers.database.serial_executor.get_thread_local_connection', return_value=MagicMock()):
            with patch('chunkhound.providers.database.serial_executor.get_thread_local_state', return_value={"last_activity_time": 100.0}):
                with patch('chunkhound.providers.database.serial_executor.time') as mock_time:
                    # Provide enough time values for all calls
                    mock_time.time.side_effect = [100.0, 100.010, 100.020, 100.030, 100.040]

                    result = executor._execute_sync_once(mock_provider, "test_operation")

                    # Should log the operation duration (actual duration may vary)
                    mock_logger.debug.assert_called_with(
                        "Database operation 'test_operation' completed in 0.020s"
                    )

    @patch('chunkhound.providers.database.serial_executor.logger')
    def test_serial_executor_failed_operation_profiling(self, mock_logger):
        """Test profiling of failed operations."""
        from chunkhound.providers.database.serial_executor import SerialDatabaseExecutor

        config = DatabaseConfig()
        executor = SerialDatabaseExecutor(config)

        # Mock provider and failing operation
        mock_provider = MagicMock()
        mock_provider._executor_failing_operation = MagicMock(side_effect=ValueError("Operation failed"))

        # Mock the connection and state functions
        with patch('chunkhound.providers.database.serial_executor.get_thread_local_connection', return_value=MagicMock()):
            with patch('chunkhound.providers.database.serial_executor.get_thread_local_state', return_value={"last_activity_time": 200.0}):
                with patch('chunkhound.providers.database.serial_executor.time') as mock_time:
                    # Provide enough time values for all calls
                    mock_time.time.side_effect = [200.0, 200.005, 200.010, 200.015, 200.020]

                    with pytest.raises(ValueError):
                        executor._execute_sync_once(mock_provider, "failing_operation")

                    # Should log the failed operation duration
                    mock_logger.warning.assert_called_with(
                        "Database operation 'failing_operation' failed after 0.010s: Operation failed"
                    )

    @patch('chunkhound.providers.database.serial_executor.logger')
    def test_timeout_error_logging_with_duration(self, mock_logger):
        """Test that timeout errors include actual duration."""
        from chunkhound.providers.database.serial_executor import SerialDatabaseExecutor
        from concurrent.futures import TimeoutError as ConcurrentTimeoutError

        config = DatabaseConfig()
        executor = SerialDatabaseExecutor(config)

        # Mock a slow operation that will timeout
        mock_provider = MagicMock()
        mock_provider._executor_timeout_operation = MagicMock(side_effect=lambda *args, **kwargs: time.sleep(1))  # Long operation

        with patch('concurrent.futures.ThreadPoolExecutor.submit') as mock_submit:
            # Mock executor to timeout
            mock_future = MagicMock()
            mock_future.result.side_effect = ConcurrentTimeoutError()

            mock_submit.return_value = mock_future

            with patch('chunkhound.providers.database.serial_executor.time') as mock_time:
                mock_time.time.side_effect = [300.0, 300.500]  # 500ms timeout

                with pytest.raises(TimeoutError, match="timed out after 0.500s"):
                    executor._execute_sync_once(mock_provider, "timeout_operation")

                # Should log with actual duration
                mock_logger.error.assert_called_with(
                    "Database operation 'timeout_operation' timed out after 0.500s (configured timeout: 30.0s)"
                )


class TestBulkInsertPerformanceProfiling:
    """Test performance profiling in bulk insert operations."""

    @patch('chunkhound.providers.database.serial_executor.logger')
    @patch('chunkhound.providers.database.lancedb_provider.time')
    def test_bulk_insert_profiling(self, mock_time, mock_logger):
        """Test that bulk insert operations are profiled."""
        with patch.object(LanceDBProvider, '_create_connection', return_value=MagicMock()):
            from pathlib import Path
            provider = LanceDBProvider(":memory:", Path("/tmp"))

            # Mock time progression - use a callable that returns incrementing values
            time_values = [100.0]
            def time_increment():
                current = time_values[0]
                time_values[0] = current + 0.1
                return current
            mock_time.time.side_effect = time_increment

            # Mock the table operations
            mock_table = MagicMock()
            provider._chunks_table = mock_table

            # Mock PyArrow operations by patching the module
            mock_pa_table = MagicMock()
            with patch('chunkhound.providers.database.lancedb_provider.pa') as mock_pa:
                mock_pa.Table.from_pylist = MagicMock(return_value=mock_pa_table)
                with patch('chunkhound.providers.database.lancedb_provider.get_chunks_schema', return_value=MagicMock()):
                    # Mock the merge_insert operation
                    mock_table.merge_insert.return_value = MagicMock()

                    chunks = [MagicMock()] * 1000  # Mock chunks

                    result = provider.insert_chunks_batch(chunks)

                    # Should log performance for each batch
                    assert mock_logger.debug.call_count >= 1

                    # Check that timing logs were made
                    debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
                    timing_logs = [log for log in debug_calls if "completed in" in log]
                    assert len(timing_logs) > 0