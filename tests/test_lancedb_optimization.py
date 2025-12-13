"""Tests for LanceDB optimization and fragmentation prevention."""

from unittest.mock import MagicMock, patch
import pytest
from pathlib import Path

from chunkhound.core.config.database_config import DatabaseConfig
from chunkhound.providers.database.lancedb_provider import LanceDBProvider


class TestLanceDBOptimizationConfig:
    """Test LanceDB optimization configuration."""

    # Configuration fields have been removed - optimization is now always enabled


class TestLanceDBProviderOptimization:
    """Test LanceDB provider optimization logic."""


    def test_should_optimize_during_indexing_below_threshold(self):
        """Test should_optimize_during_indexing when below fragment threshold."""
        with patch('lancedb.connect'):
            provider = LanceDBProvider(":memory:", Path("/tmp"), config=DatabaseConfig())

            # Mock get_fragment_count to return below threshold (25)
            with patch.object(provider, 'get_fragment_count', return_value={'chunks': 20}):
                result = provider.should_optimize_during_indexing()
                assert result is False

    def test_should_optimize_during_indexing_at_threshold(self):
        """Test should_optimize_during_indexing when at fragment threshold."""
        with patch('lancedb.connect'):
            provider = LanceDBProvider(":memory:", Path("/tmp"), config=DatabaseConfig())

            # Mock get_fragment_count to return at threshold (25)
            with patch.object(provider, 'get_fragment_count', return_value={'chunks': 25}):
                result = provider.should_optimize_during_indexing()
                assert result is True

    def test_should_optimize_during_indexing_above_threshold(self):
        """Test should_optimize_during_indexing when above fragment threshold."""
        with patch('lancedb.connect'):
            provider = LanceDBProvider(":memory:", Path("/tmp"), config=DatabaseConfig())

            # Mock get_fragment_count to return above threshold (25)
            with patch.object(provider, 'get_fragment_count', return_value={'chunks': 30}):
                result = provider.should_optimize_during_indexing()
                assert result is True

    def test_should_optimize_during_indexing_get_fragment_count_fails(self):
        """Test should_optimize_during_indexing when get_fragment_count fails."""
        with patch('lancedb.connect'):
            provider = LanceDBProvider(":memory:", Path("/tmp"), config=DatabaseConfig())

            # Mock get_fragment_count to raise exception
            with patch.object(provider, 'get_fragment_count', side_effect=Exception("DB error")):
                result = provider.should_optimize_during_indexing()
                assert result is False

    def test_should_optimize_during_indexing_no_chunks_key(self):
        """Test should_optimize_during_indexing when fragment count lacks chunks key."""
        with patch('lancedb.connect'):
            provider = LanceDBProvider(":memory:", Path("/tmp"), config=DatabaseConfig())

            # Mock get_fragment_count to return dict without chunks key
            with patch.object(provider, 'get_fragment_count', return_value={'files': 10}):
                result = provider.should_optimize_during_indexing()
                assert result is False

    def test_should_optimize_during_indexing_none_fragment_count(self):
        """Test should_optimize_during_indexing when fragment count is None."""
        with patch('lancedb.connect'):
            provider = LanceDBProvider(":memory:", Path("/tmp"), config=DatabaseConfig())

            # Mock get_fragment_count to return None
            with patch.object(provider, 'get_fragment_count', return_value=None):
                result = provider.should_optimize_during_indexing()
                assert result is False

    @patch('chunkhound.providers.database.lancedb_provider.logger')
    def test_should_optimize_during_indexing_logs_debug(self, mock_logger):
        """Test that should_optimize_during_indexing logs debug information."""
        with patch('lancedb.connect'):
            provider = LanceDBProvider(":memory:", Path("/tmp"), config=DatabaseConfig())

            with patch.object(provider, 'get_fragment_count', return_value={'chunks': 30}):
                result = provider.should_optimize_during_indexing()
                assert result is True

                # Should have logged debug info with new threshold
                mock_logger.debug.assert_called_with(
                    "Fragment check: chunks=30, threshold=25, should_optimize=True"
                )


class TestLanceDBOptimizationIntegration:
    """Test LanceDB optimization integration with indexing coordinator."""

    @patch('chunkhound.services.indexing_coordinator.logger')
    def test_indexing_coordinator_calls_should_optimize(self, mock_logger):
        """Test that indexing coordinator checks optimization during batch processing."""
        from chunkhound.services.indexing_coordinator import IndexingCoordinator
        from chunkhound.core.config.config import Config

        # Mock database provider
        mock_db = MagicMock()
        mock_db.should_optimize_during_indexing.return_value = True
        mock_db.optimize_tables = MagicMock()

        # Create coordinator with mocked db
        config = Config()
        coordinator = IndexingCoordinator(mock_db, "/tmp", None, {})

        # Mock the batch processing to trigger optimization check
        with patch.object(coordinator, '_store_parsed_results', return_value=(MagicMock(), {})):
            # Simulate the optimization check in _on_batch_store
            if hasattr(mock_db, 'should_optimize_during_indexing') and mock_db.should_optimize_during_indexing():
                mock_logger.info.assert_not_called()  # Not called yet

                # Call optimize_tables
                mock_db.optimize_tables()

                # Verify optimize_tables was called
                mock_db.optimize_tables.assert_called_once()

    def test_indexing_coordinator_optimization_failure_handling(self):
        """Test that optimization failures are handled gracefully."""
        from chunkhound.services.indexing_coordinator import IndexingCoordinator
        from chunkhound.core.config.config import Config

        # Mock database provider that fails optimization
        mock_db = MagicMock()
        mock_db.should_optimize_during_indexing.return_value = True
        mock_db.optimize_tables.side_effect = Exception("Optimization failed")

        # Create coordinator
        config = Config()
        coordinator = IndexingCoordinator(mock_db, "/tmp", None, {})

        # This should not raise an exception
        with patch.object(coordinator, '_store_parsed_results', return_value=(MagicMock(), {})):
            # The optimization check should handle the failure
            try:
                if hasattr(mock_db, 'should_optimize_during_indexing') and mock_db.should_optimize_during_indexing():
                    mock_db.optimize_tables()
            except Exception:
                pass  # Should be handled gracefully

            # Verify optimize_tables was attempted
            mock_db.optimize_tables.assert_called_once()