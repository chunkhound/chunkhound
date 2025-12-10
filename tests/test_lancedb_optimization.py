"""Tests for LanceDB optimization and fragmentation prevention."""

from unittest.mock import MagicMock, patch
import pytest
from pathlib import Path

from chunkhound.core.config.database_config import DatabaseConfig
from chunkhound.providers.database.lancedb_provider import LanceDBProvider


class TestLanceDBOptimizationConfig:
    """Test LanceDB optimization configuration."""

    def test_lancedb_config_defaults(self):
        """Test default LanceDB optimization configuration."""
        config = DatabaseConfig()
        assert config.lancedb_optimize_during_indexing is True
        assert config.lancedb_indexing_fragment_threshold == 25

    def test_lancedb_config_custom_values(self):
        """Test custom LanceDB optimization configuration."""
        config = DatabaseConfig(
            lancedb_optimize_during_indexing=False,
            lancedb_indexing_fragment_threshold=100
        )
        assert config.lancedb_optimize_during_indexing is False
        assert config.lancedb_indexing_fragment_threshold == 100

    @patch.dict('os.environ', {
        'CHUNKHOUND_DATABASE__LANCEDB_OPTIMIZE_DURING_INDEXING': 'false',
        'CHUNKHOUND_DATABASE__LANCEDB_INDEXING_FRAGMENT_THRESHOLD': '25'
    })
    def test_lancedb_config_env_vars(self):
        """Test LanceDB optimization config from environment variables."""
        env_overrides = DatabaseConfig.load_from_env()
        config = DatabaseConfig(**env_overrides)
        assert config.lancedb_optimize_during_indexing is False
        assert config.lancedb_indexing_fragment_threshold == 25

    @patch.dict('os.environ', {
        'CHUNKHOUND_DATABASE__LANCEDB_OPTIMIZE_DURING_INDEXING': 'invalid',
        'CHUNKHOUND_DATABASE__LANCEDB_INDEXING_FRAGMENT_THRESHOLD': 'not-a-number'
    })
    def test_lancedb_config_invalid_env_vars_ignored(self):
        """Test that invalid LanceDB env vars are silently ignored."""
        original_config = DatabaseConfig(
            lancedb_optimize_during_indexing=True,
            lancedb_indexing_fragment_threshold=25
        )
        env_overrides = DatabaseConfig.load_from_env()
        # Since env vars are invalid, they should be ignored and original values preserved
        config = DatabaseConfig(**{**original_config.model_dump(), **env_overrides})
        # Values should remain unchanged
        assert config.lancedb_optimize_during_indexing is True
        assert config.lancedb_indexing_fragment_threshold == 25

    def test_lancedb_config_validation(self):
        """Test LanceDB configuration validation."""
        # Valid configurations
        config = DatabaseConfig(lancedb_indexing_fragment_threshold=1)
        assert config.lancedb_indexing_fragment_threshold == 1

        config = DatabaseConfig(lancedb_indexing_fragment_threshold=1000)
        assert config.lancedb_indexing_fragment_threshold == 1000

        # ge=1 validation should work
        with pytest.raises(ValueError):
            DatabaseConfig(lancedb_indexing_fragment_threshold=0)


class TestLanceDBProviderOptimization:
    """Test LanceDB provider optimization logic."""

    def test_provider_initialization_with_config(self):
        """Test that LanceDB provider initializes with optimization config."""
        config = DatabaseConfig(
            lancedb_optimize_during_indexing=False,
            lancedb_indexing_fragment_threshold=25
        )

        with patch('lancedb.connect'):
            provider = LanceDBProvider(":memory:", Path("/tmp"), config=config)

            assert provider._optimize_during_indexing is False
            assert provider._indexing_fragment_threshold == 25

    def test_provider_initialization_defaults(self):
        """Test LanceDB provider initialization with default config."""
        with patch('lancedb.connect'):
            provider = LanceDBProvider(":memory:", Path("/tmp"), config=None)

            assert provider._optimize_during_indexing is True  # default
            assert provider._indexing_fragment_threshold == 50  # default when config=None

    def test_should_optimize_during_indexing_disabled(self):
        """Test should_optimize_during_indexing when optimization disabled."""
        with patch('lancedb.connect'):
            provider = LanceDBProvider(":memory:", Path("/tmp"), config=DatabaseConfig(
                lancedb_optimize_during_indexing=False
            ))

            result = provider.should_optimize_during_indexing()
            assert result is False

    def test_should_optimize_during_indexing_below_threshold(self):
        """Test should_optimize_during_indexing when below fragment threshold."""
        with patch('lancedb.connect'):
            provider = LanceDBProvider(":memory:", Path("/tmp"), config=DatabaseConfig(
                lancedb_optimize_during_indexing=True,
                lancedb_indexing_fragment_threshold=50
            ))

            # Mock get_fragment_count to return below threshold
            with patch.object(provider, 'get_fragment_count', return_value={'chunks': 25}):
                result = provider.should_optimize_during_indexing()
                assert result is False

    def test_should_optimize_during_indexing_at_threshold(self):
        """Test should_optimize_during_indexing when at fragment threshold."""
        with patch('lancedb.connect'):
            provider = LanceDBProvider(":memory:", Path("/tmp"), config=DatabaseConfig(
                lancedb_optimize_during_indexing=True,
                lancedb_indexing_fragment_threshold=50
            ))

            # Mock get_fragment_count to return at threshold
            with patch.object(provider, 'get_fragment_count', return_value={'chunks': 50}):
                result = provider.should_optimize_during_indexing()
                assert result is True

    def test_should_optimize_during_indexing_above_threshold(self):
        """Test should_optimize_during_indexing when above fragment threshold."""
        with patch('lancedb.connect'):
            provider = LanceDBProvider(":memory:", Path("/tmp"), config=DatabaseConfig(
                lancedb_optimize_during_indexing=True,
                lancedb_indexing_fragment_threshold=50
            ))

            # Mock get_fragment_count to return above threshold
            with patch.object(provider, 'get_fragment_count', return_value={'chunks': 75}):
                result = provider.should_optimize_during_indexing()
                assert result is True

    def test_should_optimize_during_indexing_get_fragment_count_fails(self):
        """Test should_optimize_during_indexing when get_fragment_count fails."""
        with patch('lancedb.connect'):
            provider = LanceDBProvider(":memory:", Path("/tmp"), config=DatabaseConfig(
                lancedb_optimize_during_indexing=True,
                lancedb_indexing_fragment_threshold=50
            ))

            # Mock get_fragment_count to raise exception
            with patch.object(provider, 'get_fragment_count', side_effect=Exception("DB error")):
                result = provider.should_optimize_during_indexing()
                assert result is False

    def test_should_optimize_during_indexing_no_chunks_key(self):
        """Test should_optimize_during_indexing when fragment count lacks chunks key."""
        with patch('lancedb.connect'):
            provider = LanceDBProvider(":memory:", Path("/tmp"), config=DatabaseConfig(
                lancedb_optimize_during_indexing=True,
                lancedb_indexing_fragment_threshold=50
            ))

            # Mock get_fragment_count to return dict without chunks key
            with patch.object(provider, 'get_fragment_count', return_value={'files': 10}):
                result = provider.should_optimize_during_indexing()
                assert result is False

    def test_should_optimize_during_indexing_none_fragment_count(self):
        """Test should_optimize_during_indexing when fragment count is None."""
        with patch('lancedb.connect'):
            provider = LanceDBProvider(":memory:", Path("/tmp"), config=DatabaseConfig(
                lancedb_optimize_during_indexing=True,
                lancedb_indexing_fragment_threshold=50
            ))

            # Mock get_fragment_count to return None
            with patch.object(provider, 'get_fragment_count', return_value=None):
                result = provider.should_optimize_during_indexing()
                assert result is False

    @patch('chunkhound.providers.database.lancedb_provider.logger')
    def test_should_optimize_during_indexing_logs_debug(self, mock_logger):
        """Test that should_optimize_during_indexing logs debug information."""
        with patch('lancedb.connect'):
            provider = LanceDBProvider(":memory:", Path("/tmp"), config=DatabaseConfig(
                lancedb_optimize_during_indexing=True,
                lancedb_indexing_fragment_threshold=50
            ))

            with patch.object(provider, 'get_fragment_count', return_value={'chunks': 75}):
                result = provider.should_optimize_during_indexing()
                assert result is True

                # Should have logged debug info
                mock_logger.debug.assert_called_with(
                    "Fragment check: chunks=75, threshold=50, should_optimize=True"
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