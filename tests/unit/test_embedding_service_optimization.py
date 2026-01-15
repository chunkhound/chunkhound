"""Unit tests for EmbeddingService database optimization."""

import os
from unittest.mock import Mock, patch

import pytest

from chunkhound.services.embedding_service import EmbeddingService


class TestEmbeddingServiceOptimization:
    """Test periodic database optimization during embedding generation."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database provider with optimization support."""
        db = Mock()
        db.should_optimize = Mock(return_value=True)
        db.optimize_tables = Mock()
        return db

    @pytest.fixture
    def mock_provider(self):
        """Create mock embedding provider."""
        provider = Mock()
        provider.name = "test"
        provider.model = "test-model"
        return provider

    @pytest.fixture
    def service(self, mock_db, mock_provider):
        """Create EmbeddingService with mocked dependencies."""
        service = EmbeddingService(
            database_provider=mock_db,
            embedding_provider=mock_provider,
            embedding_batch_size=10,
        )
        return service

    def test_optimization_triggers_when_threshold_reached(self, service, mock_db):
        """Verify optimization triggers when batch count reaches frequency."""
        # Set frequency to 5 batches for testing
        service._optimization_batch_frequency = 5
        service._completed_batches = 5  # At threshold

        # Call optimization check
        service._maybe_optimize_database(successful_batches=1)

        # Verify optimization was called
        assert mock_db.optimize_tables.called, (
            "optimize_tables should be called when batch count >= frequency"
        )

    def test_optimization_counter_resets_after_success(self, service, mock_db):
        """Verify counter resets to 0 after successful optimization."""
        service._optimization_batch_frequency = 5
        service._completed_batches = 5

        # Trigger optimization
        service._maybe_optimize_database(successful_batches=1)

        # Verify counter reset
        assert service._completed_batches == 0, (
            "Counter should reset to 0 after successful optimization"
        )

    def test_optimization_skipped_if_not_needed(self, service, mock_db):
        """Verify optimization skipped when should_optimize returns False."""
        service._optimization_batch_frequency = 5
        service._completed_batches = 5
        mock_db.should_optimize.return_value = False  # No optimization needed

        service._maybe_optimize_database(successful_batches=1)

        # Verify optimize_tables NOT called
        assert not mock_db.optimize_tables.called, (
            "optimize_tables should not be called when should_optimize returns False"
        )

    def test_invalid_env_var_falls_back_to_default(self, mock_db, mock_provider):
        """Verify invalid environment variable falls back to default."""
        with patch.dict(
            os.environ, {"CHUNKHOUND_EMBEDDING_OPTIMIZATION_BATCH_FREQUENCY": "invalid"}
        ):
            service = EmbeddingService(
                database_provider=mock_db,
                embedding_provider=mock_provider,
            )

            # Should fall back to default of 1000
            assert service._optimization_batch_frequency == 1000

    def test_negative_env_var_falls_back_to_default(self, mock_db, mock_provider):
        """Verify negative environment variable falls back to default."""
        with patch.dict(
            os.environ, {"CHUNKHOUND_EMBEDDING_OPTIMIZATION_BATCH_FREQUENCY": "-5"}
        ):
            service = EmbeddingService(
                database_provider=mock_db,
                embedding_provider=mock_provider,
            )

            assert service._optimization_batch_frequency == 1000

    def test_zero_env_var_falls_back_to_default(self, mock_db, mock_provider):
        """Verify zero environment variable falls back to default."""
        with patch.dict(
            os.environ, {"CHUNKHOUND_EMBEDDING_OPTIMIZATION_BATCH_FREQUENCY": "0"}
        ):
            service = EmbeddingService(
                database_provider=mock_db,
                embedding_provider=mock_provider,
            )

            assert service._optimization_batch_frequency == 1000

    def test_valid_env_var_is_used(self, mock_db, mock_provider):
        """Verify valid environment variable is used."""
        with patch.dict(
            os.environ, {"CHUNKHOUND_EMBEDDING_OPTIMIZATION_BATCH_FREQUENCY": "500"}
        ):
            service = EmbeddingService(
                database_provider=mock_db,
                embedding_provider=mock_provider,
            )

            assert service._optimization_batch_frequency == 500

    def test_optimization_counter_not_reset_on_failure(self, service, mock_db):
        """Verify counter NOT reset if optimization fails."""
        service._optimization_batch_frequency = 5
        service._completed_batches = 5

        # Make optimization fail
        mock_db.optimize_tables.side_effect = Exception("Optimization failed")

        # Trigger optimization attempt
        service._maybe_optimize_database(successful_batches=1)

        # Verify counter NOT reset (still 5)
        assert service._completed_batches == 5, (
            "Counter should NOT reset if optimization fails, "
            "allowing retry at next batch milestone"
        )

    def test_optimization_failure_is_non_fatal(self, service, mock_db):
        """Verify optimization failure doesn't halt embedding generation."""
        service._optimization_batch_frequency = 5
        service._completed_batches = 5

        # Make optimization fail
        mock_db.optimize_tables.side_effect = Exception("Optimization failed")

        # Should not raise exception
        try:
            service._maybe_optimize_database(successful_batches=1)
        except Exception as e:
            pytest.fail(f"Optimization failure should be non-fatal, but raised: {e}")
