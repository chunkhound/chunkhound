"""Unit tests for EmbeddingService database optimization."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from chunkhound.core.models import Chunk, File
from chunkhound.core.types.common import ChunkType, Language
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.embedding_service import EmbeddingService


class TestEmbeddingServiceOptimization:
    """Test periodic database optimization during embedding generation."""

    @pytest.fixture
    def provider_with_reclaimable_space(self, tmp_path: Path):
        """Create a real DuckDB provider with reclaimable free blocks."""
        db_path = tmp_path / "test.duckdb"
        provider = DuckDBProvider(str(db_path), base_directory=tmp_path)
        provider.connect()

        # Insert data then delete most of it to create free blocks
        for i in range(10):
            test_file = File(
                path=f"test_{i}.py",
                mtime=1234567890.0,
                language=Language.PYTHON,
                size_bytes=1000,
            )
            file_id = provider.insert_file(test_file)
            for j in range(20):
                chunk = Chunk(
                    file_id=file_id,
                    code=f"def func_{i}_{j}(): pass " + "x" * 500,
                    start_line=j * 5 + 1,
                    end_line=j * 5 + 5,
                    chunk_type=ChunkType.FUNCTION,
                    symbol=f"func_{i}_{j}",
                    language=Language.PYTHON,
                )
                provider.insert_chunk(chunk)

        provider.optimize_tables()

        # Delete most files to create free blocks
        for i in range(8):
            provider.delete_file_completely(f"test_{i}.py")
        provider.optimize_tables()

        yield provider

        if provider.is_connected:
            provider.disconnect()

    @pytest.fixture
    def clean_provider(self, tmp_path: Path):
        """Create a real DuckDB provider with no reclaimable space."""
        db_path = tmp_path / "clean.duckdb"
        provider = DuckDBProvider(str(db_path), base_directory=tmp_path)
        provider.connect()

        # Insert minimal data, no deletions
        test_file = File(
            path="keep.py",
            mtime=1234567890.0,
            language=Language.PYTHON,
            size_bytes=100,
        )
        file_id = provider.insert_file(test_file)
        chunk = Chunk(
            file_id=file_id,
            code="def keep(): pass",
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.FUNCTION,
            symbol="keep",
            language=Language.PYTHON,
        )
        provider.insert_chunk(chunk)
        provider.optimize_tables()

        yield provider

        if provider.is_connected:
            provider.disconnect()

    def test_optimization_triggers_when_threshold_reached(
        self, provider_with_reclaimable_space,
    ):
        """Verify optimization triggers when batch count reaches frequency."""
        service = EmbeddingService(database_provider=provider_with_reclaimable_space)
        service._optimization_batch_frequency = 5
        service._completed_batches = 5

        with patch.object(
            provider_with_reclaimable_space, "optimize_tables",
            wraps=provider_with_reclaimable_space.optimize_tables,
        ) as spy:
            service._maybe_optimize_database()
            assert spy.called, "optimize_tables should be called when reclaimable space exists"

        assert service._completed_batches == 0

    def test_optimization_counter_resets_after_success(
        self, provider_with_reclaimable_space,
    ):
        """Verify counter resets to 0 after successful optimization."""
        service = EmbeddingService(database_provider=provider_with_reclaimable_space)
        service._optimization_batch_frequency = 5
        service._completed_batches = 5

        service._maybe_optimize_database()

        assert service._completed_batches == 0, (
            "Counter should reset to 0 after successful optimization"
        )

    def test_optimization_skipped_if_not_needed(self, clean_provider):
        """Verify optimization skipped when no reclaimable space exists."""
        service = EmbeddingService(database_provider=clean_provider)
        service._optimization_batch_frequency = 5
        service._completed_batches = 5

        with patch.object(
            clean_provider, "optimize_tables",
            wraps=clean_provider.optimize_tables,
        ) as spy:
            service._maybe_optimize_database()
            assert not spy.called, "optimize_tables should not be called without reclaimable space"

        # Counter still resets to avoid per-batch overhead
        assert service._completed_batches == 0

    def test_optimization_counter_resets_even_on_failure(
        self, provider_with_reclaimable_space,
    ):
        """Counter always resets to avoid per-batch pragma_storage_info() overhead."""
        service = EmbeddingService(database_provider=provider_with_reclaimable_space)
        service._optimization_batch_frequency = 5
        service._completed_batches = 5

        # Patch just optimize_tables to raise
        with patch.object(
            provider_with_reclaimable_space, "optimize_tables",
            side_effect=Exception("Optimization failed"),
        ):
            service._maybe_optimize_database()

        assert service._completed_batches == 0, (
            "Counter should always reset to avoid per-batch overhead"
        )

    def test_optimization_failure_is_non_fatal(
        self, provider_with_reclaimable_space,
    ):
        """Verify optimization failure doesn't halt embedding generation."""
        service = EmbeddingService(database_provider=provider_with_reclaimable_space)
        service._optimization_batch_frequency = 5
        service._completed_batches = 5

        with patch.object(
            provider_with_reclaimable_space, "optimize_tables",
            side_effect=Exception("Optimization failed"),
        ):
            # Should not raise
            service._maybe_optimize_database()


class TestOptimizationFrequencyParsing:
    """Test environment variable parsing for optimization frequency."""

    @pytest.fixture
    def provider(self, tmp_path: Path):
        """Minimal DuckDB provider for construction tests."""
        db_path = tmp_path / "freq.duckdb"
        provider = DuckDBProvider(str(db_path), base_directory=tmp_path)
        provider.connect()
        yield provider
        if provider.is_connected:
            provider.disconnect()

    def test_invalid_env_var_falls_back_to_default(self, provider):
        """Verify invalid environment variable falls back to default."""
        with patch.dict(
            os.environ, {"CHUNKHOUND_EMBEDDING_OPTIMIZATION_BATCH_FREQUENCY": "invalid"}
        ):
            service = EmbeddingService(database_provider=provider)
            assert service._optimization_batch_frequency == 1000

    def test_negative_env_var_falls_back_to_default(self, provider):
        """Verify negative environment variable falls back to default."""
        with patch.dict(
            os.environ, {"CHUNKHOUND_EMBEDDING_OPTIMIZATION_BATCH_FREQUENCY": "-5"}
        ):
            service = EmbeddingService(database_provider=provider)
            assert service._optimization_batch_frequency == 1000

    def test_zero_env_var_falls_back_to_default(self, provider):
        """Verify zero environment variable falls back to default."""
        with patch.dict(
            os.environ, {"CHUNKHOUND_EMBEDDING_OPTIMIZATION_BATCH_FREQUENCY": "0"}
        ):
            service = EmbeddingService(database_provider=provider)
            assert service._optimization_batch_frequency == 1000

    def test_valid_env_var_is_used(self, provider):
        """Verify valid environment variable is used."""
        with patch.dict(
            os.environ, {"CHUNKHOUND_EMBEDDING_OPTIMIZATION_BATCH_FREQUENCY": "500"}
        ):
            service = EmbeddingService(database_provider=provider)
            assert service._optimization_batch_frequency == 500
