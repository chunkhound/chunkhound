"""Test that batch_size and max_concurrent_batches config is forwarded
to EmbeddingService during indexing (issue #244).

The registry code path correctly passes these values, but the
IndexingCoordinator.process_directory() path was missing them.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.services.indexing_coordinator import IndexingCoordinator


def _make_config(batch_size=1, max_concurrent=2, db_batch_size=500):
    """Create a mock config with embedding and indexing settings."""
    config = MagicMock()
    config.embedding.batch_size = batch_size
    config.embedding.max_concurrent_batches = max_concurrent
    config.indexing.db_batch_size = db_batch_size
    return config


def _make_coordinator(tmp_path, config=None):
    """Create an IndexingCoordinator with optional config."""
    db = MagicMock()
    embedding_provider = MagicMock()
    return IndexingCoordinator(
        database_provider=db,
        base_directory=tmp_path,
        embedding_provider=embedding_provider,
        config=config,
    )


@pytest.mark.asyncio
async def test_embedding_service_receives_batch_config(tmp_path):
    """Verify EmbeddingService is constructed with batch config from user settings."""
    config = _make_config(batch_size=1, max_concurrent=2, db_batch_size=500)
    coordinator = _make_coordinator(tmp_path, config=config)

    with patch(
        "chunkhound.services.embedding_service.EmbeddingService"
    ) as MockEmbeddingService:
        mock_service = MagicMock()
        mock_service.generate_missing_embeddings = AsyncMock(
            return_value={"status": "ok", "generated": 0}
        )
        MockEmbeddingService.return_value = mock_service

        # Patch the local import inside generate_missing_embeddings
        with patch.dict(
            "sys.modules",
            {"chunkhound.services.embedding_service": MagicMock(EmbeddingService=MockEmbeddingService)},
        ):
            await coordinator.generate_missing_embeddings()

        MockEmbeddingService.assert_called_once()
        call_kwargs = MockEmbeddingService.call_args[1]
        assert call_kwargs["embedding_batch_size"] == 1
        assert call_kwargs["max_concurrent_batches"] == 2
        assert call_kwargs["db_batch_size"] == 500


@pytest.mark.asyncio
async def test_embedding_service_defaults_without_config(tmp_path):
    """Verify EmbeddingService uses defaults when no config is set."""
    coordinator = _make_coordinator(tmp_path, config=None)

    with patch(
        "chunkhound.services.embedding_service.EmbeddingService"
    ) as MockEmbeddingService:
        mock_service = MagicMock()
        mock_service.generate_missing_embeddings = AsyncMock(
            return_value={"status": "ok", "generated": 0}
        )
        MockEmbeddingService.return_value = mock_service

        with patch.dict(
            "sys.modules",
            {"chunkhound.services.embedding_service": MagicMock(EmbeddingService=MockEmbeddingService)},
        ):
            await coordinator.generate_missing_embeddings()

        MockEmbeddingService.assert_called_once()
        call_kwargs = MockEmbeddingService.call_args[1]
        assert call_kwargs["embedding_batch_size"] == 1000
        assert call_kwargs["max_concurrent_batches"] is None
        assert call_kwargs["db_batch_size"] == 5000
