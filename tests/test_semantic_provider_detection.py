"""Test that semantic search correctly detects configured embedding provider.

This test verifies the bug where provider/model parameters with hardcoded
defaults prevent automatic detection of the configured embedding provider.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from chunkhound.mcp_server.tools import search_impl
from chunkhound.database_factory import DatabaseServices
from chunkhound.embeddings import EmbeddingManager


@pytest.mark.asyncio
async def test_semantic_search_uses_configured_provider():
    """Verify semantic search uses configured provider when not explicitly specified.

    This test catches the regression where provider/model defaults changed from
    None to hardcoded strings, breaking automatic provider detection.
    """
    # Setup: Create a mock embedding provider with non-default configuration
    mock_provider = Mock()
    mock_provider.name = "ollama"  # NOT "openai"
    mock_provider.model = "nomic-embed-text"  # NOT "text-embedding-3-small"

    # Create mock embedding manager
    mock_embedding_manager = Mock(spec=EmbeddingManager)
    mock_embedding_manager.list_providers.return_value = ["ollama"]
    mock_embedding_manager.get_provider.return_value = mock_provider

    # Create mock database services
    mock_services = Mock(spec=DatabaseServices)
    mock_provider_db = Mock()
    mock_provider_db.is_connected = True
    mock_services.provider = mock_provider_db

    # Create mock search service with async search method
    mock_search_service = Mock()
    mock_search_service.search_semantic = AsyncMock(return_value=([], {"total": 0, "page": 1, "page_size": 10}))
    mock_services.search_service = mock_search_service

    # Execute: Call search_impl with type="semantic" WITHOUT specifying provider/model
    # This should use the configured provider (ollama/nomic-embed-text)
    await search_impl(
        services=mock_services,
        embedding_manager=mock_embedding_manager,
        type="semantic",
        query="test query",
    )

    # Verify: Check that search was called with the CONFIGURED provider, not hardcoded defaults
    mock_search_service.search_semantic.assert_called_once()
    call_kwargs = mock_search_service.search_semantic.call_args.kwargs

    # CRITICAL ASSERTION: Should use configured provider, not hardcoded "openai"
    assert call_kwargs["provider"] == "ollama", \
        f"Expected provider='ollama' but got provider='{call_kwargs['provider']}'. " \
        "Semantic search should use configured provider when not explicitly specified."

    # CRITICAL ASSERTION: Should use configured model, not hardcoded "text-embedding-3-small"
    assert call_kwargs["model"] == "nomic-embed-text", \
        f"Expected model='nomic-embed-text' but got model='{call_kwargs['model']}'. " \
        "Semantic search should use configured model when not explicitly specified."


