"""Tests for rerank_api_key integration in EmbeddingProviders."""

import pytest
from unittest.mock import AsyncMock, patch
from chunkhound.core.config.embedding_config import EmbeddingConfig
from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory


@pytest.mark.asyncio
async def test_openai_provider_rerank_api_key_usage():
    """Test that OpenAI provider uses rerank_api_key for reranking requests."""
    config = EmbeddingConfig(
        provider="openai",
        api_key="provider-key",
        rerank_api_key="dedicated-rerank-key",
        rerank_url="https://rerank.example.com",
        rerank_model="rerank-1",
        rerank_format="cohere",
        model="text-embedding-3-small",
    )

    provider = EmbeddingProviderFactory.create_provider(config)

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = lambda: {"results": [{"index": 0, "relevance_score": 0.9}]}
    mock_response.raise_for_status = lambda: None

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        await provider.rerank("query", ["doc1"])

        # Check that the dedicated rerank key was used in headers
        args, kwargs = mock_post.call_args
        headers = kwargs.get("headers", {})
        assert headers.get("Authorization") == "Bearer dedicated-rerank-key"


@pytest.mark.asyncio
async def test_openai_provider_fallback_to_provider_api_key():
    """Test that OpenAI provider falls back to provider api_key if rerank_api_key is missing."""
    config = EmbeddingConfig(
        provider="openai",
        api_key="provider-key",
        rerank_url="https://rerank.example.com",
        rerank_model="rerank-1",
        rerank_format="cohere",
        model="text-embedding-3-small",
    )

    provider = EmbeddingProviderFactory.create_provider(config)

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = lambda: {"results": [{"index": 0, "relevance_score": 0.9}]}
    mock_response.raise_for_status = lambda: None

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        await provider.rerank("query", ["doc1"])

        # Check that the provider key was used in headers
        args, kwargs = mock_post.call_args
        headers = kwargs.get("headers", {})
        assert headers.get("Authorization") == "Bearer provider-key"


@pytest.mark.asyncio
async def test_mistral_provider_rerank_api_key_usage():
    """Test that Mistral provider uses rerank_api_key for reranking requests."""
    config = EmbeddingConfig(
        provider="mistral",
        api_key="mistral-provider-key",
        rerank_api_key="mistral-rerank-key",
        rerank_url="https://rerank.example.com",
        rerank_model="rerank-1",
        rerank_format="cohere",
        model="codestral-embed",
    )

    provider = EmbeddingProviderFactory.create_provider(config)

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = lambda: {"results": [{"index": 0, "relevance_score": 0.9}]}
    mock_response.raise_for_status = lambda: None

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        await provider.rerank("query", ["doc1"])

        # Check that the dedicated rerank key was used in headers
        args, kwargs = mock_post.call_args
        headers = kwargs.get("headers", {})
        assert headers.get("Authorization") == "Bearer mistral-rerank-key"
