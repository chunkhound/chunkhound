"""Unit tests for MistralEmbeddingProvider.

Tests cover:
- Provider initialization and configuration
- Batch size handling
- Model configuration updates
- Output dimension validation
- Error handling and retries
- Usage statistics
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# Skip all tests if mistralai is not installed
pytest.importorskip("mistralai", reason="mistralai package not installed")

from chunkhound.providers.embeddings.mistral_provider import (
    MISTRAL_MODEL_CONFIG,
    MistralEmbeddingProvider,
)


class TestMistralProviderInitialization:
    """Tests for MistralEmbeddingProvider initialization."""

    def test_initialization_with_defaults(self):
        """Test provider initializes with default values."""
        provider = MistralEmbeddingProvider(api_key="test-key")

        assert provider.name == "mistral"
        assert provider.model == "codestral-embed"
        assert provider.batch_size == 32
        assert provider.dims == 1536
        assert provider.distance == "cosine"
        assert provider._timeout == 30
        assert provider._retry_attempts == 3

    def test_initialization_with_custom_values(self):
        """Test provider initializes with custom values."""
        provider = MistralEmbeddingProvider(
            api_key="test-key",
            model="mistral-embed",
            batch_size=64,
            timeout=60,
            retry_attempts=5,
            retry_delay=2.0,
        )

        assert provider.model == "mistral-embed"
        assert provider.batch_size == 64
        assert provider._timeout == 60
        assert provider._retry_attempts == 5
        assert provider._retry_delay == 2.0

    def test_initialization_with_output_dimension(self):
        """Test provider initializes with Matryoshka output dimension."""
        provider = MistralEmbeddingProvider(
            api_key="test-key",
            output_dimension=768,
        )

        assert provider.dims == 768
        assert provider._output_dimension == 768

    def test_initialization_validates_output_dimension(self):
        """Test that invalid output_dimension raises ValueError."""
        # Too high
        with pytest.raises(ValueError, match="output_dimension must be between"):
            MistralEmbeddingProvider(
                api_key="test-key",
                output_dimension=5000,  # Max is 3072
            )

        # Too low
        with pytest.raises(ValueError, match="output_dimension must be between"):
            MistralEmbeddingProvider(
                api_key="test-key",
                output_dimension=0,
            )

    def test_initialization_caps_batch_size_to_model_max(self):
        """Test that batch_size is capped at model max_texts_per_batch."""
        provider = MistralEmbeddingProvider(
            api_key="test-key",
            batch_size=5000,  # Much higher than model max of 1000
        )

        # Should be capped to 1000 (model's max_texts_per_batch)
        assert provider.batch_size == 1000

    def test_model_specific_dimensions(self):
        """Test that model-specific dimensions are correctly configured."""
        # codestral-embed (default)
        provider = MistralEmbeddingProvider(api_key="test-key")
        assert provider.dims == 1536

        # mistral-embed
        provider = MistralEmbeddingProvider(api_key="test-key", model="mistral-embed")
        assert provider.dims == 1024


class TestMistralProviderConfig:
    """Tests for provider configuration management."""

    def test_config_property(self):
        """Test that config property returns correct EmbeddingConfig."""
        provider = MistralEmbeddingProvider(
            api_key="test-key",
            model="codestral-embed",
            batch_size=32,
        )

        config = provider.config

        assert config.provider == "mistral"
        assert config.model == "codestral-embed"
        assert config.dims == 1536
        assert config.distance == "cosine"
        assert config.batch_size == 32
        assert config.base_url == "https://api.mistral.ai"

    def test_update_config_model_updates_model_config(self):
        """Test that updating model also updates model_config."""
        provider = MistralEmbeddingProvider(api_key="test-key", model="codestral-embed")

        # Initial state
        assert provider.model == "codestral-embed"
        assert provider._model_config["default_dimension"] == 1536

        # Update to mistral-embed
        provider.update_config(model="mistral-embed")

        assert provider.model == "mistral-embed"
        assert provider._model_config["default_dimension"] == 1024
        assert provider._model_config["max_dimension"] == 1024

    def test_update_config_validates_output_dimension(self):
        """Test that update_config validates output_dimension."""
        provider = MistralEmbeddingProvider(api_key="test-key", model="mistral-embed")

        # Valid dimension for mistral-embed (max 1024)
        provider.update_config(output_dimension=512)
        assert provider._output_dimension == 512

        # Invalid dimension (exceeds mistral-embed max of 1024)
        with pytest.raises(ValueError, match="output_dimension must be between"):
            provider.update_config(output_dimension=2048)

    def test_update_config_api_key_resets_client(self):
        """Test that updating api_key resets the client."""
        provider = MistralEmbeddingProvider(api_key="old-key")
        provider._client_initialized = True
        provider._client = MagicMock()

        provider.update_config(api_key="new-key")

        assert provider._api_key == "new-key"
        assert provider._client is None
        assert provider._client_initialized is False

    def test_update_config_other_values(self):
        """Test updating various config values."""
        provider = MistralEmbeddingProvider(api_key="test-key")

        provider.update_config(
            batch_size=64,
            timeout=120,
            retry_attempts=5,
            retry_delay=3.0,
            max_tokens=4096,
        )

        assert provider._batch_size == 64
        assert provider._timeout == 120
        assert provider._retry_attempts == 5
        assert provider._retry_delay == 3.0
        assert provider._max_tokens == 4096


class TestMistralProviderAvailability:
    """Tests for provider availability checks."""

    def test_is_available_with_api_key(self):
        """Test is_available returns True when API key is set."""
        provider = MistralEmbeddingProvider(api_key="test-key")
        assert provider.is_available() is True

    def test_is_available_without_api_key(self):
        """Test is_available returns False when API key is not set."""
        provider = MistralEmbeddingProvider(api_key=None)
        assert provider.is_available() is False


class TestMistralProviderModelInfo:
    """Tests for model information retrieval."""

    def test_get_model_info(self):
        """Test get_model_info returns correct information."""
        provider = MistralEmbeddingProvider(
            api_key="test-key",
            model="codestral-embed",
            output_dimension=1024,
        )

        info = provider.get_model_info()

        assert info["provider"] == "mistral"
        assert info["model"] == "codestral-embed"
        assert info["dimensions"] == 1024  # Uses output_dimension
        assert info["max_dimensions"] == 3072
        assert info["distance_metric"] == "cosine"
        assert info["batch_size"] == 32
        assert info["context_length"] == 8192
        assert "codestral-embed" in info["supported_models"]
        assert "mistral-embed" in info["supported_models"]
        assert info["output_dimension"] == 1024

    def test_get_model_token_limit(self):
        """Test get_model_token_limit returns correct value."""
        provider = MistralEmbeddingProvider(api_key="test-key")
        assert provider.get_model_token_limit() == 8192

    def test_get_max_tokens_per_batch(self):
        """Test get_max_tokens_per_batch returns correct value."""
        provider = MistralEmbeddingProvider(api_key="test-key")
        assert provider.get_max_tokens_per_batch() == 100000

    def test_get_max_documents_per_batch(self):
        """Test get_max_documents_per_batch returns batch_size."""
        provider = MistralEmbeddingProvider(api_key="test-key", batch_size=32)
        assert provider.get_max_documents_per_batch() == 32


class TestMistralProviderUsageStats:
    """Tests for usage statistics tracking."""

    def test_initial_usage_stats(self):
        """Test initial usage stats are zero."""
        provider = MistralEmbeddingProvider(api_key="test-key")
        stats = provider.get_usage_stats()

        assert stats["requests_made"] == 0
        assert stats["tokens_used"] == 0
        assert stats["embeddings_generated"] == 0

    def test_reset_usage_stats(self):
        """Test reset_usage_stats clears statistics."""
        provider = MistralEmbeddingProvider(api_key="test-key")
        provider._requests_made = 10
        provider._tokens_used = 1000
        provider._embeddings_generated = 50

        provider.reset_usage_stats()

        stats = provider.get_usage_stats()
        assert stats["requests_made"] == 0
        assert stats["tokens_used"] == 0
        assert stats["embeddings_generated"] == 0


class TestMistralProviderTextValidation:
    """Tests for text validation and preprocessing."""

    def test_validate_texts_empty_list(self):
        """Test validate_texts raises for empty list."""
        provider = MistralEmbeddingProvider(api_key="test-key")

        with pytest.raises(Exception):  # ValidationError
            provider.validate_texts([])

    def test_validate_texts_non_string(self):
        """Test validate_texts raises for non-string items."""
        provider = MistralEmbeddingProvider(api_key="test-key")

        with pytest.raises(Exception):  # ValidationError
            provider.validate_texts([123, "valid"])

    def test_validate_texts_empty_strings(self):
        """Test validate_texts handles empty strings with placeholder."""
        provider = MistralEmbeddingProvider(api_key="test-key")
        result = provider.validate_texts(["valid", "", "  "])

        assert result[0] == "valid"
        assert result[1] == "[EMPTY]"
        assert result[2] == "[EMPTY]"

    def test_validate_texts_strips_whitespace(self):
        """Test validate_texts strips leading/trailing whitespace."""
        provider = MistralEmbeddingProvider(api_key="test-key")
        result = provider.validate_texts(["  hello  ", "\tworld\n"])

        assert result[0] == "hello"
        assert result[1] == "world"


class TestMistralProviderTokenEstimation:
    """Tests for token estimation."""

    def test_estimate_tokens(self):
        """Test token estimation for single text."""
        provider = MistralEmbeddingProvider(api_key="test-key")

        # Token estimation is roughly chars / 4
        text = "This is a test text for token estimation"
        tokens = provider.estimate_tokens(text)

        assert tokens > 0
        assert isinstance(tokens, int)

    def test_estimate_batch_tokens(self):
        """Test token estimation for batch of texts."""
        provider = MistralEmbeddingProvider(api_key="test-key")

        texts = ["First text", "Second longer text", "Third"]
        total = provider.estimate_batch_tokens(texts)

        assert total > 0
        assert isinstance(total, int)

    def test_chunk_text_by_tokens(self):
        """Test chunking text by token count."""
        provider = MistralEmbeddingProvider(api_key="test-key")

        # Create a long text
        text = "word " * 1000
        chunks = provider.chunk_text_by_tokens(text, max_tokens=100)

        assert len(chunks) > 1
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_chunk_text_by_tokens_invalid_max(self):
        """Test chunk_text_by_tokens raises for invalid max_tokens."""
        provider = MistralEmbeddingProvider(api_key="test-key")

        with pytest.raises(Exception):  # ValidationError
            provider.chunk_text_by_tokens("text", max_tokens=0)


class TestMistralProviderReranking:
    """Tests for reranking support."""

    def test_supports_reranking_false(self):
        """Test that Mistral does not support reranking."""
        provider = MistralEmbeddingProvider(api_key="test-key")
        assert provider.supports_reranking() is False

    @pytest.mark.asyncio
    async def test_rerank_raises_not_implemented(self):
        """Test that rerank raises NotImplementedError."""
        provider = MistralEmbeddingProvider(api_key="test-key")

        with pytest.raises(NotImplementedError, match="does not currently support"):
            await provider.rerank("query", ["doc1", "doc2"])


class TestMistralProviderConcurrency:
    """Tests for concurrency settings."""

    def test_get_recommended_concurrency(self):
        """Test recommended concurrency is set."""
        provider = MistralEmbeddingProvider(api_key="test-key")
        assert provider.get_recommended_concurrency() == 20
        assert provider.RECOMMENDED_CONCURRENCY == 20

    def test_get_optimal_batch_size(self):
        """Test optimal batch size returns configured value."""
        provider = MistralEmbeddingProvider(api_key="test-key", batch_size=48)
        assert provider.get_optimal_batch_size() == 48


class TestMistralProviderDistanceMetrics:
    """Tests for distance metric support."""

    def test_get_supported_distances(self):
        """Test supported distance metrics."""
        provider = MistralEmbeddingProvider(api_key="test-key")
        distances = provider.get_supported_distances()

        assert "cosine" in distances
        assert "l2" in distances
        assert "ip" in distances


class TestMistralProviderRateLimits:
    """Tests for rate limit information."""

    def test_get_rate_limits(self):
        """Test rate limit info is provided."""
        provider = MistralEmbeddingProvider(api_key="test-key")
        limits = provider.get_rate_limits()

        assert "note" in limits
        assert "docs" in limits
        assert "mistral.ai" in limits["docs"]


class TestMistralProviderHeaders:
    """Tests for request headers."""

    def test_get_request_headers(self):
        """Test request headers are properly formed."""
        provider = MistralEmbeddingProvider(api_key="test-api-key")
        headers = provider.get_request_headers()

        assert headers["Authorization"] == "Bearer test-api-key"
        assert headers["Content-Type"] == "application/json"
        assert "ChunkHound" in headers["User-Agent"]


class TestMistralProviderEmbedding:
    """Tests for embedding generation with mocked API."""

    @pytest.mark.asyncio
    async def test_embed_empty_list(self):
        """Test embed returns empty list for empty input."""
        provider = MistralEmbeddingProvider(api_key="test-key")
        result = await provider.embed([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_batch_empty_list(self):
        """Test embed_batch returns empty list for empty input."""
        provider = MistralEmbeddingProvider(api_key="test-key")
        result = await provider.embed_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_with_mocked_client(self):
        """Test embed with mocked Mistral client."""
        provider = MistralEmbeddingProvider(api_key="test-key")

        # Create mock response
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1] * 1536

        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 10

        # Create mock client
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response

        provider._client = mock_client
        provider._client_initialized = True

        result = await provider.embed(["test text"])

        assert len(result) == 1
        assert len(result[0]) == 1536
        assert provider._requests_made == 1
        assert provider._tokens_used == 10
        assert provider._embeddings_generated == 1

    @pytest.mark.asyncio
    async def test_embed_uses_asyncio_to_thread(self):
        """Test that embed uses asyncio.to_thread to avoid blocking."""
        provider = MistralEmbeddingProvider(api_key="test-key")

        # Create mock response
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1] * 1536

        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock_response.usage = None

        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response

        provider._client = mock_client
        provider._client_initialized = True

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = mock_response

            result = await provider.embed(["test"])

            # Verify asyncio.to_thread was called
            mock_to_thread.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_single(self):
        """Test embed_single returns single embedding."""
        provider = MistralEmbeddingProvider(api_key="test-key")

        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.5] * 1536

        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock_response.usage = None

        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response

        provider._client = mock_client
        provider._client_initialized = True

        result = await provider.embed_single("test text")

        assert len(result) == 1536


class TestMistralProviderLifecycle:
    """Tests for provider lifecycle management."""

    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test provider initialization."""
        provider = MistralEmbeddingProvider(api_key="test-key")
        await provider.initialize()

        assert provider._client_initialized is True
        assert provider._client is not None

    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test provider shutdown cleans up client."""
        provider = MistralEmbeddingProvider(api_key="test-key")
        await provider.initialize()

        await provider.shutdown()

        assert provider._client is None
        assert provider._client_initialized is False


class TestMistralProviderHealthCheck:
    """Tests for health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_not_available(self):
        """Test health check when provider is not available."""
        provider = MistralEmbeddingProvider(api_key=None)
        result = await provider.health_check()

        assert result["available"] is False
        assert "API key not configured" in result["errors"]

    @pytest.mark.asyncio
    async def test_health_check_with_mocked_client(self):
        """Test health check with mocked successful API call."""
        provider = MistralEmbeddingProvider(api_key="test-key")

        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1] * 1536

        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock_response.usage = None

        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response

        provider._client = mock_client
        provider._client_initialized = True

        result = await provider.health_check()

        assert result["available"] is True
        assert result["provider"] == "mistral"
        assert result["connectivity"] == "ok"


class TestMistralModelConfig:
    """Tests for model configuration constants."""

    def test_codestral_embed_config(self):
        """Test codestral-embed model configuration."""
        config = MISTRAL_MODEL_CONFIG["codestral-embed"]

        assert config["max_tokens_per_batch"] == 100000
        assert config["max_texts_per_batch"] == 1000
        assert config["context_length"] == 8192
        assert config["default_dimension"] == 1536
        assert config["max_dimension"] == 3072

    def test_mistral_embed_config(self):
        """Test mistral-embed model configuration."""
        config = MISTRAL_MODEL_CONFIG["mistral-embed"]

        assert config["max_tokens_per_batch"] == 100000
        assert config["max_texts_per_batch"] == 1000
        assert config["context_length"] == 8192
        assert config["default_dimension"] == 1024
        assert config["max_dimension"] == 1024


class TestMistralProviderRetry:
    """Tests for retry behavior."""

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self):
        """Test that rate limit errors trigger retry."""
        provider = MistralEmbeddingProvider(
            api_key="test-key",
            retry_attempts=3,
            retry_delay=0.01,  # Fast for testing
        )

        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1] * 1536

        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock_response.usage = None

        # Fail twice with rate limit, then succeed
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = [
            Exception("Rate limit exceeded"),
            Exception("Rate limit error"),
            mock_response,
        ]

        provider._client = mock_client
        provider._client_initialized = True

        result = await provider.embed(["test"])

        assert len(result) == 1
        assert mock_client.embeddings.create.call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test that error is raised after max retries for rate limit errors."""
        provider = MistralEmbeddingProvider(
            api_key="test-key",
            retry_attempts=2,
            retry_delay=0.01,
        )

        mock_client = MagicMock()
        # Use rate limit error to trigger retry logic
        mock_client.embeddings.create.side_effect = Exception("Rate limit exceeded")

        provider._client = mock_client
        provider._client_initialized = True

        with pytest.raises(Exception):
            await provider.embed(["test"])

        # Should have retried twice (retry_attempts=2)
        assert mock_client.embeddings.create.call_count == 2

    @pytest.mark.asyncio
    async def test_non_retryable_error_fails_immediately(self):
        """Test that non-retryable errors fail immediately without retry."""
        provider = MistralEmbeddingProvider(
            api_key="test-key",
            retry_attempts=3,
            retry_delay=0.01,
        )

        mock_client = MagicMock()
        # Generic error (not rate limit or connection) should not retry
        mock_client.embeddings.create.side_effect = Exception("Invalid API key")

        provider._client = mock_client
        provider._client_initialized = True

        with pytest.raises(Exception, match="Invalid API key"):
            await provider.embed(["test"])

        # Should only call once since it's not a retryable error
        assert mock_client.embeddings.create.call_count == 1
