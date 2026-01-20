"""Mistral embedding provider implementation for ChunkHound - concrete embedding provider using Mistral API."""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import httpx
from loguru import logger

from chunkhound.core.config.embedding_config import validate_rerank_configuration
from chunkhound.core.constants import MISTRAL_DEFAULT_MODEL
from chunkhound.core.exceptions.core import ValidationError
from chunkhound.interfaces.embedding_provider import EmbeddingConfig

from .rerank_mixin import RerankMixin
from .shared_utils import (
    chunk_text_by_words,
    estimate_tokens_rough,
    get_dimensions_for_model,
    get_usage_stats_dict,
    validate_text_input,
)

try:
    from mistralai import Mistral

    MISTRAL_AVAILABLE = True
except ImportError:
    Mistral = None  # type: ignore
    MISTRAL_AVAILABLE = False
    logger.warning("Mistral SDK not available - install with: uv pip install mistralai")


# Official Mistral embedding model configuration based on API documentation
# https://docs.mistral.ai/models/codestral-embed-25-05
MISTRAL_MODEL_CONFIG = {
    "codestral-embed": {
        "max_tokens_per_batch": 100000,  # Conservative estimate based on 8K context
        "max_texts_per_batch": 1000,
        "context_length": 8192,
        "default_dimension": 1536,
        "max_dimension": 3072,
        "price_per_million": 0.15,  # USD
    },
    "mistral-embed": {
        # Older general-purpose embedding model
        "max_tokens_per_batch": 100000,
        "max_texts_per_batch": 1000,
        "context_length": 8192,
        "default_dimension": 1024,
        "max_dimension": 1024,
        "price_per_million": 0.10,  # USD
    },
}


class MistralEmbeddingProvider(RerankMixin):
    """Mistral embedding provider using codestral-embed by default.

    Supports Mistral's embedding models including:
    - codestral-embed: State-of-the-art code embedding model (1536 dims, up to 3072)
    - mistral-embed: General-purpose embedding model (1024 dims)

    Thread Safety:
        This provider is thread-safe and stateless. Multiple concurrent calls to
        embed() are safe. The underlying Mistral client handles concurrent requests
        properly.
    """

    # Recommended concurrent batches for Mistral API
    # Conservative value based on typical API rate limits
    RECOMMENDED_CONCURRENCY = 20

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = MISTRAL_DEFAULT_MODEL,
        rerank_model: str | None = None,
        rerank_url: str = "/rerank",
        rerank_format: str = "auto",
        batch_size: int = 32,
        timeout: int = 30,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        max_tokens: int | None = None,
        output_dimension: int | None = None,
        rerank_batch_size: int | None = None,
    ):
        """Initialize Mistral embedding provider.

        Args:
            api_key: Mistral API key (defaults to MISTRAL_API_KEY env var)
            base_url: Base URL for Mistral API (defaults to https://api.mistral.ai)
            model: Model name to use for embeddings
            rerank_model: Model name to use for reranking (enables multi-hop search)
            rerank_url: Rerank endpoint URL (defaults to /rerank)
            rerank_format: Reranking API format - 'cohere', 'tei', or 'auto' (default: 'auto')
            batch_size: Maximum batch size for API requests
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts for failed requests
            retry_delay: Delay between retry attempts
            max_tokens: Maximum tokens per request (if applicable)
            output_dimension: Output embedding dimension (1-3072 for codestral-embed)
            rerank_batch_size: Max documents per rerank batch (overrides model defaults, bounded by model caps)
        """
        if not MISTRAL_AVAILABLE:
            raise ImportError(
                "Mistral SDK not available - install with: uv pip install mistralai"
            )

        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._rerank_model = rerank_model
        self._rerank_url = rerank_url
        self._rerank_format = rerank_format
        self._detected_rerank_format: str | None = (
            None  # Cache for auto-detected format
        )
        self._format_detection_lock = asyncio.Lock()  # Protect format detection cache
        self._output_dimension = output_dimension
        self._rerank_batch_size = rerank_batch_size

        # Validate rerank configuration at initialization (fail-fast)
        is_using_reranking = rerank_model or (rerank_format == "tei" and rerank_url)
        if is_using_reranking or rerank_format == "cohere":
            validate_rerank_configuration(
                provider="mistral",
                rerank_format=rerank_format,
                rerank_model=rerank_model,
                rerank_url=rerank_url,
                base_url=base_url,
            )

            # Warn about auto-detection risks in production
            if rerank_format == "auto":
                logger.warning(
                    "Using rerank_format='auto' may cause first request to fail if format guess is wrong. "
                    "For production use, explicitly set rerank_format to 'cohere' or 'tei'."
                )

        # Get model configuration or use defaults
        model_config = MISTRAL_MODEL_CONFIG.get(
            model,
            {
                "max_tokens_per_batch": 100000,
                "max_texts_per_batch": 1000,
                "context_length": 8192,
                "default_dimension": 1536,
                "max_dimension": 3072,
            },
        )

        self._batch_size = min(batch_size, int(model_config["max_texts_per_batch"]))
        self._timeout = timeout
        self._retry_attempts = retry_attempts
        self._retry_delay = retry_delay
        self._max_tokens = max_tokens or int(model_config["context_length"])
        self._model_config = model_config

        # Validate output_dimension if provided
        if output_dimension is not None:
            max_dim = model_config.get("max_dimension", 3072)
            if output_dimension < 1 or output_dimension > max_dim:
                raise ValueError(
                    f"output_dimension must be between 1 and {max_dim} for {model}, "
                    f"got {output_dimension}"
                )

        # Initialize client lazily to handle async context properly
        self._client: Mistral | None = None
        self._client_initialized = False

        # Model dimension mapping (explicitly int values)
        self._dimensions_map: dict[str, int] = {
            model_name: int(config["default_dimension"])
            for model_name, config in MISTRAL_MODEL_CONFIG.items()
        }

        # Usage tracking
        self._requests_made = 0
        self._tokens_used = 0
        self._embeddings_generated = 0

    def _ensure_client(self) -> None:
        """Ensure the Mistral client is initialized."""
        if self._client is not None and self._client_initialized:
            return

        if not MISTRAL_AVAILABLE or Mistral is None:
            raise RuntimeError(
                "Mistral SDK is not available. Install with: uv pip install mistralai"
            )

        if not self._api_key:
            raise ValueError(
                "Mistral API key is required. "
                "Set MISTRAL_API_KEY environment variable or pass api_key parameter."
            )

        self._client = Mistral(api_key=self._api_key)
        self._client_initialized = True

    @property
    def name(self) -> str:
        """Provider name."""
        return "mistral"

    @property
    def model(self) -> str:
        """Model name."""
        return self._model

    @property
    def dims(self) -> int:
        """Embedding dimensions."""
        # If output_dimension is set, use that
        if self._output_dimension is not None:
            return self._output_dimension
        return get_dimensions_for_model(
            self._model, self._dimensions_map, default_dims=1536
        )

    @property
    def distance(self) -> str:
        """Distance metric (Mistral uses cosine)."""
        return "cosine"

    @property
    def batch_size(self) -> int:
        """Maximum batch size for embedding requests."""
        return self._batch_size

    @property
    def max_tokens(self) -> int | None:
        """Maximum tokens per request."""
        return self._max_tokens

    @property
    def config(self) -> EmbeddingConfig:
        """Provider configuration."""
        return EmbeddingConfig(
            provider=self.name,
            model=self.model,
            dims=self.dims,
            distance=self.distance,
            batch_size=self.batch_size,
            max_tokens=self.max_tokens,
            api_key=self._api_key,
            base_url="https://api.mistral.ai",
            timeout=self._timeout,
            retry_attempts=self._retry_attempts,
            retry_delay=self._retry_delay,
        )

    @property
    def api_key(self) -> str | None:
        """API key for authentication."""
        return self._api_key

    @property
    def base_url(self) -> str:
        """Base URL for API requests."""
        return self._base_url or "https://api.mistral.ai"

    @property
    def timeout(self) -> int:
        """Request timeout in seconds."""
        return self._timeout

    @property
    def retry_attempts(self) -> int:
        """Number of retry attempts for failed requests."""
        return self._retry_attempts

    async def initialize(self) -> None:
        """Initialize the embedding provider."""
        self._ensure_client()
        logger.info(f"Mistral embedding provider initialized with model: {self._model}")

    async def shutdown(self) -> None:
        """Shutdown the embedding provider and cleanup resources."""
        self._client = None
        self._client_initialized = False
        logger.info("Mistral embedding provider shutdown")

    def is_available(self) -> bool:
        """Check if the provider is available and properly configured."""
        if not MISTRAL_AVAILABLE:
            return False
        return self._api_key is not None

    async def health_check(self) -> dict[str, Any]:
        """Perform health check and return status information."""
        errors: list[str] = []
        status: dict[str, Any] = {
            "provider": self.name,
            "model": self.model,
            "available": self.is_available(),
            "api_key_configured": self._api_key is not None,
            "client_initialized": self._client_initialized,
            "errors": errors,
        }

        if not self.is_available():
            if not MISTRAL_AVAILABLE:
                errors.append("Mistral SDK not installed")
            if not self._api_key:
                errors.append("API key not configured")
            return status

        try:
            # Test API connectivity with a small embedding
            test_embedding = await self.embed_single("test")
            if len(test_embedding) == self.dims:
                status["connectivity"] = "ok"
            else:
                errors.append(
                    f"Unexpected embedding dimensions: {len(test_embedding)} != {self.dims}"
                )
        except Exception as e:
            errors.append(f"API connectivity test failed: {str(e)}")

        return status

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        if not texts:
            return []

        validated_texts = validate_text_input(texts)
        if not validated_texts:
            return []

        try:
            return await self.embed_batch(validated_texts)
        except Exception as e:
            logger.error(f"[Mistral-Provider] Failed to generate embeddings: {e}")
            raise

    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        embeddings = await self.embed([text])
        return embeddings[0] if embeddings else []

    async def embed_batch(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        """Generate embeddings in batches for optimal performance."""
        if not texts:
            return []

        effective_batch_size = batch_size or self._batch_size
        all_embeddings: list[list[float]] = []

        # Process in batches
        for i in range(0, len(texts), effective_batch_size):
            batch = texts[i : i + effective_batch_size]
            batch_embeddings = await self._embed_batch_internal(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def _embed_batch_internal(self, texts: list[str]) -> list[list[float]]:
        """Internal method to embed a batch of texts."""
        self._ensure_client()
        if not self._client:
            raise RuntimeError("Mistral client not initialized")

        for attempt in range(self._retry_attempts):
            try:
                logger.debug(
                    f"Generating embeddings for {len(texts)} texts (attempt {attempt + 1})"
                )

                # Build request parameters
                # Mistral API uses 'inputs' (plural) for the text field
                kwargs: dict[str, Any] = {
                    "model": self._model,
                    "inputs": texts,
                }

                # Add output_dimension if specified (Matryoshka embeddings)
                if self._output_dimension is not None:
                    kwargs["output_dimension"] = self._output_dimension

                # Call Mistral embeddings API
                response = self._client.embeddings.create(**kwargs)

                # Extract embeddings from response
                embeddings: list[list[float]] = []
                for data in response.data:
                    if data.embedding is not None:
                        embeddings.append(list(data.embedding))

                # Update usage statistics
                self._requests_made += 1
                self._embeddings_generated += len(embeddings)
                if hasattr(response, "usage") and response.usage:
                    total_tokens = getattr(response.usage, "total_tokens", None)
                    if total_tokens is not None:
                        self._tokens_used += total_tokens

                logger.debug(f"Successfully generated {len(embeddings)} embeddings")
                return embeddings

            except Exception as e:
                error_message = str(e).lower()

                # Check for rate limit errors
                if "rate" in error_message and "limit" in error_message:
                    if attempt < self._retry_attempts - 1:
                        delay = self._retry_delay * (2**attempt)
                        logger.warning(
                            f"Rate limit exceeded, retrying in {delay}s (attempt {attempt + 1})"
                        )
                        await asyncio.sleep(delay)
                        continue

                # Check for timeout/connection errors
                if "timeout" in error_message or "connection" in error_message:
                    if attempt < self._retry_attempts - 1:
                        logger.warning(
                            f"Connection error, retrying in {self._retry_delay}s: {e}"
                        )
                        await asyncio.sleep(self._retry_delay)
                        continue

                # Non-retryable error or last attempt
                logger.error(f"Mistral API error: {e}")
                raise

        raise RuntimeError(
            f"Failed to generate embeddings after {self._retry_attempts} attempts"
        )

    async def embed_streaming(self, texts: list[str]) -> AsyncIterator[list[float]]:
        """Generate embeddings with streaming results."""
        for text in texts:
            embedding = await self.embed_single(text)
            yield embedding

    def validate_texts(self, texts: list[str]) -> list[str]:
        """Validate and preprocess texts before embedding."""
        if not texts:
            raise ValidationError("texts", texts, "No texts provided for embedding")

        validated = []
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValidationError(
                    f"texts[{i}]",
                    text,
                    f"Text at index {i} is not a string: {type(text)}",
                )

            if not text.strip():
                logger.warning(f"Empty text at index {i}, using placeholder")
                validated.append("[EMPTY]")
            else:
                validated.append(text.strip())

        return validated

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for a text."""
        return int(estimate_tokens_rough(text))

    def estimate_batch_tokens(self, texts: list[str]) -> int:
        """Estimate total token count for a batch of texts."""
        return sum(self.estimate_tokens(text) for text in texts)

    def get_model_token_limit(self) -> int:
        """Get token limit for current model."""
        return int(self._model_config.get("context_length", 8192))

    def chunk_text_by_tokens(self, text: str, max_tokens: int) -> list[str]:
        """Split text into chunks by token count."""
        if max_tokens <= 0:
            raise ValidationError(
                "max_tokens", max_tokens, "max_tokens must be positive"
            )
        return chunk_text_by_words(text, max_tokens)

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the embedding model."""
        return {
            "provider": self.name,
            "model": self.model,
            "dimensions": self.dims,
            "max_dimensions": self._model_config.get("max_dimension", 3072),
            "distance_metric": self.distance,
            "batch_size": self.batch_size,
            "max_tokens": self.max_tokens,
            "context_length": self._model_config.get("context_length", 8192),
            "supported_models": list(MISTRAL_MODEL_CONFIG.keys()),
            "output_dimension": self._output_dimension,
        }

    def get_usage_stats(self) -> dict[str, Any]:
        """Get usage statistics."""
        return get_usage_stats_dict(
            self._requests_made, self._tokens_used, self._embeddings_generated
        )

    def reset_usage_stats(self) -> None:
        """Reset usage statistics."""
        self._requests_made = 0
        self._tokens_used = 0
        self._embeddings_generated = 0

    def update_config(self, **kwargs: Any) -> None:
        """Update provider configuration."""
        if "model" in kwargs:
            self._model = kwargs["model"]
        if "batch_size" in kwargs:
            self._batch_size = kwargs["batch_size"]
        if "timeout" in kwargs:
            self._timeout = kwargs["timeout"]
        if "retry_attempts" in kwargs:
            self._retry_attempts = kwargs["retry_attempts"]
        if "retry_delay" in kwargs:
            self._retry_delay = kwargs["retry_delay"]
        if "max_tokens" in kwargs:
            self._max_tokens = kwargs["max_tokens"]
        if "output_dimension" in kwargs:
            self._output_dimension = kwargs["output_dimension"]
        if "api_key" in kwargs:
            self._api_key = kwargs["api_key"]
            # Reset client to force re-initialization with new API key
            self._client = None
            self._client_initialized = False

    def get_supported_distances(self) -> list[str]:
        """Get list of supported distance metrics."""
        return ["cosine", "l2", "ip"]

    def get_optimal_batch_size(self) -> int:
        """Get optimal batch size for this provider."""
        return self._batch_size

    def get_max_tokens_per_batch(self) -> int:
        """Get maximum tokens per batch for this provider."""
        return int(self._model_config.get("max_tokens_per_batch", 100000))

    def get_max_documents_per_batch(self) -> int:
        """Get maximum documents per batch for this provider."""
        return self._batch_size

    def get_max_rerank_batch_size(self) -> int:
        """Get maximum documents per batch for reranking operations.

        Returns model-specific batch limit for reranking to prevent OOM errors.
        Implements bounded override pattern: user can set batch size, but it's
        clamped to model caps for safety.

        Priority order:
        0. User override (rerank_batch_size) - bounded by model cap below
        1. Conservative default (128 for Mistral)

        Returns:
            Maximum number of documents to rerank in a single batch
        """
        # Conservative default for Mistral reranking
        # Research shows 32-128 is optimal for GPU reranking
        model_cap = 128

        # Priority 0: User override (bounded by model cap)
        if self._rerank_batch_size is not None:
            return min(self._rerank_batch_size, model_cap)

        # Return model cap as default
        return model_cap

    def get_recommended_concurrency(self) -> int:
        """Get recommended number of concurrent batches for Mistral.

        Returns:
            Conservative concurrency for typical API rate limits
        """
        return self.RECOMMENDED_CONCURRENCY

    # RerankMixin hook implementation
    def _get_rerank_client_kwargs(self) -> dict[str, Any]:
        """Get httpx client kwargs for rerank requests.

        Mistral uses standard httpx configuration with timeout.

        Returns:
            Dictionary of kwargs to pass to httpx.AsyncClient
        """
        return {"timeout": self._timeout}

    async def validate_api_key(self) -> bool:
        """Validate API key with the service."""
        if not self._api_key:
            return False

        try:
            # Test with a minimal request
            self._ensure_client()
            await self.embed_single("test")
            return True
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return False

    def get_rate_limits(self) -> dict[str, Any]:
        """Get rate limit information."""
        return {
            "note": "See Mistral documentation for current rate limits",
            "docs": "https://docs.mistral.ai/api/",
        }

    def get_request_headers(self) -> dict[str, str]:
        """Get headers for API requests."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": "ChunkHound-Mistral-Provider",
        }
