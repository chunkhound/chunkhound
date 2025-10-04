"""VoyageAI embedding provider implementation for ChunkHound - concrete embedding provider using VoyageAI API."""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from loguru import logger

from chunkhound.core.constants import VOYAGE_DEFAULT_MODEL, VOYAGE_DEFAULT_RERANK_MODEL
from chunkhound.interfaces.embedding_provider import EmbeddingConfig, RerankResult

from .shared_utils import (
    chunk_text_by_words,
    estimate_tokens_rough,
    get_dimensions_for_model,
    get_usage_stats_dict,
    validate_text_input,
)

try:
    import voyageai

    VOYAGEAI_AVAILABLE = True
except ImportError:
    voyageai = None  # type: ignore
    VOYAGEAI_AVAILABLE = False
    logger.warning("VoyageAI not available - install with: uv pip install voyageai")


# Official VoyageAI model configuration based on API documentation
VOYAGE_MODEL_CONFIG = {
    # Models with 120,000 token limit per batch
    "voyage-3-large": {
        "max_tokens_per_batch": 120000,
        "max_texts_per_batch": 1000,
        "context_length": 32000,
        "dimensions": [256, 512, 1024, 2048],
        "default_dimension": 1024,
    },
    "voyage-code-3": {
        "max_tokens_per_batch": 120000,
        "max_texts_per_batch": 1000,
        "context_length": 32000,
        "dimensions": [256, 512, 1024, 2048],
        "default_dimension": 1024,
    },
    "voyage-finance-2": {
        "max_tokens_per_batch": 120000,
        "max_texts_per_batch": 1000,
        "context_length": 32000,
        "dimensions": [1024],
        "default_dimension": 1024,
    },
    "voyage-law-2": {
        "max_tokens_per_batch": 120000,
        "max_texts_per_batch": 1000,
        "context_length": 16000,
        "dimensions": [1024],
        "default_dimension": 1024,
    },
    "voyage-multilingual-2": {
        "max_tokens_per_batch": 120000,
        "max_texts_per_batch": 1000,
        "context_length": 32000,
        "dimensions": [1024],
        "default_dimension": 1024,
    },
    "voyage-large-2-instruct": {
        "max_tokens_per_batch": 120000,
        "max_texts_per_batch": 1000,
        "context_length": 16000,
        "dimensions": [1024],
        "default_dimension": 1024,
    },
    # Models with 320,000 token limit per batch
    "voyage-3.5": {
        "max_tokens_per_batch": 320000,
        "max_texts_per_batch": 1000,
        "context_length": 32000,
        "dimensions": [256, 512, 1024, 2048],
        "default_dimension": 1024,
    },
    "voyage-2": {
        "max_tokens_per_batch": 320000,
        "max_texts_per_batch": 1000,
        "context_length": 4000,
        "dimensions": [1024],
        "default_dimension": 1024,
    },
    # Model with 1,000,000 token limit per batch
    "voyage-3.5-lite": {
        "max_tokens_per_batch": 1000000,
        "max_texts_per_batch": 1000,
        "context_length": 32000,
        "dimensions": [256, 512, 1024, 2048],
        "default_dimension": 1024,
    },
}


class VoyageAIEmbeddingProvider:
    """VoyageAI embedding provider using voyage-3.5 by default."""

    # Recommended concurrent batches for VoyageAI API
    # Aggressive value (40) leverages VoyageAI's high rate limits:
    # - 2000 RPM (requests per minute) for paid accounts
    # - 1M+ TPM (tokens per minute) for voyage-3.5-lite
    # With ~50ms per request and large batch sizes, 40 concurrent
    # batches saturate the API without hitting rate limits
    RECOMMENDED_CONCURRENCY = 40

    def __init__(
        self,
        api_key: str | None = None,
        model: str = VOYAGE_DEFAULT_MODEL,
        rerank_model: str | None = VOYAGE_DEFAULT_RERANK_MODEL,
        batch_size: int = 100,
        timeout: int = 30,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        max_tokens: int | None = None,
    ):
        """Initialize VoyageAI embedding provider.

        Args:
            api_key: VoyageAI API key (defaults to VOYAGE_API_KEY env var)
            model: Model name to use for embeddings
            rerank_model: Model name to use for reranking
            batch_size: Maximum batch size for API requests
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts for failed requests
            retry_delay: Delay between retry attempts
            max_tokens: Maximum tokens per request (if applicable)
        """
        if not VOYAGEAI_AVAILABLE:
            raise ImportError(
                "VoyageAI not available - install with: uv pip install voyageai"
            )

        self._model = model
        self._rerank_model = rerank_model

        # Get model configuration or use defaults
        model_config = VOYAGE_MODEL_CONFIG.get(
            model,
            {
                "max_tokens_per_batch": 320000,  # Default for unknown models
                "max_texts_per_batch": 1000,
                "context_length": 32000,
                "dimensions": [1024],
                "default_dimension": 1024,
            },
        )

        self._batch_size = min(batch_size, model_config["max_texts_per_batch"])
        self._timeout = timeout
        self._retry_attempts = retry_attempts
        self._retry_delay = retry_delay
        self._max_tokens = max_tokens or model_config["context_length"]
        self._api_key = api_key
        self._model_config = model_config

        # Initialize client
        self._client = voyageai.Client(api_key=api_key)

        # Model dimension mapping - built from configuration
        self._dimensions_map = {
            model_name: config["default_dimension"]
            for model_name, config in VOYAGE_MODEL_CONFIG.items()
        }

        # Usage tracking
        self._requests_made = 0
        self._tokens_used = 0
        self._embeddings_generated = 0

    @property
    def name(self) -> str:
        """Provider name."""
        return "voyageai"

    @property
    def model(self) -> str:
        """Model name."""
        return self._model

    @property
    def dims(self) -> int:
        """Embedding dimensions."""
        return get_dimensions_for_model(
            self._model, self._dimensions_map, default_dims=1024
        )

    @property
    def distance(self) -> str:
        """Distance metric (VoyageAI uses cosine)."""
        return "cosine"

    @property
    def batch_size(self) -> int:
        """Maximum batch size."""
        return self._batch_size

    @property
    def max_tokens(self) -> int:
        """Maximum tokens per request."""
        return self._max_tokens

    @property
    def config(self) -> EmbeddingConfig:
        """Provider configuration."""
        return EmbeddingConfig(
            provider="voyageai",
            model=self._model,
            dims=self.dims,
            distance=self.distance,
            batch_size=self._batch_size,
            max_tokens=self._max_tokens,
            api_key=self._api_key,
            timeout=self._timeout,
            retry_attempts=self._retry_attempts,
            retry_delay=self._retry_delay,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        if not texts:
            return []

        validated_texts = validate_text_input(texts)

        try:
            result = await asyncio.to_thread(
                self._client.embed,
                texts=validated_texts,
                model=self._model,
                input_type="document",
                truncation=True,
            )

            self._requests_made += 1
            self._tokens_used += result.total_tokens
            self._embeddings_generated += len(texts)

            return [embedding for embedding in result.embeddings]

        except Exception as e:
            logger.error(f"VoyageAI embedding failed: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}") from e

    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        embeddings = await self.embed([text])
        return embeddings[0]

    async def embed_batch(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        """Generate embeddings in batches for optimal performance."""
        if not texts:
            return []

        batch_size = batch_size or self._batch_size
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

        all_embeddings = []
        for batch in batches:
            embeddings = await self.embed(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    async def embed_streaming(self, texts: list[str]) -> AsyncIterator[list[float]]:
        """Generate embeddings with streaming results."""
        for text in texts:
            embedding = await self.embed_single(text)
            yield embedding

    async def initialize(self) -> None:
        """Initialize the embedding provider."""
        # Test API connection
        try:
            await self.embed_single("test")
            logger.info(f"VoyageAI provider initialized with model: {self._model}")
        except Exception as e:
            logger.error(f"VoyageAI provider initialization failed: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the embedding provider and cleanup resources."""
        logger.info("VoyageAI provider shutdown complete")

    def is_available(self) -> bool:
        """Check if the provider is available and properly configured."""
        return VOYAGEAI_AVAILABLE and self._api_key is not None

    async def health_check(self) -> dict[str, Any]:
        """Perform health check and return status information."""
        try:
            await self.embed_single("health check")
            return {
                "status": "healthy",
                "provider": "voyageai",
                "model": self._model,
                "rerank_model": self._rerank_model,
                "dimensions": self.dims,
                "requests_made": self._requests_made,
                "tokens_used": self._tokens_used,
                "embeddings_generated": self._embeddings_generated,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "voyageai",
                "error": str(e),
            }

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for a text (rough approximation)."""
        return int(estimate_tokens_rough(text))

    def chunk_text_by_tokens(self, text: str, max_tokens: int) -> list[str]:
        """Split text into chunks by token count."""
        return chunk_text_by_words(text, max_tokens)

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the embedding model."""
        return {
            "provider": "voyageai",
            "model": self._model,
            "rerank_model": self._rerank_model,
            "dimensions": self.dims,
            "max_tokens": self._max_tokens,
            "supports_reranking": True,
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
        if "rerank_model" in kwargs:
            self._rerank_model = kwargs["rerank_model"]
        if "batch_size" in kwargs:
            self._batch_size = kwargs["batch_size"]
        if "timeout" in kwargs:
            self._timeout = kwargs["timeout"]

    def get_supported_distances(self) -> list[str]:
        """Get list of supported distance metrics."""
        return ["cosine"]  # VoyageAI uses cosine similarity

    def get_optimal_batch_size(self) -> int:
        """Get optimal batch size for this provider."""
        return min(self._batch_size, 100)  # 100 is generally optimal for performance

    def get_max_tokens_per_batch(self) -> int:
        """Get maximum tokens per batch for this provider."""
        return self._model_config["max_tokens_per_batch"]

    def get_max_documents_per_batch(self) -> int:
        """Get maximum documents per batch for VoyageAI provider."""
        return self._model_config["max_texts_per_batch"]

    def get_recommended_concurrency(self) -> int:
        """Get recommended number of concurrent batches for VoyageAI.

        Returns:
            Aggressive concurrency for VoyageAI's high rate limits
        """
        return self.RECOMMENDED_CONCURRENCY

    def get_chars_to_tokens_ratio(self) -> float:
        """Get character-to-token ratio for VoyageAI.

        Based on measured data: 325,138 tokens for 975,414 chars = 3.0 chars/token
        """
        return 3.0

    # Reranking Operations
    def supports_reranking(self) -> bool:
        """VoyageAI provider supports reranking."""
        return True

    async def rerank(
        self, query: str, documents: list[str], top_k: int | None = None
    ) -> list[RerankResult]:
        """Rerank documents by relevance to query using VoyageAI reranker."""
        if not documents:
            return []

        try:
            logger.debug(
                f"VoyageAI reranking {len(documents)} documents with model {self._rerank_model}"
            )

            result = await asyncio.to_thread(
                self._client.rerank,
                query=query,
                documents=documents,
                model=self._rerank_model,
                top_k=top_k,
            )

            self._requests_made += 1

            # Check if we got valid results
            if not hasattr(result, "results") or not result.results:
                logger.warning(
                    f"VoyageAI rerank returned no results for query: {query[:100]}"
                )
                return []

            rerank_results = []
            for item in result.results:
                if hasattr(item, "index") and hasattr(item, "relevance_score"):
                    rerank_results.append(
                        RerankResult(index=item.index, score=item.relevance_score)
                    )
                else:
                    logger.warning(f"Skipping invalid rerank result: {item}")

            logger.debug(
                f"VoyageAI reranked {len(documents)} documents, got {len(rerank_results)} results"
            )
            return rerank_results

        except AttributeError as e:
            logger.error(f"VoyageAI rerank response format error: {e}")
            raise ValueError(f"Invalid rerank response format: {e}") from e
        except Exception as e:
            logger.error(f"VoyageAI reranking failed: {e}")
            raise RuntimeError(f"Reranking failed: {e}") from e
