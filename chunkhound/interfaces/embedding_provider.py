"""EmbeddingProvider protocol for ChunkHound - abstract interface for embedding implementations."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class RerankResult:
    """Result from reranking operation."""

    index: int
    score: float


@dataclass
class EmbeddingConfig:
    """Configuration for embedding providers."""

    provider: str
    model: str
    dims: int
    distance: str = "cosine"
    batch_size: int = 100
    max_tokens: int | None = None
    api_key: str | None = None
    base_url: str | None = None
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0


class EmbeddingProvider(Protocol):
    """Abstract protocol for embedding providers.

    Defines the interface that all embedding implementations must follow.
    This enables pluggable embedding backends (OpenAI, local models, etc.)
    """

    @property
    def name(self) -> str:
        """Provider name (e.g., 'openai', 'local')."""
        ...

    @property
    def model(self) -> str:
        """Model name (e.g., 'text-embedding-3-small', 'sentence-transformers/all-MiniLM-L6-v2')."""
        ...

    @property
    def dims(self) -> int:
        """Embedding dimensions."""
        ...

    @property
    def distance(self) -> str:
        """Distance metric ('cosine' | 'l2' | 'ip')."""
        ...

    @property
    def batch_size(self) -> int:
        """Maximum batch size for embedding requests."""
        ...

    @property
    def max_tokens(self) -> int | None:
        """Maximum tokens per request (if applicable)."""
        ...

    @property
    def config(self) -> EmbeddingConfig:
        """Provider configuration."""
        ...

    # Core Embedding Operations
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (one per input text)

        Raises:
            EmbeddingError: If embedding generation fails
        """
        ...

    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text string to embed

        Returns:
            Embedding vector

        Raises:
            EmbeddingError: If embedding generation fails
        """
        ...

    async def embed_batch(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        """Generate embeddings in batches for optimal performance.

        Args:
            texts: List of text strings to embed
            batch_size: Optional batch size override

        Returns:
            List of embedding vectors (one per input text)

        Raises:
            EmbeddingError: If embedding generation fails
        """
        ...

    async def embed_streaming(self, texts: list[str]) -> AsyncIterator[list[float]]:
        """Generate embeddings with streaming results.

        Args:
            texts: List of text strings to embed

        Yields:
            Embedding vectors one at a time

        Raises:
            EmbeddingError: If embedding generation fails
        """
        ...

    # Provider Management
    async def initialize(self) -> None:
        """Initialize the embedding provider (load models, validate API keys, etc.)."""
        ...

    async def shutdown(self) -> None:
        """Shutdown the embedding provider and cleanup resources."""
        ...

    def is_available(self) -> bool:
        """Check if the provider is available and properly configured."""
        ...

    async def health_check(self) -> dict[str, Any]:
        """Perform health check and return status information."""
        ...

    # Validation and Preprocessing
    def validate_texts(self, texts: list[str]) -> list[str]:
        """Validate and preprocess texts before embedding.

        Args:
            texts: List of text strings to validate

        Returns:
            List of validated/preprocessed texts

        Raises:
            ValidationError: If texts are invalid
        """
        ...

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for a text (if applicable).

        Args:
            text: Text string to analyze

        Returns:
            Estimated token count
        """
        ...

    def chunk_text_by_tokens(self, text: str, max_tokens: int) -> list[str]:
        """Split text into chunks by token count (if applicable).

        Args:
            text: Text string to chunk
            max_tokens: Maximum tokens per chunk

        Returns:
            List of text chunks
        """
        ...

    # Metadata and Information
    def get_model_info(self) -> dict[str, Any]:
        """Get information about the embedding model."""
        ...

    def get_usage_stats(self) -> dict[str, Any]:
        """Get usage statistics (tokens used, requests made, etc.)."""
        ...

    def reset_usage_stats(self) -> None:
        """Reset usage statistics."""
        ...

    # Configuration Management
    def update_config(self, **kwargs: Any) -> None:
        """Update provider configuration.

        Args:
            **kwargs: Configuration parameters to update
        """
        ...

    def get_supported_distances(self) -> list[str]:
        """Get list of supported distance metrics."""
        ...

    def get_optimal_batch_size(self) -> int:
        """Get optimal batch size for this provider."""
        ...

    def get_max_tokens_per_batch(self) -> int:
        """Get maximum tokens per batch for this provider.

        Returns:
            Maximum number of tokens that can be processed in a single batch.
            Used by service layer for provider-agnostic token-aware batching.
        """
        ...

    def get_max_documents_per_batch(self) -> int:
        """Get maximum number of documents per batch for this provider.

        Returns:
            Maximum number of documents that can be processed in a single batch.
            Used by service layer for provider-agnostic document-count-aware batching.
        """
        ...

    def get_recommended_concurrency(self) -> int:
        """Get recommended number of concurrent batches for this provider.

        Returns:
            Optimal concurrent batch count based on provider's rate limits.
            Examples: VoyageAI=40 (2000 RPM), OpenAI=8 (tier-based), Ollama=16 (local GPU)
        """
        ...


class LocalEmbeddingProvider(EmbeddingProvider, Protocol):
    """Extended protocol for local embedding providers."""

    @property
    def model_path(self) -> str:
        """Path to the local model."""
        ...

    @property
    def device(self) -> str:
        """Device used for inference ('cpu', 'cuda', 'mps')."""
        ...

    def load_model(self) -> None:
        """Load the embedding model into memory."""
        ...

    def unload_model(self) -> None:
        """Unload the embedding model from memory."""
        ...

    def is_model_loaded(self) -> bool:
        """Check if the model is loaded in memory."""
        ...


class APIEmbeddingProvider(EmbeddingProvider, Protocol):
    """Extended protocol for API-based embedding providers."""

    @property
    def api_key(self) -> str | None:
        """API key for authentication."""
        ...

    @property
    def base_url(self) -> str:
        """Base URL for API requests."""
        ...

    @property
    def timeout(self) -> int:
        """Request timeout in seconds."""
        ...

    @property
    def retry_attempts(self) -> int:
        """Number of retry attempts for failed requests."""
        ...

    async def validate_api_key(self) -> bool:
        """Validate API key with the service."""
        ...

    def get_rate_limits(self) -> dict[str, Any]:
        """Get rate limit information."""
        ...

    def get_request_headers(self) -> dict[str, str]:
        """Get headers for API requests."""
        ...

    # Reranking Operations (Optional)
    def supports_reranking(self) -> bool:
        """Return True if this provider supports reranking."""
        return False

    async def rerank(
        self, query: str, documents: list[str], top_k: int | None = None
    ) -> list[RerankResult]:
        """Rerank documents by relevance to query.

        Only called if supports_reranking() returns True.

        Args:
            query: Query text to rank against
            documents: List of document texts to rank
            top_k: Optional limit on number of results

        Returns:
            List of RerankResult with original index and relevance score

        Raises:
            NotImplementedError: If provider doesn't support reranking
        """
        raise NotImplementedError("Reranking not supported by this provider")
