"""Embedding providers for ChunkHound - pluggable vector embedding generation."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from loguru import logger

from chunkhound.interfaces.embedding_provider import (
    EmbeddingProvider as InterfaceEmbeddingProvider,
    RerankResult,
)

if TYPE_CHECKING:
    from chunkhound.providers.embeddings.openai_provider import OpenAIEmbeddingProvider

# Core domain models

# OpenAI and tiktoken imports have been moved to the specific provider implementations
# that need them. This reduces unnecessary dependencies in the core module.


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    @property
    def name(self) -> str:
        """Provider name (e.g., 'openai')."""
        ...

    @property
    def model(self) -> str:
        """Model name (e.g., 'text-embedding-3-small')."""
        ...

    @property
    def dims(self) -> int:
        """Embedding dimensions."""
        ...

    @property
    def distance(self) -> str:
        """Distance metric ('cosine' | 'l2')."""
        ...

    @property
    def batch_size(self) -> int:
        """Maximum batch size for embedding requests."""
        ...

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (one per input text)
        """
        ...

    def supports_reranking(self) -> bool:
        """Return True if this provider supports reranking."""
        ...

    async def rerank(
        self, query: str, documents: list[str], top_k: int | None = None
    ) -> list[RerankResult]:
        """Rerank documents by relevance to query.

        Args:
            query: Query text to rank against
            documents: List of document texts to rank
            top_k: Optional limit on number of results

        Returns:
            List of RerankResult with original index and relevance score
        """
        ...


@dataclass
class LocalEmbeddingResult:
    """Local result from embedding operation (legacy)."""

    embeddings: list[list[float]]
    model: str
    provider: str
    dims: int
    total_tokens: int | None = None


# OpenAIEmbeddingProvider has been moved to chunkhound.providers.embeddings.openai_provider
# The old implementation has been removed to avoid duplication and confusion.
# Use create_openai_provider() function below which imports from the new location.


class EmbeddingManager:
    """Manages embedding providers and generation."""

    def __init__(self) -> None:
        self._providers: dict[str, InterfaceEmbeddingProvider] = {}
        self._default_provider: str | None = None

    def register_provider(
        self, provider: InterfaceEmbeddingProvider, set_default: bool = False
    ) -> None:
        """Register an embedding provider.

        Args:
            provider: The embedding provider to register
            set_default: Whether to set this as the default provider
        """
        self._providers[provider.name] = provider
        logger.info(
            f"Registered embedding provider: {provider.name} (model: {provider.model})"
        )

        if set_default or self._default_provider is None:
            self._default_provider = provider.name
            logger.info(f"Set default embedding provider: {provider.name}")

    def get_provider(self, name: str | None = None) -> InterfaceEmbeddingProvider:
        """Get an embedding provider by name.

        Args:
            name: Provider name (uses default if None)

        Returns:
            The requested embedding provider
        """
        if name is None:
            if self._default_provider is None:
                raise ValueError("No default embedding provider set")
            name = self._default_provider

        if name not in self._providers:
            raise ValueError(f"Unknown embedding provider: {name}")

        return self._providers[name]

    def get_default_provider(self) -> InterfaceEmbeddingProvider | None:
        """Get the default embedding provider if one is set.

        Returns:
            The default embedding provider, or None if no default is set
        """
        if self._default_provider is None:
            return None
        return self._providers.get(self._default_provider)

    def list_providers(self) -> list[str]:
        """List all registered provider names."""
        return list(self._providers.keys())

    async def embed_texts(
        self,
        texts: list[str],
        provider_name: str | None = None,
    ) -> LocalEmbeddingResult:
        """Generate embeddings for texts using specified provider.

        Args:
            texts: List of texts to embed
            provider_name: Provider to use (uses default if None)

        Returns:
            Embedding result with vectors and metadata
        """
        provider = self.get_provider(provider_name)

        embeddings = await provider.embed(texts)

        return LocalEmbeddingResult(
            embeddings=embeddings,
            model=provider.model,
            provider=provider.name,
            dims=provider.dims,
        )


def create_openai_provider(
    api_key: str | None = None,
    base_url: str | None = None,
    model: str = "text-embedding-3-small",
    rerank_model: str | None = None,
    rerank_url: str = "/rerank",
    rerank_format: str = "auto",
    rerank_batch_size: int | None = None,
) -> "OpenAIEmbeddingProvider":
    """Create an OpenAI embedding provider with default settings.

    Args:
        api_key: OpenAI API key (uses OPENAI_API_KEY env var if None)
        base_url: Base URL for API (uses OPENAI_BASE_URL env var if None)
        model: Model name to use
        rerank_model: Model name to use for reranking (enables multi-hop search)
        rerank_url: Rerank endpoint URL (defaults to /rerank)
        rerank_format: Reranking API format - 'cohere', 'tei', or 'auto' (default: 'auto')
        rerank_batch_size: Max documents per rerank batch (overrides model defaults, bounded by model caps)

    Returns:
        Configured OpenAI embedding provider
    """
    # Import the new provider from the correct location
    from chunkhound.providers.embeddings.openai_provider import OpenAIEmbeddingProvider

    return OpenAIEmbeddingProvider(
        api_key=api_key,
        base_url=base_url,
        model=model,
        rerank_model=rerank_model,
        rerank_url=rerank_url,
        rerank_format=rerank_format,
        rerank_batch_size=rerank_batch_size,
    )
