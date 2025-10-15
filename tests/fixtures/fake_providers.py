"""Fake provider implementations for testing code research without API calls.

These providers return deterministic, predictable responses for testing
the complete code research pipeline in CI/CD without external dependencies.
"""

import asyncio
import hashlib
from collections.abc import AsyncIterator
from typing import Any

from chunkhound.interfaces.llm_provider import LLMProvider, LLMResponse
from chunkhound.interfaces.embedding_provider import EmbeddingConfig, RerankResult


class FakeLLMProvider(LLMProvider):
    """Fake LLM provider that returns scripted responses based on prompt patterns.

    Designed to test the full code research pipeline without real LLM API calls.
    Returns deterministic responses based on prompt content patterns.
    """

    def __init__(
        self,
        model: str = "fake-gpt",
        responses: dict[str, str] | None = None,
    ):
        """Initialize fake LLM provider.

        Args:
            model: Model name for identification
            responses: Optional dict mapping prompt substrings to responses
        """
        self._model = model
        self._requests_made = 0
        self._tokens_used = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0

        # Default responses for common patterns
        self._responses = responses or {
            "expand": "function definition, class implementation, code structure",
            "follow": "1. How is search implemented?\n2. What are the key algorithms?\n3. How does data flow through the system?",
            "synthesis": "## Overview\nThe codebase implements semantic search with BFS traversal.\n\n## Key Components\n- Search service handles queries\n- Deep research coordinates BFS exploration\n- Database provider stores chunks\n\n## Data Flow\nQueries → Semantic search → Chunk retrieval → Smart boundaries → Synthesis",
            "code": "semantic search, deep research, database operations",
        }

    @property
    def name(self) -> str:
        """Provider name."""
        return "fake"

    @property
    def model(self) -> str:
        """Model name."""
        return self._model

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int = 4096,
        timeout: int | None = None,
    ) -> LLMResponse:
        """Generate a completion based on prompt patterns."""
        await asyncio.sleep(0.001)  # Simulate minimal latency

        self._requests_made += 1

        # Match prompt to response pattern
        prompt_lower = prompt.lower()
        response_content = "Default test response"

        for pattern, response in self._responses.items():
            if pattern in prompt_lower:
                response_content = response
                break

        # Estimate tokens
        prompt_tokens = self.estimate_tokens(prompt)
        if system:
            prompt_tokens += self.estimate_tokens(system)
        completion_tokens = self.estimate_tokens(response_content)
        total_tokens = prompt_tokens + completion_tokens

        self._prompt_tokens += prompt_tokens
        self._completion_tokens += completion_tokens
        self._tokens_used += total_tokens

        return LLMResponse(
            content=response_content,
            tokens_used=total_tokens,
            model=self._model,
            finish_reason="stop",
        )

    async def batch_complete(
        self,
        prompts: list[str],
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> list[LLMResponse]:
        """Generate completions for multiple prompts."""
        tasks = [
            self.complete(prompt, system, max_completion_tokens) for prompt in prompts
        ]
        return await asyncio.gather(*tasks)

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (4 chars per token)."""
        return len(text) // 4

    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        return {
            "status": "healthy",
            "provider": "fake",
            "model": self._model,
            "test_response": "OK",
        }

    def get_usage_stats(self) -> dict[str, Any]:
        """Get usage statistics."""
        return {
            "requests_made": self._requests_made,
            "total_tokens": self._tokens_used,
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
        }


class FakeEmbeddingProvider:
    """Fake embedding provider that returns deterministic vectors.

    Generates consistent embeddings based on text content hash,
    allowing reproducible tests without API calls.
    """

    def __init__(
        self,
        model: str = "fake-embeddings",
        dims: int = 1536,
        batch_size: int = 100,
    ):
        """Initialize fake embedding provider.

        Args:
            model: Model name for identification
            dims: Embedding dimensions
            batch_size: Maximum batch size
        """
        self._model = model
        self._dims = dims
        self._batch_size = batch_size
        self._distance = "cosine"
        self._max_tokens = 8192

        # Usage tracking
        self._requests_made = 0
        self._tokens_used = 0
        self._embeddings_generated = 0

    @property
    def name(self) -> str:
        """Provider name."""
        return "fake"

    @property
    def model(self) -> str:
        """Model name."""
        return self._model

    @property
    def dims(self) -> int:
        """Embedding dimensions."""
        return self._dims

    @property
    def distance(self) -> str:
        """Distance metric."""
        return self._distance

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
            provider="fake",
            model=self._model,
            dims=self._dims,
            distance=self._distance,
            batch_size=self._batch_size,
            max_tokens=self._max_tokens,
        )

    def _generate_deterministic_vector(self, text: str) -> list[float]:
        """Generate deterministic embedding vector from text hash."""
        # Use text hash to seed vector generation
        text_hash = hashlib.sha256(text.encode()).digest()

        # Convert hash bytes to floats in [-1, 1] range
        vector = []
        for i in range(self._dims):
            # Use hash bytes cyclically
            byte_idx = i % len(text_hash)
            byte_val = text_hash[byte_idx]
            # Normalize to [-1, 1]
            normalized = (byte_val / 255.0) * 2 - 1
            vector.append(normalized)

        # Normalize vector to unit length for cosine similarity
        magnitude = sum(x * x for x in vector) ** 0.5
        if magnitude > 0:
            vector = [x / magnitude for x in vector]

        return vector

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        if not texts:
            return []

        await asyncio.sleep(0.001)  # Simulate minimal latency

        self._requests_made += 1
        self._embeddings_generated += len(texts)
        self._tokens_used += sum(self.estimate_tokens(text) for text in texts)

        return [self._generate_deterministic_vector(text) for text in texts]

    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        embeddings = await self.embed([text])
        return embeddings[0]

    async def embed_batch(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        """Generate embeddings in batches."""
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
        pass

    async def shutdown(self) -> None:
        """Shutdown the embedding provider."""
        pass

    def is_available(self) -> bool:
        """Check if provider is available."""
        return True

    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        return {
            "status": "healthy",
            "provider": "fake",
            "model": self._model,
            "dimensions": self._dims,
            "requests_made": self._requests_made,
            "tokens_used": self._tokens_used,
            "embeddings_generated": self._embeddings_generated,
        }

    def validate_texts(self, texts: list[str]) -> list[str]:
        """Validate and preprocess texts."""
        return [text if text else " " for text in texts]

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (3 chars per token for embeddings)."""
        return max(1, len(text) // 3)

    def chunk_text_by_tokens(self, text: str, max_tokens: int) -> list[str]:
        """Split text into chunks by token count."""
        chars_per_chunk = max_tokens * 3
        chunks = []
        for i in range(0, len(text), chars_per_chunk):
            chunks.append(text[i : i + chars_per_chunk])
        return chunks

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the embedding model."""
        return {
            "provider": "fake",
            "model": self._model,
            "dimensions": self._dims,
            "max_tokens": self._max_tokens,
            "supports_reranking": True,
        }

    def get_usage_stats(self) -> dict[str, Any]:
        """Get usage statistics."""
        return {
            "requests_made": self._requests_made,
            "tokens_used": self._tokens_used,
            "embeddings_generated": self._embeddings_generated,
        }

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

    def get_supported_distances(self) -> list[str]:
        """Get list of supported distance metrics."""
        return ["cosine", "l2", "ip"]

    def get_optimal_batch_size(self) -> int:
        """Get optimal batch size."""
        return min(self._batch_size, 100)

    def get_max_tokens_per_batch(self) -> int:
        """Get maximum tokens per batch."""
        return 320000

    def get_max_documents_per_batch(self) -> int:
        """Get maximum documents per batch."""
        return 1000

    def get_recommended_concurrency(self) -> int:
        """Get recommended concurrency."""
        return 10

    def get_chars_to_tokens_ratio(self) -> float:
        """Get character-to-token ratio."""
        return 3.0

    # Reranking Operations
    def supports_reranking(self) -> bool:
        """Fake provider supports reranking."""
        return True

    async def rerank(
        self, query: str, documents: list[str], top_k: int | None = None
    ) -> list[RerankResult]:
        """Rerank documents by relevance using deterministic scoring."""
        if not documents:
            return []

        await asyncio.sleep(0.001)  # Simulate minimal latency

        self._requests_made += 1

        # Generate deterministic relevance scores based on query-document similarity
        query_vector = self._generate_deterministic_vector(query)
        results = []

        for idx, doc in enumerate(documents):
            doc_vector = self._generate_deterministic_vector(doc)
            # Cosine similarity
            score = sum(a * b for a, b in zip(query_vector, doc_vector))
            results.append(RerankResult(index=idx, score=score))

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)

        # Apply top_k if specified
        if top_k is not None:
            results = results[:top_k]

        return results
