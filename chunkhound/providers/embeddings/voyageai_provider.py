"""VoyageAI embedding provider implementation for ChunkHound - concrete embedding provider using VoyageAI API."""

import asyncio
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx
from loguru import logger

from chunkhound.core.config.voyageai_utils import is_official_voyageai_endpoint
from chunkhound.core.constants import VOYAGE_DEFAULT_MODEL, VOYAGE_DEFAULT_RERANK_MODEL
from chunkhound.core.utils import EMBEDDING_CHARS_PER_TOKEN
from chunkhound.interfaces.embedding_provider import EmbeddingConfig, RerankResult

from .shared_utils import (
    chunk_text_by_words,
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
        rerank_batch_size: int | None = None,
        base_url: str | None = None,
        rerank_url: str | None = None,
        rerank_format: str = "auto",
        max_concurrent_batches: int | None = None,
    ):
        """Initialize VoyageAI embedding provider.

        Args:
            api_key: VoyageAI API key (defaults to VOYAGE_API_KEY env var)
            model: Model name to use for embeddings
            rerank_model: Model name to use for reranking (SDK path only)
            batch_size: Maximum batch size for API requests
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts for failed requests
            retry_delay: Delay between retry attempts
            max_tokens: Maximum tokens per request (if applicable)
            rerank_batch_size: Max documents per rerank batch (overrides default of 1000)
            base_url: Custom API base URL (overrides https://api.voyageai.com/v1)
            rerank_url: Separate reranker endpoint URL (absolute http/https).
                When set, reranking uses HTTP instead of the VoyageAI SDK.
            rerank_format: Reranking API format when using rerank_url.
                'cohere' for Cohere-compatible APIs (requires rerank_model),
                'tei' for HuggingFace TEI (model set at deployment),
                'auto' to detect from response (default).
            max_concurrent_batches: Maximum number of concurrent embed() calls.
                Defaults to 1 for custom endpoints (e.g. Azure ML) to avoid
                HTTP 424 "Failed Dependency" from concurrent-request overload,
                and to RECOMMENDED_CONCURRENCY for the official VoyageAI API.
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
        self._base_url = base_url
        self._rerank_url = rerank_url
        self._rerank_format = rerank_format
        self._model_config = model_config
        self._rerank_batch_size = rerank_batch_size

        # For non-official custom endpoints without an API key, pass a placeholder to
        # satisfy the SDK's requirement — the server ignores the auth header.
        # Official VoyageAI endpoints require a real key; "no-key" there produces a
        # cryptic auth error rather than a clear config failure.
        is_custom = base_url and not is_official_voyageai_endpoint(base_url)
        effective_api_key = api_key if api_key else ("no-key" if is_custom else None)

        # Use the system CA bundle when available so that corporate proxy CAs
        # (e.g. Blue Coat / AMAT) are trusted without patching certifi.
        # self._ssl_verify is passed directly to httpx.AsyncClient(verify=...).
        _sys_ca = "/etc/ssl/certs/ca-certificates.crt"
        if os.path.exists(_sys_ca) and not os.environ.get("REQUESTS_CA_BUNDLE"):
            self._ssl_verify: str | bool = _sys_ca
        else:
            self._ssl_verify = os.environ.get("REQUESTS_CA_BUNDLE") or True

        # Initialize client
        self._client = voyageai.Client(api_key=effective_api_key, timeout=timeout)
        if base_url:
            # voyageai >=0.3.7 uses "base_url" in _params (popped before serialization);
            # voyageai <0.3.7 uses "api_base" (named param in create()).
            # Sending "api_base" on 0.3.7+ puts it in the request body → API error.
            key = "base_url" if "base_url" in self._client._params else "api_base"
            self._client._params[key] = base_url  # per-instance, not global

        # Model dimension mapping - built from configuration
        self._dimensions_map = {
            model_name: config["default_dimension"]
            for model_name, config in VOYAGE_MODEL_CONFIG.items()
        }

        # Usage tracking
        self._requests_made = 0
        self._tokens_used = 0
        self._embeddings_generated = 0

        # Concurrency limiter: custom endpoints (e.g. Azure ML) often reject
        # simultaneous requests with HTTP 424. Default to 1 for custom base_url,
        # high value for the official API which supports 2000 RPM.
        if max_concurrent_batches is None:
            max_concurrent_batches = 1 if base_url else self.RECOMMENDED_CONCURRENCY
        self._embed_semaphore = asyncio.Semaphore(max_concurrent_batches)

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
        """Generate embeddings for a list of texts with automatic retry on network errors.

        Internally sub-batches to self._batch_size so that custom endpoints
        (e.g. Azure ML) are never overwhelmed by a single oversized request.
        """
        if not texts:
            return []

        validated_texts = validate_text_input(texts)
        if not validated_texts:
            return []

        # Sub-batch when input exceeds batch_size (protects custom/low-throughput endpoints)
        if len(validated_texts) > self._batch_size:
            all_embeddings: list[list[float]] = []
            for i in range(0, len(validated_texts), self._batch_size):
                sub_batch = validated_texts[i : i + self._batch_size]
                all_embeddings.extend(await self._embed_single_batch(sub_batch))
            return all_embeddings

        return await self._embed_single_batch(validated_texts)

    async def _embed_single_batch(self, texts: list[str]) -> list[list[float]]:
        """Send one batch to the API with retry logic."""
        async with self._embed_semaphore:
            return await self._embed_single_batch_locked(texts)

    async def _embed_single_batch_locked(self, texts: list[str]) -> list[list[float]]:
        """Inner embed implementation, called while holding the semaphore."""
        # Retry loop for transient network errors
        for attempt in range(self._retry_attempts):
            try:
                result = await asyncio.to_thread(
                    self._client.embed,
                    texts=texts,
                    model=self._model,
                    input_type="document",
                    truncation=True,
                )

                self._requests_made += 1
                self._tokens_used += result.total_tokens
                self._embeddings_generated += len(texts)

                return [embedding for embedding in result.embeddings]

            except Exception as e:
                # Classify error type for retry decision
                error_type = type(e).__name__
                error_module = type(e).__module__
                error_str = str(e)

                # Network / transient errors that should be retried
                is_network_error = any(
                    [
                        "APIConnectionError" in error_type,
                        "ConnectionError" in error_type,
                        "RemoteDisconnected" in error_type,
                        "Timeout" in error_type,
                        "TimeoutError" in error_type,
                    ]
                )

                # HTTP 408 (upstream request timeout) from Azure ML / proxies:
                # treat as transient and retry with a longer initial backoff
                is_upstream_timeout = "408" in error_str or (
                    "upstream request timeout" in error_str.lower()
                )

                if (
                    is_network_error or is_upstream_timeout
                ) and attempt < self._retry_attempts - 1:
                    # Longer backoff for upstream timeouts — endpoint needs time to recover
                    base_delay = 10.0 if is_upstream_timeout else self._retry_delay
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"VoyageAI embedding failed with {error_module}.{error_type} "
                        f"(attempt {attempt + 1}/{self._retry_attempts}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    # Non-retryable error or last attempt - log and raise
                    if is_network_error or is_upstream_timeout:
                        logger.error(
                            f"VoyageAI embedding failed after {self._retry_attempts} attempts: {e}"
                        )
                    else:
                        logger.error(
                            f"VoyageAI embedding failed with non-retryable error: {e}"
                        )
                    raise RuntimeError(f"Embedding generation failed: {e}") from e

        # Should never reach here, but provide clear error if we do
        raise RuntimeError(
            f"Embedding generation failed after {self._retry_attempts} attempts"
        )

    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        embeddings = await self.embed([text])
        return embeddings[0]

    async def embed_batch(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        """Generate embeddings in batches respecting both count and token limits."""
        if not texts:
            return []

        effective_batch_size = batch_size or self._batch_size
        max_tokens_per_batch = self._model_config["max_tokens_per_batch"]

        all_embeddings: list[list[float]] = []
        current_batch: list[str] = []
        current_tokens = 0

        for text in texts:
            text_tokens = self.estimate_tokens(text)

            if current_batch and (
                len(current_batch) >= effective_batch_size
                or current_tokens + text_tokens > max_tokens_per_batch
            ):
                all_embeddings.extend(await self.embed(current_batch))
                current_batch = []
                current_tokens = 0

            current_batch.append(text)
            current_tokens += text_tokens

        if current_batch:
            all_embeddings.extend(await self.embed(current_batch))

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
        return VOYAGEAI_AVAILABLE and (
            self._api_key is not None or self._base_url is not None
        )

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
        """Estimate token count for a text using central embedding ratio.

        Based on actual measurements: 3.0 chars/token for VoyageAI.
        """
        if not text:
            return 0
        return max(1, len(text) // EMBEDDING_CHARS_PER_TOKEN)

    def validate_texts(self, texts: list[str]) -> list[str]:
        """Validate and preprocess texts before embedding."""
        return validate_text_input(texts)

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
            "supports_reranking": self.supports_reranking(),
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

    def get_max_rerank_batch_size(self) -> int:
        """Get maximum documents per batch for reranking operations.

        VoyageAI's SDK handles batching internally, so we return a large limit.
        The actual batch splitting is managed by the VoyageAI client library.

        Implements bounded override pattern: user can set batch size, but it's
        clamped to a conservative default of 1000 for safety.

        Returns:
            Maximum documents per rerank batch (user override or 1000 default)
        """
        # Conservative default: 1000 documents (prevent OOM on large result sets)
        default_limit = 1000

        # User override (bounded by default limit for safety)
        if self._rerank_batch_size is not None:
            return min(self._rerank_batch_size, default_limit)

        # VoyageAI SDK handles batching, but we set a conservative client-side limit
        # to prevent memory issues when processing very large result sets
        return default_limit

    # Reranking Operations
    def supports_reranking(self) -> bool:
        """Return True if reranking is available with the current configuration.

        - Custom base_url (e.g. Azure ML): only supported when rerank_url is
          explicitly configured, since the embedding endpoint does not expose /rerank.
        - Official VoyageAI API (no base_url): always supported via SDK.
        """
        if self._base_url:
            return self._rerank_url is not None
        return True

    async def rerank(
        self, query: str, documents: list[str], top_k: int | None = None
    ) -> list[RerankResult]:
        """Rerank documents by relevance to query.

        Dispatches to HTTP-based reranking when rerank_url is configured,
        otherwise uses the VoyageAI SDK (official API only).
        """
        if not documents:
            return []

        if self._rerank_url:
            return await self._rerank_via_http(query, documents, top_k)

        return await self._rerank_via_sdk(query, documents, top_k)

    async def _rerank_via_sdk(
        self, query: str, documents: list[str], top_k: int | None
    ) -> list[RerankResult]:
        """Rerank using the VoyageAI SDK (official API)."""
        for attempt in range(self._retry_attempts):
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
                error_type = type(e).__name__
                error_module = type(e).__module__
                is_network_error = any(
                    [
                        "APIConnectionError" in error_type,
                        "ConnectionError" in error_type,
                        "RemoteDisconnected" in error_type,
                        "Timeout" in error_type,
                        "TimeoutError" in error_type,
                    ]
                )
                if is_network_error and attempt < self._retry_attempts - 1:
                    delay = self._retry_delay * (2**attempt)
                    logger.warning(
                        f"VoyageAI reranking failed with {error_module}.{error_type} "
                        f"(attempt {attempt + 1}/{self._retry_attempts}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    if is_network_error:
                        logger.error(
                            f"VoyageAI reranking failed after {self._retry_attempts} attempts: {e}"
                        )
                    else:
                        logger.error(
                            f"VoyageAI reranking failed with non-retryable error: {e}"
                        )
                    raise RuntimeError(f"Reranking failed: {e}") from e

        raise RuntimeError(f"Reranking failed after {self._retry_attempts} attempts")

    async def _rerank_via_http(
        self, query: str, documents: list[str], top_k: int | None
    ) -> list[RerankResult]:
        """Rerank using a separate HTTP reranker service (TEI or Cohere format).

        Handles batching when document count exceeds rerank_batch_size.
        """
        batch_limit = self.get_max_rerank_batch_size()

        if len(documents) <= batch_limit:
            results = await self._rerank_http_batch(query, documents, top_k)
            if top_k is not None:
                results = results[:top_k]
            return results

        # Split into batches and aggregate
        all_results: list[RerankResult] = []
        for start in range(0, len(documents), batch_limit):
            batch = documents[start : start + batch_limit]
            batch_results = await self._rerank_http_batch(query, batch, top_k=None)
            for r in batch_results:
                all_results.append(RerankResult(index=r.index + start, score=r.score))

        all_results.sort(key=lambda r: r.score, reverse=True)
        if top_k is not None:
            all_results = all_results[:top_k]
        return all_results

    async def _rerank_http_batch(
        self, query: str, documents: list[str], top_k: int | None
    ) -> list[RerankResult]:
        """Send one batch to the HTTP reranker and return parsed results."""
        payload = self._build_rerank_payload(query, documents, top_k)

        logger.debug(
            f"HTTP reranking {len(documents)} documents at {self._rerank_url} "
            f"(format={self._rerank_format})"
        )

        async with httpx.AsyncClient(
            timeout=self._timeout, verify=self._ssl_verify
        ) as client:
            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            response = await client.post(
                self._rerank_url, json=payload, headers=headers
            )
            response.raise_for_status()
            data = response.json()

        # Normalise bare-array response (TEI) to dict form
        if isinstance(data, list):
            data = {"results": data}

        if isinstance(data, dict) and "error" in data:
            raise ValueError(f"Rerank service error: {data['error']}")

        return self._parse_rerank_response(data, len(documents))

    def _build_rerank_payload(
        self, query: str, documents: list[str], top_k: int | None
    ) -> dict:
        """Build rerank request payload for TEI or Cohere format."""
        fmt = self._rerank_format
        if fmt == "tei":
            return {"query": query, "texts": documents}
        elif fmt == "cohere":
            payload: dict = {"query": query, "documents": documents}
            if self._rerank_model:
                payload["model"] = self._rerank_model
            if top_k is not None:
                payload["top_n"] = top_k
            return payload
        else:  # auto: try Cohere if model provided, else TEI
            if self._rerank_model:
                payload = {
                    "query": query,
                    "documents": documents,
                    "model": self._rerank_model,
                }
                if top_k is not None:
                    payload["top_n"] = top_k
                return payload
            return {"query": query, "texts": documents}

    def _parse_rerank_response(
        self, data: dict, num_documents: int
    ) -> list[RerankResult]:
        """Parse reranker HTTP response (Cohere or TEI format) into RerankResult list."""
        if "results" not in data:
            raise ValueError(
                f"Invalid rerank response: missing 'results' field. Got: {list(data.keys())}"
            )

        results = []
        for item in data["results"]:
            # Cohere: {"index": N, "relevance_score": F}
            # TEI:    {"index": N, "score": F}
            idx = item.get("index")
            score = (
                item.get("relevance_score")
                if "relevance_score" in item
                else item.get("score")
            )
            if idx is None or score is None:
                logger.warning(f"Skipping malformed rerank result: {item}")
                continue
            if not (0 <= idx < num_documents):
                logger.warning(
                    f"Rerank index {idx} out of range ({num_documents} docs), skipping"
                )
                continue
            results.append(RerankResult(index=idx, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results
