"""VoyageAI embedding provider implementation for ChunkHound - concrete embedding provider using VoyageAI API."""

import asyncio
import os
from collections.abc import AsyncIterator, Sequence
from typing import Any, TypedDict, cast

import httpx
from loguru import logger

from chunkhound.core.config.embedding_config import validate_rerank_configuration
from chunkhound.core.constants import VOYAGE_DEFAULT_MODEL, VOYAGE_DEFAULT_RERANK_MODEL
from chunkhound.core.exceptions.embedding import (
    EmbeddingConfigurationError,
    EmbeddingProviderError,
)
from chunkhound.core.utils import EMBEDDING_CHARS_PER_TOKEN
from chunkhound.core.utils.voyageai_utils import (
    OFFICIAL_VOYAGEAI_BASE_V1,
    is_official_voyageai_endpoint,
)
from chunkhound.interfaces.embedding_provider import EmbeddingConfig, RerankResult

from .shared_utils import (
    apply_client_side_truncation,
    build_dimension_request_param,
    build_runtime_supported_dimensions,
    chunk_text_by_words,
    get_dimensions_for_model,
    get_usage_stats_dict,
    validate_embedding_dims,
    validate_positive_output_dims,
    validate_runtime_output_dims_config,
    validate_text_input,
)

try:
    import voyageai

    VOYAGEAI_AVAILABLE = True
except ImportError:
    voyageai = None  # type: ignore
    VOYAGEAI_AVAILABLE = False
    logger.warning("VoyageAI not available - install with: uv pip install voyageai")


class VoyageModelConfig(TypedDict):
    """Static config for a known VoyageAI model.

    For unknown/custom models ``dimensions`` is intentionally empty ([]), and
    the provider discovers the native dimension at runtime.  See
    ``DEFAULT_UNKNOWN_MODEL_CONFIG``.
    """

    max_tokens_per_batch: int
    max_texts_per_batch: int
    context_length: int
    dimensions: list[int]
    default_dimension: int


# Official VoyageAI model configuration based on API documentation
VOYAGE_MODEL_CONFIG: dict[str, VoyageModelConfig] = {
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


DEFAULT_UNKNOWN_MODEL_CONFIG: VoyageModelConfig = {
    "max_tokens_per_batch": 320000,
    "max_texts_per_batch": 1000,
    "context_length": 32000,
    "dimensions": [],
    "default_dimension": 1024,
}


# Base backoff (seconds) for each retry category. Network errors fall through
# to the per-provider ``self._retry_delay`` so the existing instance setting
# still drives generic connection retries.
_CATEGORY_BACKOFFS: dict[str, float] = {
    # TPM / RPM windows on VoyageAI are 60s; 30s lets the window partially
    # drain before the first retry.
    "rate_limit": 30.0,
    # Azure ML / proxy 408s mean the upstream endpoint is overloaded; give it
    # room to recover before we hit it again.
    "upstream_timeout": 10.0,
}


def _classify_voyageai_error(e: Exception) -> str | None:
    """Classify a VoyageAI SDK exception for retry decisions.

    Returns one of ``"rate_limit"``, ``"upstream_timeout"``, ``"network"``,
    or ``None`` for a non-retryable error. Shared by the embedding and
    reranking code paths so their retry behavior cannot silently drift
    apart (a prior bug where the rerank path was missing ``str(e)`` went
    unnoticed for exactly this reason).
    """
    error_type = type(e).__name__
    error_str = str(e)
    error_lower = error_str.lower()

    # Rate limit / server errors from the VoyageAI SDK (HTTP 429, 5xx).
    # The ``"rate limit"`` substring match is a fallback for custom
    # endpoints (Azure ML, self-hosted proxies) that surface 429s as
    # generic ``RuntimeError`` / ``Exception`` rather than a dedicated
    # SDK exception class.
    if (
        "RateLimitError" in error_type
        or "TryAgain" in error_type
        or "ServerError" in error_type
        or "ServiceUnavailableError" in error_type
        or "rate limit" in error_lower
    ):
        return "rate_limit"

    # HTTP 408 (upstream request timeout) from Azure ML / proxies: treat
    # as transient and retry with a longer initial backoff.
    if "408" in error_str or "upstream request timeout" in error_lower:
        return "upstream_timeout"

    # Network / transient connection errors.
    if (
        "APIConnectionError" in error_type
        or "ConnectionError" in error_type
        or "RemoteDisconnected" in error_type
        or "Timeout" in error_type
        or "TimeoutError" in error_type
    ):
        return "network"

    return None


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
        output_dims: int | None = None,
        client_side_truncation: bool = False,
        base_url: str | None = None,
        rerank_url: str | None = None,
        rerank_format: str = "auto",
        max_concurrent_batches: int | None = None,
        ssl_verify: bool = True,
        rerank_ssl_verify: bool | None = None,
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
            output_dims: Optional server-side output dimension override. Known
                official Voyage models keep their whitelist. Unknown/custom
                Voyage-compatible endpoints are trusted instead and prove
                compatibility at runtime.
            client_side_truncation: If True, request full native vectors and
                truncate locally instead of sending output_dimension. Direct
                provider construction may defer the missing-output_dims failure
                until the first embed() call, but it will fail explicitly.
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
            ssl_verify: Verify TLS certificates for requests sent via explicit
                custom HTTP endpoints. The VoyageAI SDK path remains verified.
            rerank_ssl_verify: Verify TLS certificates for HTTP rerank requests.
                Defaults to ssl_verify when unset.
        """
        if not VOYAGEAI_AVAILABLE:
            raise ImportError(
                "VoyageAI not available - install with: uv pip install voyageai"
            )

        self._model = model
        self._rerank_model = rerank_model

        # Known official VoyageAI models have strict dimension whitelists.
        # Unknown/custom Voyage-compatible models must stay permissive so runtime
        # validation can learn the actual vector size from the first full native
        # response. Until then, introspection falls back to 1024 dims.
        self._is_known_model = model in VOYAGE_MODEL_CONFIG
        model_config = VOYAGE_MODEL_CONFIG.get(model, DEFAULT_UNKNOWN_MODEL_CONFIG)

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
        self._discovered_native_dims: int | None = None

        self._output_dims = validate_positive_output_dims(output_dims, model=model)
        self._client_side_truncation = client_side_truncation
        self._validate_output_dims_config()

        self._ssl_verify_enabled = ssl_verify
        self._rerank_ssl_verify_enabled = (
            rerank_ssl_verify if rerank_ssl_verify is not None else ssl_verify
        )

        # For non-official custom endpoints without an API key, pass a placeholder to
        # satisfy the SDK's requirement — the server ignores the auth header.
        # Official VoyageAI endpoints require a real key; "no-key" there produces a
        # cryptic auth error rather than a clear config failure.
        is_custom = base_url and not is_official_voyageai_endpoint(base_url)
        effective_api_key = api_key if api_key else ("no-key" if is_custom else None)

        self._ssl_verify = self._resolve_http_verify_setting(
            self._ssl_verify_enabled, endpoint=base_url
        )
        self._rerank_ssl_verify = self._resolve_http_verify_setting(
            self._rerank_ssl_verify_enabled, endpoint=rerank_url
        )

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
        # simultaneous requests with HTTP 424. Default to 1 only for custom
        # endpoints; the official VoyageAI API keeps the higher recommended
        # concurrency even when callers pass its base_url explicitly.
        if max_concurrent_batches is None:
            max_concurrent_batches = (
                1 if is_custom else self.RECOMMENDED_CONCURRENCY
            )
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
    def base_url(self) -> str:
        """Base URL for API requests."""
        return self._base_url or OFFICIAL_VOYAGEAI_BASE_V1

    @staticmethod
    def _resolve_http_verify_setting(
        ssl_verify_enabled: bool, endpoint: str | None
    ) -> str | bool:
        """Resolve the verify setting for explicit HTTP client calls."""
        if endpoint is None:
            return True
        if not ssl_verify_enabled:
            return False

        _sys_ca = "/etc/ssl/certs/ca-certificates.crt"
        if os.path.exists(_sys_ca) and not os.environ.get("REQUESTS_CA_BUNDLE"):
            return _sys_ca
        return os.environ.get("REQUESTS_CA_BUNDLE") or True

    @property
    def dims(self) -> int:
        """Actual output dimension (reflects matryoshka config if set).

        For unknown/custom models, falls through to runtime-discovered native
        dimension or a temporary 1024 fallback before the first full response.
        """
        if self._output_dims is not None:
            return self._output_dims
        if self._is_known_model:
            return get_dimensions_for_model(
                self._model, self._dimensions_map, default_dims=1024
            )
        if self._discovered_native_dims is not None:
            return self._discovered_native_dims
        return 1024

    @property
    def native_dims(self) -> int:
        """Model's full/native embedding dimension.

        For unknown/custom models, returns the runtime-discovered native
        dimension after the first full-dimension API response, or a temporary
        1024 fallback before discovery completes.
        """
        dims_list = self._model_config.get("dimensions", [])
        if dims_list:
            return max(dims_list)
        if self._discovered_native_dims is not None:
            return self._discovered_native_dims
        return 1024

    @property
    def supported_dimensions(self) -> Sequence[int]:
        """Known-valid output dimensions for this model.

        Unknown/custom Voyage-compatible models have no static whitelist before
        the first real API response. After runtime discovery, client-side
        truncation exposes every dimension from 1..native_dims; otherwise only
        the native dimension is valid.

        Use ``in`` for membership checks, not equality.
        """
        dims = self._model_config["dimensions"]
        if dims:
            return dims
        return build_runtime_supported_dimensions(
            self._discovered_native_dims,
            self._client_side_truncation,
        )

    @property
    def output_dims(self) -> int | None:
        """Configured output dimension override, or None for native."""
        return self._output_dims

    @property
    def client_side_truncation(self) -> bool:
        """Whether client-side truncation is enabled."""
        return self._client_side_truncation

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
            base_url=self._base_url,
            timeout=self._timeout,
            retry_attempts=self._retry_attempts,
            retry_delay=self._retry_delay,
            output_dims=self.output_dims,
            client_side_truncation=self.client_side_truncation,
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

    def _validate_output_dims_config_for(
        self,
        *,
        model: str,
        model_config: VoyageModelConfig,
        is_known_model: bool,
        output_dims: int | None,
        client_side_truncation: bool,
    ) -> None:
        """Validate output-dims semantics for a concrete config snapshot."""
        validate_runtime_output_dims_config(
            output_dims,
            client_side_truncation,
            model=model,
            context="client-side truncation",
        )
        if output_dims is None or not is_known_model:
            return
        default_dim = model_config.get("default_dimension", 1024)
        supported = model_config.get("dimensions", [default_dim])
        if output_dims not in supported:
            raise EmbeddingConfigurationError(
                f"output_dims {output_dims} not in supported "
                f"dimensions {supported} for model {model}"
            )

    def _validate_output_dims_config(self) -> None:
        """Validate output_dims and client_side_truncation, enforcing VoyageAI whitelist."""
        self._validate_output_dims_config_for(
            model=self._model,
            model_config=self._model_config,
            is_known_model=self._is_known_model,
            output_dims=self._output_dims,
            client_side_truncation=self._client_side_truncation,
        )

    def _reset_runtime_output_dims_state(self) -> None:
        """Clear runtime-discovered dimensions after config changes."""
        self._discovered_native_dims = None

    def _expected_raw_dims(self) -> int | None:
        """Expected native dimension from API before client-side truncation.

        Returns ``None`` for undiscovered unknown models, causing callers to
        skip dimension validation until the native size is known from the
        first full-dimension response.
        """
        if self._client_side_truncation:
            # Client-side: expect full native dims from API
            if self._is_known_model:
                return self.native_dims
            return self._discovered_native_dims
        # Server-side or no truncation: expect the configured output dims
        return self.dims

    def _maybe_discover_native_dims(
        self, raw_dims: int | None, server_side_truncation: bool
    ) -> None:
        """Discover native dims for unknown models from untruncated responses."""
        if (
            raw_dims is not None
            and not self._is_known_model
            and self._discovered_native_dims is None
            and not server_side_truncation
        ):
            self._discovered_native_dims = raw_dims
            logger.debug(
                f"Discovered native embedding dimension {raw_dims} "
                f"for model {self._model}"
            )

    async def _embed_single_batch_locked(self, texts: list[str]) -> list[list[float]]:
        """Inner embed implementation, called while holding the semaphore."""
        # Retry loop for transient network errors
        for attempt in range(self._retry_attempts):
            try:
                embed_kwargs: dict[str, Any] = {
                    "texts": texts,
                    "model": self._model,
                    "input_type": "document",
                    "truncation": True,
                }
                output_dims = validate_runtime_output_dims_config(
                    self._output_dims,
                    self._client_side_truncation,
                    model=self._model,
                    context="client-side truncation",
                )
                dim_param = build_dimension_request_param(
                    output_dims, self._client_side_truncation
                )
                # User configured output_dims — they know what they're doing.
                # Pass through; the API rejects unsupported dimensions on its own.
                if dim_param is not None:
                    embed_kwargs["output_dimension"] = dim_param
                result = await asyncio.to_thread(self._client.embed, **embed_kwargs)

                self._requests_made += 1
                self._tokens_used += result.total_tokens
                self._embeddings_generated += len(texts)

                embeddings = cast(list[list[float]], list(result.embeddings))

                raw_dims = len(embeddings[0]) if embeddings else None
                server_side_truncation = (
                    self._output_dims is not None and not self._client_side_truncation
                )

                self._maybe_discover_native_dims(raw_dims, server_side_truncation)

                # Validate raw API response dims before any client-side truncation
                expected_raw = self._expected_raw_dims()
                if raw_dims is not None and expected_raw is not None:
                    validate_embedding_dims(
                        raw_dims, expected_raw, model=self._model
                    )

                # Apply client-side truncation when server doesn't support dim param
                if self._client_side_truncation:
                    embeddings = apply_client_side_truncation(
                        embeddings, cast(int, output_dims)
                    )

                # Validate final embedding dimension after all truncation
                if embeddings:
                    validate_embedding_dims(
                        len(embeddings[0]), self.dims, model=self._model
                    )

                return embeddings

            except EmbeddingProviderError:
                raise  # Non-retryable domain errors propagate as-is

            except Exception as e:
                error_type = type(e).__name__
                error_module = type(e).__module__
                category = _classify_voyageai_error(e)

                if category is not None and attempt < self._retry_attempts - 1:
                    base_delay = _CATEGORY_BACKOFFS.get(category, self._retry_delay)
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"VoyageAI embedding failed with {error_module}.{error_type} "
                        f"(attempt {attempt + 1}/{self._retry_attempts}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    if category is not None:
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
            "native_dims": self.native_dims,
            "output_dims": self.output_dims,
            "client_side_truncation": self.client_side_truncation,
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
        next_model = kwargs.get("model", self._model)
        next_is_known_model = next_model in VOYAGE_MODEL_CONFIG
        next_model_config = VOYAGE_MODEL_CONFIG.get(
            next_model, DEFAULT_UNKNOWN_MODEL_CONFIG
        )
        next_output_dims = self._output_dims
        if "output_dims" in kwargs:
            next_output_dims = validate_positive_output_dims(
                kwargs["output_dims"], model=next_model
            )
        next_client_side_truncation = kwargs.get(
            "client_side_truncation", self._client_side_truncation
        )
        if {"model", "output_dims", "client_side_truncation"} & kwargs.keys():
            self._validate_output_dims_config_for(
                model=next_model,
                model_config=next_model_config,
                is_known_model=next_is_known_model,
                output_dims=next_output_dims,
                client_side_truncation=next_client_side_truncation,
            )

        if "model" in kwargs:
            self._model = next_model
            self._is_known_model = next_is_known_model
            self._model_config = next_model_config
            self._reset_runtime_output_dims_state()
        if "rerank_model" in kwargs:
            self._rerank_model = kwargs["rerank_model"]
        if "batch_size" in kwargs:
            self._batch_size = kwargs["batch_size"]
        if "timeout" in kwargs:
            self._timeout = kwargs["timeout"]
        if "output_dims" in kwargs:
            self._output_dims = next_output_dims
        if "client_side_truncation" in kwargs:
            self._client_side_truncation = next_client_side_truncation

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
        """Return True if reranking can run with the current configuration."""
        try:
            validate_rerank_configuration(
                provider="voyageai",
                rerank_format=self._rerank_format,
                rerank_model=self._rerank_model,
                rerank_url=self._rerank_url,
                base_url=self._base_url,
            )
        except ValueError:
            return False

        if self._rerank_url is not None:
            return True
        if self._base_url is not None:
            return False

        return self._rerank_model is not None

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
        rerank_model = self._rerank_model
        if rerank_model is None:
            raise RuntimeError("VoyageAI SDK reranking requires rerank_model")

        for attempt in range(self._retry_attempts):
            try:
                logger.debug(
                    f"VoyageAI reranking {len(documents)} documents with model {rerank_model}"
                )

                result = await asyncio.to_thread(
                    self._client.rerank,
                    query=query,
                    documents=documents,
                    model=rerank_model,
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
            except EmbeddingProviderError:
                raise
            except Exception as e:
                error_type = type(e).__name__
                error_module = type(e).__module__
                category = _classify_voyageai_error(e)

                if category is not None and attempt < self._retry_attempts - 1:
                    base_delay = _CATEGORY_BACKOFFS.get(category, self._retry_delay)
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"VoyageAI reranking failed with {error_module}.{error_type} "
                        f"(attempt {attempt + 1}/{self._retry_attempts}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    if category is not None:
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
        rerank_url = self._rerank_url
        if rerank_url is None:
            raise RuntimeError("HTTP reranking requires rerank_url")

        logger.debug(
            f"HTTP reranking {len(documents)} documents at {rerank_url} "
            f"(format={self._rerank_format})"
        )

        async with httpx.AsyncClient(
            timeout=self._timeout, verify=self._rerank_ssl_verify
        ) as client:
            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            response = await client.post(rerank_url, json=payload, headers=headers)
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
            if not isinstance(item, dict):
                logger.warning(f"Skipping non-dict rerank result: {item!r}")
                continue
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
