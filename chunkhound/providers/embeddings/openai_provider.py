"""OpenAI embedding provider implementation for ChunkHound - concrete embedding provider using OpenAI API."""

import asyncio
import heapq
import math
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any, cast

import httpx
from loguru import logger

from chunkhound.core.config.embedding_config import (
    RERANK_BASE_URL_REQUIRED,
    validate_rerank_configuration,
)
from chunkhound.core.config.openai_utils import is_azure_openai_endpoint
from chunkhound.core.exceptions.core import ValidationError
from chunkhound.core.exceptions.embedding import (
    EmbeddingConfigurationError,
    EmbeddingDimensionError,
)
from chunkhound.interfaces.embedding_provider import EmbeddingConfig, RerankResult

from .batch_utils import handle_token_limit_error
from .shared_utils import l2_normalize

try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    openai = None  # type: ignore
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available - install with: uv pip install openai")


# Qwen3 model configuration for Ollama/OpenAI-compatible endpoints
# Research: https://www.baseten.co/blog/day-zero-benchmarks-for-qwen-3
# Batch sizes optimized for throughput on GPU inference (A100 baseline)
# Rerank limits: 32-128 per batch based on model size to prevent OOM
QWEN_MODEL_CONFIG: dict[str, dict[str, Any]] = {
    # Qwen3 Embedding Models (via Ollama/OpenAI-compatible endpoints)
    # Native dimensions: 0.6B=1024, 4B=2560, 8B=4096
    # All support Matryoshka Representation Learning (MRL) for dimension reduction
    # Batch sizes balanced for GPU memory and throughput
    "dengcao/Qwen3-Embedding-0.6B:Q5_K_M": {
        "max_tokens_per_batch": 200000,  # Conservative for Q5_K_M quantization
        "max_texts_per_batch": 512,  # Smallest model: highest throughput
        "context_length": 8192,
        "max_rerank_batch": 128,  # Smallest reranker: largest batches
        "dims": 1024,
        "native_dims": 1024,
        "matryoshka": True,
        "min_dims": 32,
        "distance": "cosine",
    },
    "qwen3-embedding-0.6b": {
        "max_tokens_per_batch": 200000,
        "max_texts_per_batch": 512,
        "context_length": 8192,
        "max_rerank_batch": 128,
        "dims": 1024,
        "native_dims": 1024,
        "matryoshka": True,
        "min_dims": 32,
        "distance": "cosine",
    },
    "dengcao/Qwen3-Embedding-4B:Q5_K_M": {
        "max_tokens_per_batch": 150000,
        "max_texts_per_batch": 256,  # Medium model: balanced speed/memory
        "context_length": 8192,
        "max_rerank_batch": 96,  # Medium reranker batch size
        "dims": 2560,
        "native_dims": 2560,
        "matryoshka": True,
        "min_dims": 32,
        "distance": "cosine",
    },
    "qwen3-embedding-4b": {
        "max_tokens_per_batch": 150000,
        "max_texts_per_batch": 256,
        "context_length": 8192,
        "max_rerank_batch": 96,
        "dims": 2560,
        "native_dims": 2560,
        "matryoshka": True,
        "min_dims": 32,
        "distance": "cosine",
    },
    "dengcao/Qwen3-Embedding-8B:Q5_K_M": {
        "max_tokens_per_batch": 100000,
        "max_texts_per_batch": 128,  # Largest model: conservative for memory
        "context_length": 8192,
        "max_rerank_batch": 64,  # Largest reranker: smallest batches
        "dims": 4096,
        "native_dims": 4096,
        "matryoshka": True,
        "min_dims": 32,
        "distance": "cosine",
    },
    "qwen3-embedding-8b": {
        "max_tokens_per_batch": 100000,
        "max_texts_per_batch": 128,
        "context_length": 8192,
        "max_rerank_batch": 64,
        "dims": 4096,
        "native_dims": 4096,
        "matryoshka": True,
        "min_dims": 32,
        "distance": "cosine",
    },
    # Qwen3 Reranker Models
    # Batch sizes based on research: 32-128 for GPU inference
    # Conservative values to prevent OOM on large result sets
    "fireworks/qwen3-reranker-0.6b": {
        "max_tokens_per_batch": 100000,
        "max_texts_per_batch": 256,
        "context_length": 131072,  # jina-reranker-v3 context window
        "max_rerank_batch": 128,  # Higher for smallest model
    },
    "qwen3-reranker-0.6b": {
        "max_tokens_per_batch": 100000,
        "max_texts_per_batch": 256,
        "context_length": 131072,
        "max_rerank_batch": 128,
    },
    "fireworks/qwen3-reranker-4b": {
        "max_tokens_per_batch": 80000,
        "max_texts_per_batch": 128,
        "context_length": 32000,
        "max_rerank_batch": 96,
    },
    "qwen3-reranker-4b": {
        "max_tokens_per_batch": 80000,
        "max_texts_per_batch": 128,
        "context_length": 32000,
        "max_rerank_batch": 96,
    },
    "fireworks/qwen3-reranker-8b": {
        "max_tokens_per_batch": 60000,
        "max_texts_per_batch": 64,
        "context_length": 32000,
        "max_rerank_batch": 64,
    },
    "qwen3-reranker-8b": {
        "max_tokens_per_batch": 60000,
        "max_texts_per_batch": 64,
        "context_length": 32000,
        "max_rerank_batch": 64,
    },
}


def _validate_qwen_model_config() -> None:
    """Validate QWEN_MODEL_CONFIG structure at module load time.

    Raises:
        ValueError: If configuration is invalid
    """
    required_fields = {
        "max_tokens_per_batch",
        "max_texts_per_batch",
        "context_length",
        "max_rerank_batch",
    }

    for model_name, config in QWEN_MODEL_CONFIG.items():
        # Check all required fields present
        missing_fields = required_fields - set(config.keys())
        if missing_fields:
            raise ValueError(
                f"Qwen model '{model_name}' missing required fields: {missing_fields}"
            )

        # Validate batch sizes are positive integers
        if config["max_texts_per_batch"] <= 0:
            raise ValueError(
                f"Qwen model '{model_name}' has invalid max_texts_per_batch: "
                f"{config['max_texts_per_batch']} (must be > 0)"
            )

        if config["max_rerank_batch"] <= 0:
            raise ValueError(
                f"Qwen model '{model_name}' has invalid max_rerank_batch: "
                f"{config['max_rerank_batch']} (must be > 0)"
            )

        # Validate token limits are reasonable (between 1K and 10M)
        if not (1000 <= config["max_tokens_per_batch"] <= 10_000_000):
            raise ValueError(
                f"Qwen model '{model_name}' has unrealistic max_tokens_per_batch: "
                f"{config['max_tokens_per_batch']} (should be 1K-10M)"
            )

        # Validate context length is reasonable
        if not (512 <= config["context_length"] <= 1_000_000):
            raise ValueError(
                f"Qwen model '{model_name}' has unrealistic context_length: "
                f"{config['context_length']} (should be 512-1M)"
            )


# Validate configuration at module load
_validate_qwen_model_config()


def _normalize_qwen_model_name(model: str) -> str:
    """Strip provider prefixes from Qwen model names for config lookup.

    Handles: fireworks/qwen3-embedding-4b → qwen3-embedding-4b
    """
    prefixes = ("fireworks/", "ollama/", "together/")
    model_lower = model.lower()
    for prefix in prefixes:
        if model_lower.startswith(prefix):
            return model[len(prefix):]
    return model


class OpenAIEmbeddingProvider:
    """OpenAI embedding provider using text-embedding-3-small by default.

    Thread Safety:
        This provider is thread-safe and stateless. Multiple concurrent calls to
        embed() and rerank() are safe. The underlying OpenAI client (httpx-based)
        handles concurrent requests properly.

        Note: Provider instances should not share mutable state. Each instance
        maintains its own client and configuration, making concurrent operations
        on the same instance safe.
    """

    # Recommended concurrent batches for OpenAI API
    # Conservative value (8) accounts for tier-based rate limits:
    # - Free tier: 3 RPM, 150,000 TPM
    # - Tier 1: 500 RPM, 200,000 TPM
    # - Tier 2+: 5,000+ RPM
    # Lower concurrency prevents rate limit errors across all tiers
    RECOMMENDED_CONCURRENCY = 8

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "text-embedding-3-small",
        rerank_model: str | None = None,
        rerank_url: str = "/rerank",
        rerank_format: str = "auto",
        batch_size: int = 100,
        timeout: int = 30,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        max_tokens: int | None = None,
        rerank_batch_size: int | None = None,
        output_dims: int | None = None,
        client_side_truncation: bool = False,
        api_version: str | None = None,
        azure_endpoint: str | None = None,
        azure_deployment: str | None = None,
    ):
        """Initialize OpenAI embedding provider.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url: Base URL for OpenAI API (defaults to OPENAI_BASE_URL env var)
            model: Model name to use for embeddings
            rerank_model: Model name to use for reranking (optional for TEI format)
            rerank_url: Rerank endpoint URL (defaults to /rerank)
            rerank_format: Reranking API format - 'cohere', 'tei', or 'auto' (default: 'auto')
            batch_size: Maximum batch size for API requests
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts for failed requests
            retry_delay: Delay between retry attempts
            max_tokens: Maximum tokens per request (if applicable)
            rerank_batch_size: Max documents per rerank batch (overrides model defaults, bounded by model caps)
            output_dims: Output embedding dims for matryoshka models (None = native)
            client_side_truncation: Truncate embeddings client-side instead of using API dimensions parameter
            api_version: Azure OpenAI API version (e.g., '2024-02-01')
            azure_endpoint: Azure OpenAI endpoint URL
            azure_deployment: Azure OpenAI deployment name
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package not available. Install with: uv pip install openai"
            )

        # API key and base URL should be provided via config, not env vars
        self._api_key = api_key
        self._base_url = base_url
        self._model = model

        # Azure OpenAI configuration
        self._api_version = api_version
        self._azure_endpoint = azure_endpoint
        self._azure_deployment = azure_deployment
        self._rerank_model = rerank_model
        self._rerank_url = rerank_url
        self._rerank_format = rerank_format
        self._detected_rerank_format: str | None = (
            None  # Cache for auto-detected format
        )
        self._format_detection_lock = asyncio.Lock()  # Protect format detection cache
        self._batch_size = batch_size
        self._timeout = timeout
        self._retry_attempts = retry_attempts
        self._retry_delay = retry_delay
        self._max_tokens = max_tokens
        self._rerank_batch_size = rerank_batch_size
        self._output_dims = output_dims
        self._client_side_truncation = client_side_truncation

        # Validate rerank configuration at initialization (fail-fast)
        # Match config validation logic: check if reranking is enabled
        is_using_reranking = rerank_model or (rerank_format == "tei" and rerank_url)
        if is_using_reranking or rerank_format == "cohere":
            validate_rerank_configuration(
                provider="openai",
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

        # Configure Qwen-specific batch sizes (extracted for clarity)
        self._configure_qwen_batch_sizes(model, rerank_model, batch_size)

        # Model-specific configuration for OpenAI models
        self._model_config = {
            "text-embedding-3-small": {
                "dims": 1536,
                "native_dims": 1536,
                "matryoshka": True,
                "min_dims": 1,
                "distance": "cosine",
                "max_tokens": 8191,
            },
            "text-embedding-3-large": {
                "dims": 3072,
                "native_dims": 3072,
                "matryoshka": True,
                "min_dims": 1,
                "distance": "cosine",
                "max_tokens": 8191,
            },
            "text-embedding-ada-002": {
                "dims": 1536,
                "native_dims": 1536,
                "matryoshka": False,
                "distance": "cosine",
                "max_tokens": 8191,
            },
        }

        # Validate output_dims if specified
        if output_dims is not None:
            model_cfg = self._get_model_config()
            if model_cfg:
                if not model_cfg.get("matryoshka", False):
                    raise EmbeddingConfigurationError(
                        f"Model {model} does not support matryoshka dimensions"
                    )
                min_dims = cast(int, model_cfg.get("min_dims", 1))
                native_dims = cast(int, model_cfg.get("native_dims", model_cfg.get("dims", 1536)))
                if not (min_dims <= output_dims <= native_dims):
                    raise EmbeddingConfigurationError(
                        f"output_dims {output_dims} out of range "
                        f"[{min_dims}, {native_dims}] for {model}"
                    )

        # Usage statistics
        self._usage_stats = {
            "requests_made": 0,
            "tokens_used": 0,
            "embeddings_generated": 0,
            "errors": 0,
        }

        # Initialize OpenAI client lazily to avoid TaskGroup errors on Ubuntu
        # Creating AsyncOpenAI in __init__ can fail when no event loop is running
        self._client = None
        self._client_initialized = False

    def _configure_qwen_batch_sizes(
        self, model: str, rerank_model: str | None, batch_size: int
    ) -> None:
        """Configure Qwen-specific batch sizes if Qwen models detected.

        Detects Qwen embedding and reranker models and applies model-specific
        batch size limits to prevent OOM errors. Follows VoyageAI pattern.

        Args:
            model: Embedding model name
            rerank_model: Optional reranker model name
            batch_size: User-requested batch size
        """
        # Detect Qwen models
        qwen_config = None
        model_lower = model.lower()
        rerank_model_lower = (rerank_model or "").lower()

        # Normalize model names (strip provider prefixes like fireworks/)
        normalized_model = _normalize_qwen_model_name(model)
        normalized_rerank = _normalize_qwen_model_name(rerank_model) if rerank_model else ""

        # Check if embedding model is a Qwen model
        if "qwen" in model_lower or normalized_model in QWEN_MODEL_CONFIG:
            qwen_config = QWEN_MODEL_CONFIG.get(normalized_model)
            if qwen_config:
                logger.info(f"Detected Qwen embedding model: {model}")

        # Check if rerank model is a Qwen model
        qwen_rerank_config = None
        if rerank_model and (
            "qwen" in rerank_model_lower or normalized_rerank in QWEN_MODEL_CONFIG
        ):
            qwen_rerank_config = QWEN_MODEL_CONFIG.get(normalized_rerank)
            if qwen_rerank_config:
                logger.info(f"Detected Qwen reranker model: {rerank_model}")

        # Apply Qwen batch size limits if detected
        if qwen_config:
            # Apply min(user_batch_size, model_max_batch) pattern from VoyageAI
            effective_batch_size = min(batch_size, qwen_config["max_texts_per_batch"])
            if effective_batch_size < batch_size:
                logger.info(
                    f"Limiting batch size to {effective_batch_size} "
                    f"(model max: {qwen_config['max_texts_per_batch']})"
                )
            self._batch_size = effective_batch_size
            self._qwen_model_config = qwen_config
        else:
            self._batch_size = batch_size
            self._qwen_model_config = None

        # Store rerank config separately for get_max_rerank_batch_size()
        self._qwen_rerank_config = qwen_rerank_config

    async def _ensure_client(self) -> None:
        """Ensure the OpenAI client is initialized (must be called from async context)."""
        if self._client is not None and self._client_initialized:
            return

        if not OPENAI_AVAILABLE or openai is None:
            raise RuntimeError(
                "OpenAI library is not available. Install with: pip install openai"
            )

        # Check if using Azure OpenAI
        if is_azure_openai_endpoint(self._azure_endpoint):
            await self._ensure_azure_client()
            return

        # Only require API key for official OpenAI API
        from chunkhound.core.config.openai_utils import is_official_openai_endpoint

        is_openai_official = is_official_openai_endpoint(self._base_url)
        if is_openai_official and not self._api_key:
            raise ValueError("OpenAI API key is required for official OpenAI API")

        # Configure client options for custom endpoints
        api_key_value = self._api_key
        if not is_openai_official and not api_key_value:
            # OpenAI client requires a string value, provide placeholder for custom endpoints
            api_key_value = "not-required"

        client_kwargs = {"api_key": api_key_value, "timeout": self._timeout}

        if self._base_url:
            client_kwargs["base_url"] = self._base_url

            # For custom endpoints (non-OpenAI), disable SSL verification
            # These often use self-signed certificates (e.g., corporate servers, Ollama)
            if not is_openai_official:
                import httpx

                # Create httpx client with SSL verification disabled
                http_client = httpx.AsyncClient(
                    timeout=httpx.Timeout(timeout=self._timeout),
                    verify=False,  # Disable SSL for custom endpoints
                )
                client_kwargs["http_client"] = http_client

                logger.debug(
                    f"SSL verification disabled for custom endpoint: {self._base_url}"
                )

        # IMPORTANT: Create the client in async context to avoid TaskGroup errors on Ubuntu
        # This ensures the event loop is running when the client initializes its httpx instance
        self._client = openai.AsyncOpenAI(**client_kwargs)
        self._client_initialized = True

    async def _ensure_azure_client(self) -> None:
        """Initialize Azure OpenAI client.

        Azure OpenAI uses a different client class (AzureOpenAI) with specific
        parameters for endpoint, API version, and deployment.
        """
        if not self._api_key:
            raise ValueError("Azure OpenAI API key is required")

        if not self._api_version:
            raise ValueError("Azure OpenAI API version is required")

        # azure_endpoint is validated by is_azure_openai_endpoint before this method is called
        if not self._azure_endpoint:
            raise ValueError("Azure endpoint is required for Azure OpenAI")

        logger.debug(
            f"Creating Azure OpenAI client: endpoint={self._azure_endpoint}, "
            f"api_version={self._api_version}, deployment={self._azure_deployment}"
        )

        # AzureOpenAI client has different constructor parameters
        self._client = openai.AsyncAzureOpenAI(
            api_key=self._api_key,
            api_version=self._api_version,
            azure_endpoint=self._azure_endpoint,
            timeout=self._timeout,
        )
        self._client_initialized = True

    @property
    def name(self) -> str:
        """Provider name."""
        # Return azure_openai for Azure endpoints to distinguish in logs/metrics
        if is_azure_openai_endpoint(self._azure_endpoint):
            return "azure_openai"
        return "openai"

    @property
    def model(self) -> str:
        """Model name."""
        return self._model

    def _get_model_config(self) -> dict[str, Any] | None:
        """Get model config from OpenAI models or Qwen models."""
        if self._model in self._model_config:
            return self._model_config[self._model]

        # Normalize Qwen model names (strip provider prefixes)
        normalized = _normalize_qwen_model_name(self._model)

        if normalized in QWEN_MODEL_CONFIG:
            return QWEN_MODEL_CONFIG[normalized]
        # Case-insensitive match for Qwen models
        normalized_lower = normalized.lower()
        for key, config in QWEN_MODEL_CONFIG.items():
            if key.lower() == normalized_lower:
                return config
        return None

    def _get_deployment_model(self) -> str:
        """Get the model/deployment name for API requests.

        For Azure OpenAI, this returns the deployment name if specified,
        otherwise falls back to the model name (Azure deployments can use
        model name as deployment name).

        For standard OpenAI, this returns the model name.
        """
        if is_azure_openai_endpoint(self._azure_endpoint):
            # Azure: prefer explicit deployment, fall back to model name
            # (Azure allows using model name as deployment name)
            return self._azure_deployment or self._model
        return self._model

    @property
    def dims(self) -> int:
        """Actual output dimension (reflects matryoshka config if set)."""
        if self._output_dims is not None:
            return self._output_dims
        model_cfg = self._get_model_config()
        if model_cfg and "dims" in model_cfg:
            return cast(int, model_cfg["dims"])
        return 1536  # Default for unknown models

    @property
    def native_dims(self) -> int:
        """Model's full/native embedding dimension."""
        model_cfg = self._get_model_config()
        if model_cfg:
            return cast(int, model_cfg.get("native_dims", model_cfg.get("dims", 1536)))
        return 1536

    @property
    def supported_dimensions(self) -> list[int]:
        """List of valid output dimensions for this model."""
        model_cfg = self._get_model_config()
        if model_cfg and model_cfg.get("matryoshka", False):
            min_dims = cast(int, model_cfg.get("min_dims", 1))
            native = cast(int, model_cfg.get("native_dims", model_cfg.get("dims", 1536)))
            return list(range(min_dims, native + 1))
        return [self.native_dims]

    def supports_matryoshka(self) -> bool:
        """True if model supports variable output dimensions."""
        model_cfg = self._get_model_config()
        return bool(model_cfg.get("matryoshka", False)) if model_cfg else False

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
        """Distance metric."""
        model_cfg = self._get_model_config()
        if model_cfg and "distance" in model_cfg:
            return cast(str, model_cfg["distance"])
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
            base_url=self._base_url,
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
        return self._base_url or "https://api.openai.com/v1"

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
        if not self._client:
            await self._ensure_client()

        # Skip API key validation during initialization to avoid TaskGroup errors
        # API key validation will happen on first actual embedding request

    async def shutdown(self) -> None:
        """Shutdown the embedding provider and cleanup resources."""
        if self._client:
            await self._client.close()
            self._client = None
        logger.info("OpenAI embedding provider shutdown")

    def is_available(self) -> bool:
        """Check if the provider is available and properly configured."""
        if not OPENAI_AVAILABLE:
            return False

        # Azure OpenAI always requires API key and api_version
        if is_azure_openai_endpoint(self._azure_endpoint):
            return self._api_key is not None and self._api_version is not None

        # Import the utility function (following existing pattern)
        from chunkhound.core.config.openai_utils import is_official_openai_endpoint

        # Use the same logic as _ensure_client() and config validation
        if is_official_openai_endpoint(self._base_url):
            return self._api_key is not None
        else:
            # Custom endpoints don't require API key
            return True

    async def health_check(self) -> dict[str, Any]:
        """Perform health check and return status information."""
        status = {
            "provider": self.name,
            "model": self.model,
            "available": self.is_available(),
            "api_key_configured": self._api_key is not None,
            "client_initialized": self._client is not None,
            "errors": [],
        }

        if not self.is_available():
            if not OPENAI_AVAILABLE:
                status["errors"].append("OpenAI package not installed")
            if not self._api_key:
                status["errors"].append("API key not configured")
            if not self._client:
                status["errors"].append("Client not initialized")
            return status

        try:
            # Test API connectivity with a small embedding
            test_embedding = await self.embed_single("test")
            if len(test_embedding) == self.dims:
                status["connectivity"] = "ok"
            else:
                status["errors"].append(
                    f"Unexpected embedding dimensions: {len(test_embedding)} != {self.dims}"
                )
        except Exception as e:
            status["errors"].append(f"API connectivity test failed: {str(e)}")

        return status

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        if not texts:
            return []

        validated_texts = self.validate_texts(texts)

        try:
            # Always use token-aware batching
            return await self.embed_batch(validated_texts)

        except Exception as e:
            # CRITICAL: Log EVERY exception that passes through here to trace execution path
            logger.error(
                f"[DEBUG-TRACE] Exception caught in OpenAI embed() method: {type(e).__name__}: {str(e)[:200]}"
            )
            self._usage_stats["errors"] += 1
            # Log details of oversized chunks for root cause analysis
            text_sizes = [len(text) for text in validated_texts]
            total_chars = sum(text_sizes)
            max_chars = max(text_sizes) if text_sizes else 0

            # Find and log oversized chunks with their content preview
            oversized_chunks = []
            for i, text in enumerate(validated_texts):
                if (
                    len(text) > 100000
                ):  # Chunks over 100k chars are definitely problematic
                    preview = text[:200] + "..." if len(text) > 200 else text
                    oversized_chunks.append(
                        f"#{i}: {len(text)} chars, starts: {preview}"
                    )

            if oversized_chunks:
                logger.error(
                    "[OpenAI-Provider] OVERSIZED CHUNKS FOUND:\n"
                    + "\n".join(oversized_chunks[:3])
                )  # Limit to first 3

            logger.error(
                f"[OpenAI-Provider] Failed to generate embeddings (texts: {len(validated_texts)}, total_chars: {total_chars}, max_chars: {max_chars}): {e}"
            )

            # Add debug logging to trace the error
            debug_file = "/tmp/chunkhound_openai_debug.log"
            try:
                with open(debug_file, "a") as f:
                    f.write(
                        f"[{datetime.now().isoformat()}] OPENAI-PROVIDER ERROR: texts={len(validated_texts)}, max_chars={max_chars}, error={e}\n"
                    )
                    f.flush()
            except OSError:
                pass  # Debug logging is best-effort, OK to fail silently

            raise

    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        embeddings = await self.embed([text])
        return embeddings[0] if embeddings else []

    async def embed_batch(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        """Generate embeddings in batches with token-aware sizing."""
        if not texts:
            return []

        # Use token-aware batching
        all_embeddings = []
        current_batch = []
        current_tokens = 0
        token_limit = self.get_model_token_limit() - 100  # Safety margin

        for text in texts:
            # Handle individual texts that exceed token limit
            text_tokens = self.estimate_tokens(text)
            if text_tokens > token_limit:
                # Process current batch if not empty
                if current_batch:
                    batch_embeddings = await self._embed_batch_internal(current_batch)
                    all_embeddings.extend(batch_embeddings)
                    current_batch = []
                    current_tokens = 0

                # Split oversized text and process chunks
                chunks = self.chunk_text_by_tokens(text, token_limit)
                for chunk in chunks:
                    chunk_embedding = await self._embed_batch_internal([chunk])
                    all_embeddings.extend(chunk_embedding)
                continue

            # Check if adding this text would exceed token limit
            if current_tokens + text_tokens > token_limit and current_batch:
                # Process current batch
                batch_embeddings = await self._embed_batch_internal(current_batch)
                all_embeddings.extend(batch_embeddings)
                current_batch = []
                current_tokens = 0

            current_batch.append(text)
            current_tokens += text_tokens

        # Process remaining batch
        if current_batch:
            batch_embeddings = await self._embed_batch_internal(current_batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def embed_streaming(self, texts: list[str]) -> AsyncIterator[list[float]]:
        """Generate embeddings with streaming results."""
        for text in texts:
            embedding = await self.embed_single(text)
            yield embedding

    async def _embed_batch_internal(self, texts: list[str]) -> list[list[float]]:
        """Internal method to embed a batch of texts."""
        await self._ensure_client()
        if not self._client:
            raise RuntimeError("OpenAI client not initialized")

        for attempt in range(self._retry_attempts):
            try:
                logger.debug(
                    f"Generating embeddings for {len(texts)} texts (attempt {attempt + 1})"
                )

                # Pass dimensions parameter for matryoshka models when output_dims is set
                # Skip dimensions param if client_side_truncation is enabled
                if self._output_dims is not None and not self._client_side_truncation:
                    response = await self._client.embeddings.create(
                        model=self._get_deployment_model(),
                        input=texts,
                        dimensions=self._output_dims,
                        timeout=self._timeout,
                    )
                else:
                    response = await self._client.embeddings.create(
                        model=self._get_deployment_model(),
                        input=texts,
                        timeout=self._timeout,
                    )

                # Extract embeddings from response, sorted by original input order
                # OpenAI API does not guarantee response order - each data object
                # has an 'index' field indicating its position in the input array
                embeddings: list[list[float] | None] = [None] * len(texts)
                for data in response.data:
                    embedding = data.embedding
                    # Client-side truncation + L2 normalize
                    if self._client_side_truncation and self._output_dims is not None:
                        embedding = l2_normalize(embedding[: self._output_dims])
                    embeddings[data.index] = embedding

                # Validate all indices were filled (defensive check)
                if None in embeddings:
                    missing = [i for i, e in enumerate(embeddings) if e is None]
                    raise RuntimeError(
                        f"OpenAI API returned incomplete embeddings, missing indices: {missing}"
                    )

                # Update usage statistics
                self._usage_stats["requests_made"] += 1
                self._usage_stats["embeddings_generated"] += len(embeddings)
                if hasattr(response, "usage") and response.usage:
                    self._usage_stats["tokens_used"] += response.usage.total_tokens

                logger.debug(f"Successfully generated {len(embeddings)} embeddings")
                return cast(list[list[float]], embeddings)

            except Exception as rate_error:
                if (
                    openai
                    and hasattr(openai, "RateLimitError")
                    and isinstance(rate_error, openai.RateLimitError)
                ):
                    logger.warning(
                        f"Rate limit exceeded, retrying in {self._retry_delay * (attempt + 1)} seconds"
                    )
                    if attempt < self._retry_attempts - 1:
                        await asyncio.sleep(self._retry_delay * (attempt + 1))
                        continue
                    else:
                        raise
                elif (
                    openai
                    and hasattr(openai, "BadRequestError")
                    and isinstance(rate_error, openai.BadRequestError)
                ):
                    # Handle token limit exceeded errors
                    error_message = str(rate_error)
                    if (
                        "maximum context length" in error_message
                        and "tokens" in error_message
                    ) or (
                        "tokens" in error_message
                        and "max" in error_message
                        and "per request" in error_message
                    ):
                        total_tokens = self.estimate_batch_tokens(texts)
                        token_limit = (
                            self.get_model_token_limit() - 100
                        )  # Safety margin

                        return await handle_token_limit_error(
                            texts=texts,
                            total_tokens=total_tokens,
                            token_limit=token_limit,
                            embed_function=self._embed_batch_internal,
                            chunk_text_function=self.chunk_text_by_tokens,
                            single_text_fallback=True,
                        )
                    else:
                        raise
                elif (
                    openai
                    and hasattr(openai, "APITimeoutError")
                    and isinstance(
                        rate_error, (openai.APITimeoutError, openai.APIConnectionError)
                    )
                ):
                    # Log detailed connection error information
                    error_details = {
                        "error_type": type(rate_error).__name__,
                        "error_message": str(rate_error),
                        "base_url": self._base_url,
                        "model": self._model,
                        "timeout": self._timeout,
                        "attempt": attempt + 1,
                        "max_attempts": self._retry_attempts,
                    }
                    if hasattr(rate_error, "response"):
                        error_details["response_status"] = getattr(
                            rate_error.response, "status_code", None
                        )
                        error_details["response_headers"] = dict(
                            getattr(rate_error.response, "headers", {})
                        )

                    logger.warning(
                        f"API connection error, retrying in {self._retry_delay} seconds: {error_details}"
                    )
                    if attempt < self._retry_attempts - 1:
                        await asyncio.sleep(self._retry_delay)
                        continue
                    else:
                        raise
                else:
                    raise

        raise RuntimeError(
            f"Failed to generate embeddings after {self._retry_attempts} attempts"
        )

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
                # Basic preprocessing
                cleaned_text = text.strip()
                validated.append(cleaned_text)

        return validated

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for a text."""
        # Conservative estimation: ~3 characters per token for code/technical text
        # This accounts for more punctuation and shorter tokens in code
        return max(1, len(text) // 3)

    def estimate_batch_tokens(self, texts: list[str]) -> int:
        """Estimate total token count for a batch of texts."""
        return sum(self.estimate_tokens(text) for text in texts)

    def get_model_token_limit(self) -> int:
        """Get token limit for current model."""
        if self._model in self._model_config:
            return self._model_config[self._model]["max_tokens"]
        return 8191  # Default limit

    def chunk_text_by_tokens(self, text: str, max_tokens: int) -> list[str]:
        """Split text into chunks by token count."""
        if max_tokens <= 0:
            raise ValidationError(
                "max_tokens", max_tokens, "max_tokens must be positive"
            )

        # Use safety margin to ensure we stay well under token limits
        safety_margin = max(200, max_tokens // 5)  # 20% margin, minimum 200 tokens
        safe_max_tokens = max_tokens - safety_margin
        # Use conservative 3 chars per token for code/technical text
        max_chars = safe_max_tokens * 3

        if len(text) <= max_chars:
            return [text]

        chunks = []
        for i in range(0, len(text), max_chars):
            chunk = text[i : i + max_chars]
            chunks.append(chunk)

        return chunks

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the embedding model."""
        return {
            "provider": self.name,
            "model": self.model,
            "dimensions": self.dims,
            "distance_metric": self.distance,
            "batch_size": self.batch_size,
            "max_tokens": self.max_tokens,
            "supported_models": list(self._model_config.keys()),
        }

    def get_usage_stats(self) -> dict[str, Any]:
        """Get usage statistics."""
        return self._usage_stats.copy()

    def reset_usage_stats(self) -> None:
        """Reset usage statistics."""
        self._usage_stats = {
            "requests_made": 0,
            "tokens_used": 0,
            "embeddings_generated": 0,
            "errors": 0,
        }

    def update_config(self, **kwargs) -> None:
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
        if "api_key" in kwargs:
            self._api_key = kwargs["api_key"]
            # Reset client to force re-initialization with new API key
            self._client = None
            self._client_initialized = False
        if "base_url" in kwargs:
            self._base_url = kwargs["base_url"]
            # Reset client to force re-initialization with new base URL
            self._client = None
            self._client_initialized = False

    def get_supported_distances(self) -> list[str]:
        """Get list of supported distance metrics."""
        return ["cosine", "l2", "ip"]  # OpenAI embeddings work with multiple metrics

    def get_optimal_batch_size(self) -> int:
        """Get optimal batch size for this provider."""
        return self._batch_size

    def get_max_tokens_per_batch(self) -> int:
        """Get maximum tokens per batch for this provider."""
        # Check Qwen config first (higher limits for Ollama endpoints)
        if self._qwen_model_config:
            return self._qwen_model_config["max_tokens_per_batch"]
        # Fall back to OpenAI model config
        if self._model in self._model_config:
            return self._model_config[self._model]["max_tokens"]
        return 8191  # Default OpenAI limit

    async def validate_api_key(self) -> bool:
        """Validate API key with the service."""
        if not self._client or not self._api_key:
            return False

        try:
            # Test with a minimal request
            response = await self._client.embeddings.create(
                model=self._get_deployment_model(), input=["test"], timeout=5
            )
            return (
                len(response.data) == 1 and len(response.data[0].embedding) == self.dims
            )
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return False

    def get_rate_limits(self) -> dict[str, Any]:
        """Get rate limit information."""
        # OpenAI rate limits vary by model and tier
        return {
            "requests_per_minute": "varies by tier",
            "tokens_per_minute": "varies by tier",
            "note": "See OpenAI documentation for current limits",
        }

    def get_request_headers(self) -> dict[str, str]:
        """Get headers for API requests."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": "ChunkHound-OpenAI-Provider",
        }

    def get_max_documents_per_batch(self) -> int:
        """Get maximum documents per batch for OpenAI provider."""
        # Qwen config is already applied to self._batch_size in __init__
        return self._batch_size

    def get_max_rerank_batch_size(self) -> int:
        """Get maximum documents per batch for reranking operations.

        Returns model-specific batch limit for reranking to prevent OOM errors.
        Implements bounded override pattern: user can set batch size, but it's
        clamped to model caps for safety.

        Priority order:
        0. User override (rerank_batch_size) - bounded by model cap below
        1. Rerank model-specific config (Qwen rerankers: 64-128)
        2. Embedding model config (if rerank model matches embedding model)
        3. Conservative default (min of batch_size, 128)

        Returns:
            Maximum number of documents to rerank in a single batch
        """
        # Determine model cap (Priority 1-3: existing logic)
        model_cap = None
        if self._qwen_rerank_config and "max_rerank_batch" in self._qwen_rerank_config:
            # Priority 1: Rerank model-specific config (Qwen models)
            model_cap = self._qwen_rerank_config["max_rerank_batch"]
        elif self._qwen_model_config and "max_rerank_batch" in self._qwen_model_config:
            # Priority 2: Embedding model config (fallback)
            model_cap = self._qwen_model_config["max_rerank_batch"]
        else:
            # Priority 3: Conservative default
            # Research shows 32-128 is optimal for GPU reranking
            model_cap = min(self._batch_size, 128)

        # Priority 0: User override (bounded by model cap)
        if self._rerank_batch_size is not None:
            return min(self._rerank_batch_size, model_cap)

        # Return model cap as default
        return model_cap

    def get_recommended_concurrency(self) -> int:
        """Get recommended number of concurrent batches for OpenAI.

        Returns:
            Conservative concurrency for tier-based rate limits
        """
        return self.RECOMMENDED_CONCURRENCY

    def _resolve_rerank_format(self) -> str:
        """Resolve the reranking format to use for the next request.

        Returns cached detected format if available in auto mode, otherwise
        returns the configured format.

        Thread safety: Reading _detected_rerank_format without a lock is safe
        due to Python's GIL ensuring atomic pointer reads. The write operation
        in _parse_rerank_response uses async lock for proper synchronization.

        Returns:
            Format to use: 'cohere', 'tei', or 'auto'
        """
        if self._detected_rerank_format:
            return self._detected_rerank_format
        return self._rerank_format

    def _build_rerank_payload(
        self, query: str, documents: list[str], top_k: int | None, format_to_use: str
    ) -> dict:
        """Build rerank request payload based on format.

        Args:
            query: Search query
            documents: List of documents to rerank
            top_k: Maximum number of results to return
            format_to_use: Format to use ('cohere', 'tei', or 'auto')

        Returns:
            Request payload dictionary
        """
        if format_to_use == "tei":
            # TEI format: no model in request, uses "texts" field
            logger.debug(f"Using TEI format for reranking {len(documents)} documents")
            return {"query": query, "texts": documents}

        elif format_to_use == "cohere":
            # Cohere format: requires model, uses "documents" field
            # Validation already done in __init__, so we know model is present
            payload = {
                "model": self._rerank_model,
                "query": query,
                "documents": documents,
            }
            if top_k is not None:
                payload["top_n"] = top_k
            logger.debug(
                f"Using Cohere format for reranking {len(documents)} documents with model {self._rerank_model}"
            )
            return payload

        else:  # auto mode
            # Try Cohere first if model is set, otherwise TEI
            if self._rerank_model:
                payload = {
                    "model": self._rerank_model,
                    "query": query,
                    "documents": documents,
                }
                if top_k is not None:
                    payload["top_n"] = top_k
                logger.debug(
                    f"Auto-detecting format, trying Cohere first (model: {self._rerank_model})"
                )
                return payload
            else:
                logger.debug("Auto-detecting format, trying TEI first (no model set)")
                return {"query": query, "texts": documents}

    def supports_reranking(self) -> bool:
        """Check if reranking is supported with current configuration.

        Uses shared validation logic to determine if reranking can be performed.

        Returns:
            True if provider can perform reranking with current config
        """
        if not self._rerank_url:
            return False

        # Use shared validation logic - if validation passes, reranking is supported
        try:
            validate_rerank_configuration(
                provider="openai",
                rerank_format=self._rerank_format,
                rerank_model=self._rerank_model,
                rerank_url=self._rerank_url,
                base_url=self._base_url,
            )
            return True
        except ValueError:
            return False

    async def rerank(
        self, query: str, documents: list[str], top_k: int | None = None
    ) -> list[RerankResult]:
        """Rerank documents using configured rerank model with automatic batch splitting.

        Implements batch splitting to prevent OOM errors on large document sets.
        For Qwen3 rerankers: uses model-specific batch limits (64-128).
        Results are aggregated across batches and sorted by relevance score.

        Supports both Cohere and TEI (Text Embeddings Inference) formats.
        Format can be explicitly set or auto-detected from response.

        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Optional limit on number of results to return

        Returns:
            List of RerankResult objects sorted by relevance score (highest first)
        """
        if not documents:
            return []

        # Get model-specific batch size limit
        batch_size_limit = self.get_max_rerank_batch_size()

        # Single batch case - use original logic for efficiency
        if len(documents) <= batch_size_limit:
            logger.debug(f"Reranking {len(documents)} documents in single batch")
            results = await self._rerank_single_batch(query, documents, top_k)

            # Apply client-side top_k for formats without server-side support (TEI)
            # Cohere includes top_n in request, but we apply this uniformly for consistency
            if top_k is not None and len(results) > top_k:
                # Results from _rerank_single_batch are already sorted descending by score
                results = results[:top_k]
                logger.debug(
                    f"Applied client-side top_k filter: {len(results)} results"
                )

            return results

        # Multiple batches required - split and aggregate
        logger.info(
            f"Splitting {len(documents)} documents into batches of {batch_size_limit} "
            f"for reranking (model: {self._rerank_model})"
        )

        all_results = []
        num_batches = math.ceil(len(documents) / batch_size_limit)
        failed_batches = 0

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size_limit
            end_idx = min(start_idx + batch_size_limit, len(documents))
            batch_documents = documents[start_idx:end_idx]

            logger.debug(
                f"Reranking batch {batch_idx + 1}/{num_batches}: "
                f"documents {start_idx}-{end_idx}"
            )

            # Retry logic for this batch (following VoyageAI pattern)
            batch_results = None
            for attempt in range(self._retry_attempts):
                try:
                    # Rerank this batch without top_k limit (we'll apply globally)
                    batch_results = await self._rerank_single_batch(
                        query, batch_documents, top_k=None
                    )
                    break  # Success - exit retry loop
                except Exception as e:
                    # Classify error as retryable or not
                    error_str = str(e).lower()
                    is_retryable = any(
                        [
                            "timeout" in error_str,
                            "connection" in error_str,
                            "503" in error_str,  # Service unavailable
                            "429" in error_str,  # Rate limit
                        ]
                    )

                    if is_retryable and attempt < self._retry_attempts - 1:
                        # Exponential backoff
                        delay = self._retry_delay * (2**attempt)
                        logger.warning(
                            f"Batch {batch_idx + 1} failed (attempt {attempt + 1}), "
                            f"retrying in {delay}s: {e}"
                        )
                        await asyncio.sleep(delay)
                        continue
                    else:
                        # Last attempt or non-retryable error
                        logger.error(
                            f"Batch {batch_idx + 1} failed after {attempt + 1} attempts: {e}"
                        )
                        # Continue to next batch instead of failing entire operation
                        batch_results = []
                        failed_batches += 1
                        break

            # Process results if batch succeeded
            if batch_results:
                # Adjust indices to be relative to original document list
                # Use immutable pattern instead of mutation
                for result in batch_results:
                    # Validate index is within batch bounds (handles negative indices)
                    if result.index < 0 or result.index >= len(batch_documents):
                        logger.warning(
                            f"Invalid index {result.index} from rerank API "
                            f"(batch size: {len(batch_documents)}), skipping result"
                        )
                        continue

                    # Create new RerankResult with adjusted index
                    adjusted_result = RerankResult(
                        index=result.index + start_idx, score=result.score
                    )
                    all_results.append(adjusted_result)

        # Warn if any batches failed
        if failed_batches > 0:
            logger.warning(
                f"Reranking completed with partial failures: {failed_batches} of "
                f"{num_batches} batches failed. Results may be incomplete."
            )

        # Apply top_k selection efficiently using heapq when beneficial
        if top_k is not None and top_k < len(all_results) * 0.5:
            # Use heap-based selection for better performance when k << n
            # heapq.nlargest is O(n log k) vs sort O(n log n)
            all_results = heapq.nlargest(top_k, all_results, key=lambda r: r.score)
        else:
            # Sort all results when returning most/all of them
            all_results.sort(key=lambda r: r.score, reverse=True)
            if top_k is not None and top_k < len(all_results):
                all_results = all_results[:top_k]

        logger.debug(
            f"Reranked {len(documents)} documents across {num_batches} batches "
            f"({num_batches - failed_batches} succeeded), returning {len(all_results)} results"
        )

        return all_results

    async def _rerank_single_batch(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """Internal method to rerank a single batch of documents with format detection.

        Supports both Cohere and TEI (Text Embeddings Inference) formats.
        Format can be explicitly set or auto-detected from response.

        Args:
            query: The search query
            documents: List of documents to rerank (single batch)
            top_k: Optional limit on number of results

        Returns:
            List of RerankResult objects with indices relative to input documents
        """
        await self._ensure_client()

        # Validate base_url exists for relative URLs (redundant check for safety)
        if (
            not self._rerank_url.startswith(("http://", "https://"))
            and not self._base_url
        ):
            raise ValueError(RERANK_BASE_URL_REQUIRED)

        # Build full rerank endpoint URL
        if self._rerank_url.startswith(("http://", "https://")):
            # Full URL - use as-is for separate reranking service
            rerank_endpoint = self._rerank_url
        else:
            # Relative path - combine with base_url
            base_url = self._base_url.rstrip("/")
            rerank_url = self._rerank_url.lstrip("/")
            rerank_endpoint = f"{base_url}/{rerank_url}"

        # Resolve format and build payload
        format_to_use = self._resolve_rerank_format()
        payload = self._build_rerank_payload(query, documents, top_k, format_to_use)

        try:
            # Make API request with timeout using httpx directly
            # since OpenAI client doesn't support custom endpoints well

            # Apply consistent SSL handling (same pattern as setup wizard and client init)
            from chunkhound.core.config.openai_utils import is_official_openai_endpoint

            client_kwargs = {"timeout": self._timeout}
            if not is_official_openai_endpoint(self._base_url):
                # For custom endpoints, disable SSL verification
                # These often use self-signed certificates (corporate servers, Ollama)
                client_kwargs["verify"] = False
                logger.debug(
                    f"SSL verification disabled for rerank endpoint: {rerank_endpoint}"
                )

            async with httpx.AsyncClient(**client_kwargs) as client:
                headers = {"Content-Type": "application/json"}

                # Add Authorization header if API key is set (required for TEI with --api-key)
                if self._api_key:
                    headers["Authorization"] = f"Bearer {self._api_key}"
                    logger.debug("Added Authorization header for rerank request")

                response = await client.post(
                    rerank_endpoint, json=payload, headers=headers
                )
                response.raise_for_status()
                response_data = response.json()

            # Normalize response format: TEI servers may return bare array or wrapped dict
            # Real TEI servers: [{"index": 0, "score": 0.95}, ...]
            # Cohere/proxies: {"results": [{"index": 0, "relevance_score": 0.95}, ...]}
            if isinstance(response_data, list):
                response_data = {"results": response_data}

            # Check for error response (TEI returns HTTP 200 with error JSON)
            # This happens when the server validates the request after accepting it
            # Note: Only dict responses can have error field (arrays cannot)
            if isinstance(response_data, dict) and "error" in response_data:
                error_msg = response_data.get("error", "Unknown error")
                error_type = response_data.get("error_type", "Unknown")
                raise ValueError(f"Rerank service error ({error_type}): {error_msg}")

            # Parse response with format auto-detection
            rerank_results = await self._parse_rerank_response(
                response_data, format_to_use, num_documents=len(documents)
            )

            # Update usage statistics
            self._usage_stats["requests_made"] += 1
            self._usage_stats["documents_reranked"] = self._usage_stats.get(
                "documents_reranked", 0
            ) + len(documents)

            logger.debug(
                f"Successfully reranked {len(documents)} documents, got {len(rerank_results)} results"
            )
            return rerank_results

        except httpx.ConnectError as e:
            # Connection failed - service not available
            self._usage_stats["errors"] += 1
            logger.error(
                f"Failed to connect to rerank service at {rerank_endpoint}: {e}"
            )
            raise
        except httpx.TimeoutException as e:
            # Request timed out
            self._usage_stats["errors"] += 1
            logger.error(f"Rerank request timed out after {self._timeout}s: {e}")
            raise
        except httpx.HTTPStatusError as e:
            # HTTP error response from service
            self._usage_stats["errors"] += 1
            logger.error(
                f"Rerank service returned error {e.response.status_code}: {e.response.text}"
            )
            raise
        except ValueError as e:
            # Invalid response format
            self._usage_stats["errors"] += 1
            logger.error(f"Invalid rerank response format: {e}")
            raise
        except Exception as e:
            # Unexpected error
            self._usage_stats["errors"] += 1
            logger.error(f"Unexpected error during reranking: {e}")
            raise

    async def _parse_rerank_response(
        self, response_data: dict | list, format_hint: str, num_documents: int
    ) -> list[RerankResult]:
        """Parse rerank response with format auto-detection.

        Thread-safe format detection using async lock to prevent race conditions.
        Validates that returned indices are within bounds of the original document list.

        Supports both wrapped dict format (Cohere/proxies) and bare array format (real TEI servers).
        Bare arrays are normalized to wrapped format before processing.

        Args:
            response_data: JSON response from rerank API (dict or list)
            format_hint: Format hint ('cohere', 'tei', or 'auto')
            num_documents: Number of documents that were sent for reranking

        Returns:
            List of RerankResult objects

        Raises:
            ValueError: If response format is invalid or unrecognized
        """
        # Early validation: check num_documents is reasonable
        if num_documents <= 0:
            logger.warning(
                f"num_documents is {num_documents} (zero or negative), returning empty results"
            )
            return []

        # Validate response has results
        # Note: Bare array responses are normalized to {"results": [...]} before this point
        if "results" not in response_data:
            raise ValueError(
                "Invalid rerank response: missing 'results' field. "
                "Expected dict with 'results' key or bare array (auto-normalized)."
            )

        results = response_data["results"]
        if not isinstance(results, list):
            raise ValueError("Invalid rerank response: 'results' must be a list")

        if not results:
            logger.warning("Rerank response contains empty results list")
            return []

        # Try to detect format from first result
        first_result = results[0]
        if not isinstance(first_result, dict):
            raise ValueError(
                "Invalid rerank response: results must contain dict objects"
            )

        # Detect format based on field names
        has_relevance_score = "relevance_score" in first_result
        has_score = "score" in first_result
        has_index = "index" in first_result

        if not has_index:
            raise ValueError("Invalid rerank response: results must have 'index' field")

        # Determine score field name
        score_field = None
        detected_format = None

        if has_relevance_score:
            score_field = "relevance_score"
            detected_format = "cohere"
        elif has_score:
            score_field = "score"
            detected_format = "tei"
        else:
            raise ValueError(
                "Invalid rerank response: results must have 'relevance_score' or 'score' field"
            )

        # Cache detected format if in auto mode (thread-safe with async lock)
        if format_hint == "auto" and detected_format:
            async with self._format_detection_lock:
                # Double-check pattern: check if another task already detected the format
                if self._detected_rerank_format is None:
                    self._detected_rerank_format = detected_format
                    logger.debug(f"Auto-detected rerank format: {detected_format}")

        # Convert to ChunkHound format with validation
        rerank_results = []
        for i, result in enumerate(results):
            if not isinstance(result, dict):
                logger.warning(f"Skipping invalid result {i}: not a dict")
                continue

            if "index" not in result or score_field not in result:
                logger.warning(
                    f"Skipping result {i}: missing required fields (index, {score_field})"
                )
                continue

            try:
                index = int(result["index"])
                score = float(result[score_field])

                # Validate index is within bounds
                if index < 0:
                    logger.warning(f"Skipping result {i}: negative index {index}")
                    continue

                if index >= num_documents:
                    logger.warning(
                        f"Skipping result {i}: index {index} out of bounds (num_documents={num_documents})"
                    )
                    continue

                rerank_results.append(RerankResult(index=index, score=score))
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping result {i}: invalid data types - {e}")
                continue

        return rerank_results
