"""OpenAI embedding provider implementation for ChunkHound - concrete embedding provider using OpenAI API."""

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

import httpx
from loguru import logger

from chunkhound.core.exceptions.core import ValidationError
from chunkhound.interfaces.embedding_provider import EmbeddingConfig, RerankResult

from .batch_utils import handle_token_limit_error, with_openai_token_handling

try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    openai = None  # type: ignore
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available - install with: uv pip install openai")


class OpenAIEmbeddingProvider:
    """OpenAI embedding provider using text-embedding-3-small by default."""

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
        batch_size: int = 100,
        timeout: int = 30,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        max_tokens: int | None = None,
    ):
        """Initialize OpenAI embedding provider.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url: Base URL for OpenAI API (defaults to OPENAI_BASE_URL env var)
            model: Model name to use for embeddings
            rerank_model: Model name to use for reranking (enables multi-hop search)
            rerank_url: Rerank endpoint URL (defaults to /rerank)
            batch_size: Maximum batch size for API requests
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts for failed requests
            retry_delay: Delay between retry attempts
            max_tokens: Maximum tokens per request (if applicable)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package not available. Install with: uv pip install openai"
            )

        # API key and base URL should be provided via config, not env vars
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._rerank_model = rerank_model
        self._rerank_url = rerank_url
        self._batch_size = batch_size
        self._timeout = timeout
        self._retry_attempts = retry_attempts
        self._retry_delay = retry_delay
        self._max_tokens = max_tokens

        # Model-specific configuration
        self._model_config = {
            "text-embedding-3-small": {
                "dims": 1536,
                "distance": "cosine",
                "max_tokens": 8191,
            },
            "text-embedding-3-large": {
                "dims": 3072,
                "distance": "cosine",
                "max_tokens": 8191,
            },
            "text-embedding-ada-002": {
                "dims": 1536,
                "distance": "cosine",
                "max_tokens": 8191,
            },
        }

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

    async def _ensure_client(self) -> None:
        """Ensure the OpenAI client is initialized (must be called from async context)."""
        if self._client is not None and self._client_initialized:
            return

        if not OPENAI_AVAILABLE or openai is None:
            raise RuntimeError(
                "OpenAI library is not available. Install with: pip install openai"
            )

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

    @property
    def name(self) -> str:
        """Provider name."""
        return "openai"

    @property
    def model(self) -> str:
        """Model name."""
        return self._model

    @property
    def dims(self) -> int:
        """Embedding dimensions."""
        if self._model in self._model_config:
            return self._model_config[self._model]["dims"]
        return 1536  # Default for most OpenAI models

    @property
    def distance(self) -> str:
        """Distance metric."""
        if self._model in self._model_config:
            return self._model_config[self._model]["distance"]
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
            except:
                pass

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

                response = await self._client.embeddings.create(
                    model=self.model, input=texts, timeout=self._timeout
                )

                # Extract embeddings from response
                embeddings = []
                for data in response.data:
                    embeddings.append(data.embedding)

                # Update usage statistics
                self._usage_stats["requests_made"] += 1
                self._usage_stats["embeddings_generated"] += len(embeddings)
                if hasattr(response, "usage") and response.usage:
                    self._usage_stats["tokens_used"] += response.usage.total_tokens

                logger.debug(f"Successfully generated {len(embeddings)} embeddings")
                return embeddings

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

    @with_openai_token_handling()
    async def _embed_batch_simple(self, texts: list[str]) -> list[list[float]]:
        """Simplified embedding method using the token limit decorator.

        This demonstrates how future providers can use the decorator approach.
        """
        if not self._client:
            raise RuntimeError("OpenAI client not initialized")

        logger.debug(f"Generating embeddings for {len(texts)} texts")

        response = await self._client.embeddings.create(
            model=self.model, input=texts, timeout=self._timeout
        )

        # Extract embeddings from response
        embeddings = []
        for data in response.data:
            embeddings.append(data.embedding)

        # Update usage statistics
        self._usage_stats["requests_made"] += 1
        self._usage_stats["embeddings_generated"] += len(embeddings)
        if hasattr(response, "usage") and response.usage:
            self._usage_stats["tokens_used"] += response.usage.total_tokens

        logger.debug(f"Successfully generated {len(embeddings)} embeddings")
        return embeddings

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
                model=self.model, input=["test"], timeout=5
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
        return self._batch_size

    def get_recommended_concurrency(self) -> int:
        """Get recommended number of concurrent batches for OpenAI.

        Returns:
            Conservative concurrency for tier-based rate limits
        """
        return self.RECOMMENDED_CONCURRENCY

    def supports_reranking(self) -> bool:
        """Check if reranking is supported."""
        return self._rerank_model is not None

    async def rerank(
        self, query: str, documents: list[str], top_k: int | None = None
    ) -> list[RerankResult]:
        """Rerank documents using configured rerank model."""
        await self._ensure_client()

        # Validate base_url exists for reranking
        if not self._base_url:
            raise ValueError("base_url is required for reranking operations")

        # Build full rerank endpoint URL
        if self._rerank_url.startswith(("http://", "https://")):
            # Full URL - use as-is for separate reranking service
            rerank_endpoint = self._rerank_url
        else:
            # Relative path - combine with base_url
            base_url = self._base_url.rstrip("/")
            rerank_url = self._rerank_url.lstrip("/")
            rerank_endpoint = f"{base_url}/{rerank_url}"

        # Prepare request payload
        payload = {"model": self._rerank_model, "query": query, "documents": documents}
        if top_k is not None:
            payload["top_n"] = top_k

        try:
            logger.debug(
                f"Reranking {len(documents)} documents with model {self._rerank_model} "
                f"at endpoint {rerank_endpoint}"
            )

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
                response = await client.post(
                    rerank_endpoint, json=payload, headers=headers
                )
                response.raise_for_status()
                response_data = response.json()

            # Validate response structure
            if "results" not in response_data:
                raise ValueError("Invalid rerank response: missing 'results' field")

            results = response_data["results"]
            if not isinstance(results, list):
                raise ValueError("Invalid rerank response: 'results' must be a list")

            # Convert to ChunkHound format with validation
            rerank_results = []
            for i, result in enumerate(results):
                if not isinstance(result, dict):
                    logger.warning(f"Skipping invalid result {i}: not a dict")
                    continue

                if "index" not in result or "relevance_score" not in result:
                    logger.warning(f"Skipping result {i}: missing required fields")
                    continue

                try:
                    rerank_results.append(
                        RerankResult(
                            index=int(result["index"]),
                            score=float(result["relevance_score"]),
                        )
                    )
                except (ValueError, TypeError) as e:
                    logger.warning(f"Skipping result {i}: invalid data types - {e}")
                    continue

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
