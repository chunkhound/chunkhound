"""Shared reranking functionality for HTTP-based rerank endpoints (Cohere/TEI)."""

import asyncio
import heapq
import math
from typing import Any

import httpx
from loguru import logger

from chunkhound.core.config.embedding_config import (
    RERANK_BASE_URL_REQUIRED,
    validate_rerank_configuration,
)
from chunkhound.interfaces.embedding_provider import RerankResult


class RerankMixin:
    """Mixin providing HTTP-based reranking for Cohere/TEI compatible endpoints.

    This mixin extracts shared reranking logic used by OpenAI and Mistral providers.
    VoyageAI uses native SDK reranking and doesn't use this mixin.

    Required Provider Attributes:
        The provider class using this mixin must have these attributes:
        - _rerank_url: str
        - _rerank_model: str | None
        - _rerank_format: str (cohere/tei/auto)
        - _rerank_batch_size: int | None
        - _base_url: str | None
        - _api_key: str | None
        - _rerank_api_key: str | None
        - _timeout: int
        - _retry_attempts: int
        - _retry_delay: float
        - _detected_rerank_format: str | None (cache)
        - _format_detection_lock: asyncio.Lock

    Required Provider Methods (hooks):
        - get_max_rerank_batch_size() -> int
        - _get_rerank_client_kwargs() -> dict
    """

    def supports_reranking(self) -> bool:
        """Check if reranking is supported with current configuration.

        Uses shared validation logic to determine if reranking can be performed.

        Returns:
            True if provider can perform reranking with current config
        """
        if not hasattr(self, "_rerank_url") or not self._rerank_url:
            return False

        # Use shared validation logic - if validation passes, reranking is supported
        try:
            # Provider name is expected to be available via self.name property
            provider_name = getattr(self, "name", "unknown")
            validate_rerank_configuration(
                provider=provider_name,
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

            # Retry logic for this batch
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
                for result in batch_results:
                    # Validate index is within batch bounds
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
        # Validate base_url exists for relative URLs
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
            # Get provider-specific httpx client kwargs (for SSL handling, etc.)
            client_kwargs = self._get_rerank_client_kwargs()

            # Make API request with timeout using httpx directly
            async with httpx.AsyncClient(**client_kwargs) as client:
                headers = {"Content-Type": "application/json"}

                # Add Authorization header if API key is set
                # Use dedicated rerank API key if provided, otherwise fall back to provider API key
                rerank_api_key = None
                if hasattr(self, "_rerank_api_key") and self._rerank_api_key:
                    from pydantic import SecretStr

                    if isinstance(self._rerank_api_key, SecretStr):
                        rerank_api_key = self._rerank_api_key.get_secret_value()
                    else:
                        rerank_api_key = str(self._rerank_api_key)
                elif hasattr(self, "_api_key") and self._api_key:
                    rerank_api_key = self._api_key

                if rerank_api_key:
                    headers["Authorization"] = f"Bearer {rerank_api_key}"
                    logger.debug("Added Authorization header for rerank request")

                response = await client.post(
                    rerank_endpoint, json=payload, headers=headers
                )
                response.raise_for_status()
                response_data = response.json()

            # Normalize response format: TEI servers may return bare array or wrapped dict
            if isinstance(response_data, list):
                response_data = {"results": response_data}

            # Check for error response (TEI returns HTTP 200 with error JSON)
            if isinstance(response_data, dict) and "error" in response_data:
                error_msg = response_data.get("error", "Unknown error")
                error_type = response_data.get("error_type", "Unknown")
                raise ValueError(f"Rerank service error ({error_type}): {error_msg}")

            # Type guard: ensure we have a dict at this point
            if not isinstance(response_data, dict):
                raise ValueError("Invalid rerank response: expected dict or list")

            # Parse response with format auto-detection
            rerank_results = await self._parse_rerank_response(
                response_data, format_to_use, num_documents=len(documents)
            )

            # Update usage statistics if provider has them
            if hasattr(self, "_usage_stats"):
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
            if hasattr(self, "_usage_stats"):
                self._usage_stats["errors"] += 1
            logger.error(
                f"Failed to connect to rerank service at {rerank_endpoint}: {e}"
            )
            raise
        except httpx.TimeoutException as e:
            # Request timed out
            if hasattr(self, "_usage_stats"):
                self._usage_stats["errors"] += 1
            logger.error(f"Rerank request timed out after {self._timeout}s: {e}")
            raise
        except httpx.HTTPStatusError as e:
            # HTTP error response from service
            if hasattr(self, "_usage_stats"):
                self._usage_stats["errors"] += 1
            logger.error(
                f"Rerank service returned error {e.response.status_code}: {e.response.text}"
            )
            raise
        except ValueError as e:
            # Invalid response format
            if hasattr(self, "_usage_stats"):
                self._usage_stats["errors"] += 1
            logger.error(f"Invalid rerank response format: {e}")
            raise
        except Exception as e:
            # Unexpected error
            if hasattr(self, "_usage_stats"):
                self._usage_stats["errors"] += 1
            logger.error(f"Unexpected error during reranking: {e}")
            raise

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
    ) -> dict[str, Any]:
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

        elif format_to_use == "openarc":
            # OpenArc format: requires model, uses "documents" field
            logger.debug(
                f"Using OpenArc format for reranking {len(documents)} documents with model {self._rerank_model}"
            )
            return {"model": self._rerank_model, "query": query, "documents": documents}

        elif format_to_use == "cohere":
            # Cohere format: requires model, uses "documents" field
            payload: dict[str, Any] = {
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
                auto_payload: dict[str, Any] = {
                    "model": self._rerank_model,
                    "query": query,
                    "documents": documents,
                }
                if top_k is not None:
                    auto_payload["top_n"] = top_k
                logger.debug(
                    f"Auto-detecting format, trying Cohere first (model: {self._rerank_model})"
                )
                return auto_payload
            else:
                logger.debug("Auto-detecting format, trying TEI first (no model set)")
                return {"query": query, "texts": documents}

    async def _parse_rerank_response(
        self,
        response_data: dict[str, Any] | list[Any],
        format_hint: str,
        num_documents: int,
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

        # Check for OpenArc format: {"data": [{"ranked_documents": [{index, score, document}]}]}
        if isinstance(response_data, dict) and "data" in response_data:
            data_list = response_data["data"]
            if (
                isinstance(data_list, list)
                and data_list
                and "ranked_documents" in data_list[0]
            ):
                ranked_docs = data_list[0]["ranked_documents"]
                if not isinstance(ranked_docs, list):
                    raise ValueError(
                        "Invalid OpenArc response: 'ranked_documents' must be a list"
                    )

                # Cache detected format if in auto mode (thread-safe with async lock)
                if format_hint == "auto":
                    async with self._format_detection_lock:
                        # Double-check pattern: check if another task already detected the format
                        if self._detected_rerank_format is None:
                            self._detected_rerank_format = "openarc"
                            logger.debug("Auto-detected rerank format: openarc")

                # Parse OpenArc results with validation
                rerank_results = []
                for i, result in enumerate(ranked_docs):
                    if not isinstance(result, dict):
                        logger.warning(
                            f"Skipping invalid OpenArc result {i}: not a dict"
                        )
                        continue

                    if "index" not in result or "score" not in result:
                        logger.warning(
                            f"Skipping OpenArc result {i}: missing required fields (index, score)"
                        )
                        continue

                    try:
                        index = int(result["index"])
                        score = float(result["score"])

                        # Validate index is within bounds
                        if index < 0:
                            logger.warning(
                                f"Skipping OpenArc result {i}: negative index {index}"
                            )
                            continue

                        if index >= num_documents:
                            logger.warning(
                                f"Skipping OpenArc result {i}: index {index} out of bounds (num_documents={num_documents})"
                            )
                            continue

                        rerank_results.append(RerankResult(index=index, score=score))
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            f"Skipping OpenArc result {i}: invalid data types - {e}"
                        )
                        continue

                return rerank_results

        # Validate response has results
        # Note: Bare array responses are normalized to {"results": [...]} before this point
        if not isinstance(response_data, dict) or "results" not in response_data:
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

    # Abstract methods that providers must implement
    def get_max_rerank_batch_size(self) -> int:
        """Get maximum documents per batch for reranking operations.

        Provider-specific implementation required.
        Should return model-specific batch limit to prevent OOM errors.

        Returns:
            Maximum number of documents to rerank in a single batch
        """
        raise NotImplementedError("Provider must implement get_max_rerank_batch_size()")

    def _get_rerank_client_kwargs(self) -> dict[str, Any]:
        """Get httpx client kwargs for rerank requests.

        Provider-specific implementation required.
        Used for SSL verification, timeout configuration, etc.

        Returns:
            Dictionary of kwargs to pass to httpx.AsyncClient
        """
        raise NotImplementedError("Provider must implement _get_rerank_client_kwargs()")
