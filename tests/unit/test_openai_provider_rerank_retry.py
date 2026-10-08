"""Tests for retry coverage on OpenAIEmbeddingProvider.rerank().

Refs #408 (proposal 4): the single-batch path used to call
_rerank_single_batch() once with no retry wrapper, while the multi-batch
path retried each batch with backoff. These tests pin retry parity between
the two paths and confirm the multi-batch degrade-on-exhaustion behavior is
unchanged.
"""

import asyncio
from unittest.mock import patch

import pytest

from chunkhound.interfaces.embedding_provider import (
    EmbeddingProviderError,
    RerankResult,
)
from tests.unit.provider_test_helpers import _bare_provider


def _rerank_ready_provider(retry_attempts: int = 3, retry_delay: float = 0.01):
    """A _bare_provider with the rerank-specific attributes rerank() reads."""
    provider, fake_openai, mod = _bare_provider(
        retry_attempts=retry_attempts, retry_delay=retry_delay
    )
    provider._rerank_model = "test-rerank-model"
    provider._qwen_rerank_config = None
    provider._qwen_model_config = None
    provider._batch_size = 100
    provider._rerank_batch_size = None  # single batch unless a test overrides it
    return provider, fake_openai, mod


class TestSingleBatchRetryParity:
    @pytest.mark.asyncio
    async def test_single_batch_retries_transient_error_then_succeeds(self):
        """A single transient failure no longer aborts the whole rerank call."""
        provider, _, _ = _rerank_ready_provider(retry_attempts=3, retry_delay=1.0)

        call_count = 0

        async def flaky(query, documents, top_k):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("connection reset by peer")
            return [RerankResult(index=0, score=0.9)]

        provider._rerank_single_batch = flaky

        sleep_calls: list[float] = []

        async def fake_sleep(secs):
            sleep_calls.append(secs)

        with patch.object(asyncio, "sleep", fake_sleep):
            results = await provider.rerank("q", ["doc1"], top_k=None)

        assert call_count == 2
        assert len(sleep_calls) == 1
        assert results == [RerankResult(index=0, score=0.9)]

    @pytest.mark.asyncio
    async def test_single_batch_raises_after_exhausting_retries(self):
        """A persistent transient failure still surfaces to the caller, after
        retrying, matching the pre-existing single-batch raise-on-failure
        contract (multi_hop_strategy.py catches this and degrades)."""
        provider, _, _ = _rerank_ready_provider(retry_attempts=2, retry_delay=0.01)

        call_count = 0

        async def always_timeout(query, documents, top_k):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("request timeout")

        provider._rerank_single_batch = always_timeout

        async def fake_sleep(secs):
            pass

        with patch.object(asyncio, "sleep", fake_sleep):
            with pytest.raises(RuntimeError, match="request timeout"):
                await provider.rerank("q", ["doc1"], top_k=None)

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_single_batch_non_retryable_error_raises_without_retry(self):
        """A non-retryable error (no timeout/connection/503/429 marker) is not
        retried, matching the multi-batch path's classification."""
        provider, _, _ = _rerank_ready_provider(retry_attempts=3, retry_delay=0.01)

        call_count = 0

        async def bad_format(query, documents, top_k):
            nonlocal call_count
            call_count += 1
            raise ValueError("missing 'relevance_score' or 'score' field")

        provider._rerank_single_batch = bad_format

        with pytest.raises(ValueError, match="relevance_score"):
            await provider.rerank("q", ["doc1"], top_k=None)

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_single_batch_embedding_provider_error_skips_retry(self):
        """EmbeddingProviderError propagates immediately, same as before this
        change: it is never classified as retryable/non-retryable."""
        provider, _, _ = _rerank_ready_provider(retry_attempts=3, retry_delay=0.01)

        call_count = 0

        async def configuration_error(query, documents, top_k):
            nonlocal call_count
            call_count += 1
            raise EmbeddingProviderError("rerank_url is not configured")

        provider._rerank_single_batch = configuration_error

        with pytest.raises(EmbeddingProviderError, match="rerank_url"):
            await provider.rerank("q", ["doc1"], top_k=None)

        assert call_count == 1


class TestMultiBatchRetryUnchanged:
    @pytest.mark.asyncio
    async def test_multi_batch_still_degrades_instead_of_raising(self):
        """The multi-batch path must keep swallowing a batch's exhausted retry
        into an empty result and moving on to the next batch, not raise."""
        provider, _, _ = _rerank_ready_provider(retry_attempts=2, retry_delay=0.01)
        provider._rerank_batch_size = 1  # force 2 batches for 2 documents

        async def first_batch_always_fails_second_succeeds(query, documents, top_k):
            if documents == ["doc1"]:
                raise RuntimeError("connection refused")
            return [RerankResult(index=0, score=0.5)]

        provider._rerank_single_batch = first_batch_always_fails_second_succeeds

        async def fake_sleep(secs):
            pass

        with patch.object(asyncio, "sleep", fake_sleep):
            results = await provider.rerank("q", ["doc1", "doc2"], top_k=None)

        # batch 1 (doc1) exhausted retries and was dropped; batch 2 (doc2) succeeded
        assert len(results) == 1
        assert results[0].index == 1

    @pytest.mark.asyncio
    async def test_multi_batch_embedding_provider_error_still_aborts_immediately(self):
        """A config-shaped error on one batch must still abort the whole
        rerank() call, not be swallowed into an empty batch like a transient
        one (this is the pre-existing `except EmbeddingProviderError: raise`
        behavior, now reached through the shared retry helper)."""
        provider, _, _ = _rerank_ready_provider(retry_attempts=2, retry_delay=0.01)
        provider._rerank_batch_size = 1  # force 2 batches for 2 documents

        call_count = 0

        async def first_batch_config_error(query, documents, top_k):
            nonlocal call_count
            call_count += 1
            raise EmbeddingProviderError("rerank_url is not configured")

        provider._rerank_single_batch = first_batch_config_error

        with pytest.raises(EmbeddingProviderError, match="rerank_url"):
            await provider.rerank("q", ["doc1", "doc2"], top_k=None)

        # aborted on batch 1's first attempt; batch 2 was never reached
        assert call_count == 1
