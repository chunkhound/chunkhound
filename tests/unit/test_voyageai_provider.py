"""Unit tests for VoyageAIEmbeddingProvider — new voyage ranking features.

Covers gaps introduced by the voyage_endpoint branch:
  - _build_rerank_payload (all format branches)
  - _parse_rerank_response (sync, Cohere + TEI fields, edge cases)
  - _rerank_via_http multi-batch aggregation and index re-mapping
  - _rerank_http_batch response normalisation and error handling
  - supports_reranking with/without base_url and rerank_url
  - embed() sub-batching when input exceeds batch_size
  - embed_batch() token-budget flush
  - Semaphore initialisation (base_url → 1, official API → 40)
  - HTTP 408 / upstream-timeout retry logic
  - estimate_tokens edge cases
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import voyageai

from chunkhound.core.config.embedding_config import validate_rerank_configuration
from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory
from chunkhound.providers.embeddings.voyageai_provider import (
    VoyageAIEmbeddingProvider,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(**kwargs) -> VoyageAIEmbeddingProvider:
    """Return a VoyageAIEmbeddingProvider with the SDK client mocked out."""
    mock_client = MagicMock()
    with patch.object(voyageai, "Client", return_value=mock_client):
        return VoyageAIEmbeddingProvider(**kwargs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def provider_official():
    return _make_provider(api_key="test-key")


@pytest.fixture
def provider_custom():
    return _make_provider(api_key="test-key", base_url="http://localhost:1234")


@pytest.fixture
def provider_with_rerank_url():
    return _make_provider(
        api_key="test-key",
        base_url="http://localhost:1234",
        rerank_url="http://localhost:8001/rerank",
        rerank_format="auto",
    )


# ===========================================================================
# 1. supports_reranking
# ===========================================================================


# ===========================================================================
# 0. is_available
# ===========================================================================


class TestIsAvailable:
    def test_is_available_with_api_key(self):
        p = _make_provider(api_key="test-key")
        assert p.is_available() is True

    def test_is_available_keyless_azure_ml(self):
        """Azure ML deployments have no API key but a base_url — must be available."""
        p = _make_provider(base_url="https://az-endpoint.westus3.inference.ml.azure.com")
        assert p.is_available() is True

    def test_is_available_no_config(self):
        p = _make_provider()
        assert p.is_available() is False


# ===========================================================================
# 0b. validate_rerank_configuration
# ===========================================================================


class TestValidateRerankConfiguration:
    def test_validate_rerank_cohere_without_model_voyageai_http(self):
        """VoyageAI + HTTP rerank_url + cohere format + no model → should raise."""
        with pytest.raises(ValueError):
            validate_rerank_configuration(
                provider="voyageai",
                rerank_format="cohere",
                rerank_model=None,
                rerank_url="https://my-reranker.example.com/rerank",
                base_url=None,
            )

    def test_validate_rerank_voyageai_sdk_no_url_skips_validation(self):
        """VoyageAI with no rerank_url → SDK path, no error even without model."""
        validate_rerank_configuration(
            provider="voyageai",
            rerank_format="cohere",
            rerank_model=None,
            rerank_url=None,
            base_url=None,
        )


class TestSupportsReranking:
    def test_official_api_always_true(self, provider_official):
        assert provider_official.supports_reranking() is True

    def test_custom_endpoint_without_rerank_url_is_false(self, provider_custom):
        assert provider_custom.supports_reranking() is False

    def test_custom_endpoint_with_rerank_url_is_true(self, provider_with_rerank_url):
        assert provider_with_rerank_url.supports_reranking() is True


# ===========================================================================
# 2. Semaphore initialisation
# ===========================================================================


class TestSemaphoreInit:
    def test_custom_endpoint_defaults_to_one(self, provider_custom):
        assert provider_custom._embed_semaphore._value == 1

    def test_official_api_defaults_to_recommended_concurrency(self, provider_official):
        assert provider_official._embed_semaphore._value == VoyageAIEmbeddingProvider.RECOMMENDED_CONCURRENCY

    def test_explicit_max_concurrent_batches_honoured(self):
        p = _make_provider(api_key="test-key", max_concurrent_batches=5)
        assert p._embed_semaphore._value == 5

    def test_explicit_override_beats_base_url_default(self):
        p = _make_provider(
            api_key="test-key",
            base_url="http://localhost:1234",
            max_concurrent_batches=3,
        )
        assert p._embed_semaphore._value == 3


# ===========================================================================
# 3. estimate_tokens
# ===========================================================================


class TestEstimateTokens:
    def test_empty_string_returns_zero(self, provider_official):
        assert provider_official.estimate_tokens("") == 0

    def test_short_text(self, provider_official):
        # "abc" → len=3, 3 // 3 = 1 → max(1, 1) = 1
        assert provider_official.estimate_tokens("abc") == 1

    def test_longer_text(self, provider_official):
        text = "a" * 300  # 300 chars / 3 = 100 tokens
        assert provider_official.estimate_tokens(text) == 100

    def test_minimum_is_one_for_single_char(self, provider_official):
        # Single char: 1 // 3 = 0 → max(1, 0) = 1
        assert provider_official.estimate_tokens("x") == 1


# ===========================================================================
# 4. _build_rerank_payload
# ===========================================================================


class TestBuildRerankPayload:
    def _provider(self, fmt, model=None):
        return _make_provider(
            api_key="test-key",
            rerank_url="http://localhost:8001/rerank",
            rerank_format=fmt,
            rerank_model=model,
        )

    def test_tei_format(self):
        p = self._provider("tei")
        payload = p._build_rerank_payload("my query", ["doc1", "doc2"], top_k=None)
        assert payload == {"query": "my query", "texts": ["doc1", "doc2"]}

    def test_tei_format_ignores_top_k(self):
        p = self._provider("tei")
        payload = p._build_rerank_payload("q", ["d1"], top_k=5)
        assert "top_n" not in payload
        assert payload["texts"] == ["d1"]

    def test_cohere_format_with_model_no_top_k(self):
        p = self._provider("cohere", model="my-reranker")
        payload = p._build_rerank_payload("q", ["d1", "d2"], top_k=None)
        assert payload == {"query": "q", "documents": ["d1", "d2"], "model": "my-reranker"}
        assert "top_n" not in payload

    def test_cohere_format_with_model_and_top_k(self):
        p = self._provider("cohere", model="my-reranker")
        payload = p._build_rerank_payload("q", ["d1", "d2"], top_k=1)
        assert payload["top_n"] == 1
        assert payload["model"] == "my-reranker"

    def test_cohere_format_without_model_omits_model_key(self):
        p = self._provider("cohere")
        payload = p._build_rerank_payload("q", ["d"], top_k=None)
        assert "model" not in payload
        assert "documents" in payload

    def test_auto_with_model_uses_cohere_style(self):
        p = self._provider("auto", model="my-reranker")
        payload = p._build_rerank_payload("q", ["d1"], top_k=None)
        assert "documents" in payload
        assert payload["model"] == "my-reranker"
        assert "texts" not in payload

    def test_auto_with_model_and_top_k(self):
        p = self._provider("auto", model="my-reranker")
        payload = p._build_rerank_payload("q", ["d1"], top_k=3)
        assert payload["top_n"] == 3

    def test_auto_without_model_uses_tei_style(self):
        p = self._provider("auto", model=None)
        payload = p._build_rerank_payload("q", ["d1", "d2"], top_k=None)
        assert "texts" in payload
        assert "documents" not in payload
        assert "model" not in payload


# ===========================================================================
# 5. _parse_rerank_response (sync method)
# ===========================================================================


class TestParseRerankResponse:
    @pytest.fixture
    def p(self, provider_with_rerank_url):
        return provider_with_rerank_url

    def test_missing_results_key_raises(self, p):
        with pytest.raises(ValueError, match="missing 'results'"):
            p._parse_rerank_response({"status": "ok"}, num_documents=2)

    def test_cohere_relevance_score_field(self, p):
        data = {
            "results": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.4},
            ]
        }
        results = p._parse_rerank_response(data, num_documents=2)
        assert len(results) == 2
        assert results[0].index == 0
        assert results[0].score == pytest.approx(0.9)

    def test_tei_score_field(self, p):
        data = {
            "results": [
                {"index": 1, "score": 0.7},
                {"index": 0, "score": 0.3},
            ]
        }
        results = p._parse_rerank_response(data, num_documents=2)
        # Sorted descending: index 1 first
        assert results[0].index == 1
        assert results[1].index == 0

    def test_sorted_descending(self, p):
        data = {
            "results": [
                {"index": 0, "score": 0.1},
                {"index": 1, "score": 0.9},
                {"index": 2, "score": 0.5},
            ]
        }
        results = p._parse_rerank_response(data, num_documents=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_missing_index_skipped(self, p):
        data = {"results": [{"score": 0.8}, {"index": 1, "score": 0.5}]}
        results = p._parse_rerank_response(data, num_documents=2)
        assert len(results) == 1
        assert results[0].index == 1

    def test_missing_score_skipped(self, p):
        data = {"results": [{"index": 0}, {"index": 1, "score": 0.5}]}
        results = p._parse_rerank_response(data, num_documents=2)
        assert len(results) == 1
        assert results[0].index == 1

    def test_out_of_range_index_skipped(self, p):
        data = {
            "results": [
                {"index": 0, "score": 0.9},
                {"index": 5, "score": 0.8},  # out of range (only 3 docs)
            ]
        }
        results = p._parse_rerank_response(data, num_documents=3)
        assert len(results) == 1
        assert results[0].index == 0

    def test_negative_index_skipped(self, p):
        data = {"results": [{"index": -1, "score": 0.9}, {"index": 0, "score": 0.5}]}
        results = p._parse_rerank_response(data, num_documents=2)
        assert len(results) == 1
        assert results[0].index == 0

    def test_empty_results_list(self, p):
        assert p._parse_rerank_response({"results": []}, num_documents=3) == []

    def test_relevance_score_takes_precedence_over_score(self, p):
        # Both fields present — relevance_score wins (Cohere-style)
        data = {"results": [{"index": 0, "relevance_score": 0.95, "score": 0.1}]}
        results = p._parse_rerank_response(data, num_documents=1)
        assert results[0].score == pytest.approx(0.95)

    def test_relevance_score_zero_not_dropped(self, p):
        # relevance_score = 0.0 is falsy — must NOT fall through to "score" field
        data = {"results": [{"index": 0, "relevance_score": 0.0, "score": 0.9}]}
        results = p._parse_rerank_response(data, num_documents=1)
        assert len(results) == 1
        assert results[0].score == pytest.approx(0.0)


# ===========================================================================
# 6. _rerank_http_batch — mocked httpx
# ===========================================================================


def _mock_http_client(json_data, status_code=200):
    mock_resp = MagicMock()
    mock_resp.json.return_value = json_data
    mock_resp.raise_for_status = MagicMock()
    mock_resp.status_code = status_code

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    return mock_client


class TestRerankHttpBatch:
    @pytest.mark.asyncio
    async def test_bare_array_response_normalised(self, provider_with_rerank_url):
        """TEI servers return bare lists; provider must wrap to {"results": [...]}."""
        mock_client = _mock_http_client([
            {"index": 1, "score": 0.8},
            {"index": 0, "score": 0.3},
        ])

        with patch("chunkhound.providers.embeddings.voyageai_provider.httpx.AsyncClient",
                   return_value=mock_client):
            results = await provider_with_rerank_url._rerank_http_batch(
                "query", ["doc0", "doc1"], top_k=None
            )

        assert len(results) == 2
        assert results[0].index == 1  # sorted descending by score
        assert results[0].score == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_error_key_in_response_raises(self, provider_with_rerank_url):
        mock_client = _mock_http_client({"error": "model not found"})

        with patch("chunkhound.providers.embeddings.voyageai_provider.httpx.AsyncClient",
                   return_value=mock_client):
            with pytest.raises(ValueError, match="Rerank service error"):
                await provider_with_rerank_url._rerank_http_batch(
                    "q", ["doc"], top_k=None
                )

    @pytest.mark.asyncio
    async def test_authorization_header_sent_when_api_key_set(self, provider_with_rerank_url):
        mock_client = _mock_http_client({"results": [{"index": 0, "score": 0.5}]})

        with patch("chunkhound.providers.embeddings.voyageai_provider.httpx.AsyncClient",
                   return_value=mock_client):
            await provider_with_rerank_url._rerank_http_batch("q", ["doc"], top_k=None)

        _, kwargs = mock_client.post.call_args
        assert "Authorization" in kwargs.get("headers", {})
        assert kwargs["headers"]["Authorization"] == "Bearer test-key"

    @pytest.mark.asyncio
    async def test_no_authorization_header_when_no_api_key(self):
        p = _make_provider(
            base_url="http://localhost:1234",
            rerank_url="http://localhost:8001/rerank",
            rerank_format="tei",
        )
        assert p._api_key is None

        mock_client = _mock_http_client({"results": [{"index": 0, "score": 0.5}]})

        with patch("chunkhound.providers.embeddings.voyageai_provider.httpx.AsyncClient",
                   return_value=mock_client):
            await p._rerank_http_batch("q", ["doc"], top_k=None)

        _, kwargs = mock_client.post.call_args
        assert "Authorization" not in kwargs.get("headers", {})


# ===========================================================================
# 7. _rerank_via_http — multi-batch aggregation
# ===========================================================================


class TestRerankViaHttpBatching:
    @pytest.mark.asyncio
    async def test_single_batch_no_splitting(self, provider_with_rerank_url):
        from chunkhound.interfaces.embedding_provider import RerankResult

        docs = ["a", "b", "c"]

        async def fake_batch(query, documents, top_k):
            return [
                RerankResult(index=2, score=0.9),
                RerankResult(index=0, score=0.6),
                RerankResult(index=1, score=0.3),
            ]

        provider_with_rerank_url._rerank_http_batch = fake_batch

        results = await provider_with_rerank_url._rerank_via_http("q", docs, top_k=2)
        assert len(results) == 2
        assert results[0].index == 2

    @pytest.mark.asyncio
    async def test_multi_batch_index_remapped(self, provider_with_rerank_url):
        from chunkhound.interfaces.embedding_provider import RerankResult

        provider_with_rerank_url._rerank_batch_size = 2
        call_log: list[int] = []

        async def fake_batch(query, documents, top_k):
            call_log.append(len(documents))
            return [RerankResult(index=i, score=0.9 - i * 0.1) for i in range(len(documents))]

        provider_with_rerank_url._rerank_http_batch = fake_batch

        docs = ["d0", "d1", "d2", "d3"]
        results = await provider_with_rerank_url._rerank_via_http("q", docs, top_k=None)

        assert call_log == [2, 2]
        all_indices = {r.index for r in results}
        assert all_indices == {0, 1, 2, 3}

    @pytest.mark.asyncio
    async def test_multi_batch_top_k_applied_after_merge(self, provider_with_rerank_url):
        from chunkhound.interfaces.embedding_provider import RerankResult

        provider_with_rerank_url._rerank_batch_size = 2

        async def fake_batch(query, documents, top_k):
            return [RerankResult(index=i, score=float(i)) for i in range(len(documents))]

        provider_with_rerank_url._rerank_http_batch = fake_batch

        results = await provider_with_rerank_url._rerank_via_http("q", ["d0", "d1", "d2", "d3"], top_k=2)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_multi_batch_sorted_descending(self, provider_with_rerank_url):
        from chunkhound.interfaces.embedding_provider import RerankResult

        provider_with_rerank_url._rerank_batch_size = 2

        async def fake_batch(query, documents, top_k):
            return [RerankResult(index=i, score=float(i) * 0.1) for i in range(len(documents))]

        provider_with_rerank_url._rerank_http_batch = fake_batch

        results = await provider_with_rerank_url._rerank_via_http("q", ["d0", "d1", "d2", "d3"], top_k=None)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


# ===========================================================================
# 8. embed() sub-batching
# ===========================================================================


class TestEmbedSubBatching:
    @pytest.mark.asyncio
    async def test_single_call_when_within_batch_size(self, provider_official):
        provider_official._embed_single_batch = AsyncMock(return_value=[[0.1, 0.2]])
        result = await provider_official.embed(["hello"])
        provider_official._embed_single_batch.assert_called_once()
        assert result == [[0.1, 0.2]]

    @pytest.mark.asyncio
    async def test_sub_batching_splits_correctly(self):
        p = _make_provider(api_key="test-key", batch_size=2)
        call_sizes: list[int] = []

        async def fake_single_batch(texts):
            call_sizes.append(len(texts))
            return [[float(i)] for i in range(len(texts))]

        p._embed_single_batch = fake_single_batch

        result = await p.embed(["a", "b", "c", "d", "e"])

        assert call_sizes == [2, 2, 1]
        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_sub_batching_preserves_order(self):
        p = _make_provider(api_key="test-key", batch_size=2)

        async def fake_single_batch(texts):
            return [[float(ord(t[0]))] for t in texts]

        p._embed_single_batch = fake_single_batch

        result = await p.embed(["a", "b", "c"])
        assert result[0] == [float(ord("a"))]
        assert result[1] == [float(ord("b"))]
        assert result[2] == [float(ord("c"))]

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self, provider_official):
        result = await provider_official.embed([])
        assert result == []


# ===========================================================================
# 9. embed_batch() token-budget splitting
# ===========================================================================


class TestEmbedBatchTokenBudget:
    @pytest.mark.asyncio
    async def test_count_based_flush(self):
        p = _make_provider(api_key="test-key", batch_size=2)
        batches_sent: list[list[str]] = []

        async def fake_embed(texts):
            batches_sent.append(list(texts))
            return [[0.1] for _ in texts]

        p.embed = fake_embed

        await p.embed_batch(["a", "b", "c", "d"], batch_size=2)

        assert batches_sent == [["a", "b"], ["c", "d"]]

    @pytest.mark.asyncio
    async def test_token_budget_flush(self):
        p = _make_provider(api_key="test-key")
        p._model_config = dict(p._model_config)
        p._model_config["max_tokens_per_batch"] = 4  # forces flush every 4 tokens

        batches_sent: list[list[str]] = []

        async def fake_embed(texts):
            batches_sent.append(list(texts))
            return [[0.1] for _ in texts]

        p.embed = fake_embed

        # Each "abc" = 3 chars → 1 token; "aaaaaa" = 6 chars → 2 tokens
        # Budget=4: [abc(1), abc(1), abc(1), abc(1)] → flush → next batch
        texts = ["abc"] * 6  # 6 texts × 1 token each
        await p.embed_batch(texts, batch_size=100)

        assert len(batches_sent) >= 2
        assert sum(len(b) for b in batches_sent) == 6

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self, provider_official):
        result = await provider_official.embed_batch([])
        assert result == []


# ===========================================================================
# 10. HTTP 408 / upstream-timeout retry logic
# ===========================================================================


class TestUpstreamTimeoutRetry:
    """Test retry logic inside _embed_single_batch_locked by patching asyncio.to_thread."""

    @pytest.mark.asyncio
    async def test_408_in_error_string_retried(self):
        p = _make_provider(api_key="test-key", retry_attempts=2, retry_delay=0.0)
        call_count = 0

        def fake_to_thread(fn, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("HTTP 408 upstream request timeout")

        with patch("chunkhound.providers.embeddings.voyageai_provider.asyncio.to_thread",
                   side_effect=fake_to_thread):
            with patch("chunkhound.providers.embeddings.voyageai_provider.asyncio.sleep",
                       new_callable=AsyncMock) as mock_sleep:
                with pytest.raises(RuntimeError):
                    await p._embed_single_batch_locked(["text"])

        assert call_count == 2
        # Upstream timeout backoff: base_delay=10.0, attempt=0 → 10.0 * 2^0 = 10.0
        assert mock_sleep.call_args_list[0].args[0] == pytest.approx(10.0)

    @pytest.mark.asyncio
    async def test_upstream_timeout_phrase_retried(self):
        p = _make_provider(api_key="test-key", retry_attempts=2, retry_delay=0.0)
        call_count = 0

        def fake_to_thread(fn, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Upstream Request Timeout from proxy")

        with patch("chunkhound.providers.embeddings.voyageai_provider.asyncio.to_thread",
                   side_effect=fake_to_thread):
            with patch("chunkhound.providers.embeddings.voyageai_provider.asyncio.sleep",
                       new_callable=AsyncMock):
                with pytest.raises(RuntimeError):
                    await p._embed_single_batch_locked(["text"])

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_non_retryable_error_fails_immediately(self):
        p = _make_provider(api_key="test-key", retry_attempts=3, retry_delay=0.0)
        call_count = 0

        def fake_to_thread(fn, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise ValueError("bad input")

        with patch("chunkhound.providers.embeddings.voyageai_provider.asyncio.to_thread",
                   side_effect=fake_to_thread):
            with pytest.raises(RuntimeError, match="Embedding generation failed"):
                await p._embed_single_batch_locked(["text"])

        assert call_count == 1


# ===========================================================================
# 10. Factory — relative rerank_url resolution
# ===========================================================================


class TestFactoryRerankUrlResolution:
    """Test that the factory resolves relative rerank_url against base_url."""

    def _make_dict(self, **kwargs):
        base = {"model": "voyage-3", "api_key": "test-key", "rerank_format": "cohere"}
        base.update(kwargs)
        return base

    def test_relative_rerank_url_resolved_against_base_url(self):
        config = self._make_dict(
            base_url="https://my-endpoint.example.com",
            rerank_url="/rerank",
        )
        with patch.object(voyageai, "Client", return_value=MagicMock()):
            provider = EmbeddingProviderFactory._create_voyageai_provider(config)
        assert provider._rerank_url == "https://my-endpoint.example.com/rerank"

    def test_absolute_rerank_url_passed_through_unchanged(self):
        config = self._make_dict(
            base_url="https://my-endpoint.example.com",
            rerank_url="https://other-host.example.com/rerank",
        )
        with patch.object(voyageai, "Client", return_value=MagicMock()):
            provider = EmbeddingProviderFactory._create_voyageai_provider(config)
        assert provider._rerank_url == "https://other-host.example.com/rerank"

    def test_relative_rerank_url_without_base_url_not_forwarded(self):
        # No base_url → relative rerank_url cannot be resolved → not passed to provider
        config = self._make_dict(rerank_url="/rerank")
        with patch.object(voyageai, "Client", return_value=MagicMock()):
            provider = EmbeddingProviderFactory._create_voyageai_provider(config)
        assert provider._rerank_url is None

    def test_max_concurrent_batches_passed_through(self):
        config = self._make_dict(max_concurrent_batches=7)
        with patch.object(voyageai, "Client", return_value=MagicMock()):
            provider = EmbeddingProviderFactory._create_voyageai_provider(config)
        assert provider._embed_semaphore._value == 7


# ===========================================================================
# 11. get_model_info reflects supports_reranking()
# ===========================================================================


class TestGetModelInfo:
    def test_official_api_supports_reranking_true(self, provider_official):
        info = provider_official.get_model_info()
        assert info["supports_reranking"] is True

    def test_custom_endpoint_without_rerank_url_is_false(self, provider_custom):
        info = provider_custom.get_model_info()
        assert info["supports_reranking"] is False

    def test_custom_endpoint_with_rerank_url_is_true(self, provider_with_rerank_url):
        info = provider_with_rerank_url.get_model_info()
        assert info["supports_reranking"] is True
