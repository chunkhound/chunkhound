"""Contract tests: the embedding/reranker providers' per-attempt retry-loop
chokepoints must record one analytics provider call per attempt (including
retries), success or failure -- this is the Finding-#1 fix from the
analytics design (wrap the provider retry loop itself, not EmbeddingManager,
so both MCP-mediated and Rust-pipeline-direct calls are covered uniformly).
Uses the real chunkhound_native.AnalyticsRecorder (no mocking) with mocked
SDK/HTTP clients.
"""

import asyncio
import glob
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.core.analytics.recorder import (
    build_recorder,
    end_command,
    start_command,
)
from chunkhound.core.config.analytics_config import AnalyticsConfig
from tests.unit.provider_test_helpers import (
    _bare_provider,
    _FakeRateLimitError,
    _ok_response,
)


def _read_events(buffer_dir: Path) -> list[dict]:
    events = []
    for path in glob.glob(str(buffer_dir / "buffer-*.jsonl")):
        for line in Path(path).read_text().splitlines():
            events.append(json.loads(line))
    return events


@pytest.fixture
def open_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    buffer_dir = tmp_path / "analytics"
    monkeypatch.setattr("chunkhound.core.analytics.recorder._ANALYTICS_DIR", buffer_dir)
    config = AnalyticsConfig(
        enabled=True, flush_interval_seconds=999999, flush_batch_size=999999
    )
    recorder = build_recorder(config, tmp_path)
    handle = start_command(recorder, "index", "cli", {})
    yield recorder, handle, buffer_dir


@pytest.mark.asyncio
async def test_openai_embed_retry_then_success_counts_both_attempts(
    open_command,
) -> None:
    recorder, handle, buffer_dir = open_command
    provider, fake_openai, mod = _bare_provider(retry_attempts=2, retry_delay=0.0)

    call_count = 0

    async def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _FakeRateLimitError("429")
        return _ok_response()

    provider._client = MagicMock()
    provider._client.embeddings.create = side_effect

    with (
        patch.object(mod, "openai", fake_openai),
        patch.object(asyncio, "sleep", AsyncMock()),
    ):
        await provider._embed_batch_internal(["hello"])
    end_command(recorder, handle, True)

    embedding = _read_events(buffer_dir)[-1]["providers"]["embedding"][0]
    assert embedding["provider"] == "openai"
    assert embedding["calls"] == 2
    assert embedding["fails"] == 1
    assert embedding["error_types"] == {"_FakeRateLimitError": 1}


@pytest.mark.asyncio
async def test_openai_embed_success_records_token_usage(open_command) -> None:
    recorder, handle, buffer_dir = open_command
    provider, fake_openai, mod = _bare_provider(retry_attempts=1)
    provider._client = MagicMock()
    provider._client.embeddings.create = AsyncMock(return_value=_ok_response())

    with patch.object(mod, "openai", fake_openai):
        await provider._embed_batch_internal(["hello"])
    end_command(recorder, handle, True)

    embedding = _read_events(buffer_dir)[-1]["providers"]["embedding"][0]
    assert embedding["calls"] == 1
    assert embedding["fails"] == 0
    assert embedding["input_tokens"] == 10  # from _ok_response()'s usage mock


@pytest.mark.asyncio
async def test_openai_rerank_app_error_after_success_still_records_success(
    open_command,
) -> None:
    from chunkhound.providers.embeddings.openai_provider import OpenAIEmbeddingProvider

    recorder, handle, buffer_dir = open_command
    provider = OpenAIEmbeddingProvider(
        api_key="test-key",
        base_url="http://localhost:8080",
        model="text-embedding-3-small",
        rerank_model="test-reranker",
    )
    await provider._ensure_client()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "error": "batch too large",
        "error_type": "Validation",
    }
    mock_response.raise_for_status = MagicMock()
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch(
        "chunkhound.providers.embeddings.openai_provider.httpx.AsyncClient",
        return_value=mock_client,
    ):
        # The HTTP call itself succeeds (200 OK); the error is in the JSON
        # body, discovered by our own validation *after* the vendor call
        # already succeeded -- so this must record a successful provider
        # call, not a failure. This mirrors the embed providers' "success =
        # the vendor call succeeded, not our downstream validation" rule.
        with pytest.raises(ValueError):
            await provider.rerank("query", ["doc1", "doc2"])
    end_command(recorder, handle, True)

    reranker = _read_events(buffer_dir)[-1]["providers"]["reranker"][0]
    assert reranker["calls"] == 1
    assert reranker["fails"] == 0


@pytest.mark.asyncio
async def test_openai_rerank_connection_failure_records_a_failed_provider_call(
    open_command,
) -> None:
    from chunkhound.providers.embeddings.openai_provider import OpenAIEmbeddingProvider

    recorder, handle, buffer_dir = open_command
    provider = OpenAIEmbeddingProvider(
        api_key="test-key",
        base_url="http://localhost:8080",
        model="text-embedding-3-small",
        rerank_model="test-reranker",
    )
    await provider._ensure_client()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=RuntimeError("connection refused"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch(
        "chunkhound.providers.embeddings.openai_provider.httpx.AsyncClient",
        return_value=mock_client,
    ):
        with pytest.raises(Exception):
            await provider.rerank("query", ["doc1", "doc2"])
    end_command(recorder, handle, False)

    reranker = _read_events(buffer_dir)[-1]["providers"]["reranker"][0]
    assert reranker["fails"] == 1
    assert reranker["error_types"] == {"RuntimeError": 1}


@pytest.mark.asyncio
async def test_voyageai_embed_success_records_token_usage(open_command) -> None:
    from tests.unit.test_voyageai_provider import _FakeEmbedResult, _make_provider

    recorder, handle, buffer_dir = open_command
    provider = _make_provider(api_key="test-key", model="voyage-3")
    provider._client.embed = MagicMock(
        return_value=_FakeEmbedResult([[0.1, 0.2, 0.3]], total_tokens=7)
    )

    await provider._embed_single_batch_locked(["hello"])
    end_command(recorder, handle, True)

    embedding = _read_events(buffer_dir)[-1]["providers"]["embedding"][0]
    assert embedding["provider"] == "voyageai"
    assert embedding["calls"] == 1
    assert embedding["fails"] == 0
    assert embedding["input_tokens"] == 7


@pytest.mark.asyncio
async def test_voyageai_embed_failure_records_a_failed_provider_call(
    open_command,
) -> None:
    from tests.unit.test_voyageai_provider import _make_provider

    recorder, handle, buffer_dir = open_command
    provider = _make_provider(api_key="test-key", model="voyage-3")
    provider._client.embed = MagicMock(side_effect=RuntimeError("connection reset"))

    with pytest.raises(Exception):
        await provider._embed_single_batch_locked(["hello"])
    end_command(recorder, handle, False)

    embedding = _read_events(buffer_dir)[-1]["providers"]["embedding"][0]
    assert embedding["calls"] == 1
    assert embedding["fails"] == 1
    assert embedding["error_types"] == {"RuntimeError": 1}


@pytest.mark.asyncio
async def test_voyageai_rerank_sdk_success_records_a_provider_call(
    open_command,
) -> None:
    from tests.unit.test_voyageai_provider import _make_provider

    recorder, handle, buffer_dir = open_command
    provider = _make_provider(api_key="test-key")

    fake_result = MagicMock()
    fake_result.results = [MagicMock(index=0, relevance_score=0.9)]
    provider._client.rerank = MagicMock(return_value=fake_result)

    await provider._rerank_via_sdk("query", ["doc1"], top_k=None)
    end_command(recorder, handle, True)

    reranker = _read_events(buffer_dir)[-1]["providers"]["reranker"][0]
    assert reranker["provider"] == "voyageai"
    assert reranker["calls"] == 1
    assert reranker["fails"] == 0


@pytest.mark.asyncio
async def test_voyageai_rerank_sdk_failure_records_a_failed_provider_call(
    open_command,
) -> None:
    from tests.unit.test_voyageai_provider import _make_provider

    recorder, handle, buffer_dir = open_command
    provider = _make_provider(api_key="test-key")
    provider._client.rerank = MagicMock(side_effect=RuntimeError("service down"))

    with pytest.raises(Exception):
        await provider._rerank_via_sdk("query", ["doc1"], top_k=None)
    end_command(recorder, handle, False)

    reranker = _read_events(buffer_dir)[-1]["providers"]["reranker"][0]
    assert reranker["calls"] == 1
    assert reranker["fails"] == 1
    assert reranker["error_types"] == {"RuntimeError": 1}


@pytest.mark.asyncio
async def test_voyageai_rerank_http_single_batch_success_records_a_provider_call(
    open_command,
) -> None:
    """`_rerank_via_http`'s single-batch fast path (documents <= batch limit)
    bypasses `_rerank_via_http`'s multi-batch loop entirely -- must still
    reach `_rerank_http_batch`'s analytics recording, not just the
    already-covered SDK path.
    """
    from tests.unit.test_voyageai_provider import _make_provider, _mock_http_client

    recorder, handle, buffer_dir = open_command
    provider = _make_provider(
        api_key="test-key",
        base_url="http://localhost:1234",
        rerank_url="http://localhost:8001/rerank",
        rerank_format="auto",
    )
    mock_client = _mock_http_client({"results": [{"index": 0, "score": 0.9}]})

    with patch(
        "chunkhound.providers.embeddings.voyageai_provider.httpx.AsyncClient",
        return_value=mock_client,
    ):
        await provider._rerank_via_http("query", ["doc1"], top_k=None)
    end_command(recorder, handle, True)

    reranker = _read_events(buffer_dir)[-1]["providers"]["reranker"][0]
    assert reranker["provider"] == "voyageai"
    assert reranker["calls"] == 1
    assert reranker["fails"] == 0


@pytest.mark.asyncio
async def test_voyageai_rerank_http_single_batch_failure_records_a_failed_provider_call(
    open_command,
) -> None:
    from tests.unit.test_voyageai_provider import _make_provider

    recorder, handle, buffer_dir = open_command
    provider = _make_provider(
        api_key="test-key",
        base_url="http://localhost:1234",
        rerank_url="http://localhost:8001/rerank",
        rerank_format="auto",
    )
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=RuntimeError("connection refused"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch(
        "chunkhound.providers.embeddings.voyageai_provider.httpx.AsyncClient",
        return_value=mock_client,
    ):
        with pytest.raises(Exception):
            await provider._rerank_via_http("query", ["doc1"], top_k=None)
    end_command(recorder, handle, False)

    reranker = _read_events(buffer_dir)[-1]["providers"]["reranker"][0]
    assert reranker["calls"] == 1
    assert reranker["fails"] == 1
    assert reranker["error_types"] == {"RuntimeError": 1}


@pytest.mark.asyncio
async def test_openai_rerank_multi_batch_retry_records_one_call_per_attempt(
    open_command,
) -> None:
    """`rerank()`'s multi-batch split loop (documents > batch limit) retries
    each batch independently via `_rerank_single_batch` -- every attempt,
    including a failed-then-retried attempt on one batch and an
    immediately-successful call on the next, must be recorded separately.
    """
    from chunkhound.providers.embeddings.openai_provider import OpenAIEmbeddingProvider

    recorder, handle, buffer_dir = open_command
    provider = OpenAIEmbeddingProvider(
        api_key="test-key",
        base_url="http://localhost:8080",
        model="text-embedding-3-small",
        rerank_model="test-reranker",
        rerank_batch_size=1,
        retry_attempts=2,
        retry_delay=0.0,
    )
    await provider._ensure_client()

    success_response = MagicMock()
    success_response.status_code = 200
    success_response.json.return_value = {
        "results": [{"index": 0, "relevance_score": 0.9}]
    }
    success_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(
        side_effect=[
            RuntimeError("connection reset"),  # batch 1, attempt 1: retryable
            success_response,  # batch 1, attempt 2: succeeds
            success_response,  # batch 2, attempt 1: succeeds
        ]
    )
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with (
        patch(
            "chunkhound.providers.embeddings.openai_provider.httpx.AsyncClient",
            return_value=mock_client,
        ),
        patch.object(asyncio, "sleep", AsyncMock()),
    ):
        await provider.rerank("query", ["doc1", "doc2"], top_k=None)
    end_command(recorder, handle, True)

    reranker = _read_events(buffer_dir)[-1]["providers"]["reranker"][0]
    assert reranker["calls"] == 3
    assert reranker["fails"] == 1
    assert reranker["error_types"] == {"RuntimeError": 1}
