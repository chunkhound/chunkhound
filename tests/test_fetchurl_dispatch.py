"""Contract test for run_fetchurl's threshold-dispatch decision.

Locks the dispatch condition in fetchurl.py — the sole choice point between
option_truncate (one LLM call over sliced text) and option_chunk_rerank
(rerank + elbow + LLM call). ``rerank_threshold_tokens`` is a user-facing
config knob (CLI, env var, JSON key); a silent dispatch flip changes cost,
latency, and answer shape while still returning wrapped Markdown, so no
other test would fail.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.core.config.config import Config
from chunkhound.core.config.fetchurl_config import FetchUrlConfig
from chunkhound.interfaces.embedding_provider import EmbeddingProvider, RerankResult
from chunkhound.interfaces.llm_provider import LLMProvider
from chunkhound.llm_manager import LLMManager
from chunkhound.utils.fetchurl import run_fetchurl

# 50 LLM tokens = 200 chars via LLM_CHARS_PER_TOKEN (4). Small case sits
# well under, large case well over, and _BOUNDARY_MD lands exactly at 200
# chars → 50 tokens to lock the `<=` operator in the dispatch condition
# (a silent flip to `<` would flip only this case).
_THRESHOLD_TOKENS = 50
_SHORT_MD = "# T\n\nshort body paragraph.\n"
_LONG_MD = "# T\n\n" + ("body sentence with several words. " * 40)
_BOUNDARY_MD = "# T\n\n" + ("word " * 39)  # 5 + 195 = 200 chars → 50 tokens
assert len(_BOUNDARY_MD) == _THRESHOLD_TOKENS * 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query, payload, expect_rerank",
    [
        ("q", _SHORT_MD, False),
        ("q", _LONG_MD, True),
        ("q", _BOUNDARY_MD, False),
        ("", _LONG_MD, False),
    ],
    ids=[
        "small-with-query-uses-truncate",
        "large-with-query-uses-chunk-rerank",
        "at-threshold-uses-truncate",
        "empty-query-short-circuits-to-truncate",
    ],
)
async def test_dispatch_selects_option_by_threshold_and_query(
    tmp_path, query, payload, expect_rerank
):
    utility_provider = MagicMock(spec=LLMProvider)
    utility_provider.complete = AsyncMock(
        return_value=SimpleNamespace(content="answer")
    )
    llm_manager = MagicMock(spec=LLMManager)
    llm_manager.get_utility_provider.return_value = utility_provider

    embedding_provider = MagicMock(spec=EmbeddingProvider)
    embedding_provider.get_max_rerank_batch_size.return_value = 100
    embedding_provider.rerank = AsyncMock(
        side_effect=lambda query, documents: [
            RerankResult(index=i, score=0.9 - 0.1 * i)
            for i in range(len(documents))
        ]
    )

    async def fake_fetch(u, cfg, warning_callback=None):
        return (".md", payload, {"title": "T"})

    cfg = Config(
        target_dir=tmp_path,
        fetchurl=FetchUrlConfig(rerank_threshold_tokens=_THRESHOLD_TOKENS),
    )

    with patch(
        "chunkhound.utils.fetchurl._fetch_with_retry", side_effect=fake_fetch
    ):
        answer = await run_fetchurl(
            "http://example.com/doc.md",
            query,
            cfg,
            embedding_provider,
            llm_manager,
        )

    assert embedding_provider.rerank.called is expect_rerank
    utility_provider.complete.assert_called_once()
    assert "http://example.com/doc.md" in answer
