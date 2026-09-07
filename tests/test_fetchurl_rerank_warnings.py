"""Rerank-degradation warnings surfaced by `option_chunk_rerank`.

The three warning branches each guard against a different silent-failure
mode:

- Zero results: reranker returned nothing; scores stay at 0.0, elbow
  falls through, chunks land in original document order — visually
  indistinguishable from a healthy result.
- Partial results: some chunks retain the default 0.0 score and get
  demoted below rescored siblings for reasons unrelated to relevance.
- Uniform scores (>3 chunks): reranker responded but the ranking signal
  is flat — same original-order degeneration as the zero case.

The MCP integration routes these warnings into the response footer so
they are user-visible.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from chunkhound.core.config.config import Config
from chunkhound.core.models.chunk import Chunk
from chunkhound.core.types.common import ChunkType, FileId, Language, LineNumber
from chunkhound.interfaces.embedding_provider import EmbeddingProvider, RerankResult
from chunkhound.interfaces.llm_provider import LLMProvider
from chunkhound.llm_manager import LLMManager
from chunkhound.utils.fetchurl import option_chunk_rerank


def _chunk(line: int, code: str = "body text") -> Chunk:
    return Chunk(
        symbol=f"chunk_{line}",
        start_line=LineNumber(line),
        end_line=LineNumber(line),
        code=code,
        chunk_type=ChunkType.UNKNOWN,
        file_id=FileId(0),
        language=Language.MARKDOWN,
    )


def _make_llm_manager() -> MagicMock:
    utility_provider = MagicMock(spec=LLMProvider)
    utility_provider.complete = AsyncMock(
        return_value=SimpleNamespace(content="answer")
    )
    llm_manager = MagicMock(spec=LLMManager)
    llm_manager.get_utility_provider.return_value = utility_provider
    return llm_manager


def _make_embedding_provider(rerank_return: list[RerankResult]) -> MagicMock:
    provider = MagicMock(spec=EmbeddingProvider)
    provider.get_max_rerank_batch_size.return_value = 100
    provider.rerank = AsyncMock(return_value=rerank_return)
    return provider


async def _run(chunks: list[Chunk], provider: MagicMock, tmp_path) -> list[str]:
    warnings: list[str] = []
    llm_manager = _make_llm_manager()
    await option_chunk_rerank(
        chunks,
        query="q",
        url="http://example.com/",
        title="Doc",
        embedding_provider=provider,
        llm_manager=llm_manager,
        config=Config(target_dir=tmp_path),
        warning_callback=warnings.append,
    )
    # LLM must still be called — the warnings are diagnostics, not aborts.
    llm_manager.get_utility_provider.return_value.complete.assert_called_once()
    return warnings


@pytest.mark.asyncio
async def test_warns_when_reranker_returns_zero_results(tmp_path) -> None:
    chunks = [_chunk(i) for i in range(1, 6)]
    provider = _make_embedding_provider([])
    warnings = await _run(chunks, provider, tmp_path)
    assert any("Reranker returned no results" in w for w in warnings), warnings


@pytest.mark.asyncio
async def test_warns_when_reranker_returns_partial_results(tmp_path) -> None:
    chunks = [_chunk(i) for i in range(1, 6)]
    provider = _make_embedding_provider(
        [
            RerankResult(index=0, score=0.9),
            RerankResult(index=1, score=0.7),
            RerankResult(index=2, score=0.5),
        ]
    )
    warnings = await _run(chunks, provider, tmp_path)
    assert any("3/5" in w for w in warnings), warnings


@pytest.mark.asyncio
async def test_warns_when_reranker_returns_uniform_scores(tmp_path) -> None:
    chunks = [_chunk(i) for i in range(1, 6)]
    provider = _make_embedding_provider(
        [RerankResult(index=i, score=0.5) for i in range(5)]
    )
    warnings = await _run(chunks, provider, tmp_path)
    assert any("identical scores" in w and "5 chunks" in w for w in warnings), (
        warnings
    )


@pytest.mark.asyncio
async def test_no_warning_on_healthy_rerank(tmp_path) -> None:
    chunks = [_chunk(i) for i in range(1, 6)]
    provider = _make_embedding_provider(
        [RerankResult(index=i, score=0.9 - 0.1 * i) for i in range(5)]
    )
    warnings = await _run(chunks, provider, tmp_path)
    assert warnings == []


@pytest.mark.asyncio
async def test_no_uniform_warning_when_three_or_fewer_chunks(tmp_path) -> None:
    """The uniform-score branch is gated on `len(chunk_dicts) > 3` because a
    3-chunk uniform result is ambiguous (small samples routinely tie). Lock
    that gate in so a future >= 3 mistake doesn't spam warnings on healthy
    small documents."""
    chunks = [_chunk(i) for i in range(1, 4)]
    provider = _make_embedding_provider(
        [RerankResult(index=i, score=0.5) for i in range(3)]
    )
    warnings = await _run(chunks, provider, tmp_path)
    assert not any("identical scores" in w for w in warnings), warnings
