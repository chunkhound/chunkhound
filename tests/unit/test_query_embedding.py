"""Search queries are embedded with the provider's ``embed_query``, never ``embed``.

Some models embed queries and documents asymmetrically (Voyage's input_type):
chunks are indexed as documents, so a query embedded as a document ranks as a
document-vs-document match. Every place that embeds a search query to compare
against indexed chunk vectors must therefore use ``embed_query``.

Each test drives one call site with a provider whose query and document vectors
differ, and asserts that the QUERY vector is what gets scored. A symmetric fake
cannot catch a revert to ``embed``: both paths would return the same vector.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from chunkhound.core.models.chunk import Chunk
from chunkhound.core.types.common import ChunkType, FileId, Language, LineNumber
from chunkhound.embeddings import EmbeddingManager
from chunkhound.services.diff_aware_search_service import DiffAwareSearchService
from chunkhound.services.research.shared.unified_search import UnifiedSearch
from chunkhound.services.search.single_hop_strategy import SingleHopStrategy
from chunkhound.services.search_service import SearchService
from tests.unit.provider_test_helpers import _bare_provider

DOCUMENT_VECTOR = [1.0, 0.0]
QUERY_VECTOR = [0.0, 1.0]


class _AsymmetricProvider:
    """Embeds documents and queries to different vectors; records each call."""

    name = "fake"
    model = "fake-model"
    dims = 2

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(("document", list(texts)))
        return [list(DOCUMENT_VECTOR) for _ in texts]

    async def embed_query(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(("query", list(texts)))
        return [list(QUERY_VECTOR) for _ in texts]


def _manager(provider: _AsymmetricProvider) -> EmbeddingManager:
    manager = EmbeddingManager()
    manager.register_provider(provider, set_default=True)  # type: ignore[arg-type]
    return manager


def _chunk(symbol: str, file_path: str) -> Chunk:
    return Chunk(
        symbol=symbol,
        start_line=LineNumber(1),
        end_line=LineNumber(5),
        code=f"def {symbol}(): pass",
        chunk_type=ChunkType.FUNCTION,
        file_id=FileId(1),
        language=Language.PYTHON,
        file_path=file_path,  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_manager_embed_query_texts_uses_embed_query() -> None:
    provider = _AsymmetricProvider()

    result = await _manager(provider).embed_query_texts(["q"])

    assert result.embeddings == [QUERY_VECTOR]
    assert result.provider == "fake"
    assert provider.calls == [("query", ["q"])]


@pytest.mark.asyncio
async def test_manager_embed_texts_still_embeds_documents() -> None:
    provider = _AsymmetricProvider()

    result = await _manager(provider).embed_texts(["chunk"])

    assert result.embeddings == [DOCUMENT_VECTOR]
    assert provider.calls == [("document", ["chunk"])]


@pytest.mark.asyncio
async def test_single_hop_search_scores_the_query_vector() -> None:
    provider = _AsymmetricProvider()
    db = MagicMock()
    seen: dict[str, Any] = {}

    def search_semantic(**kwargs: Any):
        seen.update(kwargs)
        return [], {"total": 0}

    db.search_semantic = MagicMock(side_effect=search_semantic)

    await SingleHopStrategy(db, provider).search(  # type: ignore[arg-type]
        query="where is retry handled",
        page_size=5,
        offset=0,
        threshold=None,
        provider="fake",
        model="fake-model",
        path_filter=None,
    )

    assert provider.calls == [("query", ["where is retry handled"])]
    assert seen["query_embedding"] == QUERY_VECTOR


@pytest.mark.asyncio
async def test_regex_search_scores_results_with_the_query_vector() -> None:
    provider = _AsymmetricProvider()
    db = MagicMock()
    db.search_regex_async = AsyncMock(
        return_value=([{"chunk_id": 7, "file_path": "a.py", "content": "x"}], {})
    )
    db.get_chunk_similarities_async = AsyncMock(return_value={7: 0.5})

    results, _ = await SearchService(db, provider).search_regex_async(  # type: ignore[arg-type]
        pattern="retry", query="where is retry handled"
    )

    db.get_chunk_similarities_async.assert_awaited_once()
    assert db.get_chunk_similarities_async.await_args.args[1] == QUERY_VECTOR
    assert provider.calls == [("query", ["where is retry handled"])]
    assert results[0]["similarity"] == 0.5


@pytest.mark.asyncio
async def test_symbol_search_scores_results_with_the_query_vector() -> None:
    provider = _AsymmetricProvider()
    search_service = MagicMock()
    search_service.search_regex_async = AsyncMock(
        side_effect=[([{"chunk_id": 7, "content": "x"}], {}), ([], {})]
    )
    search_service.get_chunk_similarities_async = AsyncMock(return_value={7: 0.5})
    db_services = MagicMock()
    db_services.search_service = search_service

    results = await UnifiedSearch(db_services, _manager(provider)).search_by_symbols(
        ["retry_loop"], target_per_symbol=5, query="where is retry handled"
    )

    search_service.get_chunk_similarities_async.assert_awaited_once()
    scored_with = search_service.get_chunk_similarities_async.await_args.args[1]
    assert scored_with == QUERY_VECTOR
    assert provider.calls == [("query", ["where is retry handled"])]
    assert results[0]["similarity"] == 0.5


@pytest.mark.asyncio
async def test_diff_search_ranks_by_the_query_vector() -> None:
    provider = _AsymmetricProvider()
    # One diff chunk matches the document vector, the other the query vector.
    chunks = [
        _chunk("matches_document", "src/a.py"),
        _chunk("matches_query", "src/b.py"),
    ]
    svc = DiffAwareSearchService(
        MagicMock(), chunks, [DOCUMENT_VECTOR, QUERY_VECTOR], "diff", _manager(provider)
    )

    results, _ = await svc.search_semantic("where is retry handled")

    assert [r["file_path"] for r in results] == ["src/b.py", "src/a.py"]
    assert results[0]["score"] == pytest.approx(1.0)
    assert provider.calls == [("query", ["where is retry handled"])]


@pytest.mark.asyncio
async def test_openai_embed_query_embeds_like_embed() -> None:
    # OpenAI-compatible APIs have no query/document input type.
    provider, _, _ = _bare_provider()
    provider.embed = AsyncMock(return_value=[[0.5, 0.5]])  # type: ignore[method-assign]

    assert await provider.embed_query(["q"]) == [[0.5, 0.5]]
    provider.embed.assert_awaited_once_with(["q"])
