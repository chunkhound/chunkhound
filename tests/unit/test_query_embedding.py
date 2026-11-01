"""Search queries are embedded through the provider's query path, when it has one.

Providers whose models embed queries and documents asymmetrically (Voyage's
input_type) define `embed_query`; every place that embeds a search query to
compare against indexed chunk vectors must use it. Providers without it keep
embedding queries exactly as before.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from chunkhound.embeddings import EmbeddingManager
from chunkhound.interfaces.embedding_provider import embed_queries
from chunkhound.services.search.single_hop_strategy import SingleHopStrategy


class _AsymmetricProvider:
    """Records which path embedded which texts."""

    name = "fake"
    model = "fake-model"
    dims = 2

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(("document", list(texts)))
        return [[1.0, 0.0] for _ in texts]

    async def embed_query(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(("query", list(texts)))
        return [[0.0, 1.0] for _ in texts]


class _SymmetricProvider:
    name = "plain"
    model = "plain-model"
    dims = 2

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]


@pytest.mark.asyncio
async def test_embed_queries_uses_the_query_path_when_the_provider_has_one() -> None:
    provider = _AsymmetricProvider()

    vectors = await embed_queries(provider, ["where is retry handled"])

    assert vectors == [[0.0, 1.0]]
    assert provider.calls == [("query", ["where is retry handled"])]


@pytest.mark.asyncio
async def test_embed_queries_falls_back_to_embed_without_a_query_path() -> None:
    provider = _SymmetricProvider()

    vectors = await embed_queries(provider, ["where is retry handled"])

    assert vectors == [[1.0, 0.0]]
    assert provider.calls == [["where is retry handled"]]


@pytest.mark.asyncio
async def test_embed_queries_ignores_attributes_a_mock_invents() -> None:
    # A MagicMock answers every attribute, so an instance-level lookup would
    # find a non-awaitable `embed_query` on providers tests stub this way.
    provider = MagicMock()
    provider.embed = AsyncMock(return_value=[[1.0, 0.0]])

    assert await embed_queries(provider, ["q"]) == [[1.0, 0.0]]
    provider.embed.assert_awaited_once_with(["q"])


@pytest.mark.asyncio
async def test_manager_embed_query_texts_uses_the_query_path() -> None:
    manager = EmbeddingManager()
    provider = _AsymmetricProvider()
    manager.register_provider(provider, set_default=True)  # type: ignore[arg-type]

    result = await manager.embed_query_texts(["q"])

    assert result.embeddings == [[0.0, 1.0]]
    assert provider.calls == [("query", ["q"])]


@pytest.mark.asyncio
async def test_single_hop_search_embeds_the_query_as_a_query() -> None:
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
    assert seen["query_embedding"] == [0.0, 1.0]
