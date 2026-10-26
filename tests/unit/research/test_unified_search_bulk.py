"""Contracts for bulk semantic query expansion."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from chunkhound.database_factory import DatabaseServices
from chunkhound.services.research.shared.models import ResearchContext
from chunkhound.services.research.shared.unified_search import UnifiedSearch


def make_search(search_service: MagicMock) -> UnifiedSearch:
    provider = MagicMock()
    embedding_manager = MagicMock()
    embedding_provider = MagicMock()
    embedding_provider.supports_reranking.return_value = False
    embedding_manager.get_provider.return_value = embedding_provider
    services = DatabaseServices(
        provider=provider,
        indexing_coordinator=MagicMock(),
        search_service=search_service,
        embedding_service=MagicMock(),
    )
    return UnifiedSearch(services, embedding_manager)


@pytest.mark.asyncio
async def test_query_expansion_prefers_bulk_semantic_capability() -> None:
    search_service = MagicMock()
    search_service.search_semantic_many = AsyncMock(
        return_value=[
            ([{"chunk_id": "one", "content": "one"}], {"total": 1}),
            ([{"chunk_id": "two", "content": "two"}], {"total": 1}),
        ]
    )
    search_service.search_semantic = AsyncMock()
    search = make_search(search_service)
    search.extract_symbols_from_chunks = AsyncMock(return_value=[])

    results = await search.unified_search(
        "root",
        ResearchContext(root_query="root"),
        expanded_queries=["one", "two"],
        skip_rerank=True,
    )

    search_service.search_semantic_many.assert_awaited_once()
    search_service.search_semantic.assert_not_awaited()
    assert [result["chunk_id"] for result in results] == ["one", "two"]


@pytest.mark.asyncio
async def test_query_expansion_tolerates_shared_bulk_failure() -> None:
    search_service = MagicMock()
    search_service.search_semantic_many = AsyncMock(
        side_effect=TimeoutError("transient source failed")
    )
    search = make_search(search_service)
    search.extract_symbols_from_chunks = AsyncMock(return_value=[])

    results = await search.unified_search(
        "root",
        ResearchContext(root_query="root"),
        expanded_queries=["one", "two"],
        skip_rerank=True,
    )

    assert results == []


@pytest.mark.asyncio
async def test_query_expansion_falls_back_to_scalar_searches() -> None:
    search_service = MagicMock(spec=["search_semantic", "search_regex_async"])

    async def scalar_search(*, query: str, **kwargs):
        return [{"chunk_id": query, "content": query}], {"total": 1}

    search_service.search_semantic = AsyncMock(side_effect=scalar_search)
    search = make_search(search_service)
    search.extract_symbols_from_chunks = AsyncMock(return_value=[])

    results = await search.unified_search(
        "root",
        ResearchContext(root_query="root"),
        expanded_queries=["one", "two"],
        skip_rerank=True,
    )

    assert search_service.search_semantic.await_count == 2
    assert [result["chunk_id"] for result in results] == ["one", "two"]
