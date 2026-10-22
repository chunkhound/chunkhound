"""Hybrid SearchService pagination contracts."""

from typing import Any, cast

import pytest

from chunkhound.interfaces.database_provider import DatabaseProvider
from chunkhound.interfaces.embedding_provider import EmbeddingProvider
from chunkhound.services.search_service import SearchService


class BoundaryDatabase:
    """Record regex calls at the database boundary."""

    semantic_result_window_cap = 1000

    def __init__(self) -> None:
        self.regex_calls: list[dict[str, Any]] = []
        self.regex_results: list[dict[str, Any]] = []
        self.regex_has_more = False

    def search_regex(self, pattern: str, **kwargs: Any):
        self.regex_calls.append({"pattern": pattern, **kwargs})
        return self.regex_results, {"has_more": self.regex_has_more}


class BoundaryEmbeddingProvider:
    """Select single-hop without performing real embedding work."""

    name = "boundary"
    model = "boundary"

    @staticmethod
    def supports_reranking() -> bool:
        return False


class RecordingSemanticStrategy:
    """Record the semantic window passed through SearchService."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.results: list[dict[str, Any]] = []
        self.has_more = False
        self.candidate_budget_exhausted = False

    async def search(self, **kwargs: Any):
        self.calls.append(kwargs)
        return self.results, {
            "has_more": self.has_more,
            "candidate_budget_exhausted": self.candidate_budget_exhausted,
        }


def build_boundary_service(
    database: BoundaryDatabase,
) -> tuple[SearchService, RecordingSemanticStrategy]:
    service = SearchService(
        cast(DatabaseProvider, database),
        cast(EmbeddingProvider, BoundaryEmbeddingProvider()),
    )
    strategy = RecordingSemanticStrategy()
    service._single_hop_strategy = strategy  # type: ignore[assignment]
    return service, strategy


def build_regex_only_service(database: BoundaryDatabase) -> SearchService:
    """Build a hybrid service without a semantic branch."""
    return SearchService(cast(DatabaseProvider, database))


def semantic_result(chunk_id: int) -> dict[str, Any]:
    """Build a mergeable semantic result."""
    return {"chunk_id": chunk_id, "similarity": 1.0 - chunk_id / 100}


@pytest.mark.asyncio
async def test_regex_only_hybrid_ignores_semantic_window_cap() -> None:
    """Regex pagination can address offsets beyond a database's semantic cap."""
    database = BoundaryDatabase()
    service = build_regex_only_service(database)

    _, pagination = await service.search_hybrid(
        "query", regex_pattern="pattern", page_size=10, offset=995
    )

    assert database.regex_calls[0]["offset"] == 995
    assert pagination["offset"] == 995


@pytest.mark.asyncio
@pytest.mark.parametrize(("offset", "page_size"), [(-1, 1), (0, 0)])
async def test_regex_only_hybrid_rejects_invalid_windows(
    offset: int, page_size: int
) -> None:
    """Regex-only hybrid still enforces basic pagination invariants."""
    database = BoundaryDatabase()
    service = build_regex_only_service(database)

    with pytest.raises(ValueError):
        await service.search_hybrid(
            "query", regex_pattern="pattern", page_size=page_size, offset=offset
        )

    assert database.regex_calls == []


@pytest.mark.asyncio
async def test_hybrid_boundary_clamps_semantic_fetch_without_clamping_regex_fetch():
    """The final addressable semantic page keeps regex pagination independent."""
    database = BoundaryDatabase()
    service, semantic_strategy = build_boundary_service(database)

    _, pagination = await service.search_hybrid(
        "query", regex_pattern="pattern", page_size=5, offset=995
    )

    assert semantic_strategy.calls[0]["page_size"] == 5
    assert database.regex_calls[0]["page_size"] == 10
    assert pagination["next_offset"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(("offset", "page_size"), [(995, 6), (1000, 1)])
async def test_hybrid_rejects_windows_outside_semantic_cap(offset: int, page_size: int):
    """Hybrid requests share semantic search's exclusive result window."""
    database = BoundaryDatabase()
    service, semantic_strategy = build_boundary_service(database)

    with pytest.raises(ValueError, match=r"exclusive \[0, 1000\)"):
        await service.search_hybrid(
            "query", regex_pattern="pattern", page_size=page_size, offset=offset
        )

    assert semantic_strategy.calls == []
    assert database.regex_calls == []


@pytest.mark.asyncio
async def test_hybrid_propagates_semantic_candidate_budget_exhaustion():
    """Hybrid pagination preserves the semantic branch's completeness warning."""
    database = BoundaryDatabase()
    service, semantic_strategy = build_boundary_service(database)
    semantic_strategy.candidate_budget_exhausted = True

    _, pagination = await service.search_hybrid("query", page_size=5)

    assert pagination["candidate_budget_exhausted"] is True


@pytest.mark.asyncio
async def test_hybrid_exact_full_page_does_not_claim_continuation():
    """An exactly full merged page is final when both children are exhausted."""
    service, strategy = build_boundary_service(BoundaryDatabase())
    strategy.results = [semantic_result(1), semantic_result(2)]

    results, pagination = await service.search_hybrid("query", page_size=2)

    assert len(results) == 2
    assert pagination["has_more"] is False
    assert pagination["next_offset"] is None


@pytest.mark.asyncio
async def test_hybrid_merged_lookahead_proves_continuation():
    """A surplus merged result advertises another page without leaking lookahead."""
    service, strategy = build_boundary_service(BoundaryDatabase())
    strategy.results = [semantic_result(1), semantic_result(2), semantic_result(3)]

    results, pagination = await service.search_hybrid("query", page_size=2)

    assert len(results) == 2
    assert pagination["has_more"] is True
    assert pagination["next_offset"] == 2


@pytest.mark.asyncio
async def test_hybrid_child_continuation_survives_deduplication():
    """A child continuation remains visible when merge yields a short page."""
    service, strategy = build_boundary_service(BoundaryDatabase())
    strategy.results = [semantic_result(1)]
    strategy.has_more = True

    results, pagination = await service.search_hybrid("query", page_size=2)

    assert len(results) == 1
    assert pagination["has_more"] is True
    assert pagination["next_offset"] == 2
