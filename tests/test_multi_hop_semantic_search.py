"""Deterministic multi-hop semantic search contract tests."""

from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest

from chunkhound.core.types.common import Language
from chunkhound.interfaces.database_provider import DatabaseProvider
from chunkhound.interfaces.embedding_provider import EmbeddingProvider
from chunkhound.parsers.parser_factory import create_parser_for_language
from chunkhound.parsers.universal_parser import UniversalParser
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.indexing_coordinator import IndexingCoordinator
from chunkhound.services.search_service import SearchService
from tests.fixtures.fake_providers import FakeEmbeddingProvider
from tests.fixtures.multi_hop_synthetic import (
    build_insufficient_candidates_scenario,
    build_min_score_termination_scenario,
    build_multi_hop_scenario,
    build_score_drop_termination_scenario,
)


def _build_search_service(db: object, provider: object) -> SearchService:
    return SearchService(
        cast(DatabaseProvider, db),
        cast(EmbeddingProvider, provider),
    )


def _build_python_coordinator(
    db: DuckDBProvider,
    tmp_path: Path,
    embedding_provider: FakeEmbeddingProvider,
) -> IndexingCoordinator:
    parser = cast(UniversalParser, create_parser_for_language(Language.PYTHON))
    return IndexingCoordinator(
        cast(DatabaseProvider, db),
        tmp_path,
        cast(EmbeddingProvider, embedding_provider),
        {Language.PYTHON: parser},
    )


@pytest.mark.fast
@pytest.mark.asyncio
async def test_search_service_selects_expected_semantic_strategy() -> None:
    """SearchService should route requests by reranking capability and override."""
    reranking_scenario = build_multi_hop_scenario()
    reranking_service = _build_search_service(
        reranking_scenario.db,
        reranking_scenario.provider,
    )
    reranking_multi = AsyncMock(return_value=([], {}))
    reranking_single = AsyncMock(return_value=([], {}))

    with patch.object(
        reranking_service._multi_hop_strategy,
        "search",
        reranking_multi,
    ), patch.object(
        reranking_service._single_hop_strategy,
        "search",
        reranking_single,
    ):
        await reranking_service.search_semantic("strategy query", page_size=5)

    reranking_multi.assert_awaited_once()
    reranking_single.assert_not_awaited()

    non_reranking_scenario = build_multi_hop_scenario(supports_reranking=False)
    non_reranking_service = _build_search_service(
        non_reranking_scenario.db,
        non_reranking_scenario.provider,
    )
    non_reranking_multi = AsyncMock(return_value=([], {}))
    non_reranking_single = AsyncMock(return_value=([], {}))

    with patch.object(
        non_reranking_service._multi_hop_strategy,
        "search",
        non_reranking_multi,
    ), patch.object(
        non_reranking_service._single_hop_strategy,
        "search",
        non_reranking_single,
    ):
        await non_reranking_service.search_semantic("strategy query", page_size=5)
        await non_reranking_service.search_semantic(
            "strategy query",
            page_size=5,
            force_strategy="multi_hop",
        )

    non_reranking_multi.assert_not_awaited()
    assert non_reranking_single.await_count == 2


@pytest.mark.fast
@pytest.mark.asyncio
async def test_multi_hop_bridges_predefined_vocabulary_gap() -> None:
    """Multi-hop should recover relevant targets unreachable by single-hop alone."""
    scenario = build_multi_hop_scenario()
    search_service = _build_search_service(scenario.db, scenario.provider)
    query = "security validation mechanisms"

    single_hop_results, _ = await search_service.search_semantic(
        query,
        page_size=3,
        force_strategy="single_hop",
    )
    multi_hop_results, _ = await search_service.search_semantic(query, page_size=3)

    single_ids = {result["chunk_id"] for result in single_hop_results}
    multi_ids = {result["chunk_id"] for result in multi_hop_results}

    assert single_ids.isdisjoint(scenario.qrels[query]), (
        f"single-hop should not find bridge targets, got {single_ids} from {query!r}"
    )
    intersection = multi_ids & scenario.qrels[query]
    assert intersection == {3, 4}, (
        f"multi-hop should find bridge targets {{3,4}}, got {intersection}"
    )


@pytest.mark.fast
@pytest.mark.asyncio
async def test_multi_hop_results_are_deduplicated_across_converging_paths() -> None:
    """A chunk reached through multiple expansion paths should appear once."""
    scenario = build_multi_hop_scenario()
    search_service = _build_search_service(scenario.db, scenario.provider)

    results, _ = await search_service.search_semantic(
        "security validation mechanisms",
        page_size=8,
    )

    chunk_ids = [result["chunk_id"] for result in results]
    assert len(chunk_ids) == len(set(chunk_ids))


@pytest.mark.fast
@pytest.mark.asyncio
async def test_multi_hop_pagination_is_stable_for_ranked_results() -> None:
    """Pagination metadata and page boundaries should stay stable."""
    scenario = build_multi_hop_scenario()
    search_service = _build_search_service(scenario.db, scenario.provider)
    query = "security validation mechanisms"

    page_one, pagination_one = await search_service.search_semantic(query, page_size=2)
    page_two, pagination_two = await search_service.search_semantic(
        query,
        page_size=2,
        offset=2,
    )

    assert [result["chunk_id"] for result in page_one] == [3, 4], (
        f"page 1 expected [3,4], got {[r['chunk_id'] for r in page_one]}"
    )
    assert [result["chunk_id"] for result in page_two] == [2, 1], (
        f"page 2 expected [2,1], got {[r['chunk_id'] for r in page_two]}"
    )
    assert pagination_one == {
        "offset": 0,
        "page_size": 2,
        "has_more": True,
        "next_offset": 2,
        "total": 8,
    }
    assert pagination_two == {
        "offset": 2,
        "page_size": 2,
        "has_more": True,
        "next_offset": 4,
        "total": 8,
    }


@pytest.mark.fast
@pytest.mark.asyncio
async def test_multi_hop_returns_valid_results_when_rerank_fails() -> None:
    """Rerank failure should degrade to deterministic similarity-based results."""
    scenario = build_multi_hop_scenario(fail_rerank=True)
    search_service = _build_search_service(scenario.db, scenario.provider)

    results, pagination = await search_service.search_semantic(
        "security validation mechanisms",
        page_size=3,
    )

    assert [result["chunk_id"] for result in results] == [1, 5, 6], (
        f"rerank-fail fallback expected [1,5,6], got {[r['chunk_id'] for r in results]}"
    )
    assert pagination["total"] >= 5


@pytest.mark.fast
@pytest.mark.asyncio
async def test_multi_hop_terminates_on_result_limit() -> None:
    """Multi-hop stops expanding when result_limit is already met by initial search."""
    scenario = build_multi_hop_scenario()
    search_service = _build_search_service(scenario.db, scenario.provider)

    results, pagination = await search_service.search_semantic(
        "security validation mechanisms",
        page_size=10,
        result_limit=2,
    )

    chunk_ids = {result["chunk_id"] for result in results}
    assert chunk_ids <= {1, 5, 6, 7, 8}, (
        f"result_limit=2 should prevent expansion, got chunks {sorted(chunk_ids)}"
    )
    assert pagination["total"] == 5, (
        f"total should reflect only initial results (5), got {pagination['total']}"
    )


@pytest.mark.fast
@pytest.mark.asyncio
async def test_multi_hop_terminates_on_time_limit() -> None:
    """Multi-hop stops expanding when time limit expires."""
    scenario = build_multi_hop_scenario()
    search_service = _build_search_service(scenario.db, scenario.provider)

    results, pagination = await search_service.search_semantic(
        "security validation mechanisms",
        page_size=10,
        time_limit=0.0,
    )

    chunk_ids = {result["chunk_id"] for result in results}
    assert chunk_ids <= {1, 5, 6, 7, 8}, (
        f"time_limit=0 should prevent expansion, got chunks {sorted(chunk_ids)}"
    )
    assert pagination["total"] == 5, (
        f"total should reflect only initial results (5), got {pagination['total']}"
    )


@pytest.mark.fast
@pytest.mark.asyncio
async def test_multi_hop_terminates_on_insufficient_high_scoring_candidates() -> None:
    """Expansion should stop when fewer than five candidates score above zero."""
    scenario = build_insufficient_candidates_scenario()
    search_service = _build_search_service(scenario.db, scenario.provider)

    results, pagination = await search_service.search_semantic(
        "insufficient candidates",
        page_size=10,
    )

    assert pagination["total"] == 5
    assert {result["chunk_id"] for result in results} == {1, 2, 3, 4, 5}


@pytest.mark.fast
@pytest.mark.asyncio
async def test_multi_hop_terminates_on_score_drop_signal() -> None:
    """Expansion should stop when tracked chunk scores degrade sharply."""
    scenario = build_score_drop_termination_scenario()
    search_service = _build_search_service(scenario.db, scenario.provider)

    results, pagination = await search_service.search_semantic(
        "score drop termination",
        page_size=10,
    )

    assert pagination["total"] == 6
    assert results[0]["chunk_id"] == 6
    assert any(result["chunk_id"] == 1 for result in results)


@pytest.mark.fast
@pytest.mark.asyncio
async def test_multi_hop_terminates_on_low_top_five_floor() -> None:
    """Expansion should stop when the top-five floor drops below 0.3."""
    scenario = build_min_score_termination_scenario()
    search_service = _build_search_service(scenario.db, scenario.provider)

    results, pagination = await search_service.search_semantic(
        "minimum score termination",
        page_size=10,
    )

    assert pagination["total"] == 6
    top_five_scores = [result["score"] for result in results[:5]]
    assert min(top_five_scores) < 0.3


@pytest.mark.fast
@pytest.mark.asyncio
async def test_multi_hop_expansion_exhausts_naturally() -> None:
    """Multi-hop traverses the full synthetic graph without artificial limits."""
    scenario = build_multi_hop_scenario()
    search_service = _build_search_service(scenario.db, scenario.provider)

    results, pagination = await search_service.search_semantic(
        "security validation mechanisms",
        page_size=10,
    )

    chunk_ids = {result["chunk_id"] for result in results}
    assert pagination["total"] == 8, (
        f"full expansion should discover all 8 chunks, got {pagination['total']}"
    )
    assert 2 in chunk_ids, "bridge chunk 2 should be discovered via expansion"
    assert 3 in chunk_ids, "target chunk 3 should be discovered via expansion"
    assert 4 in chunk_ids, "target chunk 4 should be discovered via expansion"


@pytest.mark.fast
@pytest.mark.asyncio
async def test_multi_hop_respects_path_filter_scope(tmp_path: Path) -> None:
    """Semantic search with path_filter should not leak out-of-scope chunks."""
    db = DuckDBProvider(":memory:", base_directory=tmp_path)
    db.connect()

    embedding_provider = FakeEmbeddingProvider()
    coordinator = _build_python_coordinator(db, tmp_path, embedding_provider)

    for repo in ["repo_a", "repo_b"]:
        repo_dir = tmp_path / repo
        repo_dir.mkdir(parents=True, exist_ok=True)
        file_path = repo_dir / "module.py"
        file_path.write_text(
            f"""
def shared_function_{repo}_one():
    \"\"\"Shared multi-hop scope test in {repo}.\"\"\"
    return \"shared-{repo}-one\"

def shared_function_{repo}_two():
    \"\"\"Shared multi-hop scope test in {repo}.\"\"\"
    return \"shared-{repo}-two\"

def shared_function_{repo}_three():
    \"\"\"Shared multi-hop scope test in {repo}.\"\"\"
    return \"shared-{repo}-three\"
""",
            encoding="utf-8",
        )
        await coordinator.process_file(file_path)

    search_service = _build_search_service(
        cast(DatabaseProvider, db),
        cast(EmbeddingProvider, embedding_provider),
    )
    results, _ = await search_service.search_semantic(
        query="multi-hop scope test",
        page_size=10,
        path_filter="repo_a",
    )

    assert results
    assert all(
        result.get("file_path", "").startswith("repo_a/") for result in results
    )


@pytest.mark.fast
@pytest.mark.asyncio
async def test_find_similar_chunks_enforces_path_filter(tmp_path: Path) -> None:
    """DB-level neighbor expansion should enforce path_filter consistently."""
    db = DuckDBProvider(":memory:", base_directory=tmp_path)
    db.connect()

    embedding_provider = FakeEmbeddingProvider()
    coordinator = _build_python_coordinator(db, tmp_path, embedding_provider)

    for repo in ["repo_a", "repo_b"]:
        repo_dir = tmp_path / repo
        repo_dir.mkdir(parents=True, exist_ok=True)
        for index in range(2):
            file_path = repo_dir / f"module_{index}.py"
            file_path.write_text(
                f"""
def repo_function_{index}():
    \"\"\"Repository-specific function {index} for {repo}.\"\"\"
    return \"{repo}-value-{index}\"
""",
                encoding="utf-8",
            )
            await coordinator.process_file(file_path)

    regex_results, _ = db.search_regex(
        pattern="Repository-specific function",
        page_size=50,
    )
    repo_a_chunk = next(
        result
        for result in regex_results
        if result.get("file_path", "").startswith("repo_a/")
    )

    neighbors_unscoped = db.find_similar_chunks(
        chunk_id=repo_a_chunk["chunk_id"],
        provider=embedding_provider.name,
        model=embedding_provider.model,
        limit=20,
        threshold=None,
    )
    assert neighbors_unscoped
    assert any(
        neighbor.get("file_path", "").startswith("repo_b/")
        for neighbor in neighbors_unscoped
    )

    neighbors_scoped = db.find_similar_chunks(
        chunk_id=repo_a_chunk["chunk_id"],
        provider=embedding_provider.name,
        model=embedding_provider.model,
        limit=20,
        threshold=None,
        path_filter="repo_a",
    )
    assert neighbors_scoped
    assert all(
        neighbor.get("file_path", "").startswith("repo_a/")
        for neighbor in neighbors_scoped
    )
