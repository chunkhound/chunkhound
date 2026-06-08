"""Deterministic multi-hop semantic search contract tests."""

from pathlib import Path
from typing import cast
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
    build_graph_exhaustion_scenario,
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
async def test_search_service_selects_single_hop_when_provider_lacks_reranking(
) -> None:
    """Without reranking, search should not expand beyond initial results."""
    scenario = build_multi_hop_scenario(supports_reranking=False)
    search_service = _build_search_service(scenario.db, scenario.provider)
    query = "security validation mechanisms"

    results, pagination = await search_service.search_semantic(query, page_size=10)

    # Single-hop: only initial results (chunks 1,5,6,7,8), no bridge expansion
    assert pagination["total"] == 5, (
        f"non-reranking provider should return only initial results (5), "
        f"got {pagination['total']}"
    )

    # force_strategy="multi_hop" should also fallback to single-hop
    results_forced, pagination_forced = await search_service.search_semantic(
        query,
        page_size=10,
        force_strategy="multi_hop",
    )
    assert pagination_forced["total"] == 5, (
        f"force_strategy multi_hop on non-reranking provider should fallback "
        f"to single-hop (5), got {pagination_forced['total']}"
    )
    assert {r["chunk_id"] for r in results_forced} == {r["chunk_id"] for r in results}


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
async def test_multi_hop_handles_partial_rerank_results() -> None:
    """Partial rerank should keep deterministic fallback ordering and scores."""
    scenario = build_multi_hop_scenario(partial_rerank=True)
    search_service = _build_search_service(scenario.db, scenario.provider)

    results, pagination = await search_service.search_semantic(
        "security validation mechanisms",
        page_size=8,
    )

    assert [result["chunk_id"] for result in results] == [2, 1, 5, 6, 7, 8], (
        "partial rerank should preserve deterministic fallback ordering for "
        "documents without rerank rows"
    )
    assert pagination["total"] == 6
    assert all(
        result.get("score", 0.0) >= 0.0 for result in results
    ), "all results should have valid non-negative scores"


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

    assert pagination["total"] == 6, (
        f"score-drop should stop expansion at 6, not continue to 8, got {pagination['total']}"
    )
    # Chunks 7 and 8 should NOT be discovered (they'd be found if loop continued past score-drop)
    result_ids = {result["chunk_id"] for result in results}
    assert 7 not in result_ids, "chunk 7 should not be discovered if score-drop termination fires"
    assert 8 not in result_ids, "chunk 8 should not be discovered if score-drop termination fires"


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

    assert pagination["total"] == 6, (
        f"min-score should stop expansion at 6, not continue to 8, got {pagination['total']}"
    )
    top_five_scores = [result["score"] for result in results[:5]]
    assert min(top_five_scores) < 0.3
    # Chunks 7 and 8 should NOT be discovered
    result_ids = {result["chunk_id"] for result in results}
    assert 7 not in result_ids, "chunk 7 should not be discovered if min-score termination fires"
    assert 8 not in result_ids, "chunk 8 should not be discovered if min-score termination fires"


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
async def test_search_semantic_enforces_path_filter_component_boundary(
    tmp_path: Path,
) -> None:
    """DB semantic search should not leak sibling directories via substring matches."""
    db = DuckDBProvider(":memory:", base_directory=tmp_path)
    db.connect()

    embedding_provider = FakeEmbeddingProvider()
    coordinator = _build_python_coordinator(db, tmp_path, embedding_provider)

    shared_source = '''def shared_scope_boundary_target():
    """Shared scope boundary sentinel for path-filter regression."""
    return "shared-scope-boundary-target"
'''
    for repo in ["repo_a", "not_repo_a"]:
        repo_dir = tmp_path / repo
        repo_dir.mkdir(parents=True, exist_ok=True)
        file_path = repo_dir / "module.py"
        file_path.write_text(shared_source, encoding="utf-8")
        await coordinator.process_file(file_path)

    regex_results, _ = db.search_regex(
        pattern="scope boundary sentinel",
        page_size=20,
    )
    repo_a_chunk = next(
        result
        for result in regex_results
        if result.get("file_path", "").startswith("repo_a/")
    )
    query_embedding = await embedding_provider.embed_single(repo_a_chunk["content"])

    unscoped_results, _ = db.search_semantic(
        query_embedding=query_embedding,
        provider=embedding_provider.name,
        model=embedding_provider.model,
        page_size=20,
    )
    assert any(
        result.get("file_path", "").startswith("not_repo_a/")
        for result in unscoped_results
    )

    scoped_results, _ = db.search_semantic(
        query_embedding=query_embedding,
        provider=embedding_provider.name,
        model=embedding_provider.model,
        page_size=20,
        path_filter="repo_a",
    )
    assert scoped_results
    assert all(
        result.get("file_path", "").startswith("repo_a/")
        for result in scoped_results
    )


@pytest.mark.fast
@pytest.mark.asyncio
async def test_search_regex_enforces_path_filter_component_boundary(
    tmp_path: Path,
) -> None:
    """Regex search should not leak sibling directories via substring matches."""
    db = DuckDBProvider(":memory:", base_directory=tmp_path)
    db.connect()

    embedding_provider = FakeEmbeddingProvider()
    coordinator = _build_python_coordinator(db, tmp_path, embedding_provider)

    shared_source = '''def shared_scope_boundary_target():
    """Shared scope boundary sentinel for path-filter regression."""
    return "shared-scope-boundary-target"
'''
    for repo in ["repo_a", "not_repo_a"]:
        repo_dir = tmp_path / repo
        repo_dir.mkdir(parents=True, exist_ok=True)
        file_path = repo_dir / "module.py"
        file_path.write_text(shared_source, encoding="utf-8")
        await coordinator.process_file(file_path)

    unscoped_results, _ = db.search_regex(
        pattern="scope boundary sentinel",
        page_size=20,
    )
    assert any(
        result.get("file_path", "").startswith("not_repo_a/")
        for result in unscoped_results
    )

    scoped_results, _ = db.search_regex(
        pattern="scope boundary sentinel",
        page_size=20,
        path_filter="repo_a",
    )
    assert scoped_results
    assert all(
        result.get("file_path", "").startswith("repo_a/")
        for result in scoped_results
    )


@pytest.mark.fast
@pytest.mark.asyncio
async def test_search_by_embedding_enforces_path_filter_component_boundary(
    tmp_path: Path,
) -> None:
    """Embedding search should not leak sibling directories via substring matches."""
    db = DuckDBProvider(":memory:", base_directory=tmp_path)
    db.connect()

    embedding_provider = FakeEmbeddingProvider()
    coordinator = _build_python_coordinator(db, tmp_path, embedding_provider)

    shared_source = '''def shared_scope_boundary_target():
    """Shared scope boundary sentinel for path-filter regression."""
    return "shared-scope-boundary-target"
'''
    for repo in ["repo_a", "not_repo_a"]:
        repo_dir = tmp_path / repo
        repo_dir.mkdir(parents=True, exist_ok=True)
        file_path = repo_dir / "module.py"
        file_path.write_text(shared_source, encoding="utf-8")
        await coordinator.process_file(file_path)

    regex_results, _ = db.search_regex(
        pattern="scope boundary sentinel",
        page_size=20,
    )
    repo_a_chunk = next(
        result
        for result in regex_results
        if result.get("file_path", "").startswith("repo_a/")
    )
    query_embedding = await embedding_provider.embed_single(repo_a_chunk["content"])

    unscoped_results = db.search_by_embedding(
        query_embedding=query_embedding,
        provider=embedding_provider.name,
        model=embedding_provider.model,
        limit=20,
    )
    assert any(
        result.get("file_path", "").startswith("not_repo_a/")
        for result in unscoped_results
    )

    scoped_results = db.search_by_embedding(
        query_embedding=query_embedding,
        provider=embedding_provider.name,
        model=embedding_provider.model,
        limit=20,
        path_filter="repo_a",
    )
    assert scoped_results
    assert all(
        result.get("file_path", "").startswith("repo_a/")
        for result in scoped_results
    )


@pytest.mark.fast
@pytest.mark.asyncio
async def test_find_similar_chunks_enforces_path_filter(tmp_path: Path) -> None:
    """DB-level neighbor expansion should enforce path_filter consistently."""
    db = DuckDBProvider(":memory:", base_directory=tmp_path)
    db.connect()

    embedding_provider = FakeEmbeddingProvider()
    coordinator = _build_python_coordinator(db, tmp_path, embedding_provider)

    for repo in ["repo_a", "not_repo_a"]:
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
        neighbor.get("file_path", "").startswith("not_repo_a/")
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
