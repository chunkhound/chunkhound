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
    assert all(result.get("score", 0.0) >= 0.0 for result in results), (
        "all results should have valid non-negative scores"
    )

    # Verify fallback: ch2 (expansion index 5) is omitted from expansion rerank
    # via partial_rerank, so it should retain its original similarity score.
    ch2_result = next(r for r in results if r["chunk_id"] == 2)
    assert ch2_result.get("score", 0.0) == pytest.approx(0.70, rel=1e-3), (
        "ch2 should retain similarity score when omitted from partial rerank"
    )


@pytest.mark.fast
@pytest.mark.asyncio
async def test_search_regex_normalizes_path_filter_inputs(
    tmp_path: Path,
) -> None:
    """Equivalent path filter inputs should scope identically.

    Covers whitespace-only, backslash, leading-slash, and trailing-slash variants.
    """
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
    unscoped_ids = {result["chunk_id"] for result in unscoped_results}
    assert len(unscoped_ids) == 2

    repo_a_results, _ = db.search_regex(
        pattern="scope boundary sentinel",
        page_size=20,
        path_filter="repo_a",
    )
    repo_a_ids = {result["chunk_id"] for result in repo_a_results}
    assert len(repo_a_ids) == 1
    assert all(
        result.get("file_path", "").startswith("repo_a/") for result in repo_a_results
    )

    # Whitespace-only filter is equivalent to no filter
    whitespace_results, _ = db.search_regex(
        pattern="scope boundary sentinel",
        page_size=20,
        path_filter="   ",
    )
    assert {result["chunk_id"] for result in whitespace_results} == unscoped_ids

    # All non-whitespace variants targeting the same directory should be equivalent
    scoped_variants = {
        "backslash": "  \\repo_a\\  ",
        "leading_slash": "/repo_a",
        "trailing_slash": "repo_a/",
    }
    for variant in scoped_variants.values():
        scoped_results, _ = db.search_regex(
            pattern="scope boundary sentinel",
            page_size=20,
            path_filter=variant,
        )
        scoped_ids = {result["chunk_id"] for result in scoped_results}
        assert scoped_ids == repo_a_ids, (
            f"variant {variant!r} should match canonical repo_a scope, got {scoped_ids}"
        )
        assert all(
            result.get("file_path", "").startswith("repo_a/")
            for result in scoped_results
        )


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
    assert len(results) == 2, (
        f"result_limit=2 should cap returned count to 2, got {len(results)}"
    )
    assert pagination["total"] == 2, (
        f"result_limit=2 should cap total to 2, got {pagination['total']}"
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
async def test_multi_hop_terminates_when_no_new_candidates() -> None:
    """Multi-hop should stop once no new top candidates remain."""
    scenario = build_graph_exhaustion_scenario()
    search_service = _build_search_service(scenario.db, scenario.provider)

    results, pagination = await search_service.search_semantic(
        "graph exhaustion",
        page_size=10,
    )

    assert pagination["total"] == 6, (
        f"graph exhaustion should stop at 6, got {pagination['total']}"
    )
    result_ids = {result["chunk_id"] for result in results}
    assert 6 in result_ids, "chunk 6 should be discovered via expansion"
    assert 7 not in result_ids, (
        "chunk 7 should not be discovered when no new candidates"
    )
    assert 8 not in result_ids, (
        "chunk 8 should not be discovered when no new candidates"
    )


@pytest.mark.fast
@pytest.mark.asyncio
async def test_multi_hop_terminates_on_score_drop() -> None:
    """Expansion should stop on a tracked score drop of at least 0.15."""
    scenario = build_score_drop_termination_scenario()
    search_service = _build_search_service(scenario.db, scenario.provider)

    results, pagination = await search_service.search_semantic(
        "score drop termination",
        page_size=10,
    )

    assert pagination["total"] == 6, (
        f"score-drop should stop expansion at 6, got {pagination['total']}"
    )
    result_ids = {result["chunk_id"] for result in results}
    assert 6 in result_ids, "chunk 6 should be discovered via expansion"
    assert 7 not in result_ids, (
        "chunk 7 should not be discovered when score-drop termination fires"
    )
    assert 8 not in result_ids, (
        "chunk 8 should not be discovered when score-drop termination fires"
    )


@pytest.mark.fast
@pytest.mark.asyncio
async def test_multi_hop_score_drop_fixture_is_repeatable_across_searches() -> None:
    """Repeated searches on one scenario should not inherit rerank state."""
    scenario = build_score_drop_termination_scenario()
    search_service = _build_search_service(scenario.db, scenario.provider)
    query = "score drop termination"

    first_results, first_pagination = await search_service.search_semantic(
        query,
        page_size=10,
    )
    second_results, second_pagination = await search_service.search_semantic(
        query,
        page_size=10,
    )

    assert first_pagination == second_pagination
    assert [result["chunk_id"] for result in first_results] == [
        result["chunk_id"] for result in second_results
    ]
    assert [result["score"] for result in first_results] == [
        result["score"] for result in second_results
    ]


@pytest.mark.fast
@pytest.mark.asyncio
async def test_multi_hop_terminates_on_score_drop_threshold_boundary() -> None:
    """A score drop slightly above the threshold should terminate expansion.

    Uses round_two_chunk_one_score=0.79 so 0.95 - 0.79 = 0.16 is safely above
    SCORE_DROP_THRESHOLD=0.15 in floating point (avoiding 0.95-0.80 precision).
    """
    scenario = build_score_drop_termination_scenario(
        round_two_chunk_one_score=0.79,
        query="score drop threshold boundary",
    )
    search_service = _build_search_service(scenario.db, scenario.provider)

    results, pagination = await search_service.search_semantic(
        "score drop threshold boundary",
        page_size=10,
    )

    assert pagination["total"] == 6
    assert 7 not in {result["chunk_id"] for result in results}


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
        "min-score should stop expansion at 6, "
        f"not continue to 8, got {pagination['total']}"
    )
    top_five_scores = [result["score"] for result in results[:5]]
    assert min(top_five_scores) < 0.3
    # Chunks 7 and 8 should NOT be discovered
    result_ids = {result["chunk_id"] for result in results}
    assert 7 not in result_ids, (
        "chunk 7 should not be discovered if min-score termination fires"
    )
    assert 8 not in result_ids, (
        "chunk 8 should not be discovered if min-score termination fires"
    )


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
        result.get("file_path", "").startswith("repo_a/") for result in scoped_results
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
        result.get("file_path", "").startswith("repo_a/") for result in scoped_results
    )


@pytest.mark.fast
@pytest.mark.asyncio
async def test_path_filter_like_patterns_against_duckdb(tmp_path: Path) -> None:
    """LIKE patterns from path_filter must be correct against real DuckDB.

    Covers two scenarios the refactoring fixed:
      1. Directory boundary: path_filter="src/lib" must match src/lib/ but NOT
         src/lib2/ (the pattern %/src/lib/% is bounded on the right).
      2. Right-anchored file patterns: path_filter="module.py" must match
         module.py but NOT module.py.bak.

    Exercises both search_regex and search_semantic to cover all 4 executor
    methods that call _build_path_like_pattern.
    """
    db = DuckDBProvider(":memory:", base_directory=tmp_path)
    db.connect()

    embedding_provider = FakeEmbeddingProvider()
    coordinator = _build_python_coordinator(db, tmp_path, embedding_provider)

    shared_source = '''def path_filter_like_target():
    """Sentinel for LIKE pattern correctness tests."""
    return "path-filter-like-target"
'''

    # Directory boundary test: create files under lib/ vs lib2/
    (tmp_path / "src/lib").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src/lib2").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src/module").mkdir(parents=True, exist_ok=True)

    lib_file = tmp_path / "src/lib/util.py"
    lib_file.write_text(shared_source, encoding="utf-8")
    lib2_file = tmp_path / "src/lib2/util.py"
    lib2_file.write_text(shared_source, encoding="utf-8")
    await coordinator.process_file(lib_file)
    await coordinator.process_file(lib2_file)

    # Right-anchored file pattern test: module.py vs module.py.bak
    py_file = tmp_path / "src/module/module.py"
    py_file.write_text(shared_source, encoding="utf-8")
    bak_file = tmp_path / "src/module/module.py.bak"
    bak_file.write_text(shared_source, encoding="utf-8")
    await coordinator.process_file(py_file)
    await coordinator.process_file(bak_file)

    # ── Directory boundary: path_filter="src/lib" ──
    lib_regex_results, _ = db.search_regex(
        pattern="path-filter-like-target",
        page_size=20,
        path_filter="src/lib",
    )
    assert all(
        result.get("file_path", "").startswith("src/lib/")
        for result in lib_regex_results
    ), f"src/lib filter matched wrong paths: {[r['file_path'] for r in lib_regex_results]}"
    assert any(
        result.get("file_path", "") == "src/lib/util.py" for result in lib_regex_results
    ), "src/lib filter should match src/lib/util.py"

    # ── Right-anchored file pattern: path_filter="module.py" ──
    py_regex_results, _ = db.search_regex(
        pattern="path-filter-like-target",
        page_size=20,
        path_filter="module.py",
    )
    # Must NOT match module.py.bak
    assert not any(
        result.get("file_path", "").endswith(".bak") for result in py_regex_results
    ), (
        "path_filter='module.py' matched .bak file: "
        f"{[r['file_path'] for r in py_regex_results]}"
    )
    assert any(
        result.get("file_path", "") == "src/module/module.py"
        for result in py_regex_results
    ), "path_filter='module.py' should match module.py"

    # ── Also verify via search_semantic (exercises a different executor) ──
    query_embedding = await embedding_provider.embed_single(
        "path-filter-like-target"
    )

    lib_sem_results, _ = db.search_semantic(
        query_embedding=query_embedding,
        provider=embedding_provider.name,
        model=embedding_provider.model,
        page_size=20,
        path_filter="src/lib",
    )
    assert all(
        result.get("file_path", "").startswith("src/lib/")
        for result in lib_sem_results
    ), f"semantic src/lib filter leaked: {[r['file_path'] for r in lib_sem_results]}"

    py_sem_results, _ = db.search_semantic(
        query_embedding=query_embedding,
        provider=embedding_provider.name,
        model=embedding_provider.model,
        page_size=20,
        path_filter="module.py",
    )
    assert not any(
        result.get("file_path", "").endswith(".bak") for result in py_sem_results
    ), "semantic path_filter='module.py' matched .bak file"


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
