"""Real-DuckDB contracts for HNSW-backed semantic search."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np
import pytest
from loguru import logger

from chunkhound.api.cli.commands.search import _format_search_results
from chunkhound.core.config.database_config import DatabaseConfig
from chunkhound.core.constants import HNSW_CANDIDATE_BUDGET
from chunkhound.core.models import Chunk
from chunkhound.core.types.common import ChunkType, FileId, Language, LineNumber
from chunkhound.embeddings import LocalEmbeddingResult
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.diff_aware_search_service import DiffAwareSearchService
from chunkhound.services.search_service import SearchService
from tests.fixtures.fake_providers import FakeEmbeddingProvider
from tests.helpers.duckdb_vector_fixtures import (
    MODEL,
    PROVIDER,
    QUERY_VECTOR,
    connected_provider,
    has_hnsw_index,
    insert_chunks,
    insert_embeddings,
    require_hnsw_index,
    seed_searchable_chunks,
)


@pytest.fixture
def provider(tmp_path: Path) -> Generator[DuckDBProvider, None, None]:
    """Provide a file-backed database so DuckDB can persist its HNSW index."""
    with connected_provider(tmp_path) as database:
        yield database


def _seed_dense_table(
    provider: DuckDBProvider, count: int, path: str = "src/module.py"
) -> list[int]:
    """Seed a large single-provider table of close-to-query vectors.

    Used by the plan-shape tests so the candidate query runs against a
    realistic cardinality instead of a 3-row toy table.
    """
    chunk_ids = insert_chunks(provider, path, count)
    insert_embeddings(
        provider,
        chunk_ids,
        [[1.0, 0.001 * ((index % 50) + 1) / 50.0, 0.0] for index in range(count)],
    )
    require_hnsw_index(provider)
    return chunk_ids


@pytest.mark.hnsw
@pytest.mark.fast
def test_hnsw_eligible_candidate_query_uses_index_scan(
    provider: DuckDBProvider,
) -> None:
    """The candidate-query helper retains DuckDB's HNSW-eligible shape.

    EXPLAIN covers the helper query directly; the public call separately guards
    that semantic search remains functional against the same indexed table.
    """
    _seed_dense_table(provider, 2000)

    results, _ = provider.search_semantic(QUERY_VECTOR, PROVIDER, MODEL, page_size=10)
    query = DuckDBProvider._hnsw_eligible_candidate_query("embeddings_3", 3, 100)
    catalog = provider.execute_query(
        "SELECT index_name, sql FROM duckdb_indexes() "
        "WHERE table_name = 'embeddings_3'",
        [],
    )
    plan = provider.execute_query(f"EXPLAIN {query}", [QUERY_VECTOR, 100])

    assert len(results) == 10
    assert any("USING HNSW" in (row["sql"] or "").upper() for row in catalog)
    assert "ORDER BY DISTANCE ASC" in query.upper()
    assert "ORDER BY DISTANCE ASC, CHUNK_ID" not in query.upper()
    assert "HNSW_INDEX_SCAN" in "\n".join(str(row) for row in plan)


@pytest.mark.hnsw
@pytest.mark.fast
def test_mixed_provider_model_subset_is_complete(provider: DuckDBProvider) -> None:
    """Filtering to one provider must not starve far-but-valid matches.

    Pre-fix this returned 0 results: the HNSW beam fills with the closer
    noise rows, the Python provider filter empties the page, and the old
    early-exit read that as table exhaustion.
    """
    noise_ids = insert_chunks(provider, "noise/module.py", 30)
    target_ids = insert_chunks(provider, "target/module.py", 3)
    insert_embeddings(
        provider,
        noise_ids,
        [[0.999, 0.001, 0.0]] * len(noise_ids),
        provider_name="noise",
        model="m",
    )
    insert_embeddings(
        provider,
        target_ids,
        [[0.1, 0.2, 0.3]] * len(target_ids),
        provider_name="target",
        model="m",
    )

    results, _ = provider.search_semantic(QUERY_VECTOR, "target", "m", page_size=3)

    assert {result["chunk_id"] for result in results} == set(target_ids)
    assert all(result["file_path"] == "target/module.py" for result in results)

    by_embedding = provider.search_by_embedding(QUERY_VECTOR, "target", "m", limit=3)
    assert {result["chunk_id"] for result in by_embedding} == set(target_ids)


@pytest.mark.hnsw
@pytest.mark.fast
def test_direct_vector_apis_honor_exact_limit(provider: DuckDBProvider) -> None:
    """Direct vector APIs never return widened overfetch rows."""
    chunk_ids = seed_searchable_chunks(provider, count=5)

    by_embedding = provider.search_by_embedding(QUERY_VECTOR, PROVIDER, MODEL, limit=2)
    similar = provider.find_similar_chunks(chunk_ids[0], PROVIDER, MODEL, limit=2)

    assert len(by_embedding) == 2
    assert len(similar) == 2
    assert chunk_ids[0] not in {result["chunk_id"] for result in similar}


@pytest.mark.hnsw
@pytest.mark.fast
def test_overfetch_fills_page_after_initial_path_filter_miss(
    provider: DuckDBProvider,
) -> None:
    """Path filtering must widen beyond an initially all-noise HNSW batch."""
    noise_ids = insert_chunks(provider, "noise/module.py", 30)
    target_ids = insert_chunks(provider, "target/module.py", 5)
    insert_embeddings(provider, noise_ids, [[1.0, 0.001, 0.0]] * len(noise_ids))
    insert_embeddings(provider, target_ids, [[0.8, 0.6, 0.0]] * len(target_ids))

    results, _ = provider.search_semantic(
        QUERY_VECTOR, PROVIDER, MODEL, page_size=5, path_filter="target"
    )

    assert {result["chunk_id"] for result in results} == set(target_ids)
    assert all(result["file_path"] == "target/module.py" for result in results)


@pytest.mark.fast
def test_overfetch_telemetry_uses_full_filtered_candidate_batch(
    provider: DuckDBProvider,
) -> None:
    """Selectivity measures filtering before result truncation to the requested count."""
    telemetry: list[dict[str, object]] = []
    sink_id = logger.add(
        lambda message: telemetry.append(dict(message.record["extra"])),
        level="DEBUG",
        filter=lambda record: record["message"]
        == "Vector candidate overfetch terminated",
    )
    batch = [{"chunk_id": chunk_id} for chunk_id in range(5)]
    try:
        results, _ = provider._overfetch_vector_results(
            needed=2, execute_query=lambda _limit: (batch, 10)
        )
    finally:
        logger.remove(sink_id)

    assert len(results) == 2
    assert telemetry[-1]["results"] == 2
    assert telemetry[-1]["filtered_candidates"] == 5
    assert telemetry[-1]["filter_selectivity"] == 0.5


@pytest.mark.hnsw
@pytest.mark.fast
def test_filtered_search_short_page_at_candidate_budget(
    provider: DuckDBProvider,
) -> None:
    """Filtered ANN retrieval returns available matches at its candidate budget."""
    noise_count = 9_997
    noise_vectors = []
    for index in range(noise_count):
        distance = 0.001 + 0.5 * (index / noise_count)
        noise_vectors.append(
            [math.sqrt(max(0.0, 1.0 - distance * distance)), distance, 0.0]
        )
    noise_ids = insert_chunks(provider, "noise/module.py", noise_count)
    target_ids = insert_chunks(provider, "target/module.py", 3)
    insert_embeddings(
        provider,
        noise_ids,
        noise_vectors,
        provider_name="noise",
        model="m",
    )
    insert_embeddings(
        provider,
        target_ids,
        [[0.1, 0.2, 0.3]] * len(target_ids),
        provider_name="target",
        model="m",
    )
    # Widen the search beam so every overfetch step returns its full limit.
    provider.execute_query("SET hnsw_ef_search = 20000", [])

    results, pagination = provider.search_semantic(
        QUERY_VECTOR, "target", "m", page_size=10
    )

    assert {result["chunk_id"] for result in results} == set(target_ids)
    assert len(results) == 3
    assert pagination == {
        "offset": 0,
        "page_size": 10,
        "has_more": False,
        "next_offset": None,
        "total": None,
        "candidate_budget_exhausted": True,
    }
    by_embedding = provider.search_by_embedding(QUERY_VECTOR, "target", "m", limit=10)
    similar = provider.find_similar_chunks(target_ids[0], "target", "m", limit=10)
    assert {result["chunk_id"] for result in by_embedding} == set(target_ids)
    assert {result["chunk_id"] for result in similar} == set(target_ids) - {
        target_ids[0]
    }


@pytest.mark.hnsw
@pytest.mark.fast
def test_list_api_raises_when_budget_exhausted_with_zero_results(
    provider: DuckDBProvider,
) -> None:
    """List-only vector APIs fail loudly when budget exhaustion yields nothing.

    A zero-result page under HNSW candidate-budget exhaustion means the beam
    never reached a matching row. The paginated API reports it as metadata;
    the list-only APIs must raise so callers cannot mistake it for a genuine
    empty result.
    """
    noise_count = HNSW_CANDIDATE_BUDGET
    noise_vectors = []
    for index in range(noise_count):
        distance = 0.001 + 0.5 * (index / noise_count)
        noise_vectors.append(
            [math.sqrt(max(0.0, 1.0 - distance * distance)), distance, 0.0]
        )
    noise_ids = insert_chunks(provider, "noise/module.py", noise_count)
    insert_embeddings(
        provider,
        noise_ids,
        noise_vectors,
        provider_name="noise",
        model="m",
    )
    require_hnsw_index(provider)
    # Widen the search beam so every overfetch step returns its full limit and
    # the candidate budget is genuinely exhausted at the final scan.
    provider.execute_query("SET hnsw_ef_search = 20000", [])

    with pytest.raises(RuntimeError, match="exhausted the HNSW candidate budget"):
        provider.search_by_embedding(QUERY_VECTOR, "missing", "x", limit=10)

    # find_similar_chunks needs the searched chunk to exist under the target
    # provider/model; its own row is excluded and the noise rows are filtered,
    # so exhaustion still yields zero results.
    target_ids = insert_chunks(provider, "target/module.py", 1)
    insert_embeddings(
        provider,
        target_ids,
        [[0.9, 0.3, 0.1]],
        provider_name="missing",
        model="x",
    )
    with pytest.raises(RuntimeError, match="exhausted the HNSW candidate budget"):
        provider.find_similar_chunks(target_ids[0], "missing", "x", limit=10)


@pytest.mark.hnsw
@pytest.mark.fast
def test_semantic_pagination_uses_one_extra_result_without_total(
    provider: DuckDBProvider,
) -> None:
    """Semantic pagination reports only look-ahead state, not a full-scan total."""
    seed_searchable_chunks(provider)

    first_page, first_pagination = provider.search_semantic(
        QUERY_VECTOR, PROVIDER, MODEL, page_size=2
    )
    second_page, second_pagination = provider.search_semantic(
        QUERY_VECTOR, PROVIDER, MODEL, page_size=2, offset=2
    )

    assert len(first_page) == 2
    assert first_pagination == {
        "offset": 0,
        "page_size": 2,
        "has_more": True,
        "next_offset": 2,
        "total": None,
        "candidate_budget_exhausted": False,
    }
    assert len(second_page) == 1
    assert second_pagination == {
        "offset": 2,
        "page_size": 2,
        "has_more": False,
        "next_offset": None,
        "total": None,
        "candidate_budget_exhausted": False,
    }


@pytest.mark.hnsw
@pytest.mark.fast
def test_semantic_pagination_partitions_continuously(
    provider: DuckDBProvider,
) -> None:
    """Consecutive pages continue exactly where the previous page stopped."""
    count = 7
    chunk_ids = seed_searchable_chunks(provider, count)

    seen: list[int] = []
    offset = 0
    while True:
        page, pagination = provider.search_semantic(
            QUERY_VECTOR, PROVIDER, MODEL, page_size=3, offset=offset
        )
        seen.extend(result["chunk_id"] for result in page)
        if not pagination["has_more"]:
            break
        offset = pagination["next_offset"]

    assert len(seen) == len(set(seen)) == count
    assert set(seen) == set(chunk_ids)


@pytest.mark.fast
def test_missing_embeddings_table_contract(provider: DuckDBProvider) -> None:
    """Vector searches against an empty DB report an empty page, not an error."""
    results, pagination = provider.search_semantic(
        QUERY_VECTOR, PROVIDER, MODEL, page_size=10
    )

    assert results == []
    assert pagination == {
        "offset": 0,
        "page_size": 10,
        "has_more": False,
        "next_offset": None,
        "total": None,
        "candidate_budget_exhausted": False,
    }
    assert provider.search_by_embedding(QUERY_VECTOR, PROVIDER, MODEL, limit=10) == []
    assert provider.find_similar_chunks(1, PROVIDER, MODEL, limit=10) == []


@pytest.mark.fast
@pytest.mark.parametrize("query_embedding", [[float("nan")], [float("inf")]])
def test_missing_embeddings_table_still_rejects_non_finite_vectors(
    provider: DuckDBProvider, query_embedding: list[float]
) -> None:
    """Invalid vectors fail before an unknown dimension returns an empty result."""
    with pytest.raises(ValueError, match="non-finite"):
        provider.search_semantic(query_embedding, PROVIDER, MODEL, page_size=1)
    with pytest.raises(ValueError, match="non-finite"):
        provider.search_by_embedding(query_embedding, PROVIDER, MODEL, limit=1)


Search = Callable[[DuckDBProvider, list[int]], object]


def _search_semantic(provider: DuckDBProvider, _: list[int]) -> object:
    return provider.search_semantic(QUERY_VECTOR, PROVIDER, MODEL, page_size=1)


def _search_by_embedding(provider: DuckDBProvider, _: list[int]) -> object:
    return provider.search_by_embedding(QUERY_VECTOR, PROVIDER, MODEL, limit=1)


def _find_similar(provider: DuckDBProvider, chunk_ids: list[int]) -> object:
    return provider.find_similar_chunks(chunk_ids[0], PROVIDER, MODEL, limit=1)


@pytest.mark.hnsw
@pytest.mark.fast
@pytest.mark.parametrize(
    "search", [_search_semantic, _search_by_embedding, _find_similar]
)
def test_vector_search_recovers_missing_hnsw_index(
    provider: DuckDBProvider, search: Search
) -> None:
    """Every public vector API restores an index lost during interrupted indexing."""
    chunk_ids = seed_searchable_chunks(provider)
    provider.drop_all_hnsw_indexes()
    assert not has_hnsw_index(provider, 3)

    search(provider, chunk_ids)

    assert has_hnsw_index(provider, 3)


@pytest.mark.hnsw
@pytest.mark.fast
def test_non_cosine_hnsw_is_preserved_and_cosine_index_is_added(
    provider: DuckDBProvider,
) -> None:
    """Cosine search must not mistake another HNSW metric for compatibility."""
    seed_searchable_chunks(provider)
    provider.drop_all_hnsw_indexes()
    provider.execute_query(
        "CREATE INDEX idx_hnsw_3 ON embeddings_3 USING HNSW (embedding) "
        "WITH (metric = 'l2sq')",
        [],
    )
    before = provider.execute_query(
        "SELECT index_name, metric FROM pragma_hnsw_index_info() "
        "WHERE table_name = 'embeddings_3'",
        [],
    )

    results, _ = provider.search_semantic(QUERY_VECTOR, PROVIDER, MODEL, page_size=1)
    after = provider.execute_query(
        "SELECT index_name, metric FROM pragma_hnsw_index_info() "
        "WHERE table_name = 'embeddings_3'",
        [],
    )
    candidate_query = DuckDBProvider._hnsw_eligible_candidate_query(
        "embeddings_3", 3, 1
    )
    plan = provider.execute_query(f"EXPLAIN {candidate_query}", [QUERY_VECTOR, 1])

    assert results
    assert {(row["index_name"], row["metric"]) for row in before} <= {
        (row["index_name"], row["metric"]) for row in after
    }
    assert {row["metric"] for row in after} >= {"l2sq", "cosine"}
    assert "HNSW_INDEX_SCAN" in "\n".join(str(row) for row in plan)


@pytest.mark.hnsw
@pytest.mark.fast
def test_drop_vector_index_also_drops_cosine_variant(
    provider: DuckDBProvider,
) -> None:
    """Dropping a custom index must also drop the lazily-created _cosine variant.

    A non-cosine custom index occupies the canonical name, so the cosine
    index gets the idx_hnsw_{dims}_cosine name; drop_vector_index must remove
    it too or it leaks and the table keeps an orphan index.
    """
    seed_searchable_chunks(provider)
    provider.drop_all_hnsw_indexes()
    provider.create_vector_index(PROVIDER, MODEL, 3, "l2sq")

    provider.search_semantic(QUERY_VECTOR, PROVIDER, MODEL, page_size=1)
    provider.drop_vector_index(PROVIDER, MODEL, 3, "l2sq")

    remaining = provider.execute_query(
        "SELECT index_name FROM pragma_hnsw_index_info() "
        "WHERE table_name = 'embeddings_3'",
        [],
    )
    assert remaining == [], f"indexes leaked after drop_vector_index: {remaining}"


class _FakeEmbeddingManager:
    """Minimal EmbeddingManager stand-in returning a fixed query vector."""

    def __init__(self, query_vector: list[float]) -> None:
        self._query_vector = query_vector

    async def embed_texts(self, texts: list[str]) -> LocalEmbeddingResult:
        return LocalEmbeddingResult(
            embeddings=[self._query_vector] * len(texts),
            model="test",
            provider="test",
            dims=len(self._query_vector),
        )


@pytest.mark.hnsw
@pytest.mark.fast
@pytest.mark.asyncio
async def test_both_mode_diff_search_clamps_db_window(
    provider: DuckDBProvider,
) -> None:
    """Diff-aware 'both' mode must keep its DB fetch inside the offset window.

    Pre-fix this raised ValueError: the diff service fetched DB results with
    page_size=10_000 while the provider caps semantic windows at 1000.
    """
    chunk_ids = seed_searchable_chunks(provider, count=3)
    diff_chunks = [
        Chunk(
            symbol="new_function",
            start_line=LineNumber(1),
            end_line=LineNumber(3),
            code="def new_function(): pass",
            chunk_type=ChunkType.FUNCTION,
            file_id=FileId(999),
            language=Language.PYTHON,
            file_path="src/diff.py",  # type: ignore[arg-type]
        )
    ]
    original = SearchService(provider, FakeEmbeddingProvider(dims=3))
    svc = DiffAwareSearchService(
        original,
        diff_chunks,
        [[1.0, 0.0, 0.0]],
        "both",
        _FakeEmbeddingManager([1.0, 0.0, 0.0]),
    )

    results, pagination = await svc.search_semantic(
        query="query", page_size=2, provider=PROVIDER, model=MODEL
    )

    db_chunk_ids = [
        int(r["chunk_id"]) for r in results if isinstance(r.get("chunk_id"), int)
    ]
    assert db_chunk_ids, "merged results must include DB-backed rows"
    assert set(db_chunk_ids) <= set(chunk_ids)
    assert any(str(r.get("chunk_id")).startswith("diff:") for r in results)
    assert pagination["total"] is None


@pytest.mark.hnsw
@pytest.mark.fast
def test_missing_hnsw_in_read_only_database_fails_explicitly(tmp_path: Path) -> None:
    """HNSW recovery must not degrade to a scan when writes are forbidden."""
    db_path = tmp_path / "read-only-search.duckdb"
    writer = DuckDBProvider(db_path, base_directory=tmp_path)
    writer.connect()
    seed_searchable_chunks(writer)
    writer.drop_all_hnsw_indexes()
    writer.execute_query("CHECKPOINT", [])
    writer.disconnect(skip_checkpoint=True)

    reader = DuckDBProvider(
        db_path,
        base_directory=tmp_path,
        config=DatabaseConfig(read_only=True),
    )
    reader.connect()
    try:
        with pytest.raises(duckdb.Error):
            reader.search_semantic(QUERY_VECTOR, PROVIDER, MODEL, page_size=1)
    finally:
        reader.disconnect(skip_checkpoint=True)


@pytest.mark.parametrize("page_size", [0, -1])
@pytest.mark.fast
def test_semantic_search_rejects_nonpositive_page_size(
    provider: DuckDBProvider, page_size: int
) -> None:
    """Semantic pages must request at least one result."""
    with pytest.raises(ValueError, match="page_size must be greater than 0"):
        provider.search_semantic(
            QUERY_VECTOR, PROVIDER, MODEL, page_size=page_size, offset=0
        )


@pytest.mark.fast
def test_semantic_search_rejects_negative_offset(provider: DuckDBProvider) -> None:
    """Semantic offsets cannot precede the first result."""
    with pytest.raises(ValueError, match="offset must be greater than or equal to 0"):
        provider.search_semantic(QUERY_VECTOR, PROVIDER, MODEL, page_size=1, offset=-1)


@pytest.mark.parametrize(("offset", "page_size"), [(1000, 1), (999, 2), (1001, 1)])
@pytest.mark.fast
def test_semantic_offset_window_rejects_crossing_cap(
    provider: DuckDBProvider, offset: int, page_size: int
) -> None:
    """Semantic windows cannot start at or cross the exclusive offset cap."""
    with pytest.raises(ValueError, match="result window.*1000"):
        provider.search_semantic(
            QUERY_VECTOR,
            PROVIDER,
            MODEL,
            page_size=page_size,
            offset=offset,
        )


@pytest.mark.fast
def test_semantic_offset_window_accepts_exact_endpoint(
    provider: DuckDBProvider,
) -> None:
    """A page ending exactly at the cap is valid and cannot advertise offset 1000."""
    results, pagination = provider.search_semantic(
        QUERY_VECTOR, PROVIDER, MODEL, page_size=100, offset=900
    )

    assert results == []
    assert pagination == {
        "offset": 900,
        "page_size": 100,
        "has_more": False,
        "next_offset": None,
        "total": None,
        "candidate_budget_exhausted": False,
    }


@pytest.mark.hnsw
@pytest.mark.fast
def test_semantic_page_ending_at_cap_never_advertises_next_offset(
    provider: DuckDBProvider,
) -> None:
    """A materialized row beyond the cap cannot make offset 1000 addressable."""
    seed_searchable_chunks(provider, count=1001)

    results, pagination = provider.search_semantic(
        QUERY_VECTOR, PROVIDER, MODEL, page_size=1, offset=999
    )

    assert len(results) == 1
    assert pagination == {
        "offset": 999,
        "page_size": 1,
        "has_more": False,
        "next_offset": None,
        "total": None,
        "candidate_budget_exhausted": False,
    }


@pytest.mark.hnsw
@pytest.mark.fast
def test_vector_threshold_units_match_each_public_api(
    provider: DuckDBProvider,
) -> None:
    """Every public vector API applies threshold as an inclusive similarity floor."""
    chunk_ids = insert_chunks(provider, "src/threshold.py", 4)
    insert_embeddings(
        provider,
        chunk_ids,
        [
            QUERY_VECTOR,
            [0.999, 0.04, 0.0],
            [0.5, 0.8660254, 0.0],
            [0.91, 0.414613, 0.0],  # similarity ~= 0.91, just above the shared floor
        ],
    )

    by_embedding = provider.search_by_embedding(
        QUERY_VECTOR, PROVIDER, MODEL, limit=4, threshold=0.9
    )
    similar = provider.find_similar_chunks(
        chunk_ids[0], PROVIDER, MODEL, limit=4, threshold=0.9
    )
    results, _ = provider.search_semantic(
        QUERY_VECTOR, PROVIDER, MODEL, page_size=4, threshold=0.9
    )

    assert [result["chunk_id"] for result in by_embedding] == [
        chunk_ids[0],
        chunk_ids[1],
        chunk_ids[3],
    ]
    assert [result["chunk_id"] for result in similar] == [
        chunk_ids[1],
        chunk_ids[3],
    ]
    assert [result["chunk_id"] for result in results] == [
        chunk_ids[0],
        chunk_ids[1],
        chunk_ids[3],
    ]
    assert all(result["score"] >= 0.9 for result in by_embedding)
    assert all(result["score"] >= 0.9 for result in similar)
    assert all(result["similarity"] >= 0.9 for result in results)


@pytest.mark.hnsw
@pytest.mark.fast
def test_vector_results_break_similarity_ties_by_chunk_id(
    provider: DuckDBProvider,
) -> None:
    """Equal scores have stable ascending chunk_id order after HNSW retrieval."""
    chunk_ids = insert_chunks(provider, "src/ties.py", 5)
    insert_embeddings(provider, chunk_ids, [QUERY_VECTOR] * len(chunk_ids))

    semantic, _ = provider.search_semantic(
        QUERY_VECTOR, PROVIDER, MODEL, page_size=len(chunk_ids)
    )
    direct = provider.search_by_embedding(
        QUERY_VECTOR, PROVIDER, MODEL, limit=len(chunk_ids)
    )

    assert [result["chunk_id"] for result in semantic] == sorted(chunk_ids)
    assert [result["chunk_id"] for result in direct] == sorted(chunk_ids)


@pytest.mark.fast
def test_missing_embedding_warning_includes_source_context(
    provider: DuckDBProvider,
) -> None:
    """Missing-source diagnostics identify the requested provider and model."""
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        assert provider.find_similar_chunks(1, "source-provider", "source-model") == []
    finally:
        logger.remove(sink_id)

    assert any(
        "chunk_id=1" in message
        and "source-provider" in message
        and "source-model" in message
        for message in messages
    )


@dataclass
class _Recorder:
    messages: list[str] = field(default_factory=list)

    def section_header(self, message: str) -> None:
        self.messages.append(message)

    def info(self, message: str) -> None:
        self.messages.append(message)


def _seed_embeddings_nd(
    provider: DuckDBProvider,
    count: int,
    dims: int,
    path: str,
    *,
    provider_name: str = PROVIDER,
    model: str = MODEL,
    vector_fn: Callable[[int], list[float]] | None = None,
) -> list[int]:
    """Seed chunks+embeddings for a specific dimensionality."""
    chunk_ids = insert_chunks(provider, path, count)
    vectors = [
        vector_fn(i) if vector_fn else [random.gauss(0, 1) for _ in range(dims)]
        for i in range(count)
    ]
    insert_embeddings(
        provider, chunk_ids, vectors, provider_name=provider_name, model=model
    )
    return chunk_ids


# ---------------------------------------------------------------------------
# Recall benchmark
# ---------------------------------------------------------------------------


@pytest.mark.hnsw
@pytest.mark.slow
def test_hnsw_recall_vs_brute_force(provider: DuckDBProvider) -> None:
    """Deterministic multi-query HNSW recall stays close to brute force.

    Ground truth is computed with numpy over a full table read, so it can
    never be accelerated by a future DuckDB filter-pushdown change (an SQL
    ground-truth query would become self-referential once the planner can
    push the provider/model predicates into an HNSW scan).
    """
    n = 10_000
    dims = 8

    def make_vector(index: int) -> list[float]:
        values = [
            math.sin((index + 1) * (axis + 1) * 0.173)
            + math.cos((index + 3) * (axis + 2) * 0.071)
            for axis in range(dims)
        ]
        norm = math.sqrt(sum(value * value for value in values))
        return [value / norm for value in values]

    def large_primes(count: int) -> list[int]:
        primes: list[int] = []
        candidate = 100_003
        while len(primes) < count:
            if all(candidate % divisor for divisor in primes):
                primes.append(candidate)
            candidate += 2
        return primes

    _seed_embeddings_nd(provider, n, dims, "recall/module.py", vector_fn=make_vector)
    require_hnsw_index(provider, dims)

    rows = provider.execute_query(
        f"SELECT chunk_id, embedding FROM embeddings_{dims}", []
    )
    chunk_ids = [int(row["chunk_id"]) for row in rows]
    matrix = np.array([row["embedding"] for row in rows], dtype=np.float32)
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    hits = 0
    expected = 0
    for query_index in large_primes(32):
        query = make_vector(query_index)
        query_vec = np.array(query, dtype=np.float32)
        scores = matrix @ query_vec
        order = sorted(range(len(chunk_ids)), key=lambda i: (-scores[i], chunk_ids[i]))[
            :10
        ]
        brute_force_ids = [chunk_ids[i] for i in order]
        hnsw_results, _ = provider.search_semantic(query, PROVIDER, MODEL, page_size=10)
        hnsw_ids = {result["chunk_id"] for result in hnsw_results}
        hits += len(hnsw_ids & set(brute_force_ids))
        expected += len(brute_force_ids)

    recall = hits / expected
    assert recall >= 0.9, f"Aggregate Recall@10 = {recall:.2f}, expected >= 0.90"

    # Recall@100: fetch 100 results and compare against brute-force top-100
    hits_100 = 0
    expected_100 = 0
    for query_index in large_primes(32)[:8]:  # subset for speed
        query = make_vector(query_index)
        query_vec = np.array(query, dtype=np.float32)
        scores = matrix @ query_vec
        order_100 = sorted(
            range(len(chunk_ids)), key=lambda i: (-scores[i], chunk_ids[i])
        )[:100]
        brute_force_ids_100 = [chunk_ids[i] for i in order_100]
        hnsw_results_100, _ = provider.search_semantic(
            query, PROVIDER, MODEL, page_size=100
        )
        hnsw_ids_100 = {result["chunk_id"] for result in hnsw_results_100}
        hits_100 += len(hnsw_ids_100 & set(brute_force_ids_100))
        expected_100 += len(brute_force_ids_100)

    recall_100 = hits_100 / expected_100 if expected_100 > 0 else 0.0
    assert recall_100 >= 0.95, (
        f"Aggregate Recall@100 = {recall_100:.2f}, expected >= 0.95"
    )


# ---------------------------------------------------------------------------
# Multi-dimension HNSW
# ---------------------------------------------------------------------------


@pytest.mark.hnsw
@pytest.mark.fast
def test_multi_dimension_hnsw_targets_correct_table(
    provider: DuckDBProvider,
) -> None:
    """HNSW search targets the correct table based on query dimensions."""
    random.seed(123)

    def make_3d(i: int) -> list[float]:
        return [1.0, 0.01 * i, 0.0]

    def make_8d(i: int) -> list[float]:
        v = [0.0] * 8
        v[i % 8] = 1.0
        return v

    ids_3d = _seed_embeddings_nd(provider, 50, 3, "multi/3d.py", vector_fn=make_3d)
    ids_8d = _seed_embeddings_nd(provider, 50, 8, "multi/8d.py", vector_fn=make_8d)

    query_3d = [1.0, 0.0, 0.0]
    results_3d, _ = provider.search_semantic(query_3d, PROVIDER, MODEL, page_size=10)
    assert len(results_3d) > 0
    assert {r["chunk_id"] for r in results_3d} <= set(ids_3d)

    query_8d = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    results_8d, _ = provider.search_semantic(query_8d, PROVIDER, MODEL, page_size=10)
    assert len(results_8d) > 0
    assert {r["chunk_id"] for r in results_8d} <= set(ids_8d)

    indexes_3d = provider.execute_query(
        "SELECT sql FROM duckdb_indexes() WHERE table_name = 'embeddings_3'", []
    )
    indexes_8d = provider.execute_query(
        "SELECT sql FROM duckdb_indexes() WHERE table_name = 'embeddings_8'", []
    )
    assert any("USING HNSW" in (idx["sql"] or "").upper() for idx in indexes_3d)
    assert any("USING HNSW" in (idx["sql"] or "").upper() for idx in indexes_8d)


@pytest.mark.fast
def test_cli_formats_unknown_semantic_total() -> None:
    """The CLI must not render an unknown HNSW total as a numeric total."""
    formatter = _Recorder()

    _format_search_results(
        formatter,
        {
            "results": [{"file_path": "src/module.py", "content": ""}],
            "pagination": {"offset": 0, "page_size": 10, "total": None},
        },
        "query",
        is_regex=False,
    )

    assert "Results: 1 (showing 1-1)" in formatter.messages


@pytest.mark.fast
@pytest.mark.parametrize("threshold", [1.5, -1.5, float("nan"), float("inf")])
def test_semantic_search_rejects_invalid_threshold(
    provider: DuckDBProvider, threshold: float
) -> None:
    """Semantic search rejects thresholds outside [-1, 1] or non-finite."""
    with pytest.raises(
        ValueError,
        match="Threshold.*outside the valid cosine similarity range|Threshold must be a finite number",
    ):
        provider.search_semantic(
            QUERY_VECTOR, PROVIDER, MODEL, page_size=1, threshold=threshold
        )


@pytest.mark.fast
@pytest.mark.parametrize("threshold", [1.5, -1.5, float("nan"), float("inf")])
def test_find_similar_chunks_rejects_invalid_threshold(
    provider: DuckDBProvider, threshold: float
) -> None:
    """find_similar_chunks rejects thresholds outside [-1, 1] or non-finite."""
    with pytest.raises(
        ValueError,
        match="Threshold.*outside the valid cosine similarity range|Threshold must be a finite number",
    ):
        provider.find_similar_chunks(1, PROVIDER, MODEL, limit=1, threshold=threshold)


@pytest.mark.fast
@pytest.mark.parametrize("threshold", [1.5, -1.5, float("nan"), float("inf")])
def test_search_by_embedding_rejects_invalid_threshold(
    provider: DuckDBProvider, threshold: float
) -> None:
    """search_by_embedding rejects thresholds outside [-1, 1] or non-finite."""
    with pytest.raises(
        ValueError,
        match="Threshold.*outside the valid cosine similarity range|Threshold must be a finite number",
    ):
        provider.search_by_embedding(
            QUERY_VECTOR, PROVIDER, MODEL, limit=1, threshold=threshold
        )


@pytest.mark.fast
@pytest.mark.parametrize("limit", [0, -1])
def test_find_similar_chunks_rejects_nonpositive_limit(
    provider: DuckDBProvider, limit: int
) -> None:
    """find_similar_chunks rejects non-positive limits."""
    chunk_ids = seed_searchable_chunks(provider, count=1)
    with pytest.raises(ValueError, match="limit must be greater than 0"):
        provider.find_similar_chunks(chunk_ids[0], PROVIDER, MODEL, limit=limit)


@pytest.mark.fast
@pytest.mark.parametrize("limit", [0, -1])
def test_search_by_embedding_rejects_nonpositive_limit(
    provider: DuckDBProvider, limit: int
) -> None:
    """search_by_embedding rejects non-positive limits."""
    with pytest.raises(ValueError, match="limit must be greater than 0"):
        provider.search_by_embedding(QUERY_VECTOR, PROVIDER, MODEL, limit=limit)


@pytest.mark.fast
def test_normalize_semantic_window_cap_rejects_negative() -> None:
    """normalize_semantic_window_cap returns None for negative caps."""
    from chunkhound.services.search.semantic_window import normalize_semantic_window_cap

    assert normalize_semantic_window_cap(-5) is None
    assert normalize_semantic_window_cap(-1) is None
    assert normalize_semantic_window_cap(0) is None
    assert normalize_semantic_window_cap(1000) == 1000
