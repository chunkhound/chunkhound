"""Real-DuckDB contracts for HNSW-backed semantic search.

``slow`` tests seed 10k+ rows: candidate-budget exhaustion and Recall@10.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Generator, Sequence
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pytest
from loguru import logger

from chunkhound.api.cli.commands.search import _format_search_results
from chunkhound.core.config.database_config import DatabaseConfig
from chunkhound.core.constants import HNSW_CANDIDATE_BUDGET
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from tests.helpers.duckdb_vector_fixtures import (
    MODEL,
    PROVIDER,
    QUERY_VECTOR,
    connected_provider,
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


_BULK_INSERT_ROWS = 500


def _bulk_seed_embeddings(
    provider: DuckDBProvider,
    chunk_ids: Sequence[int],
    vectors: Sequence[Sequence[float]],
    *,
    provider_name: str = PROVIDER,
    model: str = MODEL,
    create_table: bool = True,
) -> None:
    """Fixture setup: create the table/index, then insert set-based batches.

    The first row goes through the public write API so the table and its
    canonical HNSW index are created exactly as production creates them;
    only the remaining rows are set-based. Per-row ``insert_embeddings``
    turns a 10k seed into a quadratic transaction that is pathologically slow
    on Windows (see HNSW test portability notes). Fixed-size multi-row VALUES
    statements are fixture cost, not behaviour.
    """
    if not chunk_ids:
        return
    start = 0
    if create_table:
        provider.insert_embeddings_batch(
            [
                {
                    "chunk_id": chunk_ids[0],
                    "provider": provider_name,
                    "model": model,
                    "embedding": list(vectors[0]),
                    "dims": 3,
                }
            ]
        )
        start = 1
    for start in range(start, len(chunk_ids), _BULK_INSERT_ROWS):
        rows = list(
            zip(
                chunk_ids[start : start + _BULK_INSERT_ROWS],
                vectors[start : start + _BULK_INSERT_ROWS],
                strict=True,
            )
        )
        values = ", ".join("(?, ?, ?, ?::FLOAT[3], 3)" for _ in rows)
        params: list[Any] = []
        for chunk_id, vector in rows:
            params.extend([chunk_id, provider_name, model, list(vector)])
        provider.execute_query(
            "INSERT INTO embeddings_3 (chunk_id, provider, model, embedding, dims) "
            f"VALUES {values}",
            params,
        )


@pytest.mark.hnsw
@pytest.mark.slow
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
    _bulk_seed_embeddings(
        provider, noise_ids, noise_vectors, provider_name="noise", model="m"
    )
    _bulk_seed_embeddings(
        provider,
        target_ids,
        [[0.1, 0.2, 0.3]] * 3,
        provider_name="target",
        model="m",
        create_table=False,
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
@pytest.mark.slow
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
    _bulk_seed_embeddings(
        provider, noise_ids, noise_vectors, provider_name="noise", model="m"
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
    _bulk_seed_embeddings(
        provider,
        target_ids,
        [[0.9, 0.3, 0.1]],
        provider_name="missing",
        model="x",
        create_table=False,
    )
    with pytest.raises(RuntimeError, match="exhausted the HNSW candidate budget"):
        provider.find_similar_chunks(target_ids[0], "missing", "x", limit=10)


@pytest.mark.hnsw
@pytest.mark.fast
def test_semantic_pagination_counts_the_returnable_rows(
    provider: DuckDBProvider,
) -> None:
    """Semantic pagination counts every row the pages can return."""
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
        "total": 3,
        "candidate_budget_exhausted": False,
    }
    assert len(second_page) == 1
    assert second_pagination == {
        "offset": 2,
        "page_size": 2,
        "has_more": False,
        "next_offset": None,
        "total": 3,
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
        "total": 0,
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


Search = Callable[[DuckDBProvider, list[int]], list[dict[str, Any]]]


def _search_semantic(provider: DuckDBProvider, _: list[int]) -> list[dict[str, Any]]:
    results, _ = provider.search_semantic(QUERY_VECTOR, PROVIDER, MODEL, page_size=1)
    return results


def _search_by_embedding(
    provider: DuckDBProvider, _: list[int]
) -> list[dict[str, Any]]:
    return provider.search_by_embedding(QUERY_VECTOR, PROVIDER, MODEL, limit=1)


def _find_similar(
    provider: DuckDBProvider, chunk_ids: list[int]
) -> list[dict[str, Any]]:
    return provider.find_similar_chunks(chunk_ids[0], PROVIDER, MODEL, limit=1)


def _hnsw_index_metrics(provider: DuckDBProvider) -> set[tuple[str, str]]:
    """Return (index, lower-cased metric) pairs for the 3-D table only.

    Scoped by table because the pragma sees indexes from every embedding table.
    """
    rows = provider.execute_query(
        "SELECT index_name, metric FROM pragma_hnsw_index_info() "
        "WHERE table_name = 'embeddings_3'",
        [],
    )
    return {(str(row["index_name"]), str(row["metric"]).lower()) for row in rows}


def _create_canonical_l2sq_collision(provider: DuckDBProvider) -> None:
    """Occupy the canonical HNSW name ``idx_hnsw_3`` with the l2sq metric.

    Only legacy persisted databases reach this state; no production path builds
    a non-cosine canonical index, so the fixture uses raw DDL.
    """
    provider.drop_all_hnsw_indexes()
    provider.execute_query(
        "CREATE INDEX idx_hnsw_3 ON embeddings_3 USING HNSW (embedding) "
        "WITH (metric = 'l2sq')",
        [],
    )


@pytest.mark.hnsw
@pytest.mark.fast
@pytest.mark.parametrize(
    ("search", "expected_index"),
    [(_search_semantic, 0), (_search_by_embedding, 0), (_find_similar, 1)],
)
def test_vector_search_recovers_missing_hnsw_index(
    provider: DuckDBProvider, search: Search, expected_index: int
) -> None:
    """Every public vector API still returns its best match without an index."""
    chunk_ids = seed_searchable_chunks(provider)
    provider.drop_all_hnsw_indexes()

    results = search(provider, chunk_ids)

    assert [result["chunk_id"] for result in results] == [chunk_ids[expected_index]]
    require_hnsw_index(provider)


@pytest.mark.hnsw
@pytest.mark.fast
def test_non_cosine_hnsw_is_preserved_and_cosine_index_is_added(
    provider: DuckDBProvider,
) -> None:
    """Cosine search works when the table already carries another HNSW metric."""
    chunk_ids = seed_searchable_chunks(provider)
    _create_canonical_l2sq_collision(provider)

    results, _ = provider.search_semantic(QUERY_VECTOR, PROVIDER, MODEL, page_size=3)

    assert {result["chunk_id"] for result in results} == set(chunk_ids)
    assert _hnsw_index_metrics(provider) == {
        ("idx_hnsw_3", "l2sq"),
        ("idx_hnsw_3_cosine", "cosine"),
    }


@pytest.mark.hnsw
@pytest.mark.fast
def test_drop_vector_index_also_drops_cosine_variant(
    provider: DuckDBProvider,
) -> None:
    """Dropping a custom index must also drop the lazily-created cosine one."""
    seed_searchable_chunks(provider)
    _create_canonical_l2sq_collision(provider)
    provider.create_vector_index(PROVIDER, MODEL, 3, "l2sq")

    provider.search_semantic(QUERY_VECTOR, PROVIDER, MODEL, page_size=1)
    assert ("idx_hnsw_3_cosine", "cosine") in _hnsw_index_metrics(provider)

    provider.drop_vector_index(PROVIDER, MODEL, 3, "l2sq")

    remaining = _hnsw_index_metrics(provider)
    assert remaining == set(), f"indexes leaked after drop_vector_index: {remaining}"


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
        "total": 0,
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
        "total": 1001,
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


# ---------------------------------------------------------------------------
# Multi-dimension HNSW
# ---------------------------------------------------------------------------


@pytest.mark.hnsw
@pytest.mark.fast
def test_multi_dimension_hnsw_targets_correct_table(
    provider: DuckDBProvider,
) -> None:
    """Each query dimension searches only its own embedding table."""
    ids_3d = insert_chunks(provider, "multi/3d.py", 5)
    insert_embeddings(
        provider, ids_3d, [[1.0, 0.01 * index, 0.0] for index in range(5)]
    )
    ids_8d = insert_chunks(provider, "multi/8d.py", 5)
    insert_embeddings(
        provider,
        ids_8d,
        [[1.0] + [0.01 * index] * 7 for index in range(5)],
    )

    results_3d, _ = provider.search_semantic(
        [1.0, 0.0, 0.0], PROVIDER, MODEL, page_size=5
    )
    results_8d, _ = provider.search_semantic(
        [1.0] + [0.0] * 7, PROVIDER, MODEL, page_size=5
    )

    assert {result["chunk_id"] for result in results_3d} == set(ids_3d)
    assert {result["chunk_id"] for result in results_8d} == set(ids_8d)


_THRESHOLD_ERROR = (
    "Threshold.*outside the valid cosine similarity range"
    "|Threshold must be a finite number"
)


@pytest.mark.fast
@pytest.mark.parametrize("threshold", [1.5, -1.5, float("nan"), float("inf")])
def test_semantic_search_rejects_invalid_threshold(
    provider: DuckDBProvider, threshold: float
) -> None:
    """Semantic search rejects thresholds outside [-1, 1] or non-finite."""
    with pytest.raises(ValueError, match=_THRESHOLD_ERROR):
        provider.search_semantic(
            QUERY_VECTOR, PROVIDER, MODEL, page_size=1, threshold=threshold
        )


@pytest.mark.fast
@pytest.mark.parametrize("threshold", [1.5, -1.5, float("nan"), float("inf")])
def test_find_similar_chunks_rejects_invalid_threshold(
    provider: DuckDBProvider, threshold: float
) -> None:
    """find_similar_chunks rejects thresholds outside [-1, 1] or non-finite."""
    with pytest.raises(ValueError, match=_THRESHOLD_ERROR):
        provider.find_similar_chunks(1, PROVIDER, MODEL, limit=1, threshold=threshold)


@pytest.mark.fast
@pytest.mark.parametrize("threshold", [1.5, -1.5, float("nan"), float("inf")])
def test_search_by_embedding_rejects_invalid_threshold(
    provider: DuckDBProvider, threshold: float
) -> None:
    """search_by_embedding rejects thresholds outside [-1, 1] or non-finite."""
    with pytest.raises(ValueError, match=_THRESHOLD_ERROR):
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


class _Recorder(list[str]):
    section_header = info = list.append


def test_cli_formats_unknown_semantic_total() -> None:
    """An unknown total (candidate budget exhausted) still renders a usable header."""
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

    assert "Results: 1 of 1 (showing 1-1)" in formatter


@pytest.mark.fast
def test_normalize_semantic_window_cap_rejects_negative() -> None:
    """Normalization maps non-positive caps to None and keeps positive caps."""
    from chunkhound.services.search.semantic_window import normalize_semantic_window_cap

    assert normalize_semantic_window_cap(-5) is None
    assert normalize_semantic_window_cap(-1) is None
    assert normalize_semantic_window_cap(0) is None
    assert normalize_semantic_window_cap(1000) == 1000


@pytest.mark.hnsw
@pytest.mark.slow
def test_hnsw_recall_vs_brute_force(provider: DuckDBProvider) -> None:
    """HNSW Recall@10 stays close to a numpy full-table ground truth.

    The oracle reads and scores the table in numpy, so a future filter
    pushdown into the HNSW scan cannot make the oracle self-referential.
    """
    n, dims, queries = 10_000, 8, 32
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(n + queries, dims))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    insert_embeddings(
        provider, insert_chunks(provider, "recall/module.py", n), vectors[:n].tolist()
    )
    require_hnsw_index(provider, dims)

    rows = provider.execute_query(
        f"SELECT chunk_id, embedding FROM embeddings_{dims}", []
    )
    ids = [int(row["chunk_id"]) for row in rows]
    matrix = np.array([row["embedding"] for row in rows], dtype=np.float32)

    hits = 0
    for vector in vectors[n:].astype(np.float32):
        found, _ = provider.search_semantic(
            vector.tolist(), PROVIDER, MODEL, page_size=10
        )
        truth = {ids[i] for i in np.argsort(-(matrix @ vector))[:10]}
        hits += len({result["chunk_id"] for result in found} & truth)

    recall = hits / (10 * queries)
    assert recall >= 0.9, f"Aggregate Recall@10 = {recall:.2f}, expected >= 0.90"

    # Recall@100: page_size=100 on a speed-bounded subset of queries
    hits_100 = 0
    for vector in vectors[n : n + 8].astype(np.float32):
        found_100, _ = provider.search_semantic(
            vector.tolist(), PROVIDER, MODEL, page_size=100
        )
        truth_100 = {ids[i] for i in np.argsort(-(matrix @ vector))[:100]}
        hits_100 += len({result["chunk_id"] for result in found_100} & truth_100)

    recall_100 = hits_100 / (100 * 8)
    assert recall_100 >= 0.95, (
        f"Aggregate Recall@100 = {recall_100:.2f}, expected >= 0.95"
    )
