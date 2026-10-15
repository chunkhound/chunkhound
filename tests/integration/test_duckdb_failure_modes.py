"""Failure-mode contracts for DuckDB semantic search.

Real DuckDB throughout; the only injected failure is a blocking executor
operation used to prove the configured timeout is enforced.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Generator
from pathlib import Path

import pytest

from chunkhound.core.config.database_config import DatabaseConfig
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from tests.helpers.duckdb_vector_fixtures import (
    MODEL,
    PROVIDER,
    QUERY_VECTOR,
    connected_provider,
    seed_searchable_chunks,
)

# Long enough for connect + schema creation on a loaded CI machine, short
# enough to keep the timeout test fast.
_TIMEOUT_SECONDS = 2.0


@pytest.fixture
def provider(tmp_path: Path) -> Generator[DuckDBProvider, None, None]:
    with connected_provider(tmp_path) as database:
        yield database


@pytest.fixture
def timeout_provider(tmp_path: Path) -> Generator[DuckDBProvider, None, None]:
    config = DatabaseConfig(execute_timeout_seconds=_TIMEOUT_SECONDS)
    with connected_provider(tmp_path, config=config) as database:
        yield database


@pytest.fixture
def stalled_search(
    timeout_provider: DuckDBProvider, monkeypatch: pytest.MonkeyPatch
) -> Generator[threading.Event, None, None]:
    """Block the semantic executor until the returned event is set."""
    release = threading.Event()

    def blocking_search(*_args: object, **_kwargs: object) -> object:
        release.wait(timeout=30)
        return [], {}

    monkeypatch.setattr(timeout_provider, "_executor_search_semantic", blocking_search)
    try:
        yield release
    finally:
        release.set()


@pytest.mark.integration
def test_stalled_search_times_out_and_provider_recovers(
    timeout_provider: DuckDBProvider,
    stalled_search: threading.Event,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stalled DB op raises TimeoutError naming the op, then search resumes."""
    with pytest.raises(TimeoutError, match="search_semantic"):
        timeout_provider.search_semantic(QUERY_VECTOR, PROVIDER, MODEL, page_size=1)

    stalled_search.set()
    monkeypatch.undo()
    results, pagination = timeout_provider.search_semantic(
        QUERY_VECTOR, PROVIDER, MODEL, page_size=1
    )

    assert results == []
    assert pagination["offset"] == 0


@pytest.mark.integration
@pytest.mark.hnsw
@pytest.mark.parametrize(
    "bad_vector",
    [
        [float("nan"), 0.0, 0.0],
        [float("inf"), 0.0, 0.0],
        [1.0, 0.0, float("-inf")],
    ],
)
def test_non_finite_query_embedding_is_rejected(
    provider: DuckDBProvider, bad_vector: list[float]
) -> None:
    """NaN/Inf vectors fail with ValueError, not a DuckDB internal error."""
    seed_searchable_chunks(provider)

    with pytest.raises(ValueError, match="non-finite"):
        provider.search_semantic(bad_vector, PROVIDER, MODEL, page_size=5)
    with pytest.raises(ValueError, match="non-finite"):
        provider.search_by_embedding(bad_vector, PROVIDER, MODEL, limit=5)


@pytest.mark.integration
@pytest.mark.hnsw
def test_unknown_dimension_query_returns_empty_page(provider: DuckDBProvider) -> None:
    """Querying a dimension that was never indexed yields an empty page.

    Multi-model repositories store one table per dimension; a query for a
    dimension with no table has no data rather than an error.
    """
    seed_searchable_chunks(provider)

    results, pagination = provider.search_semantic(
        [1.0] * 5, PROVIDER, MODEL, page_size=5
    )

    assert results == []
    assert pagination["has_more"] is False
    assert pagination["next_offset"] is None
    assert pagination["total"] is None
    assert pagination["candidate_budget_exhausted"] is False


@pytest.mark.integration
@pytest.mark.hnsw
def test_empty_query_embedding_returns_empty_page(provider: DuckDBProvider) -> None:
    """A zero-length query vector degrades to an empty page, never a crash."""
    seed_searchable_chunks(provider)

    results, pagination = provider.search_semantic([], PROVIDER, MODEL, page_size=5)
    by_embedding = provider.search_by_embedding([], PROVIDER, MODEL, limit=5)

    assert results == []
    assert by_embedding == []
    assert pagination["has_more"] is False
    assert pagination["candidate_budget_exhausted"] is False


@pytest.mark.integration
@pytest.mark.hnsw
def test_search_after_disconnect_fails_explicitly(tmp_path: Path) -> None:
    """Searching a closed provider raises instead of returning empty results."""
    with connected_provider(tmp_path) as database:
        seed_searchable_chunks(database)
    with pytest.raises(RuntimeError, match="after shutdown"):
        database.search_semantic(QUERY_VECTOR, PROVIDER, MODEL, page_size=1)


@pytest.mark.integration
@pytest.mark.hnsw
def test_valid_search_still_works_alongside_failure_modes(
    provider: DuckDBProvider,
) -> None:
    """Guard rails must not break the happy path."""
    chunk_ids = seed_searchable_chunks(provider, count=3)

    results, _ = provider.search_semantic(QUERY_VECTOR, PROVIDER, MODEL, page_size=3)

    assert {result["chunk_id"] for result in results} == set(chunk_ids)
    assert all(math.isfinite(result["similarity"]) for result in results)
