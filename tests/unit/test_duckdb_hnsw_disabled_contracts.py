"""Real-DuckDB contracts for exact vector search (``duckdb_hnsw_enabled=false``).

Disabled mode must rank exactly, never create or use an HNSW index, and stay
uncapped -- the approximate-search cap and candidate budget only exist because
HNSW is approximate.
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from pathlib import Path

import pytest

from chunkhound.core.config.database_config import DatabaseConfig
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from tests.helpers.duckdb_vector_fixtures import (
    MODEL,
    PROVIDER,
    QUERY_VECTOR,
    connected_provider,
    has_hnsw_index,
    insert_chunks,
    insert_embeddings,
    profile_last_query,
)

_EXACT_CONFIG = DatabaseConfig(duckdb_hnsw_enabled=False)


@pytest.fixture
def exact_provider(tmp_path: Path) -> Generator[DuckDBProvider, None, None]:
    """Provide a file-backed provider with HNSW turned off by configuration."""
    with connected_provider(tmp_path, _EXACT_CONFIG) as database:
        yield database


def _seed(
    provider: DuckDBProvider,
    count: int,
    path: str = "src/module.py",
    *,
    provider_name: str = PROVIDER,
    model: str = MODEL,
) -> list[int]:
    """Seed chunks whose vectors get monotonically further from QUERY_VECTOR."""
    chunk_ids = insert_chunks(provider, path, count)
    insert_embeddings(
        provider,
        chunk_ids,
        [[1.0, 0.01 * index, 0.0] for index in range(count)],
        provider_name=provider_name,
        model=model,
    )
    return chunk_ids


def _ids(results: Sequence[dict[str, object]]) -> list[int]:
    return [int(result["chunk_id"]) for result in results]


@pytest.mark.hnsw
@pytest.mark.fast
def test_large_batch_insert_drops_and_never_resurrects_hnsw_index_when_disabled(
    tmp_path: Path,
) -> None:
    """Bulk insert (>=50) over a pre-existing HNSW index must drop it and leave
    it dropped when HNSW is disabled.

    The bulk path drops indexes for insert speed and recreates them afterwards;
    with the configuration turned off that recreate must not happen, or the
    index would be resurrected behind the user's back.
    """
    with connected_provider(tmp_path) as enabled:
        chunk_ids = _seed(enabled, 60)
        enabled.ensure_all_hnsw_indexes()
        if not has_hnsw_index(enabled, 3):
            pytest.skip("DuckDB HNSW indexes are unavailable in this environment")

    with connected_provider(tmp_path, _EXACT_CONFIG) as disabled:
        assert has_hnsw_index(disabled, 3)

        # Exercise the repository's bulk path, which drops indexes for insert
        # speed and recreates them afterwards (the provider-level batch method
        # is a plain upsert and never touches indexes).
        disabled._embedding_repository.insert_embeddings_batch(
            [
                {
                    "chunk_id": chunk_id,
                    "provider": PROVIDER,
                    "model": MODEL,
                    "embedding": [1.0, 0.01 * index, 0.0],
                    "dims": 3,
                }
                for index, chunk_id in enumerate(chunk_ids)
            ],
            batch_size=50,
            connection=disabled.connection,
        )

        assert not has_hnsw_index(disabled, 3)


@pytest.mark.fast
def test_exact_search_serves_every_api_without_an_index(
    exact_provider: DuckDBProvider,
) -> None:
    """No HNSW index is created before, during, or after any vector search."""
    chunk_ids = _seed(exact_provider, 5)
    assert not has_hnsw_index(exact_provider, 3)

    semantic, _ = exact_provider.search_semantic(
        QUERY_VECTOR, PROVIDER, MODEL, page_size=3
    )
    by_embedding = exact_provider.search_by_embedding(
        QUERY_VECTOR, PROVIDER, MODEL, limit=2
    )
    similar = exact_provider.find_similar_chunks(chunk_ids[0], PROVIDER, MODEL, limit=2)

    assert _ids(semantic) == chunk_ids[:3]
    assert _ids(by_embedding) == chunk_ids[:2]
    assert _ids(similar) == chunk_ids[1:3]
    assert semantic[0]["similarity"] == pytest.approx(1.0)
    assert by_embedding[0]["score"] == pytest.approx(1.0)
    assert not has_hnsw_index(exact_provider, 3)


@pytest.mark.fast
def test_exact_semantic_pagination_is_uncapped_and_totalless(
    exact_provider: DuckDBProvider,
) -> None:
    """Pages partition results by look-ahead, with no total and no offset cap."""
    chunk_ids = _seed(exact_provider, 5)
    assert exact_provider.semantic_result_window_cap is None

    first, first_page = exact_provider.search_semantic(
        QUERY_VECTOR, PROVIDER, MODEL, page_size=3, offset=0
    )
    second, second_page = exact_provider.search_semantic(
        QUERY_VECTOR, PROVIDER, MODEL, page_size=3, offset=3
    )
    beyond, beyond_page = exact_provider.search_semantic(
        QUERY_VECTOR, PROVIDER, MODEL, page_size=5, offset=2000
    )

    assert _ids(first) + _ids(second) == chunk_ids
    assert first_page == {
        "offset": 0,
        "page_size": 3,
        "has_more": True,
        "next_offset": 3,
        "total": None,
        "candidate_budget_exhausted": False,
    }
    assert (second_page["has_more"], second_page["next_offset"]) == (False, None)
    assert second_page["total"] is None
    assert beyond == []
    assert (beyond_page["has_more"], beyond_page["next_offset"]) == (False, None)


@pytest.mark.fast
def test_exact_search_filters_before_ranking(exact_provider: DuckDBProvider) -> None:
    """Provider/model, path, threshold, and self-exclusion never starve results."""
    noise_ids = _seed(exact_provider, 30, "noise/module.py", provider_name="noise")
    target_ids = insert_chunks(exact_provider, "target/module.py", 3)
    insert_embeddings(
        exact_provider,
        target_ids,
        [QUERY_VECTOR, [0.91, 0.414613, 0.0], [0.5, 0.8660254, 0.0]],
    )

    semantic, _ = exact_provider.search_semantic(
        QUERY_VECTOR, PROVIDER, MODEL, page_size=10, path_filter="target"
    )
    thresholded = exact_provider.search_by_embedding(
        QUERY_VECTOR, PROVIDER, MODEL, limit=10, threshold=0.9
    )
    similar = exact_provider.find_similar_chunks(
        target_ids[0], PROVIDER, MODEL, limit=10
    )

    assert _ids(semantic) == target_ids
    assert _ids(thresholded) == target_ids[:2]
    assert all(result["score"] >= 0.9 for result in thresholded)
    assert target_ids[0] not in _ids(similar)
    assert not set(_ids(semantic)) & set(noise_ids)


@pytest.mark.hnsw
@pytest.mark.fast
def test_exact_search_ignores_a_persisted_hnsw_index(tmp_path: Path) -> None:
    """A leftover index from enabled mode must not re-enter the executed plan."""
    with connected_provider(tmp_path) as writer:
        chunk_ids = _seed(writer, 300)
        writer.ensure_all_hnsw_indexes()
        if not has_hnsw_index(writer, 3):
            pytest.skip("DuckDB HNSW indexes are unavailable in this environment")

    with connected_provider(tmp_path, _EXACT_CONFIG) as reader:
        results: list[dict[str, object]] = []
        profile = profile_last_query(
            reader,
            tmp_path / "profile.json",
            lambda: results.extend(
                reader.search_semantic(QUERY_VECTOR, PROVIDER, MODEL, page_size=3)[0]
            ),
        )

        assert has_hnsw_index(reader, 3)
        assert _ids(results) == chunk_ids[:3]
        # The ranking query is the only query the exact path runs; asserting it
        # was profiled keeps the plan assertion from passing vacuously.
        assert "ARRAY_COSINE_DISTANCE" in profile
        assert "HNSW_INDEX_SCAN" not in profile


@pytest.mark.hnsw
@pytest.mark.fast
def test_index_creation_resumes_only_after_reenabling(tmp_path: Path) -> None:
    """Disabled mode skips every creation site; re-enabling rebuilds on demand."""
    with connected_provider(tmp_path, _EXACT_CONFIG) as disabled:
        _seed(disabled, 5)
        disabled.ensure_all_hnsw_indexes()
        disabled.create_vector_index(PROVIDER, MODEL, 3)
        _seed(disabled, 3, "src/late.py")
        disabled.search_semantic(QUERY_VECTOR, PROVIDER, MODEL, page_size=2)
        assert not has_hnsw_index(disabled, 3)

    with connected_provider(tmp_path) as enabled:
        enabled.ensure_all_hnsw_indexes()
        if not has_hnsw_index(enabled, 3):
            pytest.skip("DuckDB HNSW indexes are unavailable in this environment")


@pytest.mark.hnsw
@pytest.mark.fast
def test_disabled_compaction_does_not_restore_hnsw(tmp_path: Path) -> None:
    """Compaction honors disabled mode even when the source persisted HNSW."""
    with connected_provider(tmp_path) as enabled:
        _seed(enabled, 5)
        if not has_hnsw_index(enabled, 3):
            pytest.skip("DuckDB HNSW indexes are unavailable in this environment")

    with connected_provider(tmp_path, _EXACT_CONFIG) as disabled:
        assert has_hnsw_index(disabled, 3)
        disabled.compact_database()
        assert not has_hnsw_index(disabled, 3)


@pytest.mark.fast
def test_exact_search_succeeds_on_a_read_only_database(tmp_path: Path) -> None:
    """Exact search needs no writes, so read-only databases stay searchable."""
    db_path = tmp_path / "exact-read-only.duckdb"
    writer = DuckDBProvider(db_path, base_directory=tmp_path, config=_EXACT_CONFIG)
    writer.connect()
    chunk_ids = _seed(writer, 3)
    writer.execute_query("CHECKPOINT", [])
    writer.disconnect(skip_checkpoint=True)

    reader = DuckDBProvider(
        db_path,
        base_directory=tmp_path,
        config=DatabaseConfig(read_only=True, duckdb_hnsw_enabled=False),
    )
    reader.connect()
    try:
        results, _ = reader.search_semantic(QUERY_VECTOR, PROVIDER, MODEL, page_size=3)
    finally:
        reader.disconnect(skip_checkpoint=True)

    assert _ids(results) == chunk_ids
