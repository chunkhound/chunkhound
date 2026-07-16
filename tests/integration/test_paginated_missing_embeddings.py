"""Contract tests for get_chunks_without_embeddings_paginated (Phase 1).

Verifies dual-backend paginated "chunks needing embeddings" API used by the
large-index embedding pipeline. Does not rewrite EmbeddingService yet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chunkhound.core.models import Chunk, File
from chunkhound.core.types.common import ChunkType, Language


REQUIRED_KEYS = {
    "id",
    "file_id",
    "code",
    "symbol",
    "file_path",
    "start_line",
    "end_line",
    "chunk_type",
    "language",
}


def _insert_file(provider: Any, path: str = "sample.py") -> int:
    file_id = provider.insert_file(
        File(
            path=path,
            mtime=1_700_000_000.0,
            language=Language.PYTHON,
            size_bytes=128,
        )
    )
    return int(file_id)


def _insert_chunks(provider: Any, file_id: int, count: int, prefix: str = "fn") -> list[int]:
    chunks = [
        Chunk(
            file_id=file_id,
            code=f"def {prefix}_{i}():\n    return {i}\n",
            start_line=i * 3 + 1,
            end_line=i * 3 + 2,
            chunk_type=ChunkType.FUNCTION,
            language=Language.PYTHON,
            symbol=f"{prefix}_{i}",
        )
        for i in range(count)
    ]
    ids = provider.insert_chunks_batch(chunks)
    assert len(ids) == count
    return [int(i) for i in ids]


def _embed(
    provider: Any,
    chunk_ids: list[int],
    *,
    provider_name: str = "test",
    model: str = "test-model",
    dims: int = 8,
) -> None:
    data = [
        {
            "chunk_id": cid,
            "provider": provider_name,
            "model": model,
            "dims": dims,
            # Non-zero vectors only — zero vectors are treated as invalid placeholders.
            "embedding": [0.1 + float(abs(cid) % 7) * 0.01] * dims,
        }
        for cid in chunk_ids
    ]
    stored = provider.insert_embeddings_batch(data)
    assert stored == len(chunk_ids)


def _walk_pages(
    provider: Any,
    *,
    provider_name: str,
    model: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Keyset walk: after_id advances until empty."""
    all_rows: list[dict[str, Any]] = []
    after_id: int | None = None
    for _ in range(1000):  # safety cap
        page = provider.get_chunks_without_embeddings_paginated(
            provider_name,
            model,
            limit=limit,
            after_id=after_id,
        )
        if not page:
            break
        all_rows.extend(page)
        after_id = int(page[-1]["id"])
    return all_rows


@pytest.fixture
def duckdb_provider(tmp_path: Path):
    """Connected DuckDB provider for pagination contract tests."""
    from chunkhound.providers.database.duckdb_provider import DuckDBProvider

    db_path = tmp_path / "chunks.db"
    provider = DuckDBProvider(db_path, base_directory=tmp_path)
    provider.connect()
    yield provider
    provider.disconnect()


@pytest.mark.parametrize("provider_fixture", ["duckdb_provider", "lancedb_provider"])
def test_paginated_empty_when_no_chunks(provider_fixture: str, request: pytest.FixtureRequest):
    provider = request.getfixturevalue(provider_fixture)
    page = provider.get_chunks_without_embeddings_paginated(
        "test", "test-model", limit=10
    )
    assert page == []


@pytest.mark.parametrize("provider_fixture", ["duckdb_provider", "lancedb_provider"])
def test_paginated_returns_all_when_no_embeddings(
    provider_fixture: str, request: pytest.FixtureRequest
):
    provider = request.getfixturevalue(provider_fixture)
    file_id = _insert_file(provider)
    chunk_ids = _insert_chunks(provider, file_id, 5)

    page = provider.get_chunks_without_embeddings_paginated(
        "test", "test-model", limit=100
    )
    assert len(page) == 5
    returned_ids = {int(row["id"]) for row in page}
    assert returned_ids == set(chunk_ids)

    for row in page:
        assert REQUIRED_KEYS.issubset(row.keys())
        assert row["code"]
        assert row["symbol"]
        assert row["file_path"]


@pytest.mark.parametrize("provider_fixture", ["duckdb_provider", "lancedb_provider"])
def test_paginated_excludes_embedded_for_provider_model(
    provider_fixture: str, request: pytest.FixtureRequest
):
    provider = request.getfixturevalue(provider_fixture)
    file_id = _insert_file(provider)
    chunk_ids = _insert_chunks(provider, file_id, 4)

    # Embed first two for target provider/model
    _embed(provider, chunk_ids[:2], provider_name="test", model="test-model")
    # Embed third for a different model — still "missing" for test-model
    _embed(provider, [chunk_ids[2]], provider_name="test", model="other-model")

    page = provider.get_chunks_without_embeddings_paginated(
        "test", "test-model", limit=100
    )
    returned = {int(row["id"]) for row in page}
    assert chunk_ids[0] not in returned
    assert chunk_ids[1] not in returned
    assert chunk_ids[2] in returned
    assert chunk_ids[3] in returned


@pytest.mark.parametrize("provider_fixture", ["duckdb_provider", "lancedb_provider"])
def test_paginated_keyset_walk_covers_all_without_duplicates(
    provider_fixture: str, request: pytest.FixtureRequest
):
    provider = request.getfixturevalue(provider_fixture)
    file_id = _insert_file(provider)
    chunk_ids = _insert_chunks(provider, file_id, 7)

    walked = _walk_pages(
        provider, provider_name="test", model="test-model", limit=3
    )
    walked_ids = [int(row["id"]) for row in walked]

    assert len(walked_ids) == 7
    assert set(walked_ids) == set(chunk_ids)
    assert walked_ids == sorted(walked_ids)
    # No duplicates across pages
    assert len(walked_ids) == len(set(walked_ids))


@pytest.mark.parametrize("provider_fixture", ["duckdb_provider", "lancedb_provider"])
def test_paginated_residual_shrink_after_insert(
    provider_fixture: str, request: pytest.FixtureRequest
):
    """Streaming mode: re-query without after_id after inserting embeddings."""
    provider = request.getfixturevalue(provider_fixture)
    file_id = _insert_file(provider)
    chunk_ids = _insert_chunks(provider, file_id, 6)

    first = provider.get_chunks_without_embeddings_paginated(
        "test", "test-model", limit=2
    )
    assert len(first) == 2
    _embed(provider, [int(row["id"]) for row in first])

    second = provider.get_chunks_without_embeddings_paginated(
        "test", "test-model", limit=100
    )
    second_ids = {int(row["id"]) for row in second}
    first_ids = {int(row["id"]) for row in first}
    assert first_ids.isdisjoint(second_ids)
    assert len(second_ids) == 4
    assert second_ids | first_ids == set(chunk_ids)


@pytest.mark.parametrize("provider_fixture", ["duckdb_provider", "lancedb_provider"])
def test_paginated_residual_drain_until_empty(
    provider_fixture: str, request: pytest.FixtureRequest
):
    """Large-index streaming path: page with limit until residual is empty."""
    provider = request.getfixturevalue(provider_fixture)
    file_id = _insert_file(provider)
    chunk_ids = _insert_chunks(provider, file_id, 7)

    drained: list[int] = []
    for _ in range(20):
        page = provider.get_chunks_without_embeddings_paginated(
            "test", "test-model", limit=2
        )
        if not page:
            break
        page_ids = [int(row["id"]) for row in page]
        assert set(page_ids).isdisjoint(drained)
        _embed(provider, page_ids)
        drained.extend(page_ids)

    assert set(drained) == set(chunk_ids)
    assert (
        provider.get_chunks_without_embeddings_paginated(
            "test", "test-model", limit=10
        )
        == []
    )


@pytest.mark.parametrize("provider_fixture", ["duckdb_provider", "lancedb_provider"])
def test_paginated_limit_zero_returns_empty(
    provider_fixture: str, request: pytest.FixtureRequest
):
    provider = request.getfixturevalue(provider_fixture)
    file_id = _insert_file(provider)
    _insert_chunks(provider, file_id, 2)
    assert (
        provider.get_chunks_without_embeddings_paginated(
            "test", "test-model", limit=0
        )
        == []
    )


def test_paginated_duckdb_multiple_embedding_dim_tables(duckdb_provider):
    """Missing filter must consider every embeddings_<dims> table (AND NOT EXISTS)."""
    provider = duckdb_provider
    file_id = _insert_file(provider)
    chunk_ids = _insert_chunks(provider, file_id, 3)

    # Same provider/model can live in different dim tables depending on vector length.
    _embed(provider, [chunk_ids[0]], dims=8)
    _embed(provider, [chunk_ids[1]], dims=16)

    page = provider.get_chunks_without_embeddings_paginated(
        "test", "test-model", limit=100
    )
    returned = {int(row["id"]) for row in page}
    assert chunk_ids[0] not in returned
    assert chunk_ids[1] not in returned
    assert chunk_ids[2] in returned


def test_paginated_lancedb_invalid_zero_embedding_still_missing(lancedb_provider):
    """LanceDB treats zero-vector placeholders as not a valid embedding."""
    provider = lancedb_provider
    file_id = _insert_file(provider)
    chunk_ids = _insert_chunks(provider, file_id, 2)

    # Valid embedding on first; zero-vector "embedding" on second (legacy placeholder).
    _embed(provider, [chunk_ids[0]], dims=8)
    stored = provider.insert_embeddings_batch(
        [
            {
                "chunk_id": chunk_ids[1],
                "provider": "test",
                "model": "test-model",
                "dims": 8,
                "embedding": [0.0] * 8,
            }
        ]
    )
    assert stored == 1

    page = provider.get_chunks_without_embeddings_paginated(
        "test", "test-model", limit=100
    )
    returned = {int(row["id"]) for row in page}
    assert chunk_ids[0] not in returned
    assert chunk_ids[1] in returned


def test_paginated_lancedb_keyset_includes_zero_vector_labeled_chunks(
    lancedb_provider,
):
    """Keyset walk must surface invalid labeled embeddings (not only unlabeled)."""
    provider = lancedb_provider
    file_id = _insert_file(provider)
    chunk_ids = _insert_chunks(provider, file_id, 4)

    # Two valid, one zero-vector labeled, one completely unembedded.
    _embed(provider, chunk_ids[:2], dims=8)
    stored = provider.insert_embeddings_batch(
        [
            {
                "chunk_id": chunk_ids[2],
                "provider": "test",
                "model": "test-model",
                "dims": 8,
                "embedding": [0.0] * 8,
            }
        ]
    )
    assert stored == 1

    walked = _walk_pages(
        provider, provider_name="test", model="test-model", limit=2
    )
    walked_ids = {int(row["id"]) for row in walked}
    assert chunk_ids[0] not in walked_ids
    assert chunk_ids[1] not in walked_ids
    assert chunk_ids[2] in walked_ids
    assert chunk_ids[3] in walked_ids


@pytest.mark.parametrize("provider_fixture", ["duckdb_provider", "lancedb_provider"])
def test_paginated_query_failure_does_not_look_like_complete(
    provider_fixture: str, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
):
    """Empty means done; executor errors must raise, not return []."""
    provider = request.getfixturevalue(provider_fixture)
    file_id = _insert_file(provider)
    _insert_chunks(provider, file_id, 1)

    def _boom(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        raise RuntimeError("simulated query failure")

    monkeypatch.setattr(
        provider,
        "_executor_get_chunks_without_embeddings_paginated",
        _boom,
    )

    with pytest.raises(RuntimeError, match="simulated query failure"):
        provider.get_chunks_without_embeddings_paginated(
            "test", "test-model", limit=10
        )
