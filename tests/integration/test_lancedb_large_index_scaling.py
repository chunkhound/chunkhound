"""Phase 3: LanceDB large-index read/write scaling contracts."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from chunkhound.core.models import Chunk, File
from chunkhound.core.types.common import ChunkType, Language


def _seed_chunks(provider, count: int = 20) -> list[int]:
    file_id = provider.insert_file(
        File(
            path="scale.py",
            mtime=1_700_000_000.0,
            language=Language.PYTHON,
            size_bytes=100,
        )
    )
    chunks = [
        Chunk(
            file_id=file_id,
            code=f"def scale_fn_{i}():\n    return {i}\n",
            start_line=i * 3 + 1,
            end_line=i * 3 + 2,
            chunk_type=ChunkType.FUNCTION,
            language=Language.PYTHON,
            symbol=f"scale_fn_{i}",
        )
        for i in range(count)
    ]
    return [int(i) for i in provider.insert_chunks_batch(chunks)]


def test_get_existing_embeddings_does_not_full_table_load(lancedb_provider):
    """Targeted existing-embedding lookup must not full-materialize the table."""
    chunk_ids = _seed_chunks(lancedb_provider, count=12)
    half = chunk_ids[:6]
    dims = 8
    lancedb_provider.insert_embeddings_batch(
        [
            {
                "chunk_id": cid,
                "provider": "fake",
                "model": "fake-model",
                "dims": dims,
                "embedding": [0.1 + (i * 0.01)] * dims,
            }
            for i, cid in enumerate(half)
        ]
    )

    table = lancedb_provider._chunks_table
    assert table is not None

    original_head = table.head
    head_spy = MagicMock(side_effect=original_head)
    table.head = head_spy  # type: ignore[method-assign]

    original_to_pandas = table.to_pandas
    pandas_spy = MagicMock(side_effect=original_to_pandas)
    table.to_pandas = pandas_spy  # type: ignore[method-assign]

    existing = lancedb_provider.get_existing_embeddings(
        chunk_ids, "fake", "fake-model"
    )
    assert set(half).issubset(existing)
    assert set(chunk_ids[6:]).isdisjoint(existing)
    head_spy.assert_not_called()
    # Full unfiltered to_pandas is the old OOM path — must not run on targeted lookup.
    pandas_spy.assert_not_called()


def test_get_existing_embeddings_empty_chunk_ids_scoped_to_provider(lancedb_provider):
    chunk_ids = _seed_chunks(lancedb_provider, count=5)
    dims = 8
    lancedb_provider.insert_embeddings_batch(
        [
            {
                "chunk_id": chunk_ids[0],
                "provider": "fake",
                "model": "fake-model",
                "dims": dims,
                "embedding": [0.2] * dims,
            },
            {
                "chunk_id": chunk_ids[1],
                "provider": "other",
                "model": "other-model",
                "dims": dims,
                "embedding": [0.3] * dims,
            },
        ]
    )

    existing = lancedb_provider.get_existing_embeddings(
        [], "fake", "fake-model"
    )
    assert chunk_ids[0] in existing
    assert chunk_ids[1] not in existing


def test_get_existing_embeddings_rejects_zero_vector_placeholders(lancedb_provider):
    chunk_ids = _seed_chunks(lancedb_provider, count=2)
    dims = 8
    lancedb_provider.insert_embeddings_batch(
        [
            {
                "chunk_id": chunk_ids[0],
                "provider": "fake",
                "model": "fake-model",
                "dims": dims,
                "embedding": [0.4] * dims,
            },
            {
                "chunk_id": chunk_ids[1],
                "provider": "fake",
                "model": "fake-model",
                "dims": dims,
                "embedding": [0.0] * dims,
            },
        ]
    )

    existing = lancedb_provider.get_existing_embeddings(
        chunk_ids, "fake", "fake-model"
    )
    assert chunk_ids[0] in existing
    assert chunk_ids[1] not in existing


def test_get_existing_embeddings_query_failure_raises(lancedb_provider, monkeypatch):
    chunk_ids = _seed_chunks(lancedb_provider, count=2)
    table = lancedb_provider._chunks_table
    assert table is not None

    def _boom_to_lance():
        raise RuntimeError("simulated existing-embeddings query failure")

    # Fail the native filtered path and search fallbacks so the real executor raises.
    monkeypatch.setattr(table, "to_lance", _boom_to_lance)

    def _boom_search(*_a, **_k):
        raise RuntimeError("simulated existing-embeddings query failure")

    monkeypatch.setattr(table, "search", _boom_search)

    with pytest.raises(RuntimeError, match="simulated existing-embeddings"):
        lancedb_provider.get_existing_embeddings(
            chunk_ids, "fake", "fake-model"
        )


def test_regex_pagination_fetches_page_without_duplicate_totals(lancedb_provider):
    _seed_chunks(lancedb_provider, count=15)
    results_p1, page1 = lancedb_provider.search_regex(
        "scale_fn_", page_size=5, offset=0
    )
    results_p2, page2 = lancedb_provider.search_regex(
        "scale_fn_", page_size=5, offset=5
    )

    assert page1["total"] == page2["total"]
    assert page1["total"] >= 10
    assert len(results_p1) == 5
    assert len(results_p2) == 5
    assert page1["has_more"] is True
    ids_p1 = [r["chunk_id"] for r in results_p1]
    ids_p2 = [r["chunk_id"] for r in results_p2]
    assert set(ids_p1).isdisjoint(ids_p2)
    # Ordered by id across pages
    assert ids_p1 == sorted(ids_p1)
    assert ids_p2 == sorted(ids_p2)
    assert max(ids_p1) < min(ids_p2)


def test_insert_embeddings_triggers_optimize_when_over_threshold(
    lancedb_provider, monkeypatch: pytest.MonkeyPatch
):
    """When fragment count exceeds threshold, insert path runs optimize."""
    chunk_ids = _seed_chunks(lancedb_provider, count=3)
    optimize_calls: list[str] = []

    def _fake_optimize(conn, state) -> None:
        optimize_calls.append("optimize")

    public_optimize = MagicMock(
        side_effect=lancedb_provider.optimize_tables
    )
    monkeypatch.setattr(
        lancedb_provider, "_executor_optimize_tables", _fake_optimize
    )
    monkeypatch.setattr(lancedb_provider, "optimize_tables", public_optimize)
    monkeypatch.setattr(lancedb_provider, "_fragment_threshold", 0)

    stored = lancedb_provider.insert_embeddings_batch(
        [
            {
                "chunk_id": chunk_ids[0],
                "provider": "fake",
                "model": "fake-model",
                "dims": 8,
                "embedding": [0.5] * 8,
            }
        ]
    )
    assert stored == 1
    assert optimize_calls == ["optimize"]
    # In-executor path only — public wrapper must not re-enter serial executor.
    public_optimize.assert_not_called()


def test_insert_embeddings_skips_optimize_under_threshold(
    lancedb_provider, monkeypatch: pytest.MonkeyPatch
):
    chunk_ids = _seed_chunks(lancedb_provider, count=2)
    optimize_calls: list[str] = []

    def _fake_optimize(conn, state) -> None:
        optimize_calls.append("optimize")

    monkeypatch.setattr(
        lancedb_provider, "_executor_optimize_tables", _fake_optimize
    )
    monkeypatch.setattr(lancedb_provider, "_fragment_threshold", 10_000)

    lancedb_provider.insert_embeddings_batch(
        [
            {
                "chunk_id": chunk_ids[0],
                "provider": "fake",
                "model": "fake-model",
                "dims": 8,
                "embedding": [0.5] * 8,
            }
        ]
    )
    assert optimize_calls == []


def test_insert_chunks_triggers_optimize_when_over_threshold(
    lancedb_provider, monkeypatch: pytest.MonkeyPatch
):
    optimize_calls: list[str] = []

    def _fake_optimize(conn, state) -> None:
        optimize_calls.append("optimize")

    monkeypatch.setattr(
        lancedb_provider, "_executor_optimize_tables", _fake_optimize
    )
    monkeypatch.setattr(lancedb_provider, "_fragment_threshold", 0)

    _seed_chunks(lancedb_provider, count=2)
    assert optimize_calls == ["optimize"]


def test_num_fragments_from_stats_supports_dict_and_object():
    from types import SimpleNamespace

    from chunkhound.providers.database.lancedb_provider import LanceDBProvider

    assert LanceDBProvider._num_fragments_from_stats(None) == 0
    assert (
        LanceDBProvider._num_fragments_from_stats(
            {"fragment_stats": {"num_fragments": 42}}
        )
        == 42
    )
    assert (
        LanceDBProvider._num_fragments_from_stats(
            SimpleNamespace(
                fragment_stats=SimpleNamespace(num_fragments=7)
            )
        )
        == 7
    )
    # Mixed nesting
    assert (
        LanceDBProvider._num_fragments_from_stats(
            {"fragment_stats": SimpleNamespace(num_fragments=3)}
        )
        == 3
    )
    assert (
        LanceDBProvider._num_fragments_from_stats(
            SimpleNamespace(fragment_stats={"num_fragments": 9})
        )
        == 9
    )
    assert LanceDBProvider._num_fragments_from_stats({"num_fragments": 11}) == 11
    assert LanceDBProvider._num_fragments_from_stats({"fragment_count": 13}) == 13
