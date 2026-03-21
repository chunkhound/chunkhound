import asyncio
from pathlib import Path

import pytest

from chunkhound.core.models import Embedding
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.indexing_coordinator import IndexingCoordinator


def _get_hnsw_index_names(provider: DuckDBProvider) -> list[str]:
    rows = provider.execute_query(
        """
        SELECT index_name
        FROM duckdb_indexes()
        WHERE table_name = 'embeddings_3'
          AND (index_name LIKE 'hnsw_%' OR index_name LIKE 'idx_hnsw_%')
        ORDER BY index_name
        """,
        [],
    )
    return [row["index_name"] for row in rows]


def _seed_embeddings(provider: DuckDBProvider, chunk_ids: list[int]) -> None:
    for index, chunk_id in enumerate(chunk_ids):
        provider.insert_embedding(
            Embedding(
                chunk_id=chunk_id,
                provider="test",
                model="mini",
                dims=3,
                vector=[float(index), float(index) + 0.1, float(index) + 0.2],
            )
        )


def _batch_insert_embeddings(
    provider: DuckDBProvider, chunk_ids: list[int], seed: float = 0.0
) -> int:
    payload = []
    for index, chunk_id in enumerate(chunk_ids):
        payload.append(
            {
                "chunk_id": chunk_id,
                "provider": "test",
                "model": "mini",
                "embedding": [
                    float(seed),
                    float(index) + 0.1,
                    float(seed + index) + 0.2,
                ],
                "dims": 3,
            }
        )
    return provider.insert_embeddings_batch(payload)


def test_process_file_avoids_in_transaction_hnsw_guard_for_modified_chunk_deletes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pytest.importorskip("duckdb")

    provider = DuckDBProvider(db_path=tmp_path / "db.duckdb", base_directory=tmp_path)
    provider.connect()
    try:
        coordinator = IndexingCoordinator(provider, tmp_path)

        test_file = tmp_path / "modified.py"
        test_file.write_text(
            "def keep():\n"
            "    return 1\n\n"
            "def replace_me():\n"
            "    return 2\n"
        )

        initial_result = asyncio.run(
            coordinator.process_file(test_file, skip_embeddings=True)
        )
        assert initial_result["status"] == "success"

        initial_chunks = provider.get_chunks_by_file_id(
            initial_result["file_id"], as_model=False
        )
        assert initial_chunks
        initial_chunk_ids = [int(chunk["id"]) for chunk in initial_chunks]
        _seed_embeddings(provider, initial_chunk_ids)

        initial_indexes = _get_hnsw_index_names(provider)
        if not initial_indexes:
            pytest.skip("DuckDB HNSW indexes are unavailable in this environment")

        guard_labels: list[str] = []
        original_guard = provider._executor_run_hnsw_guarded_mutation

        def _record_guard(
            conn,
            state,
            mutation_label,
            mutation_func,
            *,
            optimize_for_bulk=False,
            transactional=True,
            rollback_func=None,
        ):
            guard_labels.append(mutation_label)
            return original_guard(
                conn,
                state,
                mutation_label,
                mutation_func,
                optimize_for_bulk=optimize_for_bulk,
                transactional=transactional,
                rollback_func=rollback_func,
            )

        monkeypatch.setattr(
            provider, "_executor_run_hnsw_guarded_mutation", _record_guard
        )

        test_file.write_text(
            "def keep():\n"
            "    return 1\n\n"
            "def replaced():\n"
            "    return 3\n"
        )

        updated_result = asyncio.run(
            coordinator.process_file(test_file, skip_embeddings=True)
        )
        assert updated_result["status"] == "success"
        assert not any(
            label.startswith("delete_chunks_batch(") for label in guard_labels
        ), guard_labels
        assert _get_hnsw_index_names(provider) == initial_indexes
        assert provider.execute_query(
            "SELECT COUNT(*) AS count FROM chunks WHERE id IN (?, ?)",
            initial_chunk_ids,
        )[0]["count"] == 0
        assert provider.execute_query(
            "SELECT COUNT(*) AS count FROM embeddings_3 WHERE chunk_id IN (?, ?)",
            initial_chunk_ids,
        )[0]["count"] == 0

        updated_chunks = provider.get_chunks_by_file_id(
            updated_result["file_id"], as_model=False
        )
        updated_chunk_ids = [int(chunk["id"]) for chunk in updated_chunks]
        assert len(updated_chunk_ids) == 2
        assert _batch_insert_embeddings(provider, updated_chunk_ids, seed=10.0) == 2

        embedding_count = provider.execute_query(
            "SELECT COUNT(*) AS count FROM embeddings_3",
            [],
        )
        assert embedding_count[0]["count"] == len(updated_chunk_ids)
    finally:
        provider.disconnect(skip_checkpoint=True)


def test_delete_chunks_batch_still_uses_hnsw_guard_outside_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pytest.importorskip("duckdb")

    provider = DuckDBProvider(db_path=tmp_path / "db.duckdb", base_directory=tmp_path)
    provider.connect()
    try:
        coordinator = IndexingCoordinator(provider, tmp_path)

        test_file = tmp_path / "outside_guard.py"
        test_file.write_text(
            "def keep():\n"
            "    return 1\n\n"
            "def replace_me():\n"
            "    return 2\n"
        )

        result = asyncio.run(coordinator.process_file(test_file, skip_embeddings=True))
        stored_chunks = provider.get_chunks_by_file_id(
            result["file_id"], as_model=False
        )
        chunk_ids = [int(chunk["id"]) for chunk in stored_chunks]
        _seed_embeddings(provider, chunk_ids)

        guard_labels: list[str] = []
        original_guard = provider._executor_run_hnsw_guarded_mutation

        def _record_guard(
            conn,
            state,
            mutation_label,
            mutation_func,
            *,
            optimize_for_bulk=False,
            transactional=True,
            rollback_func=None,
        ):
            guard_labels.append(mutation_label)
            return original_guard(
                conn,
                state,
                mutation_label,
                mutation_func,
                optimize_for_bulk=optimize_for_bulk,
                transactional=transactional,
                rollback_func=rollback_func,
            )

        monkeypatch.setattr(
            provider, "_executor_run_hnsw_guarded_mutation", _record_guard
        )

        provider.delete_chunks_batch(chunk_ids)

        assert any(
            label.startswith("delete_chunks_batch(") for label in guard_labels
        ), guard_labels
        assert provider.execute_query(
            "SELECT COUNT(*) AS count FROM embeddings_3",
            [],
        )[0]["count"] == 0
    finally:
        provider.disconnect(skip_checkpoint=True)
