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


def test_process_file_uses_hnsw_guard_for_modified_chunk_deletes(
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
        _seed_embeddings(provider, [int(chunk["id"]) for chunk in initial_chunks])

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
        assert any(
            label.startswith("delete_chunks_batch(") for label in guard_labels
        ), guard_labels
        assert _get_hnsw_index_names(provider) == initial_indexes

        followup_file = tmp_path / "followup.py"
        followup_file.write_text("def followup():\n    return 4\n")
        followup_result = asyncio.run(
            coordinator.process_file(followup_file, skip_embeddings=True)
        )
        assert followup_result["status"] == "success"
        followup_chunks = provider.get_chunks_by_file_id(
            followup_result["file_id"], as_model=False
        )
        assert followup_chunks
        _seed_embeddings(provider, [int(chunk["id"]) for chunk in followup_chunks])

        embedding_count = provider.execute_query(
            "SELECT COUNT(*) AS count FROM embeddings_3",
            [],
        )
        assert embedding_count[0]["count"] >= len(followup_chunks)
    finally:
        provider.disconnect(skip_checkpoint=True)
