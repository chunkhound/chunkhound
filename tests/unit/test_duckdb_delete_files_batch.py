from pathlib import Path

import pytest

from chunkhound.core.models import Chunk, Embedding, File
from chunkhound.core.types.common import ChunkType, FileId, Language, LineNumber
from chunkhound.providers.database.duckdb_provider import DuckDBProvider


def _insert_file_with_embedding(
    provider: DuckDBProvider, file_path: str, vector_seed: float
) -> tuple[int, int]:
    file_id = provider.insert_file(
        File(path=file_path, mtime=1.0, size_bytes=24, language=Language.PYTHON)
    )
    chunk_id = provider.insert_chunk(
        Chunk(
            file_id=FileId(file_id),
            symbol=file_path,
            start_line=LineNumber(1),
            end_line=LineNumber(2),
            code=f"def {Path(file_path).stem}():\n    return {vector_seed}\n",
            chunk_type=ChunkType.FUNCTION,
            language=Language.PYTHON,
        )
    )
    provider.insert_embedding(
        Embedding(
            chunk_id=chunk_id,
            provider="test",
            model="mini",
            dims=3,
            vector=[vector_seed, vector_seed + 0.1, vector_seed + 0.2],
        )
    )
    return file_id, chunk_id


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


def test_delete_files_batch_removes_selected_files_and_preserves_indexes(
    tmp_path: Path,
):
    pytest.importorskip("duckdb")

    provider = DuckDBProvider(db_path=tmp_path / "db.duckdb", base_directory=tmp_path)
    provider.connect()

    _, first_chunk_id = _insert_file_with_embedding(provider, "orphan_one.py", 0.1)
    _, second_chunk_id = _insert_file_with_embedding(provider, "orphan_two.py", 0.4)
    keep_file_id, keep_chunk_id = _insert_file_with_embedding(provider, "keep.py", 0.7)

    initial_indexes = _get_hnsw_index_names(provider)
    if not initial_indexes:
        pytest.skip("DuckDB HNSW indexes are unavailable in this environment")

    deleted_count = provider.delete_files_batch(["orphan_one.py", "orphan_two.py"])

    assert deleted_count == 2
    assert provider.get_file_by_path("orphan_one.py", as_model=False) is None
    assert provider.get_file_by_path("orphan_two.py", as_model=False) is None
    assert provider.get_file_by_path("keep.py", as_model=False) is not None
    assert provider.get_chunks_by_file_id(keep_file_id, as_model=False) != []

    deleted_embeddings = provider.execute_query(
        "SELECT COUNT(*) AS count FROM embeddings_3 WHERE chunk_id IN (?, ?)",
        [first_chunk_id, second_chunk_id],
    )
    assert deleted_embeddings[0]["count"] == 0
    keep_embeddings = provider.execute_query(
        "SELECT COUNT(*) AS count FROM embeddings_3 WHERE chunk_id = ?",
        [keep_chunk_id],
    )
    assert keep_embeddings[0]["count"] == 1
    assert _get_hnsw_index_names(provider) == initial_indexes

    followup_file_id, followup_chunk_id = _insert_file_with_embedding(
        provider, "followup.py", 1.0
    )
    assert provider.get_file_by_path("followup.py", as_model=False) is not None
    assert provider.get_chunks_by_file_id(followup_file_id, as_model=False) != []
    followup_embeddings = provider.execute_query(
        "SELECT COUNT(*) AS count FROM embeddings_3 WHERE chunk_id = ?",
        [followup_chunk_id],
    )
    assert followup_embeddings[0]["count"] == 1


def test_delete_files_batch_restores_rows_when_hnsw_index_recreation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pytest.importorskip("duckdb")

    provider = DuckDBProvider(db_path=tmp_path / "db.duckdb", base_directory=tmp_path)
    provider.connect()

    _, first_chunk_id = _insert_file_with_embedding(provider, "orphan_one.py", 0.1)
    _, second_chunk_id = _insert_file_with_embedding(provider, "orphan_two.py", 0.4)

    initial_indexes = _get_hnsw_index_names(provider)
    if not initial_indexes:
        pytest.skip("DuckDB HNSW indexes are unavailable in this environment")

    recreate_attempts = {"count": 0}
    original_recreate = provider._executor_recreate_vector_index_from_info

    def _fail_once(conn, state, index_info):
        recreate_attempts["count"] += 1
        if recreate_attempts["count"] == 1:
            raise RuntimeError("forced hnsw recreate failure")
        return original_recreate(conn, state, index_info)

    monkeypatch.setattr(
        provider, "_executor_recreate_vector_index_from_info", _fail_once
    )

    with pytest.raises(RuntimeError, match="forced hnsw recreate failure"):
        provider.delete_files_batch(["orphan_one.py", "orphan_two.py"])

    assert recreate_attempts["count"] >= 2
    assert provider.get_file_by_path("orphan_one.py", as_model=False) is not None
    assert provider.get_file_by_path("orphan_two.py", as_model=False) is not None
    restored_embeddings = provider.execute_query(
        "SELECT COUNT(*) AS count FROM embeddings_3 WHERE chunk_id IN (?, ?)",
        [first_chunk_id, second_chunk_id],
    )
    assert restored_embeddings[0]["count"] == 2
    assert _get_hnsw_index_names(provider) == initial_indexes

    assert provider.delete_files_batch(["orphan_one.py", "orphan_two.py"]) == 2
