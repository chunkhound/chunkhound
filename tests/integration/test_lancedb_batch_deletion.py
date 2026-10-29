"""Public LanceDB batch deletion contracts, including inline vector cleanup."""

import pytest

from chunkhound.core.models import Chunk, File
from chunkhound.core.types.common import ChunkType, Language
from chunkhound.providers.database.lancedb_provider import LanceDBProvider


@pytest.mark.asyncio
@pytest.mark.parametrize("async_delete", [False, True], ids=["sync", "async"])
async def test_batch_deletion_removes_only_selected_chunks_and_vectors_after_reopen(
    lancedb_provider, tmp_path, async_delete
):
    file_id = lancedb_provider.insert_file(
        File(
            path="sample.py",
            mtime=1234567890.0,
            language=Language.PYTHON,
            size_bytes=100,
        )
    )
    chunk_ids = lancedb_provider.insert_chunks_batch(
        [
            Chunk(
                file_id=file_id,
                code=f"def function_{i}(): return {i}",
                start_line=i + 1,
                end_line=i + 1,
                chunk_type=ChunkType.FUNCTION,
                language=Language.PYTHON,
                symbol=f"function_{i}",
            )
            for i in range(101)
        ]
    )
    assert len(set(chunk_ids)) == 101
    assert (
        lancedb_provider.insert_embeddings_batch(
            [
                {
                    "chunk_id": chunk_id,
                    "provider": "test",
                    "model": "test-model",
                    "dims": 8,
                    "embedding": [0.25] * 8,
                }
                for chunk_id in chunk_ids[:100]
            ]
        )
        == 100
    )

    # Include a row without a vector, duplicates, and an absent ID. Empty and
    # repeated requests must be harmless, without deleting unrelated rows.
    selected_ids = [chunk_ids[0], chunk_ids[1], chunk_ids[100], chunk_ids[0], -1]
    for batch in ([], selected_ids, selected_ids):
        if async_delete:
            await lancedb_provider.delete_chunks_batch_async(batch)
        else:
            lancedb_provider.delete_chunks_batch(batch)

    survivors = set(chunk_ids[2:100])
    assert {
        chunk["id"] for chunk in lancedb_provider.get_chunks_by_file_id(file_id)
    } == survivors
    assert (
        lancedb_provider.get_existing_embeddings(chunk_ids, "test", "test-model")
        == survivors
    )

    lancedb_provider.disconnect()
    reopened = LanceDBProvider(lancedb_provider.db_path, base_directory=tmp_path)
    reopened.connect()
    try:
        assert {
            chunk["id"] for chunk in reopened.get_chunks_by_file_id(file_id)
        } == survivors
        assert (
            reopened.get_existing_embeddings(chunk_ids, "test", "test-model")
            == survivors
        )
        embedding = reopened.get_embedding_by_chunk_id(
            chunk_ids[2], "test", "test-model"
        )
        assert embedding is not None
        assert embedding.vector == [0.25] * 8
    finally:
        reopened.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("async_delete", [False, True], ids=["sync", "async"])
async def test_batch_deletion_reports_storage_failure(
    lancedb_provider, monkeypatch, async_delete
):
    from lancedb.table import LanceTable

    def unavailable_storage(self, predicate):
        raise OSError("storage unavailable")

    monkeypatch.setattr(LanceTable, "delete", unavailable_storage)
    with pytest.raises(OSError, match="storage unavailable"):
        if async_delete:
            await lancedb_provider.delete_chunks_batch_async([1, 2])
        else:
            lancedb_provider.delete_chunks_batch([1, 2])
