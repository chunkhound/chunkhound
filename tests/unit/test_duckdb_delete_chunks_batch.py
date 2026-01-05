from pathlib import Path

from chunkhound.core.models import Chunk, File
from chunkhound.core.types.common import ChunkType, FilePath, Language, LineNumber
from chunkhound.providers.database.duckdb_provider import DuckDBProvider


def test_delete_chunks_batch_removes_only_selected(tmp_path: Path):
    provider = DuckDBProvider(db_path=tmp_path / "db", base_directory=tmp_path)
    provider.connect()

    file_id = provider.insert_file(
        File(path=FilePath("a.yaml"), mtime=0.0, size_bytes=10, language=Language.YAML)
    )
    # Insert 5 chunks
    chunks = [
        Chunk(
            file_id=file_id,
            chunk_type=ChunkType.KEY_VALUE,
            symbol=f"k{i}",
            code=f"x: {i}",
            start_line=LineNumber(i + 1),
            end_line=LineNumber(i + 1),
            language=Language.YAML,
        )
        for i in range(5)
    ]
    ids = provider.insert_chunks_batch(chunks)
    assert len(ids) == 5

    # Delete 2 of them
    to_delete = ids[1:3]
    provider.delete_chunks_batch(to_delete)

    remaining = provider.get_chunks_by_file_id(file_id, as_model=False)
    remaining_ids = {row["id"] for row in remaining}
    assert len(remaining_ids) == 3
    assert set(to_delete).isdisjoint(remaining_ids)
