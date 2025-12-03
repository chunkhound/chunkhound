import io
import os
from pathlib import Path

from loguru import logger

from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.core.models import File, Chunk
from chunkhound.core.types.common import FilePath, Language, ChunkType, LineNumber


def test_duckdb_chunk_metrics_emitted(tmp_path: Path, monkeypatch):
    # Ensure CHUNKHOUND_MCP_MODE is not set (prevents optimize_tables from returning early)
    monkeypatch.delenv("CHUNKHOUND_MCP_MODE", raising=False)

    db_dir = tmp_path / "db"

    # Capture loguru output
    buf = io.StringIO()
    logger.remove()
    logger.add(buf, level="INFO")

    provider = DuckDBProvider(db_path=db_dir, base_directory=tmp_path)
    provider.connect()

    file_id = provider.insert_file(
        File(path=FilePath("a.yaml"), mtime=0.0, size_bytes=10, language=Language.YAML)
    )
    chunks = [
        Chunk(
            file_id=file_id,
            chunk_type=ChunkType.KEY_VALUE,
            symbol="a",
            code="x: 1",
            start_line=LineNumber(1),
            end_line=LineNumber(1),
            language=Language.YAML,
        ),
        Chunk(
            file_id=file_id,
            chunk_type=ChunkType.BLOCK,
            symbol="b",
            code="b:\n  c: 2",
            start_line=LineNumber(2),
            end_line=LineNumber(3),
            language=Language.YAML,
        ),
    ]

    provider.insert_chunks_batch(chunks)
    provider.optimize_tables()

    out = buf.getvalue()
    assert "DuckDB chunks bulk metrics:" in out
    assert "files=1" in out
    assert "rows=2" in out

