import io
from pathlib import Path

import pytest
from loguru import logger

from chunkhound.core.models import Chunk, File
from chunkhound.core.types.common import ChunkType, FilePath, Language, LineNumber
from chunkhound.providers.database.duckdb_provider import DuckDBProvider


def _insert_yaml_chunks(provider: DuckDBProvider) -> None:
    file_id = provider.insert_file(
        File(
            path=FilePath("a.yaml"),
            mtime=0.0,
            size_bytes=10,
            language=Language.YAML,
        )
    )
    provider.insert_chunks_batch([
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
    ])


def test_duckdb_chunk_metrics_emitted(tmp_path: Path, monkeypatch):
    """Chunk bulk metrics remain visible in normal CLI mode."""
    monkeypatch.delenv("CHUNKHOUND_MCP_MODE", raising=False)

    buf = io.StringIO()
    sink_id = logger.add(buf, level="INFO")

    provider = DuckDBProvider(
        db_path=tmp_path / "metrics.duckdb",
        base_directory=tmp_path,
    )
    try:
        provider.connect()
        _insert_yaml_chunks(provider)
        provider.optimize_tables()
    finally:
        provider.disconnect()
        logger.remove(sink_id)

    out = buf.getvalue()
    assert "DuckDB chunks bulk metrics:" in out
    assert "files=1" in out
    assert "rows=2" in out


def test_compaction_logging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Compaction logs contain size and reduction info (external observable)."""
    monkeypatch.delenv("CHUNKHOUND_MCP_MODE", raising=False)
    buf = io.StringIO()
    sink_id = logger.add(buf, level="INFO")

    provider = DuckDBProvider(
        db_path=tmp_path / "compact.duckdb",
        base_directory=tmp_path,
    )
    try:
        provider.connect()
        _insert_yaml_chunks(provider)
        provider.compact_database()
    finally:
        provider.disconnect()
        logger.remove(sink_id)

    out = buf.getvalue()
    assert "Compaction complete:" in out
    assert "bytes" in out
    assert "% reduction" in out
