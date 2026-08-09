"""Shared real-DuckDB seeding helpers for vector/HNSW tests.

File-backed providers only: DuckDB cannot persist an HNSW index in memory.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from pathlib import Path

import pytest

from chunkhound.core.config.database_config import DatabaseConfig
from chunkhound.core.models import Chunk, File
from chunkhound.core.types.common import (
    ChunkType,
    FileId,
    FilePath,
    Language,
    LineNumber,
    Timestamp,
)
from chunkhound.providers.database.duckdb_provider import DuckDBProvider

PROVIDER = "test"
MODEL = "mini"
QUERY_VECTOR = [1.0, 0.0, 0.0]


@contextmanager
def connected_provider(
    tmp_path: Path, config: DatabaseConfig | None = None
) -> Generator[DuckDBProvider, None, None]:
    """Yield a connected file-backed DuckDB provider."""
    database = DuckDBProvider(
        tmp_path / "search.duckdb", base_directory=tmp_path, config=config
    )
    database.connect()
    try:
        yield database
    finally:
        database.disconnect(skip_checkpoint=True)


def insert_chunks(provider: DuckDBProvider, path: str, count: int) -> list[int]:
    """Insert one file with ``count`` function chunks; return chunk ids."""
    file_id = provider.insert_file(
        File(
            path=FilePath(path),
            mtime=Timestamp(1.0),
            size_bytes=count,
            language=Language.PYTHON,
        )
    )
    return provider.insert_chunks_batch(
        [
            Chunk(
                file_id=FileId(file_id),
                symbol=f"function_{index}",
                start_line=LineNumber(index + 1),
                end_line=LineNumber(index + 1),
                code=f"def function_{index}(): return {index}",
                chunk_type=ChunkType.FUNCTION,
                language=Language.PYTHON,
            )
            for index in range(count)
        ]
    )


def insert_embeddings(
    provider: DuckDBProvider,
    chunk_ids: Sequence[int],
    vectors: Sequence[Sequence[float]],
    *,
    provider_name: str = PROVIDER,
    model: str = MODEL,
) -> None:
    """Insert embeddings tagged with an explicit provider/model pair."""
    provider.insert_embeddings_batch(
        [
            {
                "chunk_id": chunk_id,
                "provider": provider_name,
                "model": model,
                "embedding": list(vector),
                "dims": len(vector),
            }
            for chunk_id, vector in zip(chunk_ids, vectors, strict=True)
        ]
    )


def has_hnsw_index(provider: DuckDBProvider, dims: int) -> bool:
    """Report whether a real HNSW index exists for the given dimension table."""
    indexes = provider.execute_query(
        f"SELECT sql FROM duckdb_indexes() WHERE table_name = 'embeddings_{dims}'", []
    )
    return any("USING HNSW" in (index["sql"] or "").upper() for index in indexes)


def require_hnsw_index(provider: DuckDBProvider, dims: int = 3) -> None:
    """Require a real HNSW index, failing closed in the required CI lane."""
    if has_hnsw_index(provider, dims):
        return
    if os.getenv("CHUNKHOUND_REQUIRE_HNSW") == "1":
        pytest.fail(
            f"CHUNKHOUND_REQUIRE_HNSW=1 but no HNSW index was created for {dims}D"
        )
    pytest.skip("DuckDB HNSW indexes are unavailable in this environment")


def profile_last_query(
    provider: DuckDBProvider, output: Path, run: Callable[[], object]
) -> str:
    """Return DuckDB's upper-cased JSON profile of the last query ``run`` issued.

    Profiles the real executed plan (not a reconstructed EXPLAIN), so plan
    assertions stay honest across query-building refactors.
    """
    try:
        provider.execute_query("PRAGMA enable_profiling='json'", [])
        provider.execute_query(f"PRAGMA profiling_output='{output.as_posix()}'", [])
    except Exception:
        pytest.skip("DuckDB profiling unavailable in this environment")
    try:
        run()
    finally:
        try:
            provider.execute_query("PRAGMA disable_profiling", [])
        except Exception:
            pass
    try:
        text = output.read_text(encoding="utf-8")
    except FileNotFoundError:
        pytest.skip("DuckDB profiling produced no output in this environment")
    if not text.strip():
        pytest.skip("DuckDB profiling produced empty output in this environment")
    return text.upper()


def seed_searchable_chunks(
    provider: DuckDBProvider, count: int = 3, path: str = "src/module.py"
) -> list[int]:
    """Seed 3-D chunks clustered near ``QUERY_VECTOR`` with an HNSW index."""
    chunk_ids = insert_chunks(provider, path, count)
    insert_embeddings(
        provider, chunk_ids, [[1.0, 0.01 * index, 0.0] for index in range(count)]
    )
    require_hnsw_index(provider, 3)
    return chunk_ids
