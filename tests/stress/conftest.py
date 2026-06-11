"""Shared fixtures and helpers for compaction stress tests."""

from __future__ import annotations

import time
from pathlib import Path
from typing import NamedTuple

import pytest

from chunkhound.core.config.config import Config
from chunkhound.core.config.database_config import DatabaseConfig
from chunkhound.core.models import Chunk, File
from chunkhound.core.types.common import ChunkType, Language
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.compaction_service import CompactionService


class FragmentedDB(NamedTuple):
    provider: DuckDBProvider
    db_path: Path
    n_files_kept: int
    n_files_deleted: int
    n_chunks_per_file: int


def make_fragmented_db(
    tmp_path: Path,
    *,
    n_files: int,
    n_chunks_per_file: int,
    delete_ratio: float = 0.7,
) -> FragmentedDB:
    """Create a DuckDB with synthetic data, then delete a fraction to create fragmentation.

    Uses insert_chunks_batch() for speed — roughly 250× faster than single inserts.
    delete_ratio=0.7 means 70% of files are deleted, leaving ~70% row waste.
    """
    db_path = tmp_path / "stress.duckdb"
    provider = DuckDBProvider(str(db_path), base_directory=tmp_path)
    provider.connect()

    n_delete = int(n_files * delete_ratio)
    n_keep = n_files - n_delete

    file_ids: list[int] = []
    for i in range(n_files):
        f = File(
            path=f"stress_{i:04d}.py",
            mtime=1_700_000_000.0 + i,
            language=Language.PYTHON,
            size_bytes=n_chunks_per_file * 512,
        )
        file_id = provider.insert_file(f)
        file_ids.append(file_id)

        chunks = [
            Chunk(
                file_id=file_id,
                code=f"def fn_{i}_{j}(): pass  # stress payload " + "x" * 400,
                start_line=j * 8 + 1,
                end_line=j * 8 + 5,
                chunk_type=ChunkType.FUNCTION,
                symbol=f"fn_{i}_{j}",
                language=Language.PYTHON,
            )
            for j in range(n_chunks_per_file)
        ]
        provider.insert_chunks_batch(chunks)

    provider.optimize_tables()

    for i in range(n_delete):
        provider.delete_file_completely(f"stress_{i:04d}.py")

    provider.optimize_tables()

    return FragmentedDB(
        provider=provider,
        db_path=db_path,
        n_files_kept=n_keep,
        n_files_deleted=n_delete,
        n_chunks_per_file=n_chunks_per_file,
    )


def make_compaction_config(
    db_path: Path,
    *,
    threshold: float = 0.01,
    min_size_mb: int = 0,
) -> Config:
    """Build a Config that makes compaction trigger at tiny fragmentation levels."""
    config = Config(
        database=DatabaseConfig(
            path=str(db_path),
            provider="duckdb",
            compaction_enabled=True,
            compaction_threshold=threshold,
            compaction_min_size_mb=min_size_mb,
        )
    )
    return config


class TimedCompaction(NamedTuple):
    performed: bool
    elapsed_s: float
    stats_before: dict
    stats_after: dict
    size_before_bytes: int
    size_after_bytes: int


async def run_timed_compaction(
    provider: DuckDBProvider,
    db_path: Path,
    config: Config,
) -> TimedCompaction:
    """Run compact_blocking() and collect before/after metrics."""
    stats_before = provider.get_storage_stats()
    size_before = db_path.stat().st_size

    svc = CompactionService(db_path, config)
    t0 = time.perf_counter()
    performed = await svc.compact_blocking(provider)
    elapsed = time.perf_counter() - t0

    stats_after = provider.get_storage_stats()
    size_after = db_path.stat().st_size

    return TimedCompaction(
        performed=performed,
        elapsed_s=elapsed,
        stats_before=stats_before,
        stats_after=stats_after,
        size_before_bytes=size_before,
        size_after_bytes=size_after,
    )
