"""Scenario 4: Index → compact → index → compact lifecycle correctness.

Simulates a real dev session:
  1. Index a batch of files
  2. Delete stale files, index replacements (fragmentation accumulates)
  3. Compact — waste must drop
  4. Repeat: add more files, delete old ones
  5. Compact again — still cleans up

Asserts waste decreases after each compaction cycle and search stays consistent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chunkhound.core.config.config import Config
from chunkhound.core.config.database_config import DatabaseConfig
from chunkhound.core.models import Chunk, File
from chunkhound.core.types.common import ChunkType, Language
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.compaction_service import CompactionService

pytestmark = pytest.mark.heavy


def _insert_file_batch(
    provider: DuckDBProvider,
    prefix: str,
    n_files: int,
    n_chunks: int,
) -> list[str]:
    """Insert n_files each with n_chunks. Returns list of inserted paths."""
    paths: list[str] = []
    for i in range(n_files):
        path = f"{prefix}_{i:03d}.py"
        f = File(
            path=path,
            mtime=1_700_000_000.0 + i,
            language=Language.PYTHON,
            size_bytes=n_chunks * 512,
        )
        file_id = provider.insert_file(f)
        chunks = [
            Chunk(
                file_id=file_id,
                code=f"def {prefix}_fn_{i}_{j}(): return {j}  # data " + "y" * 300,
                start_line=j * 6 + 1,
                end_line=j * 6 + 4,
                chunk_type=ChunkType.FUNCTION,
                symbol=f"{prefix}_fn_{i}_{j}",
                language=Language.PYTHON,
            )
            for j in range(n_chunks)
        ]
        provider.insert_chunks_batch(chunks)
        paths.append(path)
    return paths


def _delete_batch(provider: DuckDBProvider, paths: list[str]) -> None:
    for p in paths:
        provider.delete_file_completely(p)
    provider.optimize_tables()


def _make_config(db_path: Path) -> Config:
    return Config(
        database=DatabaseConfig(
            path=str(db_path),
            provider="duckdb",
            compaction_enabled=True,
            compaction_threshold=0.01,
            compaction_min_size_mb=0,
        )
    )


@pytest.mark.asyncio
async def test_two_cycle_compaction_lifecycle(tmp_path: Path):
    """Waste decreases after each compact cycle; search stays consistent."""
    db_path = tmp_path / "lifecycle.duckdb"
    provider = DuckDBProvider(str(db_path), base_directory=tmp_path)
    provider.connect()

    config = _make_config(db_path)

    # === Cycle 1: index batch A ===
    batch_a = _insert_file_batch(provider, "batch_a", n_files=60, n_chunks=30)
    provider.optimize_tables()

    # Delete 80% of batch A to create fragmentation
    _delete_batch(provider, batch_a[:48])

    waste_pre_c1 = provider.get_storage_stats()["effective_waste"]
    assert waste_pre_c1 > 0.05, f"Expected fragmentation before cycle 1, got {waste_pre_c1:.0%}"

    svc1 = CompactionService(db_path, config)
    performed1 = await svc1.compact_blocking(provider)

    waste_post_c1 = provider.get_storage_stats()["effective_waste"]
    print(f"\n[cycle 1] waste {waste_pre_c1:.0%} → {waste_post_c1:.0%}")

    assert performed1, "Compaction skipped in cycle 1"
    assert waste_post_c1 < waste_pre_c1, "Waste must decrease after cycle 1 compaction"
    assert waste_post_c1 < 0.10, f"Waste {waste_post_c1:.0%} still too high after cycle 1"

    # Kept files still searchable
    kept_results = provider.search_chunks_regex("batch_a_fn_48_0")
    assert len(kept_results) > 0, "Kept batch_a files must still be searchable after cycle 1"

    # === Cycle 2: index batch B, delete most, compact again ===
    batch_b = _insert_file_batch(provider, "batch_b", n_files=60, n_chunks=30)
    provider.optimize_tables()

    _delete_batch(provider, batch_b[:50])

    waste_pre_c2 = provider.get_storage_stats()["effective_waste"]
    assert waste_pre_c2 > 0.05, f"Expected fragmentation before cycle 2, got {waste_pre_c2:.0%}"

    svc2 = CompactionService(db_path, config)
    performed2 = await svc2.compact_blocking(provider)

    waste_post_c2 = provider.get_storage_stats()["effective_waste"]
    print(f"[cycle 2] waste {waste_pre_c2:.0%} → {waste_post_c2:.0%}")

    assert performed2, "Compaction skipped in cycle 2"
    assert waste_post_c2 < waste_pre_c2, "Waste must decrease after cycle 2 compaction"
    assert waste_post_c2 < 0.10, f"Waste {waste_post_c2:.0%} still too high after cycle 2"

    # Both batch_a and batch_b kept files searchable (kept indices are 48+ and 50+)
    for pattern in ("batch_a_fn_55_0", "batch_b_fn_55_0"):
        results = provider.search_chunks_regex(pattern)
        assert len(results) > 0, f"Search for {pattern!r} failed after cycle 2 compaction"

    # No orphan artifacts
    assert not list(tmp_path.glob("*.swap_intent")), "Intent file not cleaned up"
    assert not list(tmp_path.glob("*.compact.duckdb")), "Compact temp file not cleaned up"

    provider.disconnect()
