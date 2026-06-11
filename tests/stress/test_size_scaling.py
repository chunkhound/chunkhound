"""Scenario 1: Size scaling — measure compaction duration and reduction as DB grows.

Three sizes:
  small:  20 files × 50 chunks  (~2 MB)
  medium: 100 files × 100 chunks (~20 MB)
  large:  500 files × 200 chunks (~200 MB)

Each is fragmented at 70% delete ratio before compaction.
"""

from __future__ import annotations

import pytest

from tests.stress.conftest import make_compaction_config, make_fragmented_db, run_timed_compaction

pytestmark = pytest.mark.heavy


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "level,n_files,n_chunks",
    [
        ("small", 20, 50),
        ("medium", 100, 100),
        ("large", 500, 200),
    ],
)
async def test_compaction_size_scaling(tmp_path, level: str, n_files: int, n_chunks: int):
    """Compaction must reduce DB size and waste for all three scale points."""
    fdb = make_fragmented_db(
        tmp_path,
        n_files=n_files,
        n_chunks_per_file=n_chunks,
        delete_ratio=0.70,
    )
    provider, db_path = fdb.provider, fdb.db_path
    config = make_compaction_config(db_path, threshold=0.01, min_size_mb=0)

    result = await run_timed_compaction(provider, db_path, config)

    before_mb = result.size_before_bytes / (1024 * 1024)
    after_mb = result.size_after_bytes / (1024 * 1024)
    reduction_pct = (1 - result.size_after_bytes / result.size_before_bytes) * 100
    before_waste = result.stats_before["effective_waste"]
    after_waste = result.stats_after["effective_waste"]

    print(
        f"\n[{level}] {before_mb:.1f} MB → {after_mb:.1f} MB "
        f"({reduction_pct:.0f}% reduction) in {result.elapsed_s:.1f}s | "
        f"waste {before_waste:.0%} → {after_waste:.0%}"
    )

    assert result.performed, "Compaction was skipped unexpectedly"
    assert after_waste < before_waste, "Waste must decrease after compaction"
    assert after_waste < 0.05, f"Post-compact waste {after_waste:.0%} must be <5%"
    assert result.size_after_bytes < result.size_before_bytes, (
        "DB size must shrink after compaction"
    )
    assert reduction_pct >= 30, (
        f"Expected ≥30% size reduction at 70% fragmentation, got {reduction_pct:.0f}%"
    )

    provider.disconnect()
