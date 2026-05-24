"""Scenario 2: Fragmentation sweep — correctness at 30/50/70/90% delete ratios.

Tests that:
- pre-compact effective_waste matches the expected delete ratio (±15%)
- post-compact effective_waste < 0.05 at every fragmentation level
- search results survive compaction intact (data integrity)
- no stale lock files or intent files remain after compaction
"""

from __future__ import annotations

import pytest

from tests.stress.conftest import make_compaction_config, make_fragmented_db, run_timed_compaction

pytestmark = pytest.mark.heavy


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "delete_ratio",
    [0.30, 0.50, 0.70, 0.90],
    ids=["del30", "del50", "del70", "del90"],
)
async def test_compaction_handles_all_fragmentation_levels(
    tmp_path, delete_ratio: float
):
    """Compaction must reduce waste to near-zero regardless of initial fragmentation."""
    fdb = make_fragmented_db(
        tmp_path,
        n_files=100,
        n_chunks_per_file=50,
        delete_ratio=delete_ratio,
    )
    provider, db_path = fdb.provider, fdb.db_path
    config = make_compaction_config(db_path, threshold=0.01, min_size_mb=0)

    stats_pre = provider.get_storage_stats()
    pre_waste = stats_pre["effective_waste"]

    # Fragmentation should roughly match the delete ratio (±15%)
    assert pre_waste >= delete_ratio - 0.15, (
        f"Expected pre-compact waste ≥ {delete_ratio - 0.15:.0%}, got {pre_waste:.0%}"
    )

    # Run compaction
    result = await run_timed_compaction(provider, db_path, config)

    assert result.performed, "Compaction was skipped unexpectedly"
    post_waste = result.stats_after["effective_waste"]

    print(
        f"\n[del={delete_ratio:.0%}] "
        f"waste {pre_waste:.0%} → {post_waste:.0%} | "
        f"{result.size_before_bytes / 1024:.0f}KB → {result.size_after_bytes / 1024:.0f}KB | "
        f"{result.elapsed_s:.2f}s"
    )

    assert post_waste < 0.05, (
        f"Post-compact waste {post_waste:.0%} should be <5%"
    )
    assert result.size_after_bytes <= result.size_before_bytes, (
        "DB must not grow after compaction"
    )

    # Data integrity: kept files should still be searchable
    # Files 0..n_delete-1 were deleted; first kept file is at index n_delete
    kept_file_idx = int(100 * delete_ratio)
    results = provider.search_chunks_regex(f"fn_{kept_file_idx}_0")
    assert len(results) > 0, (
        f"Search for fn_{kept_file_idx}_0 failed after compaction "
        f"(delete_ratio={delete_ratio:.0%})"
    )

    # No orphan compaction artifacts
    assert not (db_path.parent / (db_path.name + ".old")).exists(), "Backup file not cleaned up"
    assert not list(db_path.parent.glob("*.compact.duckdb")), "Compact temp file not cleaned up"
    assert not list(db_path.parent.glob("*.swap_intent")), "Intent file not cleaned up"

    provider.disconnect()
