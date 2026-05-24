"""Scenario 3: Concurrent access stress — connection gate under compaction load.

Verifies that:
- Operations issued before compaction complete normally
- Operations during the gate-closed window raise CompactionError(operation="connection")
- Operations after compaction complete normally
- No data corruption occurs at any point

The test uses compact_background() with a real (but instrumented) provider so that
the compaction thread genuinely closes the gate while async reader tasks run.
"""

from __future__ import annotations

import asyncio
import threading
from collections import Counter
from pathlib import Path

import pytest

from chunkhound.core.exceptions import CompactionError
from chunkhound.services.compaction_service import CompactionService
from tests.stress.conftest import make_compaction_config, make_fragmented_db

pytestmark = pytest.mark.heavy


@pytest.mark.asyncio
async def test_connection_gate_blocks_and_reopens_under_concurrent_readers(tmp_path: Path):
    """Gate closes during compaction and reopens after — no corruption at any phase."""
    fdb = make_fragmented_db(
        tmp_path,
        n_files=40,
        n_chunks_per_file=40,
        delete_ratio=0.70,
    )
    provider, db_path = fdb.provider, fdb.db_path
    config = make_compaction_config(db_path, threshold=0.01, min_size_mb=0)

    # Track results per phase
    results: Counter = Counter()
    stop_readers = threading.Event()
    gate_opened_after_compact = asyncio.Event()

    # We instrument the provider's optimize() to know exactly when the gate closes
    original_optimize = provider.optimize
    gate_closed_event = threading.Event()

    def instrumented_optimize(cancel_check=None):
        # By the time optimize() runs, the gate is already cleared.
        gate_closed_event.set()
        return original_optimize(cancel_check=cancel_check)

    provider.optimize = instrumented_optimize

    async def reader_task(phase_label: str) -> None:
        """Issue search_chunks_regex calls; record outcome per phase."""
        while not stop_readers.is_set():
            try:
                res = provider.search_chunks_regex("fn_0")
                results[f"{phase_label}_ok"] += 1
                # Data integrity: results must have expected fields
                for row in res:
                    assert "code" in row, "Missing 'code' field in search result"
            except CompactionError as e:
                assert e.operation == "connection", (
                    f"Unexpected CompactionError operation: {e.operation!r}"
                )
                results["gate_blocked"] += 1
            except Exception as e:
                results[f"unexpected_error:{type(e).__name__}"] += 1
            await asyncio.sleep(0)  # yield to event loop

    # --- Phase 1: readers run freely before compaction starts ---
    reader_tasks = [asyncio.create_task(reader_task("pre")) for _ in range(5)]

    # Let readers accumulate some successful reads
    await asyncio.sleep(0.05)
    pre_ok = results["pre_ok"]
    assert pre_ok > 0, "Readers should succeed before compaction"

    # --- Phase 2: start background compaction ---
    svc = CompactionService(db_path, config)
    started = await svc.compact_background(provider)
    assert started, "compact_background() should start compaction"

    # Wait for gate to actually close (compaction thread has started)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: gate_closed_event.wait(timeout=15.0))
    assert gate_closed_event.is_set(), "Gate did not close within 15s"

    # Give readers a moment to hit the closed gate
    await asyncio.sleep(0.05)
    assert results["gate_blocked"] > 0, "No readers were blocked by the gate — gate may not have closed"

    # --- Phase 3: wait for compaction to complete ---
    if svc._compaction_task:
        await asyncio.wait_for(svc._compaction_task, timeout=120.0)

    # At this point the gate should be open again
    assert provider.is_accepting_connections, "Gate should be open after compaction"

    # Let readers accumulate post-compaction successes
    await asyncio.sleep(0.05)
    post_ok = results["pre_ok"] + results.get("post_ok", 0)

    stop_readers.set()
    for t in reader_tasks:
        t.cancel()
    await asyncio.gather(*reader_tasks, return_exceptions=True)

    print(
        f"\n[gate stress] pre_ok={pre_ok} | gate_blocked={results['gate_blocked']} | "
        f"post-compact_ok={results.get('pre_ok', 0) - pre_ok} | "
        f"unexpected={sum(v for k, v in results.items() if 'unexpected' in k)}"
    )

    assert results["gate_blocked"] > 0, "Gate must have blocked at least one reader"
    unexpected = sum(v for k, v in results.items() if "unexpected_error" in k)
    assert unexpected == 0, f"Unexpected errors during stress: {dict(results)}"

    # DB integrity: post-compaction search should still find kept files
    # 40 files, 70% deleted = 28 deleted, kept files are indices 28..39
    final_results = provider.search_chunks_regex("fn_28_0")
    assert len(final_results) > 0, "Post-compaction search returned no results for kept chunks"

    provider.disconnect()


@pytest.mark.asyncio
async def test_same_service_concurrent_compact_calls_only_one_wins(tmp_path: Path):
    """Two concurrent compact_background() on the same service — only one proceeds."""
    fdb = make_fragmented_db(
        tmp_path,
        n_files=20,
        n_chunks_per_file=30,
        delete_ratio=0.70,
    )
    provider, db_path = fdb.provider, fdb.db_path
    config = make_compaction_config(db_path, threshold=0.01, min_size_mb=0)

    svc = CompactionService(db_path, config)

    # Fire two concurrent calls on the same service instance
    r1, r2 = await asyncio.gather(
        svc.compact_background(provider),
        svc.compact_background(provider),
    )

    # Exactly one should have acquired the compaction lock
    assert (r1 is True) != (r2 is True), (
        f"Expected exactly one to start, got r1={r1}, r2={r2}"
    )

    # Wait for compaction to finish
    if svc._compaction_task:
        await asyncio.wait_for(svc._compaction_task, timeout=120.0)

    # Verify compaction actually ran and cleaned up
    stats = provider.get_storage_stats()
    assert stats["effective_waste"] < 0.05, "Compaction should have reduced waste"

    provider.disconnect()
