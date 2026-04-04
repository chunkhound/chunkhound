"""Unit tests for the debounce mechanism in RealtimeIndexingService.

Tests call add_file / _debounced_add_file directly — no service start(),
no filesystem, no database — purely asyncio timing behaviour.
"""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from chunkhound.core.utils.path_utils import normalize_realtime_path
from chunkhound.services.realtime_indexing_service import RealtimeIndexingService


@pytest.fixture
async def service():
    """Minimal RealtimeIndexingService — enough to exercise debounce logic."""
    svc = RealtimeIndexingService(
        services=MagicMock(),
        config=MagicMock(),
    )
    svc._debounce_delay = 0.5  # Large enough for CI scheduling jitter
    yield svc
    await svc.stop()  # Cancel any lingering debounce tasks


async def _drain_queue(q: asyncio.Queue) -> list:
    """Return all items currently in a queue without blocking."""
    items = []
    while not q.empty():
        items.append(await q.get())
    return items


async def _wait_for_tasks(svc: RealtimeIndexingService, timeout: float = 2.0) -> None:
    """Wait until all debounce tasks have completed."""
    if svc._debounce_tasks:
        await asyncio.wait_for(
            asyncio.gather(*svc._debounce_tasks.copy(), return_exceptions=True),
            timeout=timeout,
        )


def _expected_mutation_path(path: Path) -> Path:
    """Match the realtime service's cross-platform path normalization."""
    return normalize_realtime_path(path)


class TestDebounce:
    async def test_rapid_changes_single_queue_entry(self, service):
        """10 rapid add_file calls for the same file produce exactly 1 queue entry."""
        path = Path("/tmp/test.py")
        for _ in range(10):
            await service.add_file(path, priority="change")

        await _wait_for_tasks(service)
        items = await _drain_queue(service.file_queue)
        assert len(items) == 1
        assert items[0][2].path == _expected_mutation_path(path)

    async def test_debounce_timestamp_refresh(self, service):
        """A second add_file during the delay resets the clock."""
        path = Path("/tmp/refresh.py")
        delay = service._debounce_delay

        await service.add_file(path, priority="change")
        # Halfway through delay, trigger another change — resets the clock
        await asyncio.sleep(delay * 0.6)
        await service.add_file(path, priority="change")
        # Sleep another 0.3×delay: 0.9×delay has elapsed from the first call but only
        # 0.3×delay from the second, so no queue entry should exist yet.
        await asyncio.sleep(delay * 0.3)
        assert service.file_queue.empty(), "File queued before debounce window expired"

        # Now wait for the full delay from the second call
        await _wait_for_tasks(service)
        assert not service.file_queue.empty(), "File not queued after debounce window"

    async def test_cleanup_after_debounce(self, service):
        """After debounce completes, _pending_debounce and _debounce_tasks are empty."""
        path = Path("/tmp/cleanup.py")
        await service.add_file(path, priority="change")
        await _wait_for_tasks(service)

        assert len(service._pending_debounce) == 0
        assert len(service._debounce_tasks) == 0

    async def test_pending_files_cleared_on_external_cleanup(self, service):
        """Early return path releases debounce bookkeeping cleanly."""
        path = Path("/tmp/external.py")
        await service.add_file(path, priority="change")

        # Simulate external cleanup of _pending_debounce (e.g. stop() cancels the task)
        service._pending_debounce.clear()

        await _wait_for_tasks(service)

        assert path not in service.pending_files, (
            "pending_files should be cleaned up when debounce entry is removed externally"
        )
        assert service._pending_mutations == {}
        assert service._pending_path_counts == {}

    async def test_scan_bypasses_debounce(self, service):
        """scan-priority events skip debouncing and land in file_queue immediately."""
        path = Path("/tmp/scan.py")
        await service.add_file(path, priority="scan")

        assert not service.file_queue.empty(), "scan event should bypass debounce"
        assert len(service._pending_debounce) == 0, "scan should not create debounce entry"

    async def test_distinct_files_independent_debounce(self, service):
        """Each file gets its own debounce task; two files produce two queue entries."""
        path_a = Path("/tmp/a.py")
        path_b = Path("/tmp/b.py")
        await service.add_file(path_a, priority="change")
        await service.add_file(path_b, priority="change")

        await _wait_for_tasks(service)
        items = await _drain_queue(service.file_queue)
        paths = {item[2].path for item in items}
        assert paths == {
            _expected_mutation_path(path_a),
            _expected_mutation_path(path_b),
        }

    async def test_scan_then_change_produces_two_queue_entries(self, service):
        """scan followed by change for the same file queues the file twice.

        scan goes straight to file_queue; the subsequent debounced change stays
        separately tracked and queues as a follow-up. Processing is idempotent,
        so the double-queue is safe.
        """
        path = Path("/tmp/scan_change.py")
        await service.add_file(path, priority="scan")
        await service.add_file(path, priority="change")

        await _wait_for_tasks(service)
        items = await _drain_queue(service.file_queue)
        assert len(items) == 2
        operations = [item[2].operation for item in items]
        assert operations == ["scan", "change"]

    async def test_debounce_multiple_refreshes(self, service):
        """While-loop iterates multiple times when timestamp is refreshed mid-sleep.

        Two refreshes are injected during the debounce window, forcing the loop
        to sleep at least 3 times before flushing. The file must not appear in
        the queue until after the final quiet period, and exactly once.
        """
        path = Path("/tmp/multi_refresh.py")
        delay = service._debounce_delay

        # Iteration 1 begins
        await service.add_file(path, priority="change")
        await asyncio.sleep(delay * 0.4)

        # Refresh 1 — resets the timestamp; iteration 1 will re-loop
        await service.add_file(path, priority="change")
        await asyncio.sleep(delay * 0.4)
        assert service.file_queue.empty(), "File queued too early after first refresh"

        await asyncio.sleep(delay * 0.4)

        # Refresh 2 — resets the timestamp again; loop must iterate a third time
        await service.add_file(path, priority="change")
        await asyncio.sleep(delay * 0.3)
        assert service.file_queue.empty(), "File queued too early after second refresh"

        # Let the debounce settle; exactly 1 entry expected
        await _wait_for_tasks(service)
        items = await _drain_queue(service.file_queue)
        assert len(items) == 1
        assert items[0][2].path == _expected_mutation_path(path)
