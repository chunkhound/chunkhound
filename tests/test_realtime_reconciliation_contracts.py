"""Boundary-level contracts for first-snapshot reconcile retry capping (f2)."""

from __future__ import annotations

import asyncio
import time

import pytest

from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from tests.helpers.realtime_harness import realtime_harness
from tests.helpers.search_contract import results_are_empty, wait_for_search

INDEXING_CAPPED = {
    "reconcile_max_attempts": 3,
    "reconcile_backoff_base_seconds": 0.0,
}


async def _wait_for_health_warning(service, needle, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if needle in ((await service.get_health()).get("last_warning") or ""):
            return True
        await asyncio.sleep(0.2)
    return False


async def _assert_no_giving_up_warning(service, duration=3.0):
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        warning = (await service.get_health()).get("last_warning") or ""
        assert "giving up" not in warning, warning
        await asyncio.sleep(0.2)


async def _wait_for_file_row(provider, path, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if provider.get_file_by_path(str(path)) is not None:
            return True
        await asyncio.sleep(0.2)
    return False


@pytest.mark.asyncio
async def test_permanent_reconcile_failure_warns_once_and_stops_retrying(
    tmp_path, monkeypatch
):
    calls: list[int] = []

    def failing_list(self, directory_prefix: str) -> list[str]:
        calls.append(1)
        raise RuntimeError("enumerate failed")

    monkeypatch.setattr(DuckDBProvider, "list_file_paths_under_directory", failing_list)

    async with realtime_harness(
        tmp_path, realtime_backend="polling", indexing=INDEXING_CAPPED
    ) as harness:
        service = harness.realtime_service
        assert await _wait_for_health_warning(service, "giving up")
        assert len(calls) == 3
        await asyncio.sleep(1.2)  # several poll cycles
        assert len(calls) == 3  # attempts stopped at the cap
        assert "giving up" in ((await service.get_health())["last_warning"] or "")
        new_file = harness.watch_dir / "still_works.py"
        new_file.write_text("def still_works(): pass")
        assert await _wait_for_file_row(harness.services.provider, new_file)


@pytest.mark.asyncio
async def test_transient_failure_recovers_and_resumes_reconciliation(
    tmp_path, monkeypatch
):
    calls: list[int] = []
    original = DuckDBProvider.list_file_paths_under_directory

    def transient_list(self, directory_prefix: str) -> list[str]:
        calls.append(1)
        if len(calls) <= 2:
            raise RuntimeError("enumerate failed")
        return original(self, directory_prefix)

    monkeypatch.setattr(
        DuckDBProvider, "list_file_paths_under_directory", transient_list
    )

    async with realtime_harness(
        tmp_path, realtime_backend="polling", indexing=INDEXING_CAPPED
    ) as harness:
        service = harness.realtime_service
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            assert "giving up" not in (
                (await service.get_health()).get("last_warning") or ""
            )
            if len(calls) >= 3:
                break
            await asyncio.sleep(0.2)
        else:
            raise AssertionError(
                "reconciliation did not resume after transient failures"
            )
        await _assert_no_giving_up_warning(service, duration=2.0)
        new_file = harness.watch_dir / "recovers.py"
        new_file.write_text("def recovers(): pass")
        assert await _wait_for_file_row(harness.services.provider, new_file)


@pytest.mark.asyncio
async def test_missing_watch_root_is_transient_and_not_counted(tmp_path):
    async with realtime_harness(
        tmp_path, realtime_backend="polling", indexing=INDEXING_CAPPED
    ) as harness:
        watch_dir = harness.watch_dir
        moved = watch_dir.parent / "project_moved"
        watch_dir.rename(moved)
        try:
            await asyncio.sleep(1.2)  # span several poll cycles with root absent
            await _assert_no_giving_up_warning(harness.realtime_service, duration=1.0)
        finally:
            moved.rename(watch_dir)
        await _assert_no_giving_up_warning(harness.realtime_service, duration=3.0)
        new_file = watch_dir / "after_restore.py"
        new_file.write_text("def after_restore(): pass")
        assert await _wait_for_file_row(harness.services.provider, new_file)


@pytest.mark.asyncio
async def test_reconciliation_root_rename_after_enumeration_queues_no_deletes(
    tmp_path, monkeypatch
):
    """A root rename after enumeration must not admit reconciliation deletes."""
    marker = "gone_marker"
    async with realtime_harness(tmp_path, realtime_backend="polling") as harness:
        gone = harness.watch_dir / "gone.py"
        gone.write_text(f"def {marker}():\n    return 1\n")
        await wait_for_search(harness.services, marker, lambda r: marker in r)
    gone.unlink()

    original_list = DuckDBProvider.list_file_paths_under_directory
    moved_to = tmp_path / "project_renamed_mid_reconcile"
    renamed_once = {"done": False}

    def rename_after_rows(self, directory_prefix: str) -> list[str]:
        rows = original_list(self, directory_prefix)
        if not renamed_once["done"]:
            renamed_once["done"] = True
            harness.watch_dir.rename(moved_to)
        return rows

    monkeypatch.setattr(DuckDBProvider, "list_file_paths_under_directory", rename_after_rows)
    async with realtime_harness(tmp_path, realtime_backend="polling") as harness:
        await wait_for_search(harness.services, marker, lambda r: marker in r)
        moved_to.rename(harness.watch_dir)
        await wait_for_search(harness.services, marker, results_are_empty)
