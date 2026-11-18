"""Realtime search integration tests against the public search contract.

These tests verify user-visible behavior: after a filesystem change, the MCP
search tool must eventually return the current file content. They do not inspect
realtime queues, debounce state, provider tables, or chunk counts.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import threading
from pathlib import Path
from typing import Any

import pytest

from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from tests.helpers.embedding_config import get_embedding_config_for_tests
from tests.helpers.realtime_harness import RealtimeHarness, realtime_harness
from tests.helpers.search_contract import (
    results_are_empty,
    results_contain,
    search_tool,
    wait_for_search,
)


@pytest.fixture
async def realtime_setup(tmp_path: Path, request: pytest.FixtureRequest):
    """Start real services; live tests use a real embedding provider."""
    embedding_config = (
        get_embedding_config_for_tests()
        if request.node.get_closest_marker("live")
        else None
    )
    async with realtime_harness(tmp_path, embedding_config=embedding_config) as harness:
        yield harness


async def _expect_searchable(harness: RealtimeHarness, marker: str) -> str:
    return await wait_for_search(
        harness.services, marker, lambda current: results_contain(current, marker)
    )


@pytest.mark.live
@pytest.mark.asyncio
async def test_semantic_search_returns_new_file(realtime_setup: RealtimeHarness):
    """Semantic search finds content from a newly created file."""
    new_file = realtime_setup.watch_dir / "mcp_test.py"
    new_file.write_text(
        """
def unique_mcp_test_function():
    '''This is a unique function for MCP integration testing'''
    return "mcp_realtime_success"
"""
    )

    results = await wait_for_search(
        realtime_setup.services,
        "unique_mcp_test_function",
        lambda current: results_contain(current, "mcp_realtime_success"),
        search_type="semantic",
        embedding_manager=realtime_setup.embedding_manager,
    )
    assert results_contain(results, "mcp_realtime_success")


@pytest.mark.asyncio
async def test_regex_search_reflects_file_modification(
    realtime_setup: RealtimeHarness,
):
    """Regex search returns the current content and drops replaced content."""
    test_file = realtime_setup.watch_dir / "modify_test.py"
    test_file.write_text('def initial_function():\n    return "version_1"\n')

    await _expect_searchable(realtime_setup, "version_1")

    test_file.write_text(
        """
def initial_function():
    return "version_2"

def modified_unique_regex_pattern():
    '''Added by modification - should be found by regex'''
    return "modification_success"
"""
    )

    results = await _expect_searchable(realtime_setup, "modified_unique_regex_pattern")
    assert results_contain(results, "modified_unique_regex_pattern")

    await wait_for_search(realtime_setup.services, "version_1", results_are_empty)


@pytest.mark.asyncio
async def test_regex_search_stops_returning_deleted_file(
    realtime_setup: RealtimeHarness,
):
    """Regex search drops content after the source file is deleted."""
    delete_file = realtime_setup.watch_dir / "delete_test.py"
    delete_file.write_text(
        """
def delete_test_unique_function():
    '''This function will be deleted'''
    return "to_be_deleted"
"""
    )

    await _expect_searchable(realtime_setup, "delete_test_unique_function")

    delete_file.unlink()

    await wait_for_search(
        realtime_setup.services,
        "delete_test_unique_function",
        results_are_empty,
    )


@pytest.mark.asyncio
async def test_regex_search_stops_returning_deleted_watch_root(tmp_path: Path):
    """Deleting the watched root drops every indexed row under it."""
    async with realtime_harness(tmp_path, watch_dir_is_index_root=True) as harness:
        files = {
            "nested_root_deletion_marker": harness.watch_dir / "pkg" / "nested.py",
            "root_level_deletion_marker": harness.watch_dir / "root.py",
        }
        for marker, path in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"def {marker}():\n    return 1\n")
            await _expect_searchable(harness, marker)

        shutil.rmtree(harness.watch_dir)

        for marker in files:
            await wait_for_search(harness.services, marker, results_are_empty)


@pytest.mark.asyncio
async def test_provider_stats_track_realtime_writes_and_deletes(
    realtime_setup: RealtimeHarness,
):
    """Provider files/chunks counts rise with realtime writes and fall on delete."""
    baseline = realtime_setup.watch_dir / "baseline.py"
    baseline.write_text("def baseline_stats_marker():\n    return 1\n")
    await _expect_searchable(realtime_setup, "baseline_stats_marker")

    provider = realtime_setup.services.provider
    initial = provider.get_stats()
    initial_files = initial["files"]
    initial_chunks = initial["chunks"]

    added = realtime_setup.watch_dir / "stats_added.py"
    added.write_text("def stats_added_marker():\n    return 1\n")
    await _expect_searchable(realtime_setup, "stats_added_marker")
    grown = provider.get_stats()
    assert grown["files"] > initial_files
    assert grown["chunks"] > initial_chunks

    added.unlink()
    await wait_for_search(realtime_setup.services, "stats_added_marker", results_are_empty)
    shrunk = provider.get_stats()
    assert shrunk["files"] == initial_files
    assert shrunk["chunks"] == initial_chunks


@pytest.mark.asyncio
async def test_polling_reconciliation_skips_missing_root(tmp_path: Path) -> None:
    async with realtime_harness(tmp_path, realtime_backend="polling") as harness:
        (harness.watch_dir / "missing.py").write_text(
            "def vanished_root_marker():\n    return 1\n"
        )
        retained = harness.watch_dir / "retained.py"
        retained.write_text("def retained_root_marker():\n    return 1\n")
        await _expect_searchable(harness, "vanished_root_marker")
        await _expect_searchable(harness, "retained_root_marker")

    # The persisted index must survive a restart whose watch root is absent.
    project = tmp_path / "project"
    moved = tmp_path / "project-moved"
    project.rename(moved)
    try:
        async with realtime_harness(
            tmp_path, realtime_backend="polling", create_watch_dir=False
        ) as harness:
            await _expect_searchable(harness, "retained_root_marker")
            removed = await harness.realtime_service.wait_for_file_removed(
                retained, timeout=1.0
            )
            assert removed is False
    finally:
        moved.rename(project)


@pytest.mark.asyncio
async def test_polling_reconciliation_skips_recreated_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale absence snapshot must not delete a file that exists again."""
    async with realtime_harness(tmp_path, realtime_backend="polling") as harness:
        recreated = harness.watch_dir / "recreated.py"
        recreated.write_text("def recreated_marker():\n    return 1\n")
        await _expect_searchable(harness, "recreated_marker")

    target = recreated.resolve()
    target_abs = os.path.realpath(os.fspath(target))
    pending_stale_absence = {"value": True}
    stale_snapshot_seen = asyncio.Event()
    loop = asyncio.get_running_loop()
    lock = threading.Lock()
    real_stat = Path.stat

    def one_shot_stat(path: Path) -> os.stat_result:
        if os.path.realpath(os.fspath(path)) != target_abs:
            return real_stat(path)
        with lock:
            was_pending = pending_stale_absence["value"]
            pending_stale_absence["value"] = False
        if not was_pending:
            return real_stat(path)
        # The first stat mimics a snapshot taken while the file was gone; the
        # reconciliation re-check must then observe the real path.
        loop.call_soon_threadsafe(stale_snapshot_seen.set)
        raise FileNotFoundError("stale absence snapshot")

    # Reconciliation probes absence through ``Path.stat`` (see _path_absence),
    # not ``Path.exists``, so a stale absence must be injected at the stat
    # boundary to exercise the real snapshot-then-recheck race.
    monkeypatch.setattr(Path, "stat", one_shot_stat)
    async with realtime_harness(tmp_path, realtime_backend="polling") as harness:
        await asyncio.wait_for(stale_snapshot_seen.wait(), timeout=10.0)
        removed = await harness.realtime_service.wait_for_file_removed(
            target, timeout=1.0
        )
        assert removed is False
        await _expect_searchable(harness, "recreated_marker")


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="POSIX mode bits do not deny root, and Windows has no chmod",
)
async def test_polling_reconciliation_tolerates_unreadable_directory(
    tmp_path: Path,
) -> None:
    """An unreadable directory must not keep deleted rows searchable forever."""
    async with realtime_harness(tmp_path, realtime_backend="polling") as harness:
        (harness.watch_dir / "doomed.py").write_text(
            "def doomed_marker():\n    return 1\n"
        )
        (blocked := harness.watch_dir / "blocked").mkdir()
        (blocked / "kept.py").write_text("def kept_marker():\n    return 1\n")
        await _expect_searchable(harness, "doomed_marker")
        await _expect_searchable(harness, "kept_marker")
    (harness.watch_dir / "doomed.py").unlink()
    blocked.chmod(0o000)
    try:
        async with realtime_harness(tmp_path, realtime_backend="polling") as harness:
            await wait_for_search(harness.services, "doomed_marker", results_are_empty)
            # The inaccessible subtree must stay indexed, not be treated as absent.
            await _expect_searchable(harness, "kept_marker")
            await asyncio.sleep(0.5)
            await _expect_searchable(harness, "kept_marker")
            health = await harness.realtime_service.get_health()
            assert health["live_indexing_state"] != "degraded", health
            assert health["last_warning"] is None, health
    finally:
        blocked.chmod(0o755)


@pytest.mark.asyncio
async def test_first_poll_snapshot_removes_missed_deletion(tmp_path: Path) -> None:
    async with realtime_harness(tmp_path, realtime_backend="polling") as harness:
        missing = harness.watch_dir / "poll_missing.py"
        retained = harness.watch_dir / "poll_retained.py"
        missing.write_text("def poll_missing_marker():\n    return 1\n")
        retained.write_text("def poll_retained_marker():\n    return 1\n")
        await _expect_searchable(harness, "poll_missing_marker")
        await _expect_searchable(harness, "poll_retained_marker")
    missing.unlink()
    async with realtime_harness(tmp_path, realtime_backend="polling") as harness:
        await wait_for_search(
            harness.services, "poll_missing_marker", results_are_empty
        )
        await _expect_searchable(harness, "poll_retained_marker")


@pytest.mark.asyncio
async def test_polling_reconciliation_without_table_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provider that only offers chunk search still drops vanished files."""
    for method in ("get_scope_file_paths", "list_file_paths_under_directory"):
        monkeypatch.delattr(DuckDBProvider, method, raising=False)

    async with realtime_harness(tmp_path, realtime_backend="polling") as harness:
        vanished = harness.watch_dir / "fallback_vanished.py"
        vanished.write_text("def fallback_vanished_marker():\n    return 1\n")
        await _expect_searchable(harness, "fallback_vanished_marker")

    vanished.unlink()
    async with realtime_harness(tmp_path, realtime_backend="polling") as harness:
        await wait_for_search(
            harness.services, "fallback_vanished_marker", results_are_empty
        )
        health = await harness.realtime_service.get_health()
        assert health["live_indexing_state"] != "degraded", health


def _write_with_mode(path: Path, content: str, mode: str) -> None:
    if mode == "write_text":
        path.write_text(content)
        return

    if mode == "open_fsync":
        with open(path, "w") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        return

    if mode == "atomic_replace":
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(content)
        os.replace(temporary, path)
        return

    raise ValueError(f"Unsupported write mode: {mode}")


@pytest.mark.parametrize("write_mode", ["write_text", "open_fsync", "atomic_replace"])
@pytest.mark.asyncio
async def test_regex_search_tracks_common_write_patterns(
    realtime_setup: RealtimeHarness,
    write_mode: str,
):
    """Search sees content written by common user-facing filesystem patterns."""
    marker = f"write_mode_{write_mode}"
    test_file = realtime_setup.watch_dir / f"{write_mode}_test.py"
    _write_with_mode(
        test_file,
        f'def {marker}():\n    return "{marker}"\n',
        write_mode,
    )

    results = await _expect_searchable(realtime_setup, marker)
    assert results_contain(results, marker)


@pytest.mark.asyncio
async def test_regex_search_after_atomic_modification(
    realtime_setup: RealtimeHarness,
):
    """A replaced file is searchable with its new content, not its old content."""
    test_file = realtime_setup.watch_dir / "atomic_modify_test.py"
    test_file.write_text('def atomic_marker():\n    return "atomic_version_1"\n')

    await _expect_searchable(realtime_setup, "atomic_version_1")

    temporary = test_file.with_suffix(".py.tmp")
    temporary.write_text('def atomic_marker():\n    return "atomic_version_2"\n')
    os.replace(temporary, test_file)

    await _expect_searchable(realtime_setup, "atomic_version_2")
    await wait_for_search(
        realtime_setup.services,
        "atomic_version_1",
        results_are_empty,
    )
    assert results_are_empty(
        await search_tool(realtime_setup.services, "atomic_version_1")
    )


@pytest.mark.asyncio
async def test_polling_reconciliation_retries_until_pass_completes(
    tmp_path: Path,
) -> None:
    """Deletions during a watch-root outage are purged once the root returns."""
    async with realtime_harness(tmp_path, realtime_backend="polling") as harness:
        gone = harness.watch_dir / "gone.py"
        gone.write_text("def gone_during_outage_marker():\n    return 1\n")
        await _expect_searchable(harness, "gone_during_outage_marker")

    project = tmp_path / "project"
    moved = tmp_path / "project-moved"
    project.rename(moved)
    (moved / "gone.py").unlink()
    restored = False
    try:
        async with realtime_harness(
            tmp_path, realtime_backend="polling", create_watch_dir=False
        ) as harness:
            # The first snapshot runs while the root is absent: nothing is
            # purged yet and the snapshot must not count as complete.
            await _expect_searchable(harness, "gone_during_outage_marker")
            moved.rename(project)
            restored = True
            # After the root returns, the still-pending first snapshot
            # reconciles and drops the stale row without a restart.
            await wait_for_search(
                harness.services,
                "gone_during_outage_marker",
                results_are_empty,
            )
    finally:
        if not restored:
            moved.rename(project)


@pytest.mark.asyncio
async def test_polling_reconciliation_purges_multiple_vanished_files(
    tmp_path: Path,
) -> None:
    """One reconciliation pass clears every vanished file, not just one."""
    async with realtime_harness(tmp_path, realtime_backend="polling") as harness:
        markers = [f"purge_marker_{i}" for i in range(5)]
        files = []
        for marker in markers:
            path = harness.watch_dir / f"purge_{marker}.py"
            path.write_text(f"def {marker}():\n    return 1\n")
            files.append(path)
            await _expect_searchable(harness, marker)

    # A fresh polling service starts with an empty baseline, so the first
    # snapshot reconciliation must purge every vanished row.
    for path in files:
        path.unlink()
    async with realtime_harness(tmp_path, realtime_backend="polling") as harness:
        for marker in markers:
            await wait_for_search(harness.services, marker, results_are_empty)


@pytest.mark.asyncio
async def test_polling_reconciliation_purges_dir_replaced_by_file(tmp_path: Path):
    """An indexed directory replaced by a regular file reconciles as vanished."""
    async with realtime_harness(tmp_path, realtime_backend="polling") as harness:
        old = harness.watch_dir / "subdir" / "old.py"
        old.parent.mkdir()
        old.write_text("def dir_replaced_marker(): pass\n")
        await _expect_searchable(harness, "dir_replaced_marker")
    subdir = harness.watch_dir / "subdir"
    shutil.rmtree(subdir)
    subdir.write_text("# now a regular file\n")
    async with realtime_harness(tmp_path, realtime_backend="polling") as harness:
        await wait_for_search(harness.services, "dir_replaced_marker", results_are_empty)


@pytest.mark.asyncio
async def test_polling_reconciliation_batches_deletes_single_provider_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reconciliation deletes many vanished files in one batched provider call."""
    async with realtime_harness(tmp_path, realtime_backend="polling") as harness:
        markers = [f"batch_purge_marker_{i}" for i in range(5)]
        files = []
        for marker in markers:
            path = harness.watch_dir / f"batch_{marker}.py"
            path.write_text(f"def {marker}():\n    return 1\n")
            files.append(path)
            await _expect_searchable(harness, marker)

    # Wrap the real provider method (not a mock) so the assertion observes the
    # actual HNSW-batching boundary.
    real_delete_files_batch = DuckDBProvider.delete_files_batch_async
    calls: list[list[str]] = []

    async def recorder(self: Any, file_paths: list[str]) -> int:
        calls.append(list(file_paths))
        return await real_delete_files_batch(self, file_paths)

    monkeypatch.setattr(DuckDBProvider, "delete_files_batch_async", recorder)

    for path in files:
        path.unlink()
    async with realtime_harness(tmp_path, realtime_backend="polling") as harness:
        for marker in markers:
            await wait_for_search(harness.services, marker, results_are_empty)
        assert calls, "expected at least one batched delete provider call"
        expected = {str(path.resolve()) for path in files}
        assert len(calls) == 1, calls
        assert set(calls[0]) == expected, calls[0]
        assert await harness.realtime_service.wait_for_file_removed(
            files[0], timeout=5.0
        )


@pytest.mark.asyncio
async def test_polling_reconciliation_removed_wait_only_after_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """wait_for_file_removed must not fire while a deletion is only queued."""
    async with realtime_harness(tmp_path, realtime_backend="polling") as harness:
        gone = harness.watch_dir / "gone.py"
        gone.write_text("def removed_wait_marker(): pass\n")
        await _expect_searchable(harness, "removed_wait_marker")
    gone.unlink()

    real_delete_single = DuckDBProvider.delete_file_completely_async
    delete_observed = asyncio.Event()
    release_delete = asyncio.Event()

    async def blocking_single(self: Any, file_path: str) -> None:
        if file_path == str(gone.resolve()):
            delete_observed.set()
            await release_delete.wait()
        await real_delete_single(self, file_path)

    monkeypatch.setattr(DuckDBProvider, "delete_file_completely_async", blocking_single)
    async with realtime_harness(tmp_path, realtime_backend="polling") as harness:
        await asyncio.wait_for(delete_observed.wait(), timeout=10.0)
        assert await harness.realtime_service.wait_for_file_removed(
            gone, timeout=0.1
        ) is False
        release_delete.set()
        await wait_for_search(harness.services, "removed_wait_marker", results_are_empty)
        assert await harness.realtime_service.wait_for_file_removed(
            gone, timeout=5.0
        )
