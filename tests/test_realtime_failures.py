"""Tests that expose real failures in the real-time indexing implementation.

These tests are designed to fail and show what's actually broken.
"""

import asyncio
import shutil
import tempfile
from pathlib import Path

import pytest

from chunkhound.core.config.config import Config
from chunkhound.database_factory import create_services
from chunkhound.services.realtime_indexing_service import RealtimeIndexingService
from tests.utils.windows_compat import (
    get_fs_event_timeout,
    is_ci,
    is_windows,
    should_use_polling,
    stabilize_polling_monitor,
)


class TestRealtimeFailures:
    """Tests that expose actual implementation failures."""

    @pytest.fixture
    async def realtime_setup(self, watcher_mode):
        """Setup real service with temp database and project directory."""
        # Resolve immediately to handle Windows 8.3 short path names
        temp_dir = Path(tempfile.mkdtemp()).resolve()
        db_path = temp_dir / ".chunkhound" / "test.db"
        watch_dir = temp_dir / "project"
        watch_dir.mkdir(parents=True)

        # Ensure database directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Use fake args to prevent find_project_root call that fails in CI
        from types import SimpleNamespace
        fake_args = SimpleNamespace(path=temp_dir)
        config = Config(
            args=fake_args,
            database={"path": str(db_path), "provider": "duckdb"},
            indexing={"include": ["*.py", "*.js"], "exclude": ["*.log"]}
        )

        services = create_services(db_path, config)
        services.provider.connect()

        # Watcher coverage is selected explicitly per test via watcher_mode.
        force_polling = should_use_polling(watcher_mode)
        realtime_service = RealtimeIndexingService(
            services, config, force_polling=force_polling
        )

        yield realtime_service, watch_dir, temp_dir, services

        # Cleanup
        try:
            await realtime_service.stop()
        except Exception:
            pass

        try:
            services.provider.disconnect()
        except Exception:
            pass

        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_indexing_coordinator_skip_embeddings_not_implemented(
        self, realtime_setup
    ):
        """Test that IndexingCoordinator.process_file doesn't support
        skip_embeddings parameter."""
        service, watch_dir, _, services = realtime_setup

        # Try to call process_file with skip_embeddings directly
        test_file = watch_dir / "skip_test.py"
        test_file.write_text("def skip_embeddings_test(): pass")

        # This should fail because process_file signature doesn't match usage
        try:
            result = await services.indexing_coordinator.process_file(
                test_file,
                skip_embeddings=True  # This parameter might not exist
            )
            # If we get here, the parameter exists but might not work correctly
            assert result.get('embeddings_skipped'), (
                "skip_embeddings parameter should actually skip embeddings"
            )
        except TypeError as e:
            pytest.fail(
                f"IndexingCoordinator.process_file doesn't support "
                f"skip_embeddings: {e}"
            )

    @pytest.mark.asyncio
    async def test_file_debouncing_creates_memory_leaks(self, realtime_setup):
        """Test that file debouncing properly cleans up timers."""
        service, watch_dir, _, _ = realtime_setup
        await service.start(watch_dir)

        # Wait for initial scan to complete
        await asyncio.sleep(1.0)

        # Create many rapid file changes to the SAME file - should reuse timer slot
        test_file = watch_dir / "reused_file.py"
        for i in range(20):
            test_file.write_text(f"def func_{i}(): pass # iteration {i}")
            await asyncio.sleep(0.01)  # Very rapid changes to same file

        # Wait for debounce delay to let timers execute and cleanup
        await asyncio.sleep(1.0)

        # Get reference to debouncer timers after cleanup should occur
        if service.event_handler and hasattr(service.event_handler, 'debouncer'):
            active_timers = len(service.event_handler.debouncer.timers)
            # Should only have 1 timer max for the single file, or 0 if cleaned up
            assert active_timers <= 1, (
                f"Too many active timers ({active_timers}) "
                "- should cleanup after execution"
            )

        await service.stop()

    @pytest.mark.asyncio
    async def test_background_scan_conflicts_with_realtime(self, realtime_setup):
        """Test that background scan and real-time processing conflict."""
        service, watch_dir, _, services = realtime_setup

        # Create file before starting (will be in initial scan)
        initial_file = watch_dir / "conflict_test.py"
        initial_file.write_text("def initial(): pass")

        await service.start(watch_dir)

        # Immediately modify the same file (real-time processing)
        initial_file.write_text("def initial_modified(): pass")

        # Wait for both to potentially process
        await asyncio.sleep(1.5)

        # Check if file was processed multiple times (race condition)
        file_record = services.provider.get_file_by_path(str(initial_file))
        if file_record:
            chunks = services.provider.get_chunks_by_file_id(file_record['id'])

            # If there are duplicate chunks or processing conflicts, this will show
            chunk_contents = [chunk.get('content', '') for chunk in chunks]
            unique_contents = set(chunk_contents)

            assert len(chunk_contents) == len(unique_contents), (
                f"Duplicate processing detected: {len(chunk_contents)} "
                f"chunks, {len(unique_contents)} unique"
            )

        await service.stop()

    @pytest.mark.xfail(
        condition=is_windows() and is_ci(),
        reason="Polling mtime detection unreliable on NTFS (fixed in PR #220)",
        strict=False,
    )
    @pytest.mark.polling_watcher
    @pytest.mark.asyncio
    async def test_observer_not_properly_recursive(self, realtime_setup):
        """Test that filesystem observer doesn't properly watch subdirectories."""
        service, watch_dir, _, services = realtime_setup
        await service.start(watch_dir)

        await stabilize_polling_monitor("polling")

        # Create subdirectory and file
        subdir = watch_dir / "subdir"
        subdir.mkdir()
        subdir_file = subdir / "nested.py"
        # Use the CI-aware timeout budget for explicit polling coverage.
        service.reset_file_tracking(subdir_file)
        subdir_file.write_text("def nested(): pass")
        found = await service.wait_for_file_indexed(subdir_file, timeout=get_fs_event_timeout())
        assert found, "Nested files should be detected by recursive monitoring"

        await service.stop()

    @pytest.mark.polling_watcher
    @pytest.mark.asyncio
    async def test_polling_detects_new_files(self, realtime_setup):
        """Polling fallback should index newly created files."""
        service, watch_dir, _, _ = realtime_setup
        await service.start(watch_dir)

        await stabilize_polling_monitor("polling")

        test_file = watch_dir / "polling_create.py"
        service.reset_file_tracking(test_file)
        test_file.write_text("def created_by_polling(): pass")
        found = await service.wait_for_file_indexed(
            test_file, timeout=get_fs_event_timeout()
        )
        assert found, "Polling fallback should detect newly created files"

        await service.stop()

    @pytest.mark.polling_watcher
    @pytest.mark.asyncio
    async def test_service_doesnt_handle_file_deletions(self, realtime_setup):
        """Test that polling mode handles file deletions after its first scan."""
        service, watch_dir, _, services = realtime_setup
        await service.start(watch_dir)
        await stabilize_polling_monitor("polling")

        # Create and process a file
        test_file = watch_dir / "delete_test.py"
        # Wait for file to be indexed
        service.reset_file_tracking(test_file)
        test_file.write_text("def to_be_deleted(): pass")
        found = await service.wait_for_file_indexed(test_file, timeout=get_fs_event_timeout())
        assert found, "File should be processed initially"

        # Delete the file
        # Wait for deletion processing
        service.reset_file_tracking(test_file)
        test_file.unlink()
        removed = await service.wait_for_file_removed(test_file, timeout=get_fs_event_timeout())
        assert removed, "Deleted files should be removed from database"

        await service.stop()

    @pytest.mark.asyncio
    async def test_error_in_processing_loop_kills_service(self, realtime_setup):
        """Test that an error in the processing loop kills the entire service."""
        service, watch_dir, _, _ = realtime_setup
        await service.start(watch_dir)

        # Force an error by creating a file and then deleting it before processing
        test_file = watch_dir / "error_test.py"
        test_file.write_text("def error_test(): pass")

        # Wait just enough for file to be queued but not processed
        await asyncio.sleep(0.3)

        # Delete file while it's queued for processing
        test_file.unlink()

        # Wait for processing to attempt and fail
        await asyncio.sleep(1.0)

        # Check if service is still alive after error
        stats = await service.get_stats()
        assert stats.get('observer_alive', False), (
            "Service should survive processing errors"
        )

        await service.stop()

    @pytest.mark.asyncio
    async def test_polling_monitor_cleanup_on_cancellation(self, realtime_setup):
        """Test that polling monitor cleans up resources when cancelled."""
        service, watch_dir, _, _ = realtime_setup

        # Force polling mode for deterministic testing
        service._force_polling = True
        await service.start(watch_dir)

        # Let polling run at least one cycle
        await asyncio.sleep(0.5)

        # Stop service (triggers cancellation of polling task)
        await service.stop()

        # Verify cleanup completed - task should be done or None
        assert service._polling_task is None or service._polling_task.done(), \
            "Polling task should be cleaned up after stop()"

    @pytest.mark.asyncio
    async def test_compaction_error_defers_file_without_counting_as_failure(
        self, realtime_setup
    ):
        """CompactionError defers retry state without inflating failed_files stats."""
        service, watch_dir, _, services = realtime_setup
        await service.start(watch_dir)

        # Wait for initial scan to finish
        assert await service.wait_for_monitoring_ready(timeout=10.0)

        test_file = watch_dir / "compaction_deferred.py"
        test_file.write_text("def deferred(): pass")

        # Simulate compaction: close gate AND drop thread-local connection
        # so next DB access raises CompactionError instead of reusing existing conn.
        services.provider._connection_allowed.clear()
        services.provider.soft_disconnect(skip_checkpoint=True)

        # Put file directly in the queue — bypasses debouncing
        await service.file_queue.put(("change", test_file))

        # wait_for_file_indexed checks both terminal failure buckets,
        # so it returns as soon as the deferred state is recorded.
        await service.wait_for_file_indexed(test_file, timeout=5.0)

        from chunkhound.services.realtime_indexing_service import normalize_file_path

        normalized = normalize_file_path(test_file)
        assert normalized not in service.failed_files, (
            "Compaction deferrals must not count as genuine failed_files"
        )
        stats = await service.get_stats()
        assert stats["failed_files"] == 0

        # Restore gate and reconnect for clean teardown
        services.provider._connection_allowed.set()
        services.provider.connect()
        await service.stop()

    @pytest.mark.asyncio
    async def test_successful_retry_clears_deferred_state(self, realtime_setup):
        """A successful retry after a CompactionError must clear deferred state.

        Regression guard for the stale deferral leak after compaction resumes.
        """
        from chunkhound.services.realtime_indexing_service import normalize_file_path

        service, watch_dir, _, services = realtime_setup
        await service.start(watch_dir)
        assert await service.wait_for_monitoring_ready(timeout=10.0)

        test_file = watch_dir / "retry_clears.py"
        test_file.write_text("def deferred(): pass")
        normalized = normalize_file_path(test_file)

        # 1. Force CompactionError — file lands in deferred tracking only.
        services.provider._connection_allowed.clear()
        services.provider.soft_disconnect(skip_checkpoint=True)
        await service.file_queue.put(("change", test_file))
        await service.wait_for_file_indexed(test_file, timeout=5.0)
        assert normalized not in service.failed_files

        # 2. Reopen the gate and reconnect so the retry can succeed.
        services.provider._connection_allowed.set()
        services.provider.connect()

        # 3. Re-queue the same file. Wait specifically for the success-path
        # update on _indexed_files — wait_for_file_indexed would return
        # immediately because failed_files still contains the entry.
        await service.file_queue.put(("change", test_file))

        async def _wait_until_indexed() -> None:
            async with service._file_condition:
                await service._file_condition.wait_for(
                    lambda: normalized in service._indexed_files
                )

        await asyncio.wait_for(_wait_until_indexed(), timeout=10.0)

        # 4. Success path must have cleared all stale failure state.
        assert normalized not in service.failed_files, (
            "Successful retry must discard entry from failed_files"
        )
        assert normalized not in service._compaction_deferred_files, (
            "Successful retry must discard compaction-deferred state"
        )
        stats = await service.get_stats()
        assert stats["failed_files"] == 0, (
            f"get_stats()['failed_files'] must be 0 after retry, got {stats['failed_files']}"
        )

        await service.stop()

    @pytest.mark.asyncio
    async def test_remove_file_clears_deferred_state(self, realtime_setup):
        """remove_file() must also clear stale deferred state for the target path.

        Symmetric regression guard: a previously deferred file that is later
        deleted must not linger in either retry bucket.
        """
        from chunkhound.services.realtime_indexing_service import normalize_file_path

        service, watch_dir, _, services = realtime_setup
        await service.start(watch_dir)
        assert await service.wait_for_monitoring_ready(timeout=10.0)

        test_file = watch_dir / "remove_clears.py"
        test_file.write_text("def deferred(): pass")
        normalized = normalize_file_path(test_file)

        # 1. Force CompactionError — file lands in deferred tracking only.
        services.provider._connection_allowed.clear()
        services.provider.soft_disconnect(skip_checkpoint=True)
        await service.file_queue.put(("change", test_file))
        await service.wait_for_file_indexed(test_file, timeout=5.0)
        assert normalized not in service.failed_files

        # 2. Reopen gate so remove_file()'s provider call can succeed.
        services.provider._connection_allowed.set()
        services.provider.connect()

        # 3. Delete on disk and call remove_file() directly. This exercises
        # the method-level contract without depending on the filesystem
        # event pipeline.
        test_file.unlink()
        await service.remove_file(test_file)

        # 4. remove_file() must discard the entry from both tracking buckets.
        assert normalized not in service.failed_files, (
            "remove_file() must discard entry from failed_files"
        )
        assert normalized not in service._compaction_deferred_files, (
            "remove_file() must discard compaction-deferred state"
        )
        assert normalized in service._removed_files
        stats = await service.get_stats()
        assert stats["failed_files"] == 0, (
            f"get_stats()['failed_files'] must be 0 after remove, got {stats['failed_files']}"
        )

        await service.stop()

    @pytest.mark.asyncio
    async def test_remove_file_defers_compaction_without_counting_as_failure(
        self, realtime_setup
    ):
        """remove_file() must treat CompactionError as deferred retry state."""
        service, watch_dir, _, services = realtime_setup
        await service.start(watch_dir)
        assert await service.wait_for_monitoring_ready(timeout=10.0)

        test_file = watch_dir / "remove_deferred.py"
        test_file.write_text("def deferred_remove(): pass")

        services.provider._connection_allowed.clear()
        services.provider.soft_disconnect(skip_checkpoint=True)

        test_file.unlink()
        await service.remove_file(test_file)
        removed = await service.wait_for_file_removed(test_file, timeout=1.0)
        assert removed is False, "Deferred removals should not report as completed"

        stats = await service.get_stats()
        assert stats["failed_files"] == 0, (
            "Compaction-deferred removals must not count as genuine failed_files"
        )

        services.provider._connection_allowed.set()
        services.provider.connect()
        await service.stop()

    @pytest.mark.asyncio
    async def test_clear_compaction_deferred_files_preserves_genuine_failures(
        self, realtime_setup
    ):
        """Selective cleanup must not hide unrelated genuine failures."""
        from chunkhound.services.realtime_indexing_service import normalize_file_path

        service, watch_dir, _, _ = realtime_setup
        await service.start(watch_dir)
        assert await service.wait_for_monitoring_ready(timeout=10.0)

        failed = normalize_file_path(watch_dir / "real_failure.py")
        deferred = normalize_file_path(watch_dir / "compaction_deferred.py")

        async with service._file_condition:
            service.failed_files.add(failed)
            service._compaction_deferred_files.add(deferred)

        await service.clear_compaction_deferred_files()

        assert failed in service.failed_files
        assert deferred not in service._compaction_deferred_files
        stats = await service.get_stats()
        assert stats["failed_files"] == 1

        await service.stop()

    @pytest.mark.asyncio
    async def test_nested_file_gets_indexed(self, realtime_setup):
        """Test that process_file indexes files in subdirectories.

        Deterministic test that calls process_file() directly, bypassing
        filesystem monitoring. Tests the reaction layer independently of
        platform-dependent change detection.
        """
        _, watch_dir, _, services = realtime_setup

        # Create subdirectory and file
        subdir = watch_dir / "subdir"
        subdir.mkdir()
        nested_file = subdir / "nested.py"
        nested_file.write_text("def nested(): pass")

        # Process directly — no watcher involved
        await services.indexing_coordinator.process_file(nested_file)

        record = services.provider.get_file_by_path(str(nested_file.resolve()))
        assert record is not None, "Nested file should be indexed by process_file"

    @pytest.mark.native_watcher
    @pytest.mark.asyncio
    async def test_stop_during_fs_monitor_setup_does_not_crash_or_leak(
        self, realtime_setup, monkeypatch
    ):
        """stop() racing with _start_fs_monitor must not crash or leak a live observer.

        Freezes the executor thread between `self.observer = Observer()` and
        `observer.start()` by gating `observer.schedule()`, then drives stop().
        Without the hoisted _fs_monitor_done clear, stop() would either crash
        with "cannot join thread before it is started" or leave a live observer
        running after return.
        """
        import threading as _threading

        from chunkhound.services import realtime_indexing_service as rt

        service, watch_dir, _, _ = realtime_setup
        # This is a native-watcher lifecycle test; keep the backend explicit.
        service._force_polling = False

        schedule_gate = _threading.Event()
        real_observer_cls = rt.Observer

        class _GatedObserver(real_observer_cls):  # type: ignore[misc, valid-type]
            def schedule(self, *args, **kwargs):
                # Block until the test lets us proceed, pinning the executor
                # thread in the exact mid-setup state stop() must survive.
                schedule_gate.wait(timeout=30.0)
                return super().schedule(*args, **kwargs)

        monkeypatch.setattr(rt, "Observer", _GatedObserver)

        start_task = asyncio.create_task(service.start(watch_dir))

        # Wait for the executor thread to reach the gate (self.observer assigned
        # but observer.start() not yet called).
        for _ in range(200):
            if service.observer is not None:
                break
            await asyncio.sleep(0.01)
        assert service.observer is not None, (
            "executor thread never reached Observer construction"
        )

        # stop() must not raise and must not strand a live observer.
        stop_task = asyncio.create_task(service.stop())
        # Release the gate so _start_fs_monitor can complete and set the event.
        schedule_gate.set()
        await asyncio.wait_for(stop_task, timeout=10.0)

        try:
            await asyncio.wait_for(start_task, timeout=5.0)
        except (asyncio.CancelledError, Exception):
            pass

        assert service.observer is None or not service.observer.is_alive(), (
            "stop() returned while a live filesystem observer was still running"
        )
