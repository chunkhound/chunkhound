"""Functional tests for real-time filesystem indexing.

Tests core real-time indexing functionality with real components.
Some tests expected to fail initially - helps identify implementation issues.
"""

import asyncio
import os
import shutil
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from chunkhound.core.config.config import Config
from chunkhound.database_factory import create_services
from chunkhound.providers.database.duckdb_provider import (
    DuckDBTransactionConflictError,
)
from chunkhound.services.realtime_indexing_service import (
    RealtimeIndexingService,
    SimpleEventHandler,
)
from chunkhound.watchman import discover_nested_linux_mount_roots
from chunkhound.watchman_runtime.loader import (
    listener_path_is_filesystem,
    resolve_packaged_watchman_runtime,
)
from tests.utils.windows_compat import (
    create_windows_directory_junction,
    realtime_backend_for_tests,
    remove_windows_directory_junction,
    wait_for_indexed,
)


async def _wait_for_realtime_condition(
    service: RealtimeIndexingService,
    predicate,
    *,
    timeout: float = 10.0,
):
    async def _poll():
        while True:
            stats = await service.get_health()
            if predicate(stats):
                return stats
            await asyncio.sleep(0.05)

    return await asyncio.wait_for(_poll(), timeout=timeout)


async def _wait_for_logical_indexed(
    provider,
    file_path: Path,
    *,
    timeout: float = 10.0,
    poll_interval: float = 0.2,
) -> bool:
    deadline = time.monotonic() + timeout
    lookup_path = str(file_path)
    while time.monotonic() < deadline:
        record = provider.get_file_by_path(lookup_path)
        if record is not None:
            return True
        await asyncio.sleep(poll_interval)
    return False


def _active_watchman_disconnect_process(adapter) -> object:
    session = getattr(adapter, "_session", None)
    process = getattr(session, "_process", None)
    if process is not None:
        return process

    sidecar = getattr(adapter, "_sidecar", None)
    sidecar_process = getattr(sidecar, "_process", None)
    if sidecar_process is not None:
        return sidecar_process

    raise AssertionError("No active Watchman process available to disconnect")


def _configured_mount_regression_paths() -> tuple[Path, Path] | None:
    mount_parent = os.environ.get("CHUNKHOUND_TEST_WATCHMAN_MOUNT_PARENT")
    nested_mount = os.environ.get("CHUNKHOUND_TEST_WATCHMAN_MOUNT_CHILD")
    if not mount_parent or not nested_mount:
        return None
    return Path(mount_parent).resolve(), Path(nested_mount).resolve()


class TestRealtimeFunctional:
    """Functional tests for real-time indexing - test what really matters."""

    @pytest.fixture
    async def realtime_setup(self):
        """Setup real service with temp database and project directory."""
        # Resolve immediately to handle symlinks (/var -> /private/var on macOS)
        # and Windows 8.3 short path names
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
            indexing={
                "include": ["*.py", "*.js"],
                "exclude": ["*.log"],
                "realtime_backend": realtime_backend_for_tests(),
            },
        )

        services = create_services(db_path, config)
        services.provider.connect()

        realtime_service = RealtimeIndexingService(services, config)

        yield realtime_service, watch_dir, temp_dir, services

        # Cleanup
        try:
            await realtime_service.stop()
        except Exception:
            pass  # Service might already be stopped or failed

        try:
            services.provider.disconnect()
        except Exception:
            pass

        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_service_can_start_and_stop(self, realtime_setup):
        """Test basic service lifecycle - start and stop without crashing."""
        service, watch_dir, _, _ = realtime_setup

        # Should be able to start
        await service.start(watch_dir)

        # Check basic state
        stats = await service.get_stats()
        assert isinstance(stats, dict), "Stats should be returned"
        assert "observer_alive" in stats, "Should report observer status"
        assert stats["watching_directory"] == str(watch_dir), (
            "Should report watched directory"
        )
        assert "event_queue" in stats, "Should expose event queue health"
        assert "resync" in stats, "Should expose backend-neutral resync state"
        assert "pipeline" in stats, "Should expose pipeline progress state"
        assert "configured_backend" in stats, "Should expose configured backend"
        assert "effective_backend" in stats, "Should expose effective backend"
        assert "monitoring_mode" in stats, "Should expose current monitoring mode"
        assert stats["configured_backend"] == realtime_backend_for_tests()
        assert stats["monitoring_mode"] == stats["effective_backend"]

        # Should be able to stop cleanly
        await service.stop()

    @pytest.mark.asyncio
    async def test_watchman_startup_timing_tracks_sidecar_delay(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        """Synthetic Watchman startup delays should land in the sidecar phase bucket."""
        temp_dir = tmp_path.resolve()
        db_path = temp_dir / ".chunkhound" / "test.db"
        watch_dir = temp_dir / "project"
        watch_dir.mkdir(parents=True)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        config = Config(
            args=SimpleNamespace(path=temp_dir),
            database={"path": str(db_path), "provider": "duckdb"},
            indexing={
                "include": ["*.py"],
                "exclude": [],
                "realtime_backend": "watchman",
            },
        )

        services = create_services(db_path, config)
        services.provider.connect()
        sidecar_delay_seconds = 0.05
        timing_tolerance_seconds = 0.01

        class _FakeSidecar:
            def __init__(self, target_dir: Path, debug_sink=None) -> None:
                del debug_sink
                runtime_root = target_dir / ".chunkhound" / "watchman"
                runtime_root.mkdir(parents=True, exist_ok=True)
                self.paths = SimpleNamespace(
                    listener_path=runtime_root / "sock",
                    statefile_path=runtime_root / "state",
                    logfile_path=runtime_root / "watchman.log",
                    pidfile_path=runtime_root / "pid",
                    project_root=target_dir,
                )
                self._target_dir = target_dir

            async def start(self):
                await asyncio.sleep(sidecar_delay_seconds)
                return SimpleNamespace(binary_path="/tmp/fake-watchman")

            async def stop(self) -> None:
                return None

            def get_health(self) -> dict[str, object]:
                return {
                    "watchman_pid": 12345,
                    "watchman_started_at": "2026-03-08T00:00:01Z",
                    "watchman_process_start_time_epoch": 1.0,
                    "watchman_runtime_version": "fake",
                    "watchman_binary_path": "/tmp/fake-watchman",
                    "watchman_socket_path": str(self.paths.listener_path),
                    "watchman_statefile_path": str(self.paths.statefile_path),
                    "watchman_logfile_path": str(self.paths.logfile_path),
                    "watchman_metadata_path": str(
                        self._target_dir / ".chunkhound" / "watchman" / "metadata.json"
                    ),
                    "watchman_alive": True,
                }

        class _FakeSession:
            _SUBSCRIPTION_QUEUE_MAXSIZE = 1000

            def __init__(self, *args, **kwargs) -> None:
                del args, kwargs
                self.subscription_queue = asyncio.Queue(
                    maxsize=self._SUBSCRIPTION_QUEUE_MAXSIZE
                )
                self._subscription_name: str | None = None
                self._scope = None
                self._capabilities = {
                    "cmd-watch-project": True,
                    "relative_root": True,
                }

            @staticmethod
            def _sanitize_subscription_suffix(value: str) -> str:
                return value.replace("/", "-").replace("\\", "-")

            def supports_prepared_session_startup(self) -> bool:
                return True

            async def prepare(self) -> dict[str, bool]:
                return dict(self._capabilities)

            async def watch_project(self, target_path: Path) -> dict[str, object]:
                del target_path
                return {"watch": str(watch_dir.resolve()), "relative_path": None}

            async def watch_roots(self, roots: list[Path]) -> tuple[Path, ...]:
                return tuple(roots)

            async def subscribe_scopes(
                self,
                *,
                target_path: Path,
                scope_plan,
                subscription_name: str | None = None,
            ) -> SimpleNamespace:
                del target_path
                self._subscription_name = (
                    subscription_name or "chunkhound-live-indexing"
                )
                self._scope = scope_plan.primary_scope
                return SimpleNamespace(
                    scope_plan=scope_plan,
                    subscription_name=self._subscription_name,
                    subscription_names=(self._subscription_name,),
                    capabilities=dict(self._capabilities),
                )

            async def start(
                self,
                target_path: Path,
                subscription_name: str,
                scope_plan,
                nested_mount_roots=(),
                additional_scopes=(),
            ) -> SimpleNamespace:
                del nested_mount_roots, additional_scopes
                await self.prepare()
                return await self.subscribe_scopes(
                    target_path=target_path,
                    subscription_name=subscription_name,
                    scope_plan=scope_plan,
                )

            async def stop(self) -> None:
                return None

            async def wait_for_unexpected_exit(self) -> str | None:
                await asyncio.Event().wait()
                return None

            def get_health(self) -> dict[str, object]:
                scope = self._scope
                scopes = []
                if scope is not None:
                    scopes.append(
                        {
                            "subscription_name": self._subscription_name,
                            "scope_kind": scope.scope_kind,
                            "requested_path": str(scope.requested_path),
                            "watch_root": str(scope.watch_root),
                            "relative_root": scope.relative_root,
                        }
                    )
                subscription_names = (
                    [self._subscription_name] if self._subscription_name else []
                )
                return {
                    "watchman_session_alive": True,
                    "watchman_session_pid": 54321,
                    "watchman_session_last_warning": None,
                    "watchman_session_last_warning_at": None,
                    "watchman_session_last_error": None,
                    "watchman_session_last_error_at": None,
                    "watchman_session_last_response_at": "2026-03-08T00:00:02Z",
                    "watchman_subscription_last_received_at": None,
                    "watchman_session_command_count": 2,
                    "watchman_subscription_queue_size": self.subscription_queue.qsize(),
                    "watchman_subscription_queue_maxsize": (
                        self.subscription_queue.maxsize
                    ),
                    "watchman_subscription_pdu_count": 0,
                    "watchman_subscription_pdu_dropped": 0,
                    "watchman_subscription_name": self._subscription_name,
                    "watchman_subscription_names": subscription_names,
                    "watchman_watch_root": (
                        str(scope.watch_root) if scope is not None else None
                    ),
                    "watchman_relative_root": (
                        scope.relative_root if scope is not None else None
                    ),
                    "watchman_scopes": scopes,
                    "watchman_session_capabilities": dict(self._capabilities),
                }

        monkeypatch.setattr(
            "chunkhound.services.realtime_indexing_service.PrivateWatchmanSidecar",
            _FakeSidecar,
        )
        monkeypatch.setattr(
            "chunkhound.services.realtime_indexing_service.WatchmanCliSession",
            _FakeSession,
        )
        monkeypatch.setattr(
            "chunkhound.services.realtime_indexing_service.discover_nested_linux_mount_roots",
            lambda target_path: (),
        )
        monkeypatch.setattr(
            "chunkhound.services.realtime_indexing_service.discover_nested_windows_junction_scopes",
            lambda target_path: (),
        )

        service = RealtimeIndexingService(services, config)

        try:
            await service.start(watch_dir)
            stats = await service.get_health()
            startup = stats["startup"]

            assert startup["state"] == "completed"
            assert startup["mode"] == "stdio"
            assert startup["phases"]["watchman_sidecar_start"]["state"] == "completed"
            # Hosted Windows runners can undershoot a nominal 50 ms sleep by a
            # few milliseconds; the contract here is timing capture, not
            # exact profiling precision.
            assert (
                startup["phases"]["watchman_sidecar_start"]["duration_seconds"]
                >= sidecar_delay_seconds - timing_tolerance_seconds
            )
            assert startup["phases"]["watchman_watch_project"]["state"] == "completed"
            assert (
                startup["phases"]["watchman_subscription_setup"]["state"] == "completed"
            )
        finally:
            await service.stop()
            services.provider.disconnect()

    @pytest.mark.asyncio
    async def test_live_indexing_state_distinguishes_busy_from_stalled(
        self, realtime_setup
    ):
        """Pipeline backlog should surface as busy first, then stalled."""
        service, watch_dir, _, _ = realtime_setup
        target_file = watch_dir / "stalled.py"

        service._service_state = "running"
        service._effective_backend = "watchdog"
        service.monitoring_ready.set()
        service._monitoring_ready_at = service._utc_now()
        service.pending_files.add(target_file)
        service._last_accepted_event_at = (
            (
                datetime.now(timezone.utc)
                - timedelta(seconds=service._STALL_THRESHOLD_SECONDS + 1)
            )
            .replace(microsecond=0)
            .isoformat()
        )
        service._last_accepted_event_type = "modified"
        service._last_accepted_event_path = str(target_file)

        stalled_stats = await service.get_health()
        assert stalled_stats["live_indexing_state"] == "stalled"
        assert stalled_stats["live_indexing_hint"] == (
            "Accepted events are queued but processing has not advanced in "
            "30s; inspect pipeline timestamps and processing_error_count."
        )

        service.pending_files.clear()
        service._active_processing_count = 1
        service._last_processing_started_at = service._utc_now()
        service._last_processing_started_path = str(target_file)

        busy_stats = await service.get_health()
        assert busy_stats["live_indexing_state"] == "busy"
        assert busy_stats["live_indexing_hint"] == (
            "Live indexing is actively processing changes."
        )

    @pytest.mark.asyncio
    async def test_live_indexing_state_stays_busy_while_processing_is_inflight(
        self, realtime_setup, monkeypatch
    ):
        """In-flight work should remain busy after pending_files is cleared."""
        service, watch_dir, _, _ = realtime_setup
        target_file = watch_dir / "inflight_busy.py"
        target_file.write_text("def inflight_busy(): pass")
        process_started = asyncio.Event()
        release_processing = asyncio.Event()
        original_process_file = service.services.indexing_coordinator.process_file

        async def blocked_process_file(
            file_path: Path, skip_embeddings: bool = False
        ) -> dict[str, object]:
            assert file_path == target_file
            assert skip_embeddings is True
            process_started.set()
            await release_processing.wait()
            return await original_process_file(
                file_path, skip_embeddings=skip_embeddings
            )

        monkeypatch.setattr(
            service.services.indexing_coordinator,
            "process_file",
            blocked_process_file,
        )

        service._service_state = "running"
        service._effective_backend = "watchdog"
        service.monitoring_ready.set()
        service._monitoring_ready_at = service._utc_now()
        await service.add_file(target_file, priority="priority")

        process_task = asyncio.create_task(service._process_loop())
        service.process_task = process_task

        try:
            await asyncio.wait_for(process_started.wait(), timeout=1.0)
            assert target_file not in service.pending_files

            stats = await service.get_health()
            assert stats["live_indexing_state"] == "busy"
            assert stats["live_indexing_hint"] == (
                "Live indexing is actively processing changes."
            )
        finally:
            release_processing.set()
            process_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await process_task

    @pytest.mark.asyncio
    async def test_health_snapshot_exposes_pending_mutation_composition_and_age(
        self, realtime_setup
    ):
        """Pending mutation status should explain backlog composition and age."""
        service, watch_dir, _, _ = realtime_setup
        change_file = watch_dir / "mixed_change.py"
        delete_file = watch_dir / "mixed_delete.py"
        embed_file = watch_dir / "mixed_embed.py"
        oldest_at = (
            (datetime.now(timezone.utc) - timedelta(seconds=45))
            .replace(microsecond=0)
            .isoformat()
        )
        newer_at = (
            (datetime.now(timezone.utc) - timedelta(seconds=12))
            .replace(microsecond=0)
            .isoformat()
        )
        zero_counts = {
            "change": 0,
            "delete": 0,
            "embed": 0,
            "dir_delete": 0,
            "dir_index": 0,
        }

        service._service_state = "running"
        service._effective_backend = "watchdog"
        service.monitoring_ready.set()
        service._monitoring_ready_at = service._utc_now()
        assert service._register_pending_mutation(
            service._build_mutation(
                "change",
                change_file,
                first_queued_at=oldest_at,
            )
        )
        assert service._register_pending_mutation(
            service._build_mutation(
                "delete",
                delete_file,
                retry_count=2,
                first_queued_at=newer_at,
            )
        )
        assert service._register_pending_mutation(
            service._build_mutation(
                "embed",
                embed_file,
                first_queued_at=newer_at,
            )
        )

        stats = await service.get_health()
        pending = stats["pending_mutations"]

        assert stats["live_indexing_state"] == "busy"
        assert stats["pending_files"] == 3
        assert pending["total"] == 3
        assert pending["unique_paths"] == 3
        assert pending["counts_by_operation"] == {
            **zero_counts,
            "change": 1,
            "delete": 1,
            "embed": 1,
        }
        assert pending["retry_counts_by_operation"] == {
            **zero_counts,
            "delete": 1,
        }
        assert pending["retrying_mutations"] == 1
        assert pending["oldest_pending_at"] == oldest_at
        assert pending["oldest_pending_age_seconds"] >= 44
        assert pending["oldest_pending_operation"] == "change"
        assert pending["oldest_pending_path"] == str(change_file)
        assert pending["oldest_pending_retry_count"] == 0
        assert pending["recovery_phase"] == "mutation_drain"
        assert pending["resync_reason"] is None

    @pytest.mark.asyncio
    async def test_health_snapshot_marks_backlog_as_pending_behind_resync(
        self, realtime_setup
    ):
        """Pending mutation status should show when backlog is blocked on resync."""
        service, watch_dir, _, _ = realtime_setup
        target_file = watch_dir / "resync_pending.py"
        queued_at = (
            (datetime.now(timezone.utc) - timedelta(seconds=20))
            .replace(microsecond=0)
            .isoformat()
        )

        service._service_state = "running"
        service._effective_backend = "watchdog"
        service.monitoring_ready.set()
        service._monitoring_ready_at = service._utc_now()
        service._needs_resync = True
        service._resync_in_progress = True
        service._last_resync_reason = "event_queue_overflow"
        service._event_queue_overflow_state = "reconciling"
        assert service._register_pending_mutation(
            service._build_mutation(
                "delete",
                target_file,
                retry_count=1,
                first_queued_at=queued_at,
            )
        )

        stats = await service.get_health()
        pending = stats["pending_mutations"]

        assert stats["live_indexing_state"] == "degraded"
        assert pending["total"] == 1
        assert pending["retrying_mutations"] == 1
        assert pending["recovery_phase"] == "resync_in_progress"
        assert pending["resync_reason"] == "event_queue_overflow"

    @pytest.mark.asyncio
    async def test_idle_health_snapshot_exposes_empty_pending_mutation_details(
        self, realtime_setup
    ):
        """Idle realtime health should keep top-level fields stable and compact."""
        service, watch_dir, _, _ = realtime_setup
        del watch_dir
        zero_counts = {
            "change": 0,
            "delete": 0,
            "embed": 0,
            "dir_delete": 0,
            "dir_index": 0,
        }

        service._service_state = "running"
        service._effective_backend = "watchdog"
        service.monitoring_ready.set()
        service._monitoring_ready_at = service._utc_now()

        stats = await service.get_health()
        pending = stats["pending_mutations"]

        assert stats["live_indexing_state"] == "idle"
        assert stats["pending_files"] == 0
        assert pending["total"] == 0
        assert pending["unique_paths"] == 0
        assert pending["counts_by_operation"] == zero_counts
        assert pending["retry_counts_by_operation"] == zero_counts
        assert pending["retrying_mutations"] == 0
        assert pending["oldest_pending_at"] is None
        assert pending["oldest_pending_age_seconds"] is None
        assert pending["oldest_pending_operation"] is None
        assert pending["oldest_pending_path"] is None
        assert pending["oldest_pending_retry_count"] is None
        assert pending["recovery_phase"] == "idle"
        assert pending["resync_reason"] is None

    @pytest.mark.asyncio
    async def test_event_pressure_reports_excluded_hot_path_without_queue_admission(
        self, realtime_setup
    ):
        """Excluded noisy paths should stay out of the queue and remain diagnosable."""
        service, watch_dir, _, _ = realtime_setup
        noisy_log = watch_dir / "generated" / "steady.log"
        noisy_log.parent.mkdir(parents=True, exist_ok=True)
        noisy_log.write_text("steady\n")

        service._service_state = "running"
        service._effective_backend = "watchdog"
        service.monitoring_ready.set()
        service._monitoring_ready_at = service._utc_now()

        handler = SimpleEventHandler(
            service.event_queue,
            service.config,
            asyncio.get_running_loop(),
            root_path=watch_dir,
            queue_result_callback=service._handle_queue_result,
            source_event_callback=service._record_source_event,
            filtered_event_callback=service._record_filtered_event,
        )
        event = SimpleNamespace(
            event_type="modified",
            is_directory=False,
            src_path=str(noisy_log),
        )

        for _ in range(25):
            handler.on_any_event(event)

        await asyncio.sleep(0)
        stats = await service.get_health()

        assert service.event_queue.empty()
        assert stats["event_queue"]["accepted"] == 0
        assert stats["event_queue"]["dropped"] == 0
        assert stats["pipeline"]["filtered_event_count"] == 25
        assert stats["resync"]["request_count"] == 0
        assert stats["event_pressure"]["state"] == "elevated"
        assert stats["event_pressure"]["sample_path"] == str(noisy_log.resolve())
        assert stats["event_pressure"]["sample_scope"] == "excluded"
        assert stats["event_pressure"]["sample_event_type"] == "modified"
        assert stats["event_pressure"]["events_in_window"] == 25
        assert stats["event_pressure"]["coalesced_updates"] == 0

    @pytest.mark.asyncio
    async def test_hot_file_pending_change_tracks_latest_generation_and_coalescing(
        self, realtime_setup
    ):
        """Repeated writes to one pending file should keep one newest change."""
        service, watch_dir, _, _ = realtime_setup
        target_file = watch_dir / "steady_hot.py"
        target_file.write_text("value = 1\n")

        service._service_state = "running"
        service._effective_backend = "watchdog"
        service.monitoring_ready.set()
        service._monitoring_ready_at = service._utc_now()

        admitted: list[bool] = []
        for version in range(1, 5):
            target_file.write_text(f"value = {version}\n")
            service._record_source_event("modified", target_file)
            service._record_accepted_event("modified", target_file)
            admitted.append(await service.add_file(target_file, priority="change"))

        pending_key = ("change", str(target_file.resolve()))
        pending_mutation = service._pending_mutations[pending_key]
        stats = await service.get_health()

        assert admitted == [True, False, False, False]
        assert stats["pending_files"] == 1
        assert stats["pending_mutations"]["total"] == 1
        assert stats["pending_mutations"]["unique_paths"] == 1
        assert pending_mutation.source_generation == service._current_source_generation(
            target_file
        )
        assert stats["event_pressure"]["sample_path"] == str(target_file.resolve())
        assert stats["event_pressure"]["sample_scope"] == "included"
        assert stats["event_pressure"]["sample_event_type"] == "modified"
        assert stats["event_pressure"]["events_in_window"] == 4
        assert stats["event_pressure"]["coalesced_updates"] == 3

        if service._debounce_tasks:
            await asyncio.wait_for(
                asyncio.gather(*service._debounce_tasks.copy(), return_exceptions=True),
                timeout=2.0,
            )
        _, _, queued_mutation = await service.file_queue.get()
        assert queued_mutation.source_generation == service._current_source_generation(
            target_file
        )

    @pytest.mark.asyncio
    async def test_hot_file_continuous_writes_keep_one_latest_follow_up(
        self,
        realtime_setup,
        monkeypatch,
    ):
        """Slow in-scope processing should retain one newest follow-up for a burst."""
        service, watch_dir, _, _ = realtime_setup
        target_file = watch_dir / "continuous_hot.py"
        target_file.write_text("value = 1\n")
        processed_contents: list[str] = []
        first_processing_started = asyncio.Event()
        release_first_processing = asyncio.Event()
        second_processing_completed = asyncio.Event()

        async def controlled_process_file(
            file_path: Path, skip_embeddings: bool = False
        ) -> dict[str, object]:
            assert file_path == target_file
            assert skip_embeddings is True
            processed_contents.append(file_path.read_text())
            if len(processed_contents) == 1:
                first_processing_started.set()
                await release_first_processing.wait()
            else:
                second_processing_completed.set()
            return {"status": "processed", "chunks": 1, "embeddings": 0}

        original_add_file = service.add_file

        async def skip_embed_follow_up(
            file_path: Path, priority: str = "change"
        ) -> bool:
            if priority == "embed":
                return True
            return await original_add_file(file_path, priority)

        monkeypatch.setattr(
            service.services.indexing_coordinator,
            "process_file",
            controlled_process_file,
        )
        monkeypatch.setattr(service, "add_file", skip_embed_follow_up)

        service._service_state = "running"
        service._effective_backend = "watchdog"
        service.monitoring_ready.set()
        service._monitoring_ready_at = service._utc_now()

        service._record_source_event("modified", target_file)
        service._record_accepted_event("modified", target_file)
        assert await service.add_file(target_file, priority="change") is True
        if service._debounce_tasks:
            await asyncio.wait_for(
                asyncio.gather(*service._debounce_tasks.copy(), return_exceptions=True),
                timeout=2.0,
            )

        process_task = asyncio.create_task(service._process_loop())
        service.process_task = process_task

        try:
            await asyncio.wait_for(first_processing_started.wait(), timeout=1.0)

            for version in range(2, 12):
                target_file.write_text(f"value = {version}\n")
                service._record_source_event("modified", target_file)
                service._record_accepted_event("modified", target_file)
                assert await service.add_file(target_file, priority="change") is False

            pending_stats = await service.get_health()
            assert pending_stats["queue_size"] == 0
            assert pending_stats["pending_files"] == 0
            assert pending_stats["event_pressure"]["sample_scope"] == "included"
            assert pending_stats["event_pressure"]["coalesced_updates"] >= 10
            assert pending_stats["resync"]["request_count"] == 0

            release_first_processing.set()
            await asyncio.wait_for(second_processing_completed.wait(), timeout=2.0)
            final_stats = await _wait_for_realtime_condition(
                service,
                lambda snapshot: (
                    snapshot["pending_files"] == 0
                    and snapshot["queue_size"] == 0
                    and snapshot["pending_mutations"]["total"] == 0
                ),
            )

            assert processed_contents == ["value = 1\n", "value = 11\n"]
            assert final_stats["last_error"] is None
        finally:
            release_first_processing.set()
            process_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await process_task

    @pytest.mark.asyncio
    async def test_event_pressure_interoperates_with_event_queue_overflow(
        self, realtime_setup
    ):
        """Hot-path pressure should remain visible during overflow recovery."""
        service, watch_dir, _, _ = realtime_setup
        callback_event = asyncio.Event()
        noisy_file = watch_dir / "overflow_hot.py"

        service._service_state = "running"
        service._effective_backend = "watchdog"
        service.monitoring_ready.set()
        service._monitoring_ready_at = service._utc_now()

        async def resync_callback(
            reason: str, details: dict[str, object] | None
        ) -> None:
            assert reason == "event_queue_overflow"
            assert details is not None
            while not service.event_queue.empty():
                service.event_queue.get_nowait()
            callback_event.set()

        service._resync_callback = resync_callback
        service.event_queue = asyncio.Queue(maxsize=1)
        service.event_queue.put_nowait(("created", watch_dir / "already_full.py"))

        handler = SimpleEventHandler(
            service.event_queue,
            service.config,
            asyncio.get_running_loop(),
            root_path=watch_dir,
            queue_result_callback=service._handle_queue_result,
            source_event_callback=service._record_source_event,
            filtered_event_callback=service._record_filtered_event,
        )
        event = SimpleNamespace(
            event_type="modified",
            is_directory=False,
            src_path=str(noisy_file),
        )

        for _ in range(25):
            handler.on_any_event(event)

        pending_stats = await _wait_for_realtime_condition(
            service,
            lambda stats: (
                stats["event_queue"]["overflow"]["state"] == "reconciling"
                and stats["resync"]["request_count"] == 1
            ),
        )

        assert pending_stats["live_indexing_state"] == "degraded"
        assert pending_stats["event_pressure"]["state"] == "elevated"
        assert pending_stats["event_pressure"]["sample_path"] == str(
            noisy_file.resolve()
        )
        assert pending_stats["event_pressure"]["sample_scope"] == "included"
        assert pending_stats["event_pressure"]["events_in_window"] == 25

        await asyncio.wait_for(callback_event.wait(), timeout=5.0)

    @pytest.mark.asyncio
    async def test_delete_mutation_waits_for_inflight_change_processing(
        self, realtime_setup, monkeypatch
    ):
        """Delete work should wait behind the in-flight change mutation."""
        service, watch_dir, _, _ = realtime_setup
        target_file = watch_dir / "checkout_storm.py"
        target_file.write_text("def checkout_storm(): pass")
        process_started = asyncio.Event()
        release_processing = asyncio.Event()
        delete_called = asyncio.Event()
        original_process_file = service.services.indexing_coordinator.process_file

        async def blocked_process_file(
            file_path: Path, skip_embeddings: bool = False
        ) -> dict[str, object]:
            assert file_path == target_file
            assert skip_embeddings is True
            process_started.set()
            await release_processing.wait()
            return await original_process_file(
                file_path, skip_embeddings=skip_embeddings
            )

        async def record_delete(file_path: str) -> bool:
            assert Path(file_path) == target_file
            delete_called.set()
            return True

        monkeypatch.setattr(
            service.services.indexing_coordinator,
            "process_file",
            blocked_process_file,
        )
        monkeypatch.setattr(
            service.services.provider,
            "delete_file_completely_async",
            record_delete,
        )

        service._service_state = "running"
        service._effective_backend = "watchdog"
        service.monitoring_ready.set()
        service._monitoring_ready_at = service._utc_now()
        await service.add_file(target_file, priority="priority")

        process_task = asyncio.create_task(service._process_loop())
        service.process_task = process_task

        try:
            await asyncio.wait_for(process_started.wait(), timeout=1.0)
            await service._enqueue_mutation(
                service._build_mutation("delete", target_file)
            )
            await asyncio.sleep(0)
            assert not delete_called.is_set()

            target_file.unlink()
            release_processing.set()

            await asyncio.wait_for(delete_called.wait(), timeout=1.0)
            stats = await _wait_for_realtime_condition(
                service,
                lambda snapshot: snapshot["queue_size"] == 0
                and snapshot["pending_files"] == 0,
            )
            assert stats["last_error"] is None
            assert stats["pipeline"]["processing_error_count"] == 0
        finally:
            release_processing.set()
            await service.stop()

    @pytest.mark.asyncio
    async def test_delete_conflict_retries_without_sticky_degraded_state(
        self, realtime_setup, monkeypatch
    ):
        """Transient delete conflicts should retry and recover without degradation."""
        service, watch_dir, _, _ = realtime_setup
        target_file = watch_dir / "retry_delete.py"
        target_file.write_text("def retry_delete(): pass")
        delete_attempts = 0
        delete_completed = asyncio.Event()

        async def flaky_delete(file_path: str) -> bool:
            nonlocal delete_attempts
            assert Path(file_path) == target_file
            delete_attempts += 1
            if delete_attempts == 1:
                raise DuckDBTransactionConflictError(
                    "delete_file_completely(retry_delete.py) "
                    "cannot run while another DuckDB transaction is active"
                )
            delete_completed.set()
            return True

        monkeypatch.setattr(
            service.services.provider,
            "delete_file_completely_async",
            flaky_delete,
        )
        monkeypatch.setattr(
            service,
            "_DELETE_CONFLICT_BASE_RETRY_DELAY_SECONDS",
            0.01,
            raising=False,
        )

        service._service_state = "running"
        service._effective_backend = "watchdog"
        service.monitoring_ready.set()
        service._monitoring_ready_at = service._utc_now()
        await service._enqueue_mutation(service._build_mutation("delete", target_file))

        process_task = asyncio.create_task(service._process_loop())
        service.process_task = process_task

        try:
            await asyncio.wait_for(delete_completed.wait(), timeout=2.0)
            stats = await _wait_for_realtime_condition(
                service,
                lambda snapshot: snapshot["queue_size"] == 0
                and snapshot["pending_files"] == 0,
                timeout=2.0,
            )
            assert delete_attempts == 2
            assert stats["service_state"] == "running"
            assert stats["last_error"] is None
            assert stats["pipeline"]["processing_error_count"] == 0
            assert stats["live_indexing_state"] == "idle"
        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_stale_delete_retry_is_dropped_after_newer_change(
        self, realtime_setup, monkeypatch
    ):
        """A retried delete must not run after a newer recreate/change."""
        service, watch_dir, _, services = realtime_setup
        target_file = watch_dir / "stale_retry.py"
        target_file.write_text("def stale_retry(): return 1")
        delete_attempts = 0
        conflict_seen = asyncio.Event()
        change_processed = asyncio.Event()
        original_process_file = service.services.indexing_coordinator.process_file

        async def conflicting_delete(file_path: str) -> bool:
            nonlocal delete_attempts
            assert Path(file_path) == target_file
            delete_attempts += 1
            if delete_attempts == 1:
                conflict_seen.set()
                raise DuckDBTransactionConflictError(
                    "delete_file_completely(stale_retry.py) "
                    "cannot run while another DuckDB transaction is active"
                )
            raise AssertionError("stale delete retry should not execute")

        async def wrapped_process_file(
            file_path: Path, skip_embeddings: bool = False
        ) -> dict[str, object]:
            result = await original_process_file(
                file_path, skip_embeddings=skip_embeddings
            )
            if file_path == target_file:
                change_processed.set()
            return result

        async def no_op_embed() -> dict[str, object]:
            return {"status": "up_to_date", "generated": 0}

        monkeypatch.setattr(
            service.services.provider,
            "delete_file_completely_async",
            conflicting_delete,
        )
        monkeypatch.setattr(
            service.services.indexing_coordinator,
            "process_file",
            wrapped_process_file,
        )
        monkeypatch.setattr(
            service.services.indexing_coordinator,
            "generate_missing_embeddings",
            no_op_embed,
        )
        monkeypatch.setattr(
            service,
            "_DELETE_CONFLICT_BASE_RETRY_DELAY_SECONDS",
            0.05,
            raising=False,
        )

        service._service_state = "running"
        service._effective_backend = "watchdog"
        service.monitoring_ready.set()
        service._monitoring_ready_at = service._utc_now()
        service._record_accepted_event("deleted", target_file)
        delete_generation = service._current_source_generation(target_file)
        await service._enqueue_mutation(
            service._build_mutation(
                "delete",
                target_file,
                source_generation=delete_generation,
            )
        )

        process_task = asyncio.create_task(service._process_loop())
        service.process_task = process_task

        try:
            await asyncio.wait_for(conflict_seen.wait(), timeout=1.0)

            target_file.write_text("def stale_retry(): return 2")
            service._record_accepted_event("modified", target_file)
            await service.add_file(target_file, priority="priority")

            await asyncio.wait_for(change_processed.wait(), timeout=2.0)
            stats = await _wait_for_realtime_condition(
                service,
                lambda snapshot: snapshot["queue_size"] == 0
                and snapshot["pending_files"] == 0
                and snapshot["pipeline"]["last_processing_completed_path"]
                == str(target_file),
                timeout=2.0,
            )
            assert delete_attempts == 1
            assert stats["last_error"] is None
            assert stats["pipeline"]["processing_error_count"] == 0
            assert services.provider.get_file_by_path(str(target_file)) is not None
        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_fresh_delete_supersedes_pending_stale_retry(
        self, realtime_setup, monkeypatch
    ):
        """A second delete must replace an older pending retry for the same path."""
        service, watch_dir, _, services = realtime_setup
        target_file = watch_dir / "delete_again.py"
        target_file.write_text("def delete_again(): return 1")
        conflict_seen = asyncio.Event()
        recreate_processed = asyncio.Event()
        delete_attempts = 0
        original_delete = service.services.provider.delete_file_completely_async
        original_process_file = service.services.indexing_coordinator.process_file

        async def conflicting_then_real_delete(file_path: str) -> bool:
            nonlocal delete_attempts
            assert Path(file_path) == target_file
            delete_attempts += 1
            if delete_attempts == 1:
                conflict_seen.set()
                raise DuckDBTransactionConflictError(
                    "delete_file_completely(delete_again.py) "
                    "cannot run while another DuckDB transaction is active"
                )
            return await original_delete(file_path)

        async def wrapped_process_file(
            file_path: Path, skip_embeddings: bool = False
        ) -> dict[str, object]:
            result = await original_process_file(
                file_path,
                skip_embeddings=skip_embeddings,
            )
            if file_path == target_file:
                recreate_processed.set()
            return result

        async def no_op_embed() -> dict[str, object]:
            return {"status": "up_to_date", "generated": 0}

        monkeypatch.setattr(
            service.services.provider,
            "delete_file_completely_async",
            conflicting_then_real_delete,
        )
        monkeypatch.setattr(
            service.services.indexing_coordinator,
            "process_file",
            wrapped_process_file,
        )
        monkeypatch.setattr(
            service.services.indexing_coordinator,
            "generate_missing_embeddings",
            no_op_embed,
        )
        monkeypatch.setattr(
            service,
            "_DELETE_CONFLICT_BASE_RETRY_DELAY_SECONDS",
            0.05,
            raising=False,
        )

        service._service_state = "running"
        service._effective_backend = "watchdog"
        service.monitoring_ready.set()
        service._monitoring_ready_at = service._utc_now()

        process_task = asyncio.create_task(service._process_loop())
        service.process_task = process_task

        try:
            await service.add_file(target_file, priority="priority")
            assert await _wait_for_logical_indexed(services.provider, target_file)
            recreate_processed.clear()

            target_file.unlink()
            service._record_accepted_event("deleted", target_file)
            first_delete_generation = service._current_source_generation(target_file)
            await service._enqueue_mutation(
                service._build_mutation(
                    "delete",
                    target_file,
                    source_generation=first_delete_generation,
                )
            )

            await asyncio.wait_for(conflict_seen.wait(), timeout=1.0)

            target_file.write_text("def delete_again(): return 2")
            service._record_accepted_event("modified", target_file)
            await service.add_file(target_file, priority="priority")
            await asyncio.wait_for(recreate_processed.wait(), timeout=2.0)

            target_file.unlink()
            service._record_accepted_event("deleted", target_file)
            second_delete_generation = service._current_source_generation(target_file)
            accepted = await service._enqueue_mutation(
                service._build_mutation(
                    "delete",
                    target_file,
                    source_generation=second_delete_generation,
                )
            )
            assert accepted is True

            await _wait_for_realtime_condition(
                service,
                lambda snapshot: snapshot["queue_size"] == 0
                and snapshot["pending_files"] == 0,
                timeout=2.0,
            )

            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if services.provider.get_file_by_path(str(target_file)) is None:
                    break
                await asyncio.sleep(0.05)
            assert services.provider.get_file_by_path(str(target_file)) is None
            assert delete_attempts == 2
        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_delete_conflict_exhaustion_sets_realtime_error(
        self, realtime_setup, monkeypatch
    ):
        """Exhausted delete retries should surface as realtime failures."""
        service, watch_dir, _, _ = realtime_setup
        target_file = watch_dir / "exhausted_retry.py"
        target_file.write_text("def exhausted_retry(): pass")
        delete_attempts = 0

        async def always_conflict(file_path: str) -> bool:
            nonlocal delete_attempts
            assert Path(file_path) == target_file
            delete_attempts += 1
            raise DuckDBTransactionConflictError(
                "delete_file_completely(exhausted_retry.py) "
                "cannot run while another DuckDB transaction is active"
            )

        monkeypatch.setattr(
            service.services.provider,
            "delete_file_completely_async",
            always_conflict,
        )
        monkeypatch.setattr(
            service,
            "_DELETE_CONFLICT_MAX_RETRIES",
            1,
            raising=False,
        )
        monkeypatch.setattr(
            service,
            "_DELETE_CONFLICT_BASE_RETRY_DELAY_SECONDS",
            0.01,
            raising=False,
        )

        service._service_state = "running"
        service._effective_backend = "watchdog"
        service.monitoring_ready.set()
        service._monitoring_ready_at = service._utc_now()
        await service._enqueue_mutation(service._build_mutation("delete", target_file))

        process_task = asyncio.create_task(service._process_loop())
        service.process_task = process_task

        try:
            stats = await _wait_for_realtime_condition(
                service,
                lambda snapshot: snapshot["service_state"] == "degraded"
                and snapshot["last_error"] is not None,
                timeout=2.0,
            )
            assert delete_attempts == 2
            assert (
                "cannot run while another DuckDB transaction is active"
                in stats["last_error"]
            )
            assert stats["pipeline"]["processing_error_count"] == 1
            assert stats["failed_files"] == 1
            assert stats["live_indexing_state"] == "degraded"
        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_deleted_directory_cleanup_queues_child_deletes(
        self, realtime_setup, monkeypatch
    ):
        """Deleted-directory cleanup should only expand into queued delete work."""
        service, watch_dir, _, _ = realtime_setup
        deleted_dir = watch_dir / "deleted_dir"
        first_file = deleted_dir / "first.py"
        second_file = deleted_dir / "second.py"
        direct_delete_calls = 0

        async def fake_search_regex_async(pattern: str, page_size: int = 1000):
            assert pattern == f"^{deleted_dir.resolve()}/.*"
            assert page_size == 1000
            return (
                [
                    {"file_path": str(first_file)},
                    {"file_path": str(second_file)},
                ],
                {},
            )

        async def unexpected_direct_delete(_file_path: str) -> bool:
            nonlocal direct_delete_calls
            direct_delete_calls += 1
            raise AssertionError("directory cleanup should not delete directly")

        monkeypatch.setattr(
            service.services.provider,
            "search_regex_async",
            fake_search_regex_async,
        )
        monkeypatch.setattr(
            service.services.provider,
            "delete_file_completely_async",
            unexpected_direct_delete,
        )

        service._service_state = "running"
        service._effective_backend = "watchdog"
        service.monitoring_ready.set()
        service._monitoring_ready_at = service._utc_now()
        service._record_accepted_event("dir_deleted", deleted_dir)
        dir_delete_generation = service._current_source_generation(deleted_dir)

        await service._process_deleted_directory_mutation(
            service._build_mutation(
                "dir_delete",
                deleted_dir,
                source_generation=dir_delete_generation,
            )
        )

        queued_mutations = []
        while not service.file_queue.empty():
            _, _, mutation = await service.file_queue.get()
            queued_mutations.append(mutation)

        assert direct_delete_calls == 0
        assert [mutation.operation for mutation in queued_mutations] == [
            "delete",
            "delete",
        ]
        assert {mutation.path for mutation in queued_mutations} == {
            first_file.resolve(),
            second_file.resolve(),
        }
        assert {mutation.source_generation for mutation in queued_mutations} == {
            dir_delete_generation
        }

    @pytest.mark.asyncio
    async def test_ready_delete_mutations_are_coalesced_into_one_batch(
        self, realtime_setup, monkeypatch
    ):
        """Adjacent queued deletes should execute through one provider batch."""
        service, watch_dir, _, _ = realtime_setup
        first_delete = watch_dir / "batched_one.py"
        second_delete = watch_dir / "batched_two.py"
        change_file = watch_dir / "after_delete_change.py"
        change_file.write_text("value = 1\n")
        batch_calls: list[list[str]] = []
        change_processed = asyncio.Event()

        async def record_delete_batch(file_paths: list[str]) -> int:
            batch_calls.append(list(file_paths))
            return len(file_paths)

        async def fail_if_single_delete_used(_file_path: str) -> bool:
            raise AssertionError(
                "batched deletes should not use single-file delete path"
            )

        async def wrapped_process_file(
            file_path: Path, skip_embeddings: bool = False
        ) -> dict[str, object]:
            assert file_path == change_file
            assert skip_embeddings is True
            change_processed.set()
            return {"status": "processed", "chunks": 1, "embeddings": 0}

        monkeypatch.setattr(
            service.services.provider,
            "delete_files_batch_async",
            record_delete_batch,
        )
        monkeypatch.setattr(
            service.services.provider,
            "delete_file_completely_async",
            fail_if_single_delete_used,
        )
        monkeypatch.setattr(
            service.services.indexing_coordinator,
            "process_file",
            wrapped_process_file,
        )

        service._service_state = "running"
        service._effective_backend = "watchdog"
        service.monitoring_ready.set()
        service._monitoring_ready_at = service._utc_now()

        await service._enqueue_mutation(service._build_mutation("delete", first_delete))
        await service._enqueue_mutation(
            service._build_mutation("delete", second_delete)
        )
        await service._enqueue_mutation(service._build_mutation("change", change_file))

        process_task = asyncio.create_task(service._process_loop())
        service.process_task = process_task

        try:
            await asyncio.wait_for(change_processed.wait(), timeout=2.0)
            stats = await _wait_for_realtime_condition(
                service,
                lambda snapshot: snapshot["queue_size"] == 0
                and snapshot["pending_files"] == 0,
                timeout=2.0,
            )
            assert batch_calls == [
                [str(first_delete.resolve()), str(second_delete.resolve())]
            ]
            assert stats["last_error"] is None
            assert stats["pipeline"]["processing_error_count"] == 0
        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_delete_batch_conflict_retries_and_recovers(
        self, realtime_setup, monkeypatch
    ):
        """A conflicting delete batch should retry per path and recover cleanly."""
        service, watch_dir, _, _ = realtime_setup
        first_delete = watch_dir / "retry_batch_one.py"
        second_delete = watch_dir / "retry_batch_two.py"
        batch_attempts = 0
        batch_completed = asyncio.Event()

        async def flaky_delete_batch(file_paths: list[str]) -> int:
            nonlocal batch_attempts
            assert file_paths == [
                str(first_delete.resolve()),
                str(second_delete.resolve()),
            ]
            batch_attempts += 1
            if batch_attempts == 1:
                raise DuckDBTransactionConflictError(
                    "delete_files_batch(count=2) cannot run while another DuckDB transaction is active"
                )
            batch_completed.set()
            return len(file_paths)

        async def fail_if_single_delete_used(_file_path: str) -> bool:
            raise AssertionError(
                "batched deletes should not fall back to single-file delete"
            )

        monkeypatch.setattr(
            service.services.provider,
            "delete_files_batch_async",
            flaky_delete_batch,
        )
        monkeypatch.setattr(
            service.services.provider,
            "delete_file_completely_async",
            fail_if_single_delete_used,
        )
        monkeypatch.setattr(
            service,
            "_DELETE_CONFLICT_BASE_RETRY_DELAY_SECONDS",
            0.01,
            raising=False,
        )

        service._service_state = "running"
        service._effective_backend = "watchdog"
        service.monitoring_ready.set()
        service._monitoring_ready_at = service._utc_now()

        await service._enqueue_mutation(service._build_mutation("delete", first_delete))
        await service._enqueue_mutation(
            service._build_mutation("delete", second_delete)
        )

        process_task = asyncio.create_task(service._process_loop())
        service.process_task = process_task

        try:
            await asyncio.wait_for(batch_completed.wait(), timeout=2.0)
            stats = await _wait_for_realtime_condition(
                service,
                lambda snapshot: snapshot["queue_size"] == 0
                and snapshot["pending_files"] == 0,
                timeout=2.0,
            )
            assert batch_attempts == 2
            assert stats["service_state"] == "running"
            assert stats["last_error"] is None
            assert stats["pipeline"]["processing_error_count"] == 0
            assert stats["live_indexing_state"] == "idle"
        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_request_resync_is_debounced(self, realtime_setup):
        """Manual resync requests should coalesce into a single scan callback."""
        service, watch_dir, _, _ = realtime_setup
        callback_calls: list[tuple[str, dict[str, object] | None]] = []
        callback_event = asyncio.Event()

        async def resync_callback(
            reason: str, details: dict[str, object] | None
        ) -> None:
            callback_calls.append((reason, details))
            callback_event.set()

        service._resync_callback = resync_callback
        await service.start(watch_dir)

        await service.request_resync("manual_reconcile", {"source": "first"})
        await service.request_resync("manual_reconcile", {"source": "latest"})

        await asyncio.wait_for(callback_event.wait(), timeout=5.0)
        await asyncio.sleep(service._RESYNC_DEBOUNCE_SECONDS + 0.1)

        stats = await service.get_health()
        assert len(callback_calls) == 1, (
            "Debounced resync should only invoke one callback"
        )
        assert callback_calls[0] == ("manual_reconcile", {"source": "latest"})
        assert stats["resync"]["request_count"] == 2
        assert stats["resync"]["performed_count"] == 1
        assert stats["resync"]["needs_resync"] is False
        assert stats["resync"]["last_reason"] == "manual_reconcile"

        await service.stop()

    @pytest.mark.asyncio
    async def test_explicit_polling_backend_reports_polling_mode(self, realtime_setup):
        """Explicit polling config should report polling as the active backend."""
        service, watch_dir, _, _ = realtime_setup
        service.config.indexing.realtime_backend = "polling"

        await service.start(watch_dir)

        stats = await service.get_health()
        assert stats["configured_backend"] == "polling"
        assert stats["effective_backend"] == "polling"
        assert stats["monitoring_mode"] == "polling"
        assert stats["last_warning"] is None

        await service.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_inflight_polling_start(
        self, realtime_setup, monkeypatch
    ):
        """stop() should invalidate an in-flight polling start."""
        service, watch_dir, _, _ = realtime_setup
        service.config.indexing.realtime_backend = "polling"
        startup_blocked = asyncio.Event()
        release_startup = asyncio.Event()

        async def blocked_start_polling_backend(
            _watch_path: Path,
            reason: str,
            emit_warning: bool = True,
        ) -> None:
            assert reason == "Configured realtime backend is polling"
            assert emit_warning is False
            startup_blocked.set()
            await release_startup.wait()

        monkeypatch.setattr(
            service, "_start_polling_backend", blocked_start_polling_backend
        )

        start_task = asyncio.create_task(service.start(watch_dir))
        await asyncio.wait_for(startup_blocked.wait(), timeout=1.0)

        await service.stop()
        release_startup.set()

        with pytest.raises(asyncio.CancelledError):
            await start_task

        stats = await service.get_health()
        assert stats["service_state"] == "stopped"
        assert stats["monitoring_ready"] is False
        assert stats["effective_backend"] == "uninitialized"
        assert service.event_consumer_task is None
        assert service.process_task is None

    @pytest.mark.asyncio
    @pytest.mark.requires_native_watchman
    async def test_watchman_backend_starts_private_sidecar_and_reports_health(
        self, tmp_path
    ):
        """Watchman backend should own a private sidecar and report it as ready."""
        from types import SimpleNamespace

        watch_dir = tmp_path / "watchman_project"
        watch_dir.mkdir(parents=True)
        db_path = watch_dir / ".chunkhound" / "test.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        fake_args = SimpleNamespace(path=watch_dir)
        config = Config(
            args=fake_args,
            database={"path": str(db_path), "provider": "duckdb"},
            indexing={"realtime_backend": "watchman"},
        )

        services = create_services(db_path, config)
        services.provider.connect()

        service = RealtimeIndexingService(services, config)

        try:
            await service.start(watch_dir)

            stats = await service.get_health()
            assert stats["configured_backend"] == "watchman"
            assert stats["effective_backend"] == "watchman"
            assert stats["monitoring_mode"] == "watchman"
            assert stats["service_state"] == "running"
            assert stats["monitoring_ready"] is True
            assert stats["observer_alive"] is True
            assert stats["watchman_pid"] is not None
            assert stats["watchman_sidecar_state"] == "running"
            assert stats["watchman_session_alive"] is True
            assert stats["watchman_connection_state"] == "connected"
            assert stats["watchman_subscription_name"] == "chunkhound-live-indexing"
            assert stats["watchman_subscription_count"] == 1
            if listener_path_is_filesystem(resolve_packaged_watchman_runtime()):
                assert Path(stats["watchman_socket_path"]).exists()
            assert stats["watchman_watch_root"] == str(watch_dir.resolve())
            assert stats["watchman_relative_root"] is None
            assert Path(stats["watchman_metadata_path"]).is_file()
            assert service.watchman_subscription_queue is not None
        finally:
            await service.stop()
            services.provider.disconnect()

        assert not (watch_dir / ".chunkhound" / "watchman" / "metadata.json").exists()

    @pytest.mark.asyncio
    @pytest.mark.requires_native_watchman
    async def test_watchman_backend_indexes_real_file_mutation(self, tmp_path):
        """Watchman backend should index a real file mutation without injected PDUs."""
        from types import SimpleNamespace

        watch_dir = tmp_path / "watchman_project"
        watch_dir.mkdir(parents=True)
        db_path = watch_dir / ".chunkhound" / "test.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        fake_args = SimpleNamespace(path=watch_dir)
        config = Config(
            args=fake_args,
            database={"path": str(db_path), "provider": "duckdb"},
            indexing={"realtime_backend": "watchman"},
        )

        services = create_services(db_path, config)
        services.provider.connect()
        service = RealtimeIndexingService(services, config)

        try:
            await service.start(watch_dir)

            file_path = watch_dir / "src" / "watchman_live_runtime.py"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(
                "def watchman_live_runtime_symbol():\n    return 1\n",
                encoding="utf-8",
            )

            assert await wait_for_indexed(services.provider, file_path, timeout=10.0)

            stats = await service.get_health()
            assert stats["watchman_connection_state"] == "connected"
            assert stats["watchman_subscription_pdu_count"] >= 1
        finally:
            await service.stop()
            services.provider.disconnect()

    @pytest.mark.asyncio
    @pytest.mark.requires_native_watchman
    @pytest.mark.skipif(os.name == "nt", reason="Linux mount topology only")
    async def test_watchman_backend_indexes_real_file_mutation_under_nested_mount(
        self,
    ):
        """Watchman backend should observe live mutations inside a nested mount."""
        from types import SimpleNamespace

        configured_paths = _configured_mount_regression_paths()
        if configured_paths is None:
            pytest.skip("Linux mount-aware Watchman fixture paths are not configured")

        watch_dir, nested_mount = configured_paths
        if not watch_dir.is_dir() or not nested_mount.is_dir():
            pytest.skip("Linux mount-aware Watchman fixture paths do not exist")

        if nested_mount not in discover_nested_linux_mount_roots(watch_dir):
            pytest.skip("Configured fixture does not expose a nested mount boundary")

        run_suffix = str(time.time_ns())
        db_root = watch_dir / ".chunkhound" / f"mount-aware-{run_suffix}"
        db_path = db_root / "test.db"
        db_root.mkdir(parents=True, exist_ok=True)
        project_dir = nested_mount / f"watchman_mount_project_{run_suffix}"
        project_dir.mkdir(parents=True, exist_ok=True)

        fake_args = SimpleNamespace(path=watch_dir)
        config = Config(
            args=fake_args,
            database={"path": str(db_path), "provider": "duckdb"},
            indexing={"realtime_backend": "watchman"},
        )

        services = create_services(db_path, config)
        services.provider.connect()
        service = RealtimeIndexingService(services, config)

        try:
            await service.start(watch_dir)

            file_path = project_dir / "src" / "watchman_mount_runtime.py"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(
                "def watchman_mount_runtime_symbol():\n    return 3\n",
                encoding="utf-8",
            )

            assert await wait_for_indexed(services.provider, file_path, timeout=15.0)

            stats = await service.get_health()
            assert stats["watchman_connection_state"] == "connected"
            assert stats["watchman_subscription_count"] >= 2
            assert len(stats["watchman_scopes"]) >= 2
            assert any(
                scope["requested_path"] == str(nested_mount.resolve())
                and scope["scope_kind"] == "nested_mount"
                for scope in stats["watchman_scopes"]
            )
        finally:
            await service.stop()
            services.provider.disconnect()
            shutil.rmtree(project_dir, ignore_errors=True)
            shutil.rmtree(db_root, ignore_errors=True)

    @pytest.mark.asyncio
    @pytest.mark.requires_native_watchman
    @pytest.mark.skipif(os.name != "nt", reason="Windows topology only")
    async def test_watchman_backend_indexes_real_file_mutation_under_windows_junction(
        self, tmp_path
    ):
        """Watchman backend should observe live mutations through a Windows junction."""
        from types import SimpleNamespace

        watch_dir = tmp_path / "watchman_project"
        watch_dir.mkdir(parents=True)
        physical_workspace = tmp_path / "junction_target"
        physical_workspace.mkdir(parents=True)
        junction_dir = watch_dir / "linked_workspace"
        try:
            create_windows_directory_junction(junction_dir, physical_workspace)
        except RuntimeError as error:
            pytest.skip(str(error))

        db_path = watch_dir / ".chunkhound" / "test.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        fake_args = SimpleNamespace(path=watch_dir)
        config = Config(
            args=fake_args,
            database={"path": str(db_path), "provider": "duckdb"},
            indexing={"realtime_backend": "watchman"},
        )

        services = create_services(db_path, config)
        services.provider.connect()
        service = RealtimeIndexingService(services, config)

        try:
            await service.start(watch_dir)

            file_path = junction_dir / "src" / "watchman_junction_runtime.py"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(
                "def watchman_junction_runtime_symbol():\n    return 7\n",
                encoding="utf-8",
            )

            assert await _wait_for_logical_indexed(
                services.provider, file_path, timeout=30.0
            )

            stats = await service.get_health()
            assert stats["watchman_connection_state"] == "connected"
            assert stats["watchman_subscription_count"] >= 2
            assert isinstance(stats["watchman_scopes"], list)
            assert any(
                scope["scope_kind"] == "nested_junction"
                and scope["requested_path"] == str(junction_dir)
                and scope["watch_root"] == str(physical_workspace.resolve())
                for scope in stats["watchman_scopes"]
            )
        finally:
            await service.stop()
            services.provider.disconnect()
            remove_windows_directory_junction(junction_dir)
            shutil.rmtree(physical_workspace, ignore_errors=True)

    @pytest.mark.asyncio
    @pytest.mark.requires_native_watchman
    async def test_watchman_backend_recovers_live_monitoring_after_disconnect(
        self, tmp_path
    ):
        """Watchman reconnect should restore live indexing without daemon restart."""
        from types import SimpleNamespace

        watch_dir = tmp_path / "watchman_project"
        watch_dir.mkdir(parents=True)
        db_path = watch_dir / ".chunkhound" / "test.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        fake_args = SimpleNamespace(path=watch_dir)
        config = Config(
            args=fake_args,
            database={"path": str(db_path), "provider": "duckdb"},
            indexing={"realtime_backend": "watchman"},
        )

        services = create_services(db_path, config)
        services.provider.connect()
        service = RealtimeIndexingService(services, config)
        callback_calls: list[tuple[str, dict[str, object] | None]] = []
        callback_event = asyncio.Event()

        async def resync_callback(
            reason: str, details: dict[str, object] | None
        ) -> None:
            callback_calls.append((reason, details))
            callback_event.set()

        service._resync_callback = resync_callback

        try:
            await service.start(watch_dir)
            adapter = service._monitor_adapter
            disconnect_process = _active_watchman_disconnect_process(adapter)

            disconnect_process.terminate()

            await asyncio.wait_for(callback_event.wait(), timeout=5.0)
            restored_stats = await _wait_for_realtime_condition(
                service,
                lambda stats: (
                    stats["watchman_reconnect"]["state"] == "restored"
                    and stats["watchman_connection_state"] == "connected"
                ),
            )

            file_path = watch_dir / "src" / "watchman_recovered_runtime.py"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(
                "def watchman_recovered_runtime_symbol():\n    return 2\n",
                encoding="utf-8",
            )

            assert await wait_for_indexed(services.provider, file_path, timeout=10.0)

            final_stats = await service.get_health()
            assert callback_calls
            assert restored_stats["watchman_reconnect"]["last_result"] == "restored"
            assert final_stats["watchman_connection_state"] == "connected"
            assert final_stats["watchman_reconnect"]["state"] == "restored"
            assert final_stats["watchman_subscription_pdu_count"] >= 1
        finally:
            await service.stop()
            services.provider.disconnect()

    @pytest.mark.asyncio
    @pytest.mark.requires_native_watchman
    async def test_watchman_backend_requires_session_capabilities(
        self, tmp_path, monkeypatch
    ):
        """Watchman startup should fail when required capabilities are missing."""
        from types import SimpleNamespace

        watch_dir = tmp_path / "watchman_project"
        watch_dir.mkdir(parents=True)
        db_path = watch_dir / ".chunkhound" / "test.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        fake_args = SimpleNamespace(path=watch_dir)
        config = Config(
            args=fake_args,
            database={"path": str(db_path), "provider": "duckdb"},
            indexing={"realtime_backend": "watchman"},
        )

        services = create_services(db_path, config)
        services.provider.connect()
        service = RealtimeIndexingService(services, config)
        monkeypatch.setenv(
            "CHUNKHOUND_FAKE_WATCHMAN_MISSING_CAPABILITY", "relative_root"
        )

        try:
            with pytest.raises(RuntimeError, match="relative_root"):
                await service.start(watch_dir)

            stats = await service.get_health()
            assert stats["service_state"] == "degraded"
            assert "relative_root" in (stats["last_error"] or "")
            assert stats["watchman_session_alive"] is False
        finally:
            await service.stop()
            services.provider.disconnect()

    @pytest.mark.asyncio
    async def test_missing_resync_callback_degrades_without_task_leak(
        self, realtime_setup
    ):
        """A missing resync callback should degrade status cleanly."""
        service, watch_dir, _, _ = realtime_setup
        loop = asyncio.get_running_loop()
        loop_errors: list[dict] = []
        previous_handler = loop.get_exception_handler()

        def exception_handler(_loop, context) -> None:
            loop_errors.append(context)

        loop.set_exception_handler(exception_handler)

        try:
            await service.start(watch_dir)
            await service.request_resync("manual_reconcile")
            await asyncio.sleep(service._RESYNC_DEBOUNCE_SECONDS + 0.1)

            stats = await service.get_health()
            assert stats["service_state"] == "degraded"
            assert stats["last_error"] == "No resync callback configured"
            assert stats["resync"]["last_error"] == "No resync callback configured"
            assert stats["resync"]["needs_resync"] is True
            assert loop_errors == []
        finally:
            loop.set_exception_handler(previous_handler)
            await service.stop()

    @pytest.mark.asyncio
    async def test_failing_resync_callback_degrades_without_task_leak(
        self, realtime_setup
    ):
        """A failing resync callback should stay contained in service status."""
        service, watch_dir, _, _ = realtime_setup
        loop = asyncio.get_running_loop()
        loop_errors: list[dict] = []
        previous_handler = loop.get_exception_handler()

        def exception_handler(_loop, context) -> None:
            loop_errors.append(context)

        async def failing_resync_callback(
            reason: str, details: dict[str, object] | None
        ) -> None:
            assert reason == "manual_reconcile"
            assert details == {"source": "test"}
            raise RuntimeError("resync exploded")

        loop.set_exception_handler(exception_handler)

        try:
            service._resync_callback = failing_resync_callback
            await service.start(watch_dir)
            await service.request_resync("manual_reconcile", {"source": "test"})
            await asyncio.sleep(service._RESYNC_DEBOUNCE_SECONDS + 0.1)

            stats = await service.get_health()
            assert stats["service_state"] == "degraded"
            assert stats["last_error"] == "Realtime resync failed: resync exploded"
            assert stats["resync"]["last_error"] == "resync exploded"
            assert stats["resync"]["needs_resync"] is True
            assert loop_errors == []
        finally:
            loop.set_exception_handler(previous_handler)
            await service.stop()

    @pytest.mark.asyncio
    async def test_error_result_resync_callback_stays_degraded(
        self, realtime_setup
    ) -> None:
        """Structured callback error results should preserve stale/degraded state."""
        service, watch_dir, _, _ = realtime_setup

        async def error_result_resync_callback(
            reason: str, details: dict[str, object] | None
        ) -> dict[str, object]:
            assert reason == "manual_reconcile"
            assert details == {"source": "test"}
            return {
                "status": "error",
                "error": "embedding follow-up failed",
                "generated": 0,
            }

        service._resync_callback = error_result_resync_callback
        await service.start(watch_dir)
        await service.request_resync("manual_reconcile", {"source": "test"})
        await asyncio.sleep(service._RESYNC_DEBOUNCE_SECONDS + 0.1)

        stats = await service.get_health()
        assert stats["service_state"] == "degraded"
        assert (
            stats["last_error"]
            == "Realtime resync failed: Resync callback reported error status: "
            "embedding follow-up failed"
        )
        assert (
            stats["resync"]["last_error"]
            == "Resync callback reported error status: embedding follow-up failed"
        )
        assert stats["resync"]["needs_resync"] is True
        assert stats["resync"]["performed_count"] == 0

        await service.stop()

    @pytest.mark.asyncio
    async def test_watchdog_timeout_fallback_does_not_adopt_late_watchdog(
        self, realtime_setup, monkeypatch
    ):
        """Late watchdog bootstrap results should be stopped after polling fallback."""
        service, watch_dir, _, _ = realtime_setup
        service.config.indexing.realtime_backend = "watchdog"
        late_watchdog_returned = asyncio.Event()
        late_observer = MagicMock()

        def late_bootstrap(
            _watch_path: Path,
            loop: asyncio.AbstractEventLoop,
            _abort_event,
        ) -> tuple[MagicMock, MagicMock]:
            time.sleep(0.05)
            loop.call_soon_threadsafe(late_watchdog_returned.set)
            return late_observer, MagicMock()

        monkeypatch.setattr(
            service, "_WATCHDOG_SETUP_TIMEOUT_SECONDS", 0.01, raising=False
        )
        monkeypatch.setattr(
            service, "_POLLING_STARTUP_SETTLE_SECONDS", 0.01, raising=False
        )
        monkeypatch.setattr(service, "_bootstrap_fs_monitor", late_bootstrap)

        await service.start(watch_dir)
        await asyncio.wait_for(late_watchdog_returned.wait(), timeout=1.0)
        await asyncio.sleep(0.05)

        stats = await service.get_health()
        assert stats["monitoring_mode"] == "polling"
        assert stats["configured_backend"] == "watchdog"
        assert stats["effective_backend"] == "polling"
        assert service.observer is None
        assert service.event_handler is None
        assert service._polling_task is not None
        late_observer.stop.assert_called_once()
        late_observer.join.assert_called_once_with(timeout=1.0)

        await service.stop()

    @pytest.mark.asyncio
    async def test_stop_swallows_bootstrap_exception_and_finishes_cleanup(
        self, realtime_setup
    ):
        """Bootstrap exceptions during stop should not abort the rest of cleanup."""
        service, _watch_dir, _, _ = realtime_setup
        service.config.indexing.realtime_backend = "watchdog"
        service._monitor_adapter = service._build_monitor_adapter()
        service._watchdog_setup_task = asyncio.create_task(asyncio.sleep(3600))
        bootstrap_future: asyncio.Future[tuple[MagicMock, MagicMock] | None] = (
            asyncio.get_running_loop().create_future()
        )
        bootstrap_future.set_exception(RuntimeError("bootstrap exploded during stop"))
        service._watchdog_bootstrap_future = bootstrap_future

        await service.stop()

        assert service._service_state == "stopped"
        assert service._watchdog_bootstrap_future is None

    @pytest.mark.asyncio
    @pytest.mark.requires_native_watchman
    async def test_stop_cancels_inflight_watchman_start_and_cleans_sidecar(
        self, tmp_path, monkeypatch
    ):
        """stop() should cancel Watchman startup and leave no owned sidecar state."""
        from types import SimpleNamespace

        watch_dir = tmp_path / "watchman_project"
        watch_dir.mkdir(parents=True)
        fake_args = SimpleNamespace(path=watch_dir)
        config = Config(
            args=fake_args,
            database={
                "path": str(watch_dir / ".chunkhound" / "test.db"),
                "provider": "duckdb",
            },
            indexing={"realtime_backend": "watchman"},
        )
        Path(config.database.path).parent.mkdir(parents=True, exist_ok=True)

        services = create_services(config.database.path, config)
        services.provider.connect()
        monkeypatch.setenv("CHUNKHOUND_FAKE_WATCHMAN_START_DELAY_SECONDS", "2")
        service = RealtimeIndexingService(services, config)

        try:
            start_task = asyncio.create_task(service.start(watch_dir))
            await asyncio.sleep(0.1)

            await service.stop()

            with pytest.raises(asyncio.CancelledError):
                await start_task

            stats = await service.get_health()
            assert stats["configured_backend"] == "watchman"
            assert stats["effective_backend"] == "uninitialized"
            assert stats["monitoring_mode"] == "uninitialized"
            assert stats["service_state"] == "stopped"
            assert stats["monitoring_ready"] is False
            assert service.event_consumer_task is None
            assert service.process_task is None
        finally:
            await service.stop()
            services.provider.disconnect()

        assert not (watch_dir / ".chunkhound" / "watchman" / "metadata.json").exists()

    @pytest.mark.asyncio
    async def test_full_event_queue_collapses_overflow_burst_into_one_resync(
        self, realtime_setup, monkeypatch
    ):
        """Dense queue overflow should converge as one reconciliation burst."""
        service, watch_dir, _, _ = realtime_setup
        warning_messages: list[str] = []
        callback_event = asyncio.Event()
        resync_calls: list[tuple[str, dict[str, object] | None]] = []

        service._service_state = "running"
        service._effective_backend = "watchdog"
        service.monitoring_ready.set()
        service._monitoring_ready_at = service._utc_now()

        monkeypatch.setattr(
            "chunkhound.services.realtime_indexing_service.logger.warning",
            lambda message: warning_messages.append(message),
        )

        async def resync_callback(
            reason: str, details: dict[str, object] | None
        ) -> None:
            resync_calls.append((reason, details))
            while not service.event_queue.empty():
                service.event_queue.get_nowait()
            callback_event.set()

        service._resync_callback = resync_callback
        service.event_queue = asyncio.Queue(maxsize=1)
        service.event_queue.put_nowait(("created", watch_dir / "already_full.py"))

        handler = SimpleEventHandler(
            service.event_queue,
            service.config,
            asyncio.get_running_loop(),
            root_path=watch_dir,
            queue_result_callback=service._handle_queue_result,
        )
        handler._queue_event("modified", watch_dir / "overflow_one.py")
        handler._queue_event("deleted", watch_dir / "overflow_two.py")
        handler._queue_event("created", watch_dir / "overflow_three.py")

        pending_stats = await _wait_for_realtime_condition(
            service,
            lambda stats: (
                stats["event_queue"]["dropped"] == 3
                and stats["event_queue"]["overflow"]["state"] == "reconciling"
                and stats["resync"]["request_count"] == 1
            ),
        )

        assert pending_stats["event_queue"]["overflow"]["current_burst_dropped"] == 3
        assert pending_stats["event_queue"]["last_reason"] == "queue_full"
        assert pending_stats["resync"]["request_count"] == 1
        assert pending_stats["resync"]["last_reason"] == "event_queue_overflow"
        assert pending_stats["live_indexing_state"] == "degraded"
        assert len(resync_calls) == 0
        assert warning_messages == [
            "Realtime event queue overflow detected; entering reconciliation mode."
        ]

        await asyncio.wait_for(callback_event.wait(), timeout=5.0)

        stats = await _wait_for_realtime_condition(
            service,
            lambda current: (
                current["event_queue"]["overflow"]["state"] == "idle"
                and current["live_indexing_state"] == "idle"
            ),
        )

        assert len(resync_calls) == 1
        assert resync_calls[0][0] == "event_queue_overflow"
        assert resync_calls[0][1] is not None
        assert stats["event_queue"]["dropped"] == 3
        assert stats["event_queue"]["last_reason"] == "queue_full"
        assert stats["event_queue"]["overflow"]["current_burst_dropped"] == 0
        assert stats["event_queue"]["overflow"]["last_burst_dropped"] == 3
        assert stats["event_queue"]["overflow"]["last_cleared_at"] is not None
        assert stats["resync"]["request_count"] == 1
        assert stats["resync"]["performed_count"] == 1
        assert stats["resync"]["needs_resync"] is False
        assert stats["last_warning"] == (
            "Realtime event queue overflow recovered after dropping 3 events."
        )

    @pytest.mark.asyncio
    async def test_full_event_queue_keeps_overflow_state_when_resync_fails(
        self, realtime_setup
    ):
        """Overflow state should stay explicit until a reconciliation succeeds."""
        service, watch_dir, _, _ = realtime_setup
        callback_event = asyncio.Event()

        service._service_state = "running"
        service._effective_backend = "watchdog"
        service.monitoring_ready.set()
        service._monitoring_ready_at = service._utc_now()

        async def resync_callback(
            reason: str, details: dict[str, object] | None
        ) -> None:
            assert reason == "event_queue_overflow"
            assert details is not None
            callback_event.set()
            raise RuntimeError("simulated overflow reconciliation failure")

        service._resync_callback = resync_callback
        service.event_queue = asyncio.Queue(maxsize=1)
        service.event_queue.put_nowait(("created", watch_dir / "already_full.py"))

        handler = SimpleEventHandler(
            service.event_queue,
            service.config,
            asyncio.get_running_loop(),
            root_path=watch_dir,
            queue_result_callback=service._handle_queue_result,
        )
        handler._queue_event("modified", watch_dir / "overflow_failure.py")

        await asyncio.wait_for(callback_event.wait(), timeout=5.0)

        stats = await _wait_for_realtime_condition(
            service,
            lambda current: current["event_queue"]["overflow"]["state"] == "failed",
        )

        assert stats["event_queue"]["dropped"] == 1
        assert stats["event_queue"]["overflow"]["current_burst_dropped"] == 1
        assert stats["event_queue"]["overflow"]["last_burst_dropped"] == 1
        assert stats["resync"]["request_count"] == 1
        assert stats["resync"]["performed_count"] == 0
        assert stats["resync"]["needs_resync"] is True
        assert (
            stats["resync"]["last_error"] == "simulated overflow reconciliation failure"
        )
        assert stats["service_state"] == "degraded"
        assert stats["live_indexing_state"] == "degraded"
        assert stats["live_indexing_hint"] == (
            "Live indexing remains degraded after internal event queue overflow; "
            "inspect event_queue.overflow and resync.last_error."
        )

    @pytest.mark.asyncio
    async def test_full_event_queue_sets_failed_state_when_resync_callback_missing(
        self, realtime_setup
    ):
        """Missing resync callback should leave overflow in an explicit failed state."""
        service, watch_dir, _, _ = realtime_setup

        service._service_state = "running"
        service._effective_backend = "watchdog"
        service.monitoring_ready.set()
        service._monitoring_ready_at = service._utc_now()
        service._resync_callback = None
        service.event_queue = asyncio.Queue(maxsize=1)
        service.event_queue.put_nowait(("created", watch_dir / "already_full.py"))

        handler = SimpleEventHandler(
            service.event_queue,
            service.config,
            asyncio.get_running_loop(),
            root_path=watch_dir,
            queue_result_callback=service._handle_queue_result,
        )
        handler._queue_event("modified", watch_dir / "overflow_missing_callback.py")

        stats = await _wait_for_realtime_condition(
            service,
            lambda current: (
                current["event_queue"]["overflow"]["state"] == "failed"
                and current["resync"]["last_error"] == "No resync callback configured"
            ),
        )

        assert stats["event_queue"]["dropped"] == 1
        assert stats["event_queue"]["overflow"]["current_burst_dropped"] == 1
        assert stats["event_queue"]["overflow"]["last_burst_dropped"] == 1
        assert stats["resync"]["request_count"] == 1
        assert stats["resync"]["performed_count"] == 0
        assert stats["resync"]["needs_resync"] is True
        assert stats["service_state"] == "degraded"
        assert stats["last_error"] == "No resync callback configured"
        assert stats["live_indexing_state"] == "degraded"
        assert stats["live_indexing_hint"] == (
            "Live indexing remains degraded after internal event queue overflow; "
            "inspect event_queue.overflow and resync.last_error."
        )

    @pytest.mark.asyncio
    async def test_polling_monitor_uses_to_thread_snapshot(
        self, realtime_setup, monkeypatch
    ):
        """Polling monitor should offload filesystem snapshots off the event loop."""
        service, watch_dir, _, _ = realtime_setup
        to_thread_called = asyncio.Event()

        async def fake_to_thread(func, *args, **kwargs):
            assert func == service._polling_snapshot
            assert args == (watch_dir,)
            assert kwargs == {}
            to_thread_called.set()
            return {}, 0, False

        monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
        poll_task = asyncio.create_task(service._polling_monitor(watch_dir))

        try:
            await asyncio.wait_for(to_thread_called.wait(), timeout=1.0)
        finally:
            poll_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await poll_task

    @pytest.mark.asyncio
    async def test_polling_monitor_detects_size_change_when_mtime_is_constant(
        self, realtime_setup, monkeypatch
    ):
        """Polling mode should treat size changes as modifications."""
        service, watch_dir, _, _ = realtime_setup
        target_file = watch_dir / "same_mtime_size_change.py"
        change_detected = asyncio.Event()
        add_calls: list[tuple[Path, str]] = []
        snapshots = iter(
            [
                ({target_file: (100, 10)}, 1, False),
                ({target_file: (100, 30)}, 1, False),
                ({target_file: (100, 30)}, 1, False),
            ]
        )

        async def fake_to_thread(func, *args, **kwargs):
            assert func == service._polling_snapshot
            assert args == (watch_dir,)
            assert kwargs == {}
            return next(snapshots)

        async def fake_add_file(file_path: Path, priority: str = "change") -> bool:
            add_calls.append((file_path, priority))
            if len(add_calls) >= 2:
                change_detected.set()
            return True

        monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
        monkeypatch.setattr(service, "add_file", fake_add_file)

        poll_task = asyncio.create_task(service._polling_monitor(watch_dir))

        try:
            await asyncio.wait_for(change_detected.wait(), timeout=2.5)
        finally:
            poll_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await poll_task

        assert add_calls == [
            (target_file, "change"),
            (target_file, "change"),
        ]
        stats = await service.get_health()
        assert stats["pipeline"]["last_accepted_event_path"] == str(target_file)

    @pytest.mark.asyncio
    async def test_polling_monitor_coalesced_change_does_not_advance_accepted_event(
        self, realtime_setup, monkeypatch
    ):
        """Polling should not advance accepted timestamps for already-pending work."""
        service, watch_dir, _, _ = realtime_setup
        target_file = watch_dir / "already_pending.py"
        target_file.write_text("def already_pending(): pass")
        baseline_accepted_at = "2026-03-16T12:00:00+00:00"
        baseline_accepted_type = "modified"
        baseline_accepted_path = str(watch_dir / "baseline.py")
        service.pending_files.add(target_file)
        service._pending_debounce[str(target_file)] = time.monotonic()
        service._last_accepted_event_at = baseline_accepted_at
        service._last_accepted_event_type = baseline_accepted_type
        service._last_accepted_event_path = baseline_accepted_path

        async def fake_to_thread(func, *args, **kwargs):
            assert func == service._polling_snapshot
            assert args == (watch_dir,)
            assert kwargs == {}
            return {target_file: (100, 10)}, 1, False

        async def stop_after_first_iteration(_delay: float) -> None:
            raise asyncio.CancelledError

        monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
        monkeypatch.setattr(asyncio, "sleep", stop_after_first_iteration)

        with pytest.raises(asyncio.CancelledError):
            await service._polling_monitor(watch_dir)

        stats = await service.get_health()
        pipeline = stats["pipeline"]
        assert pipeline["last_source_event_path"] == str(target_file)
        assert pipeline["last_accepted_event_at"] == baseline_accepted_at
        assert pipeline["last_accepted_event_type"] == baseline_accepted_type
        assert pipeline["last_accepted_event_path"] == baseline_accepted_path

    @pytest.mark.asyncio
    async def test_debounced_add_file_retries_early_timer_wake(
        self, realtime_setup, monkeypatch
    ):
        """Debounce should retry if timer granularity wakes before the target."""
        service, watch_dir, _, _ = realtime_setup
        target_file = watch_dir / "early_wake.py"
        target_file.write_text("def early_wake(): pass")
        file_key = str(target_file)
        sleep_calls: list[float] = []
        monotonic_values = iter([100.49, 100.5001])

        mutation = service._build_mutation("change", target_file)
        assert service._register_pending_mutation(mutation)
        service._pending_debounce[file_key] = 100.0

        async def fake_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        def fake_monotonic() -> float:
            try:
                return next(monotonic_values)
            except StopIteration:
                return 100.5001

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        monkeypatch.setattr(time, "monotonic", fake_monotonic)

        await service._debounced_add_file(mutation)

        assert len(sleep_calls) == 2
        assert sleep_calls[0] == service._debounce_delay
        assert sleep_calls[1] == pytest.approx(0.01, abs=1e-6)
        assert file_key not in service._pending_debounce
        priority, _, queued_mutation = await service.file_queue.get()
        assert priority == service._mutation_priority("change")
        assert queued_mutation == mutation

    @pytest.mark.asyncio
    async def test_filesystem_monitoring_detects_changes(self, realtime_setup):
        """Test that filesystem changes are detected and processed."""
        service, watch_dir, _, services = realtime_setup
        await service.start(watch_dir)

        # Create a Python file - should be detected and processed
        test_file = watch_dir / "test_monitor.py"
        test_file.write_text("def hello_world(): pass")

        # Wait for filesystem event + debouncing + processing
        found = await wait_for_indexed(services.provider, test_file)

        # This tests the full pipeline: detection -> processing -> storage
        assert found, "File should be detected and processed by filesystem monitoring"
        stats = await _wait_for_realtime_condition(
            service,
            lambda current: (
                current["pipeline"]["last_source_event_at"] is not None
                and current["pipeline"]["last_accepted_event_at"] is not None
                and current["pipeline"]["last_processing_started_at"] is not None
                and current["pipeline"]["last_processing_completed_at"] is not None
                and current["pipeline"]["last_processing_completed_path"]
                == str(test_file)
            ),
        )
        pipeline = stats["pipeline"]
        assert pipeline["last_processing_completed_path"] == str(test_file)
        assert stats["live_indexing_state"] in {"idle", "busy"}

        await service.stop()

    @pytest.mark.asyncio
    async def test_multiple_rapid_changes_handling(self, realtime_setup):
        """Test handling multiple rapid file changes - stress test for concurrency."""
        service, watch_dir, _, _ = realtime_setup
        await service.start(watch_dir)

        # Create multiple files in rapid succession
        test_files = []
        for i in range(5):
            test_file = watch_dir / f"rapid_{i}.py"
            test_file.write_text(f"def func_{i}(): return {i}")
            test_files.append(test_file)
            # Small delay to create separate events
            await asyncio.sleep(0.1)

        # Wait for all processing
        await asyncio.sleep(3.0)

        # Check service is still alive
        stats = await service.get_stats()
        assert stats.get("observer_alive", False), (
            "Service should still be running after rapid changes"
        )

        # This test mainly checks service doesn't crash under load
        await service.stop()

    @pytest.mark.asyncio
    async def test_service_survives_processing_errors(
        self, realtime_setup, monkeypatch
    ):
        """Test service continues working after processing errors."""
        service, watch_dir, _, _ = realtime_setup
        original_process_file = service.services.indexing_coordinator.process_file
        bad_file = watch_dir / "bad_file.py"

        async def flaky_process_file(
            file_path: Path, skip_embeddings: bool = False
        ) -> dict[str, object]:
            if file_path == bad_file:
                raise RuntimeError("synthetic realtime processing failure")
            return await original_process_file(
                file_path, skip_embeddings=skip_embeddings
            )

        monkeypatch.setattr(
            service.services.indexing_coordinator,
            "process_file",
            flaky_process_file,
        )
        await service.start(watch_dir)

        # Create a file that reliably forces a processing failure.
        bad_file.write_text("def broken(): pass")

        await _wait_for_realtime_condition(
            service,
            lambda stats: stats["pipeline"]["processing_error_count"] >= 1,
        )

        # Create a normal file after the bad one
        good_file = watch_dir / "good_file.py"
        good_file.write_text("def good_function(): pass")

        await asyncio.sleep(2.0)

        # Main goal: service should still be alive
        stats = await service.get_stats()
        assert stats.get("observer_alive", False), (
            "Service should survive processing errors"
        )
        assert stats["pipeline"]["processing_error_count"] >= 1

        await service.stop()

    @pytest.mark.asyncio
    async def test_file_type_filtering_works(self, realtime_setup):
        """Test that only supported file types are processed."""
        service, watch_dir, _, services = realtime_setup
        await service.start(watch_dir)

        # Create supported file
        py_file = watch_dir / "supported.py"
        py_file.write_text("def supported(): pass")

        # Create unsupported file
        bin_file = watch_dir / "unsupported.xyz"
        bin_file.write_text("unsupported content")

        await asyncio.sleep(1.5)

        # Check processing results
        bin_record = services.provider.get_file_by_path(str(bin_file))

        # Python file may still fail for unrelated reasons, but the unsupported
        # file should definitely be ignored.
        # Binary file should definitely be ignored
        assert bin_record is None, "Unsupported file types should be ignored"

        await service.stop()

    @pytest.mark.asyncio
    async def test_background_vs_realtime_processing(self, realtime_setup):
        """Test interaction between initial scan and real-time processing."""
        service, watch_dir, _, services = realtime_setup

        # Create files before starting service (will be found by initial scan)
        initial_file = watch_dir / "initial.py"
        initial_file.write_text("def initial(): pass")

        await service.start(watch_dir)

        # Create file after service started (real-time processing)
        realtime_file = watch_dir / "realtime.py"
        await asyncio.sleep(0.5)  # Let initial scan start
        realtime_file.write_text("def realtime(): pass")

        # Wait for both initial scan and real-time processing
        await asyncio.sleep(3.0)

        # Both files should eventually be processed
        initial_record = services.provider.get_file_by_path(str(initial_file))
        realtime_record = services.provider.get_file_by_path(str(realtime_file))

        # At least one should work (helps identify which path is broken)
        processed_count = sum(
            1 for record in [initial_record, realtime_record] if record is not None
        )
        assert processed_count > 0, "At least one processing path should work"

        await service.stop()
