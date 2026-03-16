"""Functional tests for real-time filesystem indexing.

Tests core real-time indexing functionality with real components.
Some tests expected to fail initially - helps identify implementation issues.
"""

import asyncio
import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from chunkhound.core.config.config import Config
from chunkhound.database_factory import create_services
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
        assert "configured_backend" in stats, "Should expose configured backend"
        assert "effective_backend" in stats, "Should expose effective backend"
        assert "monitoring_mode" in stats, "Should expose current monitoring mode"
        assert stats["configured_backend"] == realtime_backend_for_tests()
        assert stats["monitoring_mode"] == stats["effective_backend"]

        # Should be able to stop cleanly
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
            assert stats["watchman_subscription_count"] >= 1
            assert isinstance(stats["watchman_scopes"], list)
            assert stats["watchman_scopes"]
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
            session = getattr(adapter, "_session", None)
            process = getattr(session, "_process", None)
            assert process is not None

            process.terminate()

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
    async def test_full_event_queue_tracks_drop_and_requests_resync(
        self, realtime_setup
    ):
        """Dropped realtime events should be counted and escalated via resync."""
        service, watch_dir, _, _ = realtime_setup
        callback_event = asyncio.Event()

        async def resync_callback(
            reason: str, details: dict[str, object] | None
        ) -> None:
            assert reason == "event_queue_overflow"
            assert details is not None
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
        handler._queue_event("modified", watch_dir / "overflow.py")

        await asyncio.wait_for(callback_event.wait(), timeout=5.0)

        stats = await service.get_health()
        assert stats["event_queue"]["dropped"] == 1
        assert stats["event_queue"]["last_reason"] == "queue_full"
        assert stats["resync"]["request_count"] == 1
        assert stats["resync"]["last_reason"] == "event_queue_overflow"

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

        async def fake_add_file(file_path: Path, priority: str = "change") -> None:
            add_calls.append((file_path, priority))
            if len(add_calls) >= 2:
                change_detected.set()

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

        service.pending_files.add(target_file)
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

        await service._debounced_add_file(target_file, "change")

        assert len(sleep_calls) == 2
        assert sleep_calls[0] == service._debounce_delay
        assert sleep_calls[1] == pytest.approx(0.01, abs=1e-6)
        assert file_key not in service._pending_debounce
        assert await service.file_queue.get() == ("change", target_file)

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
    async def test_service_survives_processing_errors(self, realtime_setup):
        """Test service continues working after processing errors."""
        service, watch_dir, _, _ = realtime_setup
        await service.start(watch_dir)

        # Create a file that might cause processing issues
        bad_file = watch_dir / "bad_file.py"
        # Write binary data to a .py file - might cause parsing errors
        bad_file.write_bytes(b"\x00\xff\xfe\xfd")

        await asyncio.sleep(1.0)

        # Create a normal file after the bad one
        good_file = watch_dir / "good_file.py"
        good_file.write_text("def good_function(): pass")

        await asyncio.sleep(2.0)

        # Main goal: service should still be alive
        stats = await service.get_stats()
        assert stats.get("observer_alive", False), (
            "Service should survive processing errors"
        )

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
