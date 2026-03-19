"""Test that MCP server initialization is non-blocking (scan runs in background)."""

from __future__ import annotations

import asyncio
import copy
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.daemon.server import ChunkHoundDaemon
from chunkhound.mcp_server.base import MCPServerBase
from chunkhound.mcp_server.status import derive_daemon_status
from chunkhound.services.realtime_indexing_service import RealtimeIndexingService


class ConcreteMCPServer(MCPServerBase):
    """Minimal concrete implementation for testing base class behavior."""

    def _register_tools(self) -> None:
        pass

    async def run(self) -> None:
        pass


class DaemonModeMCPServer(ConcreteMCPServer):
    """Concrete base server variant that reports daemon startup mode."""

    def _realtime_startup_mode(self) -> str:
        return "daemon"


def _collect_startup_breadcrumbs(server: MCPServerBase) -> list[str]:
    breadcrumbs: list[str] = []

    def record(message: str) -> None:
        breadcrumbs.append(message)

    server._startup_log = record  # type: ignore[method-assign]
    server._startup_tracker.set_debug_sink(server._startup_log)
    return breadcrumbs


def _prime_completed_daemon_startup(server: MCPServerBase) -> None:
    server._warm_ready_started_monotonic = time.monotonic()
    server._start_startup_phase("initialize")
    server._complete_startup_phase("initialize")
    server._mark_startup_exposure_ready()
    server._complete_startup()
    server._startup_publish_complete.set()


class TestNonBlockingInitialization:
    """Verify initialization returns before scan completes."""

    @pytest.mark.asyncio
    async def test_initialization_returns_before_scan_completes(self, tmp_path: Path):
        """Verify _scan_progress shows incomplete when initialize() returns."""
        # Create minimal config mock
        config = MagicMock()
        config.database.path = str(tmp_path / "test.db")
        config.embedding = None
        config.llm = None
        config.target_dir = tmp_path

        # Mock create_services to avoid real DB operations
        with patch("chunkhound.mcp_server.base.create_services") as mock_create:
            mock_services = MagicMock()
            mock_services.provider.is_connected = False
            mock_create.return_value = mock_services

            # Mock EmbeddingManager
            with patch("chunkhound.mcp_server.base.EmbeddingManager"):
                server = ConcreteMCPServer(config=config)

                # Initialize and immediately check state
                await server.initialize()

                # Key invariant: scan has NOT completed at this point
                progress = server._scan_progress

                # Verify we haven't completed scanning yet
                # (either still scanning, or scan hasn't started because it's deferred)
                assert progress.get("scan_completed_at") is None, (
                    "Initialization should return before scan completes"
                )

                # Cleanup
                await server.cleanup()

    @pytest.mark.asyncio
    async def test_scan_progress_fields_exist(self, tmp_path: Path):
        """Verify scan_progress dict has expected structure."""
        config = MagicMock()
        config.database.path = str(tmp_path / "test.db")
        config.embedding = None
        config.llm = None
        config.target_dir = tmp_path

        with patch("chunkhound.mcp_server.base.create_services") as mock_create:
            mock_services = MagicMock()
            mock_services.provider.is_connected = False
            mock_create.return_value = mock_services

            with patch("chunkhound.mcp_server.base.EmbeddingManager"):
                server = ConcreteMCPServer(config=config)
                await server.initialize()

                progress = server._scan_progress

                # All expected fields should exist
                assert "is_scanning" in progress
                assert "files_processed" in progress
                assert "chunks_created" in progress
                assert "scan_started_at" in progress
                assert "scan_completed_at" in progress
                assert "realtime" in progress
                assert "event_queue" in progress["realtime"]
                assert "resync" in progress["realtime"]

                await server.cleanup()

    @pytest.mark.asyncio
    async def test_initialized_flag_set_before_scan_starts(self, tmp_path: Path):
        """Verify _initialized is True before background scan begins."""
        config = MagicMock()
        config.database.path = str(tmp_path / "test.db")
        config.embedding = None
        config.llm = None
        config.target_dir = tmp_path

        with patch("chunkhound.mcp_server.base.create_services") as mock_create:
            mock_services = MagicMock()
            mock_services.provider.is_connected = False
            mock_create.return_value = mock_services

            with patch("chunkhound.mcp_server.base.EmbeddingManager"):
                server = ConcreteMCPServer(config=config)

                # Before initialize
                assert not server._initialized

                await server.initialize()

                # After initialize - should be True immediately
                assert server._initialized

                # But scan should not be complete yet
                assert server._scan_progress["scan_completed_at"] is None

                await server.cleanup()

    @pytest.mark.asyncio
    async def test_realtime_start_failure_updates_scan_progress(self, tmp_path: Path):
        """Verify realtime startup task failures are surfaced into realtime status."""
        config = MagicMock()
        config.database.path = str(tmp_path / "test.db")
        config.embedding = None
        config.llm = None
        config.target_dir = tmp_path
        config.indexing.realtime_backend = "watchdog"

        server = ConcreteMCPServer(config=config)
        server.realtime_indexing = MagicMock()
        server.realtime_indexing.monitoring_ready = asyncio.Event()
        server.realtime_indexing._MONITORING_READY_TIMEOUT_SECONDS = 0.01
        server._run_directory_scan = AsyncMock()  # type: ignore[method-assign]

        async def fail_startup() -> None:
            raise RuntimeError("startup exploded")

        monitoring_task = asyncio.create_task(fail_startup())
        await server._coordinated_initial_scan(tmp_path, monitoring_task)

        realtime = server._scan_progress["realtime"]
        assert realtime["service_state"] == "degraded"
        assert "Realtime startup failed" in realtime["last_error"]
        server._run_directory_scan.assert_awaited_once()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_watchman_start_failure_skips_initial_scan(self, tmp_path: Path):
        """Watchman fail-fast startup should not enter the initial scan path."""
        config = MagicMock()
        config.database.path = str(tmp_path / "test.db")
        config.embedding = None
        config.llm = None
        config.target_dir = tmp_path
        config.indexing.realtime_backend = "watchman"

        server = ConcreteMCPServer(config=config)
        server.realtime_indexing = MagicMock()
        server.realtime_indexing.monitoring_ready = asyncio.Event()
        server.realtime_indexing._MONITORING_READY_TIMEOUT_SECONDS = 0.01
        server._run_directory_scan = AsyncMock()  # type: ignore[method-assign]

        async def fail_startup() -> None:
            raise RuntimeError("watchman startup exploded")

        monitoring_task = asyncio.create_task(fail_startup())
        await server._coordinated_initial_scan(tmp_path, monitoring_task)

        realtime = server._scan_progress["realtime"]
        assert realtime["service_state"] == "degraded"
        assert "Realtime startup failed" in realtime["last_error"]
        server._run_directory_scan.assert_not_awaited()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_initial_scan_waits_for_daemon_publish_completion(
        self, tmp_path: Path
    ) -> None:
        """Daemon-mode initial scans must wait until daemon publish completes."""
        config = MagicMock()
        config.database.path = str(tmp_path / "test.db")
        config.embedding = None
        config.llm = None
        config.target_dir = tmp_path
        config.indexing.realtime_backend = "watchdog"

        server = DaemonModeMCPServer(config=config)
        server.realtime_indexing = MagicMock()
        server.realtime_indexing.monitoring_ready = asyncio.Event()
        server.realtime_indexing.monitoring_ready.set()
        server.realtime_indexing._MONITORING_READY_TIMEOUT_SECONDS = 0.01
        server._run_directory_scan = AsyncMock()  # type: ignore[method-assign]

        async def never_finishes() -> None:
            await asyncio.Event().wait()

        monitoring_task = asyncio.create_task(never_finishes())
        with patch(
            "chunkhound.mcp_server.base.asyncio.sleep",
            AsyncMock(return_value=None),
        ):
            scan_task = asyncio.create_task(
                server._coordinated_initial_scan(tmp_path, monitoring_task)
            )
            await asyncio.sleep(0)
            server._run_directory_scan.assert_not_awaited()  # type: ignore[attr-defined]

            server._startup_publish_complete.set()
            await asyncio.wait_for(scan_task, timeout=1.0)

        server._run_directory_scan.assert_awaited_once_with(  # type: ignore[attr-defined]
            tmp_path,
            trigger="initial",
        )
        monitoring_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await monitoring_task

    @pytest.mark.asyncio
    async def test_deferred_start_failure_preserves_configured_backend(
        self, tmp_path: Path
    ):
        """Verify pre-service startup failures keep the configured backend in status."""
        config = MagicMock()
        config.database.path = str(tmp_path / "test.db")
        config.embedding = None
        config.llm = None
        config.target_dir = tmp_path
        config.indexing.realtime_backend = "polling"

        server = ConcreteMCPServer(config=config)
        server.services = MagicMock()
        server.services.provider.is_connected = False
        server.services.provider.connect.side_effect = RuntimeError("connect exploded")

        await server._deferred_connect_and_start(tmp_path)

        realtime = server._scan_progress["realtime"]
        assert realtime["configured_backend"] == "polling"
        assert realtime["service_state"] == "degraded"
        assert "Deferred connect/start failed" in realtime["last_error"]
        assert realtime["startup"]["state"] == "failed"
        assert realtime["startup"]["phases"]["db_connect"]["state"] == "failed"
        assert realtime["startup"]["phases"]["realtime_start"]["state"] == (
            "uninitialized"
        )
        assert realtime["startup"]["last_error"].startswith(
            "Deferred connect/start failed:"
        )
        assert realtime["startup"]["total_duration_seconds"] is not None

    @pytest.mark.asyncio
    async def test_watchman_config_seeds_realtime_status_surface(
        self, tmp_path: Path
    ) -> None:
        """Watchman config should pre-seed daemon status with operator fields."""
        config = MagicMock()
        config.database.path = str(tmp_path / "test.db")
        config.embedding = None
        config.llm = None
        config.target_dir = tmp_path
        config.indexing.realtime_backend = "watchman"

        with patch("chunkhound.mcp_server.base.create_services") as mock_create:
            mock_services = MagicMock()
            mock_services.provider.is_connected = False
            mock_create.return_value = mock_services

            with patch("chunkhound.mcp_server.base.EmbeddingManager"):
                server = ConcreteMCPServer(config=config)
                await server.initialize()

                realtime = server._scan_progress["realtime"]
                assert realtime["configured_backend"] == "watchman"
                assert realtime["live_indexing_state"] == "uninitialized"
                assert (
                    realtime["live_indexing_hint"]
                    == "Live indexing monitoring is not ready yet."
                )
                assert realtime["watchman_sidecar_state"] == "uninitialized"
                assert realtime["watchman_connection_state"] == "uninitialized"
                assert realtime["watchman_subscription_count"] == 0
                assert realtime["watchman_subscription_names"] == []
                assert realtime["watchman_scopes"] == []
                assert realtime["startup"]["mode"] == "stdio"
                assert realtime["startup"]["exposure_ready_at"] is None
                assert realtime["startup"]["phases"]["initialize"]["state"] in {
                    "running",
                    "completed",
                }
                assert realtime["startup"]["phases"]["watchman_sidecar_start"] == {
                    "state": "uninitialized",
                    "started_at": None,
                    "completed_at": None,
                    "duration_seconds": None,
                }
                assert realtime["pipeline"] == {
                    "last_source_event_at": None,
                    "last_source_event_type": None,
                    "last_source_event_path": None,
                    "last_accepted_event_at": None,
                    "last_accepted_event_type": None,
                    "last_accepted_event_path": None,
                    "last_processing_started_at": None,
                    "last_processing_started_path": None,
                    "last_processing_completed_at": None,
                    "last_processing_completed_path": None,
                    "filtered_event_count": 0,
                    "suppressed_duplicate_count": 0,
                    "translation_error_count": 0,
                    "processing_error_count": 0,
                    "stall_threshold_seconds": 30.0,
                }
                assert realtime["event_pressure"] == {
                    "state": "idle",
                    "sample_path": None,
                    "sample_scope": None,
                    "sample_event_type": None,
                    "events_in_window": 0,
                    "coalesced_updates": 0,
                    "window_seconds": 30.0,
                    "last_observed_at": None,
                }
                assert realtime["watchman_loss_of_sync"] == {
                    "count": 0,
                    "fresh_instance_count": 0,
                    "recrawl_count": 0,
                    "disconnect_count": 0,
                    "last_reason": None,
                    "last_at": None,
                    "last_details": None,
                }
                assert realtime["watchman_reconnect"] == {
                    "state": "idle",
                    "attempt_count": 0,
                    "max_attempts": 3,
                    "retry_delay_seconds": 1.0,
                    "last_started_at": None,
                    "last_completed_at": None,
                    "last_error": None,
                    "last_result": None,
                }

                await server.cleanup()

    @pytest.mark.asyncio
    async def test_watchman_startup_barrier_raises_recorded_failure(
        self, tmp_path: Path
    ) -> None:
        """Watchman daemon startup should fail fast when startup already degraded."""
        config = MagicMock()
        config.database.path = str(tmp_path / "test.db")
        config.embedding = None
        config.llm = None
        config.target_dir = tmp_path
        config.indexing.realtime_backend = "watchman"

        server = ConcreteMCPServer(config=config)
        server._deferred_start_task = asyncio.create_task(asyncio.sleep(0))
        server._startup_failure_message = "Watchman sidecar startup failed: boom"

        with pytest.raises(RuntimeError, match="Watchman sidecar startup failed: boom"):
            await server.await_startup_barrier()

        startup = server._scan_progress["realtime"]["startup"]
        assert startup["state"] == "failed"
        assert startup["last_error"] == "Watchman sidecar startup failed: boom"
        assert startup["phases"]["startup_barrier"]["state"] == "failed"
        assert startup["phases"]["startup_barrier"]["completed_at"] is not None

    @pytest.mark.asyncio
    async def test_daemon_run_records_publish_exposure_ready_timestamp(
        self, tmp_path: Path
    ) -> None:
        """Daemon-mode startup timing should expose the daemon publish seam."""
        config = MagicMock()
        config.database.path = str(tmp_path / "test.db")
        config.embedding = None
        config.llm = None
        config.target_dir = tmp_path
        config.indexing.realtime_backend = "watchdog"
        args = MagicMock()
        args.path = str(tmp_path)

        daemon = ChunkHoundDaemon(
            config=config,
            args=args,
            socket_path="tcp:127.0.0.1:0",
            project_dir=tmp_path,
        )
        daemon.initialize = AsyncMock()
        daemon.await_startup_barrier = AsyncMock()
        daemon.cleanup = AsyncMock()
        daemon._client_manager.poll_pids = AsyncMock(return_value=None)
        daemon._discovery.write_lock = MagicMock()
        daemon._discovery.read_lock = MagicMock(return_value={"pid": os.getpid()})
        daemon._discovery.write_registry_entry = MagicMock()

        class _FakeServer:
            async def __aenter__(self):
                daemon._shutdown_event.set()
                return self

            async def __aexit__(self, exc_type, exc, tb):
                del exc_type, exc, tb
                return False

        with patch(
            "chunkhound.daemon.server.ipc.create_server",
            AsyncMock(return_value=(_FakeServer(), "tcp:127.0.0.1:7788")),
        ):
            await daemon.run()

        startup = daemon._scan_progress["realtime"]["startup"]
        assert startup["mode"] == "daemon"
        assert startup["state"] == "completed"
        assert startup["exposure_ready_at"] is not None
        assert startup["phases"]["daemon_publish"]["state"] == "completed"
        assert startup["phases"]["daemon_publish"]["completed_at"] is not None

    def test_daemon_startup_phase_breadcrumbs_emit_to_stderr(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Daemon startup phase breadcrumbs should reach the daemon stderr log path."""
        config = MagicMock()
        config.database.path = str(tmp_path / "test.db")
        config.embedding = None
        config.llm = None
        config.target_dir = tmp_path
        config.indexing.realtime_backend = "watchdog"
        args = MagicMock()
        args.path = str(tmp_path)

        daemon = ChunkHoundDaemon(
            config=config,
            args=args,
            socket_path="tcp:127.0.0.1:0",
            project_dir=tmp_path,
        )

        daemon._start_startup_phase("daemon_publish")
        daemon._complete_startup_phase("daemon_publish")

        captured = capsys.readouterr()
        assert "[startup] startup: phase started: daemon_publish" in captured.err
        assert "[startup] startup: phase completed: daemon_publish" in captured.err

    @pytest.mark.asyncio
    async def test_run_directory_scan_surfaces_reconciliation_cleanup_failures(
        self, tmp_path: Path
    ) -> None:
        """Directory scans should record cleanup failures in daemon status."""
        config = MagicMock()
        config.database.path = str(tmp_path / "test.db")
        config.embedding = None
        config.llm = None
        config.target_dir = tmp_path
        config.indexing.include = ["**/*.py"]
        config.indexing.exclude = []
        config.indexing.config_file_size_threshold_kb = 20

        server = ConcreteMCPServer(config=config)
        server.services = MagicMock()
        server.services.indexing_coordinator.process_directory = AsyncMock(
            return_value={
                "status": "error",
                "error": (
                    "Storage reconciliation cleanup failed: "
                    "database invalidated during orphan cleanup"
                ),
            }
        )

        with pytest.raises(RuntimeError, match="Storage reconciliation cleanup failed"):
            await server._run_directory_scan(
                tmp_path,
                trigger="realtime_resync",
                reason="realtime_loss_of_sync",
                no_embeddings=True,
            )

        assert (
            "Storage reconciliation cleanup failed"
            in server._scan_progress["scan_error"]
        )

        daemon_status = derive_daemon_status(server._scan_progress)
        assert daemon_status["status"] == "degraded"
        assert daemon_status["query_ready"] is False
        assert (
            "Storage reconciliation cleanup failed"
            in daemon_status["scan_progress"]["scan_error"]
        )

    @pytest.mark.asyncio
    async def test_realtime_resync_uses_no_embedding_scan_and_single_embed_followup(
        self, tmp_path: Path
    ) -> None:
        """Realtime resyncs should rescan without embeddings, then embed once."""
        config = MagicMock()
        config.database.path = str(tmp_path / "test.db")
        config.embedding = None
        config.embeddings_disabled = False
        config.llm = None
        config.target_dir = tmp_path
        config.indexing.exclude = ["*.lock", "node_modules/**"]

        server = ConcreteMCPServer(config=config)
        server._scan_target_path = tmp_path
        server._run_directory_scan = AsyncMock()  # type: ignore[method-assign]
        server.services = MagicMock()
        server.services.indexing_coordinator.generate_missing_embeddings = AsyncMock(
            return_value={"status": "up_to_date", "generated": 0}
        )

        result = await server._request_realtime_resync(
            "realtime_loss_of_sync",
            {"backend": "watchman", "loss_of_sync_reason": "disconnect"},
        )

        server._run_directory_scan.assert_awaited_once_with(  # type: ignore[attr-defined]
            tmp_path,
            trigger="realtime_resync",
            reason="realtime_loss_of_sync",
            no_embeddings=True,
        )
        server.services.indexing_coordinator.generate_missing_embeddings.assert_awaited_once_with(
            exclude_patterns=["*.lock", "node_modules/**"]
        )
        assert result == {"status": "up_to_date", "generated": 0}

    @pytest.mark.asyncio
    async def test_realtime_resync_skips_embed_followup_when_embeddings_disabled(
        self, tmp_path: Path
    ) -> None:
        """Explicit no-embeddings mode should complete resyncs in regex-only mode."""
        config = MagicMock()
        config.database.path = str(tmp_path / "test.db")
        config.embedding = None
        config.embeddings_disabled = True
        config.llm = None
        config.target_dir = tmp_path
        config.indexing.exclude = ["*.lock"]

        server = ConcreteMCPServer(config=config)
        server._scan_target_path = tmp_path
        server._run_directory_scan = AsyncMock()  # type: ignore[method-assign]
        server.services = MagicMock()
        server.services.indexing_coordinator.generate_missing_embeddings = AsyncMock()

        result = await server._request_realtime_resync(
            "realtime_loss_of_sync",
            {"backend": "watchman", "loss_of_sync_reason": "fresh_instance"},
        )

        server._run_directory_scan.assert_awaited_once_with(  # type: ignore[attr-defined]
            tmp_path,
            trigger="realtime_resync",
            reason="realtime_loss_of_sync",
            no_embeddings=True,
        )
        server.services.indexing_coordinator.generate_missing_embeddings.assert_not_awaited()
        assert result == {
            "status": "complete",
            "generated": 0,
            "message": "Embeddings explicitly disabled",
        }

    @pytest.mark.asyncio
    async def test_realtime_resync_embed_error_status_keeps_realtime_degraded(
        self, tmp_path: Path
    ) -> None:
        """Realtime loss-of-sync should stay degraded on embed follow-up error."""
        config = MagicMock()
        config.database.path = str(tmp_path / "test.db")
        config.embedding = None
        config.embeddings_disabled = False
        config.llm = None
        config.target_dir = tmp_path
        config.indexing.exclude = ["*.lock"]

        server = ConcreteMCPServer(config=config)
        server._scan_target_path = tmp_path
        server._run_directory_scan = AsyncMock()  # type: ignore[method-assign]
        server.services = MagicMock()
        server.services.indexing_coordinator.generate_missing_embeddings = AsyncMock(
            return_value={
                "status": "error",
                "error": "embedding backend unavailable",
                "generated": 0,
            }
        )

        service = RealtimeIndexingService(
            server.services,
            config,
            status_callback=server._update_realtime_status,
            resync_callback=server._request_realtime_resync,
        )

        await service.request_resync(
            "realtime_loss_of_sync",
            {"backend": "watchman", "loss_of_sync_reason": "disconnect"},
        )
        await asyncio.sleep(service._RESYNC_DEBOUNCE_SECONDS + 0.1)

        realtime = server._scan_progress["realtime"]
        assert realtime["service_state"] == "degraded"
        assert realtime["resync"]["needs_resync"] is True
        assert realtime["resync"]["performed_count"] == 0
        assert (
            realtime["resync"]["last_error"]
            == "Resync callback reported error status: embedding backend unavailable"
        )
        assert (
            realtime["last_error"]
            == "Realtime resync failed: Resync callback reported error status: "
            "embedding backend unavailable"
        )

    @pytest.mark.asyncio
    async def test_realtime_resync_disabled_embeddings_clear_stale_state(
        self, tmp_path: Path
    ) -> None:
        """Explicit no-embeddings mode should not leave realtime resync latched."""
        config = MagicMock()
        config.database.path = str(tmp_path / "test.db")
        config.embedding = None
        config.embeddings_disabled = True
        config.llm = None
        config.target_dir = tmp_path
        config.indexing.exclude = ["*.lock"]

        server = ConcreteMCPServer(config=config)
        server._scan_target_path = tmp_path
        server._run_directory_scan = AsyncMock()  # type: ignore[method-assign]
        server.services = MagicMock()
        server.services.indexing_coordinator.generate_missing_embeddings = AsyncMock()

        service = RealtimeIndexingService(
            server.services,
            config,
            status_callback=server._update_realtime_status,
            resync_callback=server._request_realtime_resync,
        )

        await service.request_resync(
            "realtime_loss_of_sync",
            {"backend": "watchman", "loss_of_sync_reason": "fresh_instance"},
        )
        await asyncio.sleep(service._RESYNC_DEBOUNCE_SECONDS + 0.1)

        realtime = server._scan_progress["realtime"]
        assert realtime["resync"]["needs_resync"] is False
        assert realtime["resync"]["performed_count"] == 1
        assert realtime["resync"]["last_error"] is None
        assert realtime["last_error"] is None
        server.services.indexing_coordinator.generate_missing_embeddings.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_startup_window_fresh_instance_logs_double_work_and_summary(
        self, tmp_path: Path
    ) -> None:
        """Startup-window fresh-instance recovery should log both passes and summary."""
        config = MagicMock()
        config.database.path = str(tmp_path / "test.db")
        config.embedding = None
        config.embeddings_disabled = False
        config.llm = None
        config.target_dir = tmp_path
        config.indexing.realtime_backend = "watchman"
        config.indexing.exclude = []

        server = DaemonModeMCPServer(config=config)
        breadcrumbs = _collect_startup_breadcrumbs(server)
        _prime_completed_daemon_startup(server)
        server._scan_target_path = tmp_path
        server.services = MagicMock()
        server.services.indexing_coordinator.generate_missing_embeddings = AsyncMock(
            return_value={"status": "complete", "generated": 2}
        )
        server.realtime_indexing = MagicMock()
        server.realtime_indexing.monitoring_ready = asyncio.Event()
        server.realtime_indexing.monitoring_ready.set()
        server.realtime_indexing._MONITORING_READY_TIMEOUT_SECONDS = 0.01

        async def never_finishes() -> None:
            await asyncio.Event().wait()

        monitoring_task = asyncio.create_task(never_finishes())
        with patch(
            "chunkhound.mcp_server.base.DirectoryIndexingService.process_directory",
            AsyncMock(
                side_effect=[
                    SimpleNamespace(files_processed=3, chunks_created=9),
                    SimpleNamespace(files_processed=4, chunks_created=12),
                ]
            ),
        ), patch(
            "chunkhound.mcp_server.base.asyncio.sleep",
            AsyncMock(return_value=None),
        ):
            await server._request_realtime_resync(
                "realtime_loss_of_sync",
                {"backend": "watchman", "loss_of_sync_reason": "fresh_instance"},
            )
            await asyncio.wait_for(
                server._coordinated_initial_scan(tmp_path, monitoring_task),
                timeout=1.0,
            )

        monitoring_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await monitoring_task

        assert any(
            "warm-ready: realtime resync requested "
            "reason=realtime_loss_of_sync" in message
            for message in breadcrumbs
        )
        assert any(
            "warm-ready: realtime resync directory scan starting" in message
            for message in breadcrumbs
        )
        assert any(
            "warm-ready: realtime resync embedding follow-up starting" in message
            for message in breadcrumbs
        )
        assert any(
            "warm-ready: initial scan coordination monitoring_ready_to_publish="
            in message
            for message in breadcrumbs
        )
        assert any(
            "warm-ready: initial scan completed total=" in message
            for message in breadcrumbs
        )
        assert any(
            "warm-ready: startup paid both fresh-instance resync and initial scan "
            "back-to-back" in message
            for message in breadcrumbs
        )
        assert any(
            "warm-ready: summary " in message
            and "blocking_startup=" in message
            and "fresh_instance_resync=" in message
            and "initial_scan=" in message
            and "warm_ready=" in message
            for message in breadcrumbs
        )

    @pytest.mark.asyncio
    async def test_startup_warm_ready_summary_emits_without_resync(
        self, tmp_path: Path
    ) -> None:
        """Warm-ready summary should still appear when startup skips resync."""
        config = MagicMock()
        config.database.path = str(tmp_path / "test.db")
        config.embedding = None
        config.llm = None
        config.target_dir = tmp_path
        config.indexing.realtime_backend = "watchman"
        config.indexing.exclude = []

        server = DaemonModeMCPServer(config=config)
        breadcrumbs = _collect_startup_breadcrumbs(server)
        _prime_completed_daemon_startup(server)
        server.services = MagicMock()
        server.realtime_indexing = MagicMock()
        server.realtime_indexing.monitoring_ready = asyncio.Event()
        server.realtime_indexing.monitoring_ready.set()
        server.realtime_indexing._MONITORING_READY_TIMEOUT_SECONDS = 0.01

        async def never_finishes() -> None:
            await asyncio.Event().wait()

        monitoring_task = asyncio.create_task(never_finishes())
        with patch(
            "chunkhound.mcp_server.base.DirectoryIndexingService.process_directory",
            AsyncMock(
                return_value=SimpleNamespace(files_processed=5, chunks_created=15)
            ),
        ), patch(
            "chunkhound.mcp_server.base.asyncio.sleep",
            AsyncMock(return_value=None),
        ):
            await asyncio.wait_for(
                server._coordinated_initial_scan(tmp_path, monitoring_task),
                timeout=1.0,
            )

        monitoring_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await monitoring_task

        assert any(
            "warm-ready: initial scan coordination monitoring_ready_to_publish="
            in message
            for message in breadcrumbs
        )
        assert any("warm-ready: summary " in message for message in breadcrumbs)
        assert not any(
            "startup paid both fresh-instance resync and initial scan back-to-back"
            in message
            for message in breadcrumbs
        )

    @pytest.mark.asyncio
    async def test_startup_timing_breadcrumbs_do_not_mutate_public_startup_status(
        self, tmp_path: Path
    ) -> None:
        """Warm-ready breadcrumbs must not change the public startup payload."""
        config = MagicMock()
        config.database.path = str(tmp_path / "test.db")
        config.embedding = None
        config.embeddings_disabled = False
        config.llm = None
        config.target_dir = tmp_path
        config.indexing.realtime_backend = "watchman"
        config.indexing.exclude = []

        server = DaemonModeMCPServer(config=config)
        _collect_startup_breadcrumbs(server)
        _prime_completed_daemon_startup(server)
        server._scan_target_path = tmp_path
        server.services = MagicMock()
        server.services.indexing_coordinator.generate_missing_embeddings = AsyncMock(
            return_value={"status": "complete", "generated": 1}
        )
        server.realtime_indexing = MagicMock()
        server.realtime_indexing.monitoring_ready = asyncio.Event()
        server.realtime_indexing.monitoring_ready.set()
        server.realtime_indexing._MONITORING_READY_TIMEOUT_SECONDS = 0.01

        startup_before = copy.deepcopy(server._scan_progress["realtime"]["startup"])

        async def never_finishes() -> None:
            await asyncio.Event().wait()

        monitoring_task = asyncio.create_task(never_finishes())
        with patch(
            "chunkhound.mcp_server.base.DirectoryIndexingService.process_directory",
            AsyncMock(
                side_effect=[
                    SimpleNamespace(files_processed=2, chunks_created=6),
                    SimpleNamespace(files_processed=2, chunks_created=6),
                ]
            ),
        ), patch(
            "chunkhound.mcp_server.base.asyncio.sleep",
            AsyncMock(return_value=None),
        ):
            await server._request_realtime_resync(
                "realtime_loss_of_sync",
                {"backend": "watchman", "loss_of_sync_reason": "fresh_instance"},
            )
            await asyncio.wait_for(
                server._coordinated_initial_scan(tmp_path, monitoring_task),
                timeout=1.0,
            )

        monitoring_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await monitoring_task

        assert server._scan_progress["realtime"]["startup"] == startup_before
