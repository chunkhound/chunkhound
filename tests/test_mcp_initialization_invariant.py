"""Test that MCP server initialization is non-blocking (scan runs in background)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.mcp_server.base import MCPServerBase


class ConcreteMCPServer(MCPServerBase):
    """Minimal concrete implementation for testing base class behavior."""

    def _register_tools(self) -> None:
        pass

    async def run(self) -> None:
        pass


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
                assert realtime["watchman_sidecar_state"] == "uninitialized"
                assert realtime["watchman_connection_state"] == "uninitialized"
                assert realtime["watchman_subscription_count"] == 0
                assert realtime["watchman_loss_of_sync"] == {
                    "count": 0,
                    "fresh_instance_count": 0,
                    "recrawl_count": 0,
                    "disconnect_count": 0,
                    "last_reason": None,
                    "last_at": None,
                    "last_details": None,
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

    @pytest.mark.asyncio
    async def test_realtime_resync_uses_no_embedding_scan_and_single_embed_followup(
        self, tmp_path: Path
    ) -> None:
        """Realtime resyncs should rescan without embeddings, then embed once."""
        config = MagicMock()
        config.database.path = str(tmp_path / "test.db")
        config.embedding = None
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

        await server._request_realtime_resync(
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
