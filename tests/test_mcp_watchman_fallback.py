"""Test Watchman→polling fallback gating at the MCP layer.

Covers three scenarios:
1. Explicit Watchman → sidecar failure raises RuntimeError (fail-fast)
2. Install-default Watchman → sidecar failure falls back to polling
3. Done callback captures realtime task failures (exception, cancellation, success)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.mcp_server.base import MCPServerBase
from chunkhound.services.realtime.adapters.watchman import WatchmanRealtimeAdapter
from chunkhound.services.realtime.context import RealtimeServiceContext


class _TestServer(MCPServerBase):
    """Minimal concrete implementation for testing base class behavior."""

    def _register_tools(self) -> None:
        pass

    async def run(self) -> None:
        pass


class _MockWatchmanService:
    """Minimal mock service for constructing a RealtimeServiceContext.

    RealtimeServiceContext delegates all attribute access to self._service,
    so we provide callable stubs for every method the adapter touches.
    """

    def __init__(self, resolution: str = "install_default") -> None:
        self.config = MagicMock()
        self.config.target_dir = Path("/tmp/test")
        self._configured_backend_resolution = resolution

        # Phase tracking — direct callable mocks
        self._start_startup_phase = MagicMock()
        self._complete_startup_phase = MagicMock()
        self._fail_startup_phase = MagicMock()
        self._current_startup_phase = MagicMock(return_value=None)

        # Watchman monitoring state
        self._clear_watchman_monitoring_state = MagicMock()
        self._set_effective_backend = MagicMock()
        self._start_polling_backend = AsyncMock()

        # Logging / error
        self._debug = MagicMock()
        self._set_error = MagicMock()


def _make_watchman_context(
    resolution: str = "install_default",
) -> tuple[RealtimeServiceContext, _MockWatchmanService]:
    """Build a RealtimeServiceContext wired to a controllable mock service.

    Returns (context, service) so tests can assert on the service stubs.
    """
    svc = _MockWatchmanService(resolution=resolution)
    ctx = RealtimeServiceContext(
        service=svc,
        sidecar_factory=MagicMock(),
        session_factory=MagicMock(),
        nested_mount_discoverer=MagicMock(return_value=[]),
        junction_scope_discoverer=MagicMock(return_value=[]),
        scope_plan_builder=MagicMock(),
        subscription_name_builder=MagicMock(),
        subscription_names_builder=MagicMock(),
    )
    return ctx, svc


class TestWatchmanSidecarFallbackGating:
    """Verify explicit vs install-default fallback behavior."""

    @pytest.mark.asyncio
    async def test_explicit_watchman_raises_on_sidecar_failure(self):
        """When configured_backend_resolution is 'explicit', sidecar failure raises RuntimeError."""
        ctx, svc = _make_watchman_context(resolution="explicit")

        adapter = WatchmanRealtimeAdapter(ctx)
        adapter._sidecar = MagicMock()
        adapter._sidecar.start = AsyncMock(side_effect=RuntimeError("Sidecar failed"))

        with pytest.raises(RuntimeError) as exc_info:
            await adapter._establish_monitoring(
                Path("/tmp/test"),
                asyncio.get_running_loop(),
                phase="startup",
            )

        assert "Sidecar failed" in str(exc_info.value)
        assert "explicitly configured" in str(exc_info.value)
        # Verify fallback was NOT called
        svc._start_polling_backend.assert_not_called()
        svc._set_effective_backend.assert_not_called()

    @pytest.mark.asyncio
    async def test_install_default_falls_back_on_sidecar_failure(self):
        """When configured_backend_resolution is 'install_default', sidecar failure falls back to polling."""
        ctx, svc = _make_watchman_context(resolution="install_default")

        adapter = WatchmanRealtimeAdapter(ctx)
        adapter._sidecar = MagicMock()
        adapter._sidecar.start = AsyncMock(side_effect=RuntimeError("Sidecar crashed"))

        # Should NOT raise — should fall back
        await adapter._establish_monitoring(
            Path("/tmp/test"),
            asyncio.get_running_loop(),
            phase="startup",
        )

        # Verify fallback was called (reason is positional arg #1)
        svc._start_polling_backend.assert_called_once()
        call_args = svc._start_polling_backend.call_args[0]
        assert len(call_args) >= 2
        assert "Sidecar crashed" in call_args[1]
        svc._set_effective_backend.assert_called_once_with("polling")

    @pytest.mark.asyncio
    async def test_explicit_watchman_error_propagates_via_start(self):
        """Verify that start() method wraps the error correctly for explicit Watchman."""
        ctx, svc = _make_watchman_context(resolution="explicit")

        adapter = WatchmanRealtimeAdapter(ctx)
        adapter._sidecar = MagicMock()
        adapter._sidecar.start = AsyncMock(side_effect=RuntimeError("Sidecar failed"))

        with pytest.raises(RuntimeError) as exc_info:
            await adapter.start(
                Path("/tmp/test"),
                asyncio.get_running_loop(),
            )
        assert "Sidecar failed" in str(exc_info.value)
        svc._set_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_sidecar_failure_records_failed_phase(self):
        """The failed sidecar startup phase is recorded via fail_startup_phase."""
        ctx, svc = _make_watchman_context(resolution="explicit")

        adapter = WatchmanRealtimeAdapter(ctx)
        adapter._sidecar = MagicMock()
        adapter._sidecar.start = AsyncMock(side_effect=RuntimeError("Kaboom"))

        with pytest.raises(RuntimeError):
            await adapter._establish_monitoring(
                Path("/tmp/test"),
                asyncio.get_running_loop(),
                phase="startup",
            )

        # Verify the phase was failed with appropriate error message
        svc._fail_startup_phase.assert_called_once_with(
            "watchman_sidecar_start",
            "Watchman sidecar startup failed: Kaboom",
        )

    @pytest.mark.asyncio
    async def test_effective_backend_set_to_polling_after_fallback(self):
        """After install-default fallback, effective_backend is 'polling'."""
        ctx, svc = _make_watchman_context(resolution="install_default")

        adapter = WatchmanRealtimeAdapter(ctx)
        adapter._sidecar = MagicMock()
        adapter._sidecar.start = AsyncMock(side_effect=RuntimeError("Boom"))

        await adapter._establish_monitoring(
            Path("/tmp/test"),
            asyncio.get_running_loop(),
            phase="startup",
        )

        svc._set_effective_backend.assert_called_once_with("polling")


class TestRealtimeStartTaskDoneCallback:
    """Verify _handle_realtime_start_task_done captures realtime task failures."""

    @pytest.mark.asyncio
    async def test_done_callback_captures_failure(self):
        """Verify _handle_realtime_start_task_done captures exception state."""
        config = MagicMock()
        config.database.path = "/tmp/test.db"
        config.embedding = None
        config.llm = None
        config.target_dir = Path("/tmp")

        with patch("chunkhound.mcp_server.base.create_services") as mock_create:
            mock_services = MagicMock()
            mock_services.provider.is_connected = False
            mock_create.return_value = mock_services
            with patch("chunkhound.mcp_server.base.EmbeddingManager"), patch.object(
                MCPServerBase,
                "_deferred_connect_and_start",
                AsyncMock(return_value=None),
            ):
                server = _TestServer(config=config)
                await server.initialize()

                # Create a task that fails
                async def failing_task() -> None:
                    raise RuntimeError("Watchman sidecar crashed")

                task = asyncio.create_task(failing_task())
                task.add_done_callback(server._handle_realtime_start_task_done)

                # Let the task run and fail.
                try:
                    await task
                except RuntimeError:
                    pass

                # Check that failure was captured
                assert server._startup_failure_message is not None
                assert "Watchman sidecar crashed" in server._startup_failure_message

                await server.cleanup()

    @pytest.mark.asyncio
    async def test_done_callback_captures_cancellation(self):
        """Cancellation is captured via _set_startup_failure."""
        config = MagicMock()
        config.database.path = "/tmp/test.db"
        config.embedding = None
        config.llm = None
        config.target_dir = Path("/tmp")

        with patch("chunkhound.mcp_server.base.create_services") as mock_create:
            mock_services = MagicMock()
            mock_services.provider.is_connected = False
            mock_create.return_value = mock_services
            with patch("chunkhound.mcp_server.base.EmbeddingManager"), patch.object(
                MCPServerBase,
                "_deferred_connect_and_start",
                AsyncMock(return_value=None),
            ):
                server = _TestServer(config=config)
                await server.initialize()

                # Mark startup tracker as running so cancellation path is taken
                server._startup_tracker.start_phase("realtime_start")

                # Create a cancellable task, cancel it, then apply the callback
                async def never_ending() -> None:
                    await asyncio.Event().wait()

                task = asyncio.create_task(never_ending())
                task.add_done_callback(server._handle_realtime_start_task_done)

                # Cancel before it completes
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

                # The cancellation path calls _set_startup_failure
                assert server._startup_failure_message is not None
                assert "cancelled" in server._startup_failure_message.lower()

                await server.cleanup()

    @pytest.mark.asyncio
    async def test_done_callback_success_noop(self):
        """Successful completion in stdio mode calls _complete_startup.

        In stdio mode (the default for the base class), a successful realtime
        start task triggers _complete_startup() rather than storing a failure.
        """
        config = MagicMock()
        config.database.path = "/tmp/test.db"
        config.embedding = None
        config.llm = None
        config.target_dir = Path("/tmp")

        with patch("chunkhound.mcp_server.base.create_services") as mock_create:
            mock_services = MagicMock()
            mock_services.provider.is_connected = False
            mock_create.return_value = mock_services
            with patch("chunkhound.mcp_server.base.EmbeddingManager"), patch.object(
                MCPServerBase,
                "_deferred_connect_and_start",
                AsyncMock(return_value=None),
            ):
                server = _TestServer(config=config)
                await server.initialize()

                # Create a task that succeeds
                async def successful_task() -> None:
                    return

                task = asyncio.create_task(successful_task())
                task.add_done_callback(server._handle_realtime_start_task_done)

                await task

                # No failure message should be stored
                assert server._startup_failure_message is None

                # The startup tracker should have recorded completion
                snapshot = server._startup_tracker.snapshot()
                assert snapshot["state"] == "completed"

                await server.cleanup()
