"""Test that await_startup_barrier raises RuntimeError on every failure path."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from chunkhound.mcp_server.base import MCPServerBase


class _TestableMCPServer(MCPServerBase):
    """Concrete subclass for testing MCPServerBase startup barrier behavior."""

    def _register_tools(self) -> None:
        pass

    async def run(self) -> None:
        pass


class TestAwaitStartupBarrier:
    """Verify 8 failure paths raise RuntimeError; 2 success paths return cleanly."""

    @pytest.fixture
    def config(self, tmp_path: Path) -> Any:
        config = MagicMock()
        config.database.path = str(tmp_path / ".chunkhound" / "test.db")
        config.embedding = None
        config.llm = None
        config.indexing = MagicMock()
        config.indexing.realtime_backend = None
        return config

    async def _make_server(self, config: Any) -> _TestableMCPServer:
        """Create a server with the minimum viable init state for barrier tests."""
        server = _TestableMCPServer(config)
        # Ensure _scan_progress has a valid "realtime" key so
        # _set_startup_failure / _record_realtime_failure don't crash.
        server._scan_progress["realtime"] = (
            server._default_realtime_scan_status()
        )
        return server

    # ------------------------------------------------------------------
    # Success paths
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_non_strict_barrier_returns_cleanly(self, config: Any) -> None:
        """When requires_strict_startup_barrier() is False, no RuntimeError."""
        server = await self._make_server(config)
        with patch.object(server, "requires_strict_startup_barrier", return_value=False):
            await server.await_startup_barrier()
            # Should complete without raising

    @pytest.mark.asyncio
    async def test_strict_barrier_success(self, config: Any) -> None:
        """When everything is properly set up, no RuntimeError."""
        server = await self._make_server(config)
        with patch.object(server, "requires_strict_startup_barrier", return_value=True):

            async def _succeed() -> None:
                pass

            server._deferred_start_task = asyncio.create_task(_succeed())
            server._realtime_start_task = asyncio.create_task(_succeed())
            server._startup_failure_message = None
            server.realtime_indexing = MagicMock()
            server.realtime_indexing.monitoring_ready = asyncio.Event()
            server.realtime_indexing.monitoring_ready.set()

            await server.await_startup_barrier()
            # Should complete without raising

    # ------------------------------------------------------------------
    # Failure path 1: _deferred_start_task is None
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_fails_when_deferred_task_none(self, config: Any) -> None:
        """Path 1: deferred_start_task is None."""
        server = await self._make_server(config)
        with patch.object(server, "requires_strict_startup_barrier", return_value=True):
            server._deferred_start_task = None
            with pytest.raises(
                RuntimeError, match="before deferred startup began"
            ):
                await server.await_startup_barrier()

    # ------------------------------------------------------------------
    # Failure path 2: CancelledError on _deferred_start_task
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_fails_when_deferred_task_cancelled(self, config: Any) -> None:
        """Path 2: CancelledError on deferred_start_task.

        The barrier will raise RuntimeError at the deferred-task await
        before ever checking _realtime_start_task, so we leave it None.
        """
        server = await self._make_server(config)
        with patch.object(server, "requires_strict_startup_barrier", return_value=True):

            async def _cancelled() -> None:
                raise asyncio.CancelledError()

            server._deferred_start_task = asyncio.create_task(_cancelled())
            # _realtime_start_task intentionally None – not reached

            with pytest.raises(
                RuntimeError, match="was cancelled before readiness"
            ):
                await server.await_startup_barrier()

    # ------------------------------------------------------------------
    # Failure path 3: _startup_failure_message is set after deferred task
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_fails_on_deferred_startup_failure_message(
        self, config: Any
    ) -> None:
        """Path 3: _startup_failure_message is set after deferred task completes."""
        server = await self._make_server(config)
        with patch.object(server, "requires_strict_startup_barrier", return_value=True):

            async def _succeed() -> None:
                pass

            server._deferred_start_task = asyncio.create_task(_succeed())
            server._startup_failure_message = "DB connection failed"
            # _realtime_start_task intentionally None – not reached

            with pytest.raises(RuntimeError, match="DB connection failed"):
                await server.await_startup_barrier()

    # ------------------------------------------------------------------
    # Failure path 4: _realtime_start_task is None
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_fails_when_realtime_task_none(self, config: Any) -> None:
        """Path 4: _realtime_start_task is None after deferred task succeeds."""
        server = await self._make_server(config)
        with patch.object(server, "requires_strict_startup_barrier", return_value=True):

            async def _succeed() -> None:
                pass

            server._deferred_start_task = asyncio.create_task(_succeed())
            server._startup_failure_message = None
            server._realtime_start_task = None

            with pytest.raises(RuntimeError, match="was never created"):
                await server.await_startup_barrier()

    # ------------------------------------------------------------------
    # Failure path 5: CancelledError on _realtime_start_task
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_fails_when_realtime_task_cancelled(self, config: Any) -> None:
        """Path 5: CancelledError on realtime_start_task."""
        server = await self._make_server(config)
        with patch.object(server, "requires_strict_startup_barrier", return_value=True):

            async def _succeed() -> None:
                pass

            async def _cancelled() -> None:
                raise asyncio.CancelledError()

            server._deferred_start_task = asyncio.create_task(_succeed())
            server._startup_failure_message = None
            server._realtime_start_task = asyncio.create_task(_cancelled())

            with pytest.raises(
                RuntimeError, match="was cancelled before readiness"
            ):
                await server.await_startup_barrier()

    # ------------------------------------------------------------------
    # Failure path 6: Exception on _realtime_start_task
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_fails_when_realtime_task_exception(self, config: Any) -> None:
        """Path 6: Exception raised by realtime_start_task."""
        server = await self._make_server(config)
        with patch.object(server, "requires_strict_startup_barrier", return_value=True):

            async def _succeed() -> None:
                pass

            async def _failing() -> None:
                raise RuntimeError("Watchman socket error")

            server._deferred_start_task = asyncio.create_task(_succeed())
            server._startup_failure_message = None
            server._realtime_start_task = asyncio.create_task(_failing())

            with pytest.raises(RuntimeError, match="Watchman socket error"):
                await server.await_startup_barrier()

    # ------------------------------------------------------------------
    # Failure path 7: _current_startup_failure_message() non-None
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_fails_on_current_startup_failure_message(
        self, config: Any
    ) -> None:
        """Path 7: _current_startup_failure_message() is set after realtime task.

        Both tasks complete successfully, but _current_startup_failure_message
        returns a message (via override) to exercise the method-call path.
        """
        server = await self._make_server(config)
        with patch.object(server, "requires_strict_startup_barrier", return_value=True):

            async def _succeed() -> None:
                pass

            server._deferred_start_task = asyncio.create_task(_succeed())
            server._startup_failure_message = None
            server._realtime_start_task = asyncio.create_task(_succeed())

            with patch.object(
                server,
                "_current_startup_failure_message",
                return_value="Watchman session closed",
            ):
                with pytest.raises(RuntimeError, match="Watchman session closed"):
                    await server.await_startup_barrier()

    # ------------------------------------------------------------------
    # Failure path 8: monitoring_ready not set
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_fails_when_monitoring_not_ready(self, config: Any) -> None:
        """Path 8: monitoring_ready not set after all tasks complete."""
        server = await self._make_server(config)
        with patch.object(server, "requires_strict_startup_barrier", return_value=True):

            async def _succeed() -> None:
                pass

            server._deferred_start_task = asyncio.create_task(_succeed())
            server._startup_failure_message = None
            server._realtime_start_task = asyncio.create_task(_succeed())
            server.realtime_indexing = MagicMock()
            # Use a real asyncio.Event that is NOT set
            server.realtime_indexing.monitoring_ready = asyncio.Event()

            with pytest.raises(RuntimeError, match="without monitoring readiness"):
                await server.await_startup_barrier()
