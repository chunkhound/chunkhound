"""Test MCP server initialization invariants and shutdown safety."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.mcp_server.base import MCPServerBase
from chunkhound.providers.database.serial_database_provider import (
    SerialDatabaseProvider,
)


class ConcreteMCPServer(MCPServerBase):
    """Minimal concrete implementation for testing base class behavior."""

    def _register_tools(self) -> None:
        pass

    async def run(self) -> None:
        pass


class MinimalSerialProvider(SerialDatabaseProvider):
    """Minimal concrete serial provider for lifecycle regression tests."""

    def _create_connection(self) -> object:
        return object()

    def _get_schema_sql(self) -> list[str] | None:
        return None

    def _executor_ping(self, conn: object, state: dict[str, object]) -> str:
        return "ok"


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

                # Guards the internal contract between base.py and
                # common.py/tools.py: handle_tool_call passes _scan_progress
                # to tool functions that expect these keys.  If the dict
                # shape changes, this test should break so the tool
                # parameter wiring is updated in lockstep.
                progress = server._scan_progress

                assert "is_scanning" in progress
                assert "files_processed" in progress
                assert "chunks_created" in progress
                assert "scan_started_at" in progress
                assert "scan_completed_at" in progress
                assert "background_compaction" in progress

                background_compaction = progress["background_compaction"]
                assert background_compaction["phase"] == "idle"
                assert background_compaction["pending_recovery"] is False
                assert background_compaction["retry_attempted"] is False

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


class TestShutdownSafety:
    """Verify cleanup() skips provider disconnect when compaction thread is stuck."""

    @pytest.mark.asyncio
    async def test_stuck_compaction_skips_provider_disconnect(self, tmp_path: Path):
        """When compaction thread won't stop, provider.close() must not be called."""
        config = MagicMock()
        config.database.path = str(tmp_path / "test.db")
        config.embedding = None
        config.llm = None
        config.target_dir = tmp_path

        with patch("chunkhound.mcp_server.base.create_services") as mock_create:
            mock_provider = MagicMock()
            mock_provider.is_connected = True
            mock_provider.mark_terminal_after_stuck_cleanup = MagicMock()

            mock_services = MagicMock()
            mock_services.provider = mock_provider
            mock_create.return_value = mock_services

            with patch("chunkhound.mcp_server.base.EmbeddingManager"):
                server = ConcreteMCPServer(config=config)
                await server.initialize()
                assert server._initialized

                # Simulate a compaction service whose thread never finishes.
                stuck_event = threading.Event()  # never set
                mock_compaction = MagicMock()
                mock_compaction.compaction_thread_done = stuck_event
                mock_compaction.shutdown = AsyncMock()
                server._compaction_service = mock_compaction

                # Patch asyncio.to_thread so the 5s wait returns immediately.
                with patch("asyncio.to_thread", new_callable=AsyncMock):
                    await server.cleanup()

                # Provider must NOT have been closed/disconnected.
                mock_provider.close.assert_not_called()
                mock_provider.disconnect.assert_not_called()
                mock_provider.mark_terminal_after_stuck_cleanup.assert_called_once_with()
                mock_provider.shutdown_executor.assert_called_once_with()

                # _initialized stays True (cleanup is terminal, no reconnection).
                assert server._initialized

    @pytest.mark.asyncio
    async def test_stuck_compaction_cleans_non_provider_resources_before_executor_shutdown(
        self, tmp_path: Path
    ) -> None:
        """Cleanup must cancel server-managed work before killing the DB executor."""
        config = MagicMock()
        config.database.path = str(tmp_path / "test.db")
        config.embedding = None
        config.llm = None
        config.target_dir = tmp_path

        with patch("chunkhound.mcp_server.base.create_services") as mock_create:
            call_order: list[str] = []
            mock_provider = MagicMock()
            mock_provider.is_connected = True
            mock_provider.mark_terminal_after_stuck_cleanup.side_effect = (
                lambda: call_order.append("mark_terminal")
            )
            mock_provider.shutdown_executor.side_effect = lambda: call_order.append(
                "shutdown_executor"
            )

            mock_services = MagicMock()
            mock_services.provider = mock_provider
            mock_create.return_value = mock_services

            with patch("chunkhound.mcp_server.base.EmbeddingManager"):
                server = ConcreteMCPServer(config=config)
                await server.initialize()

                stuck_event = MagicMock()
                stuck_event.is_set.return_value = False
                stuck_event.wait.return_value = False

                mock_compaction = MagicMock()
                mock_compaction.compaction_thread_done = stuck_event
                mock_compaction.shutdown = AsyncMock()
                server._compaction_service = mock_compaction

                async def fake_cleanup_non_provider_resources() -> None:
                    call_order.append("cleanup_non_provider")

                server._cleanup_non_provider_resources = (
                    fake_cleanup_non_provider_resources
                )

                await server.cleanup()

                assert call_order == [
                    "cleanup_non_provider",
                    "mark_terminal",
                    "shutdown_executor",
                ]

    @pytest.mark.asyncio
    async def test_cleanup_is_idempotent_after_stuck_compaction(
        self, tmp_path: Path
    ) -> None:
        """A second cleanup() after a stuck-thread path must not race the thread.

        After the stuck-thread early-return, _compaction_service is cleared but
        the provider is still connected. A second cleanup() must not fall
        through to provider.close() — the compaction thread may still be alive.

        The contract is expressed as a ``side_effect`` on ``close``/
        ``disconnect``: deleting the ``_cleanup_done`` idempotency guard
        causes the second call to fall through and trip the AssertionError.
        """
        config = MagicMock()
        config.database.path = str(tmp_path / "test.db")
        config.embedding = None
        config.llm = None
        config.target_dir = tmp_path

        with patch("chunkhound.mcp_server.base.create_services") as mock_create:
            mock_provider = MagicMock()
            mock_provider.is_connected = True
            mock_provider.mark_terminal_after_stuck_cleanup = MagicMock()
            # Contract: while the compaction thread is stuck alive, the
            # provider must NEVER be closed or disconnected. Any invocation
            # of either method is a correctness violation.
            mock_provider.close.side_effect = AssertionError(
                "MUST NOT close provider while compaction thread is alive"
            )
            mock_provider.disconnect.side_effect = AssertionError(
                "MUST NOT disconnect provider while compaction thread is alive"
            )

            mock_services = MagicMock()
            mock_services.provider = mock_provider
            mock_create.return_value = mock_services

            with patch("chunkhound.mcp_server.base.EmbeddingManager"):
                server = ConcreteMCPServer(config=config)
                await server.initialize()

                # Event-like mock: reports "not set" forever, but wait()
                # returns immediately so the real asyncio.to_thread call
                # inside cleanup() completes without a 5-second delay.
                stuck_event = MagicMock()
                stuck_event.is_set.return_value = False
                stuck_event.wait.return_value = False

                mock_compaction = MagicMock()
                mock_compaction.compaction_thread_done = stuck_event
                mock_compaction.shutdown = AsyncMock()
                server._compaction_service = mock_compaction

                # First call hits the stuck-thread early-return branch.
                await server.cleanup()
                # Second call must be a no-op guarded by _cleanup_done. If
                # the guard is removed, this call falls through to
                # provider.close() and the side_effect fires.
                await server.cleanup()

                # Provider must never have been closed across both calls
                assert mock_provider.close.call_count == 0
                assert mock_provider.disconnect.call_count == 0
                mock_provider.mark_terminal_after_stuck_cleanup.assert_called_once_with()
                mock_provider.shutdown_executor.assert_called_once_with()

    def test_terminal_cleanup_provider_rejects_subsequent_db_work(
        self, tmp_path: Path
    ) -> None:
        """Provider operations must fail explicitly after terminal cleanup."""
        provider = MinimalSerialProvider(
            db_path=tmp_path / "test.db",
            base_directory=tmp_path,
        )
        provider.mark_terminal_after_stuck_cleanup()
        provider.shutdown_executor()

        with pytest.raises(RuntimeError, match="terminal cleanup state"):
            provider._execute_in_db_thread_sync("ping")

    @pytest.mark.asyncio
    async def test_connect_provider_rejects_terminal_provider_even_if_marked_connected(
        self, tmp_path: Path
    ) -> None:
        """_connect_provider must fail fast on a terminal provider state."""
        config = MagicMock()
        config.database.path = str(tmp_path / "test.db")
        config.embedding = None
        config.llm = None
        config.target_dir = tmp_path

        with patch("chunkhound.mcp_server.base.create_services") as mock_create:
            mock_provider = MagicMock()
            mock_provider.is_connected = True
            mock_provider.ensure_usable.side_effect = RuntimeError(
                "Cannot connect provider: provider entered terminal cleanup "
                "state after stuck compaction"
            )

            mock_services = MagicMock()
            mock_services.provider = mock_provider
            mock_create.return_value = mock_services

            with patch("chunkhound.mcp_server.base.EmbeddingManager"):
                server = ConcreteMCPServer(config=config)
                await server.initialize()

                with pytest.raises(RuntimeError, match="terminal cleanup state"):
                    await server._connect_provider()

                mock_provider.connect.assert_not_called()

    @pytest.mark.asyncio
    async def test_concurrent_cleanup_serializes_via_lock(self, tmp_path: Path) -> None:
        """Two parallel cleanup() tasks must serialize and only close once.

        Without ``_cleanup_lock``, two concurrent callers can both pass the
        ``_cleanup_done`` guard before the ``finally`` clause flips the flag,
        causing a double provider close. Asserts that only one call reaches
        the provider teardown path.
        """
        config = MagicMock()
        config.database.path = str(tmp_path / "test.db")
        config.embedding = None
        config.llm = None
        config.target_dir = tmp_path

        with patch("chunkhound.mcp_server.base.create_services") as mock_create:
            mock_provider = MagicMock()
            mock_provider.is_connected = True

            mock_services = MagicMock()
            mock_services.provider = mock_provider
            mock_create.return_value = mock_services

            with patch("chunkhound.mcp_server.base.EmbeddingManager"):
                server = ConcreteMCPServer(config=config)
                await server.initialize()

                await asyncio.gather(server.cleanup(), server.cleanup())

                # Provider closed exactly once across both concurrent calls
                close_calls = (
                    mock_provider.close.call_count + mock_provider.disconnect.call_count
                )
                assert close_calls == 1

    def test_serial_provider_operations_fail_explicitly_after_terminal_cleanup(
        self, tmp_path: Path
    ) -> None:
        """Post-terminal DB calls must fail before touching the dead executor."""
        provider = MinimalSerialProvider(
            db_path=tmp_path / "test.db",
            base_directory=tmp_path,
        )
        provider.mark_terminal_after_stuck_cleanup()

        with pytest.raises(RuntimeError, match="terminal cleanup state"):
            provider._execute_in_db_thread_sync("connect")
