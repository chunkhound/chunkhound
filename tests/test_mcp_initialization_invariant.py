"""Test that MCP server initialization is non-blocking (scan runs in background)."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import textwrap
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from chunkhound.mcp_server.base import MCPServerBase


class ConcreteMCPServer(MCPServerBase):
    """Minimal concrete implementation for testing base class behavior."""

    def _register_tools(self) -> None:
        pass

    async def run(self) -> None:
        pass


class CloseOnlyProvider:
    """Provider exposing close() for cleanup tests."""

    def __init__(self) -> None:
        self.is_connected = True
        self.close_calls = 0
        self.disconnect_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        self.is_connected = False

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.is_connected = False


class DisconnectOnlyProvider:
    """Provider exposing only disconnect() for cleanup tests."""

    def __init__(self) -> None:
        self.is_connected = True
        self.disconnect_calls = 0

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.is_connected = False


class BlockingCloseProvider:
    """Provider whose close() blocks until released by the test."""

    def __init__(self) -> None:
        self.is_connected = True
        self.close_calls = 0
        self.close_started = threading.Event()
        self.close_finished = threading.Event()
        self.release_close = threading.Event()

    def close(self) -> None:
        self.close_calls += 1
        self.close_started.set()
        self.release_close.wait(timeout=1.0)
        self.is_connected = False
        self.close_finished.set()


class FailingCloseProvider:
    """Provider whose close() raises immediately."""

    def __init__(self) -> None:
        self.is_connected = True

    def close(self) -> None:
        raise RuntimeError("close exploded")


class FailingCloseWithDisconnectProvider:
    """Provider whose close() fails even though disconnect() exists."""

    def __init__(self) -> None:
        self.is_connected = True
        self.close_calls = 0
        self.disconnect_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        raise RuntimeError("close exploded")

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.is_connected = False


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
                assert progress.get("query_ready_at") is None, (
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
                assert "query_ready_at" in progress

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
                assert server._scan_progress["query_ready_at"] is None

                await server.cleanup()

    @pytest.mark.asyncio
    async def test_initialize_skips_invalid_custom_endpoint_llm_without_manager(
        self, tmp_path: Path
    ):
        """Invalid custom endpoint LLM config should not create an MCP LLM manager."""
        config = MagicMock()
        config.database.path = str(tmp_path / "test.db")
        config.embedding = None
        config.llm.get_missing_config_for_roles.return_value = [
            "explicit model selection required for custom OpenAI-compatible "
            "endpoint roles: utility, synthesis"
        ]
        config.target_dir = tmp_path

        with patch("chunkhound.mcp_server.base.create_services") as mock_create:
            mock_services = MagicMock()
            mock_services.provider.is_connected = False
            mock_create.return_value = mock_services

            with patch("chunkhound.mcp_server.base.EmbeddingManager"):
                with patch("chunkhound.mcp_server.base.LLMManager") as mock_llm_manager:
                    server = ConcreteMCPServer(config=config)
                    await server.initialize()

                    mock_llm_manager.assert_not_called()
                    assert server.llm_manager is None

                    await server.cleanup()

    @pytest.mark.asyncio
    async def test_initialize_allows_cleanup_only_llm_misconfiguration(
        self, tmp_path: Path
    ) -> None:
        """Cleanup-only overrides must not block MCP's research-capable LLM roles."""
        config = MagicMock()
        config.database.path = str(tmp_path / "test.db")
        config.embedding = None
        config.llm.get_missing_config_for_roles.return_value = []
        config.llm.get_provider_configs.return_value = (
            {"provider": "codex-cli", "model": "codex"},
            {"provider": "codex-cli", "model": "codex"},
        )
        config.target_dir = tmp_path

        with patch("chunkhound.mcp_server.base.create_services") as mock_create:
            mock_services = MagicMock()
            mock_services.provider.is_connected = False
            mock_create.return_value = mock_services

            with patch("chunkhound.mcp_server.base.EmbeddingManager"):
                with patch("chunkhound.mcp_server.base.LLMManager") as mock_llm_manager:
                    mock_llm_manager.return_value = MagicMock()
                    server = ConcreteMCPServer(config=config)
                    await server.initialize()

                    mock_llm_manager.assert_called_once_with(
                        {"provider": "codex-cli", "model": "codex"},
                        {"provider": "codex-cli", "model": "codex"},
                        target_dir=tmp_path,
                    )
                    assert server.llm_manager is mock_llm_manager.return_value
                    config.llm.get_provider_configs.assert_called_once_with()

                    await server.cleanup()


class TestCleanup:
    """Verify cleanup invariants for connected providers."""

    @staticmethod
    def _make_server(tmp_path: Path, provider: object) -> ConcreteMCPServer:
        config = MagicMock()
        config.database.path = str(tmp_path / "test.db")
        config.embedding = None
        config.llm = None
        config.target_dir = tmp_path

        server = ConcreteMCPServer(config=config)
        server.services = SimpleNamespace(provider=provider)
        server._initialized = True
        return server

    @pytest.mark.asyncio
    async def test_cleanup_uses_close_when_available(self, tmp_path: Path) -> None:
        provider = CloseOnlyProvider()
        server = self._make_server(tmp_path, provider)

        await server.cleanup()

        assert provider.close_calls == 1
        assert provider.disconnect_calls == 0
        assert not server._initialized

    @pytest.mark.asyncio
    async def test_cleanup_falls_back_to_disconnect(self, tmp_path: Path) -> None:
        provider = DisconnectOnlyProvider()
        server = self._make_server(tmp_path, provider)

        await server.cleanup()

        assert provider.disconnect_calls == 1
        assert not server._initialized

    @pytest.mark.asyncio
    async def test_cleanup_timeout_keeps_event_loop_responsive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = BlockingCloseProvider()
        server = self._make_server(tmp_path, provider)
        debug_file = tmp_path / "cleanup.log"
        ticker = asyncio.create_task(asyncio.sleep(0.01))

        monkeypatch.setattr(
            "chunkhound.mcp_server.base._DATABASE_CLOSE_TIMEOUT_SECONDS",
            0.05,
        )
        monkeypatch.setenv("CHUNKHOUND_DEBUG_FILE", str(debug_file))

        started = asyncio.get_running_loop().time()
        try:
            await server.cleanup()
        finally:
            provider.release_close.set()
            await asyncio.to_thread(provider.close_finished.wait, 1.0)
        elapsed = asyncio.get_running_loop().time() - started

        assert provider.close_started.is_set()
        assert ticker.done()
        assert elapsed < 0.5
        assert not server._initialized
        assert "will continue in the background daemon thread" in debug_file.read_text()

    @pytest.mark.asyncio
    async def test_cleanup_reuses_in_flight_close_after_timeout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = BlockingCloseProvider()
        server = self._make_server(tmp_path, provider)

        monkeypatch.setattr(
            "chunkhound.mcp_server.base._DATABASE_CLOSE_TIMEOUT_SECONDS",
            0.05,
        )

        try:
            await server.cleanup()
            await server.cleanup()
        finally:
            provider.release_close.set()
            await asyncio.to_thread(provider.close_finished.wait, 1.0)

        assert provider.close_started.is_set()
        assert provider.close_calls == 1
        assert not server._initialized

    def test_cleanup_timeout_does_not_hang_asyncio_run_shutdown(
        self, tmp_path: Path
    ) -> None:
        script = textwrap.dedent(
            f"""
            import asyncio
            from pathlib import Path
            from types import SimpleNamespace

            from chunkhound.mcp_server.base import MCPServerBase
            import chunkhound.mcp_server.base as base_module

            base_module._DATABASE_CLOSE_TIMEOUT_SECONDS = 0.05

            class ConcreteMCPServer(MCPServerBase):
                def _register_tools(self) -> None:
                    pass

                async def run(self) -> None:
                    pass

            class BlockingCloseProvider:
                def __init__(self) -> None:
                    self.is_connected = True

                def close(self) -> None:
                    import time
                    time.sleep(60)

            async def main() -> None:
                config = SimpleNamespace(
                    database=SimpleNamespace(
                        path=str(Path({str(tmp_path / "test.db")!r}))
                    ),
                    embedding=None,
                    llm=None,
                    target_dir=Path({str(tmp_path)!r}),
                )
                server = ConcreteMCPServer(config=config)
                server.services = SimpleNamespace(provider=BlockingCloseProvider())
                server._initialized = True
                await server.cleanup()

            asyncio.run(main())
            print("cleanup returned")
            """
        )

        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True,
            text=True,
            # Importing chunkhound.mcp_server.base has a cold-start cost of
            # ~1.6-2.5s (transitive deps like llm_manager, database_factory,
            # core.config) locally, but on macOS ARM CI runners it can take
            # 4-5s.  A 5s timeout becomes flaky; 15s provides margin for
            # import + the 0.05s cleanup timeout + print + cold runner.
            timeout=15,
        )

        assert completed.returncode == 0, completed.stderr
        assert "cleanup returned" in completed.stdout

    @pytest.mark.asyncio
    async def test_cleanup_ignores_closed_loop_race_when_close_thread_finishes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = CloseOnlyProvider()
        server = self._make_server(tmp_path, provider)
        loop = asyncio.get_running_loop()

        monkeypatch.setattr(
            "chunkhound.mcp_server.base._DATABASE_CLOSE_TIMEOUT_SECONDS",
            0.01,
        )
        monkeypatch.setattr(
            loop,
            "call_soon_threadsafe",
            MagicMock(side_effect=RuntimeError("Event loop is closed")),
        )

        await server.cleanup()

        assert provider.close_calls == 1
        assert not server._initialized

    @pytest.mark.asyncio
    async def test_cleanup_logs_close_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = FailingCloseProvider()
        server = self._make_server(tmp_path, provider)
        debug_file = tmp_path / "cleanup.log"

        monkeypatch.setenv("CHUNKHOUND_DEBUG_FILE", str(debug_file))

        await server.cleanup()

        assert not server._initialized
        assert "Database close failed: close exploded" in debug_file.read_text()

    @pytest.mark.asyncio
    async def test_cleanup_does_not_fallback_to_disconnect_after_close_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = FailingCloseWithDisconnectProvider()
        server = self._make_server(tmp_path, provider)
        debug_file = tmp_path / "cleanup.log"

        monkeypatch.setenv("CHUNKHOUND_DEBUG_FILE", str(debug_file))

        await server.cleanup()

        assert provider.close_calls == 1
        assert provider.disconnect_calls == 0
        assert not server._initialized
        assert "Database close failed: close exploded" in debug_file.read_text()
