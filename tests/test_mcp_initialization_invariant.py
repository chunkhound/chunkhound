"""Test that MCP server initialization is non-blocking (scan runs in background)."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import textwrap
import threading
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import duckdb
import pytest

import chunkhound_native as native
from chunkhound.core.config.config import Config
from chunkhound.database_factory import create_services
from chunkhound.mcp_server.base import MCPServerBase
from chunkhound.mcp_server.common import handle_tool_call
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.registry import ProviderRegistry
from tests.hub.artifacts import build_database
from tests.hub.conftest import embedding_http as _embedding_http_fixture

embedding_http = _embedding_http_fixture


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


@pytest.mark.asyncio
@pytest.mark.parametrize("partial_manager", [True, False])
async def test_mcp_cleanup_releases_partial_or_retried_database_ownership(
    tmp_path, monkeypatch, partial_manager
):
    directory = tmp_path / "database"
    directory.mkdir()
    target = directory / "chunks.db"
    config = Config(
        target_dir=tmp_path,
        database={"path": directory},
        analytics={"enabled": False},
        skip_layers={"global", "env", "local_config", "config_file", "cli"},
    )
    monkeypatch.setenv("CHUNKHOUND_DEBUG_FILE", str(tmp_path / "mcp.log"))
    provider = DuckDBProvider(target, base_directory=tmp_path)
    provider.connect()
    server = ConcreteMCPServer(config)
    server.services = SimpleNamespace(provider=provider)
    manager = provider.connection
    real_close = duckdb.DuckDBPyConnection.close
    failed = False

    def fail_manager_once(conn):
        nonlocal failed
        if conn is manager and not failed:
            failed = True
            raise OSError("one-time manager close failure")
        return real_close(conn)

    try:
        if partial_manager:
            provider._connection_manager.disconnect(skip_checkpoint=True)
            assert not provider.is_connected
            assert provider.execute_query("SELECT 42 AS value") == [{"value": 42}]
            await server.cleanup()
        else:
            with monkeypatch.context() as fault:
                fault.setattr(duckdb.DuckDBPyConnection, "close", fail_manager_once)
                await server.cleanup()
                assert failed
                with pytest.raises(native.DatabaseBusyError):
                    native.acquire_install_lease(str(target), "duckdb")
                # Same public caller retries a completed failure, never a raw lease.
                await server.cleanup()
        native.acquire_install_lease(str(target), "duckdb").close()
        await server.cleanup()  # Successful cleanup remains idempotent.
        native.acquire_install_lease(str(target), "duckdb").close()
    finally:
        provider.close()  # Also makes the failing RED run safe to tear down.


@pytest.mark.asyncio
async def test_later_mcp_search_recovers_after_deferred_install_contention(tmp_path):
    directory = tmp_path / "database"
    directory.mkdir()
    target = directory / "chunks.db"
    seed = DuckDBProvider(target, base_directory=tmp_path)
    seed.connect()
    seed.close()
    config = Config(
        target_dir=tmp_path,
        database={"path": directory, "read_only": True},
        analytics={"enabled": False},
        skip_layers={"global", "env", "local_config", "config_file", "cli"},
    )
    server = ConcreteMCPServer(config)
    lease = native.acquire_install_lease(str(target), "duckdb")
    try:
        await server.initialize()
        # Model stdio/HTTP lifespan readiness, independent of deferred DB admission.
        server._initialization_complete.set()
        provider = server.services.provider
        assert server._deferred_start_task is not None
        await server._deferred_start_task
        assert not provider.is_connected

        async def search():
            contents = await handle_tool_call(
                tool_name="search",
                arguments={"type": "regex", "query": "never_matches"},
                services=server.services,
                embedding_manager=server.embedding_manager,
                initialization_complete=server._initialization_complete,
                config=config,
                ensure_services=server.ensure_tool_services,
            )
            return contents[0].text

        refused = json.loads(await search())
        assert refused["error"]["type"] == "DatabaseBusyError"
        lease.close()
        result = await search()
        assert result == "No results found."
        assert server.services.provider is provider
        assert provider.is_connected
    finally:
        lease.close()
        await server.cleanup()
    native.acquire_install_lease(str(target), "duckdb").close()


@pytest.mark.parametrize("entry_point", ["factory", "registry"])
def test_service_construction_is_eager_by_default(tmp_path, entry_point):
    config = Config(
        target_dir=tmp_path,
        database={"path": tmp_path / "database"},
        analytics={"enabled": False},
        skip_layers={"global", "env", "local_config", "config_file", "cli"},
    )
    if entry_point == "factory":
        provider = create_services(config.database.get_db_path(), config).provider
    else:
        registry = ProviderRegistry()
        registry.configure(config)
        provider = registry.get_provider("database")
    try:
        # Assert BEFORE a query can implicitly open an executor connection.
        assert provider.is_connected
        assert provider.execute_query("SELECT 42 AS value") == [{"value": 42}]
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(config.database.get_db_path()), "duckdb")
    finally:
        provider.close()
    native.acquire_install_lease(str(config.database.get_db_path()), "duckdb").close()


@pytest.mark.parametrize("config_shape", ["object", "dict"])
def test_real_factory_defers_database_ownership_only_when_requested(
    tmp_path, config_shape
):
    config = Config(
        target_dir=tmp_path,
        database={"path": tmp_path / "database"},
        analytics={"enabled": False},
        skip_layers={"global", "env", "local_config", "config_file", "cli"},
    )
    target = config.database.get_db_path()
    registry = ProviderRegistry()
    services = create_services(
        target,
        config if config_shape == "object" else {
            **config.model_dump(),
            "skip_layers": {"global", "env", "local_config", "config_file", "cli"},
        },
        connect_database=False,
        registry=registry,
    )
    provider = services.provider
    try:
        assert not provider.is_connected
        assert not target.exists()
        # A real exclusive lease proves construction acquired no hidden DB owner.
        lease = native.acquire_install_lease(str(target), "duckdb")
        lease.close()
        provider.connect()
        registry.complete_database_initialization()
        assert provider.is_connected
        assert services.search_service.search_regex("never_matches")[0] == []
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(target), "duckdb")
    finally:
        provider.close()
    native.acquire_install_lease(str(target), "duckdb").close()


@pytest.mark.asyncio
async def test_lancedb_mcp_search_reads_seeded_index_and_pins_before_indexing(
    tmp_path, embedding_http
):
    directory = tmp_path / "database"
    target = build_database(
        directory,
        backend="lancedb",
        provider="openai",
        model="hub-test",
        dims=8,
        symbol_prefix="lancedb_ready_marker",
    )
    (tmp_path / "source.py").write_text(
        "def lancedb_ready_marker_0(): return 0\n", encoding="utf-8"
    )
    config = Config(
        target_dir=tmp_path,
        database={"provider": "lancedb", "path": directory, "read_only": False},
        embedding={
            "provider": "openai",
            "model": "different-configured-model",
            "output_dims": 16,
            "api_key": "local-test-key",
            "base_url": embedding_http.url,
        },
        analytics={"enabled": False},
        skip_layers={"global", "env", "local_config", "config_file", "cli"},
    )
    assert config.database.get_db_path() == target
    server = ConcreteMCPServer(config)
    try:
        await server.initialize()
        # initialize schedules deferred startup without yielding afterward.
        # Cancel before the next await so neither monitoring nor indexing
        # can initialize tables and conceal first-tool readiness failures.
        task = server._deferred_start_task
        assert task is not None
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        assert task.cancelled()
        # Model stdio/HTTP transport readiness, NOT database readiness.
        server._initialization_complete.set()
        assert server.services is not None
        assert server.embedding_manager is not None
        database_provider = server.services.provider
        executor = database_provider._executor
        embedding_provider = server.embedding_manager.get_provider()
        assert embedding_provider.model == "different-configured-model"
        assert embedding_provider.dims == 16
        assert embedding_http.requests == []

        async def search(kind):
            contents = await handle_tool_call(
                tool_name="search",
                arguments={"type": kind, "query": "lancedb_ready_marker_0"},
                services=server.services,
                embedding_manager=server.embedding_manager,
                initialization_complete=server._initialization_complete,
                config=config,
                ensure_services=server.ensure_tool_services,
            )
            return contents[0].text

        regex = await search("regex")
        assert "def lancedb_ready_marker_0(): return 0" in regex, regex
        assert "source.py" in regex, regex
        assert "No results found." not in regex
        assert '"error"' not in regex
        assert server.embedding_manager.get_provider() is embedding_provider
        assert embedding_provider.model == "hub-test"
        assert embedding_provider.dims == 8
        assert config.embedding is not None
        assert config.embedding.model == "hub-test"
        assert config.embedding.output_dims == 8

        request_count = len(embedding_http.requests)
        semantic = await search("semantic")
        # All 100 vectors are identical: demand a real seeded hit, not
        # arbitrary tie ordering that happens to return chunk zero.
        assert "def lancedb_ready_marker_" in semantic, semantic
        assert "source.py" in semantic, semantic
        assert "No results found." not in semantic
        assert '"error"' not in semantic
        assert len(embedding_http.requests) > request_count
        assert all(r["model"] == "hub-test" for r in embedding_http.requests)
        assert server.services.provider is database_provider
        assert database_provider._executor is executor
    finally:
        await server.cleanup()
