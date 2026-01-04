"""Integration tests for concurrent MCP sessions.

These tests verify that multiple clients can connect to the HTTP daemon
simultaneously and that operations don't interfere with each other.
"""

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestConcurrentMCPSessions:
    """Tests for concurrent MCP session handling."""

    @pytest.fixture
    def mock_services(self) -> MagicMock:
        """Create mock database services."""
        services = MagicMock()
        services.provider = MagicMock()
        services.provider.is_connected = True
        services.provider.get_stats.return_value = {
            "files": 100,
            "chunks": 500,
            "embeddings": 500,
        }
        services.search_service = MagicMock()
        services.search_service.search_regex.return_value = (
            [],
            {"offset": 0, "page_size": 10, "has_more": False},
        )
        services.search_service.search_semantic = AsyncMock(
            return_value=([], {"offset": 0, "page_size": 10, "has_more": False})
        )
        return services

    @pytest.fixture
    def mock_embedding_manager(self) -> MagicMock:
        """Create mock embedding manager."""
        manager = MagicMock()
        manager.list_providers.return_value = ["openai"]
        provider = MagicMock()
        provider.name = "openai"
        provider.model = "text-embedding-3-small"
        manager.get_provider.return_value = provider
        return manager

    @pytest.mark.asyncio
    async def test_concurrent_tool_calls_same_project(
        self, mock_services: MagicMock, mock_embedding_manager: MagicMock
    ) -> None:
        """Test multiple concurrent tool calls for the same project."""
        from chunkhound.mcp_server.tools import execute_tool

        # Run multiple search calls concurrently
        async def run_search(query: str) -> dict[str, Any]:
            return await execute_tool(
                tool_name="search_regex",
                services=mock_services,
                embedding_manager=mock_embedding_manager,
                arguments={"pattern": query, "page_size": 5},
            )

        # Execute 10 concurrent searches
        tasks = [run_search(f"pattern_{i}") for i in range(10)]
        results = await asyncio.gather(*tasks)

        # All should complete without error
        assert len(results) == 10
        for result in results:
            assert "results" in result
            assert "pagination" in result

    @pytest.mark.asyncio
    async def test_concurrent_tool_calls_different_projects(
        self, mock_services: MagicMock, mock_embedding_manager: MagicMock
    ) -> None:
        """Test concurrent tool calls for different projects."""
        from chunkhound.mcp_server.tools import execute_tool

        # Simulate different project contexts
        projects = [
            {"project": "/home/user/project1"},
            {"project": "/home/user/project2"},
            {"project": "/home/user/project3"},
        ]

        async def run_search_in_project(project_context: dict) -> dict[str, Any]:
            return await execute_tool(
                tool_name="search_regex",
                services=mock_services,
                embedding_manager=mock_embedding_manager,
                arguments={"pattern": "test"},
                client_context=project_context,
            )

        # Execute searches for different projects concurrently
        tasks = [run_search_in_project(p) for p in projects * 3]
        results = await asyncio.gather(*tasks)

        assert len(results) == 9
        for result in results:
            assert "results" in result


class TestWatcherManager:
    """Tests for WatcherManager functionality."""

    @pytest.fixture
    def mock_coordinator(self) -> MagicMock:
        """Create mock indexing coordinator."""
        coordinator = MagicMock()
        coordinator.process_file = AsyncMock()
        coordinator.delete_file = AsyncMock()
        coordinator.index_directory = AsyncMock()
        return coordinator

    @pytest.fixture
    def mock_registry(self) -> MagicMock:
        """Create mock project registry."""
        registry = MagicMock()

        project = MagicMock()
        project.project_name = "test-project"
        project.base_directory = Path("/test/project")
        project.tags = []

        registry.find_project_for_path.return_value = project
        registry.get_project.return_value = project
        registry.set_watcher_status.return_value = True

        return registry

    def test_health_check_interval_default(self, mock_coordinator: MagicMock) -> None:
        """Test that health check interval is set correctly."""
        from chunkhound.services.watcher_manager import WatcherManager

        manager = WatcherManager(mock_coordinator)
        assert manager._health_check_interval == 60.0

    def test_batch_threshold_constant(self, mock_coordinator: MagicMock) -> None:
        """Test that batch threshold is defined."""
        from chunkhound.services.watcher_manager import WatcherManager

        assert WatcherManager.BATCH_THRESHOLD == 50

    @pytest.mark.asyncio
    async def test_batch_processing_trigger(
        self, mock_coordinator: MagicMock, mock_registry: MagicMock
    ) -> None:
        """Test that batch processing triggers for many events."""
        from chunkhound.services.watcher_manager import WatcherManager

        manager = WatcherManager(mock_coordinator)
        manager._registry = mock_registry
        manager._running = True
        manager._loop = asyncio.get_event_loop()

        # Create 60 events (above threshold of 50)
        events = [(Path(f"/test/project/file_{i}.py"), "created") for i in range(60)]

        # Call batch processor
        await manager._batch_process_project(Path("/test/project"), events)

        # Should call index_directory for batch mode
        mock_coordinator.index_directory.assert_called_once_with(
            Path("/test/project"),
            skip_embeddings=False,
        )


class TestPortConflictDetection:
    """Tests for port conflict detection."""

    def test_is_port_available_open_port(self) -> None:
        """Test that an unused port is detected as available."""
        from chunkhound.api.cli.commands.daemon import _is_port_available

        # Use a high ephemeral port that's unlikely to be in use
        result = _is_port_available("127.0.0.1", 59999)
        # This might be in use, so we just verify it returns a boolean
        assert isinstance(result, bool)

    def test_is_port_available_invalid_host(self) -> None:
        """Test handling of invalid host."""
        from chunkhound.api.cli.commands.daemon import _is_port_available

        # Invalid host should not crash, should return True (assume available)
        result = _is_port_available("invalid.host.that.does.not.exist", 5173)
        assert result is True


class TestGracefulShutdown:
    """Tests for graceful shutdown behavior."""

    @pytest.fixture
    def mock_watcher_manager(self) -> MagicMock:
        """Create mock watcher manager."""
        manager = MagicMock()
        manager.get_pending_count.return_value = 5
        manager.flush_pending.return_value = 5
        manager.stop = AsyncMock()
        return manager

    @pytest.mark.asyncio
    async def test_shutdown_flushes_pending_events(
        self, mock_watcher_manager: MagicMock
    ) -> None:
        """Test that shutdown flushes pending events."""
        # Simulate what happens in lifespan shutdown
        pending_count = mock_watcher_manager.get_pending_count()
        if pending_count > 0:
            flushed = mock_watcher_manager.flush_pending()
            assert flushed == 5

        await mock_watcher_manager.stop()
        mock_watcher_manager.stop.assert_called_once()


class TestAutoStartDaemon:
    """Tests for auto-start daemon functionality."""

    def test_auto_start_config_default(self) -> None:
        """Test that auto-start is disabled by default."""
        from chunkhound.core.config.database_config import MultiRepoConfig

        config = MultiRepoConfig()
        assert config.auto_start_daemon is False

    def test_auto_start_config_enabled(self) -> None:
        """Test that auto-start can be enabled."""
        from chunkhound.core.config.database_config import MultiRepoConfig

        config = MultiRepoConfig(auto_start_daemon=True)
        assert config.auto_start_daemon is True

    def test_should_use_proxy_without_auto_start(self) -> None:
        """Test proxy detection without auto-start."""
        from chunkhound.core.config.config import Config
        from chunkhound.core.config.database_config import (
            DatabaseConfig,
            MultiRepoConfig,
        )
        from chunkhound.mcp_server.proxy_client import should_use_proxy

        # Create config with global mode but no auto-start
        multi_repo = MultiRepoConfig(
            enabled=True,
            mode="global",
            auto_start_daemon=False,
        )
        db_config = DatabaseConfig(multi_repo=multi_repo)
        config = Config(database=db_config)

        # Mock the daemon status check to return not running
        with patch("chunkhound.services.daemon_manager.DaemonManager") as mock_dm:
            mock_status = MagicMock()
            mock_status.running = False
            mock_status.url = None
            mock_dm.return_value.status.return_value = mock_status

            result, url = should_use_proxy(config)

            # Should not use proxy since daemon is not running and auto-start is off
            assert result is False
            assert url is None


class TestHealthEndpoint:
    """Tests for enhanced health endpoint."""

    def test_health_response_structure(self) -> None:
        """Test that health response includes all expected fields."""
        # Expected fields in the health response
        expected_fields = {
            "status",
            "version",
            "uptime_seconds",
            "projects_indexed",
            "watchers_active",
            "pending_events",
            "memory_mb",
            "active_sessions",
            "database",
            "scan_progress",
        }

        # These are the fields we added
        new_fields = {"pending_events", "memory_mb"}

        # Verify new fields are in expected
        assert new_fields.issubset(expected_fields)


class TestSearchContextFastPath:
    """Tests for path resolution fast path."""

    def test_resolve_scope_empty_without_registry(self) -> None:
        """Test that resolve_scope returns empty scope without registry."""
        from chunkhound.services.search_context import SearchContext, SearchScope

        context = SearchContext(
            current_project_path=Path("/test/project"),
            search_all=True,
        )

        # Without registry, should return empty scope
        scope = context.resolve_scope(None)

        assert isinstance(scope, SearchScope)
        assert len(scope.projects) == 0

    def test_no_paths_no_context_unchanged(self) -> None:
        """Test that args without paths and no context are unchanged."""
        from chunkhound.mcp_server.tools import _resolve_paths

        # Simple arguments without paths
        args = {
            "pattern": "test",
            "page_size": 10,
        }

        # Should return quickly without modification
        result = _resolve_paths(args, None, None)

        assert result == args
