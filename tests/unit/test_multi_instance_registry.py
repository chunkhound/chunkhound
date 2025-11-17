"""
Tests for multi-instance registry functionality.

Verifies that multiple ChunkHound instances can run simultaneously
with isolated registries, preventing cross-contamination between
MCP server instances.
"""

import asyncio
import tempfile
from pathlib import Path

import pytest

from chunkhound.core.config.config import Config
from chunkhound.database_factory import create_services
from chunkhound.registry import (
    create_registry,
    configure_registry,
    get_registry,
    ProviderRegistry,
)


class TestRegistryIsolation:
    """Test that registry instances are properly isolated."""

    def test_create_registry_returns_new_instance(self):
        """Test that create_registry() returns a new independent instance each time."""
        registry1 = create_registry()
        registry2 = create_registry()

        assert registry1 is not registry2, "create_registry() should return new instances"
        assert isinstance(registry1, ProviderRegistry)
        assert isinstance(registry2, ProviderRegistry)

    def test_global_registry_is_singleton(self):
        """Test that get_registry() returns the same global singleton."""
        registry1 = get_registry()
        registry2 = get_registry()

        assert registry1 is registry2, "get_registry() should return singleton"

    def test_create_registry_different_from_global(self):
        """Test that create_registry() returns instances different from global singleton."""
        global_registry = get_registry()
        new_registry = create_registry()

        assert global_registry is not new_registry, (
            "create_registry() should return instance different from global"
        )

    def test_isolated_registries_dont_share_providers(self, tmp_path: Path):
        """Test that providers registered in one registry don't appear in another."""
        registry1 = create_registry()
        registry2 = create_registry()

        # Create minimal configs with different database paths
        db_path1 = tmp_path / "db1"
        db_path2 = tmp_path / "db2"
        db_path1.mkdir(exist_ok=True)
        db_path2.mkdir(exist_ok=True)

        config1 = {
            "database": {"path": str(db_path1 / "test.db"), "provider": "duckdb"},
            "embedding": {"provider": "openai", "api_key": "test-key-1"},
            "indexing": {"include": ["*.py"]},
        }
        config2 = {
            "database": {"path": str(db_path2 / "test.db"), "provider": "duckdb"},
            "embedding": {"provider": "openai", "api_key": "test-key-2"},
            "indexing": {"include": ["*.js"]},
        }

        # Configure each registry with different configs
        configure_registry(config1, registry=registry1)
        configure_registry(config2, registry=registry2)

        # Get providers from each registry
        db1 = registry1.get_provider("database")
        db2 = registry2.get_provider("database")

        # They should be different instances
        assert db1 is not db2, "Database providers should be different instances"


class TestCreateServicesIsolation:
    """Test that create_services() creates isolated service bundles."""

    def test_create_services_without_registry_creates_isolated_instance(self, tmp_path: Path):
        """Test that create_services() without registry parameter creates isolated registry."""
        db_path1 = tmp_path / "db1" / "test.db"
        db_path2 = tmp_path / "db2" / "test.db"
        db_path1.parent.mkdir(parents=True, exist_ok=True)
        db_path2.parent.mkdir(parents=True, exist_ok=True)

        config = {
            "database": {"provider": "duckdb"},
            "embedding": {"provider": "openai", "api_key": "test-key"},
            "indexing": {"include": ["*.py"]},
        }

        # Create two service bundles without providing registry
        services1 = create_services(db_path=db_path1, config=config)
        services2 = create_services(db_path=db_path2, config=config)

        # They should have different database providers
        assert services1.provider is not services2.provider, (
            "create_services() should create isolated database providers"
        )

    def test_create_services_with_explicit_registry(self, tmp_path: Path):
        """Test that create_services() respects explicitly provided registry."""
        registry = create_registry()

        db_path = tmp_path / "test.db"
        config = {
            "database": {"provider": "duckdb"},
            "embedding": {"provider": "openai", "api_key": "test-key"},
            "indexing": {"include": ["*.py"]},
        }

        # Create services with explicit registry
        services = create_services(db_path=db_path, config=config, registry=registry)

        # The database provider should be registered in the provided registry
        db_from_registry = registry.get_provider("database")
        assert db_from_registry is not None, "Database provider should be registered"

        # The services should use the same database provider as registered
        assert services.provider is db_from_registry, (
            "Services should use database provider from provided registry"
        )


class TestMCPServerIsolation:
    """Test that MCP servers can run with isolated registries."""

    @pytest.mark.asyncio
    async def test_multiple_mcp_servers_with_isolated_registries(self, tmp_path: Path):
        """Test that multiple MCP server instances can initialize with separate registries."""
        from chunkhound.mcp_server.stdio import StdioMCPServer

        # Create two different directories with different configs
        dir1 = tmp_path / "instance1"
        dir2 = tmp_path / "instance2"
        dir1.mkdir(exist_ok=True)
        dir2.mkdir(exist_ok=True)

        # Create different database paths
        db_path1 = dir1 / ".chunkhound" / "test.db"
        db_path2 = dir2 / ".chunkhound" / "test.db"
        db_path1.parent.mkdir(parents=True, exist_ok=True)
        db_path2.parent.mkdir(parents=True, exist_ok=True)

        # Create different configs
        config1 = Config(
            database={"path": str(db_path1), "provider": "duckdb"},
            embedding={"provider": "openai", "api_key": "key1"},
            indexing={"include": ["*.py"]},
        )
        config2 = Config(
            database={"path": str(db_path2), "provider": "duckdb"},
            embedding={"provider": "openai", "api_key": "key2"},
            indexing={"include": ["*.js"]},
        )

        # Create two MCP server instances
        server1 = StdioMCPServer(config=config1)
        server2 = StdioMCPServer(config=config2)

        # Initialize both servers
        await server1.initialize()
        await server2.initialize()

        try:
            # Verify they have different registries
            assert server1._registry is not None, "Server 1 should have a registry"
            assert server2._registry is not None, "Server 2 should have a registry"
            assert server1._registry is not server2._registry, (
                "Servers should have different registries"
            )

            # Verify they have different database providers
            assert server1.services.provider is not server2.services.provider, (
                "Servers should have different database providers"
            )

            # Verify database paths are different
            db1_path = server1.services.provider.db_path
            db2_path = server2.services.provider.db_path
            assert db1_path != db2_path, "Servers should use different database paths"

        finally:
            # Clean up
            await server1.cleanup()
            await server2.cleanup()

    @pytest.mark.asyncio
    async def test_mcp_server_registry_is_lazy_initialized(self):
        """Test that MCP server registry is created lazily during initialize()."""
        from chunkhound.mcp_server.stdio import StdioMCPServer

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path = temp_path / ".chunkhound" / "test.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)

            config = Config(
                database={"path": str(db_path), "provider": "duckdb"},
                embedding={"provider": "openai", "api_key": "test-key"},
                indexing={"include": ["*.py"]},
            )

            # Create server but don't initialize yet
            server = StdioMCPServer(config=config)

            # Registry should be None before initialization
            assert server._registry is None, (
                "Registry should be None before initialize() is called"
            )

            # Initialize the server
            await server.initialize()

            try:
                # Registry should now exist
                assert server._registry is not None, (
                    "Registry should be created during initialize()"
                )

            finally:
                await server.cleanup()


class TestBackwardCompatibility:
    """Test that global registry behavior is maintained for backward compatibility."""

    def test_configure_registry_without_explicit_registry_uses_global(self, tmp_path: Path):
        """Test that configure_registry() without registry parameter uses global singleton."""
        db_path = tmp_path / "test.db"
        config = {
            "database": {"path": str(db_path), "provider": "duckdb"},
            "embedding": {"provider": "openai", "api_key": "test-key"},
            "indexing": {"include": ["*.py"]},
        }

        # Get global registry before configuration
        global_registry = get_registry()

        # Configure without explicit registry parameter
        result_registry = configure_registry(config)

        # Should return the same global registry
        assert result_registry is global_registry, (
            "configure_registry() without registry param should use global singleton"
        )

    def test_database_provider_with_no_registry_uses_global(self, tmp_path: Path):
        """Test that database provider without registry parameter uses global registry."""
        from chunkhound.providers.database.duckdb_provider import DuckDBProvider

        db_path = tmp_path / "test.db"

        # Create provider without registry parameter (backward compatibility)
        provider = DuckDBProvider(
            db_path=db_path,
            base_directory=tmp_path,
        )

        # Provider should work (uses global registry internally)
        provider.connect()

        try:
            # Should be able to perform basic operations
            # Note: is_connected is a property in SerialDatabaseProvider, not a method
            assert provider.is_connected, "Provider should connect successfully"
        finally:
            provider.disconnect()


class TestDatabaseProviderRegistryParameter:
    """Test that database providers correctly accept and use registry parameter."""

    def test_duckdb_provider_with_explicit_registry(self, tmp_path: Path):
        """Test that DuckDBProvider respects explicit registry parameter."""
        from chunkhound.providers.database.duckdb_provider import DuckDBProvider

        registry = create_registry()
        db_path = tmp_path / "test.db"

        # Create provider with explicit registry
        provider = DuckDBProvider(
            db_path=db_path,
            base_directory=tmp_path,
            registry=registry,
        )

        # Store the registry internally
        assert hasattr(provider, "_registry"), "Provider should store registry"
        assert provider._registry is registry, "Provider should use provided registry"

    def test_lancedb_provider_with_explicit_registry(self, tmp_path: Path):
        """Test that LanceDBProvider respects explicit registry parameter."""
        pytest.importorskip("lancedb", reason="LanceDB not installed")

        from chunkhound.providers.database.lancedb_provider import LanceDBProvider

        registry = create_registry()
        db_path = tmp_path / "lancedb"

        # Create provider with explicit registry
        provider = LanceDBProvider(
            db_path=db_path,
            base_directory=tmp_path,
            registry=registry,
        )

        # Store the registry internally
        assert hasattr(provider, "_registry"), "Provider should store registry"
        assert provider._registry is registry, "Provider should use provided registry"

    def test_duckdb_uses_serial_wrapper_with_registry(self, tmp_path: Path):
        """Test that DuckDBProvider (which extends SerialDatabaseProvider) uses registry."""
        from chunkhound.providers.database.duckdb_provider import DuckDBProvider

        registry = create_registry()
        db_path = tmp_path / "test.db"

        # Create DuckDB provider with explicit registry
        # DuckDBProvider extends SerialDatabaseProvider which has registry support
        provider = DuckDBProvider(
            db_path=db_path,
            base_directory=tmp_path,
            registry=registry,
        )

        # Store the registry internally
        assert hasattr(provider, "_registry"), "Provider should store registry"
        assert provider._registry is registry, "Provider should use provided registry"


if __name__ == "__main__":
    # Allow running directly for debugging
    pytest.main([__file__, "-v"])
