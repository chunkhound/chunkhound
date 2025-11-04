"""Database factory module - creates services with proper dependency injection.

This module eliminates circular dependencies by serving as a dedicated composition root
for service creation. Clean internal architecture with external compatibility.

UNIFIED SERVICE CREATION PATTERN:
1. Use create_services() for clean internal architecture
2. Legacy create_database_with_dependencies() maintained for compatibility
3. Registry configuration happens automatically through this factory

BEHAVIOR GUARANTEE:
This factory ensures consistent component injection across all execution paths:
- CLI commands get same component setup as MCP servers
- All services have proper dependency injection
- Registry configuration is applied uniformly

INTEGRATION REQUIREMENT:
Any changes to this factory must be tested across all execution paths:
- CLI commands (chunkhound run)
- MCP stdio server
- MCP HTTP server
- File change processing
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from chunkhound.embeddings import EmbeddingManager
from chunkhound.registry import configure_registry, get_registry

if TYPE_CHECKING:
    from chunkhound.database import Database
    from chunkhound.interfaces.database_provider import DatabaseProvider
    from chunkhound.services.embedding_service import EmbeddingService
    from chunkhound.services.indexing_coordinator import IndexingCoordinator
    from chunkhound.services.search_service import SearchService


class DatabaseServices(NamedTuple):
    """Clean service bundle for modern internal architecture."""

    provider: "DatabaseProvider"
    indexing_coordinator: "IndexingCoordinator"
    search_service: "SearchService"
    embedding_service: "EmbeddingService"


def create_services(
    db_path: Path | str,
    config: dict[str, Any] | Any,
    embedding_manager: EmbeddingManager | None = None,
) -> DatabaseServices:
    """Create clean service bundle for modern internal architecture.

    Args:
        db_path: Path to database file
        config: Registry configuration (dict or Config object)
        embedding_manager: Optional embedding manager

    Returns:
        DatabaseServices bundle with all components
    """
    # Avoid double-configuring the registry (which can open the DB twice and lock it).
    registry = get_registry()
    try:
        existing_cfg = registry.get_config()
    except Exception:
        existing_cfg = None

    # Always (re)configure the registry with an effective per-call config so that
    # tests using distinct temporary directories get an IndexingCoordinator whose
    # base_directory matches the current tmp_path. This avoids cross-test leakage
    # from the global registry's previous target_dir.
    effective_config: dict[str, Any] | Any = config
    try:
        if isinstance(config, dict):
            effective_config = dict(config)
            db_dict = dict(effective_config.get("database", {}))
            db_dict["path"] = Path(db_path)
            db_dict.setdefault("provider", "duckdb")
            effective_config["database"] = db_dict
        else:
            if hasattr(config, "database") and hasattr(config.database, "path"):
                config.database.path = Path(db_path)
            effective_config = config
    except Exception:
        effective_config = config

    configure_registry(effective_config)
    # else: assume already configured by caller (e.g., CLI), do not reconfigure again
    # to prevent creating a second database provider connection in the same process.

    # If embedding_manager is provided, register its provider with the global registry
    # to ensure services use the same provider instance
    if embedding_manager:
        try:
            provider = embedding_manager.get_default_provider()
            if provider:
                registry.register_provider("embedding", provider, singleton=True)
        except Exception:
            # If no provider in embedding_manager, registry will handle provider creation
            pass

    return DatabaseServices(
        provider=registry.get_provider("database"),
        indexing_coordinator=registry.create_indexing_coordinator(),
        search_service=registry.create_search_service(),
        embedding_service=registry.create_embedding_service(),
    )


def create_database_with_dependencies(
    db_path: Path | str,
    config: dict[str, Any],
    embedding_manager: EmbeddingManager | None = None,
) -> "Database":
    """Legacy wrapper - use create_services() for clean internal architecture.

    Maintained for external compatibility only.
    """
    from chunkhound.database import Database

    services = create_services(db_path, config, embedding_manager)

    return Database(
        db_path=db_path,
        embedding_manager=embedding_manager,
        indexing_coordinator=services.indexing_coordinator,
        search_service=services.search_service,
        embedding_service=services.embedding_service,
        provider=services.provider,
    )
