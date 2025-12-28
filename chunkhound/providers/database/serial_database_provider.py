"""Base class for database providers requiring single-threaded execution."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from chunkhound.embeddings import EmbeddingManager
from chunkhound.file_discovery_cache import FileDiscoveryCache
from chunkhound.providers.database.serial_executor import (
    SerialDatabaseExecutor,
    _transaction_context,
)

# Type hinting only
if TYPE_CHECKING:
    from chunkhound.core.config.database_config import DatabaseConfig
    from chunkhound.services.embedding_service import EmbeddingService
    from chunkhound.services.indexing_coordinator import IndexingCoordinator
    from chunkhound.services.search_service import SearchService


class SerialDatabaseProvider(ABC):
    """Base class for database providers requiring single-threaded execution.

    This class provides the common serial execution pattern for databases like
    DuckDB and LanceDB that require all operations to be serialized through
    a single connection.

    Subclasses must implement:
    - _create_connection(): Create and return a database connection
    - _get_schema_sql(): Return SQL for creating the schema (if applicable)
    - Additional _executor_* methods for provider-specific operations
    """

    def __init__(
        self,
        db_path: Path | str,
        base_directory: Path,
        embedding_manager: EmbeddingManager | None = None,
        config: "DatabaseConfig | None" = None,
    ):
        """Initialize serial database provider.

        Args:
            db_path: Path to database file or directory
            base_directory: Base directory for path normalization (always set)
            embedding_manager: Optional embedding manager for vector generation
            config: Database configuration for provider-specific settings
        """
        self._services_initialized = False
        self.embedding_manager = embedding_manager
        self.config = config
        self._db_path = db_path

        # Create serial executor for all database operations
        self._executor = SerialDatabaseExecutor(config)

        # Service layer components
        self._indexing_coordinator: IndexingCoordinator | None = None
        self._search_service: SearchService | None = None
        self._embedding_service: EmbeddingService | None = None

        # File discovery cache for performance optimization
        self._file_discovery_cache = FileDiscoveryCache()

        # Base directory for path normalization (immutable after initialization)
        self._base_directory: Path = base_directory

    @abstractmethod
    def _create_connection(self) -> Any:
        """Create and return a database connection.

        This method is called from within the executor thread to create
        a thread-local connection.

        Returns:
            Database connection object
        """
        ...

    @abstractmethod
    def _get_schema_sql(self) -> list[str] | None:
        """Get SQL statements for creating the database schema.

        Returns:
            List of SQL statements, or None if not applicable
        """
        ...

    @property
    def db_path(self) -> Path | str:
        """Database connection path or identifier."""
        return self._db_path

    @property
    def is_connected(self) -> bool:
        """Check if database connection is active."""
        # For serial providers, we consider it connected if executor exists
        return self._executor is not None

    @property
    def last_activity_time(self) -> float | None:
        """Get the last database activity time.

        Returns:
            Unix timestamp of last activity, or None if no activity yet
        """
        if self._executor:
            return self._executor.get_last_activity_time()
        return None

    def connect(self) -> None:
        """Establish database connection and initialize schema."""
        try:
            # Execute connection in DB thread to ensure proper initialization
            self._execute_in_db_thread_sync("connect")

            # Initialize shared service instances for performance
            self._initialize_shared_instances()

            logger.info(f"{self.__class__.__name__} initialization complete")

        except Exception as e:
            logger.error(f"{self.__class__.__name__} connection failed: {e}")
            raise

    def close(self) -> None:
        """Close database connection and cleanup resources.

        This is the primary method tests and external code should use for cleanup.
        """
        self.disconnect(skip_checkpoint=False)

    def reset_connection(self, skip_checkpoint: bool = True) -> None:
        """Forcefully reset connection and executor to recover from stuck operations.

        This method disconnects, recreates the executor, and reconnects to clear
        any stuck database operations (e.g., long-running queries that have timed out).

        Args:
            skip_checkpoint: Whether to skip checkpointing during disconnect
        """
        logger.warning("Resetting database connection to clear potential stuck operation...")
        self.disconnect(skip_checkpoint=skip_checkpoint)
        # Recreate executor (original created in __init__)
        from chunkhound.providers.database.serial_executor import SerialDatabaseExecutor
        self._executor = SerialDatabaseExecutor(self.config)
        self.connect()
        logger.info("Database connection reset complete")

    def disconnect(self, skip_checkpoint: bool = False) -> None:
        """Close database connection with optional checkpointing."""
        try:
            # Perform final operations in DB thread
            self._execute_in_db_thread_sync("disconnect", skip_checkpoint)
        finally:
            # Clear thread-local storage
            self._executor.clear_thread_local()
            # Shutdown executor with Windows-specific handling
            self._executor.shutdown(wait=True)

    def _execute_in_db_thread_sync(self, operation_name: str, *args, **kwargs) -> Any:
        """Execute operation synchronously in DB thread."""
        return self._executor.execute_sync(self, operation_name, *args, **kwargs)

    async def _execute_in_db_thread(self, operation_name: str, *args, **kwargs) -> Any:
        """Execute operation asynchronously in DB thread."""
        return await self._executor.execute_async(self, operation_name, *args, **kwargs)

    def get_base_directory(self) -> Path:
        """Get the current base directory for path normalization."""
        return self._base_directory

    def _initialize_shared_instances(self):
        """Initialize service layer components and legacy compatibility objects."""
        logger.debug("Initializing service layer components")

        try:
            # Lazy import from registry to avoid circular dependency
            import importlib

            registry_module = importlib.import_module("chunkhound.registry")
            get_registry = getattr(registry_module, "get_registry")
            create_indexing_coordinator = getattr(
                registry_module, "create_indexing_coordinator"
            )
            create_search_service = getattr(registry_module, "create_search_service")
            create_embedding_service = getattr(
                registry_module, "create_embedding_service"
            )

            # Get registry and register self as database provider
            # NOTE: ProviderRegistry.register_provider expects an instance, not a factory.
            # Using a lambda here caused the registry to hold a function object, leading
            # to AttributeError when services attempted to call provider.connect().
            # Register the instance directly to maintain correct type semantics.
            registry = get_registry()
            registry.register_provider("database", self, singleton=True)

            # Initialize service layer components from registry
            if (
                not hasattr(self, "_indexing_coordinator")
                or self._indexing_coordinator is None
            ):
                self._indexing_coordinator = create_indexing_coordinator()
            if not hasattr(self, "_search_service") or self._search_service is None:
                self._search_service = create_search_service()
            if (
                not hasattr(self, "_embedding_service")
                or self._embedding_service is None
            ):
                self._embedding_service = create_embedding_service()

            logger.debug("Service layer components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize service layer components: {e}")
            # Don't raise the exception, just log it - allows test initialization to continue

    # Default _executor_* methods that subclasses can override

    def _executor_connect(self, conn: Any, state: dict[str, Any]) -> None:
        """Default executor method for connect - runs in DB thread.

        Subclasses can override to add provider-specific initialization.
        """
        logger.info("Database connection established in executor thread")

    def _executor_disconnect(
        self, conn: Any, state: dict[str, Any], skip_checkpoint: bool
    ) -> None:
        """Default executor method for disconnect - runs in DB thread.

        Subclasses should override to add provider-specific cleanup.
        """
        try:
            # Close connection
            if conn:
                conn.close()
            logger.info("Database connection closed in executor thread")
        except Exception as e:
            logger.error(f"Error closing connection: {e}")

    # Common capability detection pattern using hasattr()

    def search_semantic(
        self,
        query_embedding: list[float],
        provider: str,
        model: str,
        page_size: int = 10,
        offset: int = 0,
        threshold: float | None = None,
        path_filter: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Perform semantic vector search if supported."""
        if not hasattr(self, "_executor_search_semantic"):
            return [], {"error": "Semantic search not supported by this provider"}

        return self._execute_in_db_thread_sync(
            "search_semantic",
            query_embedding,
            provider,
            model,
            page_size,
            offset,
            threshold,
            path_filter,
        )

    def search_regex(
        self,
        pattern: str,
        page_size: int = 10,
        offset: int = 0,
        path_filter: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Perform regex search if supported (synchronous)."""
        if not hasattr(self, "_executor_search_regex"):
            return [], {"error": "Regex search not supported by this provider"}

        return self._execute_in_db_thread_sync(
            "search_regex", pattern, page_size, offset, path_filter
        )

    async def search_regex_async(
        self,
        pattern: str,
        page_size: int = 10,
        offset: int = 0,
        path_filter: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Perform regex search if supported (asynchronous).

        This method uses async execution to avoid blocking the event loop,
        allowing other concurrent operations to proceed while waiting for
        database operations.
        """
        if not hasattr(self, "_executor_search_regex"):
            return [], {"error": "Regex search not supported by this provider"}

        return await self._execute_in_db_thread(
            "search_regex", pattern, page_size, offset, path_filter
        )

    def search_chunks_regex(
        self, pattern: str, file_path: str | None = None
    ) -> list[dict[str, Any]]:
        """Backward compatibility wrapper for legacy search_chunks_regex calls."""
        results, _ = self.search_regex(
            pattern=pattern,
            path_filter=file_path,
            page_size=1000,  # Large page for legacy behavior
        )
        return results

    def search_text(
        self, query: str, page_size: int = 10, offset: int = 0
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Perform text search if supported."""
        if not hasattr(self, "_executor_search_text"):
            return [], {"error": "Text search not supported by this provider"}

        return self._execute_in_db_thread_sync("search_text", query, page_size, offset)

    # Capability detection methods

    def supports_semantic_search(self) -> bool:
        """Check if this provider supports semantic search."""
        return hasattr(self, "_executor_search_semantic")

    def supports_regex_search(self) -> bool:
        """Check if this provider supports regex search."""
        return hasattr(self, "_executor_search_regex")

    def supports_fuzzy_search(self) -> bool:
        """Check if this provider supports fuzzy search."""
        return hasattr(self, "_executor_search_fuzzy")

    def supports_text_search(self) -> bool:
        """Check if this provider supports text search."""
        return hasattr(self, "_executor_search_text")

    # Transaction management

    def begin_transaction(self) -> None:
        """Begin a database transaction if supported."""
        if not hasattr(self, "_executor_begin_transaction"):
            # No-op if transactions not supported
            return

        # Set task-local transaction state
        _transaction_context.set(True)
        # Execute in DB thread
        self._execute_in_db_thread_sync("begin_transaction")

    def commit_transaction(self, force_checkpoint: bool = False) -> None:
        """Commit the current transaction if supported."""
        if not hasattr(self, "_executor_commit_transaction"):
            # No-op if transactions not supported
            return

        try:
            self._execute_in_db_thread_sync("commit_transaction", force_checkpoint)
        finally:
            # Clear task-local transaction state
            _transaction_context.set(False)

    def rollback_transaction(self) -> None:
        """Rollback the current transaction if supported."""
        if not hasattr(self, "_executor_rollback_transaction"):
            # No-op if transactions not supported
            return

        try:
            self._execute_in_db_thread_sync("rollback_transaction")
        finally:
            # Clear task-local transaction state
            _transaction_context.set(False)

    # File processing integration



    # Default implementations that raise NotImplementedError
    # Subclasses should implement these as _executor_* methods

    def create_schema(self) -> None:
        """Create database schema for files, chunks, and embeddings."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement create_schema"
        )

    def create_indexes(self) -> None:
        """Create database indexes for performance optimization."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement create_indexes"
        )

    def health_check(self) -> dict[str, Any]:
        """Perform health check and return status information."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement health_check"
        )

    def get_connection_info(self) -> dict[str, Any]:
        """Get information about the database connection."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_connection_info"
        )

    def optimize_tables(self) -> None:
        """Optimize tables by compacting fragments and rebuilding indexes."""
        # Default no-op implementation
        pass
