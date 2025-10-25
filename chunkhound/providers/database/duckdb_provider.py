"""DuckDB provider implementation for ChunkHound - concrete database provider using DuckDB.

# FILE_CONTEXT: High-performance analytical database provider
# CRITICAL: Single-threaded access enforced by SerialDatabaseProvider
# PERFORMANCE: HNSW indexes for vector search, bulk operations optimized

## PERFORMANCE_CHARACTERISTICS
- Bulk inserts: 5000 rows optimal batch size
- Vector search: HNSW index with cosine similarity
- Index optimization: Drop/recreate for >50 embeddings (12x speedup)
- WAL mode: Automatic checkpointing, 1GB limit
"""

import os
import re
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from chunkhound.core.models import Chunk, Embedding, File
from chunkhound.core.types.common import ChunkType, Language
from chunkhound.core.utils import normalize_path_for_lookup

# Import existing components that will be used by the provider
from chunkhound.embeddings import EmbeddingManager
from chunkhound.providers.database.duckdb.chunk_repository import DuckDBChunkRepository
from chunkhound.providers.database.duckdb.connection_manager import (
    DuckDBConnectionManager,
)
from chunkhound.providers.database.duckdb.embedding_repository import (
    DuckDBEmbeddingRepository,
)
from chunkhound.providers.database.duckdb.file_repository import DuckDBFileRepository
from chunkhound.providers.database.serial_database_provider import (
    SerialDatabaseProvider,
)
from chunkhound.providers.database.serial_executor import (
    _executor_local,
    track_operation,
)

# Type hinting only
if TYPE_CHECKING:
    from chunkhound.core.config.database_config import DatabaseConfig


class DuckDBProvider(SerialDatabaseProvider):
    """DuckDB implementation of DatabaseProvider protocol.

    # CLASS_CONTEXT: Analytical database optimized for bulk operations
    # CONSTRAINT: Inherits from SerialDatabaseProvider for thread safety
    # PERFORMANCE: Uses column-store format, vectorized execution
    """

    def __init__(
        self,
        db_path: Path | str,
        base_directory: Path,
        embedding_manager: "EmbeddingManager | None" = None,
        config: "DatabaseConfig | None" = None,
    ):
        """Initialize DuckDB provider.

        Args:
            db_path: Path to DuckDB database file or ":memory:" for in-memory database
            base_directory: Base directory for path normalization
            embedding_manager: Optional embedding manager for vector generation
            config: Database configuration for provider-specific settings
        """
        # Initialize base class
        super().__init__(db_path, base_directory, embedding_manager, config)

        self.provider_type = "duckdb"  # Identify this as DuckDB provider

        # Class-level synchronization for WAL cleanup
        self._wal_cleanup_lock = threading.Lock()
        self._wal_cleanup_done = False

        # Initialize connection manager (will be simplified later)
        self._connection_manager = DuckDBConnectionManager(db_path, config)

        # Initialize file repository with provider reference for transaction awareness
        self._file_repository = DuckDBFileRepository(self._connection_manager, self)

        # Initialize chunk repository with provider reference for transaction awareness
        self._chunk_repository = DuckDBChunkRepository(self._connection_manager, self)

        # Initialize embedding repository with provider reference for transaction awareness
        self._embedding_repository = DuckDBEmbeddingRepository(
            self._connection_manager, self
        )
        self._embedding_repository.set_provider_instance(self)

    def _create_connection(self) -> Any:
        """Create and return a DuckDB connection.

        This method is called from within the executor thread to create
        a thread-local connection.

        Returns:
            DuckDB connection object
        """
        # Suppress known SWIG warning from DuckDB Python bindings
        import warnings

        warnings.filterwarnings(
            "ignore", message=".*swigvarlink.*", category=DeprecationWarning
        )
        import duckdb

        # Create a NEW connection for the executor thread
        # This ensures thread safety - only this thread will use this connection
        conn = duckdb.connect(str(self._connection_manager.db_path))

        # Load required extensions
        conn.execute("INSTALL vss")
        conn.execute("LOAD vss")
        conn.execute("SET hnsw_enable_experimental_persistence = true")

        logger.debug(
            f"Created new DuckDB connection in executor thread {threading.get_ident()}"
        )
        return conn

    def _get_schema_sql(self) -> list[str] | None:
        """Get SQL statements for creating the DuckDB schema.

        Returns:
            List of SQL statements
        """
        # DuckDB uses its own schema creation logic in _executor_create_schema
        return None

    @property
    def connection(self) -> Any | None:
        """Database connection - delegate to connection manager.

        Note: This property is maintained for backward compatibility but should not
        be used directly. All database operations should go through executor methods.
        """
        return self._connection_manager.connection

    @property
    def db_path(self) -> Path | str:
        """Database connection path or identifier - delegate to connection manager."""
        return self._connection_manager.db_path

    @property
    def is_connected(self) -> bool:
        """Check if database connection is active - delegate to connection manager."""
        return self._connection_manager.is_connected

    def _extract_file_id(self, file_record: dict[str, Any] | File) -> int | None:
        """Safely extract file ID from either dict or File model - delegate to file repository."""
        return self._file_repository._extract_file_id(file_record)

    def connect(self) -> None:
        """Establish database connection and initialize schema with WAL validation."""
        try:
            # Initialize connection manager FIRST - this handles WAL validation
            self._connection_manager.connect()

            # Call parent connect which handles executor initialization
            super().connect()

        except Exception as e:
            logger.error(f"DuckDB connection failed: {e}")
            raise

    def _executor_connect(self, conn: Any, state: dict[str, Any]) -> None:
        """Executor method for connect - runs in DB thread.

        Note: The connection is already created by _get_thread_local_connection,
        so this method just ensures schema and indexes are created.
        """
        try:
            # Perform WAL cleanup once with synchronization
            with self._wal_cleanup_lock:
                if not self._wal_cleanup_done:
                    self._perform_wal_cleanup_in_executor(conn)
                    self._wal_cleanup_done = True

            # Create schema
            self._executor_create_schema(conn, state)

            # Create indexes
            self._executor_create_indexes(conn, state)

            # Migrate legacy embeddings table if needed
            self._executor_migrate_legacy_embeddings_table(conn, state)

            logger.info("Database initialization complete in executor thread")

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    def _perform_wal_cleanup_in_executor(self, conn: Any) -> None:
        """Perform WAL cleanup within the executor thread.

        This ensures all DuckDB operations happen in the same thread.
        """
        if str(self._connection_manager.db_path) == ":memory:":
            return

        db_path = Path(self._connection_manager.db_path)
        wal_file = db_path.with_suffix(db_path.suffix + ".wal")

        if not wal_file.exists():
            return

        # Check WAL file age
        try:
            wal_age = time.time() - wal_file.stat().st_mtime
            if wal_age > 86400:  # 24 hours
                logger.warning(
                    f"Found stale WAL file (age: {wal_age / 3600:.1f}h), removing"
                )
                wal_file.unlink(missing_ok=True)
                return
        except OSError:
            pass

        # Test WAL validity by running a simple query
        try:
            conn.execute("SELECT 1").fetchone()
            logger.debug("WAL file validation passed")
        except Exception as e:
            logger.warning(f"WAL validation failed ({e}), removing WAL file")
            conn.close()
            wal_file.unlink(missing_ok=True)
            # Recreate connection after WAL cleanup
            conn = self._create_connection()
            _executor_local.connection = conn

    def disconnect(self, skip_checkpoint: bool = False) -> None:
        """Close database connection with optional checkpointing - delegate to connection manager."""
        try:
            # Call parent disconnect
            super().disconnect(skip_checkpoint)
        finally:
            # Disconnect connection manager for backward compatibility
            self._connection_manager.disconnect(
                skip_checkpoint=True
            )  # Skip checkpoint since we did it in executor

    def _executor_disconnect(
        self, conn: Any, state: dict[str, Any], skip_checkpoint: bool
    ) -> None:
        """Executor method for disconnect - runs in DB thread."""
        try:
            if not skip_checkpoint:
                # Force checkpoint before close to ensure durability
                conn.execute("CHECKPOINT")
                if not os.environ.get("CHUNKHOUND_MCP_MODE"):
                    logger.debug("Database checkpoint completed before disconnect")
            else:
                if not os.environ.get("CHUNKHOUND_MCP_MODE"):
                    logger.debug("Skipping checkpoint before disconnect (already done)")
        except Exception as e:
            if not os.environ.get("CHUNKHOUND_MCP_MODE"):
                logger.error(f"Checkpoint failed during disconnect: {e}")
        finally:
            # Close connection
            conn.close()
            # Clear thread-local connection
            if hasattr(_executor_local, "connection"):
                delattr(_executor_local, "connection")
            if not os.environ.get("CHUNKHOUND_MCP_MODE"):
                logger.info("DuckDB connection closed in executor thread")

    def health_check(self) -> dict[str, Any]:
        """Perform health check and return status information - delegate to connection manager."""
        return self._connection_manager.health_check()

    def get_connection_info(self) -> dict[str, Any]:
        """Get information about the database connection - delegate to connection manager."""
        return self._connection_manager.get_connection_info()

    def _table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database - delegate to connection manager."""
        return self._execute_in_db_thread_sync("table_exists", table_name)

    def _executor_table_exists(
        self, conn: Any, state: dict[str, Any], table_name: str
    ) -> bool:
        """Executor method for _table_exists - runs in DB thread."""
        result = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name = ?",
            [table_name],
        ).fetchone()
        return result is not None

    def _get_table_name_for_dimensions(self, dims: int) -> str:
        """Get table name for given embedding dimensions."""
        return f"embeddings_{dims}"

    def _ensure_embedding_table_exists(self, dims: int) -> str:
        """Ensure embedding table exists for given dimensions - delegate to connection manager."""
        return self._execute_in_db_thread_sync("ensure_embedding_table_exists", dims)

    def _executor_ensure_embedding_table_exists(
        self, conn: Any, state: dict[str, Any], dims: int
    ) -> str:
        """Executor method for _ensure_embedding_table_exists - runs in DB thread."""
        table_name = f"embeddings_{dims}"

        if self._executor_table_exists(conn, state, table_name):
            return table_name

        logger.info(f"Creating embedding table for {dims} dimensions: {table_name}")

        try:
            # Create table with fixed dimensions for HNSW compatibility
            conn.execute(f"""
                CREATE TABLE {table_name} (
                    id INTEGER PRIMARY KEY DEFAULT nextval('embeddings_id_seq'),
                    chunk_id INTEGER NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    embedding FLOAT[{dims}],
                    dims INTEGER NOT NULL DEFAULT {dims},
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create HNSW index for performance
            hnsw_index_name = f"idx_hnsw_{dims}"
            conn.execute(f"""
                CREATE INDEX {hnsw_index_name} ON {table_name}
                USING HNSW (embedding)
                WITH (metric = 'cosine')
            """)

            # Create regular indexes for fast lookups
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{dims}_chunk_id "
                f"ON {table_name}(chunk_id)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{dims}_provider_model "
                f"ON {table_name}(provider, model)"
            )

            logger.info(
                f"Created {table_name} with HNSW index {hnsw_index_name} "
                "and regular indexes"
            )
            return table_name

        except Exception as e:
            logger.error(f"Failed to create embedding table for {dims} dimensions: {e}")
            raise

    def _maybe_checkpoint(self, force: bool = False) -> None:
        """Perform checkpoint if needed - delegate to connection manager."""
        self._execute_in_db_thread_sync("maybe_checkpoint", force)

    def _executor_maybe_checkpoint(
        self, conn: Any, state: dict[str, Any], force: bool
    ) -> None:
        """Executor method for _maybe_checkpoint - runs in DB thread."""
        # Defer checkpoint if we're in a transaction
        if state.get("transaction_active", False):
            state["deferred_checkpoint"] = True
            if not os.environ.get("CHUNKHOUND_MCP_MODE"):
                logger.debug("Deferring checkpoint until transaction completes")
            return

        current_time = time.time()
        time_since_checkpoint = current_time - state.get(
            "last_checkpoint_time", current_time
        )
        operations_since_checkpoint = state.get("operations_since_checkpoint", 0)

        # Checkpoint if forced, operations threshold reached, or 5 minutes elapsed
        should_checkpoint = (
            force
            or operations_since_checkpoint >= 100  # Checkpoint every 100 operations
            or time_since_checkpoint >= 300  # 5 minutes
        )

        if should_checkpoint:
            try:
                conn.execute("CHECKPOINT")
                state["operations_since_checkpoint"] = 0
                state["last_checkpoint_time"] = current_time
                if not os.environ.get("CHUNKHOUND_MCP_MODE"):
                    logger.debug(
                        f"Checkpoint completed (operations: {operations_since_checkpoint}, "
                        f"time: {time_since_checkpoint:.1f}s)"
                    )
            except Exception as e:
                if not os.environ.get("CHUNKHOUND_MCP_MODE"):
                    logger.warning(f"Checkpoint failed: {e}")

    def create_schema(self) -> None:
        """Create database schema for files, chunks, and embeddings - delegate to connection manager."""
        self._execute_in_db_thread_sync("create_schema")

    def _executor_create_schema(self, conn: Any, state: dict[str, Any]) -> None:
        """Executor method for create_schema - runs in DB thread."""
        logger.info("Creating DuckDB schema")

        try:
            # Create sequence for files table
            conn.execute("CREATE SEQUENCE IF NOT EXISTS files_id_seq")

            # Files table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY DEFAULT nextval('files_id_seq'),
                    path TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    extension TEXT,
                    size INTEGER,
                    modified_time TIMESTAMP,
                    content_hash TEXT,
                    language TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Ensure content_hash exists for existing DBs
            conn.execute(
                "ALTER TABLE files ADD COLUMN IF NOT EXISTS content_hash TEXT"
            )

            # Create sequence for chunks table
            conn.execute("CREATE SEQUENCE IF NOT EXISTS chunks_id_seq")

            # Chunks table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY DEFAULT nextval('chunks_id_seq'),
                    file_id INTEGER REFERENCES files(id),
                    chunk_type TEXT NOT NULL,
                    symbol TEXT,
                    code TEXT NOT NULL,
                    start_line INTEGER,
                    end_line INTEGER,
                    start_byte INTEGER,
                    end_byte INTEGER,
                    language TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create sequence for embeddings table
            conn.execute("CREATE SEQUENCE IF NOT EXISTS embeddings_id_seq")

            # Embeddings table (1536 dimensions as default)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings_1536 (
                    id INTEGER PRIMARY KEY DEFAULT nextval('embeddings_id_seq'),
                    chunk_id INTEGER NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    embedding FLOAT[1536],
                    dims INTEGER NOT NULL DEFAULT 1536,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for 1536-dimensional embeddings
            try:
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_hnsw_1536 ON embeddings_1536
                    USING HNSW (embedding)
                    WITH (metric = 'cosine')
                """)
                logger.info(
                    "HNSW index for 1536-dimensional embeddings created successfully"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to create HNSW index for 1536-dimensional embeddings: {e}"
                )

            # Create index on chunk_id for efficient deletions
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_1536_chunk_id ON embeddings_1536(chunk_id)
            """)

            # Handle schema migrations for existing databases
            self._executor_migrate_schema(conn, state)

            logger.info(
                "DuckDB schema created successfully with multi-dimension support"
            )

        except Exception as e:
            logger.error(f"Failed to create DuckDB schema: {e}")
            raise

    def _executor_migrate_schema(self, conn: Any, state: dict[str, Any]) -> None:
        """Executor method for schema migrations - runs in DB thread."""
        try:
            # Check if 'size' and 'signature' columns exist and drop them
            columns_info = conn.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'chunks' 
                AND column_name IN ('size', 'signature')
            """).fetchall()

            if columns_info:
                logger.info(
                    "Migrating chunks table: removing unused 'size' and 'signature' columns"
                )

                # SQLite/DuckDB doesn't support DROP COLUMN directly, need to recreate table
                # First, create a temporary table with the new schema
                conn.execute("""
                    CREATE TEMP TABLE chunks_new AS
                    SELECT id, file_id, chunk_type, symbol, code, 
                           start_line, end_line, start_byte, end_byte, 
                           language, created_at, updated_at
                    FROM chunks
                """)

                # Drop the old table
                conn.execute("DROP TABLE chunks")

                # Create the new table with correct schema
                conn.execute("""
                    CREATE TABLE chunks (
                        id INTEGER PRIMARY KEY DEFAULT nextval('chunks_id_seq'),
                        file_id INTEGER REFERENCES files(id),
                        chunk_type TEXT NOT NULL,
                        symbol TEXT,
                        code TEXT NOT NULL,
                        start_line INTEGER,
                        end_line INTEGER,
                        start_byte INTEGER,
                        end_byte INTEGER,
                        language TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Copy data back
                conn.execute("""
                    INSERT INTO chunks 
                    SELECT * FROM chunks_new
                """)

                # Drop the temporary table
                conn.execute("DROP TABLE chunks_new")

                # Recreate indexes (will be done in _executor_create_indexes)
                logger.info("Successfully migrated chunks table schema")

        except Exception as e:
            logger.warning(f"Failed to migrate schema: {e}")

    def _get_all_embedding_tables(self) -> list[str]:
        """Get list of all embedding tables (dimension-specific) - delegate to connection manager."""
        return self._execute_in_db_thread_sync("get_all_embedding_tables")

    def _executor_get_all_embedding_tables(
        self, conn: Any, state: dict[str, Any]
    ) -> list[str]:
        """Executor method for _get_all_embedding_tables - runs in DB thread."""
        tables = conn.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_name LIKE 'embeddings_%'
        """).fetchall()

        return [table[0] for table in tables]

    def create_indexes(self) -> None:
        """Create database indexes for performance optimization - delegate to connection manager."""
        self._execute_in_db_thread_sync("create_indexes")

    def _executor_create_indexes(self, conn: Any, state: dict[str, Any]) -> None:
        """Executor method for create_indexes - runs in DB thread."""
        logger.info("Creating DuckDB indexes")

        try:
            # File indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_files_path ON files(path)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_files_language ON files(language)"
            )

            # Chunk indexes
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(chunk_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_symbol ON chunks(symbol)"
            )

            # Embedding indexes are created per-table in _executor_ensure_embedding_table_exists()

            logger.info("DuckDB indexes created successfully")

        except Exception as e:
            logger.error(f"Failed to create DuckDB indexes: {e}")
            raise

    def _executor_migrate_legacy_embeddings_table(
        self, conn: Any, state: dict[str, Any]
    ) -> None:
        """Executor method for migrating legacy embeddings table - runs in DB thread."""
        # Check if legacy embeddings table exists
        if not self._executor_table_exists(conn, state, "embeddings"):
            return

        logger.info(
            "Found legacy embeddings table, migrating to dimension-specific tables..."
        )

        try:
            # Get all embeddings with their dimensions
            embeddings = conn.execute("""
                SELECT id, chunk_id, provider, model, embedding, dims, created_at
                FROM embeddings
            """).fetchall()

            if not embeddings:
                logger.info("Legacy embeddings table is empty, dropping it")
                conn.execute("DROP TABLE embeddings")
                return

            # Group by dimensions
            by_dims = {}
            for emb in embeddings:
                dims = emb[5]  # dims column
                if dims not in by_dims:
                    by_dims[dims] = []
                by_dims[dims].append(emb)

            # Migrate each dimension group
            for dims, emb_list in by_dims.items():
                table_name = self._executor_ensure_embedding_table_exists(
                    conn, state, dims
                )
                logger.info(f"Migrating {len(emb_list)} embeddings to {table_name}")

                # Insert data into dimension-specific table
                for emb in emb_list:
                    vector_str = str(emb[4])  # embedding column
                    conn.execute(
                        f"""
                        INSERT INTO {table_name} 
                        (chunk_id, provider, model, embedding, dims, created_at)
                        VALUES (?, ?, ?, {vector_str}, ?, ?)
                    """,
                        [emb[1], emb[2], emb[3], emb[5], emb[6]],
                    )

            # Drop legacy table
            conn.execute("DROP TABLE embeddings")
            logger.info(
                f"Successfully migrated embeddings to {len(by_dims)} "
                "dimension-specific tables"
            )

        except Exception as e:
            logger.error(f"Failed to migrate legacy embeddings table: {e}")
            raise

    def create_vector_index(
        self, provider: str, model: str, dims: int, metric: str = "cosine"
    ) -> None:
        """Create HNSW vector index for specific provider/model/dims combination.

        # INDEX_TYPE: HNSW (Hierarchical Navigable Small World)
        # METRIC: Cosine similarity (normalized vectors)
        # BUILD_TIME: ~10s for 100k vectors
        """
        logger.info(f"Creating HNSW index for {provider}/{model} ({dims}D, {metric})")

        # Use synchronous executor for non-async method
        self._execute_in_db_thread_sync(
            "create_vector_index", provider, model, dims, metric
        )

    def _executor_create_vector_index(
        self,
        conn: Any,
        state: dict[str, Any],
        provider: str,
        model: str,
        dims: int,
        metric: str,
    ) -> None:
        """Executor method for create_vector_index - runs in DB thread."""
        try:
            # Get the correct table name for the dimensions
            table_name = f"embeddings_{dims}"

            # Ensure the table exists before creating the index
            self._executor_ensure_embedding_table_exists(conn, state, dims)

            index_name = f"hnsw_{provider}_{model}_{dims}_{metric}".replace(
                "-", "_"
            ).replace(".", "_")

            # Create HNSW index using VSS extension on the dimension-specific table
            conn.execute(f"""
                CREATE INDEX {index_name} ON {table_name}
                USING HNSW (embedding)
                WITH (metric = '{metric}')
            """)

            logger.info(f"HNSW index {index_name} created successfully on {table_name}")

        except Exception as e:
            logger.error(f"Failed to create HNSW index: {e}")
            raise

    def drop_vector_index(
        self, provider: str, model: str, dims: int, metric: str = "cosine"
    ) -> str:
        """Drop HNSW vector index for specific provider/model/dims combination."""
        return self._execute_in_db_thread_sync(
            "drop_vector_index", provider, model, dims, metric
        )

    def _executor_drop_vector_index(
        self,
        conn: Any,
        state: dict[str, Any],
        provider: str,
        model: str,
        dims: int,
        metric: str,
    ) -> str:
        """Executor method for drop_vector_index - runs in DB thread."""
        index_name = f"hnsw_{provider}_{model}_{dims}_{metric}".replace(
            "-", "_"
        ).replace(".", "_")

        try:
            conn.execute(f"DROP INDEX IF EXISTS {index_name}")
            logger.info(f"HNSW index {index_name} dropped successfully")
            return index_name

        except Exception as e:
            logger.error(f"Failed to drop HNSW index {index_name}: {e}")
            raise

    def get_existing_vector_indexes(self) -> list[dict[str, Any]]:
        """Get list of existing HNSW vector indexes on all embedding tables."""
        return self._execute_in_db_thread_sync("get_existing_vector_indexes")

    def _executor_get_existing_vector_indexes(
        self, conn: Any, state: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Executor method for get_existing_vector_indexes - runs in DB thread."""
        try:
            # Query DuckDB system tables for indexes on all embedding tables
            # Look for both legacy 'hnsw_' and standard 'idx_hnsw_' index patterns
            results = conn.execute("""
                SELECT index_name, table_name
                FROM duckdb_indexes()
                WHERE table_name LIKE 'embeddings_%'
                AND (index_name LIKE 'hnsw_%' OR index_name LIKE 'idx_hnsw_%')
            """).fetchall()

            indexes = []
            for result in results:
                index_name = result[0]
                table_name = result[1]

                # Handle different index naming patterns
                if index_name.startswith("hnsw_"):
                    # Parse custom index name: hnsw_{provider}_{model}_{dims}_{metric}
                    parts = index_name[5:].split("_")  # Remove 'hnsw_' prefix
                    if len(parts) >= 4:
                        # Reconstruct provider/model from parts (they may contain underscores)
                        metric = parts[-1]
                        dims_str = parts[-2]
                        try:
                            dims = int(dims_str)
                            # Join remaining parts as provider_model, then split on last underscore
                            provider_model = "_".join(parts[:-2])
                            # Find last underscore to separate provider and model
                            last_underscore = provider_model.rfind("_")
                            if last_underscore > 0:
                                provider = provider_model[:last_underscore]
                                model = provider_model[last_underscore + 1 :]
                            else:
                                provider = provider_model
                                model = ""

                            indexes.append(
                                {
                                    "index_name": index_name,
                                    "provider": provider,
                                    "model": model,
                                    "dims": dims,
                                    "metric": metric,
                                }
                            )
                        except ValueError:
                            logger.warning(
                                f"Could not parse dims from custom index name: {index_name}"
                            )

                elif index_name.startswith("idx_hnsw_"):
                    # Parse standard index name: idx_hnsw_{dims}
                    # Extract dims from table name: embeddings_{dims}
                    try:
                        if table_name.startswith("embeddings_"):
                            dims = int(table_name[11:])  # Remove 'embeddings_' prefix
                            indexes.append(
                                {
                                    "index_name": index_name,
                                    "provider": "generic",  # Standard index doesn't specify provider
                                    "model": "generic",  # Standard index doesn't specify model
                                    "dims": dims,
                                    "metric": "cosine",  # Default metric for standard indexes
                                }
                            )
                    except ValueError:
                        logger.warning(
                            f"Could not parse dims from standard index: {index_name} on {table_name}"
                        )

            return indexes

        except Exception as e:
            logger.error(f"Failed to get existing vector indexes: {e}")
            return []

    def bulk_operation_with_index_management(self, operation_func, *args, **kwargs):
        """Execute bulk operation with automatic HNSW index management and transaction safety.

        # PATTERN: Drop indexes → Bulk operation → Recreate indexes
        # THRESHOLD: Operations with >50 rows benefit
        # PERFORMANCE: 10-20x speedup for large batches
        """
        # Delegate to executor for proper thread safety
        return self._execute_in_db_thread_sync(
            "bulk_operation_with_index_management_executor",
            operation_func,
            args,
            kwargs,
        )

    def _executor_bulk_operation_with_index_management_executor(
        self, conn: Any, state: dict[str, Any], operation_func, args, kwargs
    ):
        """Executor method for bulk operations with index management - runs in DB thread."""
        # Get existing indexes before starting
        existing_indexes = self._executor_get_existing_vector_indexes(conn, state)
        dropped_indexes = []

        try:
            # Start transaction for atomic operation
            conn.execute("BEGIN TRANSACTION")
            state["transaction_active"] = True

            # Optimize settings for bulk loading
            conn.execute("SET preserve_insertion_order = false")

            # Drop existing HNSW vector indexes to improve bulk performance
            if existing_indexes:
                logger.info(
                    f"Dropping {len(existing_indexes)} HNSW indexes for bulk operation"
                )
                for index_info in existing_indexes:
                    try:
                        self._executor_drop_vector_index(
                            conn,
                            state,
                            index_info["provider"],
                            index_info["model"],
                            index_info["dims"],
                            index_info["metric"],
                        )
                        dropped_indexes.append(index_info)
                    except Exception as e:
                        logger.warning(
                            f"Could not drop index {index_info['index_name']}: {e}"
                        )

            # Execute the bulk operation
            result = operation_func(*args, **kwargs)

            # Recreate dropped indexes
            if dropped_indexes:
                logger.info(
                    f"Recreating {len(dropped_indexes)} HNSW indexes after bulk operation"
                )
                for index_info in dropped_indexes:
                    try:
                        self._executor_create_vector_index(
                            conn,
                            state,
                            index_info["provider"],
                            index_info["model"],
                            index_info["dims"],
                            index_info["metric"],
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to recreate index {index_info['index_name']}: {e}"
                        )
                        # Continue with other indexes

            # Commit transaction
            conn.execute("COMMIT")
            state["transaction_active"] = False

            # Force checkpoint after bulk operations to ensure durability
            self._executor_maybe_checkpoint(conn, state, True)

            logger.info("Bulk operation completed successfully with index management")
            return result

        except Exception as e:
            # Rollback transaction on any error
            try:
                conn.execute("ROLLBACK")
                state["transaction_active"] = False
                logger.info("Transaction rolled back due to error")
            except:
                pass

            # Attempt to recreate dropped indexes on failure
            if dropped_indexes:
                logger.info("Attempting to recreate dropped indexes after failure")
                for index_info in dropped_indexes:
                    try:
                        self._executor_create_vector_index(
                            conn,
                            state,
                            index_info["provider"],
                            index_info["model"],
                            index_info["dims"],
                            index_info["metric"],
                        )
                    except Exception as recreate_error:
                        logger.error(
                            f"Failed to recreate index {index_info['index_name']}: {recreate_error}"
                        )

            logger.error(f"Bulk operation failed: {e}")
            raise

    def insert_file(self, file: File) -> int:
        """Insert file record and return file ID - delegate to file repository."""
        return self._execute_in_db_thread_sync("insert_file", file)

    def _executor_insert_file(
        self, conn: Any, state: dict[str, Any], file: File
    ) -> int:
        """Executor method for insert_file - runs in DB thread."""
        try:
            # First try to find existing file by path
            existing = self._executor_get_file_by_path(
                conn, state, str(file.path), False
            )
            if existing:
                # File exists, update it
                file_id = existing["id"]
                self._executor_update_file(
                    conn,
                    state,
                    file_id,
                    file.size_bytes if hasattr(file, "size_bytes") else None,
                    file.mtime if hasattr(file, "mtime") else None,
                    getattr(file, "content_hash", None),
                )
                return file_id

            # Track operation for checkpoint management
            track_operation(state)

            # No existing file, insert new one
            result = conn.execute(
                """
                INSERT INTO files (path, name, extension, size, modified_time, content_hash, language)
                VALUES (?, ?, ?, ?, to_timestamp(?), ?, ?)
                RETURNING id
            """,
                [
                    file.path,  # Store path as-is (now relative with forward slashes)
                    file.name if hasattr(file, "name") else Path(file.path).name,
                    file.extension
                    if hasattr(file, "extension")
                    else Path(file.path).suffix,
                    file.size_bytes if hasattr(file, "size_bytes") else None,
                    file.mtime if hasattr(file, "mtime") else None,
                    getattr(file, "content_hash", None),
                    file.language.value if file.language else None,
                ],
            )

            file_id = result.fetchone()[0]
            return file_id

        except Exception as e:
            # Handle duplicate key errors
            if "Duplicate key" in str(e) and "violates unique constraint" in str(e):
                existing = self._executor_get_file_by_path(
                    conn, state, str(file.path), False
                )
                if existing and "id" in existing:
                    logger.info(f"Returning existing file ID for {file.path}")
                    return existing["id"]
            raise

    def get_file_by_path(
        self, path: str, as_model: bool = False
    ) -> dict[str, Any] | File | None:
        """Get file record by path - delegate to file repository."""
        return self._execute_in_db_thread_sync("get_file_by_path", path, as_model)

    def _executor_get_file_by_path(
        self, conn: Any, state: dict[str, Any], path: str, as_model: bool
    ) -> dict[str, Any] | File | None:
        """Executor method for get_file_by_path - runs in DB thread."""
        # Normalize path to handle both absolute and relative paths
        from chunkhound.core.utils import normalize_path_for_lookup

        base_dir = state.get("base_directory")
        lookup_path = normalize_path_for_lookup(path, base_dir)
        result = conn.execute(
            """
            SELECT id, path, name, extension, size, modified_time, language, content_hash, created_at, updated_at
            FROM files
            WHERE path = ?
        """,
            [lookup_path],
        ).fetchone()

        if result is None:
            return None

        file_dict = {
            "id": result[0],
            "path": result[1],
            "name": result[2],
            "extension": result[3],
            "size": result[4],
            "modified_time": result[5],
            "language": result[6],
            "content_hash": result[7],
            "created_at": result[8],
            "updated_at": result[9],
        }

        if as_model:
            # Convert DuckDB TIMESTAMP to epoch seconds (float)
            mval = file_dict["modified_time"]
            try:
                from datetime import datetime

                if isinstance(mval, datetime):
                    mtime = mval.timestamp()
                else:
                    mtime = float(mval) if mval is not None else 0.0
            except Exception:
                mtime = 0.0

            try:
                size_bytes = int(file_dict["size"]) if file_dict["size"] is not None else 0
            except Exception:
                size_bytes = 0

            lang_value = file_dict.get("language")
            language = Language(lang_value) if lang_value else None

            return File(
                id=file_dict["id"],
                path=Path(file_dict["path"]).as_posix(),
                mtime=mtime,
                language=language if language is not None else Language.UNKNOWN,
                size_bytes=size_bytes,
            )

        return file_dict

    def get_file_by_id(
        self, file_id: int, as_model: bool = False
    ) -> dict[str, Any] | File | None:
        """Get file record by ID - delegate to file repository."""
        return self._file_repository.get_file_by_id(file_id, as_model)

    def update_file(
        self,
        file_id: int,
        size_bytes: int | None = None,
        mtime: float | None = None,
        content_hash: str | None = None,
        **kwargs,
    ) -> None:
        """Update file record with new values - delegate to file repository."""
        self._execute_in_db_thread_sync("update_file", file_id, size_bytes, mtime, content_hash)

    def _executor_update_file(
        self,
        conn: Any,
        state: dict[str, Any],
        file_id: int,
        size_bytes: int | None,
        mtime: float | None,
        content_hash: str | None,
    ) -> None:
        """Executor method for update_file - runs in DB thread."""
        # Track operation for checkpoint management
        track_operation(state)

        # Build update query dynamically
        updates = []
        params = []

        if size_bytes is not None:
            updates.append("size = ?")
            params.append(size_bytes)

        if mtime is not None:
            updates.append("modified_time = to_timestamp(?)")
            params.append(mtime)

        if content_hash is not None:
            updates.append("content_hash = ?")
            params.append(content_hash)

        if updates:
            updates.append("updated_at = CURRENT_TIMESTAMP")
            query = f"UPDATE files SET {', '.join(updates)} WHERE id = ?"
            params.append(file_id)
            conn.execute(query, params)

    def delete_file_completely(self, file_path: str) -> bool:
        """Delete a file and all its chunks/embeddings completely - delegate to file repository."""
        return self._execute_in_db_thread_sync("delete_file_completely", file_path)

    async def delete_file_completely_async(self, file_path: str) -> bool:
        """Async version of delete_file_completely for non-blocking operation."""
        return await self._execute_in_db_thread("delete_file_completely", file_path)

    def _executor_delete_file_completely(
        self, conn: Any, state: dict[str, Any], file_path: str
    ) -> bool:
        """Executor method for delete_file_completely - runs in DB thread."""
        # Track operation for checkpoint management
        track_operation(state)

        # Get file ID first
        # Normalize path to handle both absolute and relative paths
        base_dir = state.get("base_directory")
        normalized_path = normalize_path_for_lookup(file_path, base_dir)
        result = conn.execute(
            "SELECT id FROM files WHERE path = ?", [normalized_path]
        ).fetchone()

        if not result:
            return False

        file_id = result[0]

        # Delete in correct order due to foreign key constraints
        # 1. Delete embeddings first from all embedding tables
        embedding_tables = self._executor_get_all_embedding_tables(conn, state)
        for table_name in embedding_tables:
            conn.execute(
                f"""
                DELETE FROM {table_name}
                WHERE chunk_id IN (SELECT id FROM chunks WHERE file_id = ?)
                """,
                [file_id],
            )

        # 2. Delete chunks
        conn.execute("DELETE FROM chunks WHERE file_id = ?", [file_id])

        # 3. Delete file
        conn.execute("DELETE FROM files WHERE id = ?", [file_id])

        logger.debug(f"File {file_path} and all associated data deleted")
        return True

    def insert_chunk(self, chunk: Chunk) -> int:
        """Insert chunk record and return chunk ID - delegate to chunk repository."""
        return self._chunk_repository.insert_chunk(chunk)

    def insert_chunks_batch(self, chunks: list[Chunk]) -> list[int]:
        """Insert multiple chunks in batch using optimized DuckDB bulk loading - delegate to chunk repository.

        # PERFORMANCE: 250x faster than single inserts
        # OPTIMAL_BATCH: 5000 chunks (benchmarked)
        # PATTERN: Uses VALUES clause for bulk insert
        """
        return self._execute_in_db_thread_sync("insert_chunks_batch", chunks)

    def _executor_insert_chunks_batch(
        self, conn: Any, state: dict[str, Any], chunks: list[Chunk]
    ) -> list[int]:
        """Executor method for insert_chunks_batch - runs in DB thread."""
        if not chunks:
            return []

        # Track operation for checkpoint management
        track_operation(state)

        # Prepare data for bulk insert
        chunk_data = []
        for chunk in chunks:
            chunk_data.append(
                (
                    chunk.file_id,
                    chunk.chunk_type.value,
                    chunk.symbol or "",
                    chunk.code,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.start_byte,
                    chunk.end_byte,
                    chunk.language.value if chunk.language else None,
                )
            )

        # Create temporary table
        conn.execute("""
            CREATE TEMPORARY TABLE temp_chunks (
                file_id INTEGER,
                chunk_type TEXT,
                symbol TEXT,
                code TEXT,
                start_line INTEGER,
                end_line INTEGER,
                start_byte INTEGER,
                end_byte INTEGER,
                language TEXT
            )
        """)

        # Bulk insert into temp table
        conn.executemany(
            """
            INSERT INTO temp_chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            chunk_data,
        )

        # Insert from temp to main table with RETURNING
        result = conn.execute("""
            INSERT INTO chunks (file_id, chunk_type, symbol, code, start_line, end_line,
                              start_byte, end_byte, language)
            SELECT * FROM temp_chunks
            RETURNING id
        """)

        chunk_ids = [row[0] for row in result.fetchall()]

        # Drop temp table
        conn.execute("DROP TABLE temp_chunks")

        return chunk_ids

    def get_chunk_by_id(
        self, chunk_id: int, as_model: bool = False
    ) -> dict[str, Any] | Chunk | None:
        """Get chunk record by ID - delegate to chunk repository."""
        return self._chunk_repository.get_chunk_by_id(chunk_id, as_model)

    def get_chunks_by_file_id(
        self, file_id: int, as_model: bool = False
    ) -> list[dict[str, Any] | Chunk]:
        """Get all chunks for a specific file - delegate to chunk repository."""
        return self._execute_in_db_thread_sync(
            "get_chunks_by_file_id", file_id, as_model
        )

    def _executor_get_chunks_by_file_id(
        self, conn: Any, state: dict[str, Any], file_id: int, as_model: bool
    ) -> list[dict[str, Any] | Chunk]:
        """Executor method for get_chunks_by_file_id - runs in DB thread."""
        results = conn.execute(
            """
            SELECT id, file_id, chunk_type, symbol, code, start_line, end_line,
                   start_byte, end_byte, language, created_at, updated_at
            FROM chunks
            WHERE file_id = ?
            ORDER BY start_line, start_byte
        """,
            [file_id],
        ).fetchall()

        chunks = []
        for row in results:
            chunk_dict = {
                "id": row[0],
                "file_id": row[1],
                "chunk_type": row[2],
                "symbol": row[3],
                "code": row[4],
                "start_line": row[5],
                "end_line": row[6],
                "start_byte": row[7],
                "end_byte": row[8],
                "language": row[9],
                "created_at": row[10],
                "updated_at": row[11],
            }

            if as_model:
                chunk = Chunk(
                    id=chunk_dict["id"],
                    file_id=chunk_dict["file_id"],
                    chunk_type=ChunkType(chunk_dict["chunk_type"]),
                    symbol=chunk_dict["symbol"],
                    code=chunk_dict["code"],
                    start_line=chunk_dict["start_line"],
                    end_line=chunk_dict["end_line"],
                    start_byte=chunk_dict["start_byte"],
                    end_byte=chunk_dict["end_byte"],
                    language=Language(chunk_dict["language"])
                    if chunk_dict["language"]
                    else None,
                )
                chunks.append(chunk)
            else:
                chunks.append(chunk_dict)

        return chunks

    def delete_file_chunks(self, file_id: int) -> None:
        """Delete all chunks for a file - delegate to chunk repository."""
        self._execute_in_db_thread_sync("delete_file_chunks", file_id)

    def _executor_delete_file_chunks(
        self, conn: Any, state: dict[str, Any], file_id: int
    ) -> None:
        """Executor method for delete_file_chunks - runs in DB thread."""
        # Track operation for checkpoint management
        track_operation(state)

        conn.execute("DELETE FROM chunks WHERE file_id = ?", [file_id])

    def _executor_delete_chunk(
        self, conn: Any, state: dict[str, Any], chunk_id: int
    ) -> None:
        """Executor method for delete_chunk - runs in DB thread."""
        # Track operation
        track_operation(state)

        # Delete embeddings first to avoid foreign key constraint
        # Get all embedding tables
        result = conn.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_name LIKE 'embeddings_%'
        """).fetchall()

        for (table_name,) in result:
            conn.execute(f"DELETE FROM {table_name} WHERE chunk_id = ?", [chunk_id])

        # Then delete the chunk
        conn.execute("DELETE FROM chunks WHERE id = ?", [chunk_id])

    def delete_chunk(self, chunk_id: int) -> None:
        """Delete a single chunk by ID with proper foreign key handling."""
        self._execute_in_db_thread_sync("delete_chunk", chunk_id)

    def update_chunk(self, chunk_id: int, **kwargs) -> None:
        """Update chunk record with new values - delegate to chunk repository."""
        self._chunk_repository.update_chunk(chunk_id, **kwargs)

    def _executor_insert_chunk_single(
        self, conn: Any, state: dict[str, Any], chunk: Chunk
    ) -> int:
        """Executor method for insert_chunk - runs in DB thread."""
        # Track operation for checkpoint management
        track_operation(state)

        result = conn.execute(
            """
            INSERT INTO chunks (file_id, chunk_type, symbol, code, start_line, end_line,
                              start_byte, end_byte, language)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id
        """,
            [
                chunk.file_id,
                chunk.chunk_type.value if chunk.chunk_type else None,
                chunk.symbol,
                chunk.code,
                chunk.start_line,
                chunk.end_line,
                chunk.start_byte,
                chunk.end_byte,
                chunk.language.value if chunk.language else None,
            ],
        ).fetchone()

        return result[0] if result else 0

    def _executor_get_chunk_by_id_query(
        self, conn: Any, state: dict[str, Any], chunk_id: int
    ) -> Any:
        """Executor method for get_chunk_by_id query - runs in DB thread."""
        return conn.execute(
            """
            SELECT id, file_id, chunk_type, symbol, code, start_line, end_line,
                   start_byte, end_byte, language, created_at, updated_at
            FROM chunks WHERE id = ?
        """,
            [chunk_id],
        ).fetchone()

    def _executor_get_chunks_by_file_id_query(
        self, conn: Any, state: dict[str, Any], file_id: int
    ) -> list:
        """Executor method for get_chunks_by_file_id query - runs in DB thread."""
        return conn.execute(
            """
            SELECT id, file_id, chunk_type, symbol, code, start_line, end_line,
                   start_byte, end_byte, language, created_at, updated_at
            FROM chunks WHERE file_id = ?
            ORDER BY start_line
        """,
            [file_id],
        ).fetchall()

    def _executor_update_chunk_query(
        self, conn: Any, state: dict[str, Any], chunk_id: int, query: str, values: list
    ) -> None:
        """Executor method for update_chunk query - runs in DB thread."""
        # Track operation for checkpoint management
        track_operation(state)
        conn.execute(query, values)

    def _executor_get_all_chunks_with_metadata_query(
        self, conn: Any, state: dict[str, Any], query: str
    ) -> list:
        """Executor method for get_all_chunks_with_metadata query - runs in DB thread."""
        return conn.execute(query).fetchall()

    def _executor_get_file_by_id_query(
        self, conn: Any, state: dict[str, Any], file_id: int, as_model: bool
    ) -> dict[str, Any] | File | None:
        """Executor method for get_file_by_id query - runs in DB thread."""
        result = conn.execute(
            """
            SELECT id, path, name, extension, size, modified_time, language, created_at, updated_at
            FROM files WHERE id = ?
        """,
            [file_id],
        ).fetchone()

        if not result:
            return None

        file_dict = {
            "id": result[0],
            "path": result[1],
            "name": result[2],
            "extension": result[3],
            "size": result[4],
            "modified_time": result[5],
            "language": result[6],
            "created_at": result[7],
            "updated_at": result[8],
        }

        if as_model:
            return File(
                path=result[1],
                mtime=result[5],
                size_bytes=result[4],
                language=Language(result[6]) if result[6] else Language.UNKNOWN,
            )

        return file_dict

    def insert_embedding(self, embedding: Embedding) -> int:
        """Insert embedding record and return embedding ID - delegate to embedding repository."""
        return self._embedding_repository.insert_embedding(embedding)

    def insert_embeddings_batch(
        self,
        embeddings_data: list[dict],
        batch_size: int | None = None,
        connection=None,
    ) -> int:
        """Insert multiple embedding vectors with HNSW index optimization - delegate to embedding repository.

        # OPTIMIZATION: Drops HNSW indexes for batches >50
        # PERFORMANCE: 60s → 5s for 10k embeddings (12x speedup)
        # RECOVERY: Indexes recreated after bulk insert
        """
        # Note: connection parameter is ignored in executor pattern
        return self._execute_in_db_thread_sync(
            "insert_embeddings_batch", embeddings_data, batch_size
        )

    def _executor_insert_embeddings_batch(
        self,
        conn: Any,
        state: dict[str, Any],
        embeddings_data: list[dict],
        batch_size: int | None,
    ) -> int:
        """Executor method for insert_embeddings_batch - runs in DB thread."""
        if not embeddings_data:
            return 0

        # Track operation for checkpoint management
        track_operation(state)

        # Group embeddings by dimension
        embeddings_by_dims = {}
        for emb_data in embeddings_data:
            dims = emb_data["dims"]
            if dims not in embeddings_by_dims:
                embeddings_by_dims[dims] = []
            embeddings_by_dims[dims].append(emb_data)

        total_inserted = 0

        # Insert into dimension-specific tables
        for dims, dim_embeddings in embeddings_by_dims.items():
            # Ensure table exists
            table_name = self._executor_ensure_embedding_table_exists(conn, state, dims)

            # Prepare batch data
            batch_data = []
            for emb in dim_embeddings:
                batch_data.append(
                    (
                        emb["chunk_id"],
                        emb["provider"],
                        emb["model"],
                        emb["embedding"],
                        dims,
                    )
                )

            # Insert in batches if specified
            if batch_size:
                for i in range(0, len(batch_data), batch_size):
                    batch = batch_data[i : i + batch_size]
                    conn.executemany(
                        f"""
                        INSERT INTO {table_name} (chunk_id, provider, model, embedding, dims)
                        VALUES (?, ?, ?, ?, ?)
                    """,
                        batch,
                    )
                    total_inserted += len(batch)
            else:
                # Insert all at once
                conn.executemany(
                    f"""
                    INSERT INTO {table_name} (chunk_id, provider, model, embedding, dims)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    batch_data,
                )
                total_inserted += len(batch_data)

        return total_inserted

    def get_embedding_by_chunk_id(
        self, chunk_id: int, provider: str, model: str
    ) -> Embedding | None:
        """Get embedding for specific chunk, provider, and model - delegate to embedding repository."""
        return self._embedding_repository.get_embedding_by_chunk_id(
            chunk_id, provider, model
        )

    def get_existing_embeddings(
        self, chunk_ids: list[int], provider: str, model: str
    ) -> set[int]:
        """Get set of chunk IDs that already have embeddings for given provider/model - delegate to embedding repository."""
        return self._execute_in_db_thread_sync(
            "get_existing_embeddings", chunk_ids, provider, model
        )

    def _executor_get_existing_embeddings(
        self,
        conn: Any,
        state: dict[str, Any],
        chunk_ids: list[int],
        provider: str,
        model: str,
    ) -> set[int]:
        """Executor method for get_existing_embeddings - runs in DB thread."""
        if not chunk_ids:
            return set()

        # Get all embedding tables
        embedding_tables = self._executor_get_all_embedding_tables(conn, state)
        existing_chunks = set()

        # Check each dimension-specific table
        for table_name in embedding_tables:
            # Use parameterized placeholders for chunk IDs
            placeholders = ", ".join(["?" for _ in chunk_ids])
            query = f"""
                SELECT DISTINCT chunk_id 
                FROM {table_name}
                WHERE chunk_id IN ({placeholders})
                AND provider = ? AND model = ?
            """

            params = chunk_ids + [provider, model]
            results = conn.execute(query, params).fetchall()

            for row in results:
                existing_chunks.add(row[0])

        return existing_chunks

    def delete_embeddings_by_chunk_id(self, chunk_id: int) -> None:
        """Delete all embeddings for a specific chunk - delegate to embedding repository."""
        self._embedding_repository.delete_embeddings_by_chunk_id(chunk_id)

    def get_all_chunks_with_metadata(self) -> list[dict[str, Any]]:
        """Get all chunks with their metadata including file paths - delegate to chunk repository."""
        return self._execute_in_db_thread_sync("get_all_chunks_with_metadata")

    def _executor_get_all_chunks_with_metadata(
        self, conn: Any, state: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Executor method for get_all_chunks_with_metadata - runs in DB thread."""
        query = """
            SELECT 
                c.id as chunk_id,
                c.file_id,
                c.chunk_type,
                c.symbol,
                c.code,
                c.start_line,
                c.end_line,
                c.language as chunk_language,
                f.path as file_path,
                f.language as file_language
            FROM chunks c
            JOIN files f ON c.file_id = f.id
            ORDER BY f.path, c.start_line
        """

        results = conn.execute(query).fetchall()

        chunks_with_metadata = []
        for row in results:
            chunks_with_metadata.append(
                {
                    "chunk_id": row[0],
                    "file_id": row[1],
                    "chunk_type": row[2],
                    "symbol": row[3],
                    "code": row[4],
                    "start_line": row[5],
                    "end_line": row[6],
                    "chunk_language": row[7],
                    "file_path": row[8],  # Keep stored format
                    "file_language": row[9],
                }
            )

        return chunks_with_metadata

    def _validate_and_normalize_path_filter(
        self, path_filter: str | None
    ) -> str | None:
        """Validate and normalize path filter for security and consistency.

        Args:
            path_filter: User-provided path filter

        Returns:
            Normalized path filter safe for SQL LIKE queries, or None

        Raises:
            ValueError: If path contains dangerous patterns
        """
        if path_filter is None:
            return None

        # Remove leading/trailing whitespace
        normalized = path_filter.strip()

        if not normalized:
            return None

        # Security checks - prevent directory traversal
        dangerous_patterns = ["..", "~", "*", "?", "[", "]", "\0", "\n", "\r"]
        for pattern in dangerous_patterns:
            if pattern in normalized:
                raise ValueError(f"Path filter contains forbidden pattern: {pattern}")

        # Normalize path separators to forward slashes
        normalized = normalized.replace("\\", "/")

        # Remove leading slashes to ensure relative paths
        normalized = normalized.lstrip("/")

        # Ensure trailing slash for directory patterns
        if (
            normalized
            and not normalized.endswith("/")
            and "." not in normalized.split("/")[-1]
        ):
            normalized += "/"

        return normalized

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
        """Perform semantic vector search using HNSW index with multi-dimension support.

        # PERFORMANCE: HNSW index provides ~5ms query time
        # ACCURACY: Cosine similarity metric
        # OPTIMIZATION: Dimension-specific tables (1536D, 3072D, etc.)
        """
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

    def _executor_search_semantic(
        self,
        conn: Any,
        state: dict[str, Any],
        query_embedding: list[float],
        provider: str,
        model: str,
        page_size: int,
        offset: int,
        threshold: float | None,
        path_filter: str | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Executor method for search_semantic - runs in DB thread."""
        try:
            # Validate and normalize path filter
            normalized_path = self._validate_and_normalize_path_filter(path_filter)

            # Detect dimensions from query embedding
            query_dims = len(query_embedding)
            table_name = f"embeddings_{query_dims}"

            # Check if table exists for these dimensions
            if not self._executor_table_exists(conn, state, table_name):
                logger.warning(
                    f"No embeddings table found for {query_dims} dimensions ({table_name})"
                )
                return [], {
                    "offset": offset,
                    "page_size": page_size,
                    "has_more": False,
                    "total": 0,
                }

            # Build query with dimension-specific table
            query = f"""
                SELECT
                    c.id as chunk_id,
                    c.symbol,
                    c.code,
                    c.chunk_type,
                    c.start_line,
                    c.end_line,
                    f.path as file_path,
                    f.language,
                    array_cosine_similarity(e.embedding, ?::FLOAT[{query_dims}]) as similarity
                FROM {table_name} e
                JOIN chunks c ON e.chunk_id = c.id
                JOIN files f ON c.file_id = f.id
                WHERE e.provider = ? AND e.model = ?
            """

            params = [query_embedding, provider, model]

            if threshold is not None:
                query += f" AND array_cosine_similarity(e.embedding, ?::FLOAT[{query_dims}]) >= ?"
                params.append(query_embedding)
                params.append(threshold)

            if normalized_path is not None:
                query += " AND f.path LIKE ?"
                params.append(
                    f"{normalized_path}%"
                )  # Simple prefix match on relative paths

            # Get total count for pagination
            # Build count query separately to avoid string replacement issues
            count_query = f"""
                SELECT COUNT(*)
                FROM {table_name} e
                JOIN chunks c ON e.chunk_id = c.id
                JOIN files f ON c.file_id = f.id
                WHERE e.provider = ? AND e.model = ?
            """

            count_params = [provider, model]

            if threshold is not None:
                count_query += f" AND array_cosine_similarity(e.embedding, ?::FLOAT[{query_dims}]) >= ?"
                count_params.extend([query_embedding, threshold])

            if normalized_path is not None:
                count_query += " AND f.path LIKE ?"
                count_params.append(
                    f"{normalized_path}%"
                )  # Simple prefix match on relative paths

            total_count = conn.execute(count_query, count_params).fetchone()[0]

            query += " ORDER BY similarity DESC LIMIT ? OFFSET ?"
            params.extend([page_size, offset])

            results = conn.execute(query, params).fetchall()

            result_list = [
                {
                    "chunk_id": result[0],
                    "symbol": result[1],
                    "content": result[2],
                    "chunk_type": result[3],
                    "start_line": result[4],
                    "end_line": result[5],
                    "file_path": result[6],  # Keep stored format
                    "language": result[7],
                    "similarity": result[8],
                }
                for result in results
            ]

            pagination = {
                "offset": offset,
                "page_size": page_size,
                "has_more": offset + page_size < total_count,
                "next_offset": offset + page_size
                if offset + page_size < total_count
                else None,
                "total": total_count,
            }

            return result_list, pagination

        except Exception as e:
            logger.error(f"Failed to perform semantic search: {e}")
            return [], {
                "offset": offset,
                "page_size": page_size,
                "has_more": False,
                "total": 0,
            }

    def search_regex(
        self,
        pattern: str,
        page_size: int = 10,
        offset: int = 0,
        path_filter: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Perform regex search on code content."""
        return self._execute_in_db_thread_sync(
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

    def _executor_search_regex(
        self,
        conn: Any,
        state: dict[str, Any],
        pattern: str,
        page_size: int,
        offset: int,
        path_filter: str | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Executor method for search_regex - runs in DB thread."""
        try:
            # Validate and normalize path filter
            normalized_path = self._validate_and_normalize_path_filter(path_filter)

            # Build base WHERE clause
            where_conditions = ["regexp_matches(c.code, ?)"]
            params = [pattern]

            if normalized_path is not None:
                where_conditions.append("f.path LIKE ?")
                params.append(
                    f"{normalized_path}%"
                )  # Simple prefix match on relative paths

            where_clause = " AND ".join(where_conditions)

            # Get total count for pagination
            count_query = f"""
                SELECT COUNT(*)
                FROM chunks c
                JOIN files f ON c.file_id = f.id
                WHERE {where_clause}
            """
            total_count = conn.execute(count_query, params).fetchone()[0]

            # Get results
            results_query = f"""
                SELECT
                    c.id as chunk_id,
                    c.symbol,
                    c.code,
                    c.chunk_type,
                    c.start_line,
                    c.end_line,
                    f.path as file_path,
                    f.language
                FROM chunks c
                JOIN files f ON c.file_id = f.id
                WHERE {where_clause}
                ORDER BY f.path, c.start_line
                LIMIT ? OFFSET ?
            """
            results = conn.execute(
                results_query, params + [page_size, offset]
            ).fetchall()

            result_list = [
                {
                    "chunk_id": result[0],
                    "name": result[1],
                    "content": result[2],
                    "chunk_type": result[3],
                    "start_line": result[4],
                    "end_line": result[5],
                    "file_path": result[6],  # Keep stored format
                    "language": result[7],
                }
                for result in results
            ]

            pagination = {
                "offset": offset,
                "page_size": page_size,
                "has_more": offset + page_size < total_count,
                "next_offset": offset + page_size
                if offset + page_size < total_count
                else None,
                "total": total_count,
            }

            return result_list, pagination

        except Exception as e:
            logger.error(f"Failed to perform regex search: {e}")
            return [], {
                "offset": offset,
                "page_size": page_size,
                "has_more": False,
                "total": 0,
            }

    def find_similar_chunks(
        self,
        chunk_id: int,
        provider: str,
        model: str,
        limit: int = 10,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Find chunks similar to the given chunk using its embedding."""
        return self._execute_in_db_thread_sync(
            "find_similar_chunks", chunk_id, provider, model, limit, threshold
        )

    def _executor_find_similar_chunks(
        self,
        conn: Any,
        state: dict[str, Any],
        chunk_id: int,
        provider: str,
        model: str,
        limit: int,
        threshold: float | None,
    ) -> list[dict[str, Any]]:
        """Executor method for find_similar_chunks - runs in DB thread."""
        try:
            # Find which table contains this chunk's embedding (reuse existing pattern)
            embedding_tables = self._executor_get_all_embedding_tables(conn, state)
            target_embedding = None
            dims = None
            table_name = None

            # logger.debug(f"Looking for embedding: chunk_id={chunk_id}, provider='{provider}', model='{model}'")
            # logger.debug(f"Available embedding tables: {embedding_tables}")

            for table in embedding_tables:
                result = conn.execute(
                    f"""
                    SELECT embedding
                    FROM {table}
                    WHERE chunk_id = ? AND provider = ? AND model = ?
                    LIMIT 1
                """,
                    [chunk_id, provider, model],
                ).fetchone()

                if result:
                    target_embedding = result[0]
                    # Extract dimensions from table name (e.g., "embeddings_1536" -> 1536)
                    dims_match = re.match(r"embeddings_(\d+)", table)
                    if dims_match:
                        dims = int(dims_match.group(1))
                        table_name = table
                        # logger.debug(f"Found embedding in table {table} for chunk_id={chunk_id}")
                        break
                else:
                    # Debug what's actually in this table for this chunk
                    all_for_chunk = conn.execute(
                        f"""
                        SELECT provider, model, chunk_id
                        FROM {table}
                        WHERE chunk_id = ?
                    """,
                        [chunk_id],
                    ).fetchall()
                    # if all_for_chunk:
                    #     logger.debug(f"Table {table} has chunk_id={chunk_id} but with different provider/model: {all_for_chunk}")

            if not target_embedding or dims is None:
                # Show what providers/models are actually available for this chunk
                all_providers_models = []
                for table in embedding_tables:
                    results = conn.execute(
                        f"""
                        SELECT DISTINCT provider, model
                        FROM {table}
                        WHERE chunk_id = ?
                    """,
                        [chunk_id],
                    ).fetchall()
                    all_providers_models.extend(results)

                logger.warning(
                    f"No embedding found for chunk_id={chunk_id}, provider='{provider}', model='{model}'"
                )
                logger.warning(
                    f"Available provider/model combinations for this chunk: {all_providers_models}"
                )
                return []

            embedding_type = f"FLOAT[{dims}]"

            # Use the embedding to find similar chunks
            similarity_metric = "cosine"  # Default for semantic search
            threshold_condition = (
                f"AND distance <= {threshold}" if threshold is not None else ""
            )

            # Query for similar chunks (exclude the original chunk)
            # Cast the target embedding to match the table's embedding type
            query = f"""
                SELECT 
                    c.id as chunk_id,
                    c.symbol as name,
                    c.code as content,
                    c.chunk_type,
                    c.start_line,
                    c.end_line,
                    f.path as file_path,
                    f.language,
                    array_cosine_distance(e.embedding, ?::{embedding_type}) as distance
                FROM {table_name} e
                JOIN chunks c ON e.chunk_id = c.id
                JOIN files f ON c.file_id = f.id
                WHERE e.provider = ?
                AND e.model = ?
                AND c.id != ?
                {threshold_condition}
                ORDER BY distance ASC
                LIMIT ?
            """

            results = conn.execute(
                query, [target_embedding, provider, model, chunk_id, limit]
            ).fetchall()

            # Format results
            result_list = [
                {
                    "chunk_id": result[0],
                    "name": result[1],
                    "content": result[2],
                    "chunk_type": result[3],
                    "start_line": result[4],
                    "end_line": result[5],
                    "file_path": result[6],  # Keep stored format
                    "language": result[7],
                    "score": 1.0 - result[8],  # Convert distance to similarity score
                }
                for result in results
            ]

            return result_list

        except Exception as e:
            logger.error(f"Failed to find similar chunks: {e}")
            return []

    def search_by_embedding(
        self,
        query_embedding: list[float],
        provider: str,
        model: str,
        limit: int = 10,
        threshold: float | None = None,
        path_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find chunks similar to the given embedding vector."""
        return self._execute_in_db_thread_sync(
            "search_by_embedding",
            query_embedding,
            provider,
            model,
            limit,
            threshold,
            path_filter,
        )

    def _executor_search_by_embedding(
        self,
        conn: Any,
        state: dict[str, Any],
        query_embedding: list[float],
        provider: str,
        model: str,
        limit: int,
        threshold: float | None,
        path_filter: str | None,
    ) -> list[dict[str, Any]]:
        """Executor method for search_by_embedding - runs in DB thread."""
        try:
            # Detect dimensions from query embedding (reuse pattern from search_semantic)
            query_dims = len(query_embedding)
            table_name = f"embeddings_{query_dims}"
            embedding_type = f"FLOAT[{query_dims}]"

            # Check if table exists for these dimensions (reuse existing validation pattern)
            if not self._executor_table_exists(conn, state, table_name):
                logger.warning(
                    f"No embeddings table found for {query_dims} dimensions ({table_name})"
                )
                return []

            # Build path filter condition
            path_condition = ""
            query_params = [query_embedding, provider, model, limit]

            if path_filter:
                # Convert relative path to SQL pattern
                path_pattern = f"%{path_filter}%"
                path_condition = "AND f.path LIKE ?"
                query_params.insert(-1, path_pattern)  # Insert before limit

            # Build threshold condition
            threshold_condition = (
                f"AND distance <= {threshold}" if threshold is not None else ""
            )

            # Query for similar chunks using the provided embedding
            query = f"""
                SELECT 
                    c.id as chunk_id,
                    c.symbol as name,
                    c.code as content,
                    c.chunk_type,
                    c.start_line,
                    c.end_line,
                    f.path as file_path,
                    f.language,
                    array_cosine_distance(e.embedding, ?::{embedding_type}) as distance
                FROM {table_name} e
                JOIN chunks c ON e.chunk_id = c.id
                JOIN files f ON c.file_id = f.id
                WHERE e.provider = ?
                AND e.model = ?
                {path_condition}
                {threshold_condition}
                ORDER BY distance ASC
                LIMIT ?
            """

            results = conn.execute(query, query_params).fetchall()

            # Format results
            result_list = [
                {
                    "chunk_id": result[0],
                    "name": result[1],
                    "content": result[2],
                    "chunk_type": result[3],
                    "start_line": result[4],
                    "end_line": result[5],
                    "file_path": result[6],  # Keep stored format
                    "language": result[7],
                    "score": 1.0 - result[8],  # Convert distance to similarity score
                }
                for result in results
            ]

            return result_list

        except Exception as e:
            logger.error(f"Failed to search by embedding: {e}")
            return []

    def search_text(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Perform full-text search on code content."""
        return self._execute_in_db_thread_sync("search_text", query, limit)

    def _executor_search_text(
        self, conn: Any, state: dict[str, Any], query: str, limit: int
    ) -> list[dict[str, Any]]:
        """Executor method for search_text - runs in DB thread."""
        try:
            # Simple text search using LIKE operator
            search_pattern = f"%{query}%"

            results = conn.execute(
                """
                SELECT
                    c.id as chunk_id,
                    c.symbol,
                    c.code,
                    c.chunk_type,
                    c.start_line,
                    c.end_line,
                    f.path as file_path,
                    f.language
                FROM chunks c
                JOIN files f ON c.file_id = f.id
                WHERE c.code LIKE ? OR c.symbol LIKE ?
                ORDER BY f.path, c.start_line
                LIMIT ?
            """,
                [search_pattern, search_pattern, limit],
            ).fetchall()

            return [
                {
                    "chunk_id": result[0],
                    "name": result[1],
                    "content": result[2],
                    "chunk_type": result[3],
                    "start_line": result[4],
                    "end_line": result[5],
                    "file_path": result[6],  # Keep stored format
                    "language": result[7],
                }
                for result in results
            ]

        except Exception as e:
            logger.error(f"Failed to perform text search: {e}")
            return []

    def get_stats(self) -> dict[str, int]:
        """Get database statistics (file count, chunk count, etc.)."""
        return self._execute_in_db_thread_sync("get_stats")

    def _executor_get_stats(self, conn: Any, state: dict[str, Any]) -> dict[str, int]:
        """Executor method for get_stats - runs in DB thread."""
        try:
            # Get counts from each table
            file_count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

            # Count embeddings across all dimension-specific tables
            embedding_count = 0
            embedding_tables = self._executor_get_all_embedding_tables(conn, state)
            for table_name in embedding_tables:
                count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                embedding_count += count

            # Get unique providers/models across all embedding tables
            provider_results = []
            for table_name in embedding_tables:
                results = conn.execute(f"""
                    SELECT DISTINCT provider, model, COUNT(*) as count
                    FROM {table_name}
                    GROUP BY provider, model
                """).fetchall()
                provider_results.extend(results)

            providers = {}
            for result in provider_results:
                key = f"{result[0]}/{result[1]}"
                providers[key] = result[2]

            # Convert providers dict to count for interface compliance
            provider_count = len(providers)
            return {
                "files": file_count,
                "chunks": chunk_count,
                "embeddings": embedding_count,
                "providers": provider_count,
            }

        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {"files": 0, "chunks": 0, "embeddings": 0, "providers": 0}

    def get_file_stats(self, file_id: int) -> dict[str, Any]:
        """Get statistics for a specific file - delegate to file repository."""
        return self._file_repository.get_file_stats(file_id)

    def get_provider_stats(self, provider: str, model: str) -> dict[str, Any]:
        """Get statistics for a specific embedding provider/model."""
        return self._execute_in_db_thread_sync("get_provider_stats", provider, model)

    def _executor_get_provider_stats(
        self, conn: Any, state: dict[str, Any], provider: str, model: str
    ) -> dict[str, Any]:
        """Executor method for get_provider_stats - runs in DB thread."""
        try:
            # Get embedding count across all embedding tables
            embedding_count = 0
            file_ids = set()
            dims = 0
            embedding_tables = self._executor_get_all_embedding_tables(conn, state)

            for table_name in embedding_tables:
                # Count embeddings for this provider/model in this table
                count = conn.execute(
                    f"""
                    SELECT COUNT(*) FROM {table_name}
                    WHERE provider = ? AND model = ?
                """,
                    [provider, model],
                ).fetchone()[0]
                embedding_count += count

                # Get unique file IDs for this provider/model in this table
                file_results = conn.execute(
                    f"""
                    SELECT DISTINCT c.file_id
                    FROM {table_name} e
                    JOIN chunks c ON e.chunk_id = c.id
                    WHERE e.provider = ? AND e.model = ?
                """,
                    [provider, model],
                ).fetchall()
                file_ids.update(result[0] for result in file_results)

                # Get dimensions (should be consistent across all tables for same provider/model)
                if count > 0 and dims == 0:
                    dims_result = conn.execute(
                        f"""
                        SELECT DISTINCT dims FROM {table_name}
                        WHERE provider = ? AND model = ?
                        LIMIT 1
                    """,
                        [provider, model],
                    ).fetchone()
                    if dims_result:
                        dims = dims_result[0]

            file_count = len(file_ids)

            return {
                "provider": provider,
                "model": model,
                "embeddings": embedding_count,
                "files": file_count,
                "dimensions": dims,
            }

        except Exception as e:
            logger.error(f"Failed to get provider stats for {provider}/{model}: {e}")
            return {
                "provider": provider,
                "model": model,
                "embeddings": 0,
                "files": 0,
                "dimensions": 0,
            }

    def execute_query(
        self, query: str, params: list[Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a SQL query and return results."""
        return self._execute_in_db_thread_sync("execute_query", query, params)

    def _executor_execute_query(
        self, conn: Any, state: dict[str, Any], query: str, params: list[Any] | None
    ) -> list[dict[str, Any]]:
        """Executor method for execute_query - runs in DB thread."""
        try:
            if params:
                cursor = conn.execute(query, params)
            else:
                cursor = conn.execute(query)

            results = cursor.fetchall()

            # Convert to list of dictionaries
            if results:
                # Get column names from cursor description
                column_names = [desc[0] for desc in cursor.description]
                return [dict(zip(column_names, row)) for row in results]

            return []

        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            raise

    def _executor_begin_transaction(self, conn: Any, state: dict[str, Any]) -> None:
        """Executor method for begin_transaction - runs in DB thread."""
        # Mark transaction state in executor thread
        state["transaction_active"] = True
        conn.execute("BEGIN TRANSACTION")

    def _executor_commit_transaction(
        self, conn: Any, state: dict[str, Any], force_checkpoint: bool
    ) -> None:
        """Executor method for commit_transaction - runs in DB thread."""
        try:
            conn.execute("COMMIT")

            # Clear transaction state
            state["transaction_active"] = False

            # Handle checkpoint
            if force_checkpoint or state.get("deferred_checkpoint", False):
                try:
                    conn.execute("CHECKPOINT")
                    state["operations_since_checkpoint"] = 0
                    state["last_checkpoint_time"] = time.time()
                    state["deferred_checkpoint"] = False
                    if not os.environ.get("CHUNKHOUND_MCP_MODE"):
                        logger.debug("Transaction committed with checkpoint")
                except Exception as e:
                    if not os.environ.get("CHUNKHOUND_MCP_MODE"):
                        logger.warning(f"Post-commit checkpoint failed: {e}")
        except Exception:
            # Re-raise to be handled by caller
            raise

    def _executor_rollback_transaction(self, conn: Any, state: dict[str, Any]) -> None:
        """Executor method for rollback_transaction - runs in DB thread."""
        conn.execute("ROLLBACK")
        # Clear transaction state
        state["transaction_active"] = False
        state["deferred_checkpoint"] = False

    def optimize_tables(self) -> None:
        """Optimize tables by compacting fragments and rebuilding indexes (provider-specific).

        # DUCKDB_OPTIMIZATION: Automatic via WAL and MVCC
        # CHECKPOINT: Happens at 1GB WAL size
        # MANUAL: Not needed - DuckDB self-optimizes
        """
        # DuckDB automatically manages table optimization through its WAL and MVCC system
        # No manual optimization needed for DuckDB
        pass
