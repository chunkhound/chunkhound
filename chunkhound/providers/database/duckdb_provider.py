"""DuckDB provider implementation for ChunkHound - concrete database provider using DuckDB.

# FILE_CONTEXT: High-performance analytical database provider
# CRITICAL: Single-threaded access enforced by SerialDatabaseProvider
# PERFORMANCE: Vector search via ShardManager/USearch, bulk operations optimized

## PERFORMANCE_CHARACTERISTICS
- Bulk inserts: 5000 rows optimal batch size
- Vector search: ShardManager with USearch HNSW indexes (external files)
- WAL mode: Automatic checkpointing, 1GB limit
"""

import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from chunkhound.core.models import Chunk, Embedding, File
from chunkhound.core.types.common import ChunkType, Language
from chunkhound.core.utils import normalize_path_for_lookup

# Import existing components that will be used by the provider
from chunkhound.core.config.sharding_config import ShardingConfig
from chunkhound.embeddings import EmbeddingManager
from chunkhound.providers.database.duckdb.chunk_repository import DuckDBChunkRepository
from chunkhound.providers.database.duckdb.connection_manager import (
    DuckDBConnectionManager,
    get_compaction_lock_path,
)
from chunkhound.providers.database.duckdb.embedding_repository import (
    DuckDBEmbeddingRepository,
)
from chunkhound.providers.database.duckdb.file_repository import DuckDBFileRepository
from chunkhound.providers.database.like_utils import escape_like_pattern
from chunkhound.providers.database.serial_database_provider import (
    SerialDatabaseProvider,
)
from chunkhound.providers.database.serial_executor import (
    _executor_local,
    signal_heartbeat,
)
from chunkhound.providers.database.shard_manager import ShardManager

# Type hinting only
if TYPE_CHECKING:
    from chunkhound.core.config.database_config import DatabaseConfig
    from chunkhound.providers.database.bulk_indexer import BulkIndexer


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
        sharding_config: ShardingConfig | None = None,
    ):
        """Initialize DuckDB provider.

        Args:
            db_path: Path to DuckDB database file or ":memory:" for in-memory database
            base_directory: Base directory for path normalization
            embedding_manager: Optional embedding manager for vector generation
            config: Database configuration for provider-specific settings
            sharding_config: Configuration for USearch shard management
        """
        # Initialize base class
        super().__init__(db_path, base_directory, embedding_manager, config)

        self.provider_type = "duckdb"  # Identify this as DuckDB provider

        # Store sharding config for ShardManager initialization
        self._sharding_config = sharding_config or ShardingConfig()

        # ShardManager will be initialized in connect() after schema creation
        self.shard_manager: ShardManager | None = None

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

        # Lightweight performance metrics for chunk writes (per-provider lifecycle)
        self._metrics: dict[str, dict[str, float | int]] = {
            "chunks": {
                "files": 0,
                "rows": 0,
                "batches": 0,
                "temp_create_s": 0.0,
                "temp_insert_s": 0.0,
                "main_insert_s": 0.0,
                "temp_drop_s": 0.0,
            }
        }

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
    def db_path(self) -> Path | str:
        """Database connection path or identifier - delegate to connection manager."""
        return self._connection_manager.db_path

    # NOTE: connection property was removed - use executor methods instead.
    # is_connected is inherited from SerialDatabaseProvider (checks executor existence)

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

            # Initialize ShardManager after schema creation
            self._initialize_shard_manager(conn)

            logger.info("Database initialization complete in executor thread")

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    def _initialize_shard_manager(self, conn: Any) -> None:
        """Initialize ShardManager and run fix_pass for index reconciliation.

        Creates ShardManager with configured thresholds and runs fix_pass
        to ensure USearch indexes are consistent with DuckDB state.

        Args:
            conn: Database connection (passed from executor)
        """
        if str(self._connection_manager.db_path) == ":memory:":
            # Skip ShardManager for in-memory databases
            logger.debug("Skipping ShardManager for in-memory database")
            return

        # Determine shard directory (sibling to database file)
        db_path = Path(self._connection_manager.db_path)
        shard_dir = db_path.parent / "shards"

        # Initialize ShardManager with heartbeat callback for timeout extension
        self.shard_manager = ShardManager(
            db_provider=self,
            shard_dir=shard_dir,
            config=self._sharding_config,
            heartbeat_callback=signal_heartbeat,
        )

        # Run fix_pass to reconcile USearch indexes with DuckDB state
        # This ensures indexes are valid before serving reads
        logger.info("Running ShardManager fix_pass for index reconciliation...")
        try:
            self.shard_manager.fix_pass(conn, check_quality=True)
        except Exception as e:
            logger.warning(f"ShardManager fix_pass failed: {e}")
            # Continue - database is still usable, just without shard optimization

    def get_bulk_indexer(self) -> "BulkIndexer | None":
        """Get BulkIndexer for batch operations with deferred quality checks.

        Returns BulkIndexer context manager if ShardManager is available,
        None otherwise (e.g., in-memory databases).

        Usage:
            with provider.get_bulk_indexer() as bulk:
                for batch in batches:
                    provider.insert_embeddings_batch(batch)
                    bulk.on_batch_completed()
        """
        if self.shard_manager is None:
            return None

        from chunkhound.providers.database.bulk_indexer import BulkIndexer

        return BulkIndexer(self, self._sharding_config)

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

    def soft_disconnect(self, skip_checkpoint: bool = False) -> None:
        """Close DB connection without shutting down executor.

        Use for temporary disconnections (e.g., compaction) where reconnection
        will happen soon. For final cleanup, use disconnect() instead.

        Args:
            skip_checkpoint: If True, skip final checkpoint (faster but less safe)
        """
        try:
            super().soft_disconnect(skip_checkpoint)
        finally:
            # Connection manager holds separate file locks that must also be released
            # for compaction to get exclusive access to the database file
            self._connection_manager.disconnect(skip_checkpoint=True)
            # Reset WAL cleanup flag so it runs again on reconnect
            # This is critical for compaction where we swap to a fresh database file
            self._wal_cleanup_done = False

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
        """Perform health check and return status information."""
        return self._execute_in_db_thread_sync("health_check")

    def _executor_health_check(self, conn: Any, state: dict[str, Any]) -> dict[str, Any]:
        """Executor method for health_check - runs in DB thread."""
        return self._connection_manager.health_check(conn)

    def get_connection_info(self) -> dict[str, Any]:
        """Get information about the database configuration."""
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
            # Create table with fixed dimensions
            conn.execute(f"""
                CREATE TABLE {table_name} (
                    id INTEGER PRIMARY KEY DEFAULT nextval('embeddings_id_seq'),
                    chunk_id INTEGER NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    embedding FLOAT[{dims}],
                    dims INTEGER NOT NULL DEFAULT {dims},
                    shard_id UUID REFERENCES vector_shards(shard_id),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create regular indexes for fast lookups
            # Note: Vector search is handled by ShardManager with USearch
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{dims}_chunk_id "
                f"ON {table_name}(chunk_id)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{dims}_provider_model "
                f"ON {table_name}(provider, model)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_embeddings_{dims}_shard "
                f"ON {table_name}(shard_id)"
            )

            logger.info(f"Created {table_name} with regular indexes")
            return table_name

        except Exception as e:
            logger.error(f"Failed to create embedding table for {dims} dimensions: {e}")
            raise

    def should_optimize(self, operation: str = "") -> bool:
        """Check if optimization/compaction is warranted.

        Returns True if there are free blocks pending reclamation (from DELETEs).
        This is consistent with optimize() which skips when free_blocks == 0.

        Args:
            operation: Optional context string for logging (e.g., 'post-chunking')

        Returns:
            True if optimization would reclaim space, False if database is optimal
        """
        try:
            stats = self.get_storage_stats()
            free_blocks = stats.get("free_blocks", 0)
            op_desc = f" ({operation})" if operation else ""

            if free_blocks == 0:
                logger.debug(f"Optimization check{op_desc}: skipping (no free blocks)")
                return False

            logger.debug(
                f"Optimization check{op_desc}: recommended ({free_blocks} free blocks)"
            )
            return True
        except Exception as e:
            # If we can't check, err on the side of optimizing
            logger.debug(f"Optimization check failed, defaulting to True: {e}")
            return True

    def optimize_tables(self) -> None:
        """Optimize tables by checkpointing.

        Performs:
        1. CHECKPOINT - sync WAL to main database, reclaim deleted row space
        """
        self._execute_in_db_thread_sync("optimize_tables")

    def _executor_optimize_tables(self, conn: Any, state: dict[str, Any]) -> None:
        """Executor method for optimize_tables - runs in DB thread."""
        try:
            # Checkpoint for WAL durability and space reclamation
            logger.debug("Running CHECKPOINT for optimization...")
            conn.execute("CHECKPOINT")
            logger.debug("CHECKPOINT completed")

            # Emit bulk metrics for chunk inserts
            try:
                m = self._metrics.get("chunks", {})
                if m.get("files", 0) > 0:
                    logger.info(
                        f"DuckDB chunks bulk metrics: "
                        f"files={m.get('files', 0)} "
                        f"rows={m.get('rows', 0)} "
                        f"batches={m.get('batches', 0)} "
                        f"temp_create_s={m.get('temp_create_s', 0.0):.3f} "
                        f"temp_insert_s={m.get('temp_insert_s', 0.0):.3f} "
                        f"main_insert_s={m.get('main_insert_s', 0.0):.3f} "
                        f"temp_clear_s={m.get('temp_clear_s', 0.0):.3f}"
                    )
            except Exception:
                pass  # Silent failure - don't crash on metrics logging

        except Exception as e:
            logger.warning(f"Optimization failed: {e}")

    def run_fix_pass(self, check_quality: bool = True) -> None:
        """Run ShardManager fix_pass for index maintenance.

        Triggers shard reconciliation including:
        - Orphaned file cleanup
        - Missing index rebuilds
        - Split/merge operations based on thresholds
        - Centroid cache population

        Args:
            check_quality: If True, measure self-recall for each shard
        """
        self._execute_in_db_thread_sync("run_fix_pass", check_quality)

    def _executor_run_fix_pass(
        self, conn: Any, state: dict[str, Any], check_quality: bool
    ) -> None:
        """Executor method for run_fix_pass - runs in DB thread."""
        if self.shard_manager is not None:
            self.shard_manager.fix_pass(conn, check_quality=check_quality)

    def get_storage_stats(self) -> dict[str, Any]:
        """Get DuckDB storage statistics including fragmentation.

        Returns metrics for monitoring and logging. Note that `orphaned_blocks`
        and `fragmentation_ratio` may overcount due to HNSW index structures
        being counted as orphaned when they're legitimate. For optimization
        trigger decisions, use `free_blocks` only (as should_optimize() does).

        Returns:
            Dict with keys: total_blocks, used_blocks, free_blocks,
            accounted_blocks, orphaned_blocks, block_size, fragmentation_ratio
        """
        return self._execute_in_db_thread_sync("get_storage_stats")

    def _executor_get_storage_stats(
        self, conn: Any, state: dict[str, Any]
    ) -> dict[str, Any]:
        """Executor: Calculate storage metrics including orphaned HNSW index blocks.

        Problem: pragma_database_size()'s free_blocks only counts explicitly freed
        blocks. Orphaned HNSW index blocks (from drop/recreate cycles) are still
        counted as used_blocks, causing severe underestimation of fragmentation
        (0.38% for a 96% bloated database).

        Solution: Query pragma_storage_info() to sum actual table data blocks,
        compare to used_blocks to detect orphaned space.

        Fragmentation = 1 - (accounted_blocks / used_blocks)
        """
        # Get database-level metrics
        db_result = conn.execute("""
            SELECT
                block_size,
                total_blocks,
                used_blocks,
                free_blocks
            FROM pragma_database_size()
        """).fetchone()

        block_size = db_result[0]
        total_blocks = db_result[1]
        used_blocks = db_result[2]
        free_blocks = db_result[3]

        # Get actual data blocks from all tables via pragma_storage_info()
        # This reveals orphaned HNSW index blocks that pragma_database_size misses
        accounted_blocks = 0
        try:
            # Get all user tables
            tables = conn.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'main' AND table_type = 'BASE TABLE'
            """).fetchall()

            # Sum unique blocks from each table
            # block_id is the unique block identifier, -1 means no block used
            for (table_name,) in tables:
                result = conn.execute(f"""
                    SELECT COUNT(DISTINCT block_id)
                    FROM pragma_storage_info('{table_name}')
                    WHERE block_id >= 0
                """).fetchone()
                if result and result[0]:
                    accounted_blocks += int(result[0])
        except Exception as e:
            # Fallback: assume all used blocks are accounted (old behavior)
            logger.debug(f"pragma_storage_info() failed, using fallback: {e}")
            accounted_blocks = used_blocks

        # Calculate orphaned blocks (HNSW index bloat from drop/recreate cycles)
        orphaned_blocks = max(0, used_blocks - accounted_blocks)

        # Calculate true fragmentation: orphaned + free blocks as fraction of total
        if used_blocks > 0:
            # Fragmentation = 1 - (actual_data / used_space)
            fragmentation = 1.0 - (accounted_blocks / used_blocks)
            fragmentation = max(0.0, min(1.0, fragmentation))
        else:
            fragmentation = 0.0

        return {
            "total_blocks": total_blocks,
            "used_blocks": used_blocks,
            "free_blocks": free_blocks,
            "accounted_blocks": accounted_blocks,
            "orphaned_blocks": orphaned_blocks,
            "block_size": block_size,
            "fragmentation_ratio": fragmentation,
        }

    def should_compact(self, threshold: float = 0.5) -> tuple[bool, dict[str, Any]]:
        """Check if compaction is warranted based on free blocks ratio.

        Uses free_blocks/total_blocks ratio (reliable) rather than fragmentation_ratio
        which incorrectly counts HNSW index blocks as orphaned.

        Returns:
            Tuple of (should_compact, storage_stats) to avoid duplicate queries.
        """
        stats = self.get_storage_stats()
        total = stats.get("total_blocks", 1)
        free = stats.get("free_blocks", 0)
        free_ratio = free / total if total > 0 else 0.0
        return free_ratio >= threshold, stats

    def create_schema(self) -> None:
        """Create database schema for files, chunks, and embeddings - delegate to connection manager."""
        self._execute_in_db_thread_sync("create_schema")

    def _executor_create_schema(self, conn: Any, state: dict[str, Any]) -> None:
        """Executor method for create_schema - runs in DB thread."""
        logger.info("Creating DuckDB schema")

        try:
            # Create vector_shards table for tracking embedding shards
            # Note: file_path is NOT stored - derived at runtime from shard_id
            # per portability constraint (spec I14: Path Independence)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vector_shards (
                    shard_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    dims INTEGER NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    quantization TEXT NOT NULL DEFAULT 'i8',
                    file_size_bytes BIGINT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CHECK (quantization IN ('f32', 'f16', 'i8'))
                )
            """)

            # Create index for vector_shards lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_shards_dims_provider_model
                    ON vector_shards(dims, provider, model)
            """)

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
                    shard_id UUID REFERENCES vector_shards(shard_id),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Handle schema migrations for existing databases
            # MUST run before shard index creation since migration adds shard_id column
            self._executor_migrate_schema(conn, state)

            # Create regular indexes for 1536-dimensional embeddings
            # Note: Vector search is handled by ShardManager with USearch

            # Create index on chunk_id for efficient deletions
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_1536_chunk_id ON embeddings_1536(chunk_id)
            """)

            # Create index on shard_id for shard-based queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_1536_shard ON embeddings_1536(shard_id)
            """)

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

            # Migrate embedding tables to add shard_id column for sharding support
            embedding_tables = conn.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_name LIKE 'embeddings_%'
            """).fetchall()

            for (table_name,) in embedding_tables:
                # Check if shard_id column exists
                has_shard_id = conn.execute(f"""
                    SELECT COUNT(*) FROM information_schema.columns
                    WHERE table_name = '{table_name}' AND column_name = 'shard_id'
                """).fetchone()[0] > 0

                if not has_shard_id:
                    logger.info(f"Adding shard_id column to {table_name} for sharding support")
                    # Note: DuckDB doesn't support ADD COLUMN with constraints,
                    # so we add the column without FK constraint. The FK is
                    # enforced at the application level via ShardManager.
                    conn.execute(f"""
                        ALTER TABLE {table_name}
                        ADD COLUMN shard_id UUID
                    """)
                    conn.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_{table_name}_shard
                        ON {table_name}(shard_id)
                    """)

        except Exception as e:
            logger.error(f"Failed to migrate schema: {e}")
            raise

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
        import time as _t
        _t0 = _t.perf_counter()
        conn.execute("""
            CREATE TEMPORARY TABLE IF NOT EXISTS temp_chunks (
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
        _t1 = _t.perf_counter()
        conn.execute("DELETE FROM temp_chunks")
        _t_clear = _t.perf_counter()
        # Bulk insert into temp table
        conn.executemany(
            """
            INSERT INTO temp_chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            chunk_data,
        )
        _t2 = _t.perf_counter()
        # Insert from temp to main table with RETURNING
        result = conn.execute("""
            INSERT INTO chunks (file_id, chunk_type, symbol, code, start_line, end_line,
                              start_byte, end_byte, language)
            SELECT * FROM temp_chunks
            RETURNING id
        """)
        _t3 = _t.perf_counter()
        chunk_ids = [row[0] for row in result.fetchall()]
        # Reuse temp table across calls; do not drop here

        # Update metrics
        try:
            m = self._metrics.get("chunks") or {}
            m["files"] = int(m.get("files", 0)) + 1
            m["rows"] = int(m.get("rows", 0)) + len(chunk_data)
            m["batches"] = int(m.get("batches", 0)) + 1
            m["temp_create_s"] = float(m.get("temp_create_s", 0.0)) + (_t1 - _t0)
            m["temp_insert_s"] = float(m.get("temp_insert_s", 0.0)) + (_t2 - _t_clear)
            m["main_insert_s"] = float(m.get("main_insert_s", 0.0)) + (_t3 - _t2)
            m["temp_clear_s"] = float(m.get("temp_clear_s", 0.0)) + (_t_clear - _t1)
            self._metrics["chunks"] = m
        except Exception:
            pass

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
        conn.execute("DELETE FROM chunks WHERE file_id = ?", [file_id])

    def _executor_delete_chunk(
        self, conn: Any, state: dict[str, Any], chunk_id: int
    ) -> None:
        """Executor method for delete_chunk - runs in DB thread."""
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

    def _executor_delete_chunks_batch(
        self, conn: Any, state: dict[str, Any], chunk_ids: list[int]
    ) -> None:
        """Executor method for delete_chunks_batch - runs in DB thread."""
        if not chunk_ids:
            return
        placeholders = ",".join(["?"] * len(chunk_ids))
        # Delete embeddings first across all embedding tables
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'embeddings_%'"
        ).fetchall()
        for (table_name,) in tables:
            conn.execute(
                f"DELETE FROM {table_name} WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            )
        # Delete chunks
        conn.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", chunk_ids)

    def delete_chunk(self, chunk_id: int) -> None:
        """Delete a single chunk by ID with proper foreign key handling."""
        self._execute_in_db_thread_sync("delete_chunk", chunk_id)

    def delete_chunks_batch(self, chunk_ids: list[int]) -> None:
        """Delete multiple chunks by ID efficiently (with embedding cleanup)."""
        self._execute_in_db_thread_sync("delete_chunks_batch", chunk_ids)

    def update_chunk(self, chunk_id: int, **kwargs) -> None:
        """Update chunk record with new values - delegate to chunk repository."""
        self._chunk_repository.update_chunk(chunk_id, **kwargs)

    def _executor_insert_chunk_single(
        self, conn: Any, state: dict[str, Any], chunk: Chunk
    ) -> int:
        """Executor method for insert_chunk - runs in DB thread."""
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
        """Insert multiple embedding vectors with ShardManager coordination.

        Uses external USearch indexes for vector search. Coordinates with
        ShardManager for shard assignment and triggers fix_pass when thresholds
        are crossed.
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

            # Wire to ShardManager for shard assignment and index building
            if self.shard_manager is not None:
                provider = dim_embeddings[0]["provider"]
                model = dim_embeddings[0]["model"]

                # Get IDs of just-inserted embeddings
                emb_ids_result = conn.execute(
                    f"""
                    SELECT id FROM {table_name}
                    WHERE provider = ? AND model = ?
                    ORDER BY id DESC
                    LIMIT ?
                """,
                    [provider, model, len(dim_embeddings)],
                ).fetchall()

                # Query returns DESC (newest first), reverse to match dim_embeddings order
                emb_ids = [row[0] for row in reversed(emb_ids_result)]
                emb_dicts = [
                    {"id": emb_id, "embedding": emb["embedding"]}
                    for emb_id, emb in zip(emb_ids, dim_embeddings)
                ]

                # ShardManager assigns shard_id and creates shard record if needed
                success, needs_fix = self.shard_manager.insert_embeddings(
                    emb_dicts, dims, provider, model, conn
                )

                # Always run fix_pass to ensure centroid cache is up to date
                # fix_pass is idempotent - it's a NOP if indexes are consistent
                self.shard_manager.fix_pass(conn, check_quality=False)

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

    def _executor_insert_embedding(
        self,
        conn: Any,
        state: dict[str, Any],
        embedding: Embedding,
    ) -> int:
        """Executor method for insert_embedding - runs in DB thread.

        Args:
            conn: Database connection
            state: Thread-local state dictionary
            embedding: Embedding model to insert

        Returns:
            The ID of the inserted embedding
        """
        dims = embedding.dims

        # Ensure table exists for these dimensions
        table_name = self._executor_ensure_embedding_table_exists(conn, state, dims)

        # Insert the embedding
        conn.execute(
            f"""
            INSERT INTO {table_name} (chunk_id, provider, model, embedding, dims)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                embedding.chunk_id,
                embedding.provider,
                embedding.model,
                list(embedding.vector),
                dims,
            ],
        )

        # Get the ID of the just-inserted embedding
        result = conn.execute(
            f"SELECT id FROM {table_name} WHERE chunk_id = ? AND provider = ? AND model = ? ORDER BY id DESC LIMIT 1",
            [embedding.chunk_id, embedding.provider, embedding.model],
        ).fetchone()

        embedding_id = result[0] if result else 0

        # Wire to ShardManager if available
        if self.shard_manager is not None:
            emb_dicts = [{"id": embedding_id, "embedding": list(embedding.vector)}]
            success, needs_fix = self.shard_manager.insert_embeddings(
                emb_dicts, dims, embedding.provider, embedding.model, conn
            )
            # Always run fix_pass to ensure centroid cache is up to date
            self.shard_manager.fix_pass(conn, check_quality=False)

        return embedding_id

    def _executor_get_embedding_by_chunk_id(
        self,
        conn: Any,
        state: dict[str, Any],
        chunk_id: int,
        provider: str,
        model: str,
    ) -> Embedding | None:
        """Executor method for get_embedding_by_chunk_id - runs in DB thread.

        Args:
            conn: Database connection
            state: Thread-local state dictionary
            chunk_id: ID of the chunk to find embedding for
            provider: Embedding provider name
            model: Model name used for embedding

        Returns:
            Embedding model if found, None otherwise
        """
        # Get all embedding tables
        embedding_tables = self._executor_get_all_embedding_tables(conn, state)

        for table_name in embedding_tables:
            result = conn.execute(
                f"""
                SELECT chunk_id, provider, model, embedding, dims, created_at
                FROM {table_name}
                WHERE chunk_id = ? AND provider = ? AND model = ?
                LIMIT 1
                """,
                [chunk_id, provider, model],
            ).fetchone()

            if result:
                return Embedding(
                    chunk_id=result[0],
                    provider=result[1],
                    model=result[2],
                    vector=result[3],
                    dims=result[4],
                    created_at=result[5],
                )

        return None

    def delete_embeddings_by_chunk_id(self, chunk_id: int) -> None:
        """Delete all embeddings for a specific chunk with ShardManager coordination."""
        self._execute_in_db_thread_sync("delete_embeddings_by_chunk_id", chunk_id)

    def _executor_delete_embeddings_by_chunk_id(
        self, conn: Any, state: dict[str, Any], chunk_id: int
    ) -> None:
        """Executor method for delete_embeddings_by_chunk_id - runs in DB thread.

        Coordinates with ShardManager to track shard state changes and trigger
        fix_pass if any affected shard falls below merge threshold.
        """
        try:
            # Get all embedding tables
            embedding_tables = self._executor_get_all_embedding_tables(conn, state)

            # Track affected shards BEFORE deletion
            affected_shards: dict[tuple[str, int], int] = {}
            for table_name in embedding_tables:
                dims = int(table_name.split("_")[1])
                result = conn.execute(
                    f"SELECT shard_id FROM {table_name} WHERE chunk_id = ?",
                    [chunk_id],
                ).fetchall()
                for row in result:
                    if row[0] is not None:
                        key = (str(row[0]), dims)
                        affected_shards[key] = affected_shards.get(key, 0) + 1

            # Delete from all embedding tables
            for table_name in embedding_tables:
                conn.execute(
                    f"DELETE FROM {table_name} WHERE chunk_id = ?", [chunk_id]
                )

            # Coordinate with ShardManager if available
            if self.shard_manager is not None and affected_shards:
                needs_fix = False
                for (shard_id_str, dims), _deleted_count in affected_shards.items():
                    table_name = f"embeddings_{dims}"
                    result = conn.execute(
                        f"SELECT COUNT(*) FROM {table_name} WHERE shard_id = ?",
                        [shard_id_str],
                    ).fetchone()
                    remaining = result[0] if result else 0
                    if (
                        remaining < self.shard_manager.config.merge_threshold
                        or remaining == 0
                    ):
                        needs_fix = True
                        logger.debug(
                            f"Shard {shard_id_str[:8]}... has {remaining} embeddings "
                            f"after deletion (threshold: "
                            f"{self.shard_manager.config.merge_threshold})"
                        )

                if needs_fix:
                    logger.debug(
                        f"Deletion of chunk {chunk_id} triggered shard fix_pass"
                    )
                    self.shard_manager.fix_pass(conn, check_quality=False)

        except Exception as e:
            logger.error(f"Failed to delete embeddings for chunk {chunk_id}: {e}")
            raise

    def run_fix_pass(self, check_quality: bool = True) -> None:
        """Run ShardManager fix_pass via executor thread.

        This is the public API for triggering fix_pass from outside the executor,
        such as from BulkIndexer for deferred quality checks.

        Args:
            check_quality: If True, measure self-recall for each shard
        """
        if self.shard_manager is None:
            return
        self._execute_in_db_thread_sync("run_fix_pass", check_quality)

    def _executor_run_fix_pass(
        self, conn: Any, state: dict[str, Any], check_quality: bool
    ) -> None:
        """Executor method for running ShardManager fix_pass - runs in DB thread."""
        if self.shard_manager:
            self.shard_manager.fix_pass(conn, check_quality=check_quality)

    def get_all_chunks_with_metadata(self) -> list[dict[str, Any]]:
        """Get all chunks with their metadata including file paths - delegate to chunk repository."""
        return self._execute_in_db_thread_sync("get_all_chunks_with_metadata")

    def get_scope_stats(self, scope_prefix: str | None) -> tuple[int, int]:
        """Return (total_files, total_chunks) under an optional scope prefix.

        This is used by code_mapper coverage and must avoid loading full chunk code.
        """
        return self._execute_in_db_thread_sync("get_scope_stats", scope_prefix)

    def _executor_get_scope_stats(
        self, conn: Any, state: dict[str, Any], scope_prefix: str | None
    ) -> tuple[int, int]:
        """Executor method for get_scope_stats - runs in DB thread."""
        try:
            if scope_prefix:
                normalized = scope_prefix.replace("\\", "/")
                escaped = escape_like_pattern(normalized)
                like = f"{escaped}%"
                files_row = conn.execute(
                    "SELECT COUNT(*) FROM files WHERE path LIKE ? ESCAPE '\\'",
                    [like],
                ).fetchone()
                chunks_row = conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM chunks c
                    JOIN files f ON c.file_id = f.id
                    WHERE f.path LIKE ? ESCAPE '\\'
                    """,
                    [like],
                ).fetchone()
            else:
                files_row = conn.execute("SELECT COUNT(*) FROM files").fetchone()
                chunks_row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()

            total_files = int(files_row[0]) if files_row else 0
            total_chunks = int(chunks_row[0]) if chunks_row else 0
            return total_files, total_chunks
        except Exception as exc:
            logger.debug(f"Failed to get scope stats: {exc}")
            return 0, 0

    def get_scope_file_paths(self, scope_prefix: str | None) -> list[str]:
        """Return file paths under an optional scope prefix."""
        return self._execute_in_db_thread_sync("get_scope_file_paths", scope_prefix)

    def _executor_get_scope_file_paths(
        self, conn: Any, state: dict[str, Any], scope_prefix: str | None
    ) -> list[str]:
        """Executor method for get_scope_file_paths - runs in DB thread."""
        try:
            if scope_prefix:
                normalized = scope_prefix.replace("\\", "/")
                escaped = escape_like_pattern(normalized)
                like = f"{escaped}%"
                rows = conn.execute(
                    "SELECT path FROM files WHERE path LIKE ? ESCAPE '\\' ORDER BY path",
                    [like],
                ).fetchall()
            else:
                rows = conn.execute("SELECT path FROM files ORDER BY path").fetchall()

            out: list[str] = []
            for row in rows:
                try:
                    path = str(row[0] or "").replace("\\", "/")
                except Exception:
                    path = ""
                if path:
                    out.append(path)
            return out
        except Exception as exc:
            logger.debug(f"Failed to get scope file paths: {exc}")
            return []

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
        """Executor method for search_semantic - runs in DB thread.

        Uses ShardManager for USearch-based search. Returns empty results
        if ShardManager is not available or has no shards.
        """
        empty_result: tuple[list[dict[str, Any]], dict[str, Any]] = (
            [],
            {"offset": offset, "page_size": page_size, "has_more": False, "total": 0},
        )

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
                return empty_result

            # ShardManager is required for vector search
            if self.shard_manager is None:
                logger.warning("ShardManager not initialized - vector search unavailable")
                return empty_result

            # Perform ShardManager search
            # Note: shard_manager.search() handles empty/missing centroids gracefully
            # by still searching shards that exist on disk
            try:
                # Request more results to account for filtering
                shard_results = self.shard_manager.search(
                    query=query_embedding,
                    k=page_size + offset + 100,  # Fetch extra for filtering
                    dims=query_dims,
                    provider=provider,
                    model=model,
                    conn=conn,
                )
                # Extract embedding IDs (keys) from search results
                embedding_ids = [r.key for r in shard_results]
            except Exception as e:
                logger.error(f"ShardManager search failed: {e}")
                return empty_result

            if not embedding_ids:
                return empty_result

            # Query DuckDB for chunk metadata using embedding IDs
            placeholders = ",".join(["?"] * len(embedding_ids))
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
                    array_cosine_similarity(e.embedding, ?::FLOAT[{query_dims}]) as similarity,
                    e.id as embedding_id
                FROM {table_name} e
                JOIN chunks c ON e.chunk_id = c.id
                JOIN files f ON c.file_id = f.id
                WHERE e.id IN ({placeholders})
                AND e.provider = ? AND e.model = ?
            """
            params: list[Any] = [query_embedding] + embedding_ids + [provider, model]

            path_like: str | None = None
            if normalized_path is not None:
                escaped_path = escape_like_pattern(normalized_path)
                path_like = f"%{escaped_path}%"

            if threshold is not None:
                query += f" AND array_cosine_similarity(e.embedding, ?::FLOAT[{query_dims}]) >= ?"
                params.append(query_embedding)
                params.append(threshold)

            if path_like is not None:
                query += " AND f.path LIKE ? ESCAPE '\\'"
                params.append(path_like)

            query += " ORDER BY similarity DESC"
            results = conn.execute(query, params).fetchall()

            # Apply pagination
            total_count = len(results)
            results = results[offset : offset + page_size]

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
                escaped_path = escape_like_pattern(normalized_path)
                where_conditions.append("f.path LIKE ? ESCAPE '\\'")
                # Allow matching repo-relative segments inside stored paths
                params.append(f"%{escaped_path}%")

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
        path_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find chunks similar to the given chunk using its embedding."""
        return self._execute_in_db_thread_sync(
            "find_similar_chunks",
            chunk_id,
            provider,
            model,
            limit,
            threshold,
            path_filter,
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
        path_filter: str | None,
    ) -> list[dict[str, Any]]:
        """Executor method for find_similar_chunks - runs in DB thread."""
        try:
            # Validate and normalize path filter for consistent scoping behavior
            normalized_path = self._validate_and_normalize_path_filter(path_filter)

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

            # Optional path scoping condition
            path_condition = ""
            params: list[Any] = [target_embedding, provider, model, chunk_id]
            if normalized_path is not None:
                escaped_path = escape_like_pattern(normalized_path)
                path_condition = "AND f.path LIKE ? ESCAPE '\\'"
                # Substring match so repo-relative scopes still work when base_directory is higher
                params.append(f"%{escaped_path}%")

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
                {path_condition}
                {threshold_condition}
                ORDER BY distance ASC
                LIMIT ?
            """

            params.append(limit)

            results = conn.execute(query, params).fetchall()

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
        """Executor method for search_by_embedding - runs in DB thread.

        Uses ShardManager for USearch-based search. Returns empty results
        if ShardManager is not available or has no shards.
        """
        try:
            # Detect dimensions from query embedding
            query_dims = len(query_embedding)
            table_name = f"embeddings_{query_dims}"

            # Check if table exists for these dimensions
            if not self._executor_table_exists(conn, state, table_name):
                logger.warning(
                    f"No embeddings table found for {query_dims} dimensions ({table_name})"
                )
                return []

            # ShardManager is required for vector search
            if self.shard_manager is None:
                logger.warning("ShardManager not initialized - vector search unavailable")
                return []

            if not self.shard_manager.centroids:
                logger.warning("No shard centroids available - vector search will return empty results")
                return []

            # Perform ShardManager search
            try:
                # Request more results to account for filtering (uses configured overfetch)
                shard_results = self.shard_manager.search(
                    query=query_embedding,
                    k=limit * self._sharding_config.overfetch_multiplier,
                    dims=query_dims,
                    provider=provider,
                    model=model,
                    conn=conn,
                )
                # Extract embedding IDs (keys) from search results
                embedding_ids = [r.key for r in shard_results]
            except Exception as e:
                logger.error(f"ShardManager search failed: {e}")
                return []

            if not embedding_ids:
                return []

            # Query DuckDB for chunk metadata using embedding IDs
            normalized_path = self._validate_and_normalize_path_filter(path_filter)
            placeholders = ",".join(["?"] * len(embedding_ids))
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
                    array_cosine_similarity(e.embedding, ?::FLOAT[{query_dims}]) as similarity
                FROM {table_name} e
                JOIN chunks c ON e.chunk_id = c.id
                JOIN files f ON c.file_id = f.id
                WHERE e.id IN ({placeholders})
                AND e.provider = ? AND e.model = ?
            """
            params: list[Any] = [query_embedding] + embedding_ids + [provider, model]

            if threshold is not None:
                query += f" AND array_cosine_similarity(e.embedding, ?::FLOAT[{query_dims}]) >= ?"
                params.append(query_embedding)
                params.append(threshold)

            if normalized_path is not None:
                escaped_path = escape_like_pattern(normalized_path)
                path_like = f"%{escaped_path}%"
                query += " AND f.path LIKE ? ESCAPE '\\'"
                params.append(path_like)

            query += " ORDER BY similarity DESC"
            results = conn.execute(query, params).fetchall()

            # Apply limit
            results = results[:limit]

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
                    "score": result[8],  # Similarity score (already 0-1)
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

            # Run checkpoint if forced
            if force_checkpoint:
                try:
                    conn.execute("CHECKPOINT")
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

    def optimize(self) -> bool:
        """Optimize DuckDB storage: CHECKPOINT, HNSW compact, and full compaction.

        Skips optimization only if database is in a PERFECT state (free_blocks == 0).
        We only check free_blocks because orphaned_blocks/fragmentation_ratio incorrectly
        count HNSW index storage + metadata blocks as "orphaned" when they're legitimate.

        If in doubt (stats retrieval fails), errs on the side of optimization.

        Returns:
            True if full compaction was performed (or skipped due to perfect state),
            False if it failed.
        """
        # Check if DB is in perfect state - only skip if no freed blocks pending reclamation
        # Note: We only check free_blocks because orphaned_blocks/fragmentation_ratio are
        # unreliable - they count HNSW index blocks + metadata as "orphaned" when they're not
        try:
            stats = self.get_storage_stats()
            free = stats.get("free_blocks", 1)  # Default to 1 = needs optimization

            if free == 0:
                logger.info(
                    "Database in perfect state (no free blocks pending reclamation) "
                    "- skipping optimization"
                )
                return True  # Success - nothing to do
        except Exception as e:
            # If in doubt, optimize
            logger.debug(f"Could not check storage stats, proceeding with optimization: {e}")

        # Step 1: Always run CHECKPOINT + HNSW compact (lightweight)
        self.optimize_tables()

        # Step 2: Run full compaction
        logger.info("Running full database compaction...")
        return self._run_blocking_compaction()

    def _run_blocking_compaction(self) -> bool:
        """Run full EXPORT/IMPORT/SWAP compaction cycle.

        Returns:
            True if compaction succeeded, False otherwise.
        """
        db_path = Path(self._connection_manager._db_path)

        # Check disk space (need ~2.5x current DB size)
        if not self._has_sufficient_disk_space(db_path, multiplier=2.5):
            return False

        export_dir = db_path.parent / ".chunkhound_compaction_export"
        new_db_path = db_path.with_suffix(".compact.duckdb")
        old_db_path = db_path.with_suffix(".duckdb.old")
        wal_file = db_path.with_suffix(db_path.suffix + ".wal")
        lock_file = get_compaction_lock_path(db_path)

        try:
            # Create lock file to signal compaction in progress
            lock_file.touch()

            # Soft disconnect BEFORE export to release file locks
            # DuckDB doesn't allow mixing read-only and read-write connections
            # Use soft_disconnect to keep executor alive for reconnection after swap
            self.soft_disconnect(skip_checkpoint=False)

            # 1. Export from original database (now safe - no active connections)
            logger.info("Exporting database for compaction...")
            self._export_database_for_compaction(db_path, export_dir)

            # 2. Import into fresh database
            logger.info("Importing into compacted database...")
            self._import_database_for_compaction(export_dir, new_db_path)

            # 3. Atomic swap
            logger.info("Performing atomic swap...")

            # Explicitly delete orphaned WAL before swap
            if wal_file.exists():
                wal_file.unlink()
                logger.debug(f"Removed pre-swap WAL: {wal_file}")

            # Clean up any previous old file
            if old_db_path.exists():
                old_db_path.unlink()

            # Atomic swap via renames
            db_path.rename(old_db_path)
            new_db_path.rename(db_path)

            # Reconnect to swapped database
            self.connect()

            # Clean up old file
            old_db_path.unlink()

            logger.info(f"Compaction complete: {db_path}")
            return True

        except Exception as e:
            logger.error(f"Compaction failed: {e}")
            # Attempt recovery: restore original if possible
            if old_db_path.exists() and not db_path.exists():
                logger.warning("Restoring original database from backup...")
                old_db_path.rename(db_path)
            # Reconnect (we disconnected early, need to restore connection)
            if not self.is_connected:
                try:
                    self.connect()
                except Exception as err:
                    logger.error(f"Reconnect after compaction failed: {err}")
            # Clean up temp database on failure
            if new_db_path.exists():
                try:
                    new_db_path.unlink()
                except OSError:
                    pass
            return False
        finally:
            # Always clean up export directory and lock file
            if export_dir.exists():
                shutil.rmtree(export_dir, ignore_errors=True)
            lock_file.unlink(missing_ok=True)

    def _has_sufficient_disk_space(self, db_path: Path, multiplier: float = 2.5) -> bool:
        """Check if sufficient disk space exists for compaction."""
        try:
            db_size = db_path.stat().st_size
            required = int(db_size * multiplier)
            available = shutil.disk_usage(db_path.parent).free

            if available < required:
                logger.warning(
                    f"Insufficient disk space for compaction: "
                    f"need {required / 1024 / 1024:.1f}MB, "
                    f"have {available / 1024 / 1024:.1f}MB"
                )
                return False
            return True
        except OSError as e:
            logger.warning(f"Could not check disk space: {e}")
            return True  # Proceed anyway, let OS handle errors

    def _export_database_for_compaction(self, db_path: Path, export_dir: Path) -> None:
        """Export database to Parquet files for compaction."""
        import duckdb

        # Clean up any leftover export directory
        if export_dir.exists():
            shutil.rmtree(export_dir)

        # Export with read-only connection
        conn = duckdb.connect(str(db_path), read_only=True)
        try:
            conn.execute(f"EXPORT DATABASE '{export_dir}' (FORMAT PARQUET)")
        finally:
            conn.close()

    def _import_database_for_compaction(
        self, export_dir: Path, new_db_path: Path
    ) -> None:
        """Import Parquet files into fresh database."""
        import duckdb

        if new_db_path.exists():
            new_db_path.unlink()

        conn = duckdb.connect(str(new_db_path))
        try:
            conn.execute(f"IMPORT DATABASE '{export_dir}'")
            # CRITICAL: Checkpoint to persist imported data before closing
            # Without this, the imported data may not be written to disk
            conn.execute("CHECKPOINT")
        finally:
            conn.close()

