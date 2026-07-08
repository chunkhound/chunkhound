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

import json
import os
import re
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import duckdb
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
from chunkhound.utils.logging_guard import log_if_not_mcp
from chunkhound.providers.database.like_utils import escape_like_pattern
from chunkhound.providers.database.serial_database_provider import (
    SerialDatabaseProvider,
)
from chunkhound.providers.database.serial_executor import (
    DatabaseCompactionInProgressError,
    _executor_local,
)
from chunkhound.utils.windows_constants import (
    IS_WINDOWS,
    WINDOWS_FILE_HANDLE_DELAY,
    _unlink_compacted,
)


def _atomic_replace(src: Path | str, dst: Path | str) -> None:
    """Atomically replace dst with src, with retry on Windows.

    On POSIX ``os.replace`` is atomic (rename(2)). On Windows we retry
    briefly because antivirus / delayed handle release can make a valid
    same-volume replace fail transiently.
    """
    if IS_WINDOWS:
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                os.replace(str(src), str(dst))
                return
            except OSError:
                if attempt == max_attempts - 1:
                    raise
                time.sleep(WINDOWS_FILE_HANDLE_DELAY * (2**attempt))
    else:
        os.replace(str(src), str(dst))


# Type hinting only
if TYPE_CHECKING:
    from chunkhound.core.config.database_config import DatabaseConfig

from chunkhound.providers.database.duckdb.schema_constants import (
    _CHUNKS_COLUMN_NAMES,
    _CHUNKS_TABLE_COLUMNS,
    _FILES_COLUMN_NAMES,
    _FILES_TABLE_COLUMNS,
    _SCHEMA_VERSION_TABLE_COLUMNS,
    _create_embedding_table_sql,
    _embedding_chunk_id_index_name,
    _embedding_dims_from_table_name,
    _embedding_hnsw_index_name,
    _embedding_indexes_where_clause,
    _embedding_provider_model_index_name,
    _embedding_table_columns,
    _embedding_table_name,
    _embedding_tables_where_clause,
    _embedding_unique_index_name,
    _embeddings_column_names,
    CANONICAL_TABLE_NAMES,
    VALID_CATALOGS,
    is_hnsw_index,
)


class DuckDBTransactionConflictError(RuntimeError):
    """A guarded mutation collided with an already-open DuckDB transaction."""


class DuckDBIndexedRootMismatchError(RuntimeError):
    """A DuckDB database was reopened under a different indexed root than the one it was claimed for."""


_INDEXED_ROOT_SIDECAR_SUFFIX = ".root.json"
_INDEXED_ROOT_SIDECAR_VERSION = 1


def _normalize_indexed_root(root: Path | str) -> str:
    """Normalize a requested indexed root to the authoritative logical form.

    Uses `expanduser()` + `absolute()` + `as_posix()`. Intentionally does not
    call `resolve()` — the logical requested-root contract is authoritative and
    physical/resolved paths are routing details only.
    """
    return Path(root).expanduser().absolute().as_posix()


def _indexed_root_sidecar_path(db_path: Path | str) -> Path | None:
    """Return the sidecar path for a DuckDB file, or None for :memory: databases."""
    if str(db_path) == ":memory:":
        return None
    db_file = Path(db_path)
    return db_file.with_name(db_file.name + _INDEXED_ROOT_SIDECAR_SUFFIX)


def _read_indexed_root_sidecar(sidecar: Path) -> str:
    """Read and validate the sidecar file, returning the stored logical root.

    Raises plain `RuntimeError` on malformed/unreadable/wrong-version/missing-key
    content. Callers that want a typed mismatch error handle that separately.
    """
    try:
        raw = sidecar.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(
            f"Indexed-root sidecar {sidecar} could not be read: {error}"
        ) from error
    if not raw.strip():
        raise RuntimeError(f"Indexed-root sidecar {sidecar} is empty")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Indexed-root sidecar {sidecar} is not valid JSON: {error}"
        ) from error
    if not isinstance(data, dict):
        raise RuntimeError(f"Indexed-root sidecar {sidecar} must be a JSON object")
    version = data.get("version")
    if version != _INDEXED_ROOT_SIDECAR_VERSION:
        raise RuntimeError(
            f"Indexed-root sidecar {sidecar} has unsupported version {version!r}"
        )
    stored = data.get("indexed_root_path")
    if not isinstance(stored, str) or not stored:
        raise RuntimeError(
            f"Indexed-root sidecar {sidecar} is missing indexed_root_path"
        )
    return stored


def _write_indexed_root_sidecar(sidecar: Path, logical_root: str) -> bool:
    """Exclusively claim the sidecar path, returning True only for the winner.

    The final sidecar path is only published after the payload is fully written
    and fsynced to a unique sibling temp file. That avoids exposing empty or
    partial JSON to concurrent readers while still preserving first-writer wins.
    """
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {
            "version": _INDEXED_ROOT_SIDECAR_VERSION,
            "indexed_root_path": logical_root,
        }
    )
    tmp = sidecar.with_name(f"{sidecar.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with open(tmp, "x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(tmp, sidecar)
        return True
    except FileExistsError:
        return False
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def _read_indexed_root_sidecar_after_claim_collision(sidecar: Path) -> str:
    """Read the winner sidecar after a first-writer collision.

    Some filesystems can surface the no-clobber publish collision to the loser
    just before the published path is synchronously readable from that loser
    thread/process. Retry only the transient not-found case; any other malformed
    or unreadable sidecar remains fail-closed.
    """
    for _ in range(20):
        try:
            return _read_indexed_root_sidecar(sidecar)
        except RuntimeError as error:
            if isinstance(error.__cause__, FileNotFoundError):
                time.sleep(0.01)
                continue
            raise
    return _read_indexed_root_sidecar(sidecar)


class DuckDBProvider(SerialDatabaseProvider):
    """DuckDB implementation of DatabaseProvider protocol.

    # CLASS_CONTEXT: Analytical database optimized for bulk operations
    # CONSTRAINT: Inherits from SerialDatabaseProvider for thread safety
    # PERFORMANCE: Uses column-store format, vectorized execution
    """

    _SUPPORTED_HNSW_METRICS = frozenset({"cosine", "ip", "l2sq"})

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
        # Create a NEW connection for the executor thread
        # This ensures thread safety - only this thread will use this connection
        conn = duckdb.connect(
            str(self._connection_manager.db_path),
            read_only=self._connection_manager.is_read_only,
        )

        # Load required extensions
        conn.execute("INSTALL vss")
        conn.execute("LOAD vss")
        conn.execute("SET hnsw_enable_experimental_persistence = true")

        logger.debug(
            f"Created new DuckDB connection in executor thread {threading.get_ident()}"
        )
        return conn

    def _create_connection_manager_connection(self) -> Any:
        """Create the public health/compatibility connection."""
        conn = duckdb.connect(str(self._connection_manager.db_path))
        conn.execute("INSTALL vss")
        conn.execute("LOAD vss")
        conn.execute("SET hnsw_enable_experimental_persistence = true")
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
            # Fast-fail if compaction owns the database — connection_manager's
            # recovery path would otherwise undo the in-progress compaction.
            # Must be checked BEFORE _connection_manager.connect() which also
            # runs recovery.
            if self._executor.is_compaction_in_progress():
                logger.info(
                    "Database compaction in progress — refusing connection "
                    "until compaction completes"
                )
                raise DatabaseCompactionInProgressError(
                    "Database compaction in progress — connection refused until "
                    "compaction completes"
                )

            # Validate stored indexed-root identity before any file-backed DB
            # open, WAL replay, extension load, or schema touch happens.
            self.ensure_indexed_root_identity(
                requested_root=self._base_directory,
                allow_claim_if_missing=False,
            )

            # Initialize connection manager FIRST - this handles WAL validation
            self._connection_manager.connect()

            # Call parent connect which handles executor initialization
            super().connect()

        except DatabaseCompactionInProgressError:
            raise
        except Exception as e:
            logger.error(f"DuckDB connection failed: {e}")
            raise

    def ensure_indexed_root_identity(
        self,
        *,
        requested_root: Path | str,
        allow_claim_if_missing: bool,
    ) -> None:
        """Validate (or claim) the authoritative indexed-root identity for this DB.

        The authoritative indexed root is stored in a DB-adjacent sidecar
        (``<dbfile>.root.json``). ``:memory:`` databases have no filesystem
        artifact and are treated as a no-op.

        Behavior:
        - sidecar exists and matches current root: return.
        - sidecar exists and mismatches: raise
          ``DuckDBIndexedRootMismatchError`` without touching the DB.
        - sidecar exists but is malformed / unreadable / wrong version:
          raise plain ``RuntimeError`` (fail-closed; never silently rewritten).
        - sidecar missing and ``allow_claim_if_missing`` is ``False``:
          return (legacy migration is deferred to the first mutation-capable
          call site).
        - sidecar missing and ``allow_claim_if_missing`` is ``True``:
          atomically claim the current root and log a one-time warning.
        """
        sidecar = _indexed_root_sidecar_path(self._connection_manager.db_path)
        if sidecar is None:
            return

        current_root = _normalize_indexed_root(requested_root)

        if sidecar.exists():
            stored_root = _read_indexed_root_sidecar(sidecar)
            if stored_root == current_root:
                return
            raise DuckDBIndexedRootMismatchError(
                f"Refusing to open DuckDB database {self._connection_manager.db_path} "
                f"under indexed root {current_root!r}: sidecar {sidecar} records "
                f"previously claimed root {stored_root!r}. This database must only "
                f"be reopened under its original indexed root."
            )

        if not allow_claim_if_missing:
            return

        if _write_indexed_root_sidecar(sidecar, current_root):
            logger.warning(
                "DuckDB indexed-root sidecar was missing for %s; treated as legacy "
                "DB migration and claimed current root %s. Future opens under a "
                "different root will fail.",
                self._connection_manager.db_path,
                current_root,
            )
            return

        stored_root = _read_indexed_root_sidecar_after_claim_collision(sidecar)
        if stored_root == current_root:
            return
        raise DuckDBIndexedRootMismatchError(
            f"Refusing to open DuckDB database {self._connection_manager.db_path} "
            f"under indexed root {current_root!r}: sidecar {sidecar} records "
            f"previously claimed root {stored_root!r}. This database must only "
            f"be reopened under its original indexed root."
        )

    def _executor_connect(self, conn: Any, state: dict[str, Any]) -> None:
        """Executor method for connect - runs in DB thread.

        Note: The connection is already created by _get_thread_local_connection,
        so this method just ensures schema and indexes are created.
        """
        # Read-only opens reject all DDL — schema must already exist on disk.
        if self._connection_manager.is_read_only:
            if self._executor_table_exists(conn, state, "embeddings"):
                logger.warning(
                    "Legacy 'embeddings' table present; read-only mode cannot "
                    "migrate it. Reopen without --read-only to migrate, or "
                    "queries against embeddings may behave unexpectedly."
                )
            return

        try:
            # Perform WAL cleanup once with synchronization.
            # _perform_wal_cleanup_in_executor may close `conn` and create a
            # new one (stored in _executor_local.connection) — always re-read
            # from thread-local so subsequent operations use the live connection.
            with self._wal_cleanup_lock:
                if not self._wal_cleanup_done:
                    self._perform_wal_cleanup_in_executor(conn)
                    self._wal_cleanup_done = True
                    # Re-read in case WAL cleanup replaced the connection
                    conn = _executor_local.connection

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
        if self._connection_manager.is_memory_db:
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
            _executor_local.connection = self._create_connection()

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
            if (
                not skip_checkpoint
                and not self._connection_manager.is_memory_db
                and not self._connection_manager.is_read_only
            ):
                # Force checkpoint before close to ensure durability
                conn.execute("CHECKPOINT")
                log_if_not_mcp(
                    "debug", "Database checkpoint completed before disconnect"
                )
            else:
                if self._connection_manager.is_read_only:
                    reason = "read-only"
                elif self._connection_manager.is_memory_db:
                    reason = "memory db"
                else:
                    reason = "already done"
                log_if_not_mcp(
                    "debug",
                    f"Skipping checkpoint before disconnect ({reason})",
                )
        except Exception as e:
            log_if_not_mcp("error", f"Checkpoint failed during disconnect: {e}")
        finally:
            # Close connection
            conn.close()
            # Clear thread-local connection
            if hasattr(_executor_local, "connection"):
                delattr(_executor_local, "connection")
            log_if_not_mcp("info", "DuckDB connection closed in executor thread")

    def health_check(self) -> dict[str, Any]:
        """Perform health check and return status information - delegate to connection manager."""
        return self._connection_manager.health_check()

    def get_connection_info(self) -> dict[str, Any]:
        """Get information about the database connection - delegate to connection manager."""
        return self._connection_manager.get_connection_info()

    def _table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database - delegate to connection manager."""
        return self._execute_in_db_thread_sync("table_exists", table_name)

    def _executor_table_has_primary_key(self, conn: Any, table_name: str) -> bool:
        """Check if the given table has a PRIMARY KEY constraint.

        Compaction historically used CREATE TABLE AS SELECT which strips all
        constraints, so we verify PK presence independently of CREATE TABLE
        IF NOT EXISTS.  This is now a one-time migration path — after the DDL
        refactor compaction preserves constraints.
        """
        result = conn.execute(
            """
            SELECT COUNT(*) FROM information_schema.table_constraints
            WHERE table_name = ? AND constraint_type = 'PRIMARY KEY'
            """,
            [table_name],
        ).fetchone()
        return result is not None and result[0] > 0

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
        return _embedding_table_name(dims)

    def _get_embedding_provider_model_index_name(self, dims: int) -> str:
        """Return the standard provider/model lookup index name."""
        return _embedding_provider_model_index_name(dims)

    def _get_embedding_unique_index_name(self, dims: int) -> str:
        """Return the schema-backed unique index name for embedding upserts."""
        return _embedding_unique_index_name(dims)

    def _executor_get_column_default(
        self, conn: Any, table_name: str, column_name: str
    ) -> str | None:
        """Return one column default expression from information_schema."""
        result = conn.execute(
            """
            SELECT column_default
            FROM information_schema.columns
            WHERE table_name = ? AND column_name = ?
            """,
            [table_name, column_name],
        ).fetchone()
        if result is None:
            return None
        default = result[0]
        return str(default) if default is not None else None

    def _executor_table_has_foreign_key(
        self,
        conn: Any,
        table_name: str,
        column_name: str,
        referenced_table: str,
        referenced_column: str,
    ) -> bool:
        """Return True when one specific foreign-key contract exists."""
        result = conn.execute(
            """
            SELECT 1
            FROM duckdb_constraints()
            WHERE table_name = ?
              AND constraint_type = 'FOREIGN KEY'
              AND constraint_column_names = [?]
              AND referenced_table = ?
              AND referenced_column_names = [?]
            LIMIT 1
            """,
            [table_name, column_name, referenced_table, referenced_column],
        ).fetchone()
        return result is not None

    def _executor_reseed_sequence(
        self, conn: Any, table_name: str, sequence_name: str
    ) -> None:
        """Advance one sequence so the next implicit id is above MAX(id).

        DuckDB 1.4 does not support ``ALTER SEQUENCE ... RESTART WITH`` or
        ``setval()``.  ``DROP SEQUENCE CASCADE`` would silently destroy
        dependent tables, so we keep ``SELECT nextval FROM range(N)`` for live
        databases.  The compaction path (``_compact_copy_data``) avoids this
        entirely by creating fresh sequences with the correct ``START`` value.
        """
        max_result = conn.execute(
            f"SELECT COALESCE(MAX(id), 0) FROM {self._quote_duckdb_identifier(table_name)}"
        ).fetchone()
        max_id = int(max_result[0]) if max_result is not None else 0
        if max_id <= 0:
            return

        current_value_result = conn.execute(
            """
            SELECT last_value
            FROM duckdb_sequences()
            WHERE sequence_name = ?
            """,
            [sequence_name],
        ).fetchone()
        current_value = current_value_result[0] if current_value_result else None
        if current_value is None:
            current_value = conn.execute(
                f"SELECT nextval('{sequence_name}')"
            ).fetchone()[0]

        advance_by = max_id - int(current_value)
        if advance_by <= 0:
            return

        conn.execute(f"SELECT nextval('{sequence_name}') FROM range({advance_by})")

    def _executor_recreate_sequence(
        self, conn: Any, sequence_name: str, next_value: int
    ) -> None:
        """Recreate one sequence with an explicit next value."""
        conn.execute(
            f"DROP SEQUENCE IF EXISTS {self._quote_duckdb_identifier(sequence_name)}"
        )
        conn.execute(
            f"CREATE SEQUENCE {self._quote_duckdb_identifier(sequence_name)} START {next_value}"
        )

    def _executor_get_max_embedding_id(self, conn: Any) -> int:
        """Return the global max embedding id across all embeddings_* tables."""
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            f"WHERE {_embedding_tables_where_clause()}"
        ).fetchall()
        max_id = 0
        for (table_name,) in rows:
            table_max = conn.execute(
                f"SELECT COALESCE(MAX(id), 0) FROM {self._quote_duckdb_identifier(table_name)}"
            ).fetchone()[0]
            max_id = max(max_id, int(table_max))
        return max_id

    def _executor_rebuild_table_from_canonical_schema(
        self,
        conn: Any,
        table_name: str,
        columns_sql: str,
        copy_columns: list[str],
    ) -> None:
        """Rebuild one table from canonical DDL and copy rows back explicitly."""
        backup_table = f"{table_name}_schema_rebuild_backup"
        quoted_backup = self._quote_duckdb_identifier(backup_table)
        quoted_table = self._quote_duckdb_identifier(table_name)
        column_list = ", ".join(copy_columns)
        conn.execute(f"DROP TABLE IF EXISTS {quoted_backup}")
        conn.execute(
            f"CREATE TEMP TABLE {quoted_backup} AS "
            f"SELECT {column_list} FROM {quoted_table}"
        )
        conn.execute(f"DROP TABLE {quoted_table}")
        conn.execute(f"CREATE TABLE {quoted_table} ({columns_sql})")
        conn.execute(
            f"INSERT INTO {quoted_table} ({column_list}) "
            f"SELECT {column_list} FROM {quoted_backup}"
        )
        conn.execute(f"DROP TABLE {quoted_backup}")

    def _executor_index_exists(
        self, conn: Any, table_name: str, index_name: str
    ) -> bool:
        """Return True when duckdb_indexes() reports the named index."""
        result = conn.execute(
            "SELECT 1 "
            "FROM duckdb_indexes() "
            f"WHERE {_embedding_indexes_where_clause()} "
            "AND table_name = ? AND index_name = ? "
            "LIMIT 1",
            [table_name, index_name],
        ).fetchone()
        return result is not None

    def _executor_create_embedding_provider_model_index(
        self, conn: Any, table_name: str, dims: int
    ) -> None:
        """Ensure the provider/model lookup index exists for one embedding table."""
        index_name = self._get_embedding_provider_model_index_name(dims)
        if self._executor_index_exists(conn, table_name, index_name):
            return

        conn.execute(f"CREATE INDEX {index_name} ON {table_name}(provider, model)")

    def _executor_create_embedding_unique_index(
        self, conn: Any, table_name: str, dims: int
    ) -> None:
        """Create the real upsert conflict target for one embedding table."""
        index_name = self._get_embedding_unique_index_name(dims)
        if self._executor_index_exists(conn, table_name, index_name):
            return

        conn.execute(
            f"""
            CREATE UNIQUE INDEX {index_name}
            ON {table_name}(chunk_id, provider, model)
            """
        )

    def _executor_get_embedding_duplicate_row_ids(
        self, conn: Any, table_name: str
    ) -> list[int]:
        """Return duplicate row ids that must be removed before adding uniqueness."""
        rows = conn.execute(
            f"""
            SELECT id
            FROM (
                SELECT
                    id,
                    ROW_NUMBER() OVER (
                        PARTITION BY chunk_id, provider, model
                        ORDER BY created_at DESC NULLS LAST, id DESC
                    ) AS row_num
                FROM {table_name}
            )
            WHERE row_num > 1
            ORDER BY id
            """
        ).fetchall()
        return [int(row[0]) for row in rows]

    def _executor_get_vector_indexes_for_table(
        self, conn: Any, state: dict[str, Any], table_name: str
    ) -> list[dict[str, Any]]:
        """Return only the HNSW indexes for the target embedding table."""
        return [
            index_info
            for index_info in self._executor_get_existing_vector_indexes(conn, state)
            if index_info["table_name"] == table_name
        ]

    def _executor_run_embedding_table_hnsw_guarded_mutation(
        self,
        conn: Any,
        state: dict[str, Any],
        table_name: str,
        mutation_label: str,
        mutation_func: Callable[[], Any],
        *,
        optimize_for_bulk: bool = False,
        transactional: bool = True,
    ) -> Any:
        """Run one embedding-table mutation behind a strict exact-index restore guard."""
        if state.get("transaction_active", False) and transactional:
            raise DuckDBTransactionConflictError(
                f"{mutation_label} cannot run while another DuckDB transaction is active"
            )

        existing_indexes = self._executor_get_vector_indexes_for_table(
            conn, state, table_name
        )

        try:
            if transactional:
                self._executor_begin_transaction(conn, state)
            if optimize_for_bulk:
                conn.execute("SET preserve_insertion_order = false")

            for index_info in existing_indexes:
                self._executor_drop_vector_index_by_name(conn, index_info["index_name"])

            result = mutation_func()

            # Commit data changes before recreating HNSW indexes. DuckDB VSS HNSW indexes
            # do not support CreateDeltaIndex, so committing a transaction that contains an
            # HNSW CREATE triggers a BoundIndex::CreateDeltaIndex assertion failure.
            if transactional:
                self._executor_commit_transaction(conn, state, False)

            for index_info in existing_indexes:
                self._executor_recreate_vector_index_from_info(conn, state, index_info)

            if not state.get("transaction_active", False):
                conn.execute("CHECKPOINT")
            return result
        except Exception as e:
            if transactional and state.get("transaction_active", False):
                try:
                    self._executor_rollback_transaction(conn, state)
                except Exception as rollback_error:
                    raise RuntimeError(
                        f"{mutation_label} failed: {e}; rollback failed: {rollback_error}"
                    ) from rollback_error
            else:
                restore_failures: list[str] = []
                for index_info in existing_indexes:
                    try:
                        self._executor_recreate_vector_index_from_info(
                            conn, state, index_info
                        )
                    except Exception as recreate_error:
                        restore_failures.append(
                            f"{index_info['index_name']}: {recreate_error}"
                        )
                if restore_failures:
                    joined_failures = "; ".join(restore_failures)
                    raise RuntimeError(
                        f"{mutation_label} failed and HNSW restore was incomplete: "
                        f"{joined_failures}"
                    ) from e

            raise

    def _executor_ensure_embedding_upsert_contract(
        self, conn: Any, state: dict[str, Any], table_name: str, dims: int
    ) -> None:
        """Ensure one embedding table has the required unique upsert contract."""
        self._executor_create_embedding_provider_model_index(conn, table_name, dims)

        unique_index_name = self._get_embedding_unique_index_name(dims)
        if self._executor_index_exists(conn, table_name, unique_index_name):
            return

        duplicate_row_ids = self._executor_get_embedding_duplicate_row_ids(
            conn, table_name
        )

        def _apply_contract() -> None:
            if duplicate_row_ids:
                self._executor_delete_embeddings_by_row_ids(
                    conn, table_name, duplicate_row_ids
                )
            self._executor_create_embedding_unique_index(conn, table_name, dims)

        manage_transaction = not state.get("transaction_active", False)
        self._executor_run_embedding_table_hnsw_guarded_mutation(
            conn,
            state,
            table_name,
            f"ensure_embedding_upsert_contract({table_name})",
            _apply_contract,
            transactional=manage_transaction,
        )

    def _executor_create_embedding_table_indexes(
        self, conn: Any, state: dict[str, Any], table_name: str, dims: int,
        *, create_hnsw: bool = True,
    ) -> None:
        """Restore the canonical non-table DDL for one embedding table."""
        if create_hnsw:
            hnsw_index_name = _embedding_hnsw_index_name(dims)
            try:
                conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {hnsw_index_name} ON {table_name}
                    USING HNSW (embedding)
                    WITH (metric = 'cosine')
                """)
            except Exception as e:
                logger.warning(f"Failed to create HNSW index for {table_name}: {e}")

        conn.execute(
            f"CREATE INDEX IF NOT EXISTS {_embedding_chunk_id_index_name(dims)} "
            f"ON {table_name}(chunk_id)"
        )
        self._executor_ensure_embedding_upsert_contract(conn, state, table_name, dims)

    def _executor_embedding_table_needs_rebuild(
        self, conn: Any, table_name: str, dims: int
    ) -> bool:
        """Return True when one embedding table no longer matches canonical DDL."""
        id_default = self._executor_get_column_default(conn, table_name, "id")
        if id_default != "nextval('embeddings_id_seq')":
            return True
        dims_default = self._executor_get_column_default(conn, table_name, "dims")
        return dims_default != str(dims)

    def _executor_rebuild_embedding_tables(
        self, conn: Any, state: dict[str, Any], dims_values: list[int]
    ) -> None:
        """Rebuild all embedding tables together so the shared sequence can be reseeded."""
        copy_columns = _embeddings_column_names(dims_values[0])
        column_list = ", ".join(copy_columns)
        next_value = 1
        backup_tables: list[tuple[int, str, str]] = []
        for dims in dims_values:
            table_name = _embedding_table_name(dims)
            backup_table = f"{table_name}_schema_rebuild_backup"
            quoted_backup = self._quote_duckdb_identifier(backup_table)
            quoted_table = self._quote_duckdb_identifier(table_name)
            conn.execute(f"DROP TABLE IF EXISTS {quoted_backup}")
            conn.execute(
                f"CREATE TEMP TABLE {quoted_backup} AS "
                f"SELECT {column_list} FROM {quoted_table}"
            )
            table_next_value = conn.execute(
                f"SELECT COALESCE(MAX(id), 0) + 1 FROM {quoted_table}"
            ).fetchone()[0]
            next_value = max(next_value, int(table_next_value))
            backup_tables.append((dims, table_name, backup_table))

        for _, table_name, _ in backup_tables:
            conn.execute(f"DROP TABLE {self._quote_duckdb_identifier(table_name)}")

        self._executor_recreate_sequence(conn, "embeddings_id_seq", next_value)

        for dims, table_name, backup_table in backup_tables:
            quoted_backup = self._quote_duckdb_identifier(backup_table)
            quoted_table = self._quote_duckdb_identifier(table_name)
            conn.execute(
                f"CREATE TABLE {quoted_table} ({_embedding_table_columns(dims)})"
            )
            conn.execute(
                f"INSERT INTO {quoted_table} ({column_list}) "
                f"SELECT {column_list} FROM {quoted_backup}"
            )
            conn.execute(f"DROP TABLE {quoted_backup}")
            self._executor_create_embedding_table_indexes(conn, state, table_name, dims)

    def _executor_materialize_embedding_table(
        self, conn: Any, state: dict[str, Any], dims: int
    ) -> str:
        """Create or repair one embedding table from the canonical schema."""
        conn.execute("CREATE SEQUENCE IF NOT EXISTS embeddings_id_seq")
        table_name = _embedding_table_name(dims)
        if not self._executor_table_exists(conn, state, table_name):
            logger.info(f"Creating embedding table for {dims} dimensions: {table_name}")
            conn.execute(_create_embedding_table_sql(dims))
            self._executor_create_embedding_table_indexes(conn, state, table_name, dims)
            return table_name

        self._executor_reseed_sequence(conn, table_name, "embeddings_id_seq")
        # Indexes are already in place from initial table creation
        # (CREATE INDEX IF NOT EXISTS ensures idempotency). Do NOT recreate
        # here -- that would resurrect indexes the caller deliberately dropped
        # (e.g., custom HNSW indexes replacing the canonical one).
        return table_name

    def _ensure_embedding_table_exists(self, dims: int) -> str:
        """Ensure embedding table exists for given dimensions - delegate to connection manager."""
        return self._execute_in_db_thread_sync("ensure_embedding_table_exists", dims)

    def _executor_ensure_embedding_table_exists(
        self, conn: Any, state: dict[str, Any], dims: int
    ) -> str:
        """Executor method for _ensure_embedding_table_exists - runs in DB thread."""
        try:
            return self._executor_materialize_embedding_table(conn, state, dims)
        except Exception as e:
            logger.error(f"Failed to create embedding table for {dims} dimensions: {e}")
            raise

    def _executor_maybe_checkpoint(
        self, conn: Any, state: dict[str, Any], force: bool
    ) -> None:
        """Checkpoint dispatch stub — kept for serial-executor dispatch compatibility.

        DuckDB's native auto-checkpoint (checkpoint_threshold=16MB) handles
        periodic checkpointing. This stub only runs a manual CHECKPOINT when
        explicitly forced and no transaction is active.
        """
        if force and not state.get("transaction_active", False):
            conn.execute("CHECKPOINT")

    def _executor_measure_fragmentation(
        self, conn: Any, state: dict[str, Any]
    ) -> float:
        """Return fragmentation ratio using DuckDB block-level metadata.

        Uses pragma_storage_info (block allocation metadata, not row data)
        to count blocks holding actual table data, then compares
        used_blocks against the estimated minimum.

        By definition, HNSW indexes are smaller than the embeddings tables
        they index (vectors + graph links only, not full rows).  B-tree
        indexes are also smaller than their tables.  So total live index
        storage is bounded by data_blocks (INDEX_FACTOR=1.0).

        System/catalog/sequence overhead is a small fixed number of blocks
        (SYSTEM_BLOCKS=4).

        Returns ratio where:
          < 1.5  = healthy
          > 3.0  = fragmented
          > 5.0  = significantly fragmented
        """
        if self._connection_manager.is_memory_db:
            return 0.0

        row = conn.execute("PRAGMA database_size").fetchone()
        block_size: int = row[2]

        if block_size <= 0:
            return 0.0

        try:
            current_size = os.path.getsize(cast(str, self._connection_manager.db_path))
        except OSError:
            return 0.0

        # Count blocks holding table data via block allocation metadata.
        # pragma_storage_info reads block-level allocation pages, not row
        # data — no full table scan, very fast.
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='main' AND table_type='BASE TABLE'"
        ).fetchall()

        data_blocks = 0
        for (t,) in tables:
            escaped = t.replace("'", "''")
            count = conn.execute(
                f"SELECT COUNT(DISTINCT block_id) FROM pragma_storage_info('{escaped}') "
                f"WHERE block_id >= 0"
            ).fetchone()[0]
            data_blocks += count

        # Fixed overhead for system metadata: catalog pages, sequences,
        # DuckDB internal bookkeeping, schema_version table.
        system_blocks = 4

        # Index overhead: HNSW + b-tree indexes. Since HNSW is smaller
        # than the embedding table (vectors + graph links, not full rows)
        # and b-tree indexes are smaller than their tables, total index
        # overhead is safely bounded by data_blocks.
        index_factor = 1.0

        expected_min = system_blocks + data_blocks + int(data_blocks * index_factor)

        if expected_min <= 0:
            return 0.0

        # Return ratio of current size to expected minimum size.
        # This catches both MVCC stale data and HNSW drop/recreate bloat
        # because used_blocks includes index blocks while pragma_storage_info
        # only counts table data blocks, leaving the waste visible.
        live = expected_min * block_size
        return current_size / live

    @staticmethod
    def _fragmentation_exceeds_threshold(
        ratio: float, threshold: float | None
    ) -> bool:
        """Return True when the fragmentation ratio exceeds the configured threshold."""
        if threshold is None or threshold < 0:
            return False
        return (ratio - 1.0) * 100.0 > threshold

    def _executor_compact_if_needed(self, conn: Any, state: dict[str, Any]) -> bool:
        """Executor method — checks fragmentation, compacts if needed.

        Runs entirely in the executor thread.  Calls _executor_measure_fragmentation
        and _executor_compact_database directly (no _execute_in_db_thread_sync
        round-trips, which would deadlock the single executor thread).
        """
        if state.get("transaction_active", False):
            return False
        if self._connection_manager.is_memory_db:
            return False
        threshold = self.config.fragmentation_threshold_pct if self.config else 30.0
        ratio = self._executor_measure_fragmentation(conn, state)
        if not self._fragmentation_exceeds_threshold(ratio, threshold):
            log_if_not_mcp(
                "debug",
                f"Fragmentation ratio={ratio:.2f} below threshold={threshold}% — "
                "skipping auto-compaction",
            )
            return False
        log_if_not_mcp(
            "info",
            f"Fragmentation ratio={ratio:.2f} (threshold={threshold}%) — compacting",
        )
        self._executor_compact_database(conn, state)
        return True

    def _compact_prepare(
        self,
        conn: Any,
        state: dict[str, Any],
        db_path: Path,
        backup_path: Path,
        compacted_path: Path,
    ) -> int:
        """Step 1-2: Flush WAL, close connections, move live → backup.

        Caller provides the three filesystem paths so they are computed
        exactly once (in ``_executor_compact_database``).

        Returns original file size in bytes.
        """
        conn.execute("CHECKPOINT")
        original_size = os.path.getsize(db_path)

        self._close_live_compaction_connections()

        backup_path.unlink(missing_ok=True)
        _unlink_compacted(compacted_path)

        # Pre-sleep lets the OS release the file handle before the first
        # rename attempt.  _atomic_replace has exponential-backoff retries,
        # but the pre-sleep avoids entering the retry loop at all.
        if IS_WINDOWS:
            time.sleep(WINDOWS_FILE_HANDLE_DELAY)
        _atomic_replace(db_path, backup_path)
        return original_size

    def _compact_copy_data(
        self, backup_path: Path, compacted_path: Path, state: dict[str, Any]
    ) -> tuple[list[tuple[str]], bool]:
        """Step 3: rebuild the compacted DB from ChunkHound-owned canonical tables.

        Compaction intentionally preserves only the canonical ChunkHound schema:
        schema_version, files, chunks, and embeddings_*. Unknown tables are
        dropped by design so the compacted file is a clean ChunkHound-owned DB,
        not a generic DuckDB passthrough.

        HNSW indexes are NOT created during copy — they are rebuilt eagerly
        after the atomic swap in ``_compact_finalize`` only if the original DB
        had HNSW indexes before compaction began (``had_hnsw`` flag).

        Returns (embedding_table_names, had_hnsw).
        """
        tgt_conn = duckdb.connect(str(compacted_path))
        try:
            tgt_conn.execute("LOAD vss")
            tgt_conn.execute("SET hnsw_enable_experimental_persistence = true")
            tgt_conn.execute("SET preserve_insertion_order = false")

            escaped_backup = str(backup_path).replace("'", "''")
            tgt_conn.execute(f"ATTACH '{escaped_backup}' AS src")

            # Compute MAX(id) from source tables so we can seed the fresh
            # sequences with the correct START value.  The old approach of
            # SELECT nextval FROM range(N) after the swap is O(max_id).
            max_file_id = tgt_conn.execute(
                "SELECT COALESCE(MAX(id), 0) FROM src.files"
            ).fetchone()[0]
            max_chunk_id = tgt_conn.execute(
                "SELECT COALESCE(MAX(id), 0) FROM src.chunks"
            ).fetchone()[0]
            emb_rows = tgt_conn.execute(
                "SELECT table_name FROM information_schema.tables "
                f"WHERE {_embedding_tables_where_clause(catalog='src')}"
            ).fetchall()
            max_embedding_id = 0
            for (tname,) in emb_rows:
                qtname = self._quote_duckdb_identifier(tname)
                table_max = tgt_conn.execute(
                    f"SELECT COALESCE(MAX(id), 0) FROM src.{qtname}"
                ).fetchone()[0]
                max_embedding_id = max(max_embedding_id, int(table_max))

            tgt_conn.execute(f"CREATE SEQUENCE files_id_seq START {max_file_id + 1}")
            tgt_conn.execute(f"CREATE SEQUENCE chunks_id_seq START {max_chunk_id + 1}")
            tgt_conn.execute(f"CREATE SEQUENCE embeddings_id_seq START {max_embedding_id + 1}")

            tgt_conn.execute(f"CREATE TABLE files ({_FILES_TABLE_COLUMNS})")
            tgt_conn.execute(f"CREATE TABLE chunks ({_CHUNKS_TABLE_COLUMNS})")

            # Check whether the source catalog already had HNSW indexes.
            # Compaction only restores HNSW when the pre-compaction source DB
            # had one, so a false negative here would silently degrade search.
            # Treat source-catalog discovery as fatal instead of best-effort.
            had_hnsw = (
                len(
                    self._executor_get_existing_vector_indexes_from_catalog(
                        tgt_conn, catalog="src", fail_on_error=True
                    )
                )
                > 0
            )

            tgt_conn.execute(
                f"CREATE TABLE schema_version ({_SCHEMA_VERSION_TABLE_COLUMNS})"
            )
            tgt_conn.execute(
                "INSERT INTO schema_version SELECT * FROM src.schema_version"
            )

            flist = ", ".join(_FILES_COLUMN_NAMES)
            tgt_conn.execute(
                f"INSERT INTO files ({flist}) SELECT {flist} FROM src.files"
            )

            clist = ", ".join(_CHUNKS_COLUMN_NAMES)
            tgt_conn.execute(
                f"INSERT INTO chunks ({clist}) SELECT {clist} FROM src.chunks"
            )

            emb_tables = tgt_conn.execute(
                "SELECT table_name FROM information_schema.tables "
                f"WHERE {_embedding_tables_where_clause(catalog='src')}"
            ).fetchall()

            for (tname,) in emb_tables:
                qtname = self._quote_duckdb_identifier(tname)
                dims = _embedding_dims_from_table_name(tname)
                if dims is None:
                    tgt_conn.execute(
                        f"CREATE TABLE {qtname} AS SELECT * FROM src.{qtname} LIMIT 0"
                    )
                    tgt_conn.execute(f"INSERT INTO {qtname} SELECT * FROM src.{qtname}")
                else:
                    tgt_conn.execute(
                        f"CREATE TABLE {qtname} ({_embedding_table_columns(dims)})"
                    )
                    elist = ", ".join(_embeddings_column_names(dims))
                    tgt_conn.execute(
                        f"INSERT INTO {qtname} ({elist}) SELECT {elist} FROM src.{qtname}"
                    )

            known = set(CANONICAL_TABLE_NAMES)
            known.update(t[0] for t in emb_tables)
            all_src = tgt_conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_catalog='src' AND table_type='BASE TABLE'"
            ).fetchall()
            dropped_tables = sorted(tname for (tname,) in all_src if tname not in known)
            if dropped_tables:
                msg = (
                    "DuckDB compaction dropped non-ChunkHound tables by design: "
                    f"{', '.join(dropped_tables[:5])}"
                    + (
                        f" (+{len(dropped_tables) - 5} more)"
                        if len(dropped_tables) > 5
                        else ""
                    )
                )
                if os.environ.get("CHUNKHOUND_MCP_MODE") == "1":
                    # Destructive operation — surface to stderr even in MCP
                    # mode (loguru sinks are disabled by stdio.py)
                    print(f"WARNING: {msg}", file=sys.stderr)
                else:
                    logger.warning(msg)

            tgt_conn.execute("DETACH src")

            for (tname,) in emb_tables:
                dims = _embedding_dims_from_table_name(tname)
                if dims is not None:
                    self._executor_create_embedding_table_indexes(
                        tgt_conn, state, tname, dims, create_hnsw=False,
                    )

            tgt_conn.execute("CHECKPOINT")
            return emb_tables, had_hnsw
        finally:
            tgt_conn.close()

    def _compact_finalize(
        self,
        state: dict[str, Any],
        db_path: Path,
        compacted_path: Path,
        backup_path: Path,
        original_size: int,
        had_hnsw: bool,
        _t0: float,
    ) -> int:
        """Steps 4-8: atomic swap, reconnect, restore indexes, cleanup.

        Sequences are already seeded with the correct START value in
        ``_compact_copy_data``, so no reseeding is needed here.
        """
        # Pre-sleep lets the OS release the file handle before the first
        # rename attempt.  _atomic_replace has exponential-backoff retries,
        # but the pre-sleep avoids entering the retry loop at all.
        if IS_WINDOWS:
            time.sleep(WINDOWS_FILE_HANDLE_DELAY)
        _atomic_replace(compacted_path, db_path)

        new_conn = self._create_connection()
        _executor_local.connection = new_conn
        self._connection_manager.connection = (
            self._create_connection_manager_connection()
        )

        self._executor_create_indexes(new_conn, state)

        # Only recreate HNSW if the original DB had it before compaction.
        # During batch CLI indexing, HNSW is dropped at the start (bookends)
        # and intermediate compactions must not recreate it mid-pipeline.
        if had_hnsw:
            self._executor_ensure_all_hnsw_indexes(new_conn, state)

        backup_path.unlink(missing_ok=True)

        compacted_size = os.path.getsize(db_path)
        reduction = (
            (1.0 - compacted_size / original_size) * 100.0
            if original_size > 0
            else 0.0
        )
        log_if_not_mcp(
            "info",
            f"Compaction complete: {original_size:,} → {compacted_size:,} bytes "
            f"({reduction:.1f}% reduction, {time.monotonic() - _t0:.1f}s)",
        )
        return compacted_size

    def _close_live_compaction_connections(self) -> None:
        """Close any live handles before filesystem-level restore/swap work."""
        thread_conn = getattr(_executor_local, "connection", None)
        if thread_conn is not None:
            try:
                thread_conn.close()
            finally:
                delattr(_executor_local, "connection")

        manager_conn = self._connection_manager.connection
        if manager_conn is not None:
            try:
                manager_conn.close()
            finally:
                self._connection_manager.connection = None

    def _compact_restore(
        self,
        backup_path: Path,
        compacted_path: Path,
        db_path: Path,
        state: dict[str, Any],
        *,
        had_hnsw: bool = False,
    ) -> None:
        """Restore original DB from backup after a compaction failure."""
        if not db_path or not backup_path:
            log_if_not_mcp(
                "warning", "Cannot restore: compaction paths not initialized"
            )
            _unlink_compacted(compacted_path)
            return
        self._close_live_compaction_connections()

        # Pre-sleep lets the OS release the file handle before the first
        # rename attempt.  _atomic_replace has exponential-backoff retries,
        # but the pre-sleep avoids entering the retry loop at all.
        if IS_WINDOWS:
            time.sleep(WINDOWS_FILE_HANDLE_DELAY)
        if backup_path.exists():
            _atomic_replace(backup_path, db_path)
        _unlink_compacted(compacted_path)
        restored_conn = self._create_connection()
        # Assign immediately so _close_live_compaction_connections() can
        # close it if a later step (schema/index creation) raises.
        _executor_local.connection = restored_conn
        self._executor_create_schema(restored_conn, state)
        self._executor_create_indexes(restored_conn, state)
        # Restore HNSW only if the backup DB had it.  During batch CLI
        # indexing the backup won't have HNSW (dropped at start), so
        # intermediate compaction failures won't recreate it.
        if had_hnsw:
            self._executor_ensure_all_hnsw_indexes(restored_conn, state)
        self._connection_manager.connection = (
            self._create_connection_manager_connection()
        )

    def _executor_compact_database(self, conn: Any, state: dict[str, Any]) -> int:
        """Compact the database file via canonical rebuild + atomic swap.

        The live DB is checkpointed, moved aside to a backup path, rebuilt into
        a fresh DuckDB file from ChunkHound-owned canonical tables, then swapped
        back into place atomically. Unknown/non-canonical tables are dropped by
        design during rebuild.

        Must be called with no active transaction.
        """
        if self._connection_manager.is_memory_db:
            return 0
        if state.get("transaction_active", False):
            raise RuntimeError("Cannot compact database while a transaction is active")

        self._executor.set_compaction_in_progress(True)
        _t0 = time.monotonic()
        db_path = Path(cast(str, self._connection_manager.db_path))
        backup_path = db_path.with_suffix(db_path.suffix + ".compact_backup")
        compacted_path = db_path.with_suffix(db_path.suffix + ".compact_new")
        original_size = 0
        had_hnsw = False

        try:
            original_size = self._compact_prepare(
                conn, state, db_path, backup_path, compacted_path
            )
            _, had_hnsw = self._compact_copy_data(
                backup_path, compacted_path, state
            )
            return self._compact_finalize(
                state,
                db_path,
                compacted_path,
                backup_path,
                original_size,
                had_hnsw,
                _t0,
            )
        except Exception as e:
            log_if_not_mcp("error", f"Compaction failed: {e}")
            try:
                self._compact_restore(
                    backup_path, compacted_path, db_path, state,
                    had_hnsw=had_hnsw,
                )
            except Exception as restore_err:
                log_if_not_mcp(
                    "error",
                    f"Compaction failed AND restore failed: {restore_err}. "
                    f"Backup at: {backup_path}",
                )
            raise
        finally:
            self._executor.set_compaction_in_progress(False)

    def _executor_files_table_needs_rebuild(self, conn: Any) -> bool:
        """Return True when files no longer matches canonical DDL."""
        if not self._executor_table_has_primary_key(conn, "files"):
            return True
        return self._executor_get_column_default(conn, "files", "id") != (
            "nextval('files_id_seq')"
        )

    def _executor_chunks_table_needs_rebuild(self, conn: Any) -> bool:
        """Return True when chunks no longer matches canonical DDL."""
        if not self._executor_table_has_primary_key(conn, "chunks"):
            return True
        if self._executor_get_column_default(conn, "chunks", "id") != (
            "nextval('chunks_id_seq')"
        ):
            return True
        return not self._executor_table_has_foreign_key(
            conn, "chunks", "file_id", "files", "id"
        )

    def _executor_materialize_files_table(
        self, conn: Any, state: dict[str, Any]
    ) -> None:
        """Create or rebuild files from canonical shared DDL."""
        conn.execute("CREATE SEQUENCE IF NOT EXISTS files_id_seq")
        if not self._executor_table_exists(conn, state, "files"):
            conn.execute(f"CREATE TABLE files ({_FILES_TABLE_COLUMNS})")
            return

        conn.execute("ALTER TABLE files ADD COLUMN IF NOT EXISTS content_hash TEXT")
        conn.execute("ALTER TABLE files ADD COLUMN IF NOT EXISTS skip_reason TEXT")

        if self._executor_files_table_needs_rebuild(conn):
            logger.warning("Rebuilding non-canonical files table from shared DDL")
            next_value = conn.execute(
                "SELECT COALESCE(MAX(id), 0) + 1 FROM files"
            ).fetchone()[0]
            self._executor_recreate_sequence(conn, "files_id_seq", int(next_value))
            self._executor_rebuild_table_from_canonical_schema(
                conn,
                "files",
                _FILES_TABLE_COLUMNS,
                _FILES_COLUMN_NAMES,
            )
            return

        self._executor_reseed_sequence(conn, "files", "files_id_seq")

    def _executor_materialize_chunks_table(
        self, conn: Any, state: dict[str, Any]
    ) -> None:
        """Create or rebuild chunks from canonical shared DDL."""
        conn.execute("CREATE SEQUENCE IF NOT EXISTS chunks_id_seq")
        if not self._executor_table_exists(conn, state, "chunks"):
            conn.execute(f"CREATE TABLE chunks ({_CHUNKS_TABLE_COLUMNS})")
            return

        conn.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS metadata TEXT")

        if self._executor_chunks_table_needs_rebuild(conn):
            logger.warning("Rebuilding non-canonical chunks table from shared DDL")
            next_value = conn.execute(
                "SELECT COALESCE(MAX(id), 0) + 1 FROM chunks"
            ).fetchone()[0]
            self._executor_recreate_sequence(conn, "chunks_id_seq", int(next_value))
            self._executor_rebuild_table_from_canonical_schema(
                conn,
                "chunks",
                _CHUNKS_TABLE_COLUMNS,
                _CHUNKS_COLUMN_NAMES,
            )
            return

        self._executor_reseed_sequence(conn, "chunks", "chunks_id_seq")

    def create_schema(self) -> None:
        """Create database schema for files, chunks, and embeddings - delegate to connection manager."""
        self._execute_in_db_thread_sync("create_schema")

    def _executor_create_schema(self, conn: Any, state: dict[str, Any]) -> None:
        """Executor method for create_schema - runs in DB thread."""
        logger.info("Creating DuckDB schema")

        try:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS schema_version "
                f"({_SCHEMA_VERSION_TABLE_COLUMNS})"
            )

            self._executor_materialize_files_table(conn, state)
            self._executor_materialize_chunks_table(conn, state)
            conn.execute("CREATE SEQUENCE IF NOT EXISTS embeddings_id_seq")

            # Reseed embeddings_id_seq to match existing data.  The sequence
            # was created above (start=1).  If embedding tables already have
            # rows (e.g. after compaction restore), advance the sequence so
            # the nextval call returns max(id)+1.  Missing this reseed causes
            # PRIMARY KEY violations — the sequence starts at 1 while restored
            # rows have IDs up to millions.
            max_eid = self._executor_get_max_embedding_id(conn)
            if max_eid > 0:
                cur = conn.execute(
                    "SELECT last_value FROM duckdb_sequences() "
                    "WHERE sequence_name = 'embeddings_id_seq'"
                ).fetchone()
                cv = cur[0] if cur else None
                if cv is None:
                    cv = conn.execute(
                        "SELECT nextval('embeddings_id_seq')"
                    ).fetchone()[0]
                adv = max_eid - int(cv)
                if adv > 0:
                    conn.execute(
                        f"SELECT nextval('embeddings_id_seq') FROM range({adv})"
                    )

            self._executor_migrate_schema(conn, state)

            embedding_dims: list[int] = []
            for (table_name,) in conn.execute(
                "SELECT table_name FROM information_schema.tables "
                f"WHERE {_embedding_tables_where_clause()} "
                "ORDER BY table_name"
            ).fetchall():
                dims = _embedding_dims_from_table_name(table_name)
                if dims is not None:
                    embedding_dims.append(dims)

            if embedding_dims and any(
                self._executor_embedding_table_needs_rebuild(
                    conn, _embedding_table_name(dims), dims
                )
                for dims in embedding_dims
            ):
                logger.warning(
                    "Rebuilding non-canonical embedding tables from shared DDL"
                )
                self._executor_rebuild_embedding_tables(conn, state, embedding_dims)
            else:
                for dims in embedding_dims:
                    self._executor_materialize_embedding_table(conn, state, dims)

            current_version = self._get_schema_version(conn)
            if current_version == 0:
                conn.execute("""
                    INSERT INTO schema_version (version, description)
                    VALUES (1, 'Initial schema')
                """)
                logger.info("Schema version initialized to 1")

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
                # Wrap in transaction to prevent data loss on failure
                try:
                    conn.execute("BEGIN TRANSACTION")
                    state["transaction_active"] = True

                    # First, create a temporary table with the new schema
                    conn.execute("""
                        CREATE TEMP TABLE chunks_new AS
                        SELECT id, file_id, chunk_type, symbol, code,
                               start_line, end_line, start_byte, end_byte,
                               language, NULL AS metadata, created_at, updated_at
                        FROM chunks
                    """)

                    # Drop the old table
                    conn.execute("DROP TABLE chunks")

                    # Create the new table with correct schema
                    conn.execute(f"CREATE TABLE chunks ({_CHUNKS_TABLE_COLUMNS})")

                    # Copy data back with explicit column list for safety
                    conn.execute("""
                        INSERT INTO chunks (
                            id, file_id, chunk_type, symbol, code,
                            start_line, end_line, start_byte, end_byte,
                            language, metadata, created_at, updated_at
                        )
                        SELECT id, file_id, chunk_type, symbol, code,
                               start_line, end_line, start_byte, end_byte,
                               language, metadata, created_at, updated_at
                        FROM chunks_new
                    """)

                    # Drop the temporary table
                    conn.execute("DROP TABLE chunks_new")

                    conn.execute("COMMIT")
                    state["transaction_active"] = False
                except Exception:
                    try:
                        conn.execute("ROLLBACK")
                    except Exception as rollback_error:
                        logger.error(
                            f"ROLLBACK failed during migration: {rollback_error}"
                        )
                    state["transaction_active"] = False
                    raise

                # Recreate indexes (will be done in _executor_create_indexes)
                logger.info("Successfully migrated chunks table schema")

            # Add metadata column if it doesn't exist (for databases without size/signature migration)
            conn.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS metadata TEXT")

            # Add skip_reason column for tracking files skipped during parsing
            conn.execute("ALTER TABLE files ADD COLUMN IF NOT EXISTS skip_reason TEXT")

            for (table_name,) in conn.execute(
                "SELECT table_name FROM information_schema.tables "
                f"WHERE {_embedding_tables_where_clause()} "
                "ORDER BY table_name"
            ).fetchall():
                dims = _embedding_dims_from_table_name(table_name)
                if dims is None:
                    continue
                self._executor_ensure_embedding_upsert_contract(
                    conn, state, table_name, dims
                )

        except Exception as e:
            logger.error(f"Failed to migrate schema: {e}")
            raise

    def _get_schema_version(self, conn: Any) -> int:
        """Get current schema version from database.

        Returns 0 if schema_version table doesn't exist or is empty.
        """
        try:
            # Check if table exists
            result = conn.execute("""
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_name = 'schema_version'
            """).fetchone()

            if not result or result[0] == 0:
                return 0

            # Get max version
            result = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
            return result[0] if result and result[0] is not None else 0
        except Exception:
            return 0

    def _get_all_embedding_tables(self) -> list[str]:
        """Get list of all embedding tables (dimension-specific) - delegate to connection manager."""
        return self._execute_in_db_thread_sync("get_all_embedding_tables")

    def _executor_get_all_embedding_tables(
        self, conn: Any, state: dict[str, Any]
    ) -> list[str]:
        """Executor method for _get_all_embedding_tables - runs in DB thread."""
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            f"WHERE {_embedding_tables_where_clause()}"
        ).fetchall()

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

    def optimize_tables(self) -> None:
        """Emit DuckDB batch-write metrics for diagnostics.

        DuckDB self-manages table optimization via WAL checkpointing and MVCC,
        so this method only logs bulk-insert metrics for observability.
        Actual physical compaction is handled separately via
        ``compact_database()`` / ``compact_if_needed()``.
        """
        metrics = self._metrics.get("chunks", {})
        rows = int(metrics.get("rows", 0))
        if rows == 0:
            return
        log_if_not_mcp(
            "info",
            "DuckDB chunks bulk metrics: "
            f"files={int(metrics.get('files', 0))} "
            f"rows={rows} "
            f"batches={int(metrics.get('batches', 0))} "
            f"temp_create_s={float(metrics.get('temp_create_s', 0.0)):.3f} "
            f"temp_insert_s={float(metrics.get('temp_insert_s', 0.0)):.3f} "
            f"main_insert_s={float(metrics.get('main_insert_s', 0.0)):.3f}",
        )

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
            # Read rows newest-first within each logical upsert key so duplicate
            # legacy rows collapse onto the latest version during migration.
            embeddings = conn.execute("""
                SELECT chunk_id, provider, model, embedding, dims, created_at
                FROM embeddings
                ORDER BY
                    dims,
                    chunk_id,
                    provider,
                    model,
                    created_at DESC NULLS LAST,
                    id DESC
            """).fetchall()

            if not embeddings:
                logger.info("Legacy embeddings table is empty, dropping it")
                conn.execute("DROP TABLE embeddings")
                return

            by_dims: dict[int, list[tuple[Any, Any, Any, Any, int, Any]]] = {}
            seen_keys_by_dims: dict[int, set[tuple[int, str, str]]] = {}
            duplicate_count = 0
            for emb in embeddings:
                chunk_id, provider, model, embedding, dims, created_at = emb
                dims_value = int(dims)
                dim_rows = by_dims.setdefault(dims_value, [])
                seen_keys = seen_keys_by_dims.setdefault(dims_value, set())
                logical_key = (int(chunk_id), str(provider), str(model))
                if logical_key in seen_keys:
                    duplicate_count += 1
                    continue
                seen_keys.add(logical_key)
                dim_rows.append(
                    (
                        int(chunk_id),
                        provider,
                        model,
                        embedding,
                        dims_value,
                        created_at,
                    )
                )

            # Migrate each dimension group
            for dims, emb_list in by_dims.items():
                table_name = self._executor_ensure_embedding_table_exists(
                    conn, state, dims
                )
                logger.info(f"Migrating {len(emb_list)} embeddings to {table_name}")
                conn.executemany(
                    f"""
                    INSERT INTO {table_name}
                    (chunk_id, provider, model, embedding, dims, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT (chunk_id, provider, model) DO UPDATE
                    SET
                        embedding = EXCLUDED.embedding,
                        dims = EXCLUDED.dims,
                        created_at = EXCLUDED.created_at
                    """,
                    emb_list,
                )

            # Drop legacy table
            conn.execute("DROP TABLE embeddings")
            logger.info(
                f"Successfully migrated embeddings to {len(by_dims)} "
                "dimension-specific tables"
            )
            if duplicate_count:
                logger.info(
                    "Coalesced "
                    f"{duplicate_count} duplicate legacy embedding rows during migration"
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

            normalized_metric = self._normalize_hnsw_metric(metric)
            index_name = self._custom_hnsw_index_name(
                provider,
                model,
                dims,
                normalized_metric,
            )

            # Create HNSW index using VSS extension on the dimension-specific table
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {self._quote_duckdb_identifier(index_name)}
                ON {self._quote_duckdb_identifier(table_name)}
                USING HNSW (embedding)
                WITH (metric = '{normalized_metric}')
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
        """Executor method for drop_vector_index - runs in DB thread.

        Handles both naming patterns:
        - Custom: hnsw_{provider}_{model}_{dims}_{metric} (from create_vector_index)
        - Standard: idx_hnsw_{dims} (from initial table creation)
        """
        # Custom index name pattern (from create_vector_index)
        normalized_metric = self._normalize_hnsw_metric(metric)
        custom_index_name = self._custom_hnsw_index_name(
            provider,
            model,
            dims,
            normalized_metric,
        )
        # Standard index name pattern (from table creation)
        standard_index_name = f"idx_hnsw_{dims}"

        dropped_indexes = []
        try:
            # Try to drop custom index first
            conn.execute(
                f"DROP INDEX IF EXISTS {self._quote_duckdb_identifier(custom_index_name)}"
            )
            dropped_indexes.append(custom_index_name)

            # Also try to drop standard index (created during table initialization)
            conn.execute(
                f"DROP INDEX IF EXISTS {self._quote_duckdb_identifier(standard_index_name)}"
            )
            dropped_indexes.append(standard_index_name)

            logger.info(f"HNSW index drop attempted: {', '.join(dropped_indexes)}")
            return custom_index_name  # Return primary index name for API consistency

        except Exception as e:
            logger.error(f"Failed to drop HNSW indexes: {e}")
            raise

    def _quote_duckdb_identifier(self, identifier: str) -> str:
        """Quote an identifier for DuckDB SQL."""
        return f'"{identifier.replace(chr(34), chr(34) * 2)}"'

    def _is_safe_duckdb_identifier(self, identifier: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier))

    def _sanitize_hnsw_identifier_component(self, value: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip())
        sanitized = re.sub(r"_+", "_", sanitized).strip("_")
        if not sanitized:
            return "value"
        if sanitized[0].isdigit():
            return f"v_{sanitized}"
        return sanitized

    def _normalize_hnsw_metric(self, metric: str) -> str:
        normalized = metric.strip().lower()
        if normalized not in self._SUPPORTED_HNSW_METRICS:
            raise ValueError(
                "Unsupported HNSW metric "
                f"{metric!r}; expected one of {sorted(self._SUPPORTED_HNSW_METRICS)}"
            )
        return normalized

    def _custom_hnsw_index_name(
        self, provider: str, model: str, dims: int, metric: str
    ) -> str:
        normalized_metric = self._normalize_hnsw_metric(metric)
        provider_component = self._sanitize_hnsw_identifier_component(provider)
        model_component = self._sanitize_hnsw_identifier_component(model)
        return (
            f"hnsw_{provider_component}_{model_component}_{int(dims)}_"
            f"{normalized_metric}"
        )

    def _extract_hnsw_metric(self, conn: Any, index_name: str) -> str:
        """Return HNSW similarity metric from ``pragma_hnsw_index_info()``.

        DuckDB strips the ``WITH (metric = '...')`` clause from
        ``duckdb_indexes().sql``, so we cannot recover the metric from
        the CREATE INDEX DDL.  ``pragma_hnsw_index_info()`` returns the
        live metric regardless.  The pragma scans ALL attached catalogs
        (not just ``main``), so it works correctly during compaction when
        the backup DB is ATTACHed as ``src``.
        """
        try:
            row = conn.execute(
                "SELECT metric FROM pragma_hnsw_index_info() WHERE index_name = ? LIMIT 1",
                [index_name],
            ).fetchone()
            if row:
                return str(row[0])
        except Exception:
            logger.debug(
                f"Could not query HNSW metric for {index_name}, defaulting to cosine"
            )
        return "cosine"

    def _extract_custom_hnsw_identity(
        self, index_name: str
    ) -> tuple[str | None, str | None]:
        """Best-effort provider/model extraction from custom HNSW index names."""
        if not index_name.startswith("hnsw_"):
            return None, None

        parts = index_name[5:].split("_")  # Remove 'hnsw_' prefix
        if len(parts) < 4:
            return None, None

        provider_model = "_".join(parts[:-2])
        last_underscore = provider_model.rfind("_")
        if last_underscore > 0:
            return (
                provider_model[:last_underscore],
                provider_model[last_underscore + 1 :],
            )
        if provider_model:
            return provider_model, ""
        return None, None

    def get_existing_vector_indexes(self) -> list[dict[str, Any]]:
        """Get list of existing HNSW vector indexes on all embedding tables."""
        return self._execute_in_db_thread_sync("get_existing_vector_indexes")

    def drop_all_hnsw_indexes(self) -> None:
        """Drop every HNSW index on all embedding tables.

        Called at the start of batch indexing to eliminate per-batch
        index maintenance overhead.
        """
        self._execute_in_db_thread_sync("drop_all_hnsw_indexes")

    def ensure_all_hnsw_indexes(self) -> None:
        """Ensure canonical HNSW indexes exist on all embedding tables.

        Called at the end of batch indexing / compaction to rebuild HNSW
        indexes on the clean DB.
        """
        self._execute_in_db_thread_sync("ensure_all_hnsw_indexes")

    def _executor_get_existing_vector_indexes(
        self, conn: Any, state: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Executor method for get_existing_vector_indexes - runs in DB thread."""
        try:
            results = conn.execute(
                "SELECT index_name, table_name, sql "
                "FROM duckdb_indexes() "
                f"WHERE {_embedding_indexes_where_clause()}"
            ).fetchall()

            indexes = []
            for result in results:
                index_name = result[0]
                table_name = result[1]
                create_sql = result[2]
                if not is_hnsw_index(index_name, create_sql):
                    continue

                dims = _embedding_dims_from_table_name(table_name)
                if dims is None:
                    logger.warning(
                        f"Could not parse dims from HNSW index {index_name} on {table_name}"
                    )
                    continue

                provider: str | None = None
                model: str | None = None
                if index_name.startswith("hnsw_"):
                    provider, model = self._extract_custom_hnsw_identity(index_name)
                elif index_name.startswith("idx_hnsw_"):
                    provider = "generic"
                    model = "generic"

                indexes.append(
                    {
                        "index_name": index_name,
                        "table_name": table_name,
                        "provider": provider,
                        "model": model,
                        "dims": dims,
                        "metric": self._extract_hnsw_metric(conn, index_name),
                    }
                )

            return indexes

        except Exception as e:
            logger.error(f"Failed to get existing vector indexes: {e}")
            return []

    def _executor_get_existing_vector_indexes_from_catalog(
        self,
        conn: Any,
        catalog: str = "main",
        *,
        fail_on_error: bool = False,
    ) -> list[dict[str, Any]]:
        """Return HNSW vector indexes from one DuckDB catalog.

        ``catalog='src'`` is used during compaction after ATTACHing the
        pre-compaction database. Most callers want best-effort discovery and
        can fall back to ``[]`` on catalog issues. Compaction cannot: a false
        negative would skip HNSW restoration, so it passes
        ``fail_on_error=True``.

        Valid catalogs: ``main``, ``src``, defined in ``VALID_CATALOGS`` in
        ``schema_constants.py``. If adding a new catalog, add it to that set.
        """
        if catalog not in VALID_CATALOGS:
            raise ValueError(
                f"Invalid catalog name {catalog!r}; "
                f"expected one of {sorted(VALID_CATALOGS)}"
            )
        try:
            results = conn.execute(
                "SELECT index_name, table_name, sql "
                "FROM duckdb_indexes() "
                f"WHERE {_embedding_indexes_where_clause(catalog=catalog)}"
            ).fetchall()

            indexes = []
            for result in results:
                index_name = result[0]
                table_name = result[1]
                create_sql = result[2]
                if not is_hnsw_index(index_name, create_sql):
                    continue

                dims = _embedding_dims_from_table_name(table_name)
                if dims is None:
                    logger.warning(
                        f"Could not parse dims from HNSW index {index_name} on {table_name}"
                    )
                    continue

                provider = None
                model = None
                if index_name.startswith("hnsw_"):
                    provider, model = self._extract_custom_hnsw_identity(index_name)
                elif index_name.startswith("idx_hnsw_"):
                    provider = "generic"
                    model = "generic"

                indexes.append(
                    {
                        "index_name": index_name,
                        "table_name": table_name,
                        "provider": provider,
                        "model": model,
                        "dims": dims,
                        "metric": self._extract_hnsw_metric(conn, index_name),
                    }
                )

            return indexes
        except Exception as e:
            msg = f"Failed to get existing vector indexes from catalog {catalog}: {e}"
            if fail_on_error:
                raise RuntimeError(msg) from e
            logger.error(msg)
            return []

    def _executor_drop_vector_index_by_name(self, conn: Any, index_name: str) -> None:
        """Drop a specific HNSW index by its current name."""
        conn.execute(
            f"DROP INDEX IF EXISTS {self._quote_duckdb_identifier(index_name)}"
        )

    def _executor_hnsw_index_exists(self, conn: Any, table_name: str) -> bool:
        """Return True when any HNSW index exists on the given table.

        Uses ``is_hnsw_index()`` instead of name-pattern matching so that
        custom-named HNSW indexes (e.g. ``alt_live_idx ON embeddings_3 USING HNSW``)
        are correctly identified.
        """
        results = conn.execute(
            "SELECT index_name, sql "
            "FROM duckdb_indexes() "
            f"WHERE {_embedding_indexes_where_clause()} "
            "AND table_name = ?",
            [table_name],
        ).fetchall()
        for row in results:
            idx_name = row[0]
            create_sql = row[1]
            if is_hnsw_index(idx_name, create_sql):
                return True
        return False

    def _executor_drop_all_hnsw_indexes(self, conn: Any, state: dict[str, Any]) -> None:
        """Drop every HNSW index on all embedding tables.

        Uses catalog-based detection (reuses ``_executor_get_existing_vector_indexes``)
        so custom-named HNSW indexes are properly dropped.  Previously the name-pattern
        filter (``LIKE 'hnsw_%'``) missed indexes like ``alt_live_idx`` that use
        ``USING HNSW``.
        """
        indexes = self._executor_get_existing_vector_indexes(conn, state)
        dropped = 0
        for idx in indexes:
            idx_name = str(idx["index_name"])
            conn.execute(
                f"DROP INDEX IF EXISTS {self._quote_duckdb_identifier(idx_name)}"
            )
            dropped += 1
        if dropped:
            logger.info(f"Dropped {dropped} HNSW index(es) before bulk indexing")

    def _executor_ensure_all_hnsw_indexes(
        self, conn: Any, state: dict[str, Any]
    ) -> None:
        """Create canonical HNSW indexes on all embedding tables that have data."""
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            f"WHERE {_embedding_tables_where_clause()}"
        ).fetchall()
        created = 0
        for (table_name,) in tables:
            row_count = conn.execute(
                f"SELECT COUNT(*) FROM {self._quote_duckdb_identifier(table_name)}"
            ).fetchone()[0]
            if row_count == 0:
                continue
            if self._executor_hnsw_index_exists(conn, table_name):
                continue
            dims = _embedding_dims_from_table_name(table_name)
            if dims is None:
                continue
            hnsw_name = _embedding_hnsw_index_name(dims)
            conn.execute(f"""
                CREATE INDEX {hnsw_name} ON {self._quote_duckdb_identifier(table_name)}
                USING HNSW (embedding)
                WITH (metric = 'cosine')
            """)
            created += 1
        if created:
            conn.execute("CHECKPOINT")
            logger.info(f"Created {created} HNSW index(es)")

    def _executor_recreate_vector_index_from_info(
        self, conn: Any, state: dict[str, Any], index_info: dict[str, Any]
    ) -> None:
        """Recreate one previously discovered HNSW index with its original name."""
        table_name = str(index_info["table_name"])
        dims = int(index_info["dims"])
        self._executor_ensure_embedding_table_exists(conn, state, dims)
        index_name = str(index_info["index_name"])
        if not self._is_safe_duckdb_identifier(index_name):
            logger.warning(
                "Skipping HNSW index restore for unsafe identifier "
                f"{index_name!r} on {table_name}"
            )
            return

        try:
            metric = self._normalize_hnsw_metric(
                str(index_info.get("metric", "cosine"))
            )
        except ValueError as error:
            logger.warning(
                "Skipping HNSW index restore for "
                f"{index_name!r} on {table_name}: {error}"
            )
            return

        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS {self._quote_duckdb_identifier(index_name)}
            ON {self._quote_duckdb_identifier(table_name)}
            USING HNSW (embedding)
            WITH (metric = '{metric}')
        """)

    def _executor_fetch_rows_as_dicts(
        self, conn: Any, query: str, params: list[Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a query in the DB thread and return row dictionaries."""
        cursor = conn.execute(query, params or [])
        rows = cursor.fetchall()
        if not rows:
            return []

        column_names = [desc[0] for desc in cursor.description]
        return [dict(zip(column_names, row)) for row in rows]

    def _executor_insert_row_dict(
        self, conn: Any, table_name: str, row: dict[str, Any]
    ) -> None:
        """Insert one previously snapshotted row back into a table."""
        columns = list(row.keys())
        placeholders = ", ".join(["?"] * len(columns))
        conn.execute(
            f"""
            INSERT INTO {table_name} ({", ".join(columns)})
            VALUES ({placeholders})
            """,
            [row[column] for column in columns],
        )

    def _executor_insert_row_dicts(
        self, conn: Any, table_name: str, rows: list[dict[str, Any]]
    ) -> None:
        """Insert a snapshot batch back into a table with one executemany call."""
        if not rows:
            return

        columns = list(rows[0].keys())
        placeholders = ", ".join(["?"] * len(columns))
        conn.executemany(
            f"""
            INSERT INTO {table_name} ({", ".join(columns)})
            VALUES ({placeholders})
            """,
            [[row[column] for column in columns] for row in rows],
        )

    def _executor_delete_embeddings_for_chunk_ids(
        self, conn: Any, state: dict[str, Any], chunk_ids: list[int]
    ) -> None:
        """Delete embeddings for specific chunks across all embedding tables."""
        if not chunk_ids:
            return

        placeholders = ", ".join(["?"] * len(chunk_ids))
        for table_name in self._executor_get_all_embedding_tables(conn, state):
            conn.execute(
                f"DELETE FROM {table_name} WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            )

    def _executor_get_embedding_row_ids_by_chunk_ids(
        self, conn: Any, state: dict[str, Any], chunk_ids: list[int]
    ) -> dict[str, list[int]]:
        """Resolve exact embedding row ids for the target chunk ids."""
        if not chunk_ids:
            return {}

        placeholders = ", ".join(["?"] * len(chunk_ids))
        row_ids_by_table: dict[str, list[int]] = {}
        for table_name in self._executor_get_all_embedding_tables(conn, state):
            rows = conn.execute(
                f"""
                SELECT id
                FROM {table_name}
                WHERE chunk_id IN ({placeholders})
                ORDER BY id
                """,
                chunk_ids,
            ).fetchall()
            if rows:
                row_ids_by_table[table_name] = [int(row[0]) for row in rows]

        return row_ids_by_table

    def _executor_delete_embeddings_by_row_ids(
        self, conn: Any, table_name: str, row_ids: list[int]
    ) -> None:
        """Delete exact embedding rows by primary key from one embedding table."""
        if not row_ids:
            return

        placeholders = ", ".join(["?"] * len(row_ids))
        conn.execute(f"DELETE FROM {table_name} WHERE id IN ({placeholders})", row_ids)

    def _executor_delete_chunk_ids_in_active_transaction(
        self,
        conn: Any,
        state: dict[str, Any],
        chunk_ids: list[int],
        mutation_label: str,
    ) -> None:
        """Delete chunk replacements inside an existing outer transaction.

        Step 38 reopens because routing modified-file chunk replacement through the
        generic HNSW drop/recreate guard inside `process_file(...)`'s already-open
        transaction can later crash DuckDB during follow-up embedding inserts.

        For this specific in-transaction path, delete the exact embedding rows by
        primary key and then remove the chunk rows, while keeping Step 37's guarded
        HNSW path for non-transactional chunk/file cleanup.
        """
        unique_chunk_ids = sorted({int(chunk_id) for chunk_id in chunk_ids})
        if not unique_chunk_ids:
            return

        chunk_placeholders = ", ".join(["?"] * len(unique_chunk_ids))
        row_ids_by_table = self._executor_get_embedding_row_ids_by_chunk_ids(
            conn, state, unique_chunk_ids
        )

        for table_name, row_ids in row_ids_by_table.items():
            self._executor_delete_embeddings_by_row_ids(conn, table_name, row_ids)

            remaining_embedding_count = conn.execute(
                f"""
                SELECT COUNT(*)
                FROM {table_name}
                WHERE chunk_id IN ({chunk_placeholders})
                """,
                unique_chunk_ids,
            ).fetchone()[0]
            if remaining_embedding_count:
                raise RuntimeError(
                    f"{mutation_label} left {remaining_embedding_count} stale embedding rows "
                    f"in {table_name}"
                )

        conn.execute(
            f"DELETE FROM chunks WHERE id IN ({chunk_placeholders})",
            unique_chunk_ids,
        )

        remaining_chunk_count = conn.execute(
            f"""
            SELECT COUNT(*)
            FROM chunks
            WHERE id IN ({chunk_placeholders})
            """,
            unique_chunk_ids,
        ).fetchone()[0]
        if remaining_chunk_count:
            raise RuntimeError(
                f"{mutation_label} left {remaining_chunk_count} stale chunk rows"
            )

        logger.info(
            f"{mutation_label} completed inside the active transaction "
            "without the HNSW drop/recreate guard"
        )

    def _executor_snapshot_file_delete_targets(
        self, conn: Any, state: dict[str, Any], file_ids: list[int]
    ) -> dict[str, Any]:
        """Capture the rows needed to restore a batch file delete if mutation fails."""
        unique_file_ids = sorted({int(file_id) for file_id in file_ids})
        if not unique_file_ids:
            return {
                "file_rows": [],
                "file_ids": [],
                "chunk_rows": [],
                "chunk_ids": [],
                "embedding_rows_by_table": {},
            }

        placeholders = ", ".join(["?"] * len(unique_file_ids))
        file_rows = self._executor_fetch_rows_as_dicts(
            conn,
            f"""
            SELECT *
            FROM files
            WHERE id IN ({placeholders})
            ORDER BY id
            """,
            unique_file_ids,
        )
        chunk_rows = self._executor_fetch_rows_as_dicts(
            conn,
            """
            SELECT *
            FROM chunks
            WHERE file_id IN ({placeholders})
            ORDER BY id
            """.format(placeholders=placeholders),
            unique_file_ids,
        )
        chunk_ids = [int(row["id"]) for row in chunk_rows]

        embedding_rows_by_table: dict[str, list[dict[str, Any]]] = {}
        if chunk_ids:
            placeholders = ", ".join(["?"] * len(chunk_ids))
            for table_name in self._executor_get_all_embedding_tables(conn, state):
                rows = self._executor_fetch_rows_as_dicts(
                    conn,
                    f"""
                    SELECT *
                    FROM {table_name}
                    WHERE chunk_id IN ({placeholders})
                    ORDER BY id
                    """,
                    chunk_ids,
                )
                if rows:
                    embedding_rows_by_table[table_name] = rows

        return {
            "file_rows": file_rows,
            "file_ids": [int(row["id"]) for row in file_rows],
            "chunk_rows": chunk_rows,
            "chunk_ids": chunk_ids,
            "embedding_rows_by_table": embedding_rows_by_table,
        }

    def _executor_restore_file_delete_targets(
        self,
        conn: Any,
        state: dict[str, Any],
        snapshot: dict[str, Any],
        original_indexes: list[dict[str, Any]],
    ) -> None:
        """Restore file delete targets and HNSW indexes after a failed mutation."""
        current_indexes = self._executor_get_existing_vector_indexes(conn, state)
        for index_info in current_indexes:
            self._executor_drop_vector_index_by_name(conn, index_info["index_name"])

        chunk_ids = snapshot["chunk_ids"]
        self._executor_delete_embeddings_for_chunk_ids(conn, state, chunk_ids)
        if chunk_ids:
            placeholders = ", ".join(["?"] * len(chunk_ids))
            conn.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", chunk_ids)

        file_ids = snapshot["file_ids"]
        if file_ids:
            placeholders = ", ".join(["?"] * len(file_ids))
            conn.execute(f"DELETE FROM files WHERE id IN ({placeholders})", file_ids)

        self._executor_insert_row_dicts(conn, "files", snapshot["file_rows"])
        self._executor_insert_row_dicts(conn, "chunks", snapshot["chunk_rows"])

        for table_name, rows in snapshot["embedding_rows_by_table"].items():
            self._executor_insert_row_dicts(conn, table_name, rows)

        for index_info in original_indexes:
            self._executor_recreate_vector_index_from_info(conn, state, index_info)

        if not state.get("transaction_active", False):
            conn.execute("CHECKPOINT")

    def _executor_run_hnsw_guarded_mutation(
        self,
        conn: Any,
        state: dict[str, Any],
        mutation_label: str,
        mutation_func: Callable[[], Any],
        *,
        optimize_for_bulk: bool = False,
        transactional: bool = True,
        rollback_func: Callable[[list[dict[str, Any]]], None] | None = None,
    ) -> Any:
        """Run a mutation behind one transactional HNSW drop/recreate guard."""
        if state.get("transaction_active", False) and transactional:
            raise DuckDBTransactionConflictError(
                f"{mutation_label} cannot run while another DuckDB transaction is active"
            )

        existing_indexes = self._executor_get_existing_vector_indexes(conn, state)
        indexes_recreated = False

        try:
            if transactional:
                self._executor_begin_transaction(conn, state)
            if optimize_for_bulk:
                conn.execute("SET preserve_insertion_order = false")

            if existing_indexes:
                logger.info(
                    f"Dropping {len(existing_indexes)} HNSW indexes for {mutation_label}"
                )
                for index_info in existing_indexes:
                    self._executor_drop_vector_index_by_name(
                        conn, index_info["index_name"]
                    )

            result = mutation_func()

            # Commit data changes before recreating HNSW indexes. DuckDB VSS HNSW indexes
            # do not support CreateDeltaIndex, so committing a transaction that contains an
            # HNSW CREATE triggers a BoundIndex::CreateDeltaIndex assertion failure.
            if transactional:
                self._executor_commit_transaction(conn, state, False)

            if existing_indexes:
                logger.info(
                    f"Recreating {len(existing_indexes)} HNSW indexes after {mutation_label}"
                )
                for index_info in existing_indexes:
                    self._executor_recreate_vector_index_from_info(
                        conn, state, index_info
                    )
                indexes_recreated = True

            if not state.get("transaction_active", False):
                conn.execute("CHECKPOINT")
            logger.info(f"{mutation_label} completed successfully with HNSW safety")
            return result
        except Exception as e:
            if transactional and state.get("transaction_active", False):
                try:
                    self._executor_rollback_transaction(conn, state)
                except Exception as rollback_error:
                    logger.error(
                        f"Failed to roll back {mutation_label}: {rollback_error}"
                    )
                    raise RuntimeError(
                        f"{mutation_label} failed: {e}; rollback failed: {rollback_error}"
                    ) from rollback_error
            elif rollback_func is not None:
                try:
                    rollback_func(existing_indexes)
                except Exception as rollback_error:
                    logger.error(
                        f"Failed to restore {mutation_label}: {rollback_error}"
                    )
                    raise RuntimeError(
                        f"{mutation_label} failed: {e}; rollback failed: {rollback_error}"
                    ) from rollback_error
            elif existing_indexes and not indexes_recreated:
                logger.info(
                    f"Attempting best-effort HNSW index restore after {mutation_label} failure"
                )
                for index_info in existing_indexes:
                    try:
                        self._executor_recreate_vector_index_from_info(
                            conn, state, index_info
                        )
                    except Exception as recreate_error:
                        logger.error(
                            "Failed to restore HNSW index "
                            f"{index_info['index_name']} after {mutation_label}: "
                            f"{recreate_error}"
                        )

            logger.error(f"{mutation_label} failed: {e}")
            raise

    def _executor_delete_chunk_ids_with_hnsw_safety(
        self,
        conn: Any,
        state: dict[str, Any],
        chunk_ids: list[int],
        mutation_label: str,
    ) -> None:
        """Delete chunk rows and dependent embeddings behind the HNSW guard."""
        unique_chunk_ids = sorted({int(chunk_id) for chunk_id in chunk_ids})
        if not unique_chunk_ids:
            return

        if state.get("transaction_active", False):
            self._executor_delete_chunk_ids_in_active_transaction(
                conn,
                state,
                unique_chunk_ids,
                mutation_label,
            )
            return

        placeholders = ", ".join(["?"] * len(unique_chunk_ids))

        def delete_records() -> None:
            self._executor_delete_embeddings_for_chunk_ids(
                conn, state, unique_chunk_ids
            )
            conn.execute(
                f"DELETE FROM chunks WHERE id IN ({placeholders})",
                unique_chunk_ids,
            )

        self._executor_run_hnsw_guarded_mutation(
            conn,
            state,
            mutation_label,
            delete_records,
            optimize_for_bulk=len(unique_chunk_ids) > 1,
        )

    def bulk_operation_with_index_management(
        self,
        operation_func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
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
        self,
        conn: Any,
        state: dict[str, Any],
        operation_func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Executor method for bulk operations with index management - runs in DB thread."""
        return self._executor_run_hnsw_guarded_mutation(
            conn,
            state,
            "bulk operation",
            lambda: operation_func(*args, **kwargs),
            optimize_for_bulk=True,
        )

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
                size_bytes = (
                    int(file_dict["size"]) if file_dict["size"] is not None else 0
                )
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
        **kwargs: Any,
    ) -> None:
        """Update file record with new values - delegate to file repository."""
        self._execute_in_db_thread_sync(
            "update_file", file_id, size_bytes, mtime, content_hash
        )

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
            # Clear skip_reason whenever a file is successfully re-indexed
            updates.append("skip_reason = NULL")
            updates.append(
                "updated_at = now()"
            )  # CURRENT_TIMESTAMP parses as a column name in SET clauses; use now() instead
            query = f"UPDATE files SET {', '.join(updates)} WHERE id = ?"
            params.append(file_id)
            conn.execute(query, params)

    def record_skipped_file(
        self,
        path: str,
        name: str,
        extension: str,
        size: int,
        mtime: float,
        language: str | None,
        content_hash: str | None,
        skip_reason: str,
    ) -> None:
        """Upsert a file record with skip_reason to prevent re-scanning on next run."""
        self._execute_in_db_thread_sync(
            "record_skipped_file",
            path,
            name,
            extension,
            size,
            mtime,
            language,
            content_hash,
            skip_reason,
        )

    def _executor_record_skipped_file(
        self,
        conn: Any,
        state: dict[str, Any],
        path: str,
        name: str,
        extension: str,
        size: int,
        mtime: float,
        language: str | None,
        content_hash: str | None,
        skip_reason: str,
    ) -> None:
        """Executor method for record_skipped_file - runs in DB thread."""
        conn.execute(
            """
            INSERT INTO files
                (path, name, extension, size, modified_time, content_hash, language, skip_reason)
            VALUES (?, ?, ?, ?, to_timestamp(?), ?, ?, ?)
            ON CONFLICT (path) DO UPDATE SET
                size = EXCLUDED.size,
                modified_time = EXCLUDED.modified_time,
                content_hash = EXCLUDED.content_hash,
                skip_reason = EXCLUDED.skip_reason,
                updated_at = now()  -- CURRENT_TIMESTAMP parses as a column name in ON CONFLICT SET; use now() instead
            """,
            [path, name, extension, size, mtime, content_hash, language, skip_reason],
        )

    def delete_file_completely(self, file_path: str) -> bool:
        """Delete a file and all its chunks/embeddings completely - delegate to file repository."""
        return cast(
            bool,
            self._execute_in_db_thread_sync("delete_file_completely", file_path),
        )

    def _executor_delete_file_completely(
        self, conn: Any, state: dict[str, Any], file_path: str
    ) -> bool:
        """Executor method for delete_file_completely - runs in DB thread."""
        return self._executor_delete_files_batch(conn, state, [file_path]) > 0

    def _executor_delete_files_batch(
        self, conn: Any, state: dict[str, Any], file_paths: list[str]
    ) -> int:
        """Executor method for delete_files_batch - runs in DB thread."""
        if not file_paths:
            return 0
        if state.get("transaction_active", False):
            raise DuckDBTransactionConflictError(
                "delete_files_batch cannot run while another DuckDB transaction "
                "is active"
            )

        base_dir = state.get("base_directory")
        normalized_paths: list[str] = []
        seen_paths: set[str] = set()
        for file_path in file_paths:
            normalized_path = normalize_path_for_lookup(file_path, base_dir)
            if normalized_path in seen_paths:
                continue
            normalized_paths.append(normalized_path)
            seen_paths.add(normalized_path)

        if not normalized_paths:
            return 0

        placeholders = ", ".join(["?"] * len(normalized_paths))
        existing_rows = self._executor_fetch_rows_as_dicts(
            conn,
            f"""
            SELECT id, path
            FROM files
            WHERE path IN ({placeholders})
            ORDER BY id
            """,
            normalized_paths,
        )
        if not existing_rows:
            return 0

        path_to_file_id = {
            str(row["path"]): int(row["id"]) for row in existing_rows if row.get("path")
        }
        existing_paths = [
            normalized_path
            for normalized_path in normalized_paths
            if normalized_path in path_to_file_id
        ]
        file_ids = [path_to_file_id[path] for path in existing_paths]
        snapshot = self._executor_snapshot_file_delete_targets(conn, state, file_ids)
        chunk_ids = snapshot["chunk_ids"]

        def delete_records() -> int:
            self._executor_delete_embeddings_for_chunk_ids(conn, state, chunk_ids)
            if chunk_ids:
                chunk_placeholders = ", ".join(["?"] * len(chunk_ids))
                conn.execute(
                    f"DELETE FROM chunks WHERE id IN ({chunk_placeholders})",
                    chunk_ids,
                )
            if file_ids:
                file_placeholders = ", ".join(["?"] * len(file_ids))
                conn.execute(
                    f"DELETE FROM files WHERE id IN ({file_placeholders})",
                    file_ids,
                )
            return len(file_ids)

        def rollback_delete(original_indexes: list[dict[str, Any]]) -> None:
            self._executor_restore_file_delete_targets(
                conn, state, snapshot, original_indexes
            )

        deleted_count = self._executor_run_hnsw_guarded_mutation(
            conn,
            state,
            f"delete_files_batch(count={len(file_ids)}, sample={existing_paths[0]})",
            delete_records,
            optimize_for_bulk=len(file_ids) > 1,
            # DuckDB currently rejects parent-row deletes inside the same explicit
            # transaction after child-row deletes on FK-linked tables.
            transactional=False,
            rollback_func=rollback_delete,
        )
        logger.debug(
            f"Deleted {deleted_count} files and associated data in one HNSW-safe batch"
        )
        return int(deleted_count)

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
                    json.dumps(chunk.metadata) if chunk.metadata else None,
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
                language TEXT,
                metadata TEXT
            )
        """)
        _t1 = _t.perf_counter()
        conn.execute("DELETE FROM temp_chunks")
        _t_clear = _t.perf_counter()
        # Bulk insert into temp table
        conn.executemany(
            """
            INSERT INTO temp_chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            chunk_data,
        )
        _t2 = _t.perf_counter()
        # Insert from temp to main table with RETURNING
        result = conn.execute("""
            INSERT INTO chunks (file_id, chunk_type, symbol, code, start_line, end_line,
                              start_byte, end_byte, language, metadata)
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

    def get_chunks_in_range(
        self, file_id: int, start_line: int, end_line: int
    ) -> list[dict]:
        """Get all chunks overlapping a line range - delegate to chunk repository."""
        return self._chunk_repository.get_chunks_in_range(file_id, start_line, end_line)

    def _executor_get_chunks_by_file_id(
        self, conn: Any, state: dict[str, Any], file_id: int, as_model: bool
    ) -> list[dict[str, Any] | Chunk]:
        """Executor method for get_chunks_by_file_id - runs in DB thread."""
        results = conn.execute(
            """
            SELECT id, file_id, chunk_type, symbol, code, start_line, end_line,
                   start_byte, end_byte, language, created_at, updated_at, metadata
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
                "metadata": json.loads(row[12]) if row[12] else {},
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
                    metadata=chunk_dict["metadata"],
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
        chunk_rows = conn.execute(
            "SELECT id FROM chunks WHERE file_id = ? ORDER BY id",
            [file_id],
        ).fetchall()
        chunk_ids = [int(row[0]) for row in chunk_rows]
        self._executor_delete_chunk_ids_with_hnsw_safety(
            conn,
            state,
            chunk_ids,
            f"delete_file_chunks(file_id={file_id}, count={len(chunk_ids)})",
        )

    def _executor_delete_chunk(
        self, conn: Any, state: dict[str, Any], chunk_id: int
    ) -> None:
        """Executor method for delete_chunk - runs in DB thread."""
        self._executor_delete_chunk_ids_with_hnsw_safety(
            conn, state, [chunk_id], f"delete_chunk(id={chunk_id})"
        )

    def _executor_delete_chunks_batch(
        self, conn: Any, state: dict[str, Any], chunk_ids: list[int]
    ) -> None:
        """Executor method for delete_chunks_batch - runs in DB thread."""
        self._executor_delete_chunk_ids_with_hnsw_safety(
            conn,
            state,
            chunk_ids,
            f"delete_chunks_batch(count={len(chunk_ids)})",
        )

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

        result = conn.execute(
            """
            INSERT INTO chunks (file_id, chunk_type, symbol, code, start_line, end_line,
                              start_byte, end_byte, language, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                json.dumps(chunk.metadata) if chunk.metadata else None,
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
                   start_byte, end_byte, language, created_at, updated_at, metadata
            FROM chunks WHERE id = ?
        """,
            [chunk_id],
        ).fetchone()

    def _executor_get_chunks_by_file_id_query(
        self, conn: Any, state: dict[str, Any], file_id: int
    ) -> list[Any]:
        """Executor method for get_chunks_by_file_id query - runs in DB thread."""
        return cast(
            list[Any],
            conn.execute(
                """
                SELECT id, file_id, chunk_type, symbol, code, start_line, end_line,
                       start_byte, end_byte, language, created_at, updated_at, metadata
                FROM chunks WHERE file_id = ?
                ORDER BY start_line
                """,
                [file_id],
            ).fetchall(),
        )

    def _executor_get_chunks_in_range_query(
        self,
        conn: Any,
        state: dict[str, Any],
        file_id: int,
        start_line: int,
        end_line: int,
        query: str,
    ) -> list[Any]:
        """Executor method for get_chunks_in_range query - runs in DB thread.

        Executes the overlap query to find chunks that intersect with a line range.
        """
        return cast(
            list[Any],
            conn.execute(
                query,
                [
                    file_id,
                    start_line,
                    end_line,
                    start_line,
                    end_line,
                    start_line,
                    end_line,
                ],
            ).fetchall(),
        )

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
        """Insert embedding record and return embedding ID."""
        return self._execute_in_db_thread_sync("insert_embedding", embedding)

    def _executor_insert_embedding(
        self, conn: Any, state: dict[str, Any], embedding: Embedding
    ) -> int:
        """Executor method for insert_embedding - runs in the DB thread."""

        table_name = self._executor_ensure_embedding_table_exists(
            conn, state, embedding.dims
        )
        upsert_sql = DuckDBEmbeddingRepository.build_embedding_upsert_sql(table_name)
        conn.execute(
            upsert_sql,
            [
                embedding.chunk_id,
                embedding.provider,
                embedding.model,
                embedding.vector,
                embedding.dims,
            ],
        )
        result = conn.execute(
            f"""
            SELECT id
            FROM {table_name}
            WHERE chunk_id = ? AND provider = ? AND model = ?
            """,
            [embedding.chunk_id, embedding.provider, embedding.model],
        ).fetchone()
        if result is None:
            raise RuntimeError(
                "Embedding upsert completed without a stored row for "
                f"{table_name} ({embedding.chunk_id}, {embedding.provider}, {embedding.model})"
            )
        return int(result[0])

    def insert_embeddings_batch(
        self,
        embeddings_data: list[dict],
        batch_size: int | None = None,
        connection=None,
    ) -> int:
        """Insert multiple embeddings through the executor-owned repository path.

        Args:
            embeddings_data: List of embedding dictionaries
            batch_size: Optional batch size for chunked inserts
            connection: Ignored (executor pattern uses internal connection)

        Returns:
            Number of embeddings inserted
        """
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
        """Executor method for insert_embeddings_batch - runs in DB thread.

        Uses direct batched upserts while preserving any outer executor
        transaction that is already active.
        """
        if not embeddings_data:
            return 0

        transaction_started = False
        if not state.get("transaction_active", False):
            self._executor_begin_transaction(conn, state)
            transaction_started = True

        try:
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
                table_name = self._executor_ensure_embedding_table_exists(
                    conn, state, dims
                )
                upsert_sql = DuckDBEmbeddingRepository.build_embedding_upsert_sql(
                    table_name
                )

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
                        conn.executemany(upsert_sql, batch)
                        total_inserted += len(batch)
                else:
                    # Insert all at once
                    conn.executemany(upsert_sql, batch_data)
                    total_inserted += len(batch_data)

            if transaction_started:
                self._executor_commit_transaction(conn, state, True)
                transaction_started = False

            return total_inserted
        except Exception:
            if transaction_started and state.get("transaction_active", False):
                self._executor_rollback_transaction(conn, state)
            raise

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
        return cast(
            list[dict[str, Any]],
            self._execute_in_db_thread_sync("get_all_chunks_with_metadata"),
        )

    def get_scope_stats(self, scope_prefix: str | None) -> tuple[int, int]:
        """Return (total_files, total_chunks) under an optional scope prefix.

        This is used by code_mapper coverage and must avoid loading full chunk code.
        """
        return cast(
            tuple[int, int],
            self._execute_in_db_thread_sync("get_scope_stats", scope_prefix),
        )

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
        return cast(
            list[str],
            self._execute_in_db_thread_sync("get_scope_file_paths", scope_prefix),
        )

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
                f.language as file_language,
                c.metadata
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
                    "metadata": json.loads(row[10]) if row[10] else {},
                }
            )

        return chunks_with_metadata

    @staticmethod
    def _validate_and_normalize_path_filter(path_filter: str | None) -> str | None:
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

        # Ensure trailing slash for directory patterns.
        # A file extension is a dot that appears AFTER the first character.
        # Leading-dot names without a second dot (.github, .env) are directories.
        # Leading-dot names with an extension (.eslintrc.js, .babelrc.js) are files.
        if normalized and not normalized.endswith("/"):
            last = normalized.split("/")[-1]
            has_file_extension = "." in last[1:]
            if not has_file_extension:
                normalized += "/"

        return normalized

    @staticmethod
    def _build_path_like_pattern(normalized_path: str) -> str:
        """Build a LIKE pattern from a normalized path filter.

        Directory patterns (trailing /) keep wildcards on both sides (%/dir/%).
        File patterns are right-anchored (%/file.py) so module.py does not match
        module.py.bak.

        Metacharacters (_, %) are escaped so they match literally rather than
        acting as single-character or any-sequence wildcards.
        """
        escaped = escape_like_pattern(normalized_path)
        if escaped.endswith("/"):
            return f"%/{escaped}%"
        return f"%/{escaped}"

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

            # Lazy HNSW rebuild: crash recovery from a failed/mid-crash indexing run.
            # DuckDB will brute-force scan without an index — correct but slow.
            if not self._executor_hnsw_index_exists(conn, table_name):
                hnsw_name = _embedding_hnsw_index_name(query_dims)
                logger.info(
                    f"No HNSW index on {table_name}, rebuilding (crash recovery)"
                )
                conn.execute(f"""
                    CREATE INDEX {hnsw_name} ON {table_name}
                    USING HNSW (embedding)
                    WITH (metric = 'cosine')
                """)

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
                    array_cosine_similarity(e.embedding, ?::FLOAT[{query_dims}]) as similarity,
                    c.metadata
                FROM {table_name} e
                JOIN chunks c ON e.chunk_id = c.id
                JOIN files f ON c.file_id = f.id
                WHERE e.provider = ? AND e.model = ?
            """

            params: list[Any] = [query_embedding, provider, model]

            path_like: str | None = None
            if normalized_path is not None:
                path_like = self._build_path_like_pattern(normalized_path)

            if threshold is not None:
                query += f" AND array_cosine_similarity(e.embedding, ?::FLOAT[{query_dims}]) >= ?"
                params.append(query_embedding)
                params.append(threshold)

            if path_like is not None:
                query += " AND CONCAT('/', f.path) LIKE ? ESCAPE '\\'"
                # Prepend / so %/repo_a/% only matches repo_a as a complete directory component.
                params.append(path_like)

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

            if path_like is not None:
                count_query += " AND CONCAT('/', f.path) LIKE ? ESCAPE '\\'"
                # Directory-boundary-anchored match, consistent with main query
                count_params.append(path_like)

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
                    "metadata": json.loads(result[9]) if result[9] else {},
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
        return cast(
            tuple[list[dict[str, Any]], dict[str, Any]],
            self._execute_in_db_thread_sync(
                "search_regex", pattern, page_size, offset, path_filter
            ),
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
                where_conditions.append("CONCAT('/', f.path) LIKE ? ESCAPE '\\'")
                params.append(self._build_path_like_pattern(normalized_path))

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
                    f.language,
                    c.metadata
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
                    "metadata": json.loads(result[8]) if result[8] else {},
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
                path_condition = "AND CONCAT('/', f.path) LIKE ? ESCAPE '\\'"
                params.append(self._build_path_like_pattern(normalized_path))

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
                    c.metadata,
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
                    "metadata": json.loads(result[8]) if result[8] else {},
                    "score": 1.0 - result[9],  # Convert distance to similarity score
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
        return cast(
            list[dict[str, Any]],
            self._execute_in_db_thread_sync(
                "search_by_embedding",
                query_embedding,
                provider,
                model,
                limit,
                threshold,
                path_filter,
            ),
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
            normalized_path = self._validate_and_normalize_path_filter(path_filter)
            path_condition = ""
            query_params: list[Any] = [query_embedding, provider, model]

            if normalized_path is not None:
                path_pattern = self._build_path_like_pattern(normalized_path)
                path_condition = "AND CONCAT('/', f.path) LIKE ? ESCAPE '\\'"
                query_params.append(path_pattern)
            query_params.append(limit)

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
        return cast(
            list[dict[str, Any]],
            self._execute_in_db_thread_sync("execute_query", query, params),
        )

    def list_file_paths_under_directory(self, directory_prefix: str) -> list[str]:
        escaped_prefix = escape_like_pattern(directory_prefix)
        rows = self.execute_query(
            "SELECT path FROM files WHERE path = ? OR path LIKE ? ESCAPE '\\'",
            [directory_prefix, f"{escaped_prefix}/%"],
        )
        return [row["path"] for row in rows if row.get("path")]

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
        conn.execute("BEGIN TRANSACTION")
        state["transaction_active"] = True

    def _executor_commit_transaction(
        self, conn: Any, state: dict[str, Any], force_checkpoint: bool
    ) -> None:
        """Executor method for commit_transaction - runs in DB thread."""
        conn.execute("COMMIT")
        state["transaction_active"] = False
        if force_checkpoint:
            conn.execute("CHECKPOINT")

    def _executor_rollback_transaction(self, conn: Any, state: dict[str, Any]) -> None:
        """Executor method for rollback_transaction - runs in DB thread."""
        try:
            conn.execute("ROLLBACK")
        except duckdb.TransactionException:
            # No active transaction — defensive rollback in except handlers is safe.
            logger.warning("Rollback skipped (no active transaction)")
        state["transaction_active"] = False

    # --- Public compaction / fragmentation interface ---

    def measure_fragmentation(self) -> float:
        """Return fragmentation ratio. Fully stateless.

        Uses PRAGMA database_size, pragma_storage_info (block allocation
        metadata), and os.path.getsize to compare actual file size against
        the estimated minimum live data size.
        < 1.5 = healthy, > 3.0 = fragmented, > 5.0 = significantly fragmented.
        """
        return cast(float, self._execute_in_db_thread_sync("measure_fragmentation"))

    def compact_database(self) -> int:
        """Compact the database, returns compacted size in bytes."""
        return cast(int, self._execute_in_db_thread_sync("compact_database"))

    async def compact_database_async(self) -> int:
        """Async variant of compact_database."""
        return cast(int, await self._execute_in_db_thread("compact_database"))

    def should_optimize(self, operation: str = "") -> bool:
        """Return True if fragmentation exceeds threshold. Fully stateless.

        Uses PRAGMA database_size + file stat. No sidecars or baselines.
        """
        if self._connection_manager.is_memory_db:
            return False
        threshold = self.config.fragmentation_threshold_pct if self.config else 30.0
        ratio = cast(float, self._execute_in_db_thread_sync("measure_fragmentation"))
        return self._fragmentation_exceeds_threshold(ratio, threshold)
