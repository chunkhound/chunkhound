"""DuckDB connection and schema management for ChunkHound."""

import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

# CRITICAL: Import numpy modules FIRST to prevent DuckDB threading segfaults
# This must happen before DuckDB operations start in threaded environments
# See: https://duckdb.org/docs/stable/clients/python/known_issues.html
try:
    import numpy

    # CRITICAL: Import numpy.core.multiarray specifically for threading safety
    # DuckDB docs: "If this module has not been imported from the main thread,
    # and a different thread during execution attempts to import it this causes
    # either a deadlock or a crash"
    import numpy.core.multiarray  # noqa: F401
except ImportError:
    # NumPy not available - VSS extension may not work properly
    pass

# Suppress known SWIG warning from DuckDB Python bindings
# This warning appears in CI environments and doesn't affect functionality
import warnings

warnings.filterwarnings(
    "ignore", message=".*swigvarlink.*", category=DeprecationWarning
)

import duckdb
from loguru import logger

from chunkhound.utils.logging_guard import log_if_not_mcp
from chunkhound.utils.windows_constants import _unlink_compacted


class DuckDBConnectionManager:
    """Manages DuckDB connections, schema creation, and database operations."""

    def __init__(self, db_path: Path | str, config: Any | None = None):
        """Initialize DuckDB connection manager.

        Args:
            db_path: Path to DuckDB database file or ":memory:" for in-memory database
            config: Database configuration for provider-specific settings
        """
        self._db_path = db_path
        self.connection: Any | None = None
        self.config = config
        # :memory: DBs cannot meaningfully be opened read-only — there is
        # nothing to read. Dropping the flag here keeps callers that inherit a
        # project-level read_only setting (e.g. _quickresearch under --config)
        # from asking DuckDB for an impossible open.
        self._read_only = bool(
            config and config.read_only and str(db_path) != ":memory:"
        )

        # Note: Thread safety is now handled by DuckDBProvider's executor pattern
        # All database operations are serialized to a single thread

    @property
    def db_path(self) -> Path | str:
        """Database connection path or identifier."""
        return self._db_path

    @property
    def is_memory_db(self) -> bool:
        """Whether this is an in-memory database."""
        return str(self._db_path) == ":memory:"

    @property
    def is_read_only(self) -> bool:
        """Whether the database was opened in read-only mode."""
        return self._read_only

    @property
    def is_connected(self) -> bool:
        """Check if database connection is active."""
        return self.connection is not None

    def connect(self) -> None:
        """Establish database connection and initialize schema with WAL validation."""
        logger.info(f"Connecting to DuckDB database: {self.db_path}")

        # Ensure parent directory exists for file-based databases. Skipped in
        # read-only mode so a shared / immutable DB path is never written to.
        if isinstance(self.db_path, Path) and not self._read_only:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if duckdb is None:
                raise ImportError("duckdb not available")

            # Connect to database with WAL validation
            # Thread safety is now handled by DuckDBProvider's executor pattern
            # Gate recovery on read-only mode — a read-only observer must never
            # mutate compaction artifacts or attempt WAL replay.
            if not self._read_only:
                self._recover_interrupted_compaction()
                self._preemptive_wal_cleanup()
            else:
                logger.debug(
                    "Read-only mode — skipping compaction recovery and WAL cleanup"
                )
            self._connect_with_wal_validation()

            logger.info("DuckDB connection established")

            # Load required extensions
            self._load_extensions()

            # Note: Schema and index creation is now handled by DuckDBProvider's executor

            logger.info("DuckDB connection manager initialization complete")

        except Exception as e:
            logger.error(f"DuckDB connection failed: {e}")
            raise

    def _connect_with_wal_validation(self) -> None:
        """Connect to DuckDB with WAL corruption detection and automatic cleanup."""
        try:
            # Attempt initial connection
            self.connection = duckdb.connect(
                str(self.db_path), read_only=self._read_only
            )
            logger.debug("DuckDB connection successful")

        except duckdb.Error as e:
            error_msg = str(e)
            # Recovery rewrites files on disk; never run that under read-only.
            if self._read_only:
                if not self.is_memory_db and self._is_wal_corruption_error(error_msg):
                    db_path = Path(self.db_path)
                    wal_file = db_path.with_suffix(db_path.suffix + ".wal")
                    if wal_file.exists():
                        hint = (
                            f"WAL file {wal_file} needs replay. Reopen without "
                            f"--read-only to recover, or remove the WAL file "
                            f"if you know it is stale."
                        )
                    else:
                        hint = f"No WAL file present. Underlying error: {error_msg}"
                    raise RuntimeError(
                        f"Cannot open {self.db_path} read-only: {hint}"
                    ) from e
                raise

            # Check for WAL corruption patterns
            if self._is_wal_corruption_error(error_msg):
                logger.warning(f"WAL corruption detected: {error_msg}")
                self._handle_wal_corruption()

                # Retry connection after WAL cleanup
                try:
                    self.connection = duckdb.connect(str(self.db_path))
                    logger.info("DuckDB connection successful after WAL cleanup")
                except Exception as retry_error:
                    logger.error(
                        f"Connection failed even after WAL cleanup: {retry_error}"
                    )
                    raise
            else:
                # Not a WAL corruption error, re-raise original exception
                raise

    # Method removed - MCP safety is now handled by executor pattern

    def _is_wal_corruption_error(self, error_msg: str) -> bool:
        """Check if error message indicates WAL corruption."""
        corruption_indicators = [
            "Failure while replaying WAL file",
            'Catalog "chunkhound" does not exist',
            "BinderException",
            "Binder Error",
            "Cannot bind index",
            "unknown index type",
            "HNSW",
            "You need to load the extension",
        ]

        return any(indicator in error_msg for indicator in corruption_indicators)

    def _recover_interrupted_compaction(self) -> None:
        """Restore a live DB if a prior compaction crashed mid-swap."""
        if self.is_memory_db:
            return

        db_path = Path(self.db_path)
        backup_path = db_path.with_suffix(db_path.suffix + ".compact_backup")
        staged_path = db_path.with_suffix(db_path.suffix + ".compact_new")
        wal_backup = backup_path.with_suffix(".compact_backup.wal")
        wal_path = db_path.with_suffix(db_path.suffix + ".wal")

        live_missing_or_empty = True
        if db_path.exists():
            try:
                live_missing_or_empty = db_path.stat().st_size == 0
            except OSError:
                live_missing_or_empty = True

        if backup_path.exists() and live_missing_or_empty:
            msg = (
                "WARNING: Recovering DuckDB database from interrupted "
                f"compaction backup: {backup_path}"
            )
            if os.environ.get("CHUNKHOUND_MCP_MODE") == "1":
                # Crash recovery is significant — surface to stderr even in
                # MCP mode (loguru sinks are disabled by stdio.py).
                print(msg, file=sys.stderr)
            else:
                logger.warning(msg)
            try:
                db_path.unlink(missing_ok=True)
                os.replace(backup_path, db_path)
                if wal_backup.exists():
                    wal_path.unlink(missing_ok=True)
                    os.replace(wal_backup, wal_path)
            except OSError as error:
                raise RuntimeError(
                    "Failed to recover interrupted DuckDB compaction from "
                    f"{backup_path}: {error}"
                ) from error

        if not live_missing_or_empty:
            for stale_path in (backup_path, wal_backup):
                stale_path.unlink(missing_ok=True)

        for stale_path in (
            staged_path,
            staged_path.with_suffix(staged_path.suffix + ".wal"),
        ):
            _unlink_compacted(stale_path)

    def _preemptive_wal_cleanup(self) -> None:
        """Proactively check for and clean up potentially corrupted WAL files.

        This prevents segfaults that occur when DuckDB tries to replay corrupted
        WAL files during connection, which can happen before proper error handling
        kicks in.
        """
        if self.is_memory_db:
            return  # No WAL files for in-memory databases

        db_path = Path(self.db_path)
        wal_file = db_path.with_suffix(db_path.suffix + ".wal")

        if not wal_file.exists():
            return  # No WAL file, nothing to clean up

        # Check WAL file age - if it's older than 24 hours, it's likely stale
        try:
            wal_age = time.time() - wal_file.stat().st_mtime
            if wal_age > 86400:  # 24 hours
                logger.warning(
                    f"Found stale WAL file (age: {wal_age / 3600:.1f}h), "
                    "removing preemptively"
                )
                self._handle_wal_corruption()
                return
        except OSError:
            pass

        # Validate WAL by attaching the database via an in-memory connection with VSS
        # pre-loaded. A direct duckdb.connect(file) would crash with
        # "pure virtual function called" if the WAL contains HNSW index operations
        # and VSS is not yet loaded — that crash is a C++ abort, not a catchable exception.
        test_conn = None
        try:
            test_conn = duckdb.connect(":memory:")
            try:
                test_conn.execute("INSTALL vss")
                test_conn.execute("LOAD vss")
                test_conn.execute("SET hnsw_enable_experimental_persistence = true")
            except Exception:
                pass  # If VSS unavailable, ATTACH may still work for non-HNSW DBs
            escaped = str(self.db_path).replace("'", "''")
            test_conn.execute(f"ATTACH '{escaped}' AS validation_db")
            test_conn.execute("SELECT 1").fetchone()
            test_conn.execute("DETACH validation_db")
            logger.debug("WAL file validation passed")
        except Exception as e:
            # If the database is already open by another connection, ATTACH raises
            # "Unique file handle conflict" — that means the file is healthy and in
            # active use, so validation is not needed.
            if "already attached" in str(e) or "Unique file handle conflict" in str(e):
                logger.debug("WAL file validation skipped (database already open)")
                return
            logger.warning(f"WAL validation failed ({e}), cleaning up WAL file")
            self._handle_wal_corruption()
        finally:
            if test_conn is not None:
                try:
                    test_conn.close()
                except Exception:
                    pass

    def _handle_wal_corruption(self) -> None:
        """Handle WAL corruption using advanced recovery with VSS extension."""
        db_path = Path(self.db_path)
        wal_file = db_path.with_suffix(db_path.suffix + ".wal")

        if not wal_file.exists():
            logger.warning(
                f"WAL corruption detected but no WAL file found at: {wal_file}"
            )
            return

        # Get WAL file size for logging
        file_size = wal_file.stat().st_size
        logger.warning(f"WAL corruption detected. File size: {file_size:,} bytes")

        # First attempt: Try recovery with VSS extension preloaded
        logger.info("Attempting WAL recovery with VSS extension preloaded...")

        try:
            # Create a temporary recovery connection
            recovery_conn = duckdb.connect(":memory:")

            # Load VSS extension first
            recovery_conn.execute("INSTALL vss")
            recovery_conn.execute("LOAD vss")

            # Enable experimental persistence for HNSW indexes
            recovery_conn.execute("SET hnsw_enable_experimental_persistence = true")

            # Now attach the database file - this will trigger WAL replay
            # with extension loaded
            escaped = str(db_path).replace("'", "''")
            recovery_conn.execute(f"ATTACH '{escaped}' AS recovery_db")

            # Verify tables are accessible
            recovery_conn.execute("SELECT COUNT(*) FROM recovery_db.files").fetchone()

            # Force a checkpoint to ensure WAL is integrated
            recovery_conn.execute("CHECKPOINT recovery_db")

            # Detach and close
            recovery_conn.execute("DETACH recovery_db")
            recovery_conn.close()

            logger.info("WAL recovery successful with VSS extension preloaded")
            return

        except Exception as recovery_error:
            logger.warning(f"Recovery with VSS preloading failed: {recovery_error}")

            # Second attempt: Conservative recovery - remove WAL but create backup first
            try:
                # Create backup of WAL file before removal
                backup_path = wal_file.with_suffix(".wal.corrupt")
                shutil.copy2(wal_file, backup_path)
                logger.info(f"Created WAL backup at: {backup_path}")

                # Remove corrupted WAL file
                os.remove(wal_file)
                logger.warning(f"Removed corrupted WAL file: {wal_file} (backup saved)")

            except Exception as e:
                logger.error(f"Failed to handle corrupted WAL file {wal_file}: {e}")
                raise

    def disconnect(self, skip_checkpoint: bool = False) -> None:
        """Close database connection with optional checkpointing.

        Args:
            skip_checkpoint: If True, skip the checkpoint operation (useful when
                           checkpoint was already done recently to avoid
                           checkpoint conflicts)
        """
        if self.connection is not None:
            try:
                if (
                    not skip_checkpoint
                    and not self.is_memory_db
                    and not self._read_only
                ):
                    # Force checkpoint before close to ensure durability
                    self.connection.execute("CHECKPOINT")
                    log_if_not_mcp("debug", "Database checkpoint completed before disconnect")
                else:
                    log_if_not_mcp("debug", "Skipping checkpoint before disconnect (already done)")
            except Exception as e:
                log_if_not_mcp("error", f"Checkpoint failed during disconnect: {e}")
                # Continue with close - don't block shutdown
            finally:
                self.connection.close()
                self.connection = None
                log_if_not_mcp("info", "DuckDB connection closed")

    def _load_extensions(self) -> None:
        """Load required DuckDB extensions with macOS x86 crash prevention."""
        logger.info("Loading DuckDB extensions")

        if self.connection is None:
            raise RuntimeError("No database connection")

        try:
            # Install and load VSS extension for vector operations
            self.connection.execute("INSTALL vss")
            self.connection.execute("LOAD vss")
            logger.info("VSS extension loaded successfully")

            # Enable experimental HNSW persistence AFTER VSS extension is loaded
            # This prevents segfaults when DuckDB tries to access vector functionality
            self.connection.execute("SET hnsw_enable_experimental_persistence = true")
            logger.debug("HNSW experimental persistence enabled")

        except Exception as e:
            logger.error(f"Failed to load DuckDB extensions: {e}")
            raise

    def health_check(self) -> dict[str, Any]:
        """Perform health check and return status information."""
        status = {
            "provider": "duckdb",
            "connected": self.is_connected,
            "db_path": str(self.db_path),
            "version": None,
            "extensions": [],
            "tables": [],
            "errors": [],
        }

        if not self.is_connected:
            status["errors"].append("Not connected to database")
            return status

        try:
            # Check connection before proceeding
            if self.connection is None:
                status["errors"].append("Database connection is None")
                return status

            # Get DuckDB version
            version_result = self.connection.execute("SELECT version()").fetchone()
            status["version"] = version_result[0] if version_result else "unknown"

            # Check if VSS extension is loaded
            extensions_result = self.connection.execute("""
                SELECT extension_name, loaded
                FROM duckdb_extensions()
                WHERE extension_name = 'vss'
            """).fetchone()

            if extensions_result:
                status["extensions"].append(
                    {"name": extensions_result[0], "loaded": extensions_result[1]}
                )

            # Check if tables exist
            tables_result = self.connection.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'main' AND table_type = 'BASE TABLE'
            """).fetchall()

            status["tables"] = [table[0] for table in tables_result]

            # Basic functionality test
            test_result = self.connection.execute("SELECT 1").fetchone()
            if test_result[0] != 1:
                status["errors"].append("Basic query test failed")

        except Exception as e:
            status["errors"].append(f"Health check error: {str(e)}")

        return status

    def get_connection_info(self) -> dict[str, Any]:
        """Get information about the database connection."""
        return {
            "provider": "duckdb",
            "db_path": str(self.db_path),
            "connected": self.is_connected,
            "memory_database": self.is_memory_db,
            "connection_type": (
                type(self.connection).__name__ if self.connection else None
            ),
        }
