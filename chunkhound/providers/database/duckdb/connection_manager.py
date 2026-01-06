"""DuckDB connection and schema management for ChunkHound."""

import os
import shutil
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


def get_compaction_lock_path(db_path: Path) -> Path:
    """Get the compaction lock file path for a database.

    Single source of truth for lock file location.
    """
    return Path(str(db_path) + ".compaction.lock")


class DuckDBConnectionManager:
    """Utility class for DuckDB database path management and WAL cleanup.

    NOTE: This class does NOT hold a persistent database connection.
    All database connections are managed by DuckDBProvider's executor pattern,
    which ensures thread safety by serializing operations to a single thread.

    This class provides:
    - Database path management
    - WAL file validation and cleanup utilities
    - Extension loading configuration
    """

    def __init__(self, db_path: Path | str, config: Any | None = None):
        """Initialize DuckDB connection manager.

        Args:
            db_path: Path to DuckDB database file or ":memory:" for in-memory database
            config: Database configuration for provider-specific settings
        """
        self._db_path = db_path
        self.config = config
        self._initialized = False

        # Note: Thread safety is handled by DuckDBProvider's executor pattern
        # All database operations are serialized to a single thread
        # This class does NOT hold a persistent connection

    @property
    def db_path(self) -> Path | str:
        """Database connection path or identifier."""
        return self._db_path

    @property
    def is_connected(self) -> bool:
        """Check if connection manager has been initialized.

        NOTE: This does NOT indicate an active connection - connections are
        managed by the executor. This only indicates whether connect() was called.
        """
        return self._initialized

    def connect(self) -> None:
        """Prepare database for connection with WAL validation.

        NOTE: This method does NOT create a persistent connection.
        It performs pre-connection validation and cleanup:
        - Checks for interrupted compaction
        - Ensures parent directory exists
        - Validates and cleans up WAL files if needed

        The actual database connection is created by DuckDBProvider's executor.
        """
        logger.info(f"Preparing DuckDB database: {self.db_path}")

        # Check for interrupted compaction
        if isinstance(self.db_path, Path):
            lock_file = get_compaction_lock_path(self.db_path)
            if lock_file.exists():
                logger.warning(
                    "Found compaction lock file - previous compaction may have been "
                    "interrupted. Running WAL validation..."
                )
                lock_file.unlink(missing_ok=True)

        # Ensure parent directory exists for file-based databases
        if isinstance(self.db_path, Path):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if duckdb is None:
                raise ImportError("duckdb not available")

            # Perform WAL validation and cleanup using temporary connections
            # The actual persistent connection is managed by the executor
            self._preemptive_wal_cleanup()

            # Mark as initialized
            self._initialized = True

            logger.info("DuckDB connection manager initialization complete")

        except Exception as e:
            logger.error(f"DuckDB connection preparation failed: {e}")
            raise

    # NOTE: _connect_with_wal_validation() was removed - connections are now
    # created by DuckDBProvider's executor thread, not here.

    def _is_wal_corruption_error(self, error_msg: str) -> bool:
        """Check if error message indicates WAL corruption."""
        corruption_indicators = [
            "Failure while replaying WAL file",
            'Catalog "chunkhound" does not exist',
            "BinderException",
            "Binder Error",
            "Cannot bind index",
            "unknown index type",
        ]

        return any(indicator in error_msg for indicator in corruption_indicators)

    def _preemptive_wal_cleanup(self) -> None:
        """Proactively check for and clean up potentially corrupted WAL files.

        This prevents segfaults that occur when DuckDB tries to replay corrupted
        WAL files during connection, which can happen before proper error handling
        kicks in.
        """
        if str(self.db_path) == ":memory:":
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

        # Try a quick validation by attempting to open the database
        # If it crashes or fails, clean up the WAL using existing logic
        test_conn = None
        try:
            test_conn = duckdb.connect(str(self.db_path))
            # Simple query to trigger WAL replay
            test_conn.execute("SELECT 1").fetchone()
            logger.debug("WAL file validation passed")
        except Exception as e:
            logger.warning(f"WAL validation failed ({e}), cleaning up WAL file")
            self._handle_wal_corruption()
        finally:
            # Ensure temporary validation connection is always closed
            if test_conn is not None:
                try:
                    test_conn.close()
                except Exception:
                    pass

    def _handle_wal_corruption(self) -> None:
        """Handle WAL corruption by backing up and removing corrupted WAL file."""
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

        # Conservative recovery - remove WAL but create backup first
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
        """Mark connection manager as disconnected.

        NOTE: This method no longer closes a database connection because
        ConnectionManager no longer holds one. Actual connection management
        is handled by DuckDBProvider's executor thread.

        Args:
            skip_checkpoint: Ignored - kept for API compatibility
        """
        self._initialized = False
        if not os.environ.get("CHUNKHOUND_MCP_MODE"):
            logger.debug("DuckDB connection manager marked as disconnected")

    def _load_extensions(self) -> None:
        """Load required DuckDB extensions.

        Note: vss extension has been removed - vector search is now handled
        by ShardManager with USearch indexes.
        """
        # No extensions currently required
        pass

    def health_check(self, conn: Any | None = None) -> dict[str, Any]:
        """Perform health check and return status information.

        Args:
            conn: Database connection to use for health check (from executor).
                  If None, returns basic status without querying database.
        """
        status = {
            "provider": "duckdb",
            "connected": self.is_connected,
            "db_path": str(self.db_path),
            "version": None,
            "tables": [],
            "errors": [],
        }

        if not self.is_connected:
            status["errors"].append("Connection manager not initialized")
            return status

        if conn is None:
            # No connection provided - return basic status
            status["errors"].append("No connection provided for health check")
            return status

        try:
            # Get DuckDB version
            version_result = conn.execute("SELECT version()").fetchone()
            status["version"] = version_result[0] if version_result else "unknown"

            # Check if tables exist
            tables_result = conn.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'main' AND table_type = 'BASE TABLE'
            """).fetchall()

            status["tables"] = [table[0] for table in tables_result]

            # Basic functionality test
            test_result = conn.execute("SELECT 1").fetchone()
            if test_result[0] != 1:
                status["errors"].append("Basic query test failed")

        except Exception as e:
            status["errors"].append(f"Health check error: {str(e)}")

        return status

    def get_connection_info(self) -> dict[str, Any]:
        """Get information about the database configuration.

        NOTE: This returns configuration info, not connection status.
        Connection status is managed by DuckDBProvider's executor.
        """
        return {
            "provider": "duckdb",
            "db_path": str(self.db_path),
            "initialized": self._initialized,
            "memory_database": str(self.db_path) == ":memory:",
        }
