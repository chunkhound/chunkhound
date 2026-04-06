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
import psutil
from loguru import logger

from chunkhound.core.exceptions.core import CompactionError

# Tables that must exist for a post-compaction DB to be considered valid.
# Created in DuckDBProvider._executor_create_schema.
_REQUIRED_TABLES = frozenset({"files", "chunks"})

_LONG_COMPACTION_WARNING_SECONDS = 600  # 10 minutes

# Intent-file phase constants used by both the provider (writer) and
# connection manager (reader) to coordinate crash recovery.
PHASE_PRE_SWAP = "pre_swap"
PHASE_1 = "phase1"
PHASE_2 = "phase2"


def get_compaction_lock_path(db_path: Path) -> Path:
    """Get the compaction lock file path for a database.

    Single source of truth for lock file location.
    """
    return Path(str(db_path) + ".compaction.lock")


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
    def is_connected(self) -> bool:
        """Check if database connection is active."""
        return self.connection is not None

    def _probe_db_valid(self, path: Path) -> bool:
        """Quick probe: can DuckDB open the file and read metadata?

        Used during crash recovery to verify db_path integrity before
        discarding the pre-compaction backup.
        """
        try:
            conn = duckdb.connect(str(path), read_only=True)
            try:
                tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
                if not _REQUIRED_TABLES.issubset(tables):
                    logger.debug(f"Integrity probe: missing tables (found: {tables})")
                    return False
                return True
            finally:
                conn.close()
        except Exception as e:
            logger.debug(f"Integrity probe failed for {path}: {e}")
            return False

    def _recover_from_intent(
        self,
        phase: str,
        old_db: Path,
        compact_db: Path,
        intent_path: Path,
    ) -> None:
        """Recover from interrupted compaction using intent file."""
        if phase == PHASE_PRE_SWAP:
            # Crash during pre-swap cleanup. db_path should still exist.
            # Discard any leftover compact_db and stale old_db.
            if compact_db.exists():
                compact_db.unlink()
            if old_db.exists():
                old_db.unlink()
        elif phase == PHASE_1:
            # Crash before/during first os.replace (db_path -> old_db).
            # db_path may be missing if the rename completed; restore from old_db.
            if not self.db_path.exists() and old_db.exists():
                logger.warning(
                    "phase1 recovery: restoring database from pre-compaction backup"
                )
                os.replace(old_db, self.db_path)
            # Discard compact_db — compaction didn't finish
            if compact_db.exists():
                compact_db.unlink()
        elif phase == PHASE_2:
            # First rename succeeded (db_path -> old_db).
            # Second rename (compact_db -> db_path) may not have completed.
            if not self.db_path.exists() and compact_db.exists():
                if self._probe_db_valid(compact_db):
                    logger.warning(
                        "phase2 recovery: completing interrupted swap "
                        "with compacted database"
                    )
                    os.replace(compact_db, self.db_path)
                elif old_db.exists():
                    logger.warning(
                        "phase2 recovery: compact database failed integrity probe, "
                        "restoring from pre-compaction backup"
                    )
                    compact_db.unlink()
                    os.replace(old_db, self.db_path)
                else:
                    logger.error(
                        "phase2 recovery: both compact and backup databases unavailable"
                    )
                    raise CompactionError(
                        "Unrecoverable: no valid database or backup found "
                        "after interrupted compaction",
                        operation="recovery",
                    )
            elif not self.db_path.exists() and old_db.exists():
                # compact_db also missing — fall back to old
                logger.warning(
                    "phase2 recovery: compact database missing, restoring from backup"
                )
                os.replace(old_db, self.db_path)
            # If db_path exists, probe and decide
            if self.db_path.exists() and old_db.exists():
                if self._probe_db_valid(self.db_path):
                    old_db.unlink()
                else:
                    logger.warning(
                        "phase2 recovery: database failed integrity probe, "
                        "restoring from pre-compaction backup"
                    )
                    os.replace(old_db, self.db_path)
        else:
            logger.warning(
                f"Unknown intent phase {phase!r}, falling back to legacy recovery"
            )
            self._recover_legacy(old_db, compact_db)

        intent_path.unlink(missing_ok=True)

    def _recover_legacy(self, old_db: Path, compact_db: Path) -> None:
        """Legacy recovery when no intent file exists (pre-intent databases)."""
        if not self.db_path.exists() and old_db.exists():
            logger.warning("Restoring database from pre-compaction state")
            os.replace(old_db, self.db_path)

        # old_db requires integrity check before deletion — on Windows
        # the two-step swap is not atomic, so db_path may be
        # corrupt/incomplete if a crash occurred between the two
        # os.replace() calls.
        if old_db.exists() and self.db_path.exists():
            if self._probe_db_valid(self.db_path):
                old_db.unlink()
            else:
                logger.warning(
                    "Database file failed integrity probe; "
                    "restoring from pre-compaction backup"
                )
                os.replace(old_db, self.db_path)

    def connect(self) -> None:
        """Establish database connection and initialize schema with WAL validation."""
        logger.info(f"Connecting to DuckDB database: {self.db_path}")

        # Recover from interrupted compaction
        if isinstance(self.db_path, Path):
            lock_file = get_compaction_lock_path(self.db_path)
            if lock_file.exists():
                # Check if lock is held by a live process before removing
                lock_is_stale = True
                try:
                    content = lock_file.read_text().strip()
                    # Format: "PID:TIMESTAMP" (new) or "PID" (legacy)
                    parts = content.split(":", 1)
                    pid_str = parts[0]
                    lock_time = (
                        float(parts[1]) if len(parts) > 1 else None
                    )
                    if pid_str.isdigit():
                        pid = int(pid_str)
                        age = (
                            time.time() - lock_time
                            if lock_time is not None
                            else None
                        )
                        try:
                            os.kill(pid, 0)
                            # PID exists — but check for reuse after reboot
                            if (
                                lock_time is not None
                                and lock_time < psutil.boot_time()
                            ):
                                logger.warning(
                                    f"Removing stale compaction lock "
                                    f"(PID {pid} reused after reboot)"
                                )
                            else:
                                # Process alive and lock from current boot
                                lock_is_stale = False
                                threshold = _LONG_COMPACTION_WARNING_SECONDS
                                if age is not None and age > threshold:
                                    logger.warning(
                                        f"Compaction lock held by live process "
                                        f"(PID {pid}, age {age:.0f}s > "
                                        f"{threshold}s). "
                                        "Long compaction? Not removing lock file."
                                    )
                                else:
                                    logger.warning(
                                        f"Compaction lock held by live process "
                                        f"(PID {pid}). Not removing lock file."
                                    )
                        except ProcessLookupError:
                            logger.warning(
                                f"Removing stale compaction lock "
                                f"(PID {pid} no longer running)"
                            )
                        except PermissionError:
                            # Process alive but owned by different user
                            lock_is_stale = False
                            logger.warning(
                                f"Compaction lock held by process PID "
                                f"{pid} (different user). "
                                "Not removing lock file."
                            )
                except (OSError, ValueError):
                    logger.warning(
                        "Removing compaction lock (unreadable or legacy format)"
                    )

                if lock_is_stale:
                    lock_file.unlink(missing_ok=True)

            # Recover from interrupted compaction swap
            old_db = self.db_path.with_suffix(".duckdb.old")
            compact_db = self.db_path.with_suffix(".compact.duckdb")
            export_dir = self.db_path.parent / ".chunkhound_compaction_export"
            intent_path = Path(str(self.db_path) + ".swap_intent")

            if intent_path.exists():
                phase = intent_path.read_text().strip()
                self._recover_from_intent(phase, old_db, compact_db, intent_path)
            else:
                self._recover_legacy(old_db, compact_db)

            # Clean stale compact_db (both paths may leave it)
            if compact_db.exists() and self.db_path.exists():
                compact_db.unlink()

            if export_dir.exists():
                shutil.rmtree(export_dir, ignore_errors=True)

        # Ensure parent directory exists for file-based databases
        if isinstance(self.db_path, Path):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if duckdb is None:
                raise ImportError("duckdb not available")

            # Connect to database with WAL validation
            # Thread safety is now handled by DuckDBProvider's executor pattern
            self._preemptive_wal_cleanup()
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
            self.connection = duckdb.connect(str(self.db_path))
            logger.debug("DuckDB connection successful")

        except duckdb.Error as e:
            error_msg = str(e)

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
            recovery_conn.execute(f"ATTACH '{db_path}' AS recovery_db")

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
                if not skip_checkpoint and not self.is_memory_db:
                    # Force checkpoint before close to ensure durability
                    self.connection.execute("CHECKPOINT")
                    # Only log in non-MCP mode to avoid JSON-RPC interference
                    if not os.environ.get("CHUNKHOUND_MCP_MODE"):
                        logger.debug("Database checkpoint completed before disconnect")
                else:
                    if not os.environ.get("CHUNKHOUND_MCP_MODE"):
                        logger.debug(
                            "Skipping checkpoint before disconnect (already done)"
                        )
            except Exception as e:
                # Only log errors in non-MCP mode
                if not os.environ.get("CHUNKHOUND_MCP_MODE"):
                    logger.error(f"Checkpoint failed during disconnect: {e}")
                # Continue with close - don't block shutdown
            finally:
                self.connection.close()
                self.connection = None
                if not os.environ.get("CHUNKHOUND_MCP_MODE"):
                    logger.info("DuckDB connection closed")

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
