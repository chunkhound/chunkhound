"""Thread-safe serial executor for database operations requiring single-threaded execution."""

import asyncio
import concurrent.futures
import contextvars
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from loguru import logger

from chunkhound.core.exceptions import CompactionError
from chunkhound.utils.windows_constants import IS_WINDOWS, WINDOWS_FILE_HANDLE_DELAY

# Thread-local storage for executor thread state
_executor_local = threading.local()

COMPACT_SAMPLE_INTERVAL = 100  # 1/100 writes trigger a compaction check

_WRITE_OPERATIONS = frozenset({
    "insert_file",
    "insert_chunk",
    "insert_chunks_batch",
    "insert_embedding",
    "insert_embeddings_batch",
    "delete_file_completely",
    "delete_files_batch",
    "delete_file_chunks",
    "delete_chunks_batch",
    "delete_chunk",
    "update_file",
    "update_chunk",
})


def get_thread_local_connection(provider: Any) -> Any:
    """Get thread-local database connection for executor thread.

    This function should ONLY be called from within the executor thread.

    Args:
        provider: Database provider instance that has _create_connection method

    Returns:
        Thread-local database connection

    Raises:
        RuntimeError: If connection creation fails
    """
    if not hasattr(_executor_local, "connection"):
        # Refuse to create a new connection while compaction holds the file.
        # Placed inside the `not hasattr` guard so that disconnect operations
        # (which need the cached connection) can still proceed.  The gate is
        # effective because _executor_disconnect always delattrs the cached
        # connection — any provider that skips delattr must be fixed there,
        # not here.
        # No TOCTOU risk: only the single executor thread calls this function,
        # and compaction's soft_disconnect() is submitted through the same
        # executor queue, guaranteeing ordering.
        if not provider.is_accepting_connections:
            raise CompactionError(
                "Database connection suspended: compaction in progress",
                operation="connection",
            )
        # Create new connection for this thread
        _executor_local.connection = provider._create_connection()
        if _executor_local.connection is None:
            raise RuntimeError("Failed to create database connection")
        logger.debug(
            f"Created new connection in executor thread {threading.get_ident()}"
        )
    return _executor_local.connection


def get_thread_local_state() -> dict[str, Any]:
    """Get thread-local state for executor thread.

    This function should ONLY be called from within the executor thread.

    Returns the actual dict reference (not a copy) intentionally: executor
    methods mutate the state dict in-place (e.g. toggling
    ``transaction_active``), and those mutations must be visible on the next
    call.

    Returns:
        Thread-local state dictionary
    """
    if not hasattr(_executor_local, "state"):
        _executor_local.state = {
            "transaction_active": False,
            "last_activity_time": time.time(),  # Track last database activity
            "operations_since_checkpoint": 0,
            "last_checkpoint_time": time.time(),
        }
    return _executor_local.state


def track_operation(state: dict[str, Any]) -> None:
    """Track a database operation for checkpoint management.

    This function should ONLY be called from within the executor thread.

    Args:
        state: Thread-local state dictionary
    """
    state["operations_since_checkpoint"] += 1


def reset_thread_local_state() -> None:
    """Reset thread-local state to defaults for clean reconnect.

    Clears session-specific keys (deferred_checkpoint, last_checkpoint_time,
    etc.) and restores default values.  Preserves the dict reference.
    Preserves deferred_hnsw_indexes across reconnect so HNSW indexes deferred
    during a compaction window are not silently lost.
    """
    if hasattr(_executor_local, "state"):
        preserved_hnsw = _executor_local.state.get("deferred_hnsw_indexes")
        _executor_local.state.clear()
        _executor_local.state.update({
            "transaction_active": False,
            "last_activity_time": time.time(),
            "operations_since_checkpoint": 0,
            "last_checkpoint_time": time.time(),
        })
        if preserved_hnsw:
            _executor_local.state["deferred_hnsw_indexes"] = preserved_hnsw


def _maybe_compact(provider: Any) -> None:
    """Call provider.compact_if_needed() if the provider supports it.

    Called probabilistically (every COMPACT_SAMPLE_INTERVAL writes) from the
    executor thread so it runs serially with all other DB operations.
    """
    compact_fn = getattr(provider, "compact_if_needed", None)
    if compact_fn is not None:
        try:
            compact_fn()
        except Exception as e:
            logger.debug(f"Probabilistic compaction check skipped: {e}")


class SerialDatabaseExecutor:
    """Thread-safe executor for database operations requiring single-threaded execution.

    This executor ensures all database operations are serialized through a single thread,
    which is required for databases like DuckDB and LanceDB that don't support concurrent
    access from multiple threads.
    """

    def __init__(self) -> None:
        """Initialize serial executor with single-threaded pool."""
        # Create single-threaded executor for all database operations
        # This ensures complete serialization and prevents concurrent access issues
        self._db_executor = ThreadPoolExecutor(
            max_workers=1,  # Hardcoded - not configurable
            thread_name_prefix="serial-db",
        )

    def execute_sync(
        self, provider: Any, operation_name: str, *args: Any, **kwargs: Any
    ) -> Any:
        """Execute named operation synchronously in DB thread.

        All database operations MUST go through this method to ensure serialization.
        The connection and all state management happens exclusively in the executor thread.

        Args:
            provider: Database provider instance
            operation_name: Name of the executor method to call (e.g., 'search_semantic')
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation

        Returns:
            The result of the operation, fully materialized
        """

        def executor_operation() -> Any:
            # Get thread-local connection (created on first access)
            conn = get_thread_local_connection(provider)

            # Get thread-local state
            state = get_thread_local_state()

            # Update last activity time for ALL operations
            state["last_activity_time"] = time.time()

            # Include base directory for path normalization
            state["base_directory"] = provider.get_base_directory()

            # Execute operation - look for method named _executor_{operation_name}
            op_func = getattr(provider, f"_executor_{operation_name}")
            try:
                result = op_func(conn, state, *args, **kwargs)
            except Exception as exc:
                err_str = str(exc)
                _is_disk_full = "No space left on device" in err_str
                _is_invalidated = "database has been invalidated" in err_str
                if (
                    "Out of Memory" in err_str
                    or "unsuccessful or closed pending query" in err_str
                    or _is_invalidated
                    or _is_disk_full
                ):
                    # DuckDB connection is corrupt — drop it so the next call reconnects cleanly
                    try:
                        getattr(_executor_local.connection, "close", lambda: None)()
                    except Exception:
                        pass
                    if hasattr(_executor_local, "connection"):
                        del _executor_local.connection
                    reset_thread_local_state()
                    if _is_disk_full or _is_invalidated:
                        logger.error(
                            f"DuckDB connection reset after fatal error in '{operation_name}' "
                            f"(disk full or invalidated — free disk space and retry): {exc}"
                        )
                    else:
                        logger.warning(
                            f"DuckDB connection reset after corrupting error in '{operation_name}': {exc}"
                        )
                raise
            # Probabilistic compaction: 1/COMPACT_SAMPLE_INTERVAL writes trigger a check
            if operation_name in _WRITE_OPERATIONS:
                wc = state.get("_write_count", 0) + 1
                state["_write_count"] = wc
                if wc % COMPACT_SAMPLE_INTERVAL == 0:
                    _maybe_compact(provider)
            return result

        # Run in executor synchronously with timeout (env override)
        future = self._db_executor.submit(executor_operation)
        try:
            timeout_s = float(os.getenv("CHUNKHOUND_DB_EXECUTE_TIMEOUT", "30"))
        except Exception:
            timeout_s = 30.0
        try:
            return future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            logger.error(
                f"Database operation '{operation_name}' timed out after {timeout_s} seconds"
            )
            raise TimeoutError(f"Operation '{operation_name}' timed out")

    async def execute_async(
        self, provider: Any, operation_name: str, *args, **kwargs
    ) -> Any:
        """Execute named operation asynchronously in DB thread.

        All database operations MUST go through this method to ensure serialization.
        The connection and all state management happens exclusively in the executor thread.

        Args:
            provider: Database provider instance
            operation_name: Name of the executor method to call (e.g., 'search_semantic')
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation

        Returns:
            The result of the operation, fully materialized
        """
        loop = asyncio.get_running_loop()

        def executor_operation():
            # Get thread-local connection (created on first access)
            conn = get_thread_local_connection(provider)

            # Get thread-local state
            state = get_thread_local_state()

            # Update last activity time for ALL operations
            state["last_activity_time"] = time.time()

            # Include base directory for path normalization
            state["base_directory"] = provider.get_base_directory()

            # Execute operation - look for method named _executor_{operation_name}
            op_func = getattr(provider, f"_executor_{operation_name}")
            try:
                result = op_func(conn, state, *args, **kwargs)
            except Exception as exc:
                err_str = str(exc)
                _is_disk_full = "No space left on device" in err_str
                _is_invalidated = "database has been invalidated" in err_str
                if (
                    "Out of Memory" in err_str
                    or "unsuccessful or closed pending query" in err_str
                    or _is_invalidated
                    or _is_disk_full
                ):
                    try:
                        getattr(_executor_local.connection, "close", lambda: None)()
                    except Exception:
                        pass
                    if hasattr(_executor_local, "connection"):
                        del _executor_local.connection
                    reset_thread_local_state()
                    if _is_disk_full or _is_invalidated:
                        logger.error(
                            f"DuckDB connection reset after fatal error in '{operation_name}' "
                            f"(disk full or invalidated — free disk space and retry): {exc}"
                        )
                    else:
                        logger.warning(
                            f"DuckDB connection reset after corrupting error in '{operation_name}': {exc}"
                        )
                raise
            if operation_name in _WRITE_OPERATIONS:
                wc = state.get("_write_count", 0) + 1
                state["_write_count"] = wc
                if wc % COMPACT_SAMPLE_INTERVAL == 0:
                    _maybe_compact(provider)
            return result

        # Capture context for async compatibility
        ctx = contextvars.copy_context()

        # Run in executor with context
        return await loop.run_in_executor(
            self._db_executor, ctx.run, executor_operation
        )

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor with proper cleanup.

        Args:
            wait: Whether to wait for pending operations to complete
        """
        try:
            self._force_close_connections()
        except RuntimeError:
            pass  # executor already shutdown

        try:
            # Shutdown the executor
            self._db_executor.shutdown(wait=wait)

            # Windows-specific: Small delay to allow file handles to be released
            if IS_WINDOWS:
                time.sleep(WINDOWS_FILE_HANDLE_DELAY)

        except Exception as e:
            logger.error(f"Error during executor shutdown: {e}")

    def _force_close_connections(self) -> None:
        """Force close any thread-local database connections."""

        def close_connection():
            try:
                if hasattr(_executor_local, "connection"):
                    conn = _executor_local.connection
                    if conn and hasattr(conn, "close"):
                        conn.close()
                        logger.debug("Forced close of thread-local connection")
                    # Also clean up the attribute to prevent stale references
                    delattr(_executor_local, "connection")
            except Exception as e:
                logger.error(f"Error force-closing connection: {e}")

        # Submit the close operation to the executor thread
        try:
            future = self._db_executor.submit(close_connection)
            future.result(timeout=2.0)  # Short timeout for cleanup
        except RuntimeError:
            logger.debug("Executor already shutdown, skipping force close")
        except Exception as e:
            # Downgrade to debug - this is expected during double-cleanup scenarios
            logger.debug(f"Force connection close skipped: {e}")

    def clear_thread_local(self) -> None:
        """Clear thread-local storage (for cleanup).

        This should be called when disconnecting to ensure clean state.
        """
        if hasattr(_executor_local, "connection"):
            delattr(_executor_local, "connection")
        if hasattr(_executor_local, "state"):
            delattr(_executor_local, "state")

    def get_last_activity_time(self) -> float | None:
        """Get the last activity time from the executor thread.

        Returns:
            Last activity timestamp, or None if no activity yet
        """

        def get_activity_time():
            state = get_thread_local_state()
            return state.get("last_activity_time", None)

        try:
            future = self._db_executor.submit(get_activity_time)
            return future.result(timeout=1.0)  # Quick operation, short timeout
        except Exception:
            return None
