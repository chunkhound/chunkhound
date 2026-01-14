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

from chunkhound.utils.windows_constants import IS_WINDOWS, WINDOWS_FILE_HANDLE_DELAY

# Task-local transaction state to ensure proper isolation in async contexts
_transaction_context = contextvars.ContextVar("transaction_active", default=False)

# Thread-local storage for executor thread state
_executor_local = threading.local()

# Heartbeat state for timeout extension during long-running operations
# Shared between executor thread (writes) and main thread (reads)
_heartbeat_time: float = 0.0
_heartbeat_lock = threading.Lock()


def signal_heartbeat() -> None:
    """Signal that a long-running operation is making progress.

    Call this from within executor thread during long operations
    to prevent timeout. The main thread polling loop will extend
    the deadline when it sees a recent heartbeat.

    Thread-safe: protected by _heartbeat_lock.
    """
    global _heartbeat_time
    with _heartbeat_lock:
        _heartbeat_time = time.time()


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

    Returns:
        Thread-local state dictionary
    """
    if not hasattr(_executor_local, "state"):
        _executor_local.state = {
            "transaction_active": False,
            "last_activity_time": time.time(),  # Track last database activity
        }
    return dict(_executor_local.state)  # Return a typed dict copy


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
            # Reset heartbeat at operation start
            signal_heartbeat()

            # Get thread-local connection (created on first access)
            conn = get_thread_local_connection(provider)

            # Get thread-local state
            state = get_thread_local_state()

            # Update last activity time for ALL operations
            state["last_activity_time"] = time.time()

            # Include base directory if provider has it
            if hasattr(provider, "get_base_directory"):
                state["base_directory"] = provider.get_base_directory()

            # Execute operation - look for method named _executor_{operation_name}
            op_func = getattr(provider, f"_executor_{operation_name}")
            return op_func(conn, state, *args, **kwargs)

        # Run in executor with heartbeat-based timeout extension
        future = self._db_executor.submit(executor_operation)
        try:
            timeout_s = float(os.getenv("CHUNKHOUND_DB_EXECUTE_TIMEOUT", "30"))
        except Exception:
            timeout_s = 30.0

        # Heartbeat polling: check every 1s, extend deadline if heartbeat is fresh
        poll_interval = 1.0
        deadline = time.time() + timeout_s

        while True:
            try:
                remaining = max(0.01, min(poll_interval, deadline - time.time()))
                return future.result(timeout=remaining)
            except concurrent.futures.TimeoutError:
                now = time.time()
                with _heartbeat_lock:
                    last_heartbeat = _heartbeat_time

                # If heartbeat is recent, extend deadline (operation making progress)
                heartbeat_age = now - last_heartbeat
                if heartbeat_age < timeout_s:
                    deadline = now + timeout_s
                    logger.debug(
                        f"Heartbeat received for '{operation_name}', extending timeout"
                    )
                    continue

                # No recent heartbeat - truly timed out
                if now >= deadline:
                    logger.error(
                        f"Database operation '{operation_name}' timed out "
                        f"(no heartbeat for {heartbeat_age:.1f}s)"
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
        loop = asyncio.get_event_loop()

        def executor_operation():
            # Get thread-local connection (created on first access)
            conn = get_thread_local_connection(provider)

            # Get thread-local state
            state = get_thread_local_state()

            # Update last activity time for ALL operations
            state["last_activity_time"] = time.time()

            # Include base directory if provider has it
            if hasattr(provider, "get_base_directory"):
                state["base_directory"] = provider.get_base_directory()

            # Execute operation - look for method named _executor_{operation_name}
            op_func = getattr(provider, f"_executor_{operation_name}")
            return op_func(conn, state, *args, **kwargs)

        # Capture context for async compatibility
        ctx = contextvars.copy_context()

        # Run in executor with context
        return await loop.run_in_executor(
            self._db_executor, ctx.run, executor_operation
        )

    def close_connection(self) -> None:
        """Close thread-local DB connection without shutting down executor.

        Use for temporary disconnections (e.g., compaction) where reconnection
        will happen soon. The executor thread remains alive and can be reused.
        """

        def do_close():
            try:
                if hasattr(_executor_local, "connection"):
                    conn = _executor_local.connection
                    if conn and hasattr(conn, "close"):
                        conn.close()
                        logger.debug("Closed thread-local connection (executor still active)")
                    delattr(_executor_local, "connection")
            except Exception as e:
                logger.error(f"Error closing connection: {e}")

        try:
            future = self._db_executor.submit(do_close)
            future.result(timeout=5.0)
        except Exception as e:
            logger.error(f"Error during connection close: {e}")

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor with proper cleanup.

        Args:
            wait: Whether to wait for pending operations to complete
        """
        try:
            # Only try to close connections if executor is not already shutdown
            # This prevents "cannot schedule new futures after shutdown" errors
            if not getattr(self._db_executor, '_shutdown', False):
                self._force_close_connections()

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
            # Check if executor is still accepting tasks before submitting
            if getattr(self._db_executor, '_shutdown', False):
                logger.debug("Executor already shutdown, skipping force close")
                return
            future = self._db_executor.submit(close_connection)
            future.result(timeout=2.0)  # Short timeout for cleanup
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
        except:
            return None
