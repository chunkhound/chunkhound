"""Thread-safe serial executor.

Requires single-threaded execution for database operations.
"""

import asyncio
import concurrent.futures
import contextvars
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from loguru import logger

from chunkhound.utils.logging_guard import log_if_not_mcp
from chunkhound.utils.windows_constants import IS_WINDOWS, WINDOWS_FILE_HANDLE_DELAY


class DatabaseCompactionInProgressError(RuntimeError):
    """Raised when an operation is attempted while database compaction is active."""


_COMPACTION_OPERATION_TIMEOUT_SECONDS = 660.0


# Probability sampling interval for auto-compaction checks.
# The dispatch layer (execute_sync / execute_async) triggers a stateless
# compact_if_needed() check with probability 1/N per operation, keeping the
# average rate at ~1 check per N operations without any counters or
# cross-thread state.
#
# 100 = ~1 check per 100 write operations. During a typical chunkhound index
# run (~1000-5000 write ops), this yields ~10-50 auto-compaction checks,
# which is frequent enough to catch fragmentation early while keeping the
# fragmentation measurement overhead (PRAGMA + file stat) negligible.
# Reads are excluded — they don't cause fragmentation.
COMPACT_SAMPLE_INTERVAL = 100


# Thread-local storage for executor thread state
_executor_local = threading.local()


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

    Returns the actual dict reference (not a copy) intentionally: executor
    methods mutate the state dict in-place (e.g. toggling ``transaction_active``), and
    those mutations must be visible on the next call.

    Returns:
        Thread-local state dictionary
    """
    if not hasattr(_executor_local, "state"):
        _executor_local.state = {
            "transaction_active": False,
            "last_activity_time": time.time(),  # Track last database activity
        }
    return _executor_local.state


# Write-operation prefixes — only mutations cause fragmentation, so only
# they should trigger post-operation compaction sampling.
_SAMPLEABLE_WRITE_PREFIXES = ("insert", "delete", "update", "upsert", "remove", "process")


def _should_sample_auto_compaction(operation_name: str) -> bool:
    """Return True when post-operation sampled compaction should run.

    Only write operations are sampled — reads don't cause fragmentation.
    """
    if COMPACT_SAMPLE_INTERVAL <= 0:
        return False
    if not any(operation_name.startswith(p) for p in _SAMPLEABLE_WRITE_PREFIXES):
        return False
    return random.randint(0, COMPACT_SAMPLE_INTERVAL - 1) == 0


class SerialDatabaseExecutor:
    """Thread-safe executor for single-threaded database operations.

    This executor serializes all database work through one thread, which is
    required for databases like DuckDB and LanceDB that do not support
    concurrent access from multiple threads.
    """

    def __init__(self) -> None:
        """Initialize serial executor with single-threaded pool."""
        # Create single-threaded executor for all database operations
        # This ensures complete serialization and prevents concurrent access issues
        self._db_executor = ThreadPoolExecutor(
            max_workers=1,  # Hardcoded - not configurable
            thread_name_prefix="serial-db",
        )
        # Shared visibility for callers outside the executor thread so they
        # fail fast instead of queueing behind a long compaction.
        self._compaction_in_progress = threading.Event()

    def set_compaction_in_progress(self, active: bool) -> None:
        """Publish compaction state to callers before they enqueue work."""
        if active:
            self._compaction_in_progress.set()
            return
        self._compaction_in_progress.clear()

    def is_compaction_in_progress(self) -> bool:
        """Return True when compaction is currently active."""
        return self._compaction_in_progress.is_set()

    def _raise_if_compacting_before_submit(self, operation_name: str) -> None:
        """Fast-fail new work while compaction owns the database."""
        if self.is_compaction_in_progress():
            raise DatabaseCompactionInProgressError(
                "Database compaction in progress — retry in a few seconds"
            )

    def execute_sync(
        self, provider: Any, operation_name: str, *args: Any, **kwargs: Any
    ) -> Any:
        """Execute named operation synchronously in DB thread.

        All database operations MUST go through this method to ensure
        serialization. The connection and all state management happens
        exclusively in the executor thread.

        Args:
            provider: Database provider instance
            operation_name: Name of the executor method to call
                (e.g., 'search_semantic')
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation

        Returns:
            The result of the operation, fully materialized
        """

        self._raise_if_compacting_before_submit(operation_name)

        def executor_operation() -> Any:
            # Get thread-local connection (created on first access)
            conn = get_thread_local_connection(provider)

            # Get thread-local state
            state = get_thread_local_state()

            # Update last activity time for ALL operations
            state["last_activity_time"] = time.time()

            # Include base directory if provider has it
            if hasattr(provider, "get_base_directory"):
                state["base_directory"] = provider.get_base_directory()

            # NOTE: The current operation owns the executor while it runs.
            # Compaction may be dispatched by the caller AFTER this operation
            # returns (post-operation auto-compaction).
            # No mid-operation compaction guard is needed.

            # Execute operation - look for method named _executor_{operation_name}
            op_func = getattr(provider, f"_executor_{operation_name}")
            return op_func(conn, state, *args, **kwargs)

        # Run in executor synchronously with timeout (env override)
        future = self._db_executor.submit(executor_operation)
        default_timeout = (
            _COMPACTION_OPERATION_TIMEOUT_SECONDS
            if operation_name.startswith("compact")
            else 30.0
        )
        try:
            timeout_s = float(
                os.getenv("CHUNKHOUND_DB_EXECUTE_TIMEOUT", str(default_timeout))
            )
        except Exception:
            timeout_s = default_timeout
        try:
            result = future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            logger.error(
                f"Database operation '{operation_name}' timed out after "
                f"{timeout_s} seconds"
            )
            raise TimeoutError(f"Operation '{operation_name}' timed out")

        self._maybe_run_sampled_auto_compaction(provider, operation_name)
        return result

    async def execute_async(
        self, provider: Any, operation_name: str, *args, **kwargs
    ) -> Any:
        """Execute named operation asynchronously in DB thread.

        All database operations MUST go through this method to ensure
        serialization. The connection and all state management happens
        exclusively in the executor thread.

        Args:
            provider: Database provider instance
            operation_name: Name of the executor method to call
                (e.g., 'search_semantic')
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation

        Returns:
            The result of the operation, fully materialized
        """
        self._raise_if_compacting_before_submit(operation_name)
        loop = asyncio.get_running_loop()

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

            # NOTE: The current operation owns the executor while it runs.
            # Compaction may be dispatched by the caller AFTER this operation
            # returns (post-operation auto-compaction).
            # No mid-operation compaction guard is needed.

            # Execute operation - look for method named _executor_{operation_name}
            op_func = getattr(provider, f"_executor_{operation_name}")
            return op_func(conn, state, *args, **kwargs)

        # Capture context for async compatibility
        ctx = contextvars.copy_context()

        # Run in executor with context
        result = await loop.run_in_executor(
            self._db_executor, ctx.run, executor_operation
        )

        await self._maybe_run_sampled_auto_compaction_async(provider, operation_name)
        return result

    def _maybe_run_sampled_auto_compaction(
        self, provider: Any, operation_name: str
    ) -> None:
        """Run sampled auto-compaction without failing the triggering operation."""
        if not _should_sample_auto_compaction(operation_name):
            return
        try:
            provider.compact_if_needed()
        except Exception as error:
            log_if_not_mcp(
                "warning", f"Sampled auto-compaction skipped after failure: {error}"
            )

    async def _maybe_run_sampled_auto_compaction_async(
        self, provider: Any, operation_name: str
    ) -> None:
        """Async variant of sampled auto-compaction failure isolation."""
        if not _should_sample_auto_compaction(operation_name):
            return
        try:
            await provider.compact_if_needed_async()
        except Exception as error:
            log_if_not_mcp(
                "warning", f"Sampled auto-compaction skipped after failure: {error}"
            )

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor with proper cleanup.

        Args:
            wait: Whether to wait for pending operations to complete
        """
        try:
            # Force close any thread-local connections first
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
            except Exception as e:
                logger.error(f"Error force-closing connection: {e}")

        # Submit the close operation to the executor thread
        try:
            future = self._db_executor.submit(close_connection)
            future.result(timeout=2.0)  # Short timeout for cleanup
        except Exception as e:
            logger.error(f"Error during force connection close: {e}")

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
