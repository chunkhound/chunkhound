"""Thread-safe serial executor.

Requires single-threaded execution for database operations.
"""

import asyncio
import concurrent.futures
import contextvars
import random
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from loguru import logger

from chunkhound.utils.logging_guard import log_if_not_mcp


class DatabaseOperationTimeoutError(TimeoutError):
    """The caller's wait expired; submitted database work still owns its handles."""


class DatabaseTemporarilyUnavailableError(RuntimeError):
    """Base for errors raised when the DB is fast-failed because some other
    owner (compaction, Rust pipeline) currently controls it."""


class DatabaseCompactionInProgressError(DatabaseTemporarilyUnavailableError):
    """Raised when an operation is attempted while database compaction is active."""


class DatabaseRustPipelineInProgressError(DatabaseTemporarilyUnavailableError):
    """Raised when an operation is attempted while the Rust indexing pipeline
    currently owns write access to the database file."""


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
    provider._ensure_database_ownership()
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

    def __init__(self, execute_timeout_seconds: float | None = None) -> None:
        """Initialize serial executor with single-threaded pool.

        Args:
            execute_timeout_seconds: Optional timeout for sync
                ``execute_sync`` waits only (async dispatch is unbounded).
                When None, uses built-in defaults (30s normal, 660s
                compaction). When set, applies to every sync operation
                including compaction.
        """
        # Create single-threaded executor for all database operations
        # This ensures complete serialization and prevents concurrent access issues
        self._db_executor = ThreadPoolExecutor(
            max_workers=1,  # Hardcoded - not configurable
            thread_name_prefix="serial-db",
        )
        # Shared visibility for callers outside the executor thread so they
        # fail fast instead of queueing behind a long compaction.
        self._compaction_in_progress = threading.Event()
        # Same idea, for the window where the Rust pipeline owns write
        # access to the DB file (Python's connection is closed).
        self._rust_pipeline_in_progress = threading.Event()
        self._execute_timeout_seconds = execute_timeout_seconds
        self._submission_lock = threading.RLock()
        self._finalization_lock = threading.RLock()
        self._accepting = True
        self._joined = False
        self.handles_closed = True
        self.finalization_complete = False
        self._finalization: concurrent.futures.Future[Any] | None = None

    def resolve_timeout(self, operation_name: str) -> float:
        """Resolve the wait timeout for a named ``execute_sync`` operation.

        Used only by ``execute_sync`` (async dispatch has no timeout).
        When ``execute_timeout_seconds`` is configured, that value is used for
        every sync operation (including compaction). Otherwise compaction ops
        get the extended default and all other ops get 30s.
        """
        if self._execute_timeout_seconds is not None:
            return self._execute_timeout_seconds
        if operation_name.startswith("compact"):
            return _COMPACTION_OPERATION_TIMEOUT_SECONDS
        return 30.0

    def set_compaction_in_progress(self, active: bool) -> None:
        """Publish compaction state to callers before they enqueue work."""
        if active:
            self._compaction_in_progress.set()
            return
        self._compaction_in_progress.clear()

    def is_compaction_in_progress(self) -> bool:
        """Return True when compaction is currently active."""
        return self._compaction_in_progress.is_set()

    def set_rust_pipeline_in_progress(self, active: bool) -> None:
        """Publish Rust-pipeline-owns-the-file state to callers before they enqueue work."""
        if active:
            self._rust_pipeline_in_progress.set()
            return
        self._rust_pipeline_in_progress.clear()

    def is_rust_pipeline_in_progress(self) -> bool:
        """Return True when the Rust pipeline currently owns the database file."""
        return self._rust_pipeline_in_progress.is_set()

    def _raise_if_db_unavailable_before_submit(
        self, operation_name: str, _bypass_rust_pipeline_guard: bool = False
    ) -> None:
        """Fast-fail new work while compaction or the Rust pipeline owns the database."""
        if self.is_compaction_in_progress():
            raise DatabaseCompactionInProgressError(
                "Database compaction in progress — retry in a few seconds"
            )
        if not _bypass_rust_pipeline_guard and self.is_rust_pipeline_in_progress():
            raise DatabaseRustPipelineInProgressError(
                "Rust indexing pipeline owns the database — retry once it finishes"
            )

    def _submit(
        self, function: Callable[..., Any], *args: Any
    ) -> concurrent.futures.Future[Any]:
        with self._submission_lock:
            if not self._accepting:
                raise RuntimeError("Database executor is closing or closed")
            return self._db_executor.submit(function, *args)

    def _execute_operation(
        self,
        provider: Any,
        operation_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        _bypass_rust_pipeline_guard: bool = False,
    ) -> Any:
        """Run one operation's complete handle lifecycle on the owning thread."""
        self._raise_if_db_unavailable_before_submit(
            operation_name, _bypass_rust_pipeline_guard
        )
        conn = (
            getattr(_executor_local, "connection", None)
            if operation_name == "disconnect"
            else get_thread_local_connection(provider)
        )
        self.handles_closed = not hasattr(_executor_local, "connection")
        state = get_thread_local_state()
        state["last_activity_time"] = time.time()
        if hasattr(provider, "get_base_directory"):
            state["base_directory"] = provider.get_base_directory()

        # Auto-compaction is sampled by the caller only after this operation.
        op_func = getattr(provider, f"_executor_{operation_name}")
        try:
            return op_func(conn, state, *args, **kwargs)
        except BaseException as error:
            traceback.clear_frames(error.__traceback__)
            raise
        finally:
            conn = None
            self.handles_closed = not hasattr(_executor_local, "connection")

    def _wait_for_operation(
        self,
        future: concurrent.futures.Future[Any],
        operation_name: str,
        timeout: float | None,
    ) -> Any:
        # Future.result(timeout) uses the same TimeoutError as a backend failure.
        # Observe completion separately so backend errors retain their identity,
        # even when the worker finishes at the caller's deadline boundary.
        done, _ = concurrent.futures.wait((future,), timeout=timeout)
        if future not in done:
            raise DatabaseOperationTimeoutError(
                f"Operation '{operation_name}' timed out"
            )
        return future.result()

    def prepare_connection(self, provider: Any) -> bool:
        """Check reconnect state on its owning thread without opening a connection.

        Unlike normal dispatch, the provider's lifecycle hook must be able to
        preflight a missing/partially released connection before lazy creation.
        """
        self._raise_if_db_unavailable_before_submit("connect")
        future = self._submit(provider._executor_prepare_connection)
        return bool(
            self._wait_for_operation(future, "connect", self.resolve_timeout("connect"))
        )

    def execute_sync(
        self,
        provider: Any,
        operation_name: str,
        *args: Any,
        _bypass_rust_pipeline_guard: bool = False,
        **kwargs: Any,
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
            _bypass_rust_pipeline_guard: Skip only the Rust-pipeline fast-fail
                check (the compaction check still applies). Only for
                release_for_rust_pipeline()'s own "disconnect" submission —
                set_rust_pipeline_in_progress(True) is published before that
                call runs (to close a handoff race window), which would
                otherwise make the guard reject the very disconnect that
                releases the database to the Rust pipeline.
            **kwargs: Keyword arguments for the operation

        Returns:
            The result of the operation, fully materialized
        """

        self._raise_if_db_unavailable_before_submit(
            operation_name, _bypass_rust_pipeline_guard
        )

        def executor_operation() -> Any:
            return self._execute_operation(
                provider, operation_name, args, kwargs, _bypass_rust_pipeline_guard
            )

        # Run in executor synchronously with configured timeout
        future = self._submit(executor_operation)
        timeout_s = self.resolve_timeout(operation_name)
        try:
            result = self._wait_for_operation(
                future,
                operation_name,
                None if operation_name == "disconnect" else timeout_s,
            )
        except DatabaseOperationTimeoutError:
            logger.error(
                f"Database operation '{operation_name}' timed out after "
                f"{timeout_s} seconds"
            )
            raise

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
        self._raise_if_db_unavailable_before_submit(operation_name)

        def executor_operation() -> Any:
            return self._execute_operation(provider, operation_name, args, kwargs)

        # Capture context for async compatibility
        ctx = contextvars.copy_context()

        # Run in executor with context
        result = await asyncio.wrap_future(self._submit(ctx.run, executor_operation))

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

    def disconnect_provider(self, provider: Any, skip_checkpoint: bool) -> None:
        """Drain queued work and close provider handles before joining the worker."""
        with self._finalization_lock:
            self._disconnect_provider(provider, skip_checkpoint)

    def _disconnect_provider(self, provider: Any, skip_checkpoint: bool) -> None:
        """One finalization attempt, serialized through worker join publication."""
        with self._submission_lock:
            if self._joined:
                return
            self._accepting = False

            def finish() -> None:
                conn = getattr(_executor_local, "connection", None)
                state = get_thread_local_state()
                try:
                    provider._executor_disconnect(conn, state, skip_checkpoint)
                except BaseException as error:
                    traceback.clear_frames(error.__traceback__)
                    raise
                finally:
                    conn = None
                    self.handles_closed = not hasattr(_executor_local, "connection")
                    if self.handles_closed:
                        self.clear_thread_local()  # This is the WORKER thread.

            # Final close alone bypasses ordinary admission/compaction/Rust guards.
            if self._finalization is None or (
                self._finalization.done() and not self.handles_closed
            ):
                # Only a retained handle needs another worker attempt. Reuse a
                # completed successful future if an earlier caller was interrupted
                # before joining; never resubmit after successful finalization.
                self._finalization = self._db_executor.submit(finish)
            future = self._finalization
        try:
            future.result()  # Must follow all previously queued work; no timeout.
        finally:
            # An interrupted caller does not prove the close worker stopped.
            if future.done() and self.handles_closed:
                self._db_executor.shutdown(wait=True)
                self._joined = True
                self.finalization_complete = True

    def shutdown(self, wait: bool = True) -> None:
        """Close executor-owned handles; providers must use disconnect_provider."""
        if not wait:
            raise ValueError("Database executor shutdown requires wait=True")
        with self._finalization_lock:
            if self._joined:
                return
            with self._submission_lock:
                self._accepting = False
            try:
                self._force_close_connections()
            finally:
                # Retained handles require a usable executor for cleanup retry.
                if self.handles_closed:
                    self._db_executor.shutdown(wait=True)
                    self._joined = True
                    self.finalization_complete = True

    def _force_close_connections(self) -> None:
        """Force close thread-local handles without hiding close failures."""
        def close_connection() -> None:
            conn = getattr(_executor_local, "connection", None)
            try:
                if conn is not None and hasattr(conn, "close"):
                    conn.close()
                self.clear_thread_local()
            finally:
                conn = None
                self.handles_closed = not hasattr(_executor_local, "connection")
        self._db_executor.submit(close_connection).result()

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
            future = self._submit(get_activity_time)
            return future.result(timeout=1.0)  # Quick operation, short timeout
        except Exception:
            return None
