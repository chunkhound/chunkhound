"""Background shard rebuild coordinator for non-blocking maintenance.

# FILE_CONTEXT: Coordinates background shard rebuilds with at-most-one guarantee
# CRITICAL: Non-blocking for callers - returns immediately
# CONSTRAINT: Initialization is blocking to ensure clean state on startup
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from chunkhound.providers.database.shard_manager import ShardManager


class RebuildState(Enum):
    """State of the background rebuild coordinator."""

    IDLE = auto()
    RUNNING = auto()
    QUEUED = auto()


class BackgroundRebuildCoordinator:
    """Coordinates background shard rebuilds with at-most-one guarantee.

    Key invariants:
    - At most ONE background rebuild runs at a time (mutex)
    - Non-blocking for callers - returns immediately
    - Initialization is blocking - ensures clean state on startup
    - Idempotent - queued requests coalesce
    """

    def __init__(
        self,
        shard_manager: ShardManager,
        db_provider: Any,
    ) -> None:
        """Initialize the background rebuild coordinator.

        Args:
            shard_manager: ShardManager instance for rebuild operations
            db_provider: Database provider with _execute_in_db_thread_sync method
        """
        self._shard_manager = shard_manager
        self._db_provider = db_provider
        self._rebuild_lock = threading.Lock()
        self._state = RebuildState.IDLE
        self._state_lock = threading.Lock()  # Protects _state and _queued_check_quality
        self._queued_check_quality = False
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="bg-rebuild"
        )
        self._current_future: Future[None] | None = None

    def request_rebuild(
        self, check_quality: bool = False, blocking: bool = False
    ) -> bool:
        """Request a rebuild. Returns immediately unless blocking=True.

        Args:
            check_quality: If True, measure self-recall for each shard
            blocking: If True, wait for rebuild to complete (used for initialization)

        Returns:
            True if rebuild was started or queued successfully
        """
        if blocking:
            return self._run_synchronous(check_quality)

        # Try to acquire lock without blocking
        if self._rebuild_lock.acquire(blocking=False):
            try:
                with self._state_lock:
                    self._state = RebuildState.RUNNING
                self._current_future = self._executor.submit(
                    self._do_rebuild, check_quality
                )
                self._current_future.add_done_callback(self._on_complete)
                logger.debug("Background rebuild started")
                return True
            except Exception:
                self._rebuild_lock.release()
                raise
        else:
            # Already running - queue request
            with self._state_lock:
                if check_quality:
                    self._queued_check_quality = True
                self._state = RebuildState.QUEUED
            logger.debug("Rebuild queued - another in progress")
            return True

    def _run_synchronous(self, check_quality: bool) -> bool:
        """Blocking rebuild for initialization.

        Args:
            check_quality: If True, measure self-recall for each shard

        Returns:
            True if rebuild completed successfully
        """
        with self._rebuild_lock:
            with self._state_lock:
                self._state = RebuildState.RUNNING
            try:
                self._do_rebuild(check_quality)
                return True
            finally:
                with self._state_lock:
                    self._state = RebuildState.IDLE

    def _do_rebuild(self, check_quality: bool) -> None:
        """Execute rebuild via database executor.

        Args:
            check_quality: If True, measure self-recall for each shard
        """
        self._db_provider._execute_in_db_thread_sync("run_fix_pass", check_quality)

    def _on_complete(self, future: Future[None]) -> None:
        """Handle completion, process queue.

        Args:
            future: Completed future from the executor
        """
        try:
            future.result()
            logger.debug("Background rebuild completed successfully")
        except Exception as e:
            logger.error(f"Background rebuild failed: {e}")
        finally:
            # Check for queued request before releasing lock
            with self._state_lock:
                queued = self._state == RebuildState.QUEUED
                quality = self._queued_check_quality
                self._queued_check_quality = False
                self._state = RebuildState.IDLE

            self._rebuild_lock.release()

            if queued:
                logger.debug("Processing queued rebuild")
                self.request_rebuild(check_quality=quality)

    def is_running(self) -> bool:
        """Check if a rebuild is currently running.

        Returns:
            True if a rebuild is in progress
        """
        with self._state_lock:
            return self._state == RebuildState.RUNNING

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the background executor.

        Args:
            wait: If True, wait for pending rebuilds to complete
        """
        self._executor.shutdown(wait=wait)
        logger.debug("Background rebuild coordinator shutdown")
