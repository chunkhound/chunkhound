"""Compaction service for DuckDB databases.

Provides both blocking (CLI) and background (MCP) compaction modes.
"""

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from chunkhound.core.config.config import Config
    from chunkhound.providers.database.duckdb_provider import DuckDBProvider


def estimate_reclaimable_bytes(stats: dict[str, Any]) -> int:
    """Upper-bound estimate of reclaimable bytes from storage stats.

    Takes the larger of free-block space and row-waste projection.
    Overestimates slightly because used_blocks includes index/metadata
    blocks — acceptable since this feeds a min-size gate where false
    positives just trigger a compaction that finds little to reclaim.
    """
    block_size = stats.get("block_size", 262144)
    free_blocks = stats.get("free_blocks", 0)
    used_blocks = stats.get("used_blocks", 0)
    row_waste = stats.get("row_waste_ratio", 0.0)
    return max(
        free_blocks * block_size,
        int(row_waste * used_blocks * block_size),
    )


class CompactionService:
    """Performs database compaction via EXPORT/IMPORT cycle.

    Modes:
    - Blocking: For CLI usage. Caller waits until compaction completes.
    - Background: For MCP usage. Returns immediately, triggers reindex callback.
    """

    def __init__(self, db_path: Path, config: "Config"):
        self._db_path = db_path
        self._config = config
        self._compaction_in_progress = False
        self._compaction_task: asyncio.Task[None] | None = None
        self._shutdown_requested = False
        # Tracks whether the blocking thread spawned by asyncio.to_thread has
        # truly exited.  asyncio task cancellation does NOT interrupt threads —
        # the thread runs to completion regardless.  Callers (e.g. cleanup())
        # must wait on this event before tearing down the provider.
        self._compaction_thread_done = threading.Event()
        self._compaction_thread_done.set()  # No thread running initially
        self._last_error: Exception | None = None

    @property
    def is_compacting(self) -> bool:
        """Check if compaction is currently in progress."""
        return self._compaction_in_progress

    @property
    def compaction_thread_done(self) -> threading.Event:
        """Event that is set when the compaction thread has exited."""
        return self._compaction_thread_done

    @property
    def last_error(self) -> Exception | None:
        """Last error from a compaction attempt, or None if last attempt succeeded."""
        return self._last_error

    def check_should_compact(
        self, provider: "DuckDBProvider"
    ) -> tuple[bool, dict[str, Any]]:
        """Check if compaction is warranted. Returns (should_compact, stats)."""
        # check-then-set on _compaction_in_progress is safe: CLI uses
        # compact_blocking (single caller) and MCP calls compact_background
        # once from the post-index hook — no concurrent callers in practice.
        if self._compaction_in_progress:
            # Self-heal after task cancellation: if the thread finished but
            # the asyncio task was already cancelled (so its finally block
            # couldn't reset the flag), clear it now.
            if self._compaction_thread_done.is_set() and self._compaction_task is None:
                self._compaction_in_progress = False
            else:
                return False, {}

        if not self._config.database.compaction_enabled:
            return False, {}

        threshold = self._config.database.compaction_threshold
        should, stats = provider.should_compact(threshold=threshold)

        if not should:
            return False, {}

        # Two gates: (1) ratio threshold above decides IF compaction is needed,
        # (2) absolute size gate below avoids compacting tiny databases where
        # the overhead isn't worth the reclaimed space.
        reclaimable_bytes = estimate_reclaimable_bytes(stats)
        min_bytes = self._config.database.compaction_min_size_mb * 1024 * 1024

        if reclaimable_bytes < min_bytes:
            logger.debug(
                f"Reclaimable space {reclaimable_bytes / 1024 / 1024:.1f}MB "
                f"below threshold {self._config.database.compaction_min_size_mb}MB"
            )
            return False, {}

        # Disk space check delegated to provider.optimize()
        return True, stats

    def _log_compaction_trigger(
        self, stats: dict[str, Any], *, background: bool = False
    ) -> None:
        mode = "background compaction" if background else "compaction"
        logger.info(
            f"Storage waste {stats.get('effective_waste', 0.0):.0%} exceeds threshold "
            f"(row_waste={stats.get('row_waste_ratio', 0.0):.0%}, "
            f"free_ratio={stats.get('free_ratio', 0.0):.0%}), "
            f"starting {mode}..."
        )

    async def compact_blocking(self, provider: "DuckDBProvider") -> bool:
        """Perform compaction synchronously. Blocks until complete.

        Use this in CLI mode where we want to pause all operations during compaction.
        No catch-up needed since no writes occur during blocking compaction.

        Returns True if compaction was performed, False if skipped.

        Raises:
            CompactionError: If compaction fails (propagated from provider).
        """
        should, stats = self.check_should_compact(provider)
        if not should:
            return False

        self._log_compaction_trigger(stats)

        self._compaction_in_progress = True
        try:
            return await self._do_compaction(provider)
        finally:
            self._compaction_in_progress = False

    async def compact_background(
        self,
        provider: "DuckDBProvider",
        on_complete: Callable[[], Awaitable[None]] | None = None,
    ) -> bool:
        """Start compaction in background. Returns immediately.

        Use this in MCP mode. After swap completes, calls on_complete callback
        which should trigger a background incremental index pass.

        Args:
            provider: The DuckDB provider to compact
            on_complete: Async callback invoked after successful compaction.
                        Use this to trigger post-compaction reindexing.

        Returns True if compaction was started, False if skipped.
        """
        should, stats = self.check_should_compact(provider)
        if not should:
            return False

        self._log_compaction_trigger(stats, background=True)

        self._compaction_in_progress = True
        try:
            self._compaction_task = asyncio.create_task(
                self._do_compaction_with_callback(provider, on_complete)
            )
        except Exception:
            self._compaction_in_progress = False
            raise
        return True

    async def _do_compaction_with_callback(
        self,
        provider: "DuckDBProvider",
        on_complete: Callable[[], Awaitable[None]] | None,
    ) -> None:
        """Wrapper that invokes callback after successful compaction."""
        try:
            self._last_error = None
            result = await self._do_compaction(provider)

            if result and on_complete is not None:
                try:
                    logger.info(
                        "Compaction complete, triggering post-compaction reindex..."
                    )
                    await on_complete()
                except Exception as cb_err:
                    logger.error(f"Post-compaction callback failed: {cb_err}")

        except asyncio.CancelledError:
            logger.info("Background compaction cancelled")
            raise
        except Exception as e:
            self._last_error = e
            logger.error(f"Background compaction failed: {e}")
        finally:
            # Thread may outlive a cancelled asyncio task — only reset flag once thread exits.
            if self._compaction_thread_done.is_set():
                self._compaction_in_progress = False
            self._compaction_task = None

    async def _do_compaction(self, provider: "DuckDBProvider") -> bool:
        """Perform compaction by delegating to provider.optimize().

        The provider handles all mechanics: lock file, EXPORT/IMPORT/SWAP, recovery.
        This service handles: shutdown coordination, async dispatch, cancel propagation.

        Returns True if compaction completed, False if cancelled.

        Raises:
            CompactionError: If provider.optimize() fails (propagated to caller).
        """
        # Check for shutdown request before starting
        if self._shutdown_requested:
            logger.info("Compaction aborted due to shutdown request")
            return False

        self._compaction_thread_done.clear()

        def _run_in_thread() -> bool:
            """Run provider.optimize() and signal completion via threading.Event.

            asyncio.to_thread cancellation does NOT interrupt threads — the
            EXPORT/IMPORT/SWAP operations run to completion.  The event is set
            in ``finally`` so callers waiting on it are unblocked regardless of
            success, failure, or asyncio-level cancellation.
            """
            try:
                return provider.optimize(
                    cancel_check=lambda: self._shutdown_requested,
                )
            finally:
                self._compaction_thread_done.set()

        result = await asyncio.to_thread(_run_in_thread)

        if result:
            logger.info("Compaction complete")
        return result

    async def shutdown(self, timeout: float = 30.0) -> None:
        """Gracefully shutdown compaction service.

        Cancels any in-progress compaction and waits for the underlying thread
        to finish.  asyncio.to_thread cancellation does NOT interrupt a running
        EXPORT or IMPORT — those are blocking calls inside the thread.
        cancel_check only fires between steps.  We therefore also wait on
        ``_compaction_thread_done`` (a threading.Event set by the thread's
        ``finally`` block) to ensure the thread has truly exited before the
        caller tears down the database provider.
        """
        self._shutdown_requested = True
        deadline = time.monotonic() + timeout

        if self._compaction_task is not None and not self._compaction_task.done():
            logger.info("Cancelling in-progress compaction...")
            self._compaction_task.cancel()

            remaining = max(0, deadline - time.monotonic())
            try:
                await asyncio.wait_for(self._compaction_task, timeout=remaining)
            except asyncio.TimeoutError:
                logger.warning("Compaction did not stop within timeout")
            except asyncio.CancelledError:
                pass

        # Wait for the actual compaction thread to exit.  The asyncio task may
        # be "done" (cancelled) while the thread is still running EXPORT/IMPORT.
        # Use asyncio.to_thread to avoid blocking the event loop.
        if not self._compaction_thread_done.is_set():
            remaining = max(0, deadline - time.monotonic())
            logger.info("Waiting for compaction thread to finish...")
            done = await asyncio.to_thread(
                self._compaction_thread_done.wait, timeout=remaining
            )
            if not done:
                logger.warning(
                    "Compaction thread still running after shutdown timeout"
                )

        self._compaction_task = None
        if self._compaction_thread_done.is_set():
            self._compaction_in_progress = False
        else:
            logger.error(
                "Compaction thread still running after shutdown — "
                "leaving _compaction_in_progress=True to prevent new compaction"
            )
