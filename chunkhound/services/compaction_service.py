"""Compaction service for DuckDB databases.

Provides both blocking (CLI) and background (MCP) compaction modes.
"""

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from chunkhound.core.config.config import Config
    from chunkhound.providers.database.duckdb_provider import DuckDBProvider


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

    @property
    def is_compacting(self) -> bool:
        """Check if compaction is currently in progress."""
        return self._compaction_in_progress

    def check_should_compact(
        self, provider: "DuckDBProvider"
    ) -> tuple[bool, dict[str, Any]]:
        """Check if compaction is warranted. Returns (should_compact, stats)."""
        if self._compaction_in_progress:
            return False, {}

        if not self._config.database.compaction_enabled:
            return False, {}

        threshold = self._config.database.compaction_threshold
        should, stats = provider.should_compact(threshold=threshold)

        if not should:
            return False, stats

        # Check minimum size threshold (use free_blocks only - reliable metric)
        block_size = stats.get("block_size", 262144)
        free_blocks = stats.get("free_blocks", 0)
        reclaimable_bytes = free_blocks * block_size
        min_bytes = self._config.database.compaction_min_size_mb * 1024 * 1024

        if reclaimable_bytes < min_bytes:
            logger.debug(
                f"Reclaimable space {reclaimable_bytes / 1024 / 1024:.1f}MB "
                f"below threshold {self._config.database.compaction_min_size_mb}MB"
            )
            return False, stats

        # Disk space check delegated to provider.optimize()
        return True, stats

    async def compact_blocking(self, provider: "DuckDBProvider") -> bool:
        """Perform compaction synchronously. Blocks until complete.

        Use this in CLI mode where we want to pause all operations during compaction.
        No catch-up needed since no writes occur during blocking compaction.

        Returns True if compaction succeeded, False otherwise.
        """
        should, stats = self.check_should_compact(provider)
        if not should:
            return False

        free_ratio = stats.get("free_blocks", 0) / max(stats.get("total_blocks", 1), 1)
        logger.info(
            f"Free blocks ratio {free_ratio:.0%} exceeds threshold, "
            f"starting compaction..."
        )

        self._compaction_in_progress = True
        try:
            await self._do_compaction(provider)
            return True
        except Exception as e:
            logger.error(f"Compaction failed: {e}")
            return False
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

        free_ratio = stats.get("free_blocks", 0) / max(stats.get("total_blocks", 1), 1)
        logger.info(
            f"Free blocks ratio {free_ratio:.0%} exceeds threshold, "
            f"starting background compaction..."
        )

        self._compaction_in_progress = True
        self._compaction_task = asyncio.create_task(
            self._do_compaction_with_callback(provider, on_complete)
        )
        return True

    async def _do_compaction_with_callback(
        self,
        provider: "DuckDBProvider",
        on_complete: Callable[[], Awaitable[None]] | None,
    ) -> None:
        """Wrapper that invokes callback after successful compaction."""
        try:
            await self._do_compaction(provider)

            if on_complete is not None:
                logger.info(
                    "Compaction complete, triggering post-compaction reindex..."
                )
                await on_complete()

        except asyncio.CancelledError:
            logger.info("Background compaction cancelled")
            raise
        except Exception as e:
            logger.error(f"Background compaction failed: {e}")
        finally:
            self._compaction_in_progress = False
            self._compaction_task = None

    async def _do_compaction(self, provider: "DuckDBProvider") -> None:
        """Perform compaction by delegating to provider.optimize().

        The provider handles all mechanics: lock file, EXPORT/IMPORT/SWAP, recovery.
        This service handles: shutdown coordination, async dispatch.
        """
        # Check for shutdown request before starting
        if self._shutdown_requested:
            logger.info("Compaction aborted due to shutdown request")
            return

        # Delegate actual compaction to provider (runs in thread pool)
        # Provider handles: lock file, EXPORT, IMPORT, atomic swap, recovery
        await asyncio.to_thread(provider.optimize)

        logger.info("Compaction complete")

    async def shutdown(self, timeout: float = 5.0) -> None:
        """Gracefully shutdown compaction service.

        Cancels any in-progress compaction and waits for cleanup.
        """
        self._shutdown_requested = True

        if self._compaction_task is not None and not self._compaction_task.done():
            logger.info("Cancelling in-progress compaction...")
            self._compaction_task.cancel()

            try:
                await asyncio.wait_for(
                    asyncio.shield(self._compaction_task),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning("Compaction did not stop within timeout")
            except asyncio.CancelledError:
                pass

        self._compaction_task = None
        self._compaction_in_progress = False
