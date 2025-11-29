"""Real-time indexing service for MCP servers.

This service provides continuous filesystem monitoring and incremental updates
while maintaining search responsiveness. It leverages the existing indexing
infrastructure and respects the single-threaded database constraint.

Architecture:
- Single event queue for filesystem changes
- Background scan iterator for initial indexing
- No cancellation - operations complete naturally
- SerialDatabaseProvider handles all concurrency
"""

import asyncio
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable

from loguru import logger
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from chunkhound.utils.windows_constants import IS_WINDOWS

from chunkhound.core.config.config import Config
from chunkhound.database_factory import DatabaseServices


def normalize_file_path(path: Path | str) -> str:
    """Single source of truth for path normalization across ChunkHound."""
    return str(Path(path).resolve())


class SimpleEventHandler(FileSystemEventHandler):
    """Simple sync event handler - no async complexity."""

    def __init__(
        self,
        event_queue: asyncio.Queue,
        config: Config | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        self.event_queue = event_queue
        self.config = config
        self.loop = loop
        self._engine = None
        self._include_patterns: list[str] | None = None
        self._pattern_cache: dict[str, Any] = {}
        try:
            self._root = (config.target_dir if config and config.target_dir else Path.cwd()).resolve()
        except Exception:
            self._root = Path.cwd().resolve()

    def on_any_event(self, event: Any) -> None:
        """Handle filesystem events - simple queue operation."""
        # Handle directory creation
        if event.event_type == "created" and event.is_directory:
            # Queue directory creation for processing
            self._queue_event("dir_created", Path(normalize_file_path(event.src_path)))
            return

        # Handle directory deletion
        if event.event_type == "deleted" and event.is_directory:
            # Queue directory deletion for cleanup
            self._queue_event("dir_deleted", Path(normalize_file_path(event.src_path)))
            return

        # Skip other directory events (modified, moved)
        if event.is_directory:
            return

        # Handle move events for atomic writes
        if event.event_type == "moved" and hasattr(event, "dest_path"):
            self._handle_move_event(event.src_path, event.dest_path)
            return

        # Resolve path to canonical form to avoid /var vs /private/var issues
        file_path = Path(normalize_file_path(event.src_path))

        # Simple filtering for supported file types
        if not self._should_index(file_path):
            return

        # Put event in async queue from watchdog thread
        try:
            if self.loop and not self.loop.is_closed():
                future = asyncio.run_coroutine_threadsafe(
                    self.event_queue.put((event.event_type, file_path)), self.loop
                )
                future.result(timeout=5.0)  # More tolerance for queue operations
        except Exception as e:
            logger.warning(f"Failed to queue event for {file_path}: {e}")

    def _should_index(self, file_path: Path) -> bool:
        """Check if file should be indexed based on config patterns.

        Uses config-based filtering if available, otherwise falls back to
        Language enum which derives all patterns from parser_factory.
        This ensures realtime indexing supports all languages without
        requiring manual updates.
        """
        if not self.config:
            # Fallback: derive from Language enum (which derives from parser_factory)
            # Uses lazy import to avoid heavyweight startup cost
            from chunkhound.core.types.common import Language

            # Check extension-based patterns
            if file_path.suffix.lower() in Language.get_all_extensions():
                return True

            # Check filename-based patterns (Makefile, Dockerfile, etc.)
            if file_path.name.lower() in Language.get_all_filename_patterns():
                return True

            return False

        # Repo-aware ignore engine (lazy init)
        try:
            if self._engine is None:
                from chunkhound.utils.ignore_engine import build_repo_aware_ignore_engine
                sources = self.config.indexing.resolve_ignore_sources()
                cfg_ex = self.config.indexing.get_effective_config_excludes()
                chf = self.config.indexing.chignore_file
                overlay = bool(getattr(self.config.indexing, "workspace_gitignore_nonrepo", False))
                self._engine = build_repo_aware_ignore_engine(self._root, sources, chf, cfg_ex, workspace_root_only_gitignore=overlay)
        except Exception:
            self._engine = None

        # Exclude via engine
        try:
            if self._engine is not None and self._engine.matches(file_path, is_dir=False):
                return False
        except Exception:
            pass

        # Include via normalized patterns (fallback to Language defaults)
        try:
            if self._include_patterns is None:
                from chunkhound.utils.file_patterns import normalize_include_patterns
                inc = list(self.config.indexing.include)
                self._include_patterns = normalize_include_patterns(inc)

            from chunkhound.utils.file_patterns import should_include_file
            return should_include_file(file_path, self._root, self._include_patterns, self._pattern_cache)
        except Exception:
            # Fallback to Language-based detection if include matching fails
            from chunkhound.core.types.common import Language
            if file_path.suffix.lower() in Language.get_all_extensions():
                return True
            if file_path.name.lower() in Language.get_all_filename_patterns():
                return True
            return False

    def _handle_move_event(self, src_path: str, dest_path: str) -> None:
        """Handle atomic file moves (temp -> final file)."""
        src_file = Path(normalize_file_path(src_path))
        dest_file = Path(normalize_file_path(dest_path))

        # If moving FROM temp file TO supported file -> index destination
        if not self._should_index(src_file) and self._should_index(dest_file):
            logger.debug(f"Atomic write detected: {src_path} -> {dest_path}")
            self._queue_event("created", dest_file)

        # If moving FROM supported file -> handle as deletion + creation
        elif self._should_index(src_file) and self._should_index(dest_file):
            logger.debug(f"File rename: {src_path} -> {dest_path}")
            self._queue_event("deleted", src_file)
            self._queue_event("created", dest_file)

        # If moving FROM supported file TO temp/unsupported -> deletion
        elif self._should_index(src_file) and not self._should_index(dest_file):
            logger.debug(f"File moved to temp/unsupported: {src_path}")
            self._queue_event("deleted", src_file)

    def _queue_event(self, event_type: str, file_path: Path) -> None:
        """Queue an event for async processing."""
        try:
            if self.loop and not self.loop.is_closed():
                future = asyncio.run_coroutine_threadsafe(
                    self.event_queue.put((event_type, file_path)), self.loop
                )
                future.result(timeout=5.0)  # More tolerance for queue operations
        except Exception as e:
            logger.warning(f"Failed to queue {event_type} event for {file_path}: {e}")


class RealtimeIndexingService:
    """Simple real-time indexing service with search responsiveness."""

    # Event deduplication window - suppress duplicate events within this period
    _EVENT_DEDUP_WINDOW_SECONDS = 2.0
    # Retention period for event history - entries older than this are cleaned up
    _EVENT_HISTORY_RETENTION_SECONDS = 10.0

    def __init__(
        self,
        services: DatabaseServices,
        config: Config,
        debug_sink: Callable[[str], None] | None = None,
    ):
        self.services = services
        self.config = config
        # Optional sink that writes to MCPServerBase.debug_log so events land in
        # /tmp/chunkhound_mcp_debug.log when CHUNKHOUND_DEBUG is enabled.
        self._debug_sink = debug_sink

        # Existing asyncio queue for priority processing
        self.file_queue: asyncio.Queue[tuple[str, Path]] = asyncio.Queue()

        # NEW: Async queue for events from watchdog (thread-safe via asyncio)
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)

        # Deduplication and error tracking
        self.pending_files: set[Path] = set()
        self.failed_files: set[str] = set()

        # Simple debouncing for rapid file changes
        self._pending_debounce: dict[str, float] = {}  # file_path -> timestamp
        self._debounce_delay = 0.5  # 500ms delay from research
        self._debounce_tasks: set[asyncio.Task] = set()  # Track active debounce tasks

        self._recent_file_events: dict[str, tuple[str, float]] = {}  # Layer 3: event dedup

        # Background scan state
        self.scan_iterator: Iterator | None = None
        self.scan_complete = False

        # Filesystem monitoring
        self.observer: Any | None = None
        self.event_handler: SimpleEventHandler | None = None
        self.watch_path: Path | None = None

        # Processing tasks
        self.process_task: asyncio.Task | None = None
        self.event_consumer_task: asyncio.Task | None = None
        self._polling_task: asyncio.Task | None = None

        # Directory watch management for progressive monitoring
        self.watched_directories: set[str] = set()  # Track watched dirs
        self.watch_lock = asyncio.Lock()  # Protect concurrent access

        # Monitoring readiness coordination
        self.monitoring_ready = asyncio.Event()  # Signals when monitoring is ready
        self._monitoring_ready_time: float | None = (
            None  # Track when monitoring became ready
        )

    # Internal helper to forward realtime events into the MCP debug log file
    def _debug(self, message: str) -> None:
        try:
            if self._debug_sink:
                # Prefix with RT to make it easy to filter
                self._debug_sink(f"RT: {message}")
        except Exception:
            # Never let debug plumbing affect runtime
            pass

    async def start(self, watch_path: Path) -> None:
        """Start real-time indexing service."""
        logger.debug(f"Starting real-time indexing for {watch_path}")
        self._debug(f"start watch on {watch_path}")

        # Store the watch path
        self.watch_path = watch_path

        # Always start with watchdog but with reasonable timeout
        # If it takes too long, we'll fall back to polling
        loop = asyncio.get_event_loop()

        # Start all necessary tasks
        self.event_consumer_task = asyncio.create_task(self._consume_events())
        self.process_task = asyncio.create_task(self._process_loop())

        # Setup watchdog with timeout
        self._watchdog_setup_task = asyncio.create_task(
            self._setup_watchdog_with_timeout(watch_path, loop)
        )

        # Wait for monitoring to be confirmed ready
        monitoring_ok = await self.wait_for_monitoring_ready(timeout=10.0)
        if monitoring_ok:
            self._debug("monitoring ready")
        else:
            self._debug("monitoring timeout; continuing")

    async def stop(self) -> None:
        """Stop the service gracefully."""
        logger.debug("Stopping real-time indexing service")
        self._debug("stopping service")

        # Cancel watchdog setup if still running
        if hasattr(self, "_watchdog_setup_task") and self._watchdog_setup_task:
            self._watchdog_setup_task.cancel()
            try:
                await self._watchdog_setup_task
            except asyncio.CancelledError:
                pass

        # Stop filesystem observer
        if self.observer:
            self.observer.stop()
            # Join with timeout to prevent hanging
            try:
                loop = asyncio.get_event_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(None, self.observer.join), timeout=1.0
                )
            except asyncio.TimeoutError:
                logger.warning("Observer thread did not exit within timeout")

        # Cancel event consumer task
        if self.event_consumer_task:
            self.event_consumer_task.cancel()
            try:
                await self.event_consumer_task
            except asyncio.CancelledError:
                pass

        # Cancel processing task
        if self.process_task:
            self.process_task.cancel()
            try:
                await self.process_task
            except asyncio.CancelledError:
                pass

        # Cancel polling task if running
        if self._polling_task:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass

        # Cancel all active debounce tasks
        for task in self._debounce_tasks.copy():
            task.cancel()

        # Wait for debounce tasks to finish cancelling
        if self._debounce_tasks:
            await asyncio.gather(*self._debounce_tasks, return_exceptions=True)
            self._debounce_tasks.clear()

    async def _setup_watchdog_async(
        self, watch_path: Path, loop: asyncio.AbstractEventLoop
    ) -> None:
        """Setup watchdog in background thread without blocking initialization."""
        try:
            await loop.run_in_executor(None, self._start_fs_monitor, watch_path, loop)
            logger.debug("Watchdog setup completed successfully")
        except Exception as e:
            logger.error(f"Failed to setup watchdog monitoring: {e}")
            # Server continues to work even if watchdog setup fails

    async def _setup_watchdog_with_timeout(
        self, watch_path: Path, loop: asyncio.AbstractEventLoop
    ) -> None:
        """Setup watchdog with timeout - fall back to polling if it takes too long."""
        # run_in_executor returns an awaitable Future - no create_task needed
        watchdog_task = loop.run_in_executor(
            None, self._start_fs_monitor, watch_path, loop
        )

        try:
            # Try recursive setup with reasonable timeout for large directories
            await asyncio.wait_for(watchdog_task, timeout=5.0)
            logger.debug("Watchdog setup completed successfully (recursive mode)")
            self._debug("watchdog setup complete (recursive)")
            self._monitoring_ready_time = time.time()
            self.monitoring_ready.set()  # Signal monitoring is ready

        except asyncio.TimeoutError:
            # Cancel the watchdog task before falling back to polling
            watchdog_task.cancel()
            try:
                await watchdog_task
            except asyncio.CancelledError:
                pass

            logger.info(
                f"Watchdog setup timed out for {watch_path} - falling back to polling"
            )
            self._using_polling = True
            self._polling_task = asyncio.create_task(self._polling_monitor(watch_path))
            # Wait a moment for polling to start
            await asyncio.sleep(0.5)
            self._monitoring_ready_time = time.time()
            self.monitoring_ready.set()  # Signal monitoring is ready (polling mode)
            self._debug("watchdog timed out; switched to polling")
        except Exception as e:
            # Cancel watchdog task on other errors too
            if not watchdog_task.done():
                watchdog_task.cancel()
                try:
                    await watchdog_task
                except asyncio.CancelledError:
                    pass

            logger.warning(f"Watchdog setup failed: {e} - falling back to polling")
            self._using_polling = True
            self._polling_task = asyncio.create_task(self._polling_monitor(watch_path))
            # Wait a moment for polling to start
            await asyncio.sleep(0.5)
            self._monitoring_ready_time = time.time()
            self.monitoring_ready.set()  # Signal monitoring is ready (polling mode)
            self._debug("watchdog failed; switched to polling")

    def _start_fs_monitor(
        self, watch_path: Path, loop: asyncio.AbstractEventLoop
    ) -> None:
        """Start filesystem monitoring with recursive watching for complete coverage."""
        self.event_handler = SimpleEventHandler(self.event_queue, self.config, loop)
        self.observer = Observer()

        # Use recursive=True to ensure all directory events are captured
        # This is necessary for proper real-time monitoring of new directories
        self.observer.schedule(
            self.event_handler,
            str(watch_path),
            recursive=True,  # Use recursive for complete event coverage
        )
        self.watched_directories.add(str(watch_path))
        self.observer.start()

        # Wait for observer thread to be fully running
        # On Windows, observer thread startup can be noticeably slower.
        # Give it more time to become alive to avoid falling back to polling unnecessarily.
        max_wait = 5.0 if IS_WINDOWS else 1.0
        start = time.time()
        while not self.observer.is_alive() and (time.time() - start) < max_wait:
            time.sleep(0.01)

        if self.observer.is_alive():
            logger.debug(f"Started recursive filesystem monitoring for {watch_path}")
        else:
            raise RuntimeError("Observer failed to start within timeout")

    async def _add_subdirectories_progressively(self, root_path: Path) -> None:
        """No longer needed - using recursive monitoring."""
        logger.debug(
            "Progressive directory addition skipped (using recursive monitoring)"
        )

    async def _polling_monitor(self, watch_path: Path) -> None:
        """Simple polling monitor for large directories."""
        logger.debug(f"Starting polling monitor for {watch_path}")
        self._debug(f"polling monitor active for {watch_path}")
        known_files = set()

        # Create a simple event handler for shouldIndex check once
        simple_handler = SimpleEventHandler(None, self.config, None)

        # Use a shorter interval during the first few seconds to ensure
        # freshly created files are detected quickly after startup/fallback.
        polling_start = time.time()

        while True:
            try:
                current_files = set()
                files_checked = 0

                # Walk directory tree but with limits to avoid hanging
                for file_path in watch_path.rglob("*"):
                    try:
                        if file_path.is_file():
                            files_checked += 1
                            if simple_handler._should_index(file_path):
                                current_files.add(file_path)
                                if file_path not in known_files:
                                    # New file detected
                                    logger.debug(
                                        f"Polling detected new file: {file_path}"
                                    )
                                    self._debug(
                                        f"polling detected new file: {file_path}"
                                    )
                                    await self.add_file(file_path, priority="change")

                        # Yield control periodically and limit total files checked
                        if files_checked % 100 == 0:
                            await asyncio.sleep(0)  # Yield control
                            if files_checked > 5000:  # Limit to prevent hanging
                                logger.warning(
                                    f"Polling checked {files_checked} files, skipping rest to avoid blocking"
                                )
                                break
                    except (OSError, PermissionError):
                        # Skip files we can't access
                        continue

                # Check for deleted files
                deleted = known_files - current_files
                for file_path in deleted:
                    logger.debug(f"Polling detected deleted file: {file_path}")
                    await self.remove_file(file_path)
                    self._debug(f"polling detected deleted file: {file_path}")

                known_files = current_files

                # Adaptive poll interval: 1s for the first 10s, then 5s
                elapsed = time.time() - polling_start
                interval = 1.0 if elapsed < 10.0 else 5.0
                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Polling monitor error: {e}")
                await asyncio.sleep(5)

    async def add_file(self, file_path: Path, priority: str = "change") -> None:
        """Add file to processing queue with deduplication and debouncing."""
        if file_path not in self.pending_files:
            self.pending_files.add(file_path)

            # Simple debouncing for change events
            if priority == "change":
                file_str = str(file_path)
                current_time = time.time()

                if file_str in self._pending_debounce:
                    # Update timestamp for existing pending file
                    self._pending_debounce[file_str] = current_time
                    return
                else:
                    # Schedule debounced processing
                    self._pending_debounce[file_str] = current_time
                    task = asyncio.create_task(
                        self._debounced_add_file(file_path, priority)
                    )
                    self._debounce_tasks.add(task)
                    task.add_done_callback(self._debounce_tasks.discard)
                    self._debug(f"queued (debounced) {file_path} priority={priority}")
            else:
                # Priority scan events bypass debouncing
                await self.file_queue.put((priority, file_path))
                self._debug(f"queued {file_path} priority={priority}")

    async def _debounced_add_file(self, file_path: Path, priority: str) -> None:
        """Process file after debounce delay."""
        await asyncio.sleep(self._debounce_delay)

        file_str = str(file_path)
        if file_str in self._pending_debounce:
            last_update = self._pending_debounce[file_str]

            # Check if no recent updates during delay
            if time.time() - last_update >= self._debounce_delay:
                del self._pending_debounce[file_str]
                await self.file_queue.put((priority, file_path))
                logger.debug(f"Processing debounced file: {file_path}")
                self._debug(f"processing debounced file: {file_path}")

    async def _consume_events(self) -> None:
        """Simple event consumer - pure asyncio queue."""
        while True:
            try:
                # Get event from async queue with timeout
                try:
                    event_type, file_path = await asyncio.wait_for(
                        self.event_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    # Normal timeout, continue to check if task should stop
                    continue

                # Layer 3: Event deduplication to prevent redundant processing
                # Suppress duplicate events within 2-second window (e.g., created + modified from same editor save)
                file_key = str(file_path)
                current_time = time.time()

                if file_key in self._recent_file_events:
                    last_event_type, last_event_time = self._recent_file_events[file_key]
                    if last_event_type == event_type and (current_time - last_event_time) < self._EVENT_DEDUP_WINDOW_SECONDS:
                        logger.debug(f"Suppressing duplicate {event_type} event for {file_path} (within {self._EVENT_DEDUP_WINDOW_SECONDS}s window)")
                        self._debug(f"suppressed duplicate {event_type}: {file_path}")
                        self.event_queue.task_done()
                        continue

                # Record this event
                self._recent_file_events[file_key] = (event_type, current_time)

                # Cleanup old entries to keep dict bounded (max 1000 files)
                if len(self._recent_file_events) > 1000:
                    cutoff = current_time - self._EVENT_HISTORY_RETENTION_SECONDS
                    self._recent_file_events = {
                        k: v for k, v in self._recent_file_events.items()
                        if v[1] > cutoff
                    }

                if event_type in ("created", "modified"):
                    # Use existing add_file method for deduplication and priority
                    await self.add_file(file_path, priority="change")
                    self._debug(f"event {event_type}: {file_path}")
                elif event_type == "deleted":
                    # Handle deletion immediately
                    await self.remove_file(file_path)
                    self._debug(f"event deleted: {file_path}")
                elif event_type == "dir_created":
                    # Handle new directory creation - with recursive monitoring,
                    # we don't need to add individual watches
                    # Index files in new directory
                    await self._index_directory(file_path)
                    self._debug(f"event dir_created: {file_path}")
                elif event_type == "dir_deleted":
                    # Handle directory deletion - cleanup database
                    await self._cleanup_deleted_directory(str(file_path))
                    self._debug(f"event dir_deleted: {file_path}")

                self.event_queue.task_done()

            except Exception as e:
                logger.error(f"Error consuming event: {e}")
                await asyncio.sleep(0.1)  # Brief pause on error

    async def remove_file(self, file_path: Path) -> None:
        """Remove file from database."""
        try:
            logger.debug(f"Removing file from database: {file_path}")
            self.services.provider.delete_file_completely(str(file_path))
            self._debug(f"removed file from database: {file_path}")
        except Exception as e:
            logger.error(f"Error removing file {file_path}: {e}")

    async def _add_directory_watch(self, dir_path: str) -> None:
        """Add a new directory to monitoring with recursive watching for real-time events."""
        async with self.watch_lock:
            if dir_path not in self.watched_directories:
                if self.observer and self.event_handler:
                    self.observer.schedule(
                        self.event_handler,
                        dir_path,
                        recursive=True,  # Use recursive for dynamically created directories
                    )
                    self.watched_directories.add(dir_path)
                    logger.debug(f"Added recursive watch for new directory: {dir_path}")

    async def _remove_directory_watch(self, dir_path: str) -> None:
        """Remove directory from monitoring and clean up database."""
        async with self.watch_lock:
            if dir_path in self.watched_directories:
                # Note: Watchdog auto-removes watches for deleted dirs
                self.watched_directories.discard(dir_path)

                # Clean up database entries for files in deleted directory
                await self._cleanup_deleted_directory(dir_path)
                logger.debug(f"Removed watch for deleted directory: {dir_path}")

    async def _cleanup_deleted_directory(self, dir_path: str) -> None:
        """Clean up database entries for files in a deleted directory."""
        try:
            # Get all files that were in this directory from database
            # Use the provider's search capability to find files with this path prefix
            search_results, _ = self.services.provider.search_regex(
                pattern=f"^{dir_path}/.*",
                page_size=1000,  # Large page to get all matches
            )

            # Delete each file found in the directory
            for result in search_results:
                file_path = result.get("file_path", result.get("path", ""))
                if file_path:
                    logger.debug(f"Cleaning up deleted file: {file_path}")
                    self.services.provider.delete_file_completely(file_path)

            logger.info(
                f"Cleaned up {len(search_results)} files from deleted directory: {dir_path}"
            )

        except Exception as e:
            logger.error(f"Error cleaning up deleted directory {dir_path}: {e}")

    async def _index_directory(self, dir_path: Path) -> None:
        """Index files in a newly created directory."""
        try:
            # Get all supported files in the new directory
            supported_files = []
            for file_path in dir_path.rglob("*"):
                if (
                    file_path.is_file()
                    and self.event_handler
                    and self.event_handler._should_index(file_path)
                ):
                    supported_files.append(file_path)

            # Add files to processing queue
            for file_path in supported_files:
                await self.add_file(file_path, priority="change")

            logger.debug(
                f"Queued {len(supported_files)} files from new directory: {dir_path}"
            )
            self._debug(
                f"queued {len(supported_files)} files from new directory: {dir_path}"
            )

        except Exception as e:
            logger.error(f"Error indexing new directory {dir_path}: {e}")

    async def _process_loop(self) -> None:
        """Main processing loop - simple and robust."""
        logger.debug("Starting processing loop")

        while True:
            try:
                # Wait for next file (blocks if queue is empty)
                priority, file_path = await self.file_queue.get()

                # Remove from pending set
                self.pending_files.discard(file_path)

                # Check if file still exists (prevent race condition with deletion)
                if not file_path.exists():
                    logger.debug(f"Skipping {file_path} - file no longer exists")
                    continue

                # Process the file
                logger.debug(f"Processing {file_path} (priority: {priority})")

                # Fast path for embedding pass: generate missing embeddings for all chunks
                # without re-parsing the file. Keeps the loop snappy and avoids diffing.
                if priority == "embed":
                    try:
                        await self.services.indexing_coordinator.generate_missing_embeddings()
                    except Exception as e:
                        logger.warning(f"Embedding generation failed in realtime (embed pass): {e}")
                    continue

                # Skip embeddings for initial and change events to keep loop responsive.
                # An explicit 'embed' follow-up event will generate embeddings.
                skip_embeddings = True

                # Use existing indexing coordinator
                result = await self.services.indexing_coordinator.process_file(
                    file_path, skip_embeddings=skip_embeddings
                )

                # Ensure database transaction is flushed for immediate visibility
                if hasattr(self.services.provider, "flush"):
                    await self.services.provider.flush()

                # If we skipped embeddings, queue for embedding generation
                if skip_embeddings:
                    await self.add_file(file_path, priority="embed")

                # Record processing summary into MCP debug log
                try:
                    chunks = (
                        result.get("chunks", None) if isinstance(result, dict) else None
                    )
                    embeds = (
                        result.get("embeddings", None)
                        if isinstance(result, dict)
                        else None
                    )
                    self._debug(
                        f"processed {file_path} priority={priority} "
                        f"skip_embeddings={skip_embeddings} chunks={chunks} embeddings={embeds}"
                    )
                except Exception:
                    pass

            except asyncio.CancelledError:
                logger.debug("Processing loop cancelled")
                raise
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                # Track failed files for debugging and monitoring
                self.failed_files.add(str(file_path))
                # Continue processing other files

    async def get_stats(self) -> dict:
        """Get current service statistics."""
        # Check if observer is running OR we're using polling mode
        monitoring_active = False
        if self.observer and self.observer.is_alive():
            monitoring_active = True
        elif hasattr(self, "_using_polling"):
            # If we're using polling mode, consider it "alive"
            monitoring_active = True

        return {
            "queue_size": self.file_queue.qsize(),
            "pending_files": len(self.pending_files),
            "failed_files": len(self.failed_files),
            "scan_complete": self.scan_complete,
            "observer_alive": monitoring_active,
            "watching_directory": str(self.watch_path) if self.watch_path else None,
            "watched_directories_count": len(self.watched_directories),  # Added
        }

    async def wait_for_monitoring_ready(self, timeout: float = 10.0) -> bool:
        """Wait for filesystem monitoring to be ready."""
        try:
            await asyncio.wait_for(self.monitoring_ready.wait(), timeout=timeout)
            logger.debug("Monitoring became ready after setup")
            return True
        except asyncio.TimeoutError:
            logger.warning(f"Monitoring not ready after {timeout}s")
            return False
