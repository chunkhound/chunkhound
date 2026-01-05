"""Watcher manager for multi-repository file monitoring.

This service manages file watchers for all registered projects in global database mode.
It provides centralized file monitoring with:
- One watchdog Observer per project
- Debounced event processing
- Direct indexing (no queue needed - daemon owns write lock)
- Coordinated status updates with ProjectRegistry

The WatcherManager is owned by the HTTP server daemon and handles all file
change detection across indexed projects.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
from watchdog.events import (
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

if TYPE_CHECKING:
    from chunkhound.core.config.config import Config
    from chunkhound.services.indexing_coordinator import IndexingCoordinator
    from chunkhound.services.project_registry import ProjectInfo, ProjectRegistry


class ProjectEventHandler(FileSystemEventHandler):
    """Event handler for a single project's file changes.

    Collects events and forwards them to the central WatcherManager
    for debounced processing.
    """

    def __init__(
        self,
        project_path: Path,
        callback: Callable[[Path, str], None],
        should_index: Callable[[Path], bool],
    ):
        """Initialize event handler.

        Args:
            project_path: Root path of the project being watched
            callback: Function to call with (file_path, event_type)
            should_index: Function to check if a file should be indexed
        """
        super().__init__()
        self.project_path = project_path
        self.callback = callback
        self.should_index = should_index

    def _normalize_path(self, path: str) -> Path:
        """Normalize path to canonical form."""
        return Path(path).resolve()

    def on_created(self, event: FileCreatedEvent) -> None:
        """Handle file creation."""
        if event.is_directory:
            return
        file_path = self._normalize_path(event.src_path)
        if self.should_index(file_path):
            self.callback(file_path, "created")

    def on_modified(self, event: FileModifiedEvent) -> None:
        """Handle file modification."""
        if event.is_directory:
            return
        file_path = self._normalize_path(event.src_path)
        if self.should_index(file_path):
            self.callback(file_path, "modified")

    def on_deleted(self, event: FileDeletedEvent) -> None:
        """Handle file deletion."""
        if event.is_directory:
            return
        file_path = self._normalize_path(event.src_path)
        # Always callback for deletions - file may have been indexed
        self.callback(file_path, "deleted")

    def on_moved(self, event: FileMovedEvent) -> None:
        """Handle file move/rename."""
        if event.is_directory:
            return

        src_path = self._normalize_path(event.src_path)
        dest_path = self._normalize_path(event.dest_path)

        src_indexable = self.should_index(src_path)
        dest_indexable = self.should_index(dest_path)

        # Handle different move scenarios
        if src_indexable and dest_indexable:
            # Both indexable: treat as delete + create
            self.callback(src_path, "deleted")
            self.callback(dest_path, "created")
        elif src_indexable:
            # Moving away from indexable: delete
            self.callback(src_path, "deleted")
        elif dest_indexable:
            # Moving to indexable: create (atomic write pattern)
            self.callback(dest_path, "created")


class ProjectWatcher:
    """Manages a single project's file watcher."""

    def __init__(
        self,
        project_path: Path,
        observer: Observer,
        handler: ProjectEventHandler,
        watch_id: Any,
    ):
        """Initialize project watcher.

        Args:
            project_path: Root path of the project
            observer: Watchdog Observer instance
            handler: Event handler for this project
            watch_id: Watch identifier from observer.schedule()
        """
        self.project_path = project_path
        self.observer = observer
        self.handler = handler
        self.watch_id = watch_id
        self.started_at = time.time()
        self.events_processed = 0
        self.last_event_at: float | None = None


class WatcherManager:
    """Manages file watchers for all registered projects.

    The WatcherManager is responsible for:
    - Starting/stopping file watchers per project
    - Debouncing file change events
    - Triggering re-indexing through IndexingCoordinator
    - Tracking watcher health and statistics

    Thread-safe: Uses locks for concurrent access from watchdog threads.

    Usage:
        manager = WatcherManager(indexing_coordinator)
        manager.set_project_registry(registry)

        # Start watcher for a project
        manager.start_watcher(project_info)

        # Stop specific watcher
        manager.stop_watcher(project_path)

        # Stop all watchers
        manager.stop_all()
    """

    # Default values (used when no config provided)
    DEFAULT_DEBOUNCE_DELAY = 2.0
    DEFAULT_MAX_PENDING_PER_PROJECT = 500
    DEFAULT_MAX_PENDING_EVENTS = 10000

    def __init__(
        self,
        indexing_coordinator: IndexingCoordinator,
        debug_sink: Callable[[str], None] | None = None,
        config: Config | None = None,
    ):
        """Initialize watcher manager.

        Args:
            indexing_coordinator: Coordinator for processing file changes
            debug_sink: Optional function for debug logging
            config: Optional Config object for loading settings
        """
        self._coordinator = indexing_coordinator
        self._debug_sink = debug_sink

        # Load settings from config or use defaults
        if config is not None:
            mr = config.database.multi_repo
            self._debounce_delay = mr.watcher_debounce_seconds
            self._max_pending_events = mr.watcher_max_pending_events
            self._max_pending_per_project = mr.watcher_max_pending_per_project
        else:
            self._debounce_delay = self.DEFAULT_DEBOUNCE_DELAY
            self._max_pending_events = self.DEFAULT_MAX_PENDING_EVENTS
            self._max_pending_per_project = self.DEFAULT_MAX_PENDING_PER_PROJECT

        # Project watchers: project_path -> ProjectWatcher
        self._watchers: dict[str, ProjectWatcher] = {}
        self._watchers_lock = threading.RLock()

        # Pending events: file_path -> (event_type, first_seen_time)
        # Uses first_seen for debouncing (not last_modified)
        self._pending_events: dict[str, tuple[str, float]] = {}
        self._pending_lock = threading.RLock()

        # Project registry reference (set later to avoid circular import)
        self._registry: ProjectRegistry | None = None

        # Event loop for async operations
        self._loop: asyncio.AbstractEventLoop | None = None

        # Background processor task
        self._processor_task: asyncio.Task | None = None
        self._health_check_task: asyncio.Task | None = None
        self._running = False

        # Health check interval (seconds)
        self._health_check_interval = 60.0

    def set_project_registry(self, registry: ProjectRegistry) -> None:
        """Set the project registry for status coordination.

        Args:
            registry: ProjectRegistry instance
        """
        self._registry = registry

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set the event loop for async operations.

        Args:
            loop: asyncio event loop
        """
        self._loop = loop

    async def start(self) -> None:
        """Start the watcher manager background processor."""
        self._running = True
        self._loop = asyncio.get_event_loop()
        self._processor_task = asyncio.create_task(self._process_pending_events())
        self._health_check_task = asyncio.create_task(self._health_check_watchers())
        self._debug("WatcherManager started")

    async def stop(self) -> None:
        """Stop the watcher manager and all watchers."""
        self._running = False

        # Cancel processor task
        if self._processor_task and not self._processor_task.done():
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

        # Cancel health check task
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Stop all watchers
        self.stop_all()

        self._debug("WatcherManager stopped")

    def _debug(self, message: str) -> None:
        """Log debug message."""
        try:
            if self._debug_sink:
                self._debug_sink(f"WM: {message}")
            else:
                logger.debug(f"WatcherManager: {message}")
        except Exception:
            pass

    def _should_index_file(self, file_path: Path) -> bool:
        """Check if a file should be indexed.

        Uses Language enum to determine supported file types.

        Args:
            file_path: Path to check

        Returns:
            True if file should be indexed
        """
        # Skip hidden files and directories
        if any(part.startswith(".") for part in file_path.parts):
            # But allow specific dotfiles
            allowed_dotfiles = {".gitignore", ".env.example"}
            if file_path.name not in allowed_dotfiles:
                return False

        # Skip common non-code directories
        skip_dirs = {
            "node_modules",
            "__pycache__",
            ".git",
            ".hg",
            ".svn",
            "venv",
            ".venv",
            "dist",
            "build",
            ".chunkhound",
        }
        if any(part in skip_dirs for part in file_path.parts):
            return False

        # Use Language enum for supported extensions
        try:
            from chunkhound.core.types.common import Language

            # Check extension
            if file_path.suffix.lower() in Language.get_all_extensions():
                return True

            # Check filename patterns (Makefile, Dockerfile, etc.)
            if file_path.name.lower() in Language.get_all_filename_patterns():
                return True

        except ImportError:
            # Fallback to common extensions
            common_extensions = {
                ".py",
                ".js",
                ".ts",
                ".tsx",
                ".jsx",
                ".go",
                ".rs",
                ".java",
                ".c",
                ".cpp",
                ".h",
                ".hpp",
                ".rb",
                ".php",
                ".swift",
                ".kt",
                ".scala",
                ".cs",
                ".md",
                ".json",
                ".yaml",
                ".yml",
                ".toml",
            }
            if file_path.suffix.lower() in common_extensions:
                return True

        return False

    def _on_file_event(self, file_path: Path, event_type: str) -> None:
        """Handle a file event from watchdog.

        Called from watchdog thread - must be thread-safe.

        Args:
            file_path: Path that changed
            event_type: Type of change (created, modified, deleted)
        """
        file_key = str(file_path)
        current_time = time.time()

        # Lookup project info BEFORE acquiring lock to prevent deadlock.
        # Lock ordering: registry lock -> pending_lock (never reverse).
        # Even if project is unregistered between lookup and lock acquisition,
        # we only use immutable fields (base_directory, project_name strings).
        # Worst case: events queued for a deleted project are harmlessly processed.
        project_info = self._find_project_for_path(file_path)
        project_base = str(project_info.base_directory) if project_info else None
        project_name = project_info.project_name if project_info else None

        with self._pending_lock:
            # Only set timestamp if NOT already pending (preserves first_seen)
            if file_key not in self._pending_events:
                # Check per-project limit first
                if project_base:
                    project_event_count = sum(
                        1 for k in self._pending_events if k.startswith(project_base)
                    )
                    if project_event_count >= self._max_pending_per_project:
                        # Force-process oldest events for this project
                        project_events = sorted(
                            [
                                (k, v)
                                for k, v in self._pending_events.items()
                                if k.startswith(project_base)
                            ],
                            key=lambda x: x[1][1],  # Sort by first_seen time
                        )[:50]  # Process oldest 50

                        events_to_process = [
                            (Path(k), et) for k, (et, _) in project_events
                        ]
                        for k, _ in project_events:
                            del self._pending_events[k]

                        evt_count = len(events_to_process)
                        self._debug(
                            f"Per-project limit ({self._max_pending_per_project}) "
                            f"reached for {project_name}, "
                            f"force-processing {evt_count} events"
                        )
                        logger.warning(
                            f"WatcherManager: Per-project pending limit reached "
                            f"for {project_name}, "
                            f"force-processing {evt_count} events"
                        )

                        if self._loop:
                            for fp, et in events_to_process:
                                asyncio.run_coroutine_threadsafe(
                                    self._process_file_event(fp, et),
                                    self._loop,
                                )

                # Check global limit
                if len(self._pending_events) >= self._max_pending_events:
                    # Force-process oldest events to make room
                    # Sort by first_seen time to get oldest events first
                    oldest_keys = sorted(
                        self._pending_events.keys(),
                        key=lambda k: self._pending_events[k][1],
                    )[:100]

                    events_to_process = [
                        (Path(k), self._pending_events[k][0]) for k in oldest_keys
                    ]
                    for k in oldest_keys:
                        del self._pending_events[k]

                    self._debug(
                        f"max_pending_events ({self._max_pending_events}) reached, "
                        f"force-processing {len(events_to_process)} oldest events"
                    )
                    logger.warning(
                        f"WatcherManager: Pending events limit reached, "
                        f"force-processing {len(events_to_process)} oldest events"
                    )

                    # Schedule async processing of overflow events
                    if self._loop:
                        for fp, et in events_to_process:
                            asyncio.run_coroutine_threadsafe(
                                self._process_file_event(fp, et),
                                self._loop,
                            )

                self._pending_events[file_key] = (event_type, current_time)
                self._debug(f"Queued {event_type}: {file_path}")
            else:
                # Update event type but keep first_seen time
                _, first_seen = self._pending_events[file_key]
                self._pending_events[file_key] = (event_type, first_seen)

    # Threshold for switching to batch mode (files per project)
    BATCH_THRESHOLD = 50

    # Maximum events to collect per processing cycle (memory bound)
    MAX_READY_EVENTS = 1000

    async def _process_pending_events(self) -> None:
        """Background task to process debounced events.

        Runs continuously, checking for events that have been stable
        for the configured debounce delay (default 2.0 seconds).

        Uses batch processing when many files in the same project change
        (e.g., git checkout) to improve efficiency.
        """
        self._debug("Event processor started")

        while self._running:
            try:
                await asyncio.sleep(0.5)  # Check every 500ms

                # Collect events ready for processing
                ready_events: list[tuple[Path, str]] = []
                current_time = time.time()

                with self._pending_lock:
                    keys_to_remove = []

                    for file_key, (
                        event_type,
                        first_seen,
                    ) in self._pending_events.items():
                        # Check if event has been stable for debounce period
                        if current_time - first_seen >= self._debounce_delay:
                            ready_events.append((Path(file_key), event_type))
                            keys_to_remove.append(file_key)
                            # Bound memory usage per cycle
                            if len(ready_events) >= self.MAX_READY_EVENTS:
                                break

                    # Remove processed events
                    for key in keys_to_remove:
                        del self._pending_events[key]

                if not ready_events:
                    continue

                # Group events by project for potential batch processing
                project_events: dict[str, list[tuple[Path, str]]] = defaultdict(list)
                orphan_events: list[tuple[Path, str]] = []

                for file_path, event_type in ready_events:
                    project = self._find_project_for_path(file_path)
                    if project:
                        project_key = str(project.base_directory)
                        project_events[project_key].append((file_path, event_type))
                    else:
                        orphan_events.append((file_path, event_type))

                # Process each project's events
                for project_key, events in project_events.items():
                    # Check if we should use batch mode
                    non_delete_count = sum(1 for _, et in events if et != "deleted")

                    if non_delete_count >= self.BATCH_THRESHOLD:
                        # Batch mode: re-index entire project incrementally
                        self._debug(
                            f"Batch processing {len(events)} events "
                            f"for project at {project_key}"
                        )
                        await self._batch_process_project(Path(project_key), events)
                    else:
                        # Individual mode: process each file
                        for file_path, event_type in events:
                            await self._process_file_event(file_path, event_type)

                # Process orphan events individually
                for file_path, event_type in orphan_events:
                    await self._process_file_event(file_path, event_type)

            except asyncio.CancelledError:
                self._debug("Event processor cancelled")
                break
            except Exception as e:
                self._debug(f"Error in event processor: {e}")
                logger.exception("Error in WatcherManager event processor")
                await asyncio.sleep(5.0)  # Back off on error

        self._debug("Event processor stopped")

    async def _batch_process_project(
        self,
        project_path: Path,
        events: list[tuple[Path, str]],
    ) -> None:
        """Batch process multiple file events for a single project.

        When many files change at once (e.g., git checkout, branch switch),
        it's more efficient to do an incremental re-index of the project
        rather than processing each file individually.

        Args:
            project_path: Project base directory
            events: List of (file_path, event_type) tuples
        """
        try:
            # First handle deletions individually (can't batch these)
            delete_events = [(fp, et) for fp, et in events if et == "deleted"]
            for file_path, event_type in delete_events:
                await self._process_file_event(file_path, event_type)

            # Then do incremental index for creates/modifies
            non_delete_count = len(events) - len(delete_events)
            if non_delete_count > 0:
                self._debug(
                    f"Running incremental index for {non_delete_count} files "
                    f"at {project_path}"
                )

                # Use coordinator's incremental indexing if available
                if hasattr(self._coordinator, "index_directory"):
                    await self._coordinator.index_directory(
                        project_path,
                        skip_embeddings=False,
                    )
                    self._debug(f"Batch index complete for {project_path}")
                else:
                    # Fallback to individual processing
                    create_modify_events = [
                        (fp, et) for fp, et in events if et != "deleted"
                    ]
                    for file_path, event_type in create_modify_events:
                        await self._process_file_event(file_path, event_type)

            # Update watcher stats
            project = self._find_project_for_path(project_path)
            if project:
                with self._watchers_lock:
                    watcher = self._watchers.get(str(project.base_directory))
                    if watcher:
                        watcher.events_processed += len(events)
                        watcher.last_event_at = time.time()

        except Exception as e:
            self._debug(f"Error in batch processing for {project_path}: {e}")
            logger.exception(f"Failed to batch process project: {project_path}")
            # Fallback to individual processing on error
            for file_path, event_type in events:
                try:
                    await self._process_file_event(file_path, event_type)
                except Exception:
                    pass

    async def _process_file_event(self, file_path: Path, event_type: str) -> None:
        """Process a single file event.

        Args:
            file_path: Path that changed
            event_type: Type of change
        """
        try:
            self._debug(f"Processing {event_type}: {file_path}")

            # Find the project to get base_directory for correct path resolution
            project = self._find_project_for_path(file_path)
            project_base = project.base_directory if project else None

            if event_type == "deleted":
                # Remove file from index
                if hasattr(self._coordinator, "delete_file"):
                    await self._coordinator.delete_file(file_path)
                elif hasattr(self._coordinator._db, "delete_file_completely"):
                    self._coordinator._db.delete_file_completely(str(file_path))
                self._debug(f"Deleted from index: {file_path}")
            else:
                # Index or re-index file
                if file_path.exists():
                    await self._coordinator.process_file(
                        file_path,
                        skip_embeddings=False,
                        base_directory=project_base,
                    )
                    self._debug(f"Indexed: {file_path}")
                else:
                    self._debug(f"File no longer exists, skipping: {file_path}")

            # Update watcher stats (reuse project from earlier lookup)
            if project:
                with self._watchers_lock:
                    watcher = self._watchers.get(str(project.base_directory))
                    if watcher:
                        watcher.events_processed += 1
                        watcher.last_event_at = time.time()

        except Exception as e:
            self._debug(f"Error processing {file_path}: {e}")
            logger.exception(f"Failed to process file event: {file_path}")

    def _find_project_for_path(self, path: Path) -> ProjectInfo | None:
        """Find which project contains the given path."""
        if self._registry:
            return self._registry.find_project_for_path(path)
        return None

    async def _health_check_watchers(self) -> None:
        """Background task to check watcher health and restart dead watchers.

        Runs periodically (default every 60 seconds) to detect and recover
        from watcher failures. If a watcher's observer thread has died,
        it will be stopped and restarted.
        """
        self._debug("Watcher health check started")

        while self._running:
            try:
                await asyncio.sleep(self._health_check_interval)

                if not self._running:
                    break

                # Check all watchers for dead observer threads
                dead_watchers: list[tuple[str, str]] = []

                with self._watchers_lock:
                    for project_key, watcher in self._watchers.items():
                        if not watcher.observer.is_alive():
                            dead_watchers.append(
                                (project_key, watcher.project_path.name)
                            )

                # Restart dead watchers (outside lock to avoid deadlock)
                for project_key, project_name in dead_watchers:
                    self._debug(
                        f"Detected dead watcher for {project_name}, restarting..."
                    )
                    logger.warning(
                        f"WatcherManager: Watcher for {project_name} died, restarting"
                    )

                    # Stop the dead watcher
                    self.stop_watcher(Path(project_key))

                    # Try to restart if registry is available
                    if self._registry:
                        project = self._registry.get_project(project_key)
                        if project:
                            success = self.start_watcher(project)
                            if success:
                                self._debug(
                                    f"Successfully restarted watcher for {project_name}"
                                )
                                logger.info(
                                    f"WatcherManager: Restarted watcher "
                                    f"for {project_name}"
                                )
                            else:
                                self._debug(
                                    f"Failed to restart watcher for {project_name}"
                                )
                                logger.error(
                                    f"WatcherManager: Failed to restart "
                                    f"watcher for {project_name}"
                                )

            except asyncio.CancelledError:
                self._debug("Watcher health check cancelled")
                break
            except Exception as e:
                self._debug(f"Error in watcher health check: {e}")
                logger.exception("Error in WatcherManager health check")
                await asyncio.sleep(10.0)  # Back off on error

        self._debug("Watcher health check stopped")

    def start_watcher(self, project: ProjectInfo) -> bool:
        """Start file watcher for a project.

        Args:
            project: ProjectInfo for the project to watch

        Returns:
            True if watcher started successfully
        """
        project_key = str(project.base_directory)

        with self._watchers_lock:
            # Check if already watching
            if project_key in self._watchers:
                self._debug(f"Already watching: {project.project_name}")
                return True

            try:
                # Create observer and handler
                observer = Observer()
                handler = ProjectEventHandler(
                    project_path=project.base_directory,
                    callback=self._on_file_event,
                    should_index=self._should_index_file,
                )

                # Schedule recursive watch
                watch_id = observer.schedule(
                    handler,
                    str(project.base_directory),
                    recursive=True,
                )

                # Start observer
                observer.start()

                # Store watcher
                watcher = ProjectWatcher(
                    project_path=project.base_directory,
                    observer=observer,
                    handler=handler,
                    watch_id=watch_id,
                )
                self._watchers[project_key] = watcher

                self._debug(f"Started watcher for: {project.project_name}")

                # Update registry status
                if self._registry:
                    self._registry.set_watcher_status(project_key, active=True)

                return True

            except Exception as e:
                self._debug(f"Failed to start watcher for {project.project_name}: {e}")
                logger.exception(f"Failed to start watcher: {project.project_name}")

                # Update registry with error
                if self._registry:
                    self._registry.set_watcher_status(
                        project_key, active=False, error=str(e)
                    )

                return False

    def stop_watcher(self, project_path: Path) -> bool:
        """Stop file watcher for a project.

        Args:
            project_path: Root path of the project

        Returns:
            True if watcher was stopped
        """
        project_key = str(project_path)

        with self._watchers_lock:
            watcher = self._watchers.get(project_key)
            if not watcher:
                self._debug(f"No watcher found for: {project_path}")
                return False

            try:
                # Stop observer
                watcher.observer.stop()
                watcher.observer.join(timeout=5.0)

                if watcher.observer.is_alive():
                    self._debug(
                        f"Warning: Observer thread did not stop: {project_path}"
                    )

                # Remove from watchers
                del self._watchers[project_key]

                self._debug(f"Stopped watcher for: {project_path}")

                # Update registry status
                if self._registry:
                    self._registry.set_watcher_status(project_key, active=False)

                return True

            except Exception as e:
                self._debug(f"Error stopping watcher for {project_path}: {e}")
                logger.exception(f"Error stopping watcher: {project_path}")
                return False

    def stop_all(self) -> None:
        """Stop all file watchers."""
        with self._watchers_lock:
            project_paths = list(self._watchers.keys())

        for project_path in project_paths:
            self.stop_watcher(Path(project_path))

        # Clear any remaining pending events
        with self._pending_lock:
            count = len(self._pending_events)
            self._pending_events.clear()
            if count > 0:
                self._debug(f"Cleared {count} pending events")

    def get_watcher_status(self, project_path: Path) -> dict[str, Any] | None:
        """Get status information for a project's watcher.

        Args:
            project_path: Root path of the project

        Returns:
            Status dictionary or None if not watching
        """
        project_key = str(project_path)

        with self._watchers_lock:
            watcher = self._watchers.get(project_key)
            if not watcher:
                return None

            return {
                "project_path": str(watcher.project_path),
                "started_at": watcher.started_at,
                "events_processed": watcher.events_processed,
                "last_event_at": watcher.last_event_at,
                "observer_alive": watcher.observer.is_alive(),
            }

    def get_all_status(self) -> dict[str, dict[str, Any]]:
        """Get status for all watchers.

        Returns:
            Dictionary mapping project paths to status dictionaries
        """
        result = {}

        with self._watchers_lock:
            for project_key, watcher in self._watchers.items():
                result[project_key] = {
                    "project_path": str(watcher.project_path),
                    "started_at": watcher.started_at,
                    "events_processed": watcher.events_processed,
                    "last_event_at": watcher.last_event_at,
                    "observer_alive": watcher.observer.is_alive(),
                }

        return result

    def get_pending_count(self) -> int:
        """Get number of pending (debouncing) events.

        Returns:
            Number of events waiting to be processed
        """
        with self._pending_lock:
            return len(self._pending_events)

    def flush_pending(self, max_events: int = 100) -> int:
        """Force-flush pending events immediately.

        Processes up to max_events to bound shutdown time.
        Remaining events are discarded (will be re-indexed on restart).

        Args:
            max_events: Maximum events to process (default 100)

        Returns:
            Number of events flushed
        """
        with self._pending_lock:
            # Sort by first_seen time to process oldest first
            sorted_events = sorted(
                self._pending_events.items(),
                key=lambda x: x[1][1],  # Sort by first_seen time
            )
            # Take only max_events
            events_to_process = sorted_events[:max_events]
            discarded = len(sorted_events) - len(events_to_process)
            self._pending_events.clear()

        if discarded > 0:
            self._debug(f"Discarding {discarded} pending events during shutdown")
            logger.warning(
                f"WatcherManager: Discarding {discarded} pending events during "
                f"shutdown (exceeds max_events={max_events})"
            )

        count = 0
        for file_key, (event_type, _) in events_to_process:
            try:
                # Process synchronously with short timeout
                if self._loop:
                    future = asyncio.run_coroutine_threadsafe(
                        self._process_file_event(Path(file_key), event_type),
                        self._loop,
                    )
                    future.result(timeout=5.0)  # 5s timeout per event
                    count += 1
            except Exception as e:
                self._debug(f"Error flushing event for {file_key}: {e}")

        self._debug(f"Flushed {count} pending events")
        return count
