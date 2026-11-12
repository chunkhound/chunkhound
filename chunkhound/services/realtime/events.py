from __future__ import annotations

import asyncio
import os
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from chunkhound.core.config.config import Config
from chunkhound.services.realtime_path_filter import RealtimePathFilter

# The watchdog event types that can mean a file's content changed, and the type
# the consumer sees for each. This map is the only place a watchdog type is
# admitted; every other type is dropped where it arrives, before it takes an
# event-queue slot.
# On Linux, inotify reports every open and close (`opened`, `closed`,
# `closed_no_write`); a start-up scan or a type-check run reads the whole tree,
# so letting those through filled the queue, overflowed it and forced a full
# resync. The consumer never acted on `closed` (a write also yields
# `modified`, `created` or `moved`, except an mmap write, which inotify
# reports only on close). Without them two writes reach the consumer as
# `modified, modified`, so its duplicate check must neither drop a write made
# while the file is being processed nor let a dropped duplicate skip the
# debounce refresh (see `_consume_events`).
# `attrib` is a change: on Windows, watchdog 7 reports an in-place write as
# `attrib` whenever the record's change time is ahead of its write time, which
# an ordinary append usually is, so for most writes it is the only signal. On
# Linux it is IN_ATTRIB (chmod, touch), which watchdog 6 already reported as
# `modified`. Reads raise no `attrib` on Windows because
# ignore_windows_last_access() unsubscribes from last-access changes.
WATCHDOG_TO_CONSUMER_EVENT = {
    "created": "created",
    "modified": "modified",
    "attrib": "modified",
    "deleted": "deleted",
    "moved": "moved",
}


def ignore_windows_last_access() -> None:
    """Stop watchdog's Windows emitter from reporting last-access changes.

    When last-access updates are on (`fsutil behavior query
    disablelastaccess`), every read of a file not read for an hour changes its
    access time, and watchdog subscribes to that
    (FILE_NOTIFY_CHANGE_LAST_ACCESS): without this, each read raises an event
    (7: `attrib`, 6: `modified`) that takes a queue slot, and the start-up scan
    reads every file. Writes still arrive through LAST_WRITE, SIZE and
    ATTRIBUTES.

    watchdog has no per-watch flag: it reads the module global
    WATCHDOG_FILE_NOTIFY_FLAGS on every ReadDirectoryChangesW call (6.0 and
    7.1), so clearing the bit before an observer starts covers every watch.
    It is process-wide; ChunkHound's watchdog observer is its only user here.
    """
    if os.name != "nt":
        return
    try:
        from watchdog.observers import winapi
    except ImportError:
        return
    flags = getattr(winapi, "WATCHDOG_FILE_NOTIFY_FLAGS", None)
    last_access = getattr(winapi, "FILE_NOTIFY_CHANGE_LAST_ACCESS", None)
    if flags is None or last_access is None:
        return
    winapi.WATCHDOG_FILE_NOTIFY_FLAGS = flags & ~last_access


def watchdog_event_filter() -> list[type]:
    """The `event_filter` for every watchdog `schedule` call.

    On inotify it also narrows the kernel watch mask, so reads are never
    reported at all. Every schedule on one observer must pass the same filter:
    watchdog 7 shares one inotify descriptor per observer and each
    `inotify_add_watch` replaces the inode's mask, so a narrower filter on one
    watch would silently narrow an overlapping one.
    """
    # Imported here so this module stays importable without watchdog.
    import watchdog.events as watchdog_events

    names = {
        "created": ("FileCreatedEvent", "DirCreatedEvent"),
        "modified": ("FileModifiedEvent",),
        "deleted": ("FileDeletedEvent", "DirDeletedEvent"),
        "moved": ("FileMovedEvent", "DirMovedEvent"),
        # Only watchdog 7 has the attrib classes; 6 reports attrib as modified.
        "attrib": ("FileAttribEvent",),
    }
    return [
        getattr(watchdog_events, name)
        for event_type in sorted(names)
        for name in names[event_type]
        if hasattr(watchdog_events, name)
    ]


def normalize_file_path(path: Path | str) -> str:
    """Single source of truth for path normalization across ChunkHound."""
    return str(Path(path).resolve())


QueueResultCallback = Callable[[str, Path, bool, str | None], None]
AdmissionCallback = Callable[[str, Path, bool], bool]


class _WatchmanTranslationError(RuntimeError):
    def __init__(self, message: str, *, failure_kind: str) -> None:
        super().__init__(message)
        self.failure_kind = failure_kind


def _record_realtime_queue_result(
    queue_result_callback: QueueResultCallback | None,
    event_type: str,
    file_path: Path,
    accepted: bool,
    reason: str | None,
) -> None:
    try:
        if queue_result_callback:
            queue_result_callback(event_type, file_path, accepted, reason)
    except Exception:
        pass


def _enqueue_realtime_event(
    event_queue: asyncio.Queue[tuple[str, Path]] | None,
    queue_result_callback: QueueResultCallback | None,
    event_type: str,
    file_path: Path,
) -> None:
    if event_queue is None:
        _record_realtime_queue_result(
            queue_result_callback,
            event_type,
            file_path,
            False,
            "queue_unavailable",
        )
        return

    try:
        event_queue.put_nowait((event_type, file_path))
        _record_realtime_queue_result(
            queue_result_callback,
            event_type,
            file_path,
            True,
            None,
        )
    except asyncio.QueueFull:
        if queue_result_callback is None:
            logger.warning(
                f"Realtime event queue full; dropped {event_type} for {file_path}"
            )
        _record_realtime_queue_result(
            queue_result_callback,
            event_type,
            file_path,
            False,
            "queue_full",
        )
    except Exception as error:
        logger.warning(f"Failed to queue {event_type} event for {file_path}: {error}")
        _record_realtime_queue_result(
            queue_result_callback,
            event_type,
            file_path,
            False,
            type(error).__name__,
        )


def forward_to_loop(
    loop: asyncio.AbstractEventLoop, callback: Callable[[str, Path], None]
) -> Callable[[str, Path], None]:
    """Run an observer-thread event callback on the service's event loop.

    The event loop owns the service's event counters, event-pressure state and
    status snapshots; the watchdog observer calls back from its own thread.
    call_soon_threadsafe runs callbacks in the order they were scheduled, so a
    source event is still recorded before the queueing of the same event.
    """

    def forward(event_type: str, file_path: Path) -> None:
        loop.call_soon_threadsafe(callback, event_type, file_path)

    return forward


@dataclass(frozen=True, slots=True)
class RealtimeMutation:
    """One downstream realtime pipeline operation."""

    mutation_id: int
    operation: str
    path: Path
    first_queued_at: str
    retry_count: int = 0
    source_generation: int | None = None


@dataclass(slots=True)
class HotPathPressure:
    """Bounded rolling event-pressure accounting for one logical path."""

    event_timestamps: deque[float] = field(default_factory=deque)
    coalesced_timestamps: deque[float] = field(default_factory=deque)
    last_scope: str | None = None
    last_event_type: str | None = None
    last_observed_at: str | None = None
    last_observed_monotonic: float = 0.0


class SimpleEventHandler:
    """Watchdog-compatible event handler without import-time watchdog coupling."""

    def dispatch(self, event: Any) -> None:
        self.on_any_event(event)

    def __init__(
        self,
        event_queue: asyncio.Queue[tuple[str, Path]] | None,
        config: Config | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        root_path: Path | None = None,
        queue_result_callback: QueueResultCallback | None = None,
        source_event_callback: Callable[[str, Path], None] | None = None,
        filtered_event_callback: Callable[[str, Path], None] | None = None,
        admission_callback: AdmissionCallback | None = None,
    ):
        self.event_queue = event_queue
        self.config = config
        self.loop = loop
        self._queue_result_callback = queue_result_callback
        self._source_event_callback = source_event_callback
        self._filtered_event_callback = filtered_event_callback
        self._admission_callback = admission_callback
        if root_path is not None:
            self._root = root_path.resolve()
        else:
            try:
                self._root = (
                    config.target_dir if config and config.target_dir else Path.cwd()
                ).resolve()
            except Exception:
                self._root = Path.cwd().resolve()
        self._path_filter = RealtimePathFilter(config=config, root_path=self._root)

    def on_any_event(self, event: Any) -> None:
        """Handle filesystem events - simple queue operation."""
        # No callbacks for a dropped read: each would cost a cross-thread call.
        event_type = WATCHDOG_TO_CONSUMER_EVENT.get(event.event_type)
        if event_type is None:
            return

        # Handle directory creation
        if event_type == "created" and event.is_directory:
            # Queue directory creation for processing
            file_path = Path(normalize_file_path(event.src_path))
            self._record_source_event("dir_created", file_path)
            self._queue_event("dir_created", file_path)
            return

        # Handle directory deletion
        if event_type == "deleted" and event.is_directory:
            # Queue directory deletion for cleanup
            file_path = Path(normalize_file_path(event.src_path))
            self._record_source_event("dir_deleted", file_path)
            self._queue_event("dir_deleted", file_path)
            return

        # Skip other directory events (modified, moved)
        if event.is_directory:
            return

        # Handle move events for atomic writes
        if event_type == "moved" and hasattr(event, "dest_path"):
            self._handle_move_event(event.src_path, event.dest_path)
            return

        # Resolve path to canonical form to avoid /var vs /private/var issues
        file_path = Path(normalize_file_path(event.src_path))
        self._record_source_event(event_type, file_path)

        # Simple filtering for supported file types, with a shared
        # previously-indexed delete exception for cleanup correctness.
        if not self._should_admit_event(event_type, file_path):
            self._record_filtered_event(event_type, file_path)
            return

        self._queue_event(event_type, file_path)

    def _should_index(self, file_path: Path) -> bool:
        """Check if file should be indexed based on config patterns.

        Uses config-based filtering if available, otherwise falls back to
        Language enum which derives all patterns from parser_factory.
        This ensures realtime indexing supports all languages without
        requiring manual updates.
        """
        return self._path_filter.should_index(file_path)

    def _handle_move_event(self, src_path: str, dest_path: str) -> None:
        """Handle atomic file moves (temp -> final file)."""
        src_file = Path(normalize_file_path(src_path))
        dest_file = Path(normalize_file_path(dest_path))
        src_should_index = self._should_index(src_file)
        dest_should_index = self._should_index(dest_file)
        src_should_admit_delete = self._should_admit_event(
            "deleted",
            src_file,
            should_index=src_should_index,
        )

        # If moving FROM temp file TO supported file -> index destination
        if not src_should_index and dest_should_index:
            logger.debug(f"Atomic write detected: {src_path} -> {dest_path}")
            self._record_source_event("created", dest_file)
            self._queue_event("created", dest_file)

        # If moving FROM supported file -> handle as deletion + creation
        elif src_should_admit_delete and dest_should_index:
            logger.debug(f"File rename: {src_path} -> {dest_path}")
            self._record_source_event("deleted", src_file)
            self._queue_event("deleted", src_file)
            self._record_source_event("created", dest_file)
            self._queue_event("created", dest_file)

        # If moving FROM supported file TO temp/unsupported -> deletion
        elif src_should_admit_delete and not dest_should_index:
            logger.debug(f"File moved to temp/unsupported: {src_path}")
            self._record_source_event("deleted", src_file)
            self._queue_event("deleted", src_file)

    def _should_admit_event(
        self,
        event_type: str,
        file_path: Path,
        *,
        should_index: bool | None = None,
    ) -> bool:
        path_allowed = (
            should_index if should_index is not None else self._should_index(file_path)
        )
        if self._admission_callback is None:
            return path_allowed
        return self._admission_callback(event_type, file_path, path_allowed)

    def _queue_event(self, event_type: str, file_path: Path) -> None:
        """Queue an event for async processing."""
        if not self.loop or self.loop.is_closed() or self.event_queue is None:
            self._record_queue_result(event_type, file_path, False, "loop_unavailable")
            return

        try:
            self.loop.call_soon_threadsafe(
                _enqueue_realtime_event,
                self.event_queue,
                self._queue_result_callback,
                event_type,
                file_path,
            )
        except Exception as error:
            logger.warning(
                f"Failed to queue {event_type} event for {file_path}: {error}"
            )
            self._record_queue_result(
                event_type,
                file_path,
                False,
                type(error).__name__,
            )

    def _record_queue_result(
        self, event_type: str, file_path: Path, accepted: bool, reason: str | None
    ) -> None:
        try:
            if self._queue_result_callback:
                self._queue_result_callback(event_type, file_path, accepted, reason)
        except Exception:
            # Never let bookkeeping interfere with monitoring.
            pass

    def _record_source_event(self, event_type: str, file_path: Path) -> None:
        try:
            if self._source_event_callback:
                self._source_event_callback(event_type, file_path)
        except Exception:
            pass

    def _record_filtered_event(self, event_type: str, file_path: Path) -> None:
        try:
            if self._filtered_event_callback:
                self._filtered_event_callback(event_type, file_path)
        except Exception:
            pass
