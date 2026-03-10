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
import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from loguru import logger
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from chunkhound.core.config.config import Config
from chunkhound.database_factory import DatabaseServices
from chunkhound.services.realtime_path_filter import RealtimePathFilter
from chunkhound.utils.windows_constants import IS_WINDOWS
from chunkhound.watchman import (
    PrivateWatchmanSidecar,
    WatchmanCliSession,
    WatchmanScopePlan,
    WatchmanSubscriptionScope,
)


def normalize_file_path(path: Path | str) -> str:
    """Single source of truth for path normalization across ChunkHound."""
    return str(Path(path).resolve())


QueueResultCallback = Callable[[str, Path, bool, str | None], None]


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


class SimpleEventHandler(FileSystemEventHandler):
    """Simple sync event handler - no async complexity."""

    def __init__(
        self,
        event_queue: asyncio.Queue[tuple[str, Path]] | None,
        config: Config | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        root_path: Path | None = None,
        queue_result_callback: QueueResultCallback | None = None,
    ):
        self.event_queue = event_queue
        self.config = config
        self.loop = loop
        self._queue_result_callback = queue_result_callback
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

        self._queue_event(event.event_type, file_path)

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


class RealtimeMonitorAdapter(Protocol):
    """Backend-specific filesystem monitoring lifecycle."""

    backend_name: str

    async def start(
        self, watch_path: Path, loop: asyncio.AbstractEventLoop
    ) -> None: ...

    async def stop(self) -> None: ...

    def get_health(self) -> dict[str, Any]: ...


class WatchdogRealtimeAdapter:
    """Watchdog-backed monitor with polling fallback."""

    backend_name = "watchdog"

    def __init__(self, service: "RealtimeIndexingService") -> None:
        self._service = service

    async def start(self, watch_path: Path, loop: asyncio.AbstractEventLoop) -> None:
        self._service._set_effective_backend(self.backend_name)
        await self._service._setup_watchdog_with_timeout(watch_path, loop)

    async def stop(self) -> None:
        await self._service._cancel_watchdog_setup_task()
        await self._service._cancel_watchdog_bootstrap_future()
        await self._service._stop_observer()
        await self._service._cancel_polling_task()

    def get_health(self) -> dict[str, Any]:
        observer_alive = False
        if self._service.observer and self._service.observer.is_alive():
            observer_alive = True
        elif (
            self._service._using_polling
            and self._service._polling_task
            and not self._service._polling_task.done()
        ):
            observer_alive = True
        return {"observer_alive": observer_alive}


class PollingRealtimeAdapter:
    """Explicit polling backend."""

    backend_name = "polling"

    def __init__(self, service: "RealtimeIndexingService") -> None:
        self._service = service

    async def start(self, watch_path: Path, loop: asyncio.AbstractEventLoop) -> None:
        del loop
        await self._service._start_polling_backend(
            watch_path,
            reason="Configured realtime backend is polling",
            emit_warning=False,
        )

    async def stop(self) -> None:
        await self._service._cancel_polling_task()

    def get_health(self) -> dict[str, Any]:
        return {
            "observer_alive": bool(
                self._service._polling_task and not self._service._polling_task.done()
            )
        }


class WatchmanRealtimeAdapter:
    """Private Watchman sidecar and session bridge adapter."""

    backend_name = "watchman"
    _SUBSCRIPTION_NAME = "chunkhound-live-indexing"

    def __init__(self, service: "RealtimeIndexingService") -> None:
        self._service = service
        target_dir = getattr(service.config, "target_dir", None)
        if not isinstance(target_dir, Path):
            raise RuntimeError(
                "Watchman backend requires config.target_dir "
                "to resolve a private runtime root"
            )
        self._sidecar = PrivateWatchmanSidecar(target_dir, debug_sink=service._debug)
        self._session: WatchmanCliSession | None = None
        self._path_filter: RealtimePathFilter | None = None
        self._subscription_consumer_task: asyncio.Task[None] | None = None

    async def start(self, watch_path: Path, loop: asyncio.AbstractEventLoop) -> None:
        try:
            metadata = await self._sidecar.start()
        except Exception as error:
            message = f"Watchman sidecar startup failed: {error}"
            self._service._set_error(message)
            raise RuntimeError(message) from error

        self._service.watchman_scope_plan = None
        self._service.watchman_subscription_queue = None

        try:
            self._session = WatchmanCliSession(
                binary_path=Path(metadata.binary_path),
                socket_path=self._sidecar.paths.socket_path,
                project_root=self._sidecar.paths.project_root,
                debug_sink=self._service._debug,
            )
            setup = await self._session.start(
                target_path=watch_path,
                subscription_name=self._SUBSCRIPTION_NAME,
            )
        except Exception as error:
            if self._session is not None:
                await self._session.stop()
                self._session = None
            await self._sidecar.stop()
            self._service.watchman_scope_plan = None
            self._service.watchman_subscription_queue = None
            message = f"Watchman session startup failed: {error}"
            self._service._set_error(message)
            raise RuntimeError(message) from error

        self._service.watchman_scope_plan = setup.scope_plan
        self._service.watchman_subscription_queue = self._session.subscription_queue
        self._path_filter = RealtimePathFilter(
            config=self._service.config,
            root_path=setup.scope_plan.primary_scope.requested_path,
        )
        self._subscription_consumer_task = loop.create_task(
            self._consume_subscription_pdus(setup.scope_plan.primary_scope)
        )
        self._service._set_effective_backend(self.backend_name)
        self._service._monitoring_ready_at = self._service._utc_now()
        self._service.monitoring_ready.set()
        self._service._emit_status_update()

    async def stop(self) -> None:
        self._service.watchman_scope_plan = None
        self._service.watchman_subscription_queue = None
        self._path_filter = None
        if self._subscription_consumer_task is not None:
            self._subscription_consumer_task.cancel()
            try:
                await self._subscription_consumer_task
            except asyncio.CancelledError:
                pass
            self._subscription_consumer_task = None
        if self._session is not None:
            await self._session.stop()
            self._session = None
        await self._sidecar.stop()

    async def _consume_subscription_pdus(
        self, scope: WatchmanSubscriptionScope
    ) -> None:
        session = self._session
        if session is None:
            return

        while True:
            payload = await session.subscription_queue.get()
            try:
                self._translate_subscription_pdu(payload, scope)
            except Exception as error:
                message = f"Watchman event translation failed: {error}"
                logger.warning(message)
                self._service._set_warning(message)
            finally:
                session.subscription_queue.task_done()

    def _translate_subscription_pdu(
        self, payload: dict[str, object], scope: WatchmanSubscriptionScope
    ) -> None:
        files = payload.get("files")
        if not isinstance(files, list):
            self._warn_translation_issue(
                "Watchman subscription PDU did not include a files list"
            )
            return

        path_filter = self._path_filter
        if path_filter is None:
            self._warn_translation_issue(
                "Watchman event translation ran without an active path filter"
            )
            return

        for entry in files:
            if not isinstance(entry, dict):
                self._warn_translation_issue(
                    "Watchman subscription entry was not an object"
                )
                continue
            translated = self._translate_watchman_file_entry(entry, scope)
            if translated is None:
                continue
            event_type, file_path = translated
            if not path_filter.should_index(file_path):
                continue
            _enqueue_realtime_event(
                self._service.event_queue,
                self._service._handle_queue_result,
                event_type,
                file_path,
            )

    def _translate_watchman_file_entry(
        self,
        entry: dict[str, object],
        scope: WatchmanSubscriptionScope,
    ) -> tuple[str, Path] | None:
        file_type = entry.get("type")
        if file_type not in {None, "f"}:
            self._warn_translation_issue(
                f"Skipping unexpected Watchman file type {file_type!r}"
            )
            return None

        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            self._warn_translation_issue(
                "Skipping Watchman subscription entry without a valid name"
            )
            return None

        relative_name = PurePosixPath(name.strip().replace("\\", "/"))
        if (
            relative_name.is_absolute()
            or ".." in relative_name.parts
            or not relative_name.parts
        ):
            self._warn_translation_issue(
                f"Skipping unsafe Watchman subscription path {name!r}"
            )
            return None

        relative_root = (
            PurePosixPath(scope.relative_root) if scope.relative_root else None
        )
        mapped_parts = []
        if relative_root is not None:
            mapped_parts.extend(relative_root.parts)
        mapped_parts.extend(relative_name.parts)
        canonical_path = Path(
            normalize_file_path(scope.watch_root.joinpath(*mapped_parts))
        )
        try:
            canonical_path.relative_to(scope.requested_path)
        except ValueError:
            self._warn_translation_issue(
                f"Skipping out-of-scope Watchman path {canonical_path}"
            )
            return None

        exists = entry.get("exists")
        is_new = entry.get("new")
        if exists is False:
            event_type = "deleted"
        elif exists is True and is_new is True:
            event_type = "created"
        else:
            event_type = "modified"
        return event_type, canonical_path

    def _warn_translation_issue(self, message: str) -> None:
        warning = f"Watchman event translation warning: {message}"
        logger.warning(warning)
        self._service._set_warning(warning)

    def get_health(self) -> dict[str, Any]:
        health = self._sidecar.get_health()
        if self._session is not None:
            health.update(self._session.get_health())
        else:
            health.update(
                {
                    "watchman_session_alive": False,
                    "watchman_session_pid": None,
                    "watchman_session_last_warning": None,
                    "watchman_session_last_warning_at": None,
                    "watchman_session_last_error": None,
                    "watchman_session_last_error_at": None,
                    "watchman_session_last_response_at": None,
                    "watchman_subscription_last_received_at": None,
                    "watchman_session_command_count": 0,
                    "watchman_subscription_queue_size": 0,
                    "watchman_subscription_queue_maxsize": 1000,
                    "watchman_subscription_pdu_count": 0,
                    "watchman_subscription_pdu_dropped": 0,
                    "watchman_subscription_name": None,
                    "watchman_watch_root": None,
                    "watchman_relative_root": None,
                    "watchman_session_capabilities": {},
                }
            )
        health["observer_alive"] = bool(health.get("watchman_alive")) and bool(
            health.get("watchman_session_alive")
        )
        return health


class RealtimeIndexingService:
    """Simple real-time indexing service with search responsiveness."""

    # Event deduplication window - suppress duplicate events within this period
    _EVENT_DEDUP_WINDOW_SECONDS = 2.0
    # Retention period for event history - entries older than this are cleaned up
    _EVENT_HISTORY_RETENTION_SECONDS = 10.0
    _EVENT_QUEUE_MAXSIZE = 1000
    _RESYNC_DEBOUNCE_SECONDS = 1.0
    _WATCHDOG_SETUP_TIMEOUT_SECONDS = 5.0
    _MONITORING_READY_TIMEOUT_SECONDS = 10.0
    _POLLING_STARTUP_SETTLE_SECONDS = 0.5

    def __init__(
        self,
        services: DatabaseServices,
        config: Config,
        debug_sink: Callable[[str], None] | None = None,
        status_callback: Callable[[dict[str, Any]], None] | None = None,
        resync_callback: Callable[[str, dict[str, Any] | None], Awaitable[None]]
        | None = None,
    ):
        self.services = services
        self.config = config
        # Optional sink that writes to MCPServerBase.debug_log so events land in
        # /tmp/chunkhound_mcp_debug.log when CHUNKHOUND_DEBUG is enabled.
        self._debug_sink = debug_sink
        self._status_callback = status_callback
        self._resync_callback = resync_callback
        self._configured_backend = self._resolve_configured_backend()
        self._effective_backend = "uninitialized"
        self._monitor_adapter: RealtimeMonitorAdapter | None = None
        self.watchman_scope_plan: WatchmanScopePlan | None = None
        self.watchman_subscription_queue: asyncio.Queue[dict[str, object]] | None = None

        # Existing asyncio queue for priority processing
        self.file_queue: asyncio.Queue[tuple[str, Path]] = asyncio.Queue()

        # NEW: Async queue for events from watchdog (thread-safe via asyncio)
        self.event_queue: asyncio.Queue[tuple[str, Path]] = asyncio.Queue(
            maxsize=self._EVENT_QUEUE_MAXSIZE
        )

        # Deduplication and error tracking
        self.pending_files: set[Path] = set()
        self.failed_files: set[str] = set()
        self._last_warning: str | None = None
        self._last_warning_at: str | None = None
        self._last_error: str | None = None
        self._last_error_at: str | None = None

        # Simple debouncing for rapid file changes
        self._pending_debounce: dict[str, float] = {}  # file_path -> timestamp
        self._debounce_delay = 0.5  # 500ms delay from research
        self._debounce_tasks: set[asyncio.Task] = set()  # Track active debounce tasks

        self._recent_file_events: dict[
            str, tuple[str, float]
        ] = {}  # Layer 3: event dedup
        self._event_queue_accepted = 0
        self._event_queue_dropped = 0
        self._event_queue_last_reason: str | None = None
        self._event_queue_last_event_type: str | None = None
        self._event_queue_last_file_path: str | None = None
        self._event_queue_last_enqueued_at: str | None = None
        self._event_queue_last_dropped_at: str | None = None

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
        self._watchdog_setup_task: asyncio.Task | None = None
        self._watchdog_bootstrap_future: (
            asyncio.Future[tuple[Observer, SimpleEventHandler] | None] | None
        ) = None
        self._watchdog_bootstrap_abort = threading.Event()
        self._resync_dispatch_task: asyncio.Task | None = None
        self._active_start_task: asyncio.Task | None = None
        self._start_generation = 0
        self._using_polling = False
        self._service_state = "idle"
        self._last_poll_snapshot_at: str | None = None
        self._last_poll_files_checked = 0
        self._last_poll_snapshot_truncated = False

        # Directory watch management for progressive monitoring
        self.watched_directories: set[str] = set()  # Track watched dirs
        self.watch_lock = asyncio.Lock()  # Protect concurrent access

        # Monitoring readiness coordination
        self.monitoring_ready = asyncio.Event()  # Signals when monitoring is ready
        self._monitoring_ready_at: str | None = None

        # Backend-neutral resync substrate
        self._needs_resync = False
        self._resync_in_progress = False
        self._resync_request_count = 0
        self._resync_performed_count = 0
        self._last_resync_reason: str | None = None
        self._last_resync_details: dict[str, Any] | None = None
        self._last_resync_requested_at: str | None = None
        self._last_resync_started_at: str | None = None
        self._last_resync_completed_at: str | None = None
        self._last_resync_error: str | None = None
        self._last_resync_request_monotonic: float | None = None

    @staticmethod
    def _utc_now() -> str:
        """Return an ISO8601 UTC timestamp."""
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    @classmethod
    def default_health_snapshot(
        cls, configured_backend: str | None = None
    ) -> dict[str, Any]:
        """Return the neutral realtime health structure used by MCP status plumbing."""
        return {
            "configured_backend": configured_backend,
            "effective_backend": "uninitialized",
            "service_state": "idle",
            "monitoring_mode": "uninitialized",
            "monitoring_ready": False,
            "monitoring_ready_at": None,
            "observer_alive": False,
            "watching_directory": None,
            "watched_directories_count": 0,
            "queue_size": 0,
            "pending_files": 0,
            "failed_files": 0,
            "last_warning": None,
            "last_warning_at": None,
            "last_error": None,
            "last_error_at": None,
            "event_queue": {
                "size": 0,
                "maxsize": cls._EVENT_QUEUE_MAXSIZE,
                "accepted": 0,
                "dropped": 0,
                "last_reason": None,
                "last_event_type": None,
                "last_file_path": None,
                "last_enqueued_at": None,
                "last_dropped_at": None,
            },
            "resync": {
                "needs_resync": False,
                "in_progress": False,
                "debounce_seconds": cls._RESYNC_DEBOUNCE_SECONDS,
                "request_count": 0,
                "performed_count": 0,
                "last_reason": None,
                "last_details": None,
                "last_requested_at": None,
                "last_started_at": None,
                "last_completed_at": None,
                "last_error": None,
            },
            "polling": {
                "last_snapshot_at": None,
                "last_files_checked": 0,
                "last_snapshot_truncated": False,
            },
        }

    @classmethod
    def health_snapshot_for_config(cls, config: Any | None) -> dict[str, Any]:
        """Return the neutral realtime health snapshot seeded from config."""
        configured_backend = None
        try:
            backend = getattr(
                getattr(config, "indexing", None), "realtime_backend", None
            )
            if backend in {"watchman", "watchdog", "polling"}:
                configured_backend = str(backend)
        except Exception:
            configured_backend = None
        return cls.default_health_snapshot(configured_backend=configured_backend)

    # Internal helper to forward realtime events into the MCP debug log file
    def _debug(self, message: str) -> None:
        try:
            if self._debug_sink:
                # Prefix with RT to make it easy to filter
                self._debug_sink(f"RT: {message}")
        except Exception:
            # Never let debug plumbing affect runtime
            pass

    def _set_warning(self, message: str) -> None:
        self._last_warning = message
        self._last_warning_at = self._utc_now()
        self._emit_status_update()

    def _set_error(self, message: str) -> None:
        self._last_error = message
        self._last_error_at = self._utc_now()
        if self._service_state not in {"stopping", "stopped"}:
            self._service_state = "degraded"
        self._emit_status_update()

    def _clear_resync_error_state(self) -> None:
        """Clear degraded state only when it originated from resync plumbing."""
        self._last_resync_error = None
        if self._last_error == "No resync callback configured" or (
            self._last_error and self._last_error.startswith("Realtime resync failed:")
        ):
            self._last_error = None
            self._last_error_at = None
        if self._service_state not in {"stopping", "stopped"} and not self._last_error:
            self._service_state = "running"
        self._emit_status_update()

    def _resolve_configured_backend(self) -> str:
        backend = getattr(self.config.indexing, "realtime_backend", "watchdog")
        if backend in {"watchman", "watchdog", "polling"}:
            return str(backend)
        return "watchdog"

    def _set_effective_backend(self, backend: str) -> None:
        self._effective_backend = backend

    def _start_still_owned(self, start_generation: int) -> bool:
        """Return whether a start() invocation still owns service startup state."""
        return (
            start_generation == self._start_generation
            and self._service_state not in {"stopping", "stopped"}
        )

    def _build_monitor_adapter(self) -> RealtimeMonitorAdapter:
        if self._configured_backend == "watchman":
            return WatchmanRealtimeAdapter(self)
        if self._configured_backend == "polling":
            return PollingRealtimeAdapter(self)
        return WatchdogRealtimeAdapter(self)

    def _build_health_snapshot(self) -> dict[str, Any]:
        monitoring_active = False
        if self.observer and self.observer.is_alive():
            monitoring_active = True
        elif (
            self._using_polling and self._polling_task and not self._polling_task.done()
        ):
            monitoring_active = True
        adapter_health = (
            self._monitor_adapter.get_health() if self._monitor_adapter else {}
        )
        if "observer_alive" in adapter_health:
            monitoring_active = bool(adapter_health["observer_alive"])

        effective_backend = self._effective_backend
        if self.watch_path is None and effective_backend == "uninitialized":
            monitoring_mode = "uninitialized"
        else:
            monitoring_mode = effective_backend

        status = self.default_health_snapshot()
        status.update(
            {
                "configured_backend": self._configured_backend,
                "effective_backend": effective_backend,
                "service_state": self._service_state,
                "monitoring_mode": monitoring_mode,
                "monitoring_ready": self.monitoring_ready.is_set(),
                "monitoring_ready_at": self._monitoring_ready_at,
                "observer_alive": monitoring_active,
                "watching_directory": str(self.watch_path) if self.watch_path else None,
                "watched_directories_count": len(self.watched_directories),
                "queue_size": self.file_queue.qsize(),
                "pending_files": len(self.pending_files),
                "failed_files": len(self.failed_files),
                "last_warning": self._last_warning,
                "last_warning_at": self._last_warning_at,
                "last_error": self._last_error,
                "last_error_at": self._last_error_at,
            }
        )
        for key, value in adapter_health.items():
            if key != "observer_alive":
                status[key] = value
        status["event_queue"].update(
            {
                "size": self.event_queue.qsize(),
                "maxsize": self.event_queue.maxsize,
                "accepted": self._event_queue_accepted,
                "dropped": self._event_queue_dropped,
                "last_reason": self._event_queue_last_reason,
                "last_event_type": self._event_queue_last_event_type,
                "last_file_path": self._event_queue_last_file_path,
                "last_enqueued_at": self._event_queue_last_enqueued_at,
                "last_dropped_at": self._event_queue_last_dropped_at,
            }
        )
        status["resync"].update(
            {
                "needs_resync": self._needs_resync,
                "in_progress": self._resync_in_progress,
                "debounce_seconds": self._RESYNC_DEBOUNCE_SECONDS,
                "request_count": self._resync_request_count,
                "performed_count": self._resync_performed_count,
                "last_reason": self._last_resync_reason,
                "last_details": self._last_resync_details,
                "last_requested_at": self._last_resync_requested_at,
                "last_started_at": self._last_resync_started_at,
                "last_completed_at": self._last_resync_completed_at,
                "last_error": self._last_resync_error,
            }
        )
        status["polling"].update(
            {
                "last_snapshot_at": self._last_poll_snapshot_at,
                "last_files_checked": self._last_poll_files_checked,
                "last_snapshot_truncated": self._last_poll_snapshot_truncated,
            }
        )
        return status

    def _emit_status_update(self) -> None:
        try:
            if self._status_callback:
                self._status_callback(self._build_health_snapshot())
        except Exception:
            # Status plumbing must never affect runtime behavior.
            pass

    def _handle_queue_result(
        self, event_type: str, file_path: Path, accepted: bool, reason: str | None
    ) -> None:
        timestamp = self._utc_now()
        self._event_queue_last_event_type = event_type
        self._event_queue_last_file_path = str(file_path)

        if accepted:
            self._event_queue_accepted += 1
            self._event_queue_last_enqueued_at = timestamp
        else:
            self._event_queue_dropped += 1
            self._event_queue_last_reason = reason
            self._event_queue_last_dropped_at = timestamp
            self._set_warning(f"realtime event dropped ({reason or 'unknown_reason'})")
            if reason == "queue_full":
                asyncio.create_task(
                    self.request_resync(
                        "event_queue_overflow",
                        {
                            "event_type": event_type,
                            "file_path": str(file_path),
                            "drop_reason": reason,
                        },
                    )
                )

        self._emit_status_update()

    async def request_resync(
        self, reason: str, details: dict[str, Any] | None = None
    ) -> bool:
        """Request a debounced backend-neutral reconciliation scan."""
        self._needs_resync = True
        self._resync_request_count += 1
        self._last_resync_reason = reason
        self._last_resync_details = details
        self._last_resync_requested_at = self._utc_now()
        self._last_resync_request_monotonic = time.monotonic()
        self._emit_status_update()

        if self._resync_dispatch_task and not self._resync_dispatch_task.done():
            return False

        self._resync_dispatch_task = asyncio.create_task(self._dispatch_resync())
        return True

    async def _dispatch_resync(self) -> None:
        """Coalesce resync requests and run the callback on the trailing edge."""
        try:
            while True:
                requested_at = self._last_resync_request_monotonic
                await asyncio.sleep(self._RESYNC_DEBOUNCE_SECONDS)
                if requested_at == self._last_resync_request_monotonic:
                    break

            reason = self._last_resync_reason or "unspecified"
            details = self._last_resync_details
            callback = self._resync_callback
            if callback is None:
                self._last_resync_error = "No resync callback configured"
                self._set_error(self._last_resync_error)
                return

            while True:
                started_request_at = self._last_resync_request_monotonic
                self._resync_in_progress = True
                self._last_resync_started_at = self._utc_now()
                self._last_resync_error = None
                self._emit_status_update()

                try:
                    await callback(reason, details)
                    self._needs_resync = False
                    self._resync_performed_count += 1
                    self._last_resync_completed_at = self._utc_now()
                    if self._service_state not in {"stopping", "stopped"}:
                        self._clear_resync_error_state()
                except Exception as e:
                    self._last_resync_error = str(e)
                    self._set_error(f"Realtime resync failed: {e}")
                    break
                finally:
                    self._resync_in_progress = False
                    self._emit_status_update()

                if started_request_at == self._last_resync_request_monotonic:
                    break
                reason = self._last_resync_reason or reason
                details = self._last_resync_details
        finally:
            self._resync_dispatch_task = None

    async def _cancel_watchdog_setup_task(self) -> None:
        if self._watchdog_setup_task:
            self._watchdog_setup_task.cancel()
            try:
                await self._watchdog_setup_task
            except asyncio.CancelledError:
                pass
            self._watchdog_setup_task = None

    async def _cancel_watchdog_bootstrap_future(self) -> None:
        self._watchdog_bootstrap_abort.set()
        if (
            self._watchdog_bootstrap_future
            and not self._watchdog_bootstrap_future.done()
        ):
            try:
                await asyncio.wait_for(self._watchdog_bootstrap_future, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            except Exception as error:
                logger.debug(
                    "Watchdog bootstrap future raised during shutdown: "
                    f"{type(error).__name__}: {error}"
                )
        if self._watchdog_bootstrap_future and self._watchdog_bootstrap_future.done():
            self._watchdog_bootstrap_future = None

    async def _stop_observer(self) -> None:
        if self.observer:
            self.observer.stop()
            try:
                loop = asyncio.get_event_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(None, self.observer.join), timeout=1.0
                )
            except asyncio.TimeoutError:
                logger.warning("Observer thread did not exit within timeout")
            self.observer = None
            self.event_handler = None

    async def _cancel_polling_task(self) -> None:
        if self._polling_task:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass
            self._polling_task = None
        self._using_polling = False

    async def _cancel_processing_tasks(self) -> None:
        if self.event_consumer_task:
            self.event_consumer_task.cancel()
            try:
                await self.event_consumer_task
            except asyncio.CancelledError:
                pass
            self.event_consumer_task = None

        if self.process_task:
            self.process_task.cancel()
            try:
                await self.process_task
            except asyncio.CancelledError:
                pass
            self.process_task = None

    async def start(self, watch_path: Path) -> None:
        """Start real-time indexing service."""
        start_task = asyncio.current_task()
        self._active_start_task = start_task
        self._start_generation += 1
        start_generation = self._start_generation

        # Resolve path to canonical form for Windows 8.3 short name handling
        # This ensures polling monitor paths stay aligned with the coordinator's
        # normalized base directory.
        watch_path = watch_path.resolve()

        try:
            logger.debug(f"Starting real-time indexing for {watch_path}")
            self._debug(f"start watch on {watch_path}")
            self._service_state = "starting"
            self._configured_backend = self._resolve_configured_backend()
            self._effective_backend = "uninitialized"
            self._monitor_adapter = self._build_monitor_adapter()
            self.monitoring_ready.clear()
            self._monitoring_ready_at = None

            # Store the watch path
            self.watch_path = watch_path
            self._emit_status_update()

            loop = asyncio.get_event_loop()

            # Start all necessary tasks
            self.event_consumer_task = asyncio.create_task(self._consume_events())
            self.process_task = asyncio.create_task(self._process_loop())

            await self._monitor_adapter.start(watch_path, loop)
            if not self._start_still_owned(start_generation):
                return

            # Wait for monitoring to be confirmed ready
            monitoring_ok = await self.wait_for_monitoring_ready(
                timeout=self._MONITORING_READY_TIMEOUT_SECONDS
            )
            if not self._start_still_owned(start_generation):
                return

            if monitoring_ok:
                self._service_state = "running"
                self._debug("monitoring ready")
            else:
                self._service_state = "degraded"
                self._set_warning(
                    "Monitoring did not become ready before startup timeout"
                )
                self._debug("monitoring timeout; continuing")
            self._emit_status_update()
        except Exception:
            await self._cancel_processing_tasks()
            raise
        finally:
            if self._active_start_task is start_task:
                self._active_start_task = None

    async def stop(self) -> None:
        """Stop the service gracefully."""
        logger.debug("Stopping real-time indexing service")
        self._debug("stopping service")
        self._start_generation += 1
        self._service_state = "stopping"
        self.monitoring_ready.clear()
        self._monitoring_ready_at = None
        self._emit_status_update()

        active_start_task = self._active_start_task
        if (
            active_start_task
            and active_start_task is not asyncio.current_task()
            and not active_start_task.done()
        ):
            active_start_task.cancel()

        if self._monitor_adapter:
            await self._monitor_adapter.stop()

        await self._cancel_processing_tasks()

        if self._resync_dispatch_task:
            self._resync_dispatch_task.cancel()
            try:
                await self._resync_dispatch_task
            except asyncio.CancelledError:
                pass
            self._resync_dispatch_task = None

        # Cancel all active debounce tasks
        for task in self._debounce_tasks.copy():
            task.cancel()

        # Wait for debounce tasks to finish cancelling
        if self._debounce_tasks:
            await asyncio.gather(*self._debounce_tasks, return_exceptions=True)
            self._debounce_tasks.clear()

        self._service_state = "stopped"
        self._monitor_adapter = None
        self._emit_status_update()

    async def _setup_watchdog_with_timeout(
        self, watch_path: Path, loop: asyncio.AbstractEventLoop
    ) -> None:
        """Setup watchdog with timeout - fall back to polling if it takes too long."""
        self._watchdog_bootstrap_abort = threading.Event()
        self._watchdog_bootstrap_future = loop.run_in_executor(
            None,
            self._bootstrap_fs_monitor,
            watch_path,
            loop,
            self._watchdog_bootstrap_abort,
        )
        self._watchdog_bootstrap_future.add_done_callback(
            self._handle_watchdog_bootstrap_done
        )

        try:
            bootstrap_result = await asyncio.wait_for(
                asyncio.shield(self._watchdog_bootstrap_future),
                timeout=self._WATCHDOG_SETUP_TIMEOUT_SECONDS,
            )
            if bootstrap_result is None:
                return
            observer, event_handler = bootstrap_result
            self._adopt_watchdog_monitor(observer, event_handler, watch_path)
        except asyncio.TimeoutError:
            self._watchdog_bootstrap_abort.set()
            logger.info(
                f"Watchdog setup timed out for {watch_path} - falling back to polling"
            )
            await self._start_polling_backend(
                watch_path,
                reason="Watchdog setup timed out; switched to polling mode",
            )
        except Exception as e:
            self._watchdog_bootstrap_abort.set()
            logger.warning(f"Watchdog setup failed: {e} - falling back to polling")
            await self._start_polling_backend(
                watch_path,
                reason=f"Watchdog setup failed; switched to polling mode: {e}",
            )
        finally:
            if (
                self._watchdog_bootstrap_future
                and self._watchdog_bootstrap_future.done()
            ):
                self._watchdog_bootstrap_future = None

    def _handle_watchdog_bootstrap_done(
        self, future: asyncio.Future[tuple[Observer, SimpleEventHandler] | None]
    ) -> None:
        """Drain watchdog bootstrap exceptions and reflect unexpected failures."""
        if self._watchdog_bootstrap_future is future:
            self._watchdog_bootstrap_future = None
        if future.cancelled():
            return

        try:
            bootstrap_result = future.result()
        except Exception as error:
            if (
                not self._watchdog_bootstrap_abort.is_set()
                and self._service_state not in {"stopping", "stopped"}
            ):
                logger.warning(f"Watchdog bootstrap failed: {error}")
                self._set_error(f"Watchdog bootstrap failed: {error}")
            return

        if bootstrap_result is None:
            return

        observer, _event_handler = bootstrap_result
        if (
            self._watchdog_bootstrap_abort.is_set()
            or self._using_polling
            or self._service_state in {"stopping", "stopped"}
        ):
            self._stop_bootstrap_observer(observer)

    async def _start_polling_backend(
        self, watch_path: Path, reason: str, emit_warning: bool = True
    ) -> None:
        """Start polling mode and optionally record it as a fallback warning."""
        if not self._using_polling or not self._polling_task:
            self._using_polling = True
            self._polling_task = asyncio.create_task(self._polling_monitor(watch_path))
        self._set_effective_backend("polling")
        await asyncio.sleep(self._POLLING_STARTUP_SETTLE_SECONDS)
        self._monitoring_ready_at = self._utc_now()
        self.monitoring_ready.set()
        self._debug(reason)
        if emit_warning:
            self._set_warning(reason)
        else:
            self._emit_status_update()

    def _adopt_watchdog_monitor(
        self,
        observer: Observer,
        event_handler: SimpleEventHandler,
        watch_path: Path,
    ) -> None:
        """Adopt a successfully bootstrapped watchdog observer into service state."""
        if (
            self._watchdog_bootstrap_abort.is_set()
            or self._using_polling
            or self._service_state in {"stopping", "stopped"}
        ):
            self._stop_bootstrap_observer(observer)
            return

        self.event_handler = event_handler
        self.observer = observer
        self._using_polling = False
        self.watched_directories.add(str(watch_path))
        logger.debug("Watchdog setup completed successfully (recursive mode)")
        self._debug("watchdog setup complete (recursive)")
        self._set_effective_backend("watchdog")
        self._monitoring_ready_at = self._utc_now()
        self.monitoring_ready.set()
        self._emit_status_update()

    @staticmethod
    def _stop_bootstrap_observer(observer: Observer) -> None:
        """Stop a watchdog observer created during a bootstrap race."""
        try:
            observer.stop()
        except Exception:
            pass
        try:
            observer.join(timeout=1.0)
        except Exception:
            pass

    def _bootstrap_fs_monitor(
        self,
        watch_path: Path,
        loop: asyncio.AbstractEventLoop,
        abort_event: threading.Event,
    ) -> tuple[Observer, SimpleEventHandler] | None:
        """Create and start a watchdog observer without mutating shared state."""
        event_handler = SimpleEventHandler(
            self.event_queue,
            self.config,
            loop,
            root_path=watch_path,
            queue_result_callback=self._handle_queue_result,
        )
        observer = Observer()

        # Use recursive=True to ensure all directory events are captured
        # This is necessary for proper real-time monitoring of new directories
        observer.schedule(
            event_handler,
            str(watch_path),
            recursive=True,  # Use recursive for complete event coverage
        )
        observer.start()

        # Wait for observer thread to be fully running
        # On Windows, observer thread startup can be noticeably slower.
        # Give it more time to become alive to avoid unnecessary polling fallback.
        max_wait = 5.0 if IS_WINDOWS else 1.0
        start = time.time()
        while not observer.is_alive() and (time.time() - start) < max_wait:
            if abort_event.is_set():
                self._stop_bootstrap_observer(observer)
                return None
            time.sleep(0.01)

        if abort_event.is_set():
            self._stop_bootstrap_observer(observer)
            return None

        if observer.is_alive():
            logger.debug(f"Started recursive filesystem monitoring for {watch_path}")
            return observer, event_handler

        self._stop_bootstrap_observer(observer)
        raise RuntimeError("Observer failed to start within timeout")

    async def _add_subdirectories_progressively(self, root_path: Path) -> None:
        """No longer needed - using recursive monitoring."""
        logger.debug(
            "Progressive directory addition skipped (using recursive monitoring)"
        )

    def _polling_snapshot(self, watch_path: Path) -> tuple[dict[Path, int], int, bool]:
        """Collect a filesystem snapshot off the event loop for polling mode."""
        current_files: dict[Path, int] = {}
        files_checked = 0
        truncated = False
        simple_handler = SimpleEventHandler(
            None, self.config, None, root_path=watch_path
        )

        for file_path in watch_path.rglob("*"):
            try:
                if not file_path.is_file():
                    continue

                files_checked += 1
                if simple_handler._should_index(file_path):
                    try:
                        current_mtime = file_path.stat().st_mtime_ns
                    except OSError:
                        continue

                    current_files[file_path] = current_mtime

                if files_checked >= 5000:
                    truncated = True
                    break
            except (OSError, PermissionError):
                continue

        return current_files, files_checked, truncated

    def _collect_supported_files(self, dir_path: Path) -> list[Path]:
        """Collect supported files in a directory off the event loop."""
        simple_handler = SimpleEventHandler(None, self.config, None, root_path=dir_path)
        supported_files: list[Path] = []

        for file_path in dir_path.rglob("*"):
            try:
                if file_path.is_file() and simple_handler._should_index(file_path):
                    supported_files.append(file_path)
            except (OSError, PermissionError):
                continue

        return supported_files

    async def _polling_monitor(self, watch_path: Path) -> None:
        """Simple polling monitor for large directories."""
        logger.debug(f"Starting polling monitor for {watch_path}")
        self._debug(f"polling monitor active for {watch_path}")
        # Track files with their mtime to detect modifications (not just new/deleted)
        known_files: dict[Path, int] = {}

        # Use a shorter interval during the first few seconds to ensure
        # freshly created files are detected quickly after startup/fallback.
        polling_start = time.time()

        while True:
            try:
                current_files, files_checked, truncated = await asyncio.to_thread(
                    self._polling_snapshot, watch_path
                )
                self._last_poll_snapshot_at = self._utc_now()
                self._last_poll_files_checked = files_checked
                self._last_poll_snapshot_truncated = truncated
                if truncated:
                    self._set_warning(
                        "Polling snapshot truncated after 5000 files to avoid "
                        "event-loop starvation"
                    )

                for file_path, current_mtime in current_files.items():
                    if file_path not in known_files:
                        logger.debug(f"Polling detected new file: {file_path}")
                        self._debug(f"polling detected new file: {file_path}")
                        await self.add_file(file_path, priority="change")
                    elif known_files[file_path] != current_mtime:
                        logger.debug(f"Polling detected modified file: {file_path}")
                        self._debug(f"polling detected modified file: {file_path}")
                        await self.add_file(file_path, priority="change")

                # Check for deleted files
                deleted = set(known_files.keys()) - set(current_files.keys())
                for file_path in deleted:
                    logger.debug(f"Polling detected deleted file: {file_path}")
                    await self.remove_file(file_path)
                    self._debug(f"polling detected deleted file: {file_path}")

                known_files = current_files

                # Adaptive poll interval: 1s for the first 10s, then 5s
                elapsed = time.time() - polling_start
                interval = 1.0 if elapsed < 10.0 else 5.0
                self._emit_status_update()
                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Polling monitor error: {e}")
                self._set_error(f"Polling monitor error: {e}")
                await asyncio.sleep(5)

    async def add_file(self, file_path: Path, priority: str = "change") -> None:
        """Add file to processing queue with deduplication and debouncing."""
        if file_path not in self.pending_files:
            self.pending_files.add(file_path)

            # Simple debouncing for change events
            if priority == "change":
                file_str = str(file_path)
                current_time = time.monotonic()

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
                    self._emit_status_update()
            else:
                # Priority scan events bypass debouncing
                await self.file_queue.put((priority, file_path))
                self._debug(f"queued {file_path} priority={priority}")
                self._emit_status_update()

    async def _debounced_add_file(self, file_path: Path, priority: str) -> None:
        """Process file after debounce delay."""
        await asyncio.sleep(self._debounce_delay)

        file_str = str(file_path)
        if file_str in self._pending_debounce:
            last_update = self._pending_debounce[file_str]

            # Check if no recent updates during delay
            if time.monotonic() - last_update >= self._debounce_delay:
                del self._pending_debounce[file_str]
                await self.file_queue.put((priority, file_path))
                logger.debug(f"Processing debounced file: {file_path}")
                self._debug(f"processing debounced file: {file_path}")
                self._emit_status_update()

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

                # Layer 3: Event deduplication to prevent redundant processing.
                file_key = str(file_path)
                current_time = time.time()

                if file_key in self._recent_file_events:
                    last_event_type, last_event_time = self._recent_file_events[
                        file_key
                    ]
                    if (
                        last_event_type == event_type
                        and (current_time - last_event_time)
                        < self._EVENT_DEDUP_WINDOW_SECONDS
                    ):
                        logger.debug(
                            "Suppressing duplicate "
                            f"{event_type} event for {file_path} "
                            f"(within {self._EVENT_DEDUP_WINDOW_SECONDS}s window)"
                        )
                        self._debug(f"suppressed duplicate {event_type}: {file_path}")
                        self.event_queue.task_done()
                        continue

                # Record this event
                self._recent_file_events[file_key] = (event_type, current_time)

                # Cleanup old entries to keep dict bounded (max 1000 files)
                if len(self._recent_file_events) > 1000:
                    cutoff = current_time - self._EVENT_HISTORY_RETENTION_SECONDS
                    self._recent_file_events = {
                        k: v
                        for k, v in self._recent_file_events.items()
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
                self._set_error(f"Error consuming realtime event: {e}")
                await asyncio.sleep(0.1)  # Brief pause on error

    async def remove_file(self, file_path: Path) -> None:
        """Remove file from database."""
        try:
            logger.debug(f"Removing file from database: {file_path}")
            self.services.provider.delete_file_completely(str(file_path))
            self._debug(f"removed file from database: {file_path}")
            self._emit_status_update()
        except Exception as e:
            logger.error(f"Error removing file {file_path}: {e}")
            self._set_error(f"Error removing file {file_path}: {e}")

    async def _add_directory_watch(self, dir_path: str) -> None:
        """Add a new directory to monitoring with recursive watching."""
        async with self.watch_lock:
            if dir_path not in self.watched_directories:
                if self.observer and self.event_handler:
                    self.observer.schedule(
                        self.event_handler,
                        dir_path,
                        recursive=True,  # Keep new directories recursively covered.
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
                "Cleaned up "
                f"{len(search_results)} files from deleted directory: {dir_path}"
            )

        except Exception as e:
            logger.error(f"Error cleaning up deleted directory {dir_path}: {e}")
            self._set_error(f"Error cleaning up deleted directory {dir_path}: {e}")

    async def _index_directory(self, dir_path: Path) -> None:
        """Index files in a newly created directory."""
        try:
            supported_files = await asyncio.to_thread(
                self._collect_supported_files, dir_path
            )

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
            self._set_error(f"Error indexing new directory {dir_path}: {e}")

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

                # Fast path for embedding generation without re-parsing the file.
                if priority == "embed":
                    try:
                        indexing_coordinator = self.services.indexing_coordinator
                        await indexing_coordinator.generate_missing_embeddings()
                    except Exception as e:
                        logger.warning(
                            f"Embedding generation failed in realtime (embed pass): {e}"
                        )
                        self._set_warning(
                            f"Embedding generation failed in realtime embed pass: {e}"
                        )
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

                # Clear event dedup entry so future modifications aren't suppressed
                self._recent_file_events.pop(str(file_path), None)

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
                        f"skip_embeddings={skip_embeddings} "
                        f"chunks={chunks} embeddings={embeds}"
                    )
                except Exception:
                    pass
                self._emit_status_update()

            except asyncio.CancelledError:
                logger.debug("Processing loop cancelled")
                raise
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                # Track failed files for debugging and monitoring
                self.failed_files.add(str(file_path))
                self._set_error(f"Error processing {file_path}: {e}")
                # Continue processing other files

    async def get_health(self) -> dict[str, Any]:
        """Return the richer backend-neutral realtime health snapshot."""
        status = self._build_health_snapshot()
        status["scan_complete"] = self.scan_complete
        return status

    async def get_stats(self) -> dict[str, Any]:
        """Get current service statistics."""
        return await self.get_health()

    async def wait_for_monitoring_ready(self, timeout: float = 10.0) -> bool:
        """Wait for filesystem monitoring to be ready."""
        try:
            await asyncio.wait_for(self.monitoring_ready.wait(), timeout=timeout)
            logger.debug("Monitoring became ready after setup")
            return True
        except asyncio.TimeoutError:
            logger.warning(f"Monitoring not ready after {timeout}s")
            self._set_warning(f"Monitoring not ready after {timeout}s")
            return False
