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
import copy
import gc
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from loguru import logger
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from chunkhound.core.config.config import Config
from chunkhound.core.utils.path_utils import normalize_realtime_path
from chunkhound.database_factory import DatabaseServices
from chunkhound.providers.database.duckdb_provider import (
    DuckDBTransactionConflictError,
)
from chunkhound.services.realtime_path_filter import RealtimePathFilter
from chunkhound.utils.windows_constants import IS_WINDOWS
from chunkhound.watchman import (
    PrivateWatchmanSidecar,
    WatchmanCliSession,
    WatchmanScopePlan,
    WatchmanSubscriptionScope,
    build_watchman_scope_plan,
    discover_nested_linux_mount_roots,
    discover_nested_windows_junction_scopes,
)
from chunkhound.watchman_runtime.loader import (
    default_realtime_backend_for_current_install,
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


class RealtimeStartupStatusTracker:
    """Track bounded daemon-side startup timing for public status surfaces."""

    _PHASE_NAMES = (
        "initialize",
        "db_connect",
        "realtime_start",
        "startup_barrier",
        "daemon_publish",
        "watchman_sidecar_start",
        "watchman_watch_project",
        "watchman_scope_discovery",
        "watchman_subscription_setup",
        "watchdog_setup",
        "polling_setup",
    )

    def __init__(
        self,
        mode: str = "stdio",
        debug_sink: Callable[[str], None] | None = None,
    ) -> None:
        self._debug_sink = debug_sink
        self.reset(mode)

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    @staticmethod
    def _default_phase_snapshot() -> dict[str, Any]:
        return {
            "state": "uninitialized",
            "started_at": None,
            "completed_at": None,
            "duration_seconds": None,
        }

    @staticmethod
    def _duration_seconds(
        start_monotonic: float | None,
        end_monotonic: float | None,
    ) -> float | None:
        if start_monotonic is None or end_monotonic is None:
            return None
        return round(max(end_monotonic - start_monotonic, 0.0), 3)

    @classmethod
    def default_snapshot(cls, mode: str = "stdio") -> dict[str, Any]:
        normalized_mode = mode if mode in {"daemon", "stdio"} else "stdio"
        return {
            "state": "uninitialized",
            "mode": normalized_mode,
            "started_at": None,
            "completed_at": None,
            "exposure_ready_at": None,
            "total_duration_seconds": None,
            "current_phase": None,
            "last_error": None,
            "phases": {
                phase_name: cls._default_phase_snapshot()
                for phase_name in cls._PHASE_NAMES
            },
        }

    def reset(self, mode: str = "stdio") -> None:
        self._snapshot = self.default_snapshot(mode)
        self._startup_started_monotonic: float | None = None
        self._phase_started_monotonic: dict[str, float] = {}
        self._phase_stack: list[str] = []

    def set_debug_sink(self, debug_sink: Callable[[str], None] | None) -> None:
        self._debug_sink = debug_sink

    def _log(self, message: str) -> None:
        if self._debug_sink is None:
            return
        try:
            self._debug_sink(f"startup: {message}")
        except Exception:
            pass

    def _ensure_started(self) -> None:
        if self._snapshot["state"] != "uninitialized":
            return
        self._snapshot["state"] = "running"
        self._snapshot["started_at"] = self._utc_now()
        self._snapshot["completed_at"] = None
        self._snapshot["total_duration_seconds"] = None
        self._snapshot["current_phase"] = None
        self._snapshot["last_error"] = None
        self._startup_started_monotonic = time.monotonic()
        self._log(f"startup tracking began mode={self._snapshot['mode']}")

    def start_phase(self, phase_name: str) -> None:
        if phase_name not in self._snapshot["phases"]:
            return
        if self._snapshot["state"] in {"completed", "failed"}:
            return
        self._ensure_started()
        phase = self._snapshot["phases"][phase_name]
        if phase["state"] == "running":
            return
        phase["state"] = "running"
        phase["started_at"] = self._utc_now()
        phase["completed_at"] = None
        phase["duration_seconds"] = None
        self._phase_started_monotonic[phase_name] = time.monotonic()
        self._phase_stack = [name for name in self._phase_stack if name != phase_name]
        self._phase_stack.append(phase_name)
        self._snapshot["current_phase"] = phase_name
        self._log(f"phase started: {phase_name}")

    def _close_phase(
        self, phase_name: str, state: str, error: str | None = None
    ) -> None:
        if phase_name not in self._snapshot["phases"]:
            return
        phase = self._snapshot["phases"][phase_name]
        if phase["state"] not in {"running", "uninitialized"}:
            return
        if phase["state"] == "uninitialized":
            self.start_phase(phase_name)
            phase = self._snapshot["phases"][phase_name]
        completed_at = self._utc_now()
        end_monotonic = time.monotonic()
        started_monotonic = self._phase_started_monotonic.pop(phase_name, None)
        phase["state"] = state
        phase["completed_at"] = completed_at
        phase["duration_seconds"] = self._duration_seconds(
            started_monotonic,
            end_monotonic,
        )
        self._phase_stack = [name for name in self._phase_stack if name != phase_name]
        self._snapshot["current_phase"] = (
            self._phase_stack[-1] if self._phase_stack else None
        )
        if state == "failed":
            self._log(
                "phase failed: "
                f"{phase_name} duration={phase['duration_seconds']}s "
                f"error={error}"
            )
        else:
            self._log(
                f"phase completed: {phase_name} duration={phase['duration_seconds']}s"
            )

    def complete_phase(self, phase_name: str) -> None:
        self._close_phase(phase_name, "completed")

    def fail_phase(self, phase_name: str, error: str) -> None:
        self._close_phase(phase_name, "failed", error)

    def fail_startup(self, error: str, *, phase_name: str | None = None) -> None:
        if phase_name:
            self.fail_phase(phase_name, error)
        if self._snapshot["state"] == "completed":
            return
        self._ensure_started()
        self._snapshot["state"] = "failed"
        self._snapshot["completed_at"] = self._utc_now()
        self._snapshot["last_error"] = error
        self._snapshot["total_duration_seconds"] = self._duration_seconds(
            self._startup_started_monotonic,
            time.monotonic(),
        )
        self._phase_stack = []
        self._snapshot["current_phase"] = phase_name
        self._log(
            "startup failed"
            f" duration={self._snapshot['total_duration_seconds']}s error={error}"
        )

    def complete_startup(self) -> None:
        if self._snapshot["state"] in {"completed", "failed"}:
            return
        self._ensure_started()
        self._snapshot["state"] = "completed"
        self._snapshot["completed_at"] = self._utc_now()
        self._snapshot["total_duration_seconds"] = self._duration_seconds(
            self._startup_started_monotonic,
            time.monotonic(),
        )
        self._phase_stack = []
        self._snapshot["current_phase"] = None
        self._log(
            f"startup completed duration={self._snapshot['total_duration_seconds']}s"
        )

    def mark_exposure_ready(self) -> None:
        if self._snapshot["mode"] != "daemon":
            return
        if self._snapshot["exposure_ready_at"] is None:
            self._snapshot["exposure_ready_at"] = self._utc_now()
            self._log("daemon exposure became ready")

    def snapshot(self) -> dict[str, Any]:
        snapshot = copy.deepcopy(self._snapshot)
        if self._snapshot["state"] == "running":
            snapshot["total_duration_seconds"] = self._duration_seconds(
                self._startup_started_monotonic,
                time.monotonic(),
            )
        for phase_name, phase in snapshot["phases"].items():
            if phase["state"] != "running":
                continue
            phase["duration_seconds"] = self._duration_seconds(
                self._phase_started_monotonic.get(phase_name),
                time.monotonic(),
            )
        return snapshot


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


class SimpleEventHandler(FileSystemEventHandler):
    """Simple sync event handler - no async complexity."""

    def __init__(
        self,
        event_queue: asyncio.Queue[tuple[str, Path]] | None,
        config: Config | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        root_path: Path | None = None,
        queue_result_callback: QueueResultCallback | None = None,
        source_event_callback: Callable[[str, Path], None] | None = None,
        filtered_event_callback: Callable[[str, Path], None] | None = None,
    ):
        self.event_queue = event_queue
        self.config = config
        self.loop = loop
        self._queue_result_callback = queue_result_callback
        self._source_event_callback = source_event_callback
        self._filtered_event_callback = filtered_event_callback
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
            file_path = Path(normalize_file_path(event.src_path))
            self._record_source_event("dir_created", file_path)
            self._queue_event("dir_created", file_path)
            return

        # Handle directory deletion
        if event.event_type == "deleted" and event.is_directory:
            # Queue directory deletion for cleanup
            file_path = Path(normalize_file_path(event.src_path))
            self._record_source_event("dir_deleted", file_path)
            self._queue_event("dir_deleted", file_path)
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
        self._record_source_event(event.event_type, file_path)

        # Simple filtering for supported file types
        if not self._should_index(file_path):
            self._record_filtered_event(event.event_type, file_path)
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
            self._record_source_event("created", dest_file)
            self._queue_event("created", dest_file)

        # If moving FROM supported file -> handle as deletion + creation
        elif self._should_index(src_file) and self._should_index(dest_file):
            logger.debug(f"File rename: {src_path} -> {dest_path}")
            self._record_source_event("deleted", src_file)
            self._queue_event("deleted", src_file)
            self._record_source_event("created", dest_file)
            self._queue_event("created", dest_file)

        # If moving FROM supported file TO temp/unsupported -> deletion
        elif self._should_index(src_file) and not self._should_index(dest_file):
            logger.debug(f"File moved to temp/unsupported: {src_path}")
            self._record_source_event("deleted", src_file)
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


class RealtimeMonitorAdapter(Protocol):
    """Backend-specific filesystem monitoring lifecycle."""

    backend_name: str

    async def start(
        self, watch_path: Path, loop: asyncio.AbstractEventLoop
    ) -> None: ...

    async def stop(self) -> None: ...

    def get_health(self) -> dict[str, Any]: ...


def _default_watchman_loss_of_sync_snapshot() -> dict[str, Any]:
    """Return the stable Watchman loss-of-sync status payload."""
    return {
        "count": 0,
        "fresh_instance_count": 0,
        "recrawl_count": 0,
        "disconnect_count": 0,
        "last_reason": None,
        "last_at": None,
        "last_details": None,
    }


def _default_watchman_reconnect_snapshot() -> dict[str, Any]:
    """Return the stable Watchman reconnect status payload."""
    return {
        "state": "idle",
        "attempt_count": 0,
        "max_attempts": WatchmanRealtimeAdapter._RECONNECT_MAX_ATTEMPTS,
        "retry_delay_seconds": WatchmanRealtimeAdapter._RECONNECT_RETRY_DELAY_SECONDS,
        "last_started_at": None,
        "last_completed_at": None,
        "last_error": None,
        "last_result": None,
    }


def _default_watchman_health_snapshot() -> dict[str, Any]:
    """Return Watchman-specific realtime fields for daemon status surfaces."""
    return {
        "watchman_pid": None,
        "watchman_started_at": None,
        "watchman_process_start_time_epoch": None,
        "watchman_runtime_version": None,
        "watchman_binary_path": None,
        "watchman_socket_path": None,
        "watchman_statefile_path": None,
        "watchman_logfile_path": None,
        "watchman_metadata_path": None,
        "watchman_alive": False,
        "watchman_sidecar_state": "uninitialized",
        "watchman_connection_state": "uninitialized",
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
        "watchman_subscription_names": [],
        "watchman_subscription_count": 0,
        "watchman_watch_root": None,
        "watchman_relative_root": None,
        "watchman_scopes": [],
        "watchman_session_capabilities": {},
        "watchman_loss_of_sync": _default_watchman_loss_of_sync_snapshot(),
        "watchman_reconnect": _default_watchman_reconnect_snapshot(),
    }


class WatchdogRealtimeAdapter:
    """Watchdog-backed monitor with polling fallback."""

    backend_name = "watchdog"

    def __init__(self, service: "RealtimeIndexingService") -> None:
        self._service = service

    async def start(self, watch_path: Path, loop: asyncio.AbstractEventLoop) -> None:
        self._service._set_effective_backend(self.backend_name)
        self._service._start_startup_phase("watchdog_setup")
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
        self._service._start_startup_phase("polling_setup")
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
    _RECONNECT_MAX_ATTEMPTS = 3
    _RECONNECT_RETRY_DELAY_SECONDS = 1.0

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
        self._sessions: list[WatchmanCliSession] = []
        self._path_filter: RealtimePathFilter | None = None
        self._shared_subscription_queue: asyncio.Queue[dict[str, object]] | None = None
        self._subscription_consumer_task: asyncio.Task[None] | None = None
        self._subscription_bridge_tasks: list[asyncio.Task[None]] = []
        self._session_monitor_task: asyncio.Task[None] | None = None
        self._subscription_scope_map: dict[str, WatchmanSubscriptionScope] = {}
        self._reconnect_task: asyncio.Task[None] | None = None
        self._watch_path: Path | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loss_of_sync_count = 0
        self._fresh_instance_count = 0
        self._recrawl_count = 0
        self._disconnect_count = 0
        self._last_loss_of_sync_reason: str | None = None
        self._last_loss_of_sync_at: str | None = None
        self._last_loss_of_sync_details: dict[str, object] | None = None
        self._reconnect_state = "idle"
        self._reconnect_attempt_count = 0
        self._last_reconnect_started_at: str | None = None
        self._last_reconnect_completed_at: str | None = None
        self._last_reconnect_error: str | None = None
        self._last_reconnect_result: str | None = None

    async def start(self, watch_path: Path, loop: asyncio.AbstractEventLoop) -> None:
        self._watch_path = watch_path
        self._loop = loop
        self._reset_loss_of_sync_state()
        self._reset_reconnect_state()
        try:
            await self._establish_monitoring(watch_path, loop, phase="startup")
        except Exception as error:
            message = str(error)
            self._service._set_error(message)
            raise RuntimeError(message) from error

    async def stop(self) -> None:
        self._watch_path = None
        self._loop = None
        if self._reconnect_task is not None:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
            self._reconnect_task = None
        await self._clear_monitoring_runtime(stop_sidecar=True)

    async def _consume_subscription_pdus(self) -> None:
        subscription_queue = self._shared_subscription_queue
        if subscription_queue is None:
            return

        while True:
            payload = await subscription_queue.get()
            try:
                if self._handle_loss_of_sync_payload(payload):
                    continue
                scope = self._scope_for_payload(payload)
                if scope is None:
                    self._warn_translation_issue(
                        "Watchman subscription PDU did not map to a known scope"
                    )
                    continue
                self._translate_subscription_pdu(payload, scope)
            except Exception as error:
                message = f"Watchman event translation failed: {error}"
                logger.warning(message)
                self._service._record_translation_error()
                self._service._set_warning(message)
            finally:
                subscription_queue.task_done()

    def _scope_for_payload(
        self, payload: dict[str, object]
    ) -> WatchmanSubscriptionScope | None:
        subscription_name = payload.get("subscription")
        scope = self._subscription_scope_map.get(
            subscription_name if isinstance(subscription_name, str) else ""
        )
        if scope is not None:
            return scope
        if self._service.watchman_scope_plan is None:
            return None
        if len(self._service.watchman_scope_plan.scopes) == 1:
            return self._service.watchman_scope_plan.primary_scope
        return None

    async def _bridge_session_subscription_pdus(
        self, session: WatchmanCliSession
    ) -> None:
        shared_queue = self._shared_subscription_queue
        if shared_queue is None:
            return

        while True:
            payload = await session.subscription_queue.get()
            try:
                shared_queue.put_nowait(payload)
            except asyncio.QueueFull:
                self._handle_subscription_queue_overflow(
                    payload,
                    shared_queue.qsize() + 1,
                    shared_queue.maxsize,
                )
            finally:
                session.subscription_queue.task_done()

    async def _monitor_unexpected_session_exits(self) -> None:
        sessions = tuple(self._sessions)
        if not sessions:
            return

        wait_tasks = {
            asyncio.create_task(session.wait_for_unexpected_exit()): session
            for session in sessions
        }
        message: str | None = None
        try:
            done, pending = await asyncio.wait(
                wait_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                message = task.result()
                if message is not None:
                    break
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        finally:
            wait_tasks.clear()

        if message is None:
            return

        if self._session_monitor_task is asyncio.current_task():
            self._session_monitor_task = None
        await self._cancel_subscription_consumer_task()
        await self._cancel_subscription_bridge_tasks()
        self._path_filter = None
        self._service.watchman_scope_plan = None
        self._service.watchman_subscription_queue = None
        self._service.monitoring_ready.clear()
        self._service._monitoring_ready_at = None

        sidecar_health = self._sidecar.get_health()
        details = {
            "backend": "watchman",
            "loss_of_sync_reason": "disconnect",
            "watchman_session_alive": False,
            "watchman_alive": bool(sidecar_health.get("watchman_alive")),
            "watchman_session_error": message,
        }
        self._record_loss_of_sync(
            "disconnect",
            message=f"Watchman session disconnected: {message}",
            details=details,
            as_error=True,
        )
        self._begin_reconnect_cycle()

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
            self._service._record_source_event(event_type, file_path)
            if not path_filter.should_index(file_path):
                self._service._record_filtered_event(event_type, file_path)
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

        # Preserve the logical handled path under config.target_dir even when
        # the Watchman watch root is a physical junction target outside it.
        canonical_path = scope.requested_path.joinpath(*relative_name.parts)
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
        self._service._record_translation_error()
        self._service._set_warning(warning)

    def _reset_loss_of_sync_state(self) -> None:
        self._loss_of_sync_count = 0
        self._fresh_instance_count = 0
        self._recrawl_count = 0
        self._disconnect_count = 0
        self._last_loss_of_sync_reason = None
        self._last_loss_of_sync_at = None
        self._last_loss_of_sync_details = None

    def _reset_reconnect_state(self) -> None:
        self._reconnect_state = "idle"
        self._reconnect_attempt_count = 0
        self._last_reconnect_started_at = None
        self._last_reconnect_completed_at = None
        self._last_reconnect_error = None
        self._last_reconnect_result = None

    async def _cancel_subscription_consumer_task(self) -> None:
        task = self._subscription_consumer_task
        if task is None:
            return
        self._subscription_consumer_task = None
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _cancel_subscription_bridge_tasks(self) -> None:
        tasks = list(self._subscription_bridge_tasks)
        self._subscription_bridge_tasks = []
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _cancel_session_monitor_task(self) -> None:
        task = self._session_monitor_task
        if task is None:
            return
        if task is asyncio.current_task():
            self._session_monitor_task = None
            return
        self._session_monitor_task = None
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _clear_monitoring_runtime(self, *, stop_sidecar: bool) -> None:
        self._service.watchman_scope_plan = None
        self._service.watchman_subscription_queue = None
        self._path_filter = None
        self._shared_subscription_queue = None
        self._subscription_scope_map = {}
        self._service.monitoring_ready.clear()
        self._service._monitoring_ready_at = None
        await self._cancel_session_monitor_task()
        await self._cancel_subscription_consumer_task()
        await self._cancel_subscription_bridge_tasks()
        sessions = self._sessions
        primary_session = self._session
        cleanup_complete = False
        try:
            for session in list(sessions):
                await session.stop()
            if stop_sidecar:
                await self._sidecar.stop()
            cleanup_complete = True
        finally:
            # If teardown is cancelled mid-flight, keep the live adapter-owned
            # session handles attached so a follow-up stop() call can still
            # terminate the same CLI processes safely.
            if cleanup_complete:
                if self._sessions is sessions:
                    self._sessions = []
                if self._session is primary_session:
                    self._session = None
            self._service._emit_status_update()

    async def _establish_monitoring(
        self,
        watch_path: Path,
        loop: asyncio.AbstractEventLoop,
        *,
        phase: str,
    ) -> None:
        self._service._start_startup_phase("watchman_sidecar_start")
        try:
            metadata = await self._sidecar.start()
        except Exception as error:
            self._service._fail_startup_phase(
                "watchman_sidecar_start",
                f"Watchman sidecar {phase} failed: {error}",
            )
            raise RuntimeError(f"Watchman sidecar {phase} failed: {error}") from error
        self._service._complete_startup_phase("watchman_sidecar_start")

        self._service.watchman_scope_plan = None
        self._service.watchman_subscription_queue = None

        planning_session = WatchmanCliSession(
            binary_path=Path(metadata.binary_path),
            socket_path=self._sidecar.paths.listener_path,
            statefile_path=self._sidecar.paths.statefile_path,
            logfile_path=self._sidecar.paths.logfile_path,
            pidfile_path=self._sidecar.paths.pidfile_path,
            project_root=self._sidecar.paths.project_root,
            debug_sink=self._service._debug,
            subscription_overflow_handler=self._handle_subscription_queue_overflow,
        )
        sessions: list[WatchmanCliSession] = []
        subscription_scope_map: dict[str, WatchmanSubscriptionScope] = {}
        try:
            self._service._start_startup_phase("watchman_watch_project")
            watch_project_response = await planning_session._run_one_shot_command(
                ["watch-project", str(watch_path.resolve())]
            )
            self._service._complete_startup_phase("watchman_watch_project")
            self._service._start_startup_phase("watchman_scope_discovery")
            nested_mount_roots = discover_nested_linux_mount_roots(watch_path)
            additional_scopes = discover_nested_windows_junction_scopes(watch_path)
            watched_roots: set[Path] = set()
            for mount_root in nested_mount_roots:
                if mount_root in watched_roots:
                    continue
                watched_roots.add(mount_root)
                await planning_session._run_one_shot_command(["watch", str(mount_root)])
            for extra_scope in additional_scopes:
                if extra_scope.watch_root in watched_roots:
                    continue
                watched_roots.add(extra_scope.watch_root)
                await planning_session._run_one_shot_command(
                    ["watch", str(extra_scope.watch_root)]
                )
            scope_plan = build_watchman_scope_plan(
                watch_path,
                watch_project_response,
                nested_mount_roots=nested_mount_roots,
                additional_scopes=additional_scopes,
            )
            self._service._complete_startup_phase("watchman_scope_discovery")

            self._service._start_startup_phase("watchman_subscription_setup")
            self._shared_subscription_queue = asyncio.Queue(
                maxsize=WatchmanCliSession._SUBSCRIPTION_QUEUE_MAXSIZE
            )
            self._subscription_scope_map = {}

            for scope_index, scope in enumerate(scope_plan.scopes):
                session = WatchmanCliSession(
                    binary_path=Path(metadata.binary_path),
                    socket_path=self._sidecar.paths.listener_path,
                    statefile_path=self._sidecar.paths.statefile_path,
                    logfile_path=self._sidecar.paths.logfile_path,
                    pidfile_path=self._sidecar.paths.pidfile_path,
                    project_root=self._sidecar.paths.project_root,
                    debug_sink=self._service._debug,
                    subscription_overflow_handler=self._handle_subscription_queue_overflow,
                )
                scoped_subscription_name = self._subscription_name_for_scope(
                    base_name=self._SUBSCRIPTION_NAME,
                    target_path=watch_path,
                    scope=scope,
                    scope_index=scope_index,
                )
                single_scope_plan = WatchmanScopePlan(scopes=(scope,))
                await session.start(
                    target_path=scope.requested_path,
                    subscription_name=scoped_subscription_name,
                    scope_plan=single_scope_plan,
                    nested_mount_roots=(),
                )
                sessions.append(session)
                subscription_scope_map[scoped_subscription_name] = scope
        except Exception as error:
            current_phase = self._service._startup_tracker.snapshot().get(
                "current_phase"
            )
            if isinstance(current_phase, str):
                self._service._fail_startup_phase(current_phase, str(error))
            for session in sessions:
                await session.stop()
            await self._sidecar.stop()
            self._service.watchman_scope_plan = None
            self._service.watchman_subscription_queue = None
            self._shared_subscription_queue = None
            self._subscription_scope_map = {}
            raise RuntimeError(f"Watchman session {phase} failed: {error}") from error

        self._sessions = sessions
        self._session = sessions[0] if sessions else None
        self._subscription_scope_map = dict(subscription_scope_map)
        self._service.watchman_scope_plan = scope_plan
        self._service.watchman_subscription_queue = self._shared_subscription_queue
        self._path_filter = RealtimePathFilter(
            config=self._service.config,
            root_path=watch_path,
        )
        self._subscription_bridge_tasks = [
            loop.create_task(self._bridge_session_subscription_pdus(session))
            for session in sessions
        ]
        self._subscription_consumer_task = loop.create_task(
            self._consume_subscription_pdus()
        )
        self._session_monitor_task = loop.create_task(
            self._monitor_unexpected_session_exits()
        )
        self._service._set_effective_backend(self.backend_name)
        self._service._monitoring_ready_at = self._service._utc_now()
        self._service.monitoring_ready.set()
        self._service._complete_startup_phase("watchman_subscription_setup")
        self._service._emit_status_update()

    def _begin_reconnect_cycle(self) -> None:
        if self._reconnect_task is not None and not self._reconnect_task.done():
            return
        if self._watch_path is None or self._loop is None:
            return
        self._reconnect_state = "pending"
        self._reconnect_attempt_count = 0
        self._last_reconnect_started_at = None
        self._last_reconnect_completed_at = None
        self._last_reconnect_error = None
        self._last_reconnect_result = None
        self._service._emit_status_update()
        self._reconnect_task = asyncio.create_task(self._run_reconnect_loop())

    async def _run_reconnect_loop(self) -> None:
        watch_path = self._watch_path
        loop = self._loop
        if watch_path is None or loop is None:
            return

        try:
            for attempt in range(1, self._RECONNECT_MAX_ATTEMPTS + 1):
                if attempt > 1:
                    self._reconnect_state = "pending"
                    self._service._emit_status_update()
                    await asyncio.sleep(self._RECONNECT_RETRY_DELAY_SECONDS)

                self._reconnect_state = "running"
                self._reconnect_attempt_count = attempt
                self._last_reconnect_started_at = self._service._utc_now()
                self._last_reconnect_error = None
                self._last_reconnect_result = None
                self._service._emit_status_update()

                try:
                    await self._clear_monitoring_runtime(stop_sidecar=False)
                    await self._establish_monitoring(
                        watch_path,
                        loop,
                        phase="reconnect",
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    self._last_reconnect_error = str(error)
                    self._last_reconnect_completed_at = self._service._utc_now()
                    if attempt < self._RECONNECT_MAX_ATTEMPTS:
                        self._service._set_warning(
                            "Watchman reconnect attempt "
                            f"{attempt}/{self._RECONNECT_MAX_ATTEMPTS} failed: "
                            f"{self._last_reconnect_error}"
                        )
                        continue
                    self._reconnect_state = "failed"
                    self._last_reconnect_result = "failed"
                    self._service._set_error(
                        f"Watchman reconnect failed: {self._last_reconnect_error}"
                    )
                    self._service._emit_status_update()
                    return

                self._reconnect_state = "restored"
                self._last_reconnect_completed_at = self._service._utc_now()
                self._last_reconnect_result = "restored"
                self._service._clear_error_state(
                    prefixes=(
                        "Watchman session disconnected:",
                        "Watchman reconnect failed:",
                    )
                )
                self._service._refresh_runtime_service_state()
                self._service._emit_status_update()
                return
        finally:
            self._reconnect_task = None

    def _handle_loss_of_sync_payload(self, payload: dict[str, object]) -> bool:
        reason: str | None = None
        message: str | None = None

        if (
            payload.get("is_fresh_instance") is True
            or payload.get("fresh_instance") is True
        ):
            reason = "fresh_instance"
            message = (
                "Watchman reported a fresh instance; scheduling a reconciliation resync"
            )
        else:
            warning = payload.get("warning")
            if isinstance(warning, str) and "recrawl" in warning.lower():
                reason = "recrawl"
                message = (
                    "Watchman reported a recrawl warning; "
                    "scheduling a reconciliation resync"
                )

        if reason is None:
            return False

        details: dict[str, object] = {
            "backend": "watchman",
            "loss_of_sync_reason": reason,
            "subscription": str(payload.get("subscription") or self._SUBSCRIPTION_NAME),
        }
        clock = payload.get("clock")
        if isinstance(clock, str) and clock:
            details["clock"] = clock
        warning = payload.get("warning")
        if isinstance(warning, str) and warning:
            details["warning"] = warning

        self._record_loss_of_sync(reason, message=message, details=details)
        return True

    def _handle_subscription_queue_overflow(
        self,
        payload: dict[str, object],
        dropped_count: int,
        queue_maxsize: int,
    ) -> None:
        if self._service._needs_resync:
            self._service._emit_status_update()
            return

        details: dict[str, object] = {
            "backend": "watchman",
            "loss_of_sync_reason": "subscription_pdu_dropped",
            "subscription": str(payload.get("subscription") or self._SUBSCRIPTION_NAME),
            "watchman_subscription_pdu_dropped": dropped_count,
            "watchman_subscription_queue_maxsize": queue_maxsize,
        }
        clock = payload.get("clock")
        if isinstance(clock, str) and clock:
            details["clock"] = clock

        self._record_loss_of_sync(
            "subscription_pdu_dropped",
            message=(
                "Watchman subscription queue overflowed; "
                "scheduling a reconciliation resync"
            ),
            details=details,
        )

    def _record_loss_of_sync(
        self,
        reason: str,
        *,
        message: str | None = None,
        details: dict[str, object] | None = None,
        as_error: bool = False,
    ) -> None:
        self._loss_of_sync_count += 1
        if reason == "fresh_instance":
            self._fresh_instance_count += 1
        elif reason == "recrawl":
            self._recrawl_count += 1
        elif reason == "disconnect":
            self._disconnect_count += 1

        self._last_loss_of_sync_reason = reason
        self._last_loss_of_sync_at = self._service._utc_now()
        self._last_loss_of_sync_details = dict(details) if details else None

        if message:
            if as_error:
                self._service._set_error(message)
            else:
                self._service._set_warning(message)
        else:
            self._service._emit_status_update()

        self._schedule_resync_request(reason, details)

    def _schedule_resync_request(
        self, reason: str, details: dict[str, object] | None = None
    ) -> None:
        async def _dispatch() -> None:
            try:
                await self._service.request_resync("realtime_loss_of_sync", details)
            except Exception as error:
                self._service._set_error(f"Watchman resync request failed: {error}")

        asyncio.create_task(_dispatch())

    @staticmethod
    def _latest_session_value(
        session_healths: list[dict[str, Any]],
        value_key: str,
        timestamp_key: str,
    ) -> tuple[Any, str | None]:
        latest_value: Any = None
        latest_timestamp: str | None = None
        for health in session_healths:
            timestamp = health.get(timestamp_key)
            value = health.get(value_key)
            if not isinstance(timestamp, str) or not timestamp:
                continue
            if latest_timestamp is None or timestamp > latest_timestamp:
                latest_timestamp = timestamp
                latest_value = value
        return latest_value, latest_timestamp

    def get_health(self) -> dict[str, Any]:
        health = _default_watchman_health_snapshot()
        health.update(self._sidecar.get_health())
        session_healths = [session.get_health() for session in self._sessions]
        if not session_healths and self._session is not None:
            session_healths = [self._session.get_health()]
        if session_healths:
            primary_session_health = session_healths[0]
            warning, warning_at = self._latest_session_value(
                session_healths,
                "watchman_session_last_warning",
                "watchman_session_last_warning_at",
            )
            error, error_at = self._latest_session_value(
                session_healths,
                "watchman_session_last_error",
                "watchman_session_last_error_at",
            )
            last_response_at, _ = self._latest_session_value(
                session_healths,
                "watchman_session_last_response_at",
                "watchman_session_last_response_at",
            )
            last_subscription_at, _ = self._latest_session_value(
                session_healths,
                "watchman_subscription_last_received_at",
                "watchman_subscription_last_received_at",
            )
            health.update(
                {
                    "watchman_session_alive": all(
                        bool(item.get("watchman_session_alive"))
                        for item in session_healths
                    ),
                    "watchman_session_pid": primary_session_health.get(
                        "watchman_session_pid"
                    ),
                    "watchman_session_last_warning": warning,
                    "watchman_session_last_warning_at": warning_at,
                    "watchman_session_last_error": error,
                    "watchman_session_last_error_at": error_at,
                    "watchman_session_last_response_at": last_response_at,
                    "watchman_subscription_last_received_at": last_subscription_at,
                    "watchman_session_command_count": sum(
                        int(item.get("watchman_session_command_count") or 0)
                        for item in session_healths
                    ),
                    "watchman_subscription_queue_size": (
                        self._shared_subscription_queue.qsize()
                        if self._shared_subscription_queue is not None
                        else sum(
                            int(item.get("watchman_subscription_queue_size") or 0)
                            for item in session_healths
                        )
                    ),
                    "watchman_subscription_queue_maxsize": (
                        self._shared_subscription_queue.maxsize
                        if self._shared_subscription_queue is not None
                        else int(
                            primary_session_health.get(
                                "watchman_subscription_queue_maxsize"
                            )
                            or 0
                        )
                    ),
                    "watchman_subscription_pdu_count": sum(
                        int(item.get("watchman_subscription_pdu_count") or 0)
                        for item in session_healths
                    ),
                    "watchman_subscription_pdu_dropped": sum(
                        int(item.get("watchman_subscription_pdu_dropped") or 0)
                        for item in session_healths
                    ),
                    "watchman_subscription_name": primary_session_health.get(
                        "watchman_subscription_name"
                    ),
                    "watchman_subscription_names": [
                        str(name)
                        for item in session_healths
                        for name in item.get("watchman_subscription_names", [])
                        if isinstance(name, str)
                    ],
                    "watchman_watch_root": primary_session_health.get(
                        "watchman_watch_root"
                    ),
                    "watchman_relative_root": primary_session_health.get(
                        "watchman_relative_root"
                    ),
                    "watchman_scopes": [
                        scope
                        for item in session_healths
                        for scope in item.get("watchman_scopes", [])
                        if isinstance(scope, dict)
                    ],
                    "watchman_session_capabilities": dict(
                        primary_session_health.get("watchman_session_capabilities")
                        or {}
                    ),
                }
            )
        sidecar_alive = bool(health.get("watchman_alive"))
        session_alive = bool(health.get("watchman_session_alive"))
        health["watchman_sidecar_state"] = "running" if sidecar_alive else "stopped"
        if session_alive:
            health["watchman_connection_state"] = "connected"
        elif sidecar_alive:
            health["watchman_connection_state"] = "sidecar_only"
        else:
            health["watchman_connection_state"] = "disconnected"
        subscription_names = health.get("watchman_subscription_names")
        if isinstance(subscription_names, list):
            health["watchman_subscription_count"] = len(subscription_names)
        elif isinstance(health.get("watchman_scopes"), list):
            health["watchman_subscription_count"] = len(health["watchman_scopes"])
        elif health.get("watchman_subscription_name"):
            health["watchman_subscription_count"] = 1
        else:
            health["watchman_subscription_count"] = 0
        health["watchman_loss_of_sync"] = {
            "count": self._loss_of_sync_count,
            "fresh_instance_count": self._fresh_instance_count,
            "recrawl_count": self._recrawl_count,
            "disconnect_count": self._disconnect_count,
            "last_reason": self._last_loss_of_sync_reason,
            "last_at": self._last_loss_of_sync_at,
            "last_details": self._last_loss_of_sync_details,
        }
        health["watchman_reconnect"] = {
            "state": self._reconnect_state,
            "attempt_count": self._reconnect_attempt_count,
            "max_attempts": self._RECONNECT_MAX_ATTEMPTS,
            "retry_delay_seconds": self._RECONNECT_RETRY_DELAY_SECONDS,
            "last_started_at": self._last_reconnect_started_at,
            "last_completed_at": self._last_reconnect_completed_at,
            "last_error": self._last_reconnect_error,
            "last_result": self._last_reconnect_result,
        }
        health["observer_alive"] = sidecar_alive and session_alive
        return health

    def _subscription_name_for_scope(
        self,
        *,
        base_name: str,
        target_path: Path,
        scope: WatchmanSubscriptionScope,
        scope_index: int,
    ) -> str:
        if scope.scope_kind == "primary" or scope_index == 0:
            return base_name
        try:
            suffix_source = scope.requested_path.relative_to(target_path).as_posix()
        except ValueError:
            suffix_source = scope.requested_path.as_posix()
        suffix = WatchmanCliSession._sanitize_subscription_suffix(suffix_source)
        if not suffix:
            suffix = f"scope-{scope_index}"
        return f"{base_name}--{suffix}"


class RealtimeIndexingService:
    """Simple real-time indexing service with search responsiveness."""

    _PENDING_MUTATION_STATUS_OPERATIONS = (
        "change",
        "delete",
        "embed",
        "dir_delete",
        "dir_index",
    )
    # Event deduplication window - suppress duplicate events within this period
    _EVENT_DEDUP_WINDOW_SECONDS = 2.0
    # Retention period for event history - entries older than this are cleaned up
    _EVENT_HISTORY_RETENTION_SECONDS = 10.0
    _EVENT_QUEUE_MAXSIZE = 1000
    _RESYNC_DEBOUNCE_SECONDS = 1.0
    _STALL_THRESHOLD_SECONDS = 30.0
    _EVENT_PRESSURE_WINDOW_SECONDS = 30.0
    _EVENT_PRESSURE_MAX_TRACKED_PATHS = 64
    _EVENT_PRESSURE_ELEVATED_EVENTS = 20
    _EVENT_PRESSURE_OVERLOADED_EVENTS = 100
    _EVENT_PRESSURE_ELEVATED_COALESCED_UPDATES = 5
    _EVENT_PRESSURE_OVERLOADED_COALESCED_UPDATES = 20
    _WATCHDOG_SETUP_TIMEOUT_SECONDS = 5.0
    _MONITORING_READY_TIMEOUT_SECONDS = 10.0
    _POLLING_STARTUP_SETTLE_SECONDS = 0.5
    _DELETE_CONFLICT_MAX_RETRIES = 5
    _DELETE_CONFLICT_BASE_RETRY_DELAY_SECONDS = 0.1
    _MUTATION_PRIORITIES = {
        "delete": 0,
        "dir_delete": 0,
        "change": 1,
        "scan": 1,
        "dir_index": 1,
        "embed": 2,
    }

    def __init__(
        self,
        services: DatabaseServices,
        config: Config,
        debug_sink: Callable[[str], None] | None = None,
        startup_log_sink: Callable[[str], None] | None = None,
        status_callback: Callable[[dict[str, Any]], None] | None = None,
        resync_callback: Callable[
            [str, dict[str, Any] | None], Awaitable[dict[str, Any] | None]
        ]
        | None = None,
        startup_tracker: RealtimeStartupStatusTracker | None = None,
    ):
        self.services = services
        self.config = config
        # Optional sink that writes to MCPServerBase.debug_log so events land in
        # /tmp/chunkhound_mcp_debug.log when CHUNKHOUND_DEBUG is enabled.
        self._debug_sink = debug_sink
        self._status_callback = status_callback
        self._resync_callback = resync_callback
        resolved_startup_log_sink = startup_log_sink or debug_sink
        self._startup_tracker = startup_tracker or RealtimeStartupStatusTracker(
            debug_sink=resolved_startup_log_sink
        )
        self._startup_tracker.set_debug_sink(resolved_startup_log_sink)
        self._configured_backend = self._resolve_configured_backend()
        self._effective_backend = "uninitialized"
        self._monitor_adapter: RealtimeMonitorAdapter | None = None
        self.watchman_scope_plan: WatchmanScopePlan | None = None
        self.watchman_subscription_queue: asyncio.Queue[dict[str, object]] | None = None

        # Downstream mutation queue. Deletes, changes, and embed follow-ups all
        # converge here so DB-changing realtime work stays serialized.
        self.file_queue: asyncio.PriorityQueue[tuple[int, int, RealtimeMutation]] = (
            asyncio.PriorityQueue()
        )
        self._queue_sequence = 0
        self._next_mutation_id = 0

        # NEW: Async queue for events from watchdog (thread-safe via asyncio)
        self.event_queue: asyncio.Queue[tuple[str, Path]] = asyncio.Queue(
            maxsize=self._EVENT_QUEUE_MAXSIZE
        )

        # Deduplication and error tracking
        self.pending_files: set[Path] = set()
        self._pending_mutations: dict[tuple[str, str], RealtimeMutation] = {}
        self._pending_path_counts: dict[str, int] = {}
        self.failed_files: set[str] = set()
        self._last_warning: str | None = None
        self._last_warning_at: str | None = None
        self._last_error: str | None = None
        self._last_error_at: str | None = None

        # Simple debouncing for rapid file changes
        self._pending_debounce: dict[str, float] = {}  # file_path -> timestamp
        self._debounce_delay = 0.5  # 500ms delay from research
        self._debounce_tasks: set[asyncio.Task] = set()  # Track active debounce tasks
        self._retry_tasks: set[asyncio.Task[None]] = set()

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
        self._event_queue_overflow_state = "idle"
        self._event_queue_overflow_burst_count = 0
        self._event_queue_overflow_current_burst_dropped = 0
        self._event_queue_overflow_last_burst_dropped = 0
        self._event_queue_overflow_last_started_at: str | None = None
        self._event_queue_overflow_last_cleared_at: str | None = None
        self._event_queue_overflow_sample_event_type: str | None = None
        self._event_queue_overflow_sample_file_path: str | None = None
        self._last_source_event_at: str | None = None
        self._last_source_event_type: str | None = None
        self._last_source_event_path: str | None = None
        self._last_accepted_event_at: str | None = None
        self._last_accepted_event_type: str | None = None
        self._last_accepted_event_path: str | None = None
        self._next_source_generation = 0
        self._latest_source_generation_by_path: dict[str, int] = {}
        self._last_processing_started_at: str | None = None
        self._last_processing_started_path: str | None = None
        self._last_processing_completed_at: str | None = None
        self._last_processing_completed_path: str | None = None
        self._filtered_event_count = 0
        self._suppressed_duplicate_count = 0
        self._translation_error_count = 0
        self._processing_error_count = 0
        self._active_processing_count = 0
        self._event_pressure_by_path: dict[str, HotPathPressure] = {}

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
    def _default_pipeline_snapshot(cls) -> dict[str, Any]:
        return {
            "last_source_event_at": None,
            "last_source_event_type": None,
            "last_source_event_path": None,
            "last_accepted_event_at": None,
            "last_accepted_event_type": None,
            "last_accepted_event_path": None,
            "last_processing_started_at": None,
            "last_processing_started_path": None,
            "last_processing_completed_at": None,
            "last_processing_completed_path": None,
            "filtered_event_count": 0,
            "suppressed_duplicate_count": 0,
            "translation_error_count": 0,
            "processing_error_count": 0,
            "stall_threshold_seconds": cls._STALL_THRESHOLD_SECONDS,
        }

    @classmethod
    def _default_event_pressure_snapshot(cls) -> dict[str, Any]:
        return {
            "state": "idle",
            "sample_path": None,
            "sample_scope": None,
            "sample_event_type": None,
            "events_in_window": 0,
            "coalesced_updates": 0,
            "window_seconds": cls._EVENT_PRESSURE_WINDOW_SECONDS,
            "last_observed_at": None,
        }

    @classmethod
    def _default_pending_mutation_snapshot(cls) -> dict[str, Any]:
        counts_by_operation = {
            operation: 0 for operation in cls._PENDING_MUTATION_STATUS_OPERATIONS
        }
        return {
            "total": 0,
            "unique_paths": 0,
            "counts_by_operation": counts_by_operation,
            "retry_counts_by_operation": dict(counts_by_operation),
            "retrying_mutations": 0,
            "oldest_pending_at": None,
            "oldest_pending_age_seconds": None,
            "oldest_pending_operation": None,
            "oldest_pending_path": None,
            "oldest_pending_retry_count": None,
            "recovery_phase": "idle",
            "resync_reason": None,
        }

    @staticmethod
    def _parse_status_timestamp(value: Any) -> datetime | None:
        if not isinstance(value, str) or not value:
            return None
        normalized = value.replace("Z", "+00:00") if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @classmethod
    def _latest_timestamp(cls, *values: Any) -> datetime | None:
        latest: datetime | None = None
        for value in values:
            parsed = cls._parse_status_timestamp(value)
            if parsed is None:
                continue
            if latest is None or parsed > latest:
                latest = parsed
        return latest

    @classmethod
    def default_health_snapshot(
        cls,
        configured_backend: str | None = None,
        startup_mode: str = "stdio",
    ) -> dict[str, Any]:
        """Return the neutral realtime health structure used by MCP status plumbing."""
        status = {
            "configured_backend": configured_backend,
            "effective_backend": "uninitialized",
            "service_state": "idle",
            "monitoring_mode": "uninitialized",
            "live_indexing_state": "uninitialized",
            "live_indexing_hint": "Live indexing monitoring is not ready yet.",
            "monitoring_ready": False,
            "monitoring_ready_at": None,
            "observer_alive": False,
            "watching_directory": None,
            "watched_directories_count": 0,
            "queue_size": 0,
            "pending_files": 0,
            "pending_mutations": cls._default_pending_mutation_snapshot(),
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
                "overflow": {
                    "state": "idle",
                    "burst_count": 0,
                    "current_burst_dropped": 0,
                    "last_burst_dropped": 0,
                    "last_started_at": None,
                    "last_cleared_at": None,
                    "sample_event_type": None,
                    "sample_file_path": None,
                },
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
            "event_pressure": cls._default_event_pressure_snapshot(),
            "pipeline": cls._default_pipeline_snapshot(),
            "startup": RealtimeStartupStatusTracker.default_snapshot(startup_mode),
        }
        if configured_backend == "watchman":
            status.update(_default_watchman_health_snapshot())
        return status

    @classmethod
    def health_snapshot_for_config(
        cls,
        config: Any | None,
        startup_mode: str = "stdio",
    ) -> dict[str, Any]:
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
        return cls.default_health_snapshot(
            configured_backend=configured_backend,
            startup_mode=startup_mode,
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

    def _start_startup_phase(self, phase_name: str) -> None:
        self._startup_tracker.start_phase(phase_name)
        self._emit_status_update()

    def _complete_startup_phase(self, phase_name: str) -> None:
        self._startup_tracker.complete_phase(phase_name)
        self._emit_status_update()

    def _fail_startup_phase(self, phase_name: str, error: str) -> None:
        self._startup_tracker.fail_phase(phase_name, error)
        self._emit_status_update()

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
        self._refresh_runtime_service_state()
        self._emit_status_update()

    def _clear_error_state(
        self,
        *,
        exact_messages: tuple[str, ...] = (),
        prefixes: tuple[str, ...] = (),
    ) -> None:
        current_error = self._last_error
        if not current_error:
            return
        if current_error in exact_messages or any(
            current_error.startswith(prefix) for prefix in prefixes
        ):
            self._last_error = None
            self._last_error_at = None
        self._refresh_runtime_service_state()
        self._emit_status_update()

    @staticmethod
    def _resync_callback_error(result: Any) -> str | None:
        """Normalize backend-neutral callback error results into service failures."""
        if not isinstance(result, dict) or result.get("status") != "error":
            return None
        error = result.get("error")
        if isinstance(error, str) and error:
            return f"Resync callback reported error status: {error}"
        return "Resync callback reported error status"

    def _refresh_runtime_service_state(self) -> None:
        if self._service_state in {"starting", "stopping", "stopped"}:
            return
        if self._last_error:
            self._service_state = "degraded"
            return
        adapter_health = (
            self._monitor_adapter.get_health() if self._monitor_adapter else {}
        )
        reconnect = adapter_health.get("watchman_reconnect")
        reconnect_state = None
        if isinstance(reconnect, dict):
            reconnect_state = reconnect.get("state")
        connection_state = adapter_health.get("watchman_connection_state")
        if self._effective_backend == "watchman" and (
            reconnect_state in {"pending", "running", "failed"}
            or connection_state in {"disconnected", "sidecar_only"}
        ):
            self._service_state = "degraded"
            return
        self._service_state = "running"

    def _resolve_configured_backend(self) -> str:
        backend = getattr(self.config.indexing, "realtime_backend", None)
        if backend in {"watchman", "watchdog", "polling"}:
            return str(backend)
        return default_realtime_backend_for_current_install()

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

        status = self.default_health_snapshot(
            startup_mode=self._startup_tracker.snapshot()["mode"]
        )
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
        status["pending_mutations"] = self._build_pending_mutation_snapshot()
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
        status["event_queue"]["overflow"].update(
            {
                "state": self._event_queue_overflow_state,
                "burst_count": self._event_queue_overflow_burst_count,
                "current_burst_dropped": (
                    self._event_queue_overflow_current_burst_dropped
                ),
                "last_burst_dropped": self._event_queue_overflow_last_burst_dropped,
                "last_started_at": self._event_queue_overflow_last_started_at,
                "last_cleared_at": self._event_queue_overflow_last_cleared_at,
                "sample_event_type": self._event_queue_overflow_sample_event_type,
                "sample_file_path": self._event_queue_overflow_sample_file_path,
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
        status["event_pressure"].update(self._build_event_pressure_snapshot())
        pipeline = self._build_pipeline_snapshot()
        status["pipeline"].update(pipeline)
        status["startup"] = self._startup_tracker.snapshot()
        live_indexing_state = self._derive_live_indexing_state(pipeline)
        status["live_indexing_state"] = live_indexing_state
        status["live_indexing_hint"] = self._derive_live_indexing_hint(
            live_indexing_state
        )
        return status

    def _pending_mutation_recovery_phase(
        self, total_pending_mutations: int
    ) -> tuple[str, str | None]:
        if self._resync_in_progress:
            return "resync_in_progress", self._last_resync_reason
        if self._needs_resync:
            return "resync_pending", self._last_resync_reason
        if total_pending_mutations > 0:
            return "mutation_drain", None
        return "idle", None

    def _build_pending_mutation_snapshot(self) -> dict[str, Any]:
        snapshot = self._default_pending_mutation_snapshot()
        pending_mutations = list(self._pending_mutations.values())
        total_pending_mutations = len(pending_mutations)
        recovery_phase, resync_reason = self._pending_mutation_recovery_phase(
            total_pending_mutations
        )
        snapshot["total"] = total_pending_mutations
        snapshot["unique_paths"] = max(
            len(self._pending_path_counts), len(self.pending_files)
        )
        snapshot["recovery_phase"] = recovery_phase
        snapshot["resync_reason"] = resync_reason

        if not pending_mutations:
            return snapshot

        counts_by_operation = snapshot["counts_by_operation"]
        retry_counts_by_operation = snapshot["retry_counts_by_operation"]
        retrying_mutations = 0
        oldest_mutation: RealtimeMutation | None = None
        oldest_pending_at: datetime | None = None

        for mutation in pending_mutations:
            status_operation = self._status_operation(mutation.operation)
            counts_by_operation.setdefault(status_operation, 0)
            counts_by_operation[status_operation] += 1
            if mutation.retry_count > 0:
                retrying_mutations += 1
                retry_counts_by_operation.setdefault(status_operation, 0)
                retry_counts_by_operation[status_operation] += 1

            queued_at = self._parse_status_timestamp(mutation.first_queued_at)
            if queued_at is None:
                continue
            if oldest_pending_at is None or queued_at < oldest_pending_at:
                oldest_pending_at = queued_at
                oldest_mutation = mutation

        snapshot["retrying_mutations"] = retrying_mutations
        if oldest_mutation is None or oldest_pending_at is None:
            return snapshot

        snapshot["oldest_pending_at"] = oldest_mutation.first_queued_at
        snapshot["oldest_pending_age_seconds"] = max(
            int((datetime.now(timezone.utc) - oldest_pending_at).total_seconds()),
            0,
        )
        snapshot["oldest_pending_operation"] = self._status_operation(
            oldest_mutation.operation
        )
        snapshot["oldest_pending_path"] = str(oldest_mutation.path)
        snapshot["oldest_pending_retry_count"] = oldest_mutation.retry_count
        return snapshot

    def _prune_event_pressure_entry(
        self, entry: HotPathPressure, *, now_monotonic: float
    ) -> tuple[int, int]:
        cutoff = now_monotonic - self._EVENT_PRESSURE_WINDOW_SECONDS
        while entry.event_timestamps and entry.event_timestamps[0] < cutoff:
            entry.event_timestamps.popleft()
        while (
            entry.coalesced_timestamps
            and entry.coalesced_timestamps[0] < cutoff
        ):
            entry.coalesced_timestamps.popleft()
        return len(entry.event_timestamps), len(entry.coalesced_timestamps)

    def _trim_event_pressure_state(self, *, now_monotonic: float) -> None:
        removable_paths: list[str] = []
        ranked_paths: list[tuple[int, int, float, str]] = []

        for path, entry in self._event_pressure_by_path.items():
            events_in_window, coalesced_updates = self._prune_event_pressure_entry(
                entry,
                now_monotonic=now_monotonic,
            )
            if events_in_window == 0 and coalesced_updates == 0:
                removable_paths.append(path)
                continue
            ranked_paths.append(
                (
                    events_in_window,
                    coalesced_updates,
                    entry.last_observed_monotonic,
                    path,
                )
            )

        for path in removable_paths:
            self._event_pressure_by_path.pop(path, None)

        if len(self._event_pressure_by_path) <= self._EVENT_PRESSURE_MAX_TRACKED_PATHS:
            return

        ranked_paths.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
        keep_paths = {
            path
            for _, _, _, path in ranked_paths[: self._EVENT_PRESSURE_MAX_TRACKED_PATHS]
        }
        self._event_pressure_by_path = {
            path: entry
            for path, entry in self._event_pressure_by_path.items()
            if path in keep_paths
        }

    def _track_event_pressure(
        self,
        file_path: Path | str,
        *,
        event_type: str | None = None,
        scope: str | None = None,
        count_event: bool = False,
        count_coalesced: bool = False,
    ) -> None:
        normalized_path = str(self._normalize_mutation_path(file_path))
        now_monotonic = time.monotonic()
        entry = self._event_pressure_by_path.get(normalized_path)
        if entry is None:
            entry = HotPathPressure()
            self._event_pressure_by_path[normalized_path] = entry

        if count_event:
            entry.event_timestamps.append(now_monotonic)
        if count_coalesced:
            entry.coalesced_timestamps.append(now_monotonic)
        if scope in {"included", "excluded"}:
            entry.last_scope = scope
        if event_type is not None:
            entry.last_event_type = event_type
        entry.last_observed_at = self._utc_now()
        entry.last_observed_monotonic = now_monotonic
        self._prune_event_pressure_entry(entry, now_monotonic=now_monotonic)
        self._trim_event_pressure_state(now_monotonic=now_monotonic)

    def _event_pressure_state_for_counts(
        self, *, events_in_window: int, coalesced_updates: int
    ) -> str:
        if (
            events_in_window >= self._EVENT_PRESSURE_OVERLOADED_EVENTS
            or coalesced_updates >= self._EVENT_PRESSURE_OVERLOADED_COALESCED_UPDATES
        ):
            return "overloaded"
        if (
            events_in_window >= self._EVENT_PRESSURE_ELEVATED_EVENTS
            or coalesced_updates >= self._EVENT_PRESSURE_ELEVATED_COALESCED_UPDATES
        ):
            return "elevated"
        return "idle"

    def _build_event_pressure_snapshot(self) -> dict[str, Any]:
        snapshot = self._default_event_pressure_snapshot()
        if not self._event_pressure_by_path:
            return snapshot

        now_monotonic = time.monotonic()
        self._trim_event_pressure_state(now_monotonic=now_monotonic)
        if not self._event_pressure_by_path:
            return snapshot

        ranked_entries: list[tuple[int, int, float, str, HotPathPressure]] = []
        for path, entry in self._event_pressure_by_path.items():
            events_in_window, coalesced_updates = self._prune_event_pressure_entry(
                entry,
                now_monotonic=now_monotonic,
            )
            if events_in_window == 0 and coalesced_updates == 0:
                continue
            ranked_entries.append(
                (
                    events_in_window,
                    coalesced_updates,
                    entry.last_observed_monotonic,
                    path,
                    entry,
                )
            )

        if not ranked_entries:
            return snapshot

        ranked_entries.sort(
            key=lambda item: (-item[0], -item[1], -item[2], item[3]),
        )
        events_in_window, coalesced_updates, _, sample_path, entry = ranked_entries[0]
        snapshot.update(
            {
                "state": self._event_pressure_state_for_counts(
                    events_in_window=events_in_window,
                    coalesced_updates=coalesced_updates,
                ),
                "sample_path": sample_path,
                "sample_scope": entry.last_scope,
                "sample_event_type": entry.last_event_type,
                "events_in_window": events_in_window,
                "coalesced_updates": coalesced_updates,
                "last_observed_at": entry.last_observed_at,
            }
        )
        return snapshot

    def _build_pipeline_snapshot(self) -> dict[str, Any]:
        pipeline = self._default_pipeline_snapshot()
        pipeline.update(
            {
                "last_source_event_at": self._last_source_event_at,
                "last_source_event_type": self._last_source_event_type,
                "last_source_event_path": self._last_source_event_path,
                "last_accepted_event_at": self._last_accepted_event_at,
                "last_accepted_event_type": self._last_accepted_event_type,
                "last_accepted_event_path": self._last_accepted_event_path,
                "last_processing_started_at": self._last_processing_started_at,
                "last_processing_started_path": self._last_processing_started_path,
                "last_processing_completed_at": self._last_processing_completed_at,
                "last_processing_completed_path": self._last_processing_completed_path,
                "filtered_event_count": self._filtered_event_count,
                "suppressed_duplicate_count": self._suppressed_duplicate_count,
                "translation_error_count": self._translation_error_count,
                "processing_error_count": self._processing_error_count,
            }
        )
        return pipeline

    def _derive_live_indexing_state(self, pipeline: dict[str, Any]) -> str:
        if (
            self._service_state == "degraded"
            or self._last_error is not None
            or self._last_resync_error is not None
            or self._needs_resync
        ):
            return "degraded"
        if (
            self._effective_backend == "uninitialized"
            or not self.monitoring_ready.is_set()
        ):
            return "uninitialized"

        if self._active_processing_count > 0:
            return "busy"

        backlog_size = (
            self.event_queue.qsize() + self.file_queue.qsize() + len(self.pending_files)
        )
        if backlog_size <= 0:
            return "idle"

        accepted_at = self._parse_status_timestamp(pipeline["last_accepted_event_at"])
        latest_progress_at = self._latest_timestamp(
            pipeline["last_processing_started_at"],
            pipeline["last_processing_completed_at"],
        )
        if accepted_at is not None:
            now = datetime.now(timezone.utc)
            accepted_age_seconds = (now - accepted_at).total_seconds()
            progress_is_stale = (
                latest_progress_at is None
                or latest_progress_at < accepted_at
                or (now - latest_progress_at).total_seconds()
                > self._STALL_THRESHOLD_SECONDS
            )
            if (
                accepted_age_seconds > self._STALL_THRESHOLD_SECONDS
                and progress_is_stale
            ):
                return "stalled"
        return "busy"

    def _derive_live_indexing_hint(self, live_indexing_state: str) -> str:
        if live_indexing_state == "degraded":
            if self._event_queue_overflow_state == "reconciling":
                return (
                    "Live indexing is reconciling after internal event queue "
                    "overflow; inspect event_queue.overflow and resync.last_reason."
                )
            if self._event_queue_overflow_state == "failed":
                return (
                    "Live indexing remains degraded after internal event queue "
                    "overflow; inspect event_queue.overflow and resync.last_error."
                )
            if self._needs_resync:
                return (
                    "Live indexing needs reconciliation; inspect resync.last_reason "
                    "and last_error."
                )
            return (
                "Live indexing is degraded; inspect last_error and resync.last_error."
            )
        if live_indexing_state == "stalled":
            return (
                "Accepted events are queued but processing has not advanced in "
                "30s; inspect pipeline timestamps and processing_error_count."
            )
        if live_indexing_state == "busy":
            return "Live indexing is actively processing changes."
        if live_indexing_state == "idle":
            return "Live indexing is connected and idle."
        return "Live indexing monitoring is not ready yet."

    def _emit_status_update(self) -> None:
        try:
            if self._status_callback:
                self._status_callback(self._build_health_snapshot())
        except Exception:
            # Status plumbing must never affect runtime behavior.
            pass

    def _record_source_event(self, event_type: str, file_path: Path | str) -> None:
        normalized_path = str(self._normalize_mutation_path(file_path))
        self._last_source_event_at = self._utc_now()
        self._last_source_event_type = event_type
        self._last_source_event_path = normalized_path
        self._track_event_pressure(
            normalized_path,
            event_type=event_type,
            count_event=True,
        )

    def _record_accepted_event(self, event_type: str, file_path: Path | str) -> None:
        normalized_path = str(self._normalize_mutation_path(file_path))
        source_generation = self._advance_source_generation(normalized_path)
        self._refresh_pending_change_generation(normalized_path, source_generation)
        self._last_accepted_event_at = self._utc_now()
        self._last_accepted_event_type = event_type
        self._last_accepted_event_path = normalized_path
        self._track_event_pressure(
            normalized_path,
            event_type=event_type,
            scope="included",
        )
        self._emit_status_update()

    def _record_filtered_event(self, event_type: str, file_path: Path | str) -> None:
        self._filtered_event_count += 1
        self._track_event_pressure(
            file_path,
            event_type=event_type,
            scope="excluded",
        )
        self._emit_status_update()

    def _record_translation_error(self) -> None:
        self._translation_error_count += 1

    def _record_duplicate_suppression(
        self, event_type: str, file_path: Path | str
    ) -> None:
        self._suppressed_duplicate_count += 1
        self._track_event_pressure(
            file_path,
            event_type=event_type,
            scope="included",
        )
        self._emit_status_update()

    def _record_processing_started(self, file_path: Path | str) -> None:
        self._active_processing_count += 1
        self._last_processing_started_at = self._utc_now()
        self._last_processing_started_path = str(file_path)
        self._emit_status_update()

    def _record_processing_finished(
        self, file_path: Path | str, *, completed: bool
    ) -> None:
        if completed:
            self._last_processing_completed_at = self._utc_now()
            self._last_processing_completed_path = str(file_path)
        if self._active_processing_count > 0:
            self._active_processing_count -= 1
        self._emit_status_update()

    def _record_processing_error(self) -> None:
        self._processing_error_count += 1

    def _normalize_mutation_path(self, file_path: Path | str) -> Path:
        path_obj = Path(file_path)
        base_dir = self.watch_path or getattr(self.config, "target_dir", None)
        return normalize_realtime_path(
            path_obj, base_dir if isinstance(base_dir, Path) else None
        )

    @classmethod
    def _mutation_priority(cls, operation: str) -> int:
        return cls._MUTATION_PRIORITIES.get(
            operation, cls._MUTATION_PRIORITIES["change"]
        )

    @classmethod
    def _normalize_add_priority(cls, priority: str) -> tuple[str, bool]:
        if priority == "change":
            return "change", True
        if priority in {"priority", "scan"}:
            return "scan", False
        if priority == "embed":
            return "embed", False
        if priority in cls._MUTATION_PRIORITIES:
            return priority, False
        return "change", False

    @staticmethod
    def _status_operation(operation: str) -> str:
        if operation == "scan":
            return "change"
        return operation

    def _build_mutation(
        self,
        operation: str,
        file_path: Path | str,
        retry_count: int = 0,
        source_generation: int | None = None,
        first_queued_at: str | None = None,
    ) -> RealtimeMutation:
        self._next_mutation_id += 1
        return RealtimeMutation(
            mutation_id=self._next_mutation_id,
            operation=operation,
            path=self._normalize_mutation_path(file_path),
            first_queued_at=first_queued_at or self._utc_now(),
            retry_count=retry_count,
            source_generation=source_generation,
        )

    def _advance_source_generation(self, file_path: Path | str) -> int:
        normalized_path = str(self._normalize_mutation_path(file_path))
        self._next_source_generation += 1
        self._latest_source_generation_by_path[normalized_path] = (
            self._next_source_generation
        )
        return self._next_source_generation

    def _current_source_generation(self, file_path: Path | str) -> int | None:
        normalized_path = str(self._normalize_mutation_path(file_path))
        return self._latest_source_generation_by_path.get(normalized_path)

    def _refresh_pending_change_generation(
        self, file_path: Path | str, source_generation: int | None = None
    ) -> None:
        normalized_path = str(self._normalize_mutation_path(file_path))
        current_generation = source_generation
        if current_generation is None:
            current_generation = self._latest_source_generation_by_path.get(
                normalized_path
            )
        if current_generation is None:
            return

        key = ("change", normalized_path)
        existing = self._pending_mutations.get(key)
        if existing is None or existing.source_generation == current_generation:
            return
        self._pending_mutations[key] = replace(
            existing,
            source_generation=current_generation,
        )

    def _mark_coalesced_change(
        self, file_path: Path | str, event_type: str = "modified"
    ) -> None:
        self._refresh_pending_change_generation(file_path)
        self._track_event_pressure(
            file_path,
            event_type=event_type,
            scope="included",
            count_coalesced=True,
        )

    def _mutation_for_processing(self, mutation: RealtimeMutation) -> RealtimeMutation:
        if mutation.operation not in {"change", "scan"}:
            return mutation
        current_generation = self._current_source_generation(mutation.path)
        if (
            current_generation is None
            or mutation.source_generation == current_generation
        ):
            return mutation
        return replace(mutation, source_generation=current_generation)

    def _delete_mutation_is_stale(self, mutation: RealtimeMutation) -> bool:
        if mutation.source_generation is None:
            return False

        current_generation = self._current_source_generation(mutation.path)
        return (
            current_generation is not None
            and current_generation > mutation.source_generation
        )

    @staticmethod
    def _pending_mutation_key(mutation: RealtimeMutation) -> tuple[str, str]:
        return (mutation.operation, str(mutation.path))

    def _owns_pending_mutation(self, mutation: RealtimeMutation) -> bool:
        current = self._pending_mutations.get(self._pending_mutation_key(mutation))
        return current is not None and current.mutation_id == mutation.mutation_id

    def _delete_mutation_supersedes_existing(
        self, mutation: RealtimeMutation, existing: RealtimeMutation
    ) -> bool:
        if mutation.operation != "delete" or existing.operation != "delete":
            return False

        incoming_generation = mutation.source_generation
        existing_generation = existing.source_generation

        if incoming_generation is not None and existing_generation is None:
            return True
        if incoming_generation is None:
            return False
        if existing_generation is None:
            return True
        if incoming_generation > existing_generation:
            return True
        if (
            incoming_generation == existing_generation
            and mutation.retry_count < existing.retry_count
        ):
            return True
        return False

    def _register_pending_mutation(self, mutation: RealtimeMutation) -> bool:
        key = self._pending_mutation_key(mutation)
        existing = self._pending_mutations.get(key)
        if existing is not None:
            if self._delete_mutation_supersedes_existing(mutation, existing):
                self._pending_mutations[key] = mutation
                self._debug(
                    "replaced pending delete ownership "
                    f"path={mutation.path} old_generation="
                    f"{existing.source_generation} new_generation="
                    f"{mutation.source_generation}"
                )
                self._emit_status_update()
                return True
            return False

        self._pending_mutations[key] = mutation
        path_key = str(mutation.path)
        self._pending_path_counts[path_key] = (
            self._pending_path_counts.get(path_key, 0) + 1
        )
        self.pending_files.add(mutation.path)
        return True

    def _release_pending_mutation(self, mutation: RealtimeMutation) -> None:
        key = self._pending_mutation_key(mutation)
        current = self._pending_mutations.get(key)
        if current is None or current.mutation_id != mutation.mutation_id:
            return
        self._pending_mutations.pop(key, None)

        path_key = str(mutation.path)
        remaining = self._pending_path_counts.get(path_key, 0) - 1
        if remaining <= 0:
            self._pending_path_counts.pop(path_key, None)
            self.pending_files.discard(mutation.path)
            return
        self._pending_path_counts[path_key] = remaining

    async def _enqueue_mutation(
        self, mutation: RealtimeMutation, *, register: bool = True
    ) -> bool:
        if register and not self._register_pending_mutation(mutation):
            return False
        if not register and not self._owns_pending_mutation(mutation):
            return False

        self._queue_sequence += 1
        await self.file_queue.put(
            (
                self._mutation_priority(mutation.operation),
                self._queue_sequence,
                mutation,
            )
        )
        self._debug(
            "queued "
            f"{mutation.path} operation={mutation.operation} "
            f"retry={mutation.retry_count}"
        )
        self._emit_status_update()
        return True

    async def _retry_mutation_after_delay(
        self, mutation: RealtimeMutation, delay_seconds: float
    ) -> None:
        try:
            await asyncio.sleep(delay_seconds)
            if not self._owns_pending_mutation(mutation):
                self._debug(
                    "dropped superseded delete retry "
                    f"path={mutation.path} source_generation="
                    f"{mutation.source_generation}"
                )
                return
            if self._delete_mutation_is_stale(mutation):
                self._debug(
                    "dropped stale delete retry "
                    f"path={mutation.path} source_generation="
                    f"{mutation.source_generation}"
                )
                self._release_pending_mutation(mutation)
                self._emit_status_update()
                return
            await self._enqueue_mutation(mutation, register=False)
        except asyncio.CancelledError:
            self._release_pending_mutation(mutation)
            raise
        except Exception:
            self._release_pending_mutation(mutation)
            raise

    def _schedule_delete_retry(self, mutation: RealtimeMutation) -> bool:
        if self._delete_mutation_is_stale(mutation):
            self._debug(
                "skipped stale delete retry scheduling "
                f"path={mutation.path} source_generation={mutation.source_generation}"
            )
            return True

        if mutation.retry_count >= self._DELETE_CONFLICT_MAX_RETRIES:
            return False

        retry_mutation = self._build_mutation(
            "delete",
            mutation.path,
            retry_count=mutation.retry_count + 1,
            source_generation=mutation.source_generation,
            first_queued_at=mutation.first_queued_at,
        )
        if not self._register_pending_mutation(retry_mutation):
            return True

        delay_seconds = self._DELETE_CONFLICT_BASE_RETRY_DELAY_SECONDS * (
            2**mutation.retry_count
        )
        retry_task = asyncio.create_task(
            self._retry_mutation_after_delay(retry_mutation, delay_seconds)
        )
        self._retry_tasks.add(retry_task)
        retry_task.add_done_callback(self._retry_tasks.discard)
        self._debug(
            "scheduled delete retry "
            f"{retry_mutation.retry_count} for {retry_mutation.path} "
            f"after {delay_seconds:.2f}s"
        )
        self._emit_status_update()
        return True

    async def _queue_follow_up_change(
        self,
        file_path: Path | str,
        *,
        source_generation: int,
        first_queued_at: str | None = None,
    ) -> bool:
        follow_up = self._build_mutation(
            "change",
            file_path,
            source_generation=source_generation,
            first_queued_at=first_queued_at,
        )
        if not self._register_pending_mutation(follow_up):
            self._refresh_pending_change_generation(file_path, source_generation)
            return False
        return await self._enqueue_mutation(follow_up, register=False)

    @staticmethod
    def _overflow_drop_label(drop_count: int) -> str:
        return "event" if drop_count == 1 else "events"

    def _record_event_queue_overflow(
        self, event_type: str, file_path: Path, *, timestamp: str
    ) -> None:
        if self._event_queue_overflow_state == "idle":
            self._event_queue_overflow_state = "reconciling"
            self._event_queue_overflow_burst_count += 1
            self._event_queue_overflow_current_burst_dropped = 1
            self._event_queue_overflow_sample_event_type = event_type
            self._event_queue_overflow_sample_file_path = str(file_path)
            self._event_queue_overflow_last_started_at = timestamp
            message = (
                "Realtime event queue overflow detected; entering reconciliation mode."
            )
            logger.warning(message)
            self._set_warning(message)
            asyncio.create_task(
                self.request_resync(
                    "event_queue_overflow",
                    {
                        "event_type": event_type,
                        "file_path": str(file_path),
                        "drop_reason": "queue_full",
                        "overflow_burst": self._event_queue_overflow_burst_count,
                        "dropped_events": (
                            self._event_queue_overflow_current_burst_dropped
                        ),
                    },
                )
            )
            return

        self._event_queue_overflow_current_burst_dropped += 1

    def _complete_event_queue_overflow_burst(self, *, success: bool) -> None:
        if self._event_queue_overflow_state == "idle":
            return

        drop_count = max(self._event_queue_overflow_current_burst_dropped, 1)
        self._event_queue_overflow_last_burst_dropped = drop_count

        if success:
            self._event_queue_overflow_state = "idle"
            self._event_queue_overflow_current_burst_dropped = 0
            self._event_queue_overflow_last_cleared_at = self._utc_now()
            message = (
                "Realtime event queue overflow recovered after dropping "
                f"{drop_count} {self._overflow_drop_label(drop_count)}."
            )
            logger.info(message)
            self._set_warning(message)
            return

        self._event_queue_overflow_state = "failed"
        message = (
            "Realtime event queue overflow reconciliation failed after dropping "
            f"{drop_count} {self._overflow_drop_label(drop_count)}."
        )
        logger.warning(message)
        self._set_warning(message)

    def _handle_queue_result(
        self, event_type: str, file_path: Path, accepted: bool, reason: str | None
    ) -> None:
        timestamp = self._utc_now()
        self._event_queue_last_event_type = event_type
        self._event_queue_last_file_path = str(file_path)

        if accepted:
            self._event_queue_accepted += 1
            self._event_queue_last_enqueued_at = timestamp
            self._record_accepted_event(event_type, file_path)
        else:
            self._event_queue_dropped += 1
            self._event_queue_last_reason = reason
            self._event_queue_last_dropped_at = timestamp
            self._track_event_pressure(
                file_path,
                event_type=event_type,
                scope="included",
            )
            if reason == "queue_full":
                self._record_event_queue_overflow(
                    event_type,
                    file_path,
                    timestamp=timestamp,
                )
            else:
                self._set_warning(
                    f"realtime event dropped ({reason or 'unknown_reason'})"
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
                self._complete_event_queue_overflow_burst(success=False)
                self._set_error(self._last_resync_error)
                return

            while True:
                started_request_at = self._last_resync_request_monotonic
                self._resync_in_progress = True
                self._last_resync_started_at = self._utc_now()
                self._last_resync_error = None
                self._emit_status_update()

                try:
                    result = await callback(reason, details)
                    callback_error = self._resync_callback_error(result)
                    if callback_error is not None:
                        raise RuntimeError(callback_error)
                    self._needs_resync = False
                    self._resync_performed_count += 1
                    self._last_resync_completed_at = self._utc_now()
                    self._complete_event_queue_overflow_burst(success=True)
                    if self._service_state not in {"stopping", "stopped"}:
                        self._clear_resync_error_state()
                except Exception as e:
                    self._last_resync_error = str(e)
                    self._complete_event_queue_overflow_burst(success=False)
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
        future = self._watchdog_bootstrap_future
        if future is None:
            return
        if not future.done():
            try:
                await asyncio.wait_for(future, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            except Exception as error:
                logger.debug(
                    "Watchdog bootstrap future raised during shutdown: "
                    f"{type(error).__name__}: {error}"
                )
        if self._watchdog_bootstrap_future is future:
            self._watchdog_bootstrap_future = None

    async def _stop_observer(self) -> None:
        if self.observer:
            self.observer.stop()
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.observer.join, 1.0)
            if self.observer.is_alive():
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
        self._start_startup_phase("realtime_start")

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
                self._complete_startup_phase("realtime_start")
                if self._startup_tracker.snapshot()["mode"] == "stdio":
                    self._startup_tracker.complete_startup()
            else:
                self._service_state = "degraded"
                timeout_message = (
                    "Monitoring did not become ready before startup timeout"
                )
                self._set_warning(timeout_message)
                self._debug("monitoring timeout; continuing")
                self._fail_startup_phase("realtime_start", timeout_message)
                if self._startup_tracker.snapshot()["mode"] == "stdio":
                    self._startup_tracker.fail_startup(
                        timeout_message,
                        phase_name="realtime_start",
                    )
            self._emit_status_update()
        except Exception as error:
            self._fail_startup_phase("realtime_start", str(error))
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
        self._watchdog_setup_task = None
        self._watchdog_bootstrap_future = None
        self._watchdog_bootstrap_abort = threading.Event()

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

        # Wait for debounce tasks to finish cancelling.
        # Each task removes itself from the tracking set via its done callback.
        if self._debounce_tasks:
            await asyncio.gather(*self._debounce_tasks, return_exceptions=True)

        # Defensive: clear all deduplication state (debounce + scan-tracked files).
        # pending_files tracks both change-debounced and scan-priority files; stop() is
        # the only place that purges scan entries.
        self._pending_debounce.clear()
        self.pending_files.clear()

        for task in self._retry_tasks.copy():
            task.cancel()

        if self._retry_tasks:
            await asyncio.gather(*self._retry_tasks, return_exceptions=True)
            self._retry_tasks.clear()

        self._pending_debounce.clear()
        self._pending_mutations.clear()
        self._pending_path_counts.clear()
        self.pending_files.clear()
        self._next_source_generation = 0
        self._latest_source_generation_by_path.clear()
        self._event_pressure_by_path.clear()
        self.file_queue = asyncio.PriorityQueue()
        self.event_queue = asyncio.Queue(maxsize=self._EVENT_QUEUE_MAXSIZE)
        self._queue_sequence = 0
        self._next_mutation_id = 0

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
            self._fail_startup_phase(
                "watchdog_setup",
                "Watchdog setup timed out",
            )
            logger.info(
                f"Watchdog setup timed out for {watch_path} - falling back to polling"
            )
            await self._start_polling_backend(
                watch_path,
                reason="Watchdog setup timed out; switched to polling mode",
            )
        except Exception as e:
            self._watchdog_bootstrap_abort.set()
            self._fail_startup_phase(
                "watchdog_setup",
                f"Watchdog setup failed: {e}",
            )
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
        self._start_startup_phase("polling_setup")
        if not self._using_polling or not self._polling_task:
            self._using_polling = True
            self._polling_task = asyncio.create_task(self._polling_monitor(watch_path))
        self._set_effective_backend("polling")
        await asyncio.sleep(self._POLLING_STARTUP_SETTLE_SECONDS)
        self._monitoring_ready_at = self._utc_now()
        self.monitoring_ready.set()
        self._debug(reason)
        self._complete_startup_phase("polling_setup")
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
        self._complete_startup_phase("watchdog_setup")
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
            source_event_callback=self._record_source_event,
            filtered_event_callback=self._record_filtered_event,
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

    def _polling_snapshot(
        self, watch_path: Path
    ) -> tuple[dict[Path, tuple[int, int]], int, bool]:
        """Collect a filesystem snapshot off the event loop for polling mode."""
        current_files: dict[Path, tuple[int, int]] = {}
        files_checked = 0
        truncated = False
        simple_handler = SimpleEventHandler(
            None, self.config, None, root_path=watch_path
        )

        rglob_gen = watch_path.rglob("*")
        try:
            for file_path in rglob_gen:
                try:
                    if not file_path.is_file():
                        continue

                    files_checked += 1
                    if simple_handler._should_index(file_path):
                        try:
                            stat_result = file_path.stat()
                        except OSError:
                            continue

                        current_files[file_path] = (
                            stat_result.st_mtime_ns,
                            stat_result.st_size,
                        )

                    if files_checked >= 5000:
                        truncated = True
                        break
                except (OSError, PermissionError):
                    continue
        finally:
            rglob_gen.close()

        return current_files, files_checked, truncated

    def _collect_supported_files(self, dir_path: Path) -> list[Path]:
        """Collect supported files in a directory off the event loop."""
        simple_handler = SimpleEventHandler(
            None,
            self.config,
            None,
            root_path=self._path_filter_root(dir_path),
        )
        supported_files: list[Path] = []

        for file_path in dir_path.rglob("*"):
            try:
                if file_path.is_file() and simple_handler._should_index(file_path):
                    supported_files.append(file_path)
            except (OSError, PermissionError):
                continue

        return supported_files

    def _path_filter_root(self, fallback_path: Path | None = None) -> Path:
        """Return the logical workspace root for realtime scope decisions."""
        if self.watch_path is not None:
            return self.watch_path

        target_dir = getattr(self.config, "target_dir", None)
        if isinstance(target_dir, Path):
            return target_dir

        if fallback_path is not None:
            return fallback_path

        return Path.cwd()

    async def _polling_monitor(self, watch_path: Path) -> None:
        """Simple polling monitor for large directories."""
        logger.debug(f"Starting polling monitor for {watch_path}")
        self._debug(f"polling monitor active for {watch_path}")
        # Track both mtime and size so Windows polling catches overwrites that
        # fail to advance mtime reliably on CI filesystems.
        known_files: dict[Path, tuple[int, int]] = {}

        # Use a shorter interval during the first few seconds to ensure
        # freshly created files are detected quickly after startup/fallback.
        polling_start = time.time()

        try:
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

                    for file_path, current_fingerprint in current_files.items():
                        if file_path not in known_files:
                            logger.debug(f"Polling detected new file: {file_path}")
                            self._debug(f"polling detected new file: {file_path}")
                            self._record_source_event("created", file_path)
                            accepted = await self.add_file(file_path, priority="change")
                            if accepted:
                                self._record_accepted_event("created", file_path)
                            else:
                                self._advance_source_generation(file_path)
                                self._refresh_pending_change_generation(file_path)
                        elif known_files[file_path] != current_fingerprint:
                            logger.debug(f"Polling detected modified file: {file_path}")
                            self._debug(f"polling detected modified file: {file_path}")
                            self._record_source_event("modified", file_path)
                            accepted = await self.add_file(file_path, priority="change")
                            if accepted:
                                self._record_accepted_event("modified", file_path)
                            else:
                                self._advance_source_generation(file_path)
                                self._refresh_pending_change_generation(file_path)

                    # Check for deleted files.
                    deleted = set(known_files.keys()) - set(current_files.keys())
                    for file_path in deleted:
                        logger.debug(f"Polling detected deleted file: {file_path}")
                        self._record_source_event("deleted", file_path)
                        self._record_accepted_event("deleted", file_path)
                        source_generation = self._current_source_generation(file_path)
                        await self._enqueue_mutation(
                            self._build_mutation(
                                "delete",
                                file_path,
                                source_generation=source_generation,
                            )
                        )
                        self._debug(f"polling detected deleted file: {file_path}")

                    known_files = current_files

                    # Adaptive poll interval: 0.5s for the first 30s, then 3s
                    # Extended fast polling window ensures reliable detection during
                    # multi-file test sequences on Windows CI where setup + indexing
                    # can consume the initial fast-polling budget
                    elapsed = time.time() - polling_start
                    interval = 0.5 if elapsed < 30.0 else 3.0
                    self._emit_status_update()
                    await asyncio.sleep(interval)

                except Exception as e:
                    logger.error(f"Polling monitor error: {e}")
                    self._set_error(f"Polling monitor error: {e}")
                    await asyncio.sleep(5)
        except asyncio.CancelledError:
            logger.debug("Polling monitor cancelled")
            raise
        finally:
            # Force cleanup of any lingering file handles on Windows
            gc.collect()
            logger.debug("Polling monitor stopped")

    async def add_file(self, file_path: Path, priority: str = "change") -> bool:
        """Add file to the realtime pipeline and report whether work was admitted."""
        operation, debounced = self._normalize_add_priority(priority)
        source_generation = (
            self._current_source_generation(file_path)
            if operation == "change"
            else None
        )
        mutation = self._build_mutation(
            operation,
            file_path,
            source_generation=source_generation,
        )
        if debounced:
            file_str = str(mutation.path)
            if file_str in self._pending_debounce:
                # Keep the already-pending debounce horizon fresh.
                self._pending_debounce[file_str] = time.monotonic()
                self._mark_coalesced_change(mutation.path)
                self._emit_status_update()
                return False

        if not self._register_pending_mutation(mutation):
            if debounced:
                file_str = str(mutation.path)
                if file_str in self._pending_debounce:
                    # Keep the already-pending debounce horizon fresh.
                    self._pending_debounce[file_str] = time.monotonic()
                self._mark_coalesced_change(mutation.path)
                self._emit_status_update()
            return False

        # Simple debouncing for change events
        if debounced:
            file_str = str(mutation.path)
            self._pending_debounce[file_str] = time.monotonic()
            task = asyncio.create_task(self._debounced_add_file(mutation))
            self._debounce_tasks.add(task)
            task.add_done_callback(self._debounce_tasks.discard)
            self._debug(f"queued (debounced) {mutation.path} operation={operation}")
            self._emit_status_update()
            return True

        # Immediate mutations bypass debouncing.
        return await self._enqueue_mutation(mutation, register=False)

    async def _debounced_add_file(
        self, file_or_mutation: Path | RealtimeMutation, priority: str = "change"
    ) -> None:
        """Process file after debounce delay.

        Loops until the debounce window has been quiet for at least
        ``_debounce_delay``. If debounce state is cleared externally, release the
        registered mutation ownership so pending counts do not leak.
        """
        if isinstance(file_or_mutation, RealtimeMutation):
            mutation = file_or_mutation
        else:
            operation, _ = self._normalize_add_priority(priority)
            mutation = self._build_mutation(operation, file_or_mutation)
        file_str = str(mutation.path)
        remaining_delay = self._debounce_delay

        try:
            while True:
                await asyncio.sleep(remaining_delay)

                if file_str not in self._pending_debounce:
                    self._release_pending_mutation(mutation)
                    return

                last_update = self._pending_debounce[file_str]
                remaining_delay = self._debounce_delay - (
                    time.monotonic() - last_update
                )

                # Windows timer granularity can wake slightly before the debounce
                # horizon; retry instead of leaving the file stuck in pending state.
                if remaining_delay > 0:
                    continue
                break
        except asyncio.CancelledError:
            self._pending_debounce.pop(file_str, None)
            self._release_pending_mutation(mutation)
            raise

        del self._pending_debounce[file_str]
        current_mutation = self._pending_mutations.get(
            self._pending_mutation_key(mutation),
            mutation,
        )
        if not await self._enqueue_mutation(current_mutation, register=False):
            self._release_pending_mutation(current_mutation)
            return
        logger.debug(f"Processing debounced file: {current_mutation.path}")
        self._debug(f"processing debounced file: {current_mutation.path}")
        return

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
                        self._record_duplicate_suppression(event_type, file_path)
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
                    source_generation = self._current_source_generation(file_path)
                    await self._enqueue_mutation(
                        self._build_mutation(
                            "delete",
                            file_path,
                            source_generation=source_generation,
                        )
                    )
                    self._debug(f"event deleted: {file_path}")
                elif event_type == "dir_created":
                    await self._enqueue_mutation(
                        self._build_mutation("dir_index", file_path)
                    )
                    self._debug(f"event dir_created: {file_path}")
                elif event_type == "dir_deleted":
                    source_generation = self._current_source_generation(file_path)
                    await self._enqueue_mutation(
                        self._build_mutation(
                            "dir_delete",
                            file_path,
                            source_generation=source_generation,
                        )
                    )
                    self._debug(f"event dir_deleted: {file_path}")

                self.event_queue.task_done()

            except Exception as e:
                logger.error(f"Error consuming event: {e}")
                self._set_error(f"Error consuming realtime event: {e}")
                await asyncio.sleep(0.1)  # Brief pause on error

    async def remove_file(self, file_path: Path) -> None:
        """Remove file from database."""
        logger.debug(f"Removing file from database: {file_path}")
        await self.services.provider.delete_file_completely_async(str(file_path))
        self._debug(f"removed file from database: {file_path}")

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
                await self._cleanup_deleted_directory(
                    dir_path,
                    source_generation=self._current_source_generation(dir_path),
                )
                logger.debug(f"Removed watch for deleted directory: {dir_path}")

    async def _cleanup_deleted_directory(
        self, dir_path: str | Path, *, source_generation: int | None = None
    ) -> int:
        """Queue cleanup work for files that were under a deleted directory."""
        normalized_dir = str(self._normalize_mutation_path(dir_path))

        search_results, _ = await self.services.provider.search_regex_async(
            pattern=f"^{normalized_dir}/.*",
            page_size=1000,
        )

        queued_files = 0
        for result in search_results:
            file_path = result.get("file_path", result.get("path", ""))
            if not file_path:
                continue
            accepted = await self._enqueue_mutation(
                self._build_mutation(
                    "delete",
                    file_path,
                    source_generation=source_generation,
                )
            )
            if accepted:
                queued_files += 1

        logger.info(
            "Queued cleanup for "
            f"{queued_files} files from deleted directory: {normalized_dir}"
        )
        self._debug(
            f"queued deleted directory cleanup {normalized_dir} files={queued_files}"
        )
        return queued_files

    async def _process_delete_mutation(
        self, mutation: RealtimeMutation, *, owned_when_dequeued: bool
    ) -> None:
        """Apply one queued delete with bounded retry for transaction conflicts."""
        if not owned_when_dequeued and not self._delete_mutation_is_stale(mutation):
            self._debug(
                "skipped superseded delete "
                f"path={mutation.path} source_generation={mutation.source_generation}"
            )
            return

        self._record_processing_started(mutation.path)
        completed = False
        try:
            if self._delete_mutation_is_stale(mutation):
                self._debug(
                    "skipped stale delete "
                    f"path={mutation.path} source_generation="
                    f"{mutation.source_generation}"
                )
                completed = True
                return

            await self.remove_file(mutation.path)
            completed = True
        except DuckDBTransactionConflictError as error:
            if self._delete_mutation_is_stale(mutation):
                self._debug(
                    "ignored stale delete conflict "
                    f"path={mutation.path} source_generation="
                    f"{mutation.source_generation}"
                )
                completed = True
                return

            if self._schedule_delete_retry(mutation):
                if self._delete_mutation_is_stale(mutation):
                    completed = True
                    return
                logger.info(
                    "Retrying realtime delete for "
                    f"{mutation.path} after transaction conflict "
                    f"(attempt {mutation.retry_count + 1}/"
                    f"{self._DELETE_CONFLICT_MAX_RETRIES})"
                )
                self._debug(
                    "retrying delete after transaction conflict "
                    f"path={mutation.path} attempt={mutation.retry_count + 1}"
                )
                return

            logger.error(f"Error removing file {mutation.path}: {error}")
            self.failed_files.add(str(mutation.path))
            self._record_processing_error()
            self._set_error(f"Error removing file {mutation.path}: {error}")
        except Exception as error:
            logger.error(f"Error removing file {mutation.path}: {error}")
            self.failed_files.add(str(mutation.path))
            self._record_processing_error()
            self._set_error(f"Error removing file {mutation.path}: {error}")
        finally:
            self._record_processing_finished(mutation.path, completed=completed)

    async def _process_deleted_directory_mutation(
        self, mutation: RealtimeMutation
    ) -> None:
        self._record_processing_started(mutation.path)
        completed = False
        try:
            await self._cleanup_deleted_directory(
                mutation.path,
                source_generation=mutation.source_generation,
            )
            completed = True
        except Exception as error:
            logger.error(
                f"Error cleaning up deleted directory {mutation.path}: {error}"
            )
            self.failed_files.add(str(mutation.path))
            self._record_processing_error()
            self._set_error(
                f"Error cleaning up deleted directory {mutation.path}: {error}"
            )
        finally:
            self._record_processing_finished(mutation.path, completed=completed)

    async def _index_directory(self, dir_path: Path) -> None:
        """Index files in a newly created directory."""
        self._record_processing_started(dir_path)
        completed = False
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
            completed = True

        except Exception as e:
            logger.error(f"Error indexing new directory {dir_path}: {e}")
            self._record_processing_error()
            self._set_error(f"Error indexing new directory {dir_path}: {e}")
        finally:
            self._record_processing_finished(dir_path, completed=completed)

    async def _process_loop(self) -> None:
        """Main processing loop - simple and robust."""
        logger.debug("Starting processing loop")

        while True:
            try:
                # Wait for next mutation (blocks if queue is empty)
                _, _, mutation = await self.file_queue.get()
                owned_when_dequeued = self._owns_pending_mutation(mutation)
                self._release_pending_mutation(mutation)
                mutation = self._mutation_for_processing(mutation)

                # Fast path for embedding generation without re-parsing the file.
                if mutation.operation == "embed":
                    completed = False
                    try:
                        self._record_processing_started(mutation.path)
                        indexing_coordinator = self.services.indexing_coordinator
                        await indexing_coordinator.generate_missing_embeddings()
                        completed = True
                    except Exception as error:
                        logger.warning(
                            "Embedding generation failed in realtime "
                            f"(embed pass): {error}"
                        )
                        self._record_processing_error()
                        self._set_warning(
                            "Embedding generation failed in realtime "
                            f"embed pass: {error}"
                        )
                    finally:
                        self._record_processing_finished(
                            mutation.path, completed=completed
                        )
                    continue

                if mutation.operation == "delete":
                    await self._process_delete_mutation(
                        mutation,
                        owned_when_dequeued=owned_when_dequeued,
                    )
                    continue

                if mutation.operation == "dir_delete":
                    await self._process_deleted_directory_mutation(mutation)
                    continue

                if mutation.operation == "dir_index":
                    await self._index_directory(mutation.path)
                    continue

                file_path = mutation.path

                # Check if file still exists (prevent race condition with deletion)
                if not file_path.exists():
                    logger.debug(f"Skipping {file_path} - file no longer exists")
                    continue

                # Process the file
                logger.debug(
                    f"Processing {file_path} (operation: {mutation.operation})"
                )

                # Skip embeddings for initial and change events to keep loop responsive.
                # An explicit 'embed' follow-up event will generate embeddings.
                skip_embeddings = True

                # Use existing indexing coordinator
                self._record_processing_started(file_path)
                completed = False
                follow_up_generation: int | None = None
                try:
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
                            result.get("chunks", None)
                            if isinstance(result, dict)
                            else None
                        )
                        embeds = (
                            result.get("embeddings", None)
                            if isinstance(result, dict)
                            else None
                        )
                        self._debug(
                            f"processed {file_path} operation={mutation.operation} "
                            f"skip_embeddings={skip_embeddings} "
                            f"chunks={chunks} embeddings={embeds}"
                        )
                    except Exception:
                        pass
                    completed = True
                    current_generation = self._current_source_generation(file_path)
                    if (
                        mutation.operation in {"change", "scan"}
                        and mutation.source_generation is not None
                        and current_generation is not None
                        and current_generation > mutation.source_generation
                    ):
                        follow_up_generation = current_generation
                finally:
                    self._record_processing_finished(file_path, completed=completed)

                if follow_up_generation is not None:
                    await self._queue_follow_up_change(
                        file_path,
                        source_generation=follow_up_generation,
                    )

            except asyncio.CancelledError:
                logger.debug("Processing loop cancelled")
                raise
            except Exception as error:
                mutation_path = (
                    mutation.path if "mutation" in locals() else Path("<unknown>")
                )
                logger.error(f"Error processing {mutation_path}: {error}")
                # Track failed files for debugging and monitoring
                self.failed_files.add(str(mutation_path))
                self._record_processing_error()
                self._set_error(f"Error processing {mutation_path}: {error}")
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
