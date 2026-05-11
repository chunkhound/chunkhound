"""Real-time indexing service for MCP servers.

This service provides continuous filesystem monitoring and incremental updates
while maintaining search responsiveness. It leverages the existing indexing
infrastructure and respects the single-threaded database constraint.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from loguru import logger
from watchdog.observers.api import BaseObserver

from chunkhound.core.config.config import Config
from chunkhound.database_factory import DatabaseServices
from chunkhound.services.realtime_path_filter import (
    RealtimePathFilter,
    RealtimePathFilterSettings,
)
from chunkhound.watchman import (
    PrivateWatchmanSidecar,
    WatchmanCliSession,
    WatchmanScopePlan,
    WatchmanSubscriptionScope,
    build_watchman_scope_plan,
    build_watchman_subscription_name_for_scope,
    build_watchman_subscription_names_for_scope_plan,
    discover_nested_linux_mount_roots,
    discover_nested_windows_junction_scopes,
)

from .adapters import (
    PollingRealtimeAdapter,
    WatchdogRealtimeAdapter,
    WatchmanRealtimeAdapter,
    _default_watchman_health_snapshot,
)
from .context import RealtimeServiceContext
from .events import (
    HotPathPressure,
    QueueResultCallback,
    RealtimeMutation,
    SimpleEventHandler,
    normalize_file_path,
)
from .pipeline import RealtimePipelineMixin
from .startup import (
    RealtimeStartupMixin,
    RealtimeStartupStatusTracker,
    default_realtime_backend_for_current_install,
)
from .task_supervision import OwnedTaskSet


class RealtimeMonitorAdapter(Protocol):
    """Backend-specific filesystem monitoring lifecycle."""

    backend_name: str

    async def start(
        self, watch_path: Path, loop: asyncio.AbstractEventLoop
    ) -> None: ...

    async def stop(self) -> None: ...

    def get_health(self) -> dict[str, Any]: ...


class RealtimeIndexingService(RealtimeStartupMixin, RealtimePipelineMixin):
    """Simple real-time indexing service with search responsiveness."""

    _PENDING_MUTATION_STATUS_OPERATIONS = (
        "change",
        "delete",
        "embed",
        "dir_delete",
        "dir_index",
    )
    _EVENT_DEDUP_WINDOW_SECONDS = 2.0
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
    _DELETE_BATCH_SIZE = 64
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
        self._debug_sink = debug_sink
        self._status_callback = status_callback
        self._resync_callback = resync_callback
        resolved_startup_log_sink = startup_log_sink or debug_sink
        self._startup_tracker = startup_tracker or RealtimeStartupStatusTracker(
            debug_sink=resolved_startup_log_sink
        )
        self._startup_tracker.set_debug_sink(resolved_startup_log_sink)
        self._configured_backend_raw: object | None = None
        self._configured_backend_resolution = "explicit"
        self._configured_backend = self._resolve_configured_backend()
        self._effective_backend = "uninitialized"
        self._realtime_context = RealtimeServiceContext(
            self,
            sidecar_factory=lambda target_dir, debug_sink: PrivateWatchmanSidecar(
                target_dir,
                debug_sink=debug_sink,
            ),
            session_factory=lambda metadata,
            sidecar,
            overflow_handler: WatchmanCliSession(
                binary_path=Path(metadata.binary_path),
                socket_path=sidecar.paths.listener_path,
                statefile_path=sidecar.paths.statefile_path,
                logfile_path=sidecar.paths.logfile_path,
                pidfile_path=sidecar.paths.pidfile_path,
                project_root=sidecar.paths.project_root,
                debug_sink=self._debug,
                subscription_overflow_handler=overflow_handler,
            ),
            nested_mount_discoverer=lambda watch_path: discover_nested_linux_mount_roots(
                watch_path
            ),
            junction_scope_discoverer=lambda watch_path: discover_nested_windows_junction_scopes(
                watch_path
            ),
            scope_plan_builder=lambda watch_path,
            watch_project_response,
            **kwargs: build_watchman_scope_plan(
                watch_path,
                watch_project_response,
                **kwargs,
            ),
            subscription_name_builder=lambda **kwargs: build_watchman_subscription_name_for_scope(
                **kwargs
            ),
            subscription_names_builder=lambda **kwargs: build_watchman_subscription_names_for_scope_plan(
                **kwargs
            ),
        )
        self._monitor_adapter: RealtimeMonitorAdapter | None = None
        self.watchman_scope_plan: WatchmanScopePlan | None = None
        self.watchman_subscription_queue: asyncio.Queue[dict[str, object]] | None = None
        self.file_queue: asyncio.PriorityQueue[tuple[int, int, RealtimeMutation]] = (
            asyncio.PriorityQueue()
        )
        self._queue_sequence = 0
        self._next_mutation_id = 0
        self.event_queue: asyncio.Queue[tuple[str, Path]] = asyncio.Queue(
            maxsize=self._EVENT_QUEUE_MAXSIZE
        )
        self.pending_files: set[Path] = set()
        self._pending_mutations: dict[tuple[str, str], RealtimeMutation] = {}
        self._pending_path_counts: dict[str, int] = {}
        self.failed_files: set[str] = set()
        self._last_warning: str | None = None
        self._last_warning_at: str | None = None
        self._last_error: str | None = None
        self._last_error_at: str | None = None
        self._pending_debounce: dict[str, float] = {}
        self._debounce_delay = 0.5
        self._transient_tasks = OwnedTaskSet()
        # Transitional compatibility for existing tests and call sites that
        # still inspect the legacy debounce-task set directly.
        self._debounce_tasks: set[asyncio.Task[Any]] = self._transient_tasks._tasks
        self._recent_file_events: dict[str, tuple[str, float]] = {}
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
        self._active_change_generations: dict[str, int | None] = {}
        self._reserved_follow_up_change_generations: dict[str, int] = {}
        self.scan_iterator: Iterator | None = None
        self.scan_complete = False
        self.observer: BaseObserver | None = None
        self.event_handler: SimpleEventHandler | None = None
        self.watch_path: Path | None = None
        self.process_task: asyncio.Task | None = None
        self.event_consumer_task: asyncio.Task | None = None
        self._polling_task: asyncio.Task | None = None
        self._watchdog_setup_task: asyncio.Task | None = None
        self._watchdog_bootstrap_future: (
            asyncio.Future[tuple[BaseObserver, SimpleEventHandler] | None] | None
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
        self.watched_directories: set[str] = set()
        self.watch_lock = asyncio.Lock()
        self.monitoring_ready = asyncio.Event()
        self._monitoring_ready_at: str | None = None
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
        self._file_condition = asyncio.Condition()
        self._indexed_files: set[str] = set()
        self._removed_files: set[str] = set()
        self._stopping = False

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
        while entry.coalesced_timestamps and entry.coalesced_timestamps[0] < cutoff:
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

    async def wait_for_file_indexed(
        self, path: Path | str, timeout: float = 10.0
    ) -> bool:
        """Wait for a file to be indexed.

        Call reset_file_tracking(path) BEFORE triggering the file change
        to ensure the wait observes the new event, not a stale record.
        """
        normalized = normalize_file_path(path)

        async def _wait() -> None:
            async with self._file_condition:
                await self._file_condition.wait_for(
                    lambda: normalized in self._indexed_files
                    or normalized in self.failed_files
                    or self._stopping
                )

        try:
            await asyncio.wait_for(_wait(), timeout=timeout)
            return normalized in self._indexed_files
        except asyncio.TimeoutError:
            return False

    async def wait_for_file_removed(
        self, path: Path | str, timeout: float = 10.0
    ) -> bool:
        """Wait for a file to be removed from the index.

        Call reset_file_tracking(path) BEFORE triggering the file change.
        """
        normalized = normalize_file_path(path)

        async def _wait() -> None:
            async with self._file_condition:
                await self._file_condition.wait_for(
                    lambda: normalized in self._removed_files
                    or normalized in self.failed_files
                    or self._stopping
                )

        try:
            await asyncio.wait_for(_wait(), timeout=timeout)
            return normalized in self._removed_files
        except asyncio.TimeoutError:
            return False

    def reset_file_tracking(self, path: Path | str) -> None:
        """Clear indexing/removal records for a path so the next wait starts fresh.

        Call before triggering the file change, not after.
        """
        # No lock needed: asyncio is single-threaded and there are no await points.
        normalized = normalize_file_path(path)
        self._indexed_files.discard(normalized)
        self._removed_files.discard(normalized)
        self.failed_files.discard(normalized)
