from __future__ import annotations

import asyncio
import copy
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from loguru import logger
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from chunkhound.watchman import WatchmanScopePlan
from chunkhound.watchman_runtime.loader import default_realtime_backend_for_current_install
from chunkhound.utils.windows_constants import IS_WINDOWS

from .adapters import PollingRealtimeAdapter, WatchdogRealtimeAdapter, WatchmanRealtimeAdapter
from .events import SimpleEventHandler

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

class RealtimeStartupMixin:
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
            reconnect_state in {"retrying", "running"}
            or connection_state in {"disconnected", "sidecar_only"}
        ):
            self._service_state = "degraded"
            return
        self._service_state = "running"
    def _resolve_configured_backend(self) -> str:
        backend = getattr(self.config.indexing, "realtime_backend", None)
        self._configured_backend_raw = backend
        if backend in {"watchman", "watchdog", "polling"}:
            self._configured_backend_resolution = "explicit"
            return str(backend)
        self._configured_backend_resolution = "install_default"
        resolved_backend = default_realtime_backend_for_current_install()
        if backend is None:
            reason = "no explicit realtime backend is configured"
        else:
            reason = f"configured realtime backend {backend!r} is invalid"
        logger.info(
            "Realtime backend resolved to install default "
            f"{resolved_backend!r} because {reason}."
        )
        return resolved_backend
    def _set_effective_backend(self, backend: str) -> None:
        self._effective_backend = backend
    def _clear_watchman_monitoring_state(self) -> None:
        self.watchman_scope_plan = None
        self.watchman_subscription_queue = None
        self.monitoring_ready.clear()
        self._monitoring_ready_at = None
    def _activate_watchman_monitoring(
        self,
        *,
        scope_plan: WatchmanScopePlan,
        subscription_queue: asyncio.Queue[dict[str, object]],
        backend_name: str,
    ) -> None:
        self.watchman_scope_plan = scope_plan
        self.watchman_subscription_queue = subscription_queue
        self._set_effective_backend(backend_name)
        self._monitoring_ready_at = self._utc_now()
        self.monitoring_ready.set()
    def _current_startup_phase(self) -> str | None:
        current_phase = self._startup_tracker.snapshot().get("current_phase")
        return current_phase if isinstance(current_phase, str) else None
    def _start_still_owned(self, start_generation: int) -> bool:
        """Return whether a start() invocation still owns service startup state."""
        return (
            start_generation == self._start_generation
            and self._service_state not in {"stopping", "stopped"}
        )
    def _build_monitor_adapter(self) -> Any:
        if self._configured_backend == "watchman":
            return WatchmanRealtimeAdapter(self._realtime_context)
        if self._configured_backend == "polling":
            return PollingRealtimeAdapter(self._realtime_context)
        return WatchdogRealtimeAdapter(self._realtime_context)
    @staticmethod
    def _normalize_requested_watch_path(path: Path) -> Path:
        """Return an absolute watch path without resolving junctions/symlinks."""
        return path.expanduser().absolute()
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
    async def start(self, watch_path: Path) -> None:
        """Start real-time indexing service."""
        # Fail-closed indexed-root identity guard BEFORE any state mutation,
        # path normalization, monitor construction, or watcher setup.
        # Validate against the provider's authoritative base_directory so the
        # sidecar stays consistent with connect-time validation across
        # normalization differences (e.g. macOS /var ↔ /private/var, Windows
        # 8.3 short-name expansion) that can otherwise cause the caller-passed
        # `watch_path` to diverge from the already-validated provider base.
        provider = getattr(getattr(self, "services", None), "provider", None)
        ensure_root = getattr(provider, "ensure_indexed_root_identity", None)
        if callable(ensure_root):
            get_base = getattr(provider, "get_base_directory", None)
            requested_root = get_base() if callable(get_base) else watch_path
            ensure_root(
                requested_root=requested_root,
                allow_claim_if_missing=True,
            )

        start_task = asyncio.current_task()
        self._active_start_task = start_task
        self._start_generation += 1
        start_generation = self._start_generation
        self._start_startup_phase("realtime_start")

        try:
            requested_watch_path = self._normalize_requested_watch_path(watch_path)
            self._stopping = False
            self._indexed_files.clear()
            self._removed_files.clear()
            self.failed_files.clear()
            logger.debug(f"Starting real-time indexing for {requested_watch_path}")
            self._debug(f"start watch on {requested_watch_path}")
            self._service_state = "starting"
            self._configured_backend = self._resolve_configured_backend()
            self._effective_backend = "uninitialized"
            self._monitor_adapter = self._build_monitor_adapter()
            self.monitoring_ready.clear()
            self._monitoring_ready_at = None

            effective_watch_path = requested_watch_path
            if self._monitor_adapter.backend_name != "watchman":
                # Polling/watchdog still use resolved monitor paths so observer and
                # coordinator normalization stay aligned on Windows.
                effective_watch_path = requested_watch_path.resolve()

            # Store the watch path
            self.watch_path = effective_watch_path
            self._emit_status_update()

            loop = asyncio.get_event_loop()

            # Start all necessary tasks
            self.event_consumer_task = asyncio.create_task(self._consume_events())
            self.process_task = asyncio.create_task(self._process_loop())

            await self._monitor_adapter.start(effective_watch_path, loop)
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
        self._stopping = True
        self._service_state = "stopping"
        self.monitoring_ready.clear()
        self._monitoring_ready_at = None
        self._emit_status_update()
        async with self._file_condition:
            self._file_condition.notify_all()

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
        # Defensive: clear all deduplication state (debounce + scan-tracked files).
        # pending_files tracks both change-debounced and scan-priority files; stop() is
        # the only place that purges scan entries.
        self._pending_debounce.clear()
        self.pending_files.clear()
        await self._transient_tasks.cancel_all()

        self._pending_debounce.clear()
        self._pending_mutations.clear()
        self._pending_path_counts.clear()
        self.pending_files.clear()
        self._next_source_generation = 0
        self._latest_source_generation_by_path.clear()
        self._event_pressure_by_path.clear()
        self._active_change_generations.clear()
        self._reserved_follow_up_change_generations.clear()
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
        self, future: asyncio.Future[tuple[BaseObserver, SimpleEventHandler] | None]
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
        observer: BaseObserver,
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
    def _stop_bootstrap_observer(observer: BaseObserver) -> None:
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
    ) -> tuple[BaseObserver, SimpleEventHandler] | None:
        """Create and start a watchdog observer without mutating shared state."""
        event_handler = SimpleEventHandler(
            self.event_queue,
            self.config,
            loop,
            root_path=watch_path,
            queue_result_callback=self._handle_queue_result,
            source_event_callback=self._record_source_event,
            filtered_event_callback=self._record_filtered_event,
            admission_callback=self._should_admit_realtime_event,
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
