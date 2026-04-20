from __future__ import annotations

import asyncio
import time
from pathlib import Path, PurePosixPath
from typing import Any

from loguru import logger

from chunkhound.services.realtime_path_filter import RealtimePathFilter
from chunkhound.watchman import (
    WatchmanCliSession,
    WatchmanScopePlan,
    WatchmanSubscriptionScope,
)

from ..context import RealtimeServiceContext, coerce_realtime_context
from ..events import _WatchmanTranslationError


def _default_watchman_loss_of_sync_snapshot() -> dict[str, Any]:
    """Return the stable Watchman loss-of-sync status payload."""
    return {
        "count": 0,
        "fresh_instance_count": 0,
        "recrawl_count": 0,
        "disconnect_count": 0,
        "translation_failure_count": 0,
        "subscription_pdu_dropped_count": 0,
        "last_reason": None,
        "last_at": None,
        "last_details": None,
    }


def _default_watchman_reconnect_snapshot() -> dict[str, Any]:
    """Return the stable Watchman reconnect status payload."""
    return {
        "state": "idle",
        "attempt_count": 0,
        "retry_delay_seconds": None,
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


class WatchmanRealtimeAdapter:
    """Private Watchman sidecar and session bridge adapter."""

    backend_name = "watchman"
    _SUBSCRIPTION_NAME = "chunkhound-live-indexing"
    _RECONNECT_INITIAL_RETRY_DELAY_SECONDS = 1.0
    _RECONNECT_MAX_RETRY_DELAY_SECONDS = 60.0

    def __init__(self, context: RealtimeServiceContext | object) -> None:
        self._context = coerce_realtime_context(context)
        target_dir = getattr(self._context.config, "target_dir", None)
        if not isinstance(target_dir, Path):
            raise RuntimeError(
                "Watchman backend requires config.target_dir "
                "to resolve a private runtime root"
            )
        self._sidecar = self._context.create_sidecar(target_dir)
        self._session: WatchmanCliSession | None = None
        self._sessions: list[WatchmanCliSession] = []
        self._path_filter: RealtimePathFilter | None = None
        self._scope_path_filters: dict[str, RealtimePathFilter] = {}
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
        self._translation_failure_count = 0
        self._subscription_pdu_dropped_count = 0
        self._last_loss_of_sync_reason: str | None = None
        self._last_loss_of_sync_at: str | None = None
        self._last_loss_of_sync_details: dict[str, object] | None = None
        self._bridge_subscription_pdu_dropped = 0
        self._reconnect_state = "idle"
        self._reconnect_attempt_count = 0
        self._reconnect_retry_delay_seconds: float | None = None
        self._last_reconnect_started_at: str | None = None
        self._last_reconnect_completed_at: str | None = None
        self._last_reconnect_error: str | None = None
        self._last_reconnect_result: str | None = None

    async def start(self, watch_path: Path, loop: asyncio.AbstractEventLoop) -> None:
        self._watch_path = watch_path
        self._loop = loop
        self._reset_loss_of_sync_state()
        self._reset_reconnect_state()
        self._bridge_subscription_pdu_dropped = 0
        try:
            await self._establish_monitoring(watch_path, loop, phase="startup")
        except Exception as error:
            message = str(error)
            self._context.set_error(message)
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
                    self._record_translation_loss_of_sync(
                        "scope_mapping_failure",
                        payload,
                        "Watchman subscription PDU did not map to a known scope",
                        failure_kind="unknown_scope",
                    )
                    continue
                self._translate_subscription_pdu(payload, scope)
            except Exception as error:
                self._record_translation_loss_of_sync(
                    "translation_failure",
                    payload,
                    f"Watchman event translation failed: {error}",
                    failure_kind="translation_exception",
                )
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
        if self._context.watchman_scope_plan is None:
            return None
        if len(self._context.watchman_scope_plan.scopes) == 1:
            return self._context.watchman_scope_plan.primary_scope
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
                self._record_bridge_subscription_queue_overflow(
                    payload, shared_queue.maxsize
                )
            finally:
                session.subscription_queue.task_done()

    def _record_bridge_subscription_queue_overflow(
        self,
        payload: dict[str, object],
        queue_maxsize: int,
    ) -> None:
        self._bridge_subscription_pdu_dropped += 1
        self._handle_subscription_queue_overflow(
            payload,
            dropped_count=1,
            queue_maxsize=queue_maxsize,
        )

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
        self._scope_path_filters = {}
        self._context.clear_watchman_monitoring_state()

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
            self._record_translation_loss_of_sync(
                "translation_failure",
                payload,
                "Watchman subscription PDU did not include a files list",
                failure_kind="missing_files_list",
            )
            return

        path_filter = self._path_filter_for_scope(scope)
        if path_filter is None:
            self._record_translation_loss_of_sync(
                "translation_failure",
                payload,
                "Watchman event translation ran without an active path filter",
                failure_kind="missing_path_filter",
            )
            return

        for entry in files:
            if not isinstance(entry, dict):
                self._record_translation_loss_of_sync(
                    "translation_failure",
                    payload,
                    "Watchman subscription entry was not an object",
                    failure_kind="invalid_entry_object",
                )
                return
            try:
                translated = self._translate_watchman_file_entry(entry, scope)
            except _WatchmanTranslationError as error:
                self._record_translation_loss_of_sync(
                    "translation_failure",
                    payload,
                    str(error),
                    failure_kind=error.failure_kind,
                )
                return
            if translated is None:
                continue
            event_type, file_path = translated
            self._context.ingest_realtime_event(
                event_type,
                file_path,
                should_index=self._should_admit_event(
                    event_type,
                    file_path,
                    path_filter,
                ),
            )

    def _path_filter_for_scope(
        self, scope: WatchmanSubscriptionScope
    ) -> RealtimePathFilter | None:
        return self._scope_path_filters.get(
            str(scope.requested_path),
            self._path_filter,
        )

    def _should_admit_event(
        self,
        event_type: str,
        file_path: Path,
        path_filter: RealtimePathFilter,
    ) -> bool:
        if event_type in {"dir_created", "dir_deleted"}:
            return True
        return path_filter.should_index(file_path)

    def _translate_watchman_file_entry(
        self,
        entry: dict[str, object],
        scope: WatchmanSubscriptionScope,
    ) -> tuple[str, Path] | None:
        file_type = entry.get("type")
        if file_type not in {None, "f", "d"}:
            self._warn_translation_issue(
                f"Skipping unexpected Watchman file type {file_type!r}"
            )
            return None

        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise _WatchmanTranslationError(
                "Skipping Watchman subscription entry without a valid name",
                failure_kind="invalid_entry_name",
            )

        relative_name = PurePosixPath(name.strip().replace("\\", "/"))
        if (
            relative_name.is_absolute()
            or ".." in relative_name.parts
            or not relative_name.parts
        ):
            raise _WatchmanTranslationError(
                f"Skipping unsafe Watchman subscription path {name!r}",
                failure_kind="unsafe_path",
            )

        # Preserve the logical handled path under config.target_dir even when
        # the Watchman watch root is a physical junction target outside it.
        canonical_path = scope.requested_path.joinpath(*relative_name.parts)
        try:
            canonical_path.relative_to(scope.requested_path)
        except ValueError:
            raise _WatchmanTranslationError(
                f"Skipping out-of-scope Watchman path {canonical_path}",
                failure_kind="out_of_scope_path",
            )

        exists = entry.get("exists")
        is_new = entry.get("new")
        if file_type == "d":
            if exists is False:
                event_type = "dir_deleted"
            elif exists is True and is_new is True:
                event_type = "dir_created"
            else:
                return None
        elif exists is False:
            event_type = "deleted"
        elif exists is True and is_new is True:
            event_type = "created"
        else:
            event_type = "modified"
        return event_type, canonical_path

    def _warn_translation_issue(self, message: str) -> None:
        warning = f"Watchman event translation warning: {message}"
        logger.warning(warning)
        self._context.record_translation_error()
        self._context.set_warning(warning)

    def _record_translation_issue(
        self,
        payload: dict[str, object] | None,
        message: str,
    ) -> None:
        if payload is None:
            self._warn_translation_issue(message)
            return
        self._record_translation_loss_of_sync(
            "translation_failure",
            payload,
            message,
            failure_kind="translation_issue",
        )

    def _record_translation_loss_of_sync(
        self,
        reason: str,
        payload: dict[str, object],
        message: str,
        *,
        failure_kind: str,
    ) -> None:
        warning = f"Watchman event translation warning: {message}"
        logger.warning(warning)
        self._context.record_translation_error()

        details: dict[str, object] = {
            "backend": "watchman",
            "loss_of_sync_reason": reason,
            "subscription": str(payload.get("subscription") or self._SUBSCRIPTION_NAME),
            "translation_failure_kind": failure_kind,
            "translation_failure_message": message,
            "translation_warning": warning,
        }
        clock = payload.get("clock")
        if isinstance(clock, str) and clock:
            details["clock"] = clock

        self._record_loss_of_sync(reason, message=warning, details=details)

    def _reset_loss_of_sync_state(self) -> None:
        self._loss_of_sync_count = 0
        self._fresh_instance_count = 0
        self._recrawl_count = 0
        self._disconnect_count = 0
        self._translation_failure_count = 0
        self._subscription_pdu_dropped_count = 0
        self._last_loss_of_sync_reason = None
        self._last_loss_of_sync_at = None
        self._last_loss_of_sync_details = None

    def _reset_reconnect_state(self) -> None:
        self._reconnect_state = "idle"
        self._reconnect_attempt_count = 0
        self._reconnect_retry_delay_seconds = None
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
        self._context.clear_watchman_monitoring_state()
        self._path_filter = None
        self._scope_path_filters = {}
        self._shared_subscription_queue = None
        self._subscription_scope_map = {}
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
            self._context.emit_status_update()

    async def _establish_monitoring(
        self,
        watch_path: Path,
        loop: asyncio.AbstractEventLoop,
        *,
        phase: str,
    ) -> None:
        self._context.start_startup_phase("watchman_sidecar_start")
        try:
            metadata = await self._sidecar.start()
        except Exception as error:
            self._context.fail_startup_phase(
                "watchman_sidecar_start",
                f"Watchman sidecar {phase} failed: {error}",
            )
            raise RuntimeError(f"Watchman sidecar {phase} failed: {error}") from error
        self._context.complete_startup_phase("watchman_sidecar_start")

        self._context.clear_watchman_monitoring_state()

        def _new_session() -> WatchmanCliSession:
            return self._context.create_watchman_session(
                metadata,
                self._sidecar,
                self._handle_subscription_queue_overflow,
            )

        primary_session = _new_session()
        use_prepared_startup = primary_session.supports_prepared_session_startup()
        sessions: list[WatchmanCliSession] = []
        subscription_scope_map: dict[str, WatchmanSubscriptionScope] = {}
        try:
            self._context.start_startup_phase("watchman_watch_project")
            if use_prepared_startup:
                await primary_session.prepare()
                watch_project_response = await primary_session.watch_project(watch_path)
            else:
                watch_project_response = (
                    await primary_session.startup_watch_project_once(watch_path)
                )
            self._context.complete_startup_phase("watchman_watch_project")
            self._context.start_startup_phase("watchman_scope_discovery")

            def _log_scope_discovery(message: str) -> None:
                self._context.debug(f"watchman scope discovery: {message}")

            def _elapsed_seconds(started_at: float) -> float:
                return round(max(time.monotonic() - started_at, 0.0), 3)

            def _render_paths(paths: list[Path]) -> str:
                if not paths:
                    return "[]"
                return "[" + ", ".join(str(path) for path in paths) + "]"

            scope_discovery_started_at = time.monotonic()

            nested_mount_discovery_started_at = time.monotonic()
            nested_mount_roots = self._context.discover_nested_linux_mount_roots(
                watch_path
            )
            _log_scope_discovery(
                "linux nested mounts "
                f"count={len(nested_mount_roots)} "
                f"duration={_elapsed_seconds(nested_mount_discovery_started_at)}s "
                f"roots={_render_paths(list(nested_mount_roots))}"
            )

            junction_discovery_started_at = time.monotonic()
            additional_scopes = self._context.discover_nested_windows_junction_scopes(
                watch_path
            )
            _log_scope_discovery(
                "windows junction scopes "
                f"count={len(additional_scopes)} "
                f"duration={_elapsed_seconds(junction_discovery_started_at)}s "
                "roots="
                f"{_render_paths([scope.watch_root for scope in additional_scopes])}"
            )

            startup_watch_roots = [
                *nested_mount_roots,
                *(scope.watch_root for scope in additional_scopes),
            ]
            watch_roots_started_at = time.monotonic()
            if use_prepared_startup:
                await primary_session.watch_roots(startup_watch_roots)
            else:
                await primary_session.startup_watch_roots_once(startup_watch_roots)
            _log_scope_discovery(
                "watch roots "
                f"mode={'prepared_session' if use_prepared_startup else 'one_shot'} "
                f"count={len(startup_watch_roots)} "
                f"duration={_elapsed_seconds(watch_roots_started_at)}s "
                f"roots={_render_paths(list(startup_watch_roots))}"
            )

            scope_plan_started_at = time.monotonic()
            scope_plan = self._context.build_watchman_scope_plan(
                watch_path,
                watch_project_response,
                nested_mount_roots=nested_mount_roots,
                additional_scopes=additional_scopes,
            )
            _log_scope_discovery(
                "scope plan built "
                f"count={len(scope_plan.scopes)} "
                f"duration={_elapsed_seconds(scope_plan_started_at)}s "
                f"scopes={[scope.scope_kind for scope in scope_plan.scopes]}"
            )
            _log_scope_discovery(
                f"phase total duration={_elapsed_seconds(scope_discovery_started_at)}s"
            )
            self._context.complete_startup_phase("watchman_scope_discovery")

            self._context.start_startup_phase("watchman_subscription_setup")
            self._shared_subscription_queue = asyncio.Queue(
                maxsize=WatchmanCliSession._SUBSCRIPTION_QUEUE_MAXSIZE
            )
            self._subscription_scope_map = {}
            if not scope_plan.scopes:
                raise RuntimeError("Watchman scope plan did not contain any scopes")
            resolved_scope_names = (
                self._context.build_watchman_subscription_names_for_scope_plan(
                    base_name=self._SUBSCRIPTION_NAME,
                    target_path=watch_path,
                    scope_plan=scope_plan,
                )
            )

            primary_scope = scope_plan.primary_scope
            primary_scope_plan = WatchmanScopePlan(scopes=(primary_scope,))
            primary_subscription_name = resolved_scope_names[0]
            if use_prepared_startup:
                primary_setup = await primary_session.subscribe_scopes(
                    target_path=watch_path,
                    subscription_name=primary_subscription_name,
                    scope_plan=primary_scope_plan,
                )
            else:
                primary_setup = await primary_session.start(
                    target_path=watch_path,
                    subscription_name=primary_subscription_name,
                    scope_plan=primary_scope_plan,
                    nested_mount_roots=(),
                    additional_scopes=(),
                )
            sessions.append(primary_session)
            for subscription_name in primary_setup.subscription_names:
                subscription_scope_map[subscription_name] = primary_scope

            for scope, scoped_subscription_name in zip(
                scope_plan.scopes[1:],
                resolved_scope_names[1:],
                strict=True,
            ):
                session = _new_session()
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
            current_phase = self._context.current_startup_phase()
            if current_phase is not None:
                self._context.fail_startup_phase(current_phase, str(error))
            for session in sessions:
                await session.stop()
            if primary_session not in sessions:
                await primary_session.stop()
            await self._sidecar.stop()
            self._context.clear_watchman_monitoring_state()
            self._shared_subscription_queue = None
            self._subscription_scope_map = {}
            raise RuntimeError(f"Watchman session {phase} failed: {error}") from error

        self._sessions = sessions
        self._session = sessions[0] if sessions else None
        self._subscription_scope_map = dict(subscription_scope_map)
        if self._shared_subscription_queue is None:
            raise RuntimeError("Watchman shared subscription queue was not initialized")
        self._scope_path_filters = {
            str(scope.requested_path): RealtimePathFilter(
                config=self._context.config,
                root_path=scope.requested_path,
            )
            for scope in scope_plan.scopes
        }
        self._path_filter = self._scope_path_filters.get(
            str(scope_plan.primary_scope.requested_path)
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
        self._context.activate_watchman_monitoring(
            scope_plan=scope_plan,
            subscription_queue=self._shared_subscription_queue,
            backend_name=self.backend_name,
        )
        self._context.complete_startup_phase("watchman_subscription_setup")
        self._context.emit_status_update()

    def _begin_reconnect_cycle(self) -> None:
        if self._reconnect_task is not None and not self._reconnect_task.done():
            return
        if self._watch_path is None or self._loop is None:
            return
        self._reconnect_state = "running"
        self._reconnect_attempt_count = 0
        self._reconnect_retry_delay_seconds = None
        self._last_reconnect_started_at = None
        self._last_reconnect_completed_at = None
        self._last_reconnect_error = None
        self._last_reconnect_result = None
        self._context.emit_status_update()
        self._reconnect_task = asyncio.create_task(self._run_reconnect_loop())

    async def _run_reconnect_loop(self) -> None:
        watch_path = self._watch_path
        loop = self._loop
        if watch_path is None or loop is None:
            return

        try:
            retry_delay = self._RECONNECT_INITIAL_RETRY_DELAY_SECONDS
            attempt = 0
            while True:
                attempt += 1
                self._reconnect_state = "running"
                self._reconnect_attempt_count = attempt
                self._reconnect_retry_delay_seconds = None
                self._last_reconnect_started_at = self._context.utc_now()
                self._context.emit_status_update()

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
                    self._last_reconnect_completed_at = self._context.utc_now()
                    self._last_reconnect_result = "failed"
                    self._reconnect_state = "retrying"
                    self._reconnect_retry_delay_seconds = retry_delay
                    self._context.emit_status_update()
                    self._context.set_warning(
                        "Watchman reconnect attempt "
                        f"{attempt} failed: {self._last_reconnect_error}; "
                        f"retrying in {retry_delay:.1f}s"
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(
                        retry_delay * 2.0,
                        self._RECONNECT_MAX_RETRY_DELAY_SECONDS,
                    )
                    continue

                self._reconnect_state = "restored"
                self._reconnect_retry_delay_seconds = None
                self._last_reconnect_completed_at = self._context.utc_now()
                self._last_reconnect_result = "restored"
                self._last_reconnect_error = None
                self._context.clear_error_state(
                    prefixes=(
                        "Watchman session disconnected:",
                        "Watchman reconnect failed:",
                    )
                )
                self._schedule_post_restore_resync()
                self._context.refresh_runtime_service_state()
                self._context.emit_status_update()
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
        if self._context.needs_resync:
            self._context.emit_status_update()
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
        elif reason in {"translation_failure", "scope_mapping_failure"}:
            self._translation_failure_count += 1
        elif reason == "subscription_pdu_dropped":
            self._subscription_pdu_dropped_count += 1

        self._last_loss_of_sync_reason = reason
        self._last_loss_of_sync_at = self._context.utc_now()
        self._last_loss_of_sync_details = dict(details) if details else None

        if message:
            if as_error:
                self._context.set_error(message)
            else:
                self._context.set_warning(message)
        else:
            self._context.emit_status_update()

        self._schedule_resync_request(reason, details)

    def _schedule_resync_request(
        self, reason: str, details: dict[str, object] | None = None
    ) -> None:
        async def _dispatch() -> None:
            try:
                await self._context.request_resync("realtime_loss_of_sync", details)
            except Exception as error:
                self._context.set_error(f"Watchman resync request failed: {error}")

        self._context.start_transient_task(_dispatch())

    def _schedule_post_restore_resync(self) -> None:
        details = {
            "backend": "watchman",
            "loss_of_sync_reason": "disconnect",
            "post_restore_reconciliation": True,
            "reconnect_state": "restored",
        }
        self._schedule_resync_request("disconnect", details)

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
        session_subscription_pdu_dropped = sum(
            int(item.get("watchman_subscription_pdu_dropped") or 0)
            for item in session_healths
        )
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
                    "watchman_subscription_pdu_dropped": (
                        session_subscription_pdu_dropped
                        + self._bridge_subscription_pdu_dropped
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
        else:
            health["watchman_subscription_pdu_dropped"] = (
                self._bridge_subscription_pdu_dropped
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
            "translation_failure_count": self._translation_failure_count,
            "subscription_pdu_dropped_count": self._subscription_pdu_dropped_count,
            "last_reason": self._last_loss_of_sync_reason,
            "last_at": self._last_loss_of_sync_at,
            "last_details": self._last_loss_of_sync_details,
        }
        health["watchman_reconnect"] = {
            "state": self._reconnect_state,
            "attempt_count": self._reconnect_attempt_count,
            "retry_delay_seconds": self._reconnect_retry_delay_seconds,
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
        return self._context.build_watchman_subscription_name_for_scope(
            base_name=base_name,
            target_path=target_path,
            scope=scope,
            scope_index=scope_index,
        )
