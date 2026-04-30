from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from chunkhound.watchman import WatchmanScopePlan, WatchmanSubscriptionScope

from .events import _enqueue_realtime_event


class RealtimeServiceContext:
    def __init__(
        self,
        service: Any,
        *,
        sidecar_factory: Callable[[Path, Callable[[str], None] | None], Any],
        session_factory: Callable[
            [Any, Any, Callable[[dict[str, object], int, int], None] | None],
            Any,
        ],
        nested_mount_discoverer: Callable[[Path], tuple[Path, ...] | list[Path]],
        junction_scope_discoverer: Callable[
            [Path],
            tuple[WatchmanSubscriptionScope, ...] | list[WatchmanSubscriptionScope],
        ],
        scope_plan_builder: Callable[..., WatchmanScopePlan],
        subscription_name_builder: Callable[..., str],
        subscription_names_builder: Callable[..., tuple[str, ...]],
    ) -> None:
        self._service = service
        self._sidecar_factory = sidecar_factory
        self._session_factory = session_factory
        self._nested_mount_discoverer = nested_mount_discoverer
        self._junction_scope_discoverer = junction_scope_discoverer
        self._scope_plan_builder = scope_plan_builder
        self._subscription_name_builder = subscription_name_builder
        self._subscription_names_builder = subscription_names_builder

    @property
    def config(self) -> Any:
        return self._service.config

    @property
    def watchman_scope_plan(self) -> Any:
        return self._service.watchman_scope_plan

    @property
    def needs_resync(self) -> bool:
        return self._service._needs_resync

    @property
    def observer(self) -> Any:
        return self._service.observer

    @property
    def using_polling(self) -> bool:
        return self._service._using_polling

    @property
    def polling_task(self) -> Any:
        return self._service._polling_task

    def debug(self, message: str) -> None:
        self._service._debug(message)

    def set_error(self, message: str) -> None:
        self._service._set_error(message)

    def set_warning(self, message: str) -> None:
        self._service._set_warning(message)

    def emit_status_update(self) -> None:
        self._service._emit_status_update()

    def record_source_event(self, event_type: str, file_path: Path) -> None:
        self._service._record_source_event(event_type, file_path)

    def record_filtered_event(self, event_type: str, file_path: Path) -> None:
        self._service._record_filtered_event(event_type, file_path)

    def record_translation_error(self) -> None:
        self._service._record_translation_error()

    def ingest_realtime_event(
        self,
        event_type: str,
        file_path: Path,
        *,
        should_index: bool,
    ) -> bool:
        self._service._record_source_event(event_type, file_path)
        if not self._service._should_admit_realtime_event(
            event_type,
            file_path,
            should_index=should_index,
        ):
            self._service._record_filtered_event(event_type, file_path)
            return False
        _enqueue_realtime_event(
            self._service.event_queue,
            self._service._handle_queue_result,
            event_type,
            file_path,
        )
        return True

    def start_startup_phase(self, phase_name: str) -> None:
        self._service._start_startup_phase(phase_name)

    def complete_startup_phase(self, phase_name: str) -> None:
        self._service._complete_startup_phase(phase_name)

    def fail_startup_phase(self, phase_name: str, error: str) -> None:
        self._service._fail_startup_phase(phase_name, error)

    def current_startup_phase(self) -> str | None:
        return self._service._current_startup_phase()

    def set_effective_backend(self, backend: str) -> None:
        self._service._set_effective_backend(backend)

    def clear_watchman_monitoring_state(self) -> None:
        self._service._clear_watchman_monitoring_state()

    def activate_watchman_monitoring(
        self,
        *,
        scope_plan: WatchmanScopePlan,
        subscription_queue: Any,
        backend_name: str,
    ) -> None:
        self._service._activate_watchman_monitoring(
            scope_plan=scope_plan,
            subscription_queue=subscription_queue,
            backend_name=backend_name,
        )

    def clear_error_state(
        self,
        *,
        exact_messages: tuple[str, ...] = (),
        prefixes: tuple[str, ...] = (),
    ) -> None:
        self._service._clear_error_state(
            exact_messages=exact_messages,
            prefixes=prefixes,
        )

    def refresh_runtime_service_state(self) -> None:
        self._service._refresh_runtime_service_state()

    def utc_now(self) -> str:
        return self._service._utc_now()

    def start_transient_task(self, awaitable: Awaitable[Any]) -> Any:
        return self._service._start_transient_task(awaitable)

    async def request_resync(
        self, reason: str, details: dict[str, Any] | None = None
    ) -> bool:
        return await self._service.request_resync(reason, details)

    async def setup_watchdog_with_timeout(self, watch_path: Path, loop: Any) -> None:
        await self._service._setup_watchdog_with_timeout(watch_path, loop)

    async def start_polling_backend(
        self, watch_path: Path, reason: str, emit_warning: bool = True
    ) -> None:
        await self._service._start_polling_backend(watch_path, reason, emit_warning)

    async def cancel_watchdog_setup_task(self) -> None:
        await self._service._cancel_watchdog_setup_task()

    async def cancel_watchdog_bootstrap_future(self) -> None:
        await self._service._cancel_watchdog_bootstrap_future()

    async def stop_observer(self) -> None:
        await self._service._stop_observer()

    async def cancel_polling_task(self) -> None:
        await self._service._cancel_polling_task()

    def create_sidecar(self, target_dir: Path) -> Any:
        return self._sidecar_factory(target_dir, self._service._debug)

    def create_watchman_session(
        self,
        metadata: Any,
        sidecar: Any,
        overflow_handler: Callable[[dict[str, object], int, int], None] | None,
    ) -> Any:
        return self._session_factory(metadata, sidecar, overflow_handler)

    def discover_nested_linux_mount_roots(self, watch_path: Path) -> Any:
        return self._nested_mount_discoverer(watch_path)

    def discover_nested_windows_junction_scopes(self, watch_path: Path) -> Any:
        return self._junction_scope_discoverer(watch_path)

    def build_watchman_scope_plan(
        self,
        watch_path: Path,
        watch_project_response: dict[str, object],
        *,
        nested_mount_roots: Any,
        additional_scopes: Any,
    ) -> WatchmanScopePlan:
        return self._scope_plan_builder(
            watch_path,
            watch_project_response,
            nested_mount_roots=nested_mount_roots,
            additional_scopes=additional_scopes,
        )

    def build_watchman_subscription_name_for_scope(self, **kwargs: Any) -> str:
        return self._subscription_name_builder(**kwargs)

    def build_watchman_subscription_names_for_scope_plan(
        self, **kwargs: Any
    ) -> tuple[str, ...]:
        return self._subscription_names_builder(**kwargs)


def coerce_realtime_context(value: Any) -> RealtimeServiceContext:
    if isinstance(value, RealtimeServiceContext):
        return value
    context = getattr(value, "_realtime_context", None)
    if isinstance(context, RealtimeServiceContext):
        return context
    raise TypeError(
        "Realtime adapter requires RealtimeServiceContext or a service carrying "
        "a _realtime_context"
    )
