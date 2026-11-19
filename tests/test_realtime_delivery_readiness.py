"""Readiness must mean usable filesystem delivery, not just a live thread."""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from loguru import logger
from watchdog.events import FileCreatedEvent
from watchdog.observers import Observer

from chunkhound.services.realtime import startup
from chunkhound.services.realtime.delivery_probe import PROBE_FILENAME
from tests.helpers.realtime_harness import RealtimeHarness, realtime_harness
from tests.helpers.search_contract import (
    results_are_empty,
    search_tool,
    wait_for_health,
    wait_for_search,
)


@dataclass
class ProbeWrites:
    opened: asyncio.Event = field(default_factory=asyncio.Event)
    paths: list[Path] = field(default_factory=list)
    failure: str | None = None


def observe_probe_writes(
    monkeypatch: pytest.MonkeyPatch,
    on_probe_write: Any = None,
) -> ProbeWrites:
    """Observe the filesystem boundary without replacing probe or event logic."""
    writes = ProbeWrites()
    loop = asyncio.get_running_loop()
    original_open = Path.open

    def open_path(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path.name == PROBE_FILENAME:
            writes.paths.append(path)
            if writes.failure:
                raise PermissionError(writes.failure)
            result = original_open(path, *args, **kwargs)
            loop.call_soon_threadsafe(writes.opened.set)
            if on_probe_write is not None:
                on_probe_write(path)
            return result
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_path)
    return writes


def observe_probe_cleanup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail probe-file cleanup only; delivery and indexing stay real."""
    original_unlink = Path.unlink

    def unlink(path: Path, *args: Any, **kwargs: Any) -> None:
        if path.name == PROBE_FILENAME:
            raise PermissionError("delivery-contract-cleanup-denied")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)


def discard_native_delivery(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Keep real native emitters alive, but lose delivery at the OS boundary."""
    observers: list[Any] = []

    class NonDeliveringObserver(Observer):
        def dispatch_events(self, event_queue: Any) -> None:
            event_queue.get(block=True, timeout=self.timeout)
            event_queue.task_done()

    def create_observer(on_overflow: Any) -> Any:
        observer = NonDeliveringObserver()
        observers.append(observer)
        return observer, "active"

    monkeypatch.setattr(startup, "_create_watchdog_observer", create_observer)
    return observers


class FakeNativeEmitter:
    """Emitter surface the supervision port reads: liveness + dropped batches."""

    def __init__(self, observer: FakeNativeObserver) -> None:
        self._observer = observer
        self._alive = False
        self.dropped_event_batches = 0

    def is_alive(self) -> bool:
        return self._alive

    def stop(self) -> None:
        self._alive = False

    def join(self, timeout: float | None = None) -> None:
        return None

    def queue_event(self, event: Any) -> None:
        self._observer.offer(event)


class FakeNativeObserver:
    """Monitoring-port fake behind the RealtimeObserver contract.

    Adopts the service's real event handler and dispatches injected events on
    its own dispatch thread, so fault injection (dead stream, dropped batches)
    and recovery outcomes stay valid against any native backend.
    """

    _STOP: Any = object()

    def __init__(self) -> None:
        self.emitters: list[FakeNativeEmitter] = []
        self._handlers: list[Any] = []
        self._queue: queue.Queue[Any] = queue.Queue()
        self._alive = False
        self._thread: threading.Thread | None = None

    def schedule(
        self,
        event_handler: Any,
        path: str,
        recursive: bool = True,
        event_filter: list[type] | None = None,
    ) -> Any:
        self._handlers.append(event_handler)
        self.emitters.append(FakeNativeEmitter(self))
        return (path, recursive)

    def start(self) -> None:
        self._alive = True
        for emitter in self.emitters:
            emitter._alive = True
        self._thread = threading.Thread(target=self._dispatch_loop, daemon=True)
        self._thread.start()

    def is_alive(self) -> bool:
        return self._alive

    def stop(self) -> None:
        self._alive = False
        self.offer(self._STOP)

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def offer(self, event: Any) -> None:
        """Inject one native event; the dispatch thread hands it to the handler."""
        self._queue.put(event)

    def _dispatch_loop(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                if not self._alive:
                    return
                continue
            if item is self._STOP:
                return
            for handler in list(self._handlers):
                handler.dispatch(item)


def controlled_native_observer(monkeypatch: pytest.MonkeyPatch) -> FakeNativeObserver:
    """Replace the native-observer factory with the port-level fake."""
    observer = FakeNativeObserver()
    monkeypatch.setattr(
        startup,
        "_create_watchdog_observer",
        lambda on_overflow: (observer, "active"),
    )
    return observer


def _make_ping_feed(observer: FakeNativeObserver, *, once: bool = False) -> Any:
    """Route real probe-file writes into the fake stream (healthy delivery).

    ``once`` delivers only the first write (startup verification) and keeps
    later pings undelivered — a verified-but-subsequently-deaf stream.
    """
    fed = threading.Event()

    def on_probe_write(path: Path) -> None:
        if once and not fed.is_set():
            fed.set()
            observer.offer(FileCreatedEvent(str(path)))
            return
        if not once:
            observer.offer(FileCreatedEvent(str(path)))

    return on_probe_write


@asynccontextmanager
async def controlled_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    indexing: dict[str, Any] | None = None,
    deliver_pings: bool = True,
):
    observer = controlled_native_observer(monkeypatch)
    on_probe_write = _make_ping_feed(observer, once=not deliver_pings)
    writes = observe_probe_writes(monkeypatch, on_probe_write=on_probe_write)
    async with realtime_harness(
        tmp_path,
        include=("*.py",),
        exclude=(PROBE_FILENAME,),
        realtime_backend="watchdog",
        indexing=indexing,
    ) as harness:
        yield harness, observer, writes


async def assert_native_probe_is_private(
    harness: RealtimeHarness, writes: ProbeWrites
) -> None:
    health = await harness.realtime_service.get_health()
    assert health["effective_backend"] == "watchdog", health
    assert health["monitoring_ready"], health
    assert health["pipeline"]["last_source_event_at"] is None, health
    assert health["pipeline"]["last_source_event_path"] is None, health
    await assert_no_probe_artifacts(harness, writes)


async def assert_immediate_search(harness: RealtimeHarness) -> None:
    target = harness.watch_dir / "ready_contract.py"
    target.write_text("def delivery_contract_marker():\n    return 'ready'\n")
    result = await wait_for_search(
        harness.services,
        "delivery_contract_marker",
        lambda output: "ready_contract.py" in output,
    )
    assert "delivery_contract_marker" in result


async def assert_no_probe_artifacts(
    harness: RealtimeHarness, writes: ProbeWrites
) -> None:
    # Supervision pings reuse the probe path; quiesce first so the artifact
    # assertions cannot race an in-flight ping file.
    await harness.realtime_service._cancel_supervision_task()
    assert writes.paths, "Native startup must attempt a filesystem delivery probe"
    assert not list(harness.watch_dir.glob(PROBE_FILENAME))
    for path in writes.paths:
        assert harness.services.provider.get_file_by_path(str(path)) is None
    assert results_are_empty(
        await search_tool(harness.services, "ChunkHound filesystem delivery probe")
    )


async def assert_polling_fallback(harness: RealtimeHarness) -> dict[str, Any]:
    health = await harness.realtime_service.get_health()
    assert health["effective_backend"] == "polling", health
    assert health["monitoring_ready"], health
    assert health["last_warning"], health
    # Recovery must surface as a warning; a bootstrap failure that falls back
    # successfully must never poison live_indexing_state into "degraded".
    assert health["service_state"] == "running", health
    assert health["last_error"] is None, health
    assert health["live_indexing_state"] in {"idle", "busy"}, health
    return health


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("include", "exclude"),
    [(("*",), ()), (("*.py",), (PROBE_FILENAME,))],
)
async def test_native_start_is_searchable_and_probe_never_enters_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    include: tuple[str, ...],
    exclude: tuple[str, ...],
) -> None:
    writes = observe_probe_writes(monkeypatch)
    async with realtime_harness(
        tmp_path,
        include=include,
        exclude=exclude,
        realtime_backend="watchdog",
    ) as harness:
        # Native delivery is OS-dependent; a verified polling fallback is valid.
        await assert_immediate_search(harness)
        health = await harness.realtime_service.get_health()
        assert health["monitoring_ready"], health
        assert health["effective_backend"] in {"watchdog", "polling"}, health
        await assert_no_probe_artifacts(harness, writes)


@pytest.mark.asyncio
async def test_native_probe_bypasses_restrictive_patterns_without_source_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with controlled_harness(tmp_path, monkeypatch) as (harness, _, writes):
        await assert_native_probe_is_private(harness, writes)


@pytest.mark.asyncio
async def test_live_native_observer_without_delivery_falls_back_to_searchable_polling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writes = observe_probe_writes(monkeypatch)
    observers = discard_native_delivery(monkeypatch)
    async with realtime_harness(tmp_path, realtime_backend="watchdog") as harness:
        await assert_immediate_search(harness)
        health = await assert_polling_fallback(harness)
        # The timeout-specific fallback must win over the generic setup failure,
        # which would embed the same exception text and pass by accident.
        assert health["last_warning"].startswith(
            "Watchdog setup or filesystem delivery probe timed out"
        ), health
        assert len(observers) == 1
        assert not observers[0].is_alive()
        await assert_no_probe_artifacts(harness, writes)


@pytest.mark.asyncio
async def test_probe_write_failure_falls_back_to_searchable_polling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writes = observe_probe_writes(monkeypatch)
    writes.failure = "delivery-contract-write-denied"
    async with realtime_harness(tmp_path, realtime_backend="watchdog") as harness:
        await assert_immediate_search(harness)
        health = await assert_polling_fallback(harness)
        assert writes.failure in health["last_warning"], health
        await assert_no_probe_artifacts(harness, writes)


async def test_crash_leftover_probe_artifact_is_replaced_and_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file leaked by a killed process must not block or outlive the next start."""
    stale = tmp_path / "project" / PROBE_FILENAME
    stale.parent.mkdir()
    stale.write_text("crash leftover")
    async with controlled_harness(tmp_path, monkeypatch) as (harness, _, writes):
        assert stale in writes.paths
        health = await harness.realtime_service.get_health()
        assert health["effective_backend"] == "watchdog", health
    assert not stale.exists()


async def interrupt_probe_startup(
    harness: RealtimeHarness, writes: ProbeWrites, observers: list[Any], action: str
) -> None:
    service = harness.realtime_service
    start_task = asyncio.create_task(service.start(harness.watch_dir))
    try:
        await asyncio.wait_for(writes.opened.wait(), timeout=10)
        assert observers[0].is_alive(), "The probe must run on a live native observer"
        assert not service.monitoring_ready.is_set()
        if action == "cancel":
            start_task.cancel()
        await asyncio.wait_for(service.stop(), timeout=10)
        with pytest.raises(asyncio.CancelledError):
            await start_task
    finally:
        if not start_task.done():
            start_task.cancel()
        await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["cancel", "stop"])
async def test_interrupting_delivery_probe_cleans_workspace_and_native_observer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    # Reuse the real harness's connected services for an interrupted second start.
    async with realtime_harness(tmp_path, realtime_backend="polling") as harness:
        await harness.realtime_service.stop()
        harness.realtime_service.config.indexing.realtime_backend = "watchdog"
        writes = observe_probe_writes(monkeypatch)
        observers = discard_native_delivery(monkeypatch)
        await interrupt_probe_startup(harness, writes, observers, action)
        assert len(observers) == 1
        assert not observers[0].is_alive()
        assert not harness.realtime_service.monitoring_ready.is_set()
        await assert_no_probe_artifacts(harness, writes)


@pytest.mark.asyncio
async def test_probe_cleanup_failure_keeps_verified_watchdog_observer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A leaked probe file must not demote a verified observer to polling."""
    observe_probe_cleanup_failure(monkeypatch)
    async with controlled_harness(tmp_path, monkeypatch) as (harness, observer, _):
        health = await harness.realtime_service.get_health()
        assert health["effective_backend"] == "watchdog", health
        assert health["monitoring_ready"], health
        assert health["service_state"] == "running", health
        assert health["last_error"] is None, health

        target = harness.watch_dir / "cleanup_survivor.py"
        target.write_text("def cleanup_survivor_marker():\n    return 'ok'\n")
        (emitter,) = observer.emitters
        emitter.queue_event(FileCreatedEvent(str(target)))
        result = await wait_for_search(
            harness.services,
            "cleanup_survivor_marker",
            lambda output: "cleanup_survivor.py" in output,
        )
        assert "cleanup_survivor_marker" in result


@pytest.mark.asyncio
async def test_dead_watchdog_emitters_fall_back_to_polling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dead emitter stream goes deaf silently; recovery must keep writes indexed."""
    async with controlled_harness(
        tmp_path,
        monkeypatch,
        indexing={"realtime_supervision_interval_seconds": 0.05},
    ) as (harness, observer, _):
        service = harness.realtime_service
        health = await service.get_health()
        assert health["effective_backend"] == "watchdog", health
        (emitter,) = observer.emitters
        emitter.stop()
        assert not emitter.is_alive(), "The stream must be dead for the test"
        assert observer.is_alive(), "A live dispatch thread alone is not healthy"

        target = harness.watch_dir / "after_emitter_death.py"
        target.write_text("def emitter_death_recovery_marker():\n    return 'ok'\n")
        result = await wait_for_search(
            harness.services,
            "emitter_death_recovery_marker",
            lambda output: "after_emitter_death.py" in output,
        )
        assert "emitter_death_recovery_marker" in result

        health = await wait_for_health(
            service, lambda h: h["effective_backend"] == "polling"
        )
        assert health["live_indexing_state"] in {"idle", "busy"}, health


@pytest.mark.asyncio
async def test_dropped_native_batch_rescans_missed_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A native lost-window signal must reconcile missed files, keeping watchdog."""
    async with controlled_harness(
        tmp_path,
        monkeypatch,
        indexing={"realtime_supervision_interval_seconds": 0.05},
    ) as (harness, observer, _):
        target = harness.watch_dir / "dropped.py"
        target.write_text("def dropped_batch_marker():\n    return 'recovered'\n")
        assert results_are_empty(
            await search_tool(harness.services, "dropped_batch_marker")
        )
        (emitter,) = observer.emitters
        emitter.dropped_event_batches = 1
        result = await wait_for_search(
            harness.services,
            "dropped_batch_marker",
            lambda output: "dropped.py" in output,
        )
        assert "dropped_batch_marker" in result
        health = await harness.realtime_service.get_health()
        assert health["effective_backend"] == "watchdog", health


@pytest.mark.asyncio
async def test_failed_dropped_batch_rescan_backs_off_then_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed rescan backs off, keeps the window, and retries to recovery."""
    async with controlled_harness(tmp_path, monkeypatch) as (harness, observer, _):
        await harness.realtime_service._cancel_supervision_task()
        (harness.watch_dir / "retried.py").write_text(
            "def retried_rescan_marker():\n    return 'ok'\n"
        )
        (emitter,) = observer.emitters
        emitter.dropped_event_batches = 1
        failures = [OSError("transient rescan failure")]
        real_rescan = harness.realtime_service._request_dir_rescan

        async def flaky_rescan(reason: str) -> None:
            if failures:
                raise failures.pop()
            await real_rescan(reason)

        service = harness.realtime_service
        monkeypatch.setattr(service, "_request_dir_rescan", flaky_rescan)
        assert await service._supervision_tick() is True
        assert "rescan failed" in (await service.get_health())["last_warning"]
        assert service._dropped_rescan_failures == 1
        # The backoff suppresses the immediate next tick; the window survives.
        service._next_dropped_rescan_attempt_at = float("inf")
        assert await service._supervision_tick() is True
        assert service._dropped_rescan_failures == 1
        # Once the backoff elapses the retry succeeds and reconciles the window.
        service._next_dropped_rescan_attempt_at = 0.0
        assert await service._supervision_tick() is True
        result = await wait_for_search(
            harness.services, "retried_rescan_marker", lambda out: "retried.py" in out
        )
        assert "retried_rescan_marker" in result


@pytest.mark.asyncio
async def test_healthy_watchdog_pings_only_at_the_slow_cadence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A proven-healthy stream must not ping the probe file every tick."""
    async with controlled_harness(tmp_path, monkeypatch) as (harness, _, _):
        service = harness.realtime_service
        await service._cancel_supervision_task()
        probe = service._delivery_probe
        pings: list[bool] = []
        probe.ping = lambda _window: pings.append(True) or True  # type: ignore[method-assign]

        service._last_ping_at = None
        assert await service._supervision_tick() is True
        assert len(pings) == 1
        # A healthy tick within the slow interval must not ping again.
        service._last_ping_at = time.monotonic()
        assert await service._supervision_tick() is True
        assert len(pings) == 1
        # After the slow interval elapses the heartbeat resumes.
        service._last_ping_at = 0.0
        assert await service._supervision_tick() is True
        assert len(pings) == 2


@pytest.mark.asyncio
async def test_native_errors_reach_application_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    messages = []
    sink = logger.add(messages.append, format="{message}")
    try:
        async with controlled_harness(tmp_path, monkeypatch):
            logging.getLogger("fsevents").error("native_delivery_error_marker")
        assert any(
            "native_delivery_error_marker" in str(message) for message in messages
        )
    finally:
        logger.remove(sink)


@pytest.mark.asyncio
async def test_transient_ping_miss_keeps_watchdog_live_and_indexes_later_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One delayed FSEvents batch must not downgrade a healthy stream to polling."""
    async with controlled_harness(tmp_path, monkeypatch) as (harness, observer, _):
        service = harness.realtime_service
        await service._cancel_supervision_task()
        service._last_ping_at = None
        probe = service._delivery_probe
        outcomes = [False, True]
        probe.ping = lambda _window: outcomes.pop(0)  # type: ignore[method-assign]

        assert await service._supervision_tick() is True
        health = await service.get_health()
        assert health["effective_backend"] == "watchdog", health
        assert health["delivery_pings"]["consecutive_misses"] == 1, health
        assert health["last_warning"] is None, health

        assert await service._supervision_tick() is True
        health = await service.get_health()
        assert health["effective_backend"] == "watchdog", health
        assert health["delivery_pings"]["consecutive_misses"] == 0, health
        assert health["delivery_pings"]["last_delivered"] is True, health

        # The stream is still connected: a native event indexes normally.
        target = harness.watch_dir / "after_transient_miss.py"
        target.write_text("def transient_miss_marker():\n    return 'ok'\n")
        (emitter,) = observer.emitters
        emitter.queue_event(FileCreatedEvent(str(target)))
        result = await wait_for_search(
            harness.services,
            "transient_miss_marker",
            lambda output: "after_transient_miss.py" in output,
        )
        assert "transient_miss_marker" in result


@pytest.mark.asyncio
async def test_silent_native_stream_falls_back_to_polling_via_ping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A verified probe does not guarantee future delivery; pings must catch deafness.

    macOS 26 FSEvents streams can adopt successfully (probe verified) and then
    silently stop delivering. Thread liveness cannot detect that shape; the
    supervision ping must, and polling must fully catch up afterwards. A tick
    that raises must not disable that detection.
    """
    async with controlled_harness(
        tmp_path,
        monkeypatch,
        indexing={"realtime_supervision_interval_seconds": 0.01},
    ) as (harness, _, _):
        service = harness.realtime_service
        health = await service.get_health()
        assert health["effective_backend"] == "watchdog", health

        await service._cancel_supervision_task()
        service._last_ping_at = None
        probe, calls = service._delivery_probe, 0

        def flaky_ping(_window: float) -> bool:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("transient supervision failure")
            return False

        probe.ping = flaky_ping  # type: ignore[method-assign]
        await service._start_monitor_supervision()

        # The controlled emitter stays alive but never delivers again.
        target = harness.watch_dir / "after_silence.py"
        target.write_text("def silent_stream_recovery_marker():\n    return 'ok'\n")
        result = await wait_for_search(
            harness.services,
            "silent_stream_recovery_marker",
            lambda output: "after_silence.py" in output,
        )
        assert "silent_stream_recovery_marker" in result

        health = await service.get_health()
        assert health["effective_backend"] == "polling", health
        assert health["live_indexing_state"] in {"idle", "busy"}, health
        assert (
            health["delivery_pings"]["consecutive_misses"]
            >= health["delivery_pings"]["max_consecutive_misses"]
        ), health
        task = service._monitor_supervision_task
        await asyncio.wait_for(task, timeout=10)
        assert task is not None and task.exception() is None
        # The first miss raised; sustained misses came from ping results.
        assert calls >= 3


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.name == "nt", reason="symlink creation requires privileges on Windows"
)
async def test_ping_never_follows_a_symlink_at_the_probe_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A symlink planted at the probe path must never touch its target file."""
    outside = tmp_path / "outside.txt"
    outside.write_text("precious original content\n")

    async with controlled_harness(
        tmp_path, monkeypatch, deliver_pings=False
    ) as (harness, _, _):
        service = harness.realtime_service
        await service._cancel_supervision_task()
        probe = service._delivery_probe
        probe.remove()
        probe.path.symlink_to(outside)
        try:
            assert probe.ping(0.05) is False
            assert outside.read_text() == "precious original content\n"
            assert not probe.path.exists(), "ping must clean up its own probe file"
        finally:
            probe.remove()


@pytest.mark.asyncio
async def test_shutdown_waits_for_in_flight_probe_ping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """stop() must release the probe file before returning (no restart overlap)."""
    async with controlled_harness(
        tmp_path,
        monkeypatch,
        indexing={"realtime_supervision_interval_seconds": 0.02},
        deliver_pings=False,
    ) as (harness, _, _):
        service = harness.realtime_service
        # Supervision is auto-started on adoption with the config cadence above,
        # so the ping is in flight when stop() is called.
        service._DELIVERY_PING_WINDOW_SECONDS = 1.0
        service._last_ping_at = None

        probe_path = harness.watch_dir.resolve() / PROBE_FILENAME
        deadline = time.monotonic() + 10.0
        while not probe_path.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("supervision ping never opened the probe path")
            await asyncio.sleep(0.01)

        await service.stop()
        assert not probe_path.exists()
