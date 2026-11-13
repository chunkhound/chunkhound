"""A status read reports the realtime state as of now.

The realtime service publishes its snapshot only when it records something.
Between publishes the event queue can drain and time can pass, so a status read
that served the published snapshot would show a drained queue as `busy` until
the next filesystem event, and could never show a stall past the threshold.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from chunkhound.core.config.config import Config
from chunkhound.database_factory import create_services
from chunkhound.mcp_server.base import MCPServerBase
from chunkhound.mcp_server.status import derive_daemon_status
from chunkhound.providers.database.serial_database_provider import (
    SerialDatabaseProvider,
)
from chunkhound.services.realtime.service import RealtimeIndexingService


class _Server(MCPServerBase):
    def _register_tools(self) -> None:
        pass

    async def run(self) -> None:
        pass


async def _started() -> None:
    pass


@pytest.fixture
async def server_and_service(
    tmp_path: Path,
) -> AsyncIterator[tuple[_Server, RealtimeIndexingService, Path]]:
    root = tmp_path / "repo"
    root.mkdir()
    db_path = tmp_path / "db.duckdb"
    config = Config(
        database={"path": str(db_path), "provider": "duckdb"},
        indexing={"realtime_backend": "watchdog"},
        target_dir=root,
    )
    server = _Server(config)
    services = create_services(db_path, config)
    services.provider.connect()
    service = RealtimeIndexingService(
        services, config, status_callback=server._update_realtime_status
    )
    # A service whose start completed with monitoring up, as after query_ready.
    service._service_state = "running"
    service._effective_backend = "watchdog"
    service.monitoring_ready.set()
    server.realtime_indexing = service
    server._realtime_start_task = asyncio.create_task(_started())
    await server._realtime_start_task
    try:
        yield server, service, root
    finally:
        await service.stop()
        services.provider.disconnect()


def _realtime(server: _Server) -> dict:
    """What the daemon_status tool is given, as both tool dispatchers pass it."""
    return server._scan_progress_for_tool("daemon_status")["realtime"]


@pytest.mark.asyncio
async def test_status_read_reflects_state_since_the_last_publish(
    server_and_service: tuple[_Server, RealtimeIndexingService, Path],
) -> None:
    server, service, root = server_and_service
    path = root / "module.py"
    # A change is queued and published; then nothing processes it.
    service.event_queue.put_nowait(("modified", path))
    service._handle_queue_result("modified", path, True, None)
    assert server._scan_progress["realtime"]["live_indexing_state"] == "busy"

    # Time passes past the stall threshold with no publish.
    stale = datetime.now(timezone.utc) - timedelta(
        seconds=service._STALL_THRESHOLD_SECONDS + 30
    )
    service._last_accepted_event_at = stale.replace(microsecond=0).isoformat()
    realtime = _realtime(server)
    assert realtime["live_indexing_state"] == "stalled"
    assert realtime["event_queue"]["size"] == 1

    # The queue drains with no publish.
    service.event_queue.get_nowait()
    service.event_queue.task_done()
    realtime = _realtime(server)
    assert realtime["live_indexing_state"] == "idle"
    assert realtime["event_queue"]["size"] == 0


@pytest.mark.asyncio
async def test_status_read_keeps_a_recorded_realtime_failure(
    server_and_service: tuple[_Server, RealtimeIndexingService, Path],
) -> None:
    server, _service, _root = server_and_service
    server._record_realtime_failure("watcher lost")

    realtime = _realtime(server)
    assert realtime["service_state"] == "degraded"
    assert realtime["last_error"] == "watcher lost"


def _scan_progress_after(
    server: _Server, monkeypatch: pytest.MonkeyPatch, seconds: float
) -> dict:
    """Read the daemon_status scan progress as if ``seconds`` had passed."""
    real = time.monotonic
    with monkeypatch.context() as patch:
        patch.setattr(time, "monotonic", lambda: real() + seconds)
        return server._scan_progress_for_tool("daemon_status")


@pytest.mark.asyncio
async def test_hung_rust_hold_reads_stalled(
    server_and_service: tuple[_Server, RealtimeIndexingService, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Rust run holding the database without progress reads stalled.

    The initial scan's hold leaves the backlog empty, which read idle.
    """
    server, service, _root = server_and_service
    provider = service.services.provider
    assert isinstance(provider, SerialDatabaseProvider)
    provider.set_rust_pipeline_in_progress(True)
    try:
        provider.note_rust_pipeline_progress("write-data")
        progress = _scan_progress_after(server, monkeypatch, 601)
        realtime = progress["realtime"]
        assert realtime["live_indexing_state"] == "stalled"
        assert "holding the database" in realtime["live_indexing_hint"]
        assert "last phase: write-data" in realtime["live_indexing_hint"]
        assert derive_daemon_status(progress)["status"] == "degraded"

        progress = _scan_progress_after(server, monkeypatch, 590)
        assert progress["realtime"]["live_indexing_state"] == "idle"

        # A single native step reports nothing until it ends.
        provider.note_rust_pipeline_progress("write-compact")
        progress = _scan_progress_after(server, monkeypatch, 700)
        assert progress["realtime"]["live_indexing_state"] == "idle"
    finally:
        provider.set_rust_pipeline_in_progress(False)


@pytest.mark.asyncio
async def test_late_main_report_keeps_the_store_step_ceiling(
    server_and_service: tuple[_Server, RealtimeIndexingService, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A parse report after the store starts an index build keeps its ceiling.

    After a disk-limit trip the store thread starts building indexes while the
    parse and embed threads can still report their in-flight batch.
    """
    server, service, _root = server_and_service
    provider = service.services.provider
    assert isinstance(provider, SerialDatabaseProvider)
    provider.set_rust_pipeline_in_progress(True)
    try:
        provider.note_rust_pipeline_progress("write-index")
        provider.note_rust_pipeline_progress("parse")
        progress = _scan_progress_after(server, monkeypatch, 700)
        assert progress["realtime"]["live_indexing_state"] == "idle"
    finally:
        provider.set_rust_pipeline_in_progress(False)


@pytest.mark.asyncio
async def test_hold_before_the_store_reports_uses_the_long_ceiling(
    server_and_service: tuple[_Server, RealtimeIndexingService, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hold whose store thread has not reported gets the single-step ceiling.

    The store thread opens the database first, which can rebuild missing
    vector indexes without reporting.
    """
    server, service, _root = server_and_service
    provider = service.services.provider
    assert isinstance(provider, SerialDatabaseProvider)
    provider.set_rust_pipeline_in_progress(True)
    try:
        provider.note_rust_pipeline_progress("parse")
        provider.note_rust_pipeline_progress("embed")
        progress = _scan_progress_after(server, monkeypatch, 700)
        assert progress["realtime"]["live_indexing_state"] == "idle"

        progress = _scan_progress_after(server, monkeypatch, 1321)
        realtime = progress["realtime"]
        assert realtime["live_indexing_state"] == "stalled"
        assert "last phase: embed" in realtime["live_indexing_hint"]
    finally:
        provider.set_rust_pipeline_in_progress(False)
