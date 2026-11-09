"""A burst of filtered events publishes a few status snapshots, not one each.

Each filtered event used to build a full health snapshot, which the daemon then
deep-copies: about 0.58 ms per event, 44% of a filtered event's cost.

The pipeline module's clock is frozen during each burst, so the number of
publishes does not depend on how fast the machine runs the burst.
"""

from __future__ import annotations

import asyncio
import copy
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chunkhound.core.config.config import Config
from chunkhound.database_factory import create_services
from chunkhound.services.realtime import pipeline
from chunkhound.services.realtime.service import RealtimeIndexingService

_BURST = 1000
_INTERVAL_SECONDS = 0.05


class _Published:
    def __init__(self) -> None:
        self.snapshots: list[dict[str, Any]] = []
        self.thread_ids: list[int] = []

    def __call__(self, snapshot: dict[str, Any]) -> None:
        self.snapshots.append(copy.deepcopy(snapshot))
        self.thread_ids.append(threading.get_ident())

    def filtered_counts(self) -> list[int]:
        return [s["pipeline"]["filtered_event_count"] for s in self.snapshots]


@pytest.fixture
async def service_and_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[RealtimeIndexingService, _Published, Path]]:
    monkeypatch.setattr(pipeline, "time", SimpleNamespace(monotonic=lambda: 1000.0))
    root = tmp_path / "repo"
    root.mkdir()
    db_path = tmp_path / "db.duckdb"
    config = Config(
        database={"path": str(db_path), "provider": "duckdb"},
        indexing={"realtime_backend": "watchdog"},
        target_dir=root,
    )
    published = _Published()
    services = create_services(db_path, config)
    services.provider.connect()
    service = RealtimeIndexingService(services, config, status_callback=published)
    service._FILTERED_STATUS_INTERVAL_SECONDS = _INTERVAL_SECONDS
    try:
        yield service, published, root
    finally:
        await service.stop()
        services.provider.disconnect()


def _record_burst(service: RealtimeIndexingService, root: Path) -> None:
    for i in range(_BURST):
        service._record_filtered_event("created", root / "node_modules" / f"f{i}.js")


async def _wait_for_publishes(published: _Published, count: int) -> None:
    deadline = time.monotonic() + 5
    while len(published.snapshots) < count and time.monotonic() < deadline:
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_filtered_burst_publishes_once_then_once_more_with_final_counts(
    service_and_published: tuple[RealtimeIndexingService, _Published, Path],
) -> None:
    service, published, root = service_and_published

    _record_burst(service, root)
    # The first event publishes at once; the rest of the interval waits.
    assert published.filtered_counts() == [1]

    await _wait_for_publishes(published, 2)
    await asyncio.sleep(_INTERVAL_SECONDS * 2)
    # One trailing publish carries the counters the burst's last event left.
    assert published.filtered_counts() == [1, _BURST]


@pytest.mark.asyncio
async def test_trailing_publish_runs_on_loop_while_event_pressure_changes(
    service_and_published: tuple[RealtimeIndexingService, _Published, Path],
) -> None:
    service, published, root = service_and_published
    loop_thread = threading.get_ident()

    _record_burst(service, root)
    # While the trailing publish is pending, the loop keeps adding and trimming
    # event-pressure entries, as it does under load.
    i = 0
    while len(published.snapshots) < 2 and i < 5000:
        service._record_source_event("modified", root / "src" / f"s{i}.py")
        i += 1
        await asyncio.sleep(0)

    assert published.filtered_counts() == [1, _BURST]
    assert published.thread_ids == [loop_thread, loop_thread]


@pytest.mark.asyncio
async def test_stop_cancels_pending_trailing_publish_and_carries_its_counts(
    service_and_published: tuple[RealtimeIndexingService, _Published, Path],
) -> None:
    service, published, root = service_and_published

    _record_burst(service, root)
    await service.stop()
    published_at_stop = len(published.snapshots)
    assert published.filtered_counts()[-1] == _BURST

    await asyncio.sleep(_INTERVAL_SECONDS * 4)
    assert len(published.snapshots) == published_at_stop

    # A filtered event after stop publishes again: no flush is left pending.
    service._record_filtered_event("created", root / "node_modules" / "late.js")
    assert published.filtered_counts()[-1] == _BURST + 1
