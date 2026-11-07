"""A burst of filtered events publishes a few status snapshots, not one each.

Each filtered event used to build a full health snapshot, which the daemon then
deep-copies: about 0.58 ms per event, 44% of a filtered event's cost.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

from chunkhound.core.config.config import Config
from chunkhound.database_factory import create_services
from chunkhound.services.realtime.service import RealtimeIndexingService


def test_filtered_burst_is_throttled_and_converges(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    db_path = tmp_path / "db.duckdb"
    config = Config(
        database={"path": str(db_path), "provider": "duckdb"},
        indexing={"realtime_backend": "watchdog"},
        target_dir=root,
    )
    published: list[dict] = []
    services = create_services(db_path, config)
    services.provider.connect()
    try:
        service = RealtimeIndexingService(
            services,
            config,
            status_callback=lambda s: published.append(copy.deepcopy(s)),
        )
        burst = 1000
        for i in range(burst):
            service._record_filtered_event(
                "created", root / "node_modules" / f"f{i}.js"
            )
        # About one publish per 0.5 s the burst spans (plus event-pressure state
        # changes), never one per event.
        assert len(published) <= burst // 20, len(published)

        # The burst is over: the last published counters reach their final value.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and (
            not published
            or published[-1]["pipeline"]["filtered_event_count"] != burst
        ):
            time.sleep(0.05)
        assert published[-1]["pipeline"]["filtered_event_count"] == burst
    finally:
        services.provider.disconnect()
