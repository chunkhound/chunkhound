"""Regression test: daemon-proxied MCP tool calls must record analytics.

`ChunkHoundDaemon` builds its own `AnalyticsRecorder` once at startup, via
the same `MCPServerBase.__init__` every other MCP transport uses (see
`chunkhound/mcp_server/base.py`'s "one recorder per server process,
matching the design's per-process buffer model" comment). But
`ChunkHoundDaemon._handle_tools_call()` used to forget to pass that
recorder into `handle_tool_call()`, silently defaulting to
`analytics_recorder=None` -- a documented no-op (see
`chunkhound/core/analytics/recorder.py`'s `start_command`/`end_command`).
Every daemon-proxied tool call was invisible to analytics despite the
recorder existing, contradicting the design's "hooking handle_tool_call()
once already covers MCP calls regardless of transport" decision, which
only actually held for the in-process stdio/HTTP path.
"""

import glob
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from chunkhound.core.config.config import Config
from chunkhound.daemon.server import ChunkHoundDaemon


def _read_events(buffer_dir: Path) -> list[dict]:
    events = []
    for path in glob.glob(str(buffer_dir / "buffer-*.jsonl")):
        for line in Path(path).read_text().splitlines():
            events.append(json.loads(line))
    return events


@pytest.fixture
def daemon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ChunkHoundDaemon, Path]:
    buffer_dir = tmp_path / "analytics"
    monkeypatch.setattr("chunkhound.core.analytics.recorder._ANALYTICS_DIR", buffer_dir)

    db_path = tmp_path / ".chunkhound" / "test.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    config = Config(
        target_dir=tmp_path,
        database={"path": str(db_path), "provider": "duckdb"},
        indexing={},
        analytics={
            "enabled": True,
            "flush_interval_seconds": 999999,
            "flush_batch_size": 999999,
        },
    )
    instance = ChunkHoundDaemon(
        config,
        args=MagicMock(path=tmp_path),
        socket_path=str(tmp_path / "test.sock"),
        project_dir=tmp_path,
    )
    # Real startup sets this once Watchman/services are ready; the tool-call
    # path under test just needs it set, not the full startup sequence.
    instance._initialization_complete.set()
    return instance, buffer_dir


@pytest.mark.asyncio
async def test_daemon_proxied_tool_call_records_one_command_summary(
    daemon: tuple[ChunkHoundDaemon, Path],
) -> None:
    instance, buffer_dir = daemon

    result = await instance._handle_tools_call(
        {"id": 1, "params": {"name": "daemon_status", "arguments": {}}}
    )

    assert result["result"]["isError"] is False
    events = _read_events(buffer_dir)
    assert len(events) == 1
    assert events[0]["command"] == "daemon_status"
    assert events[0]["source"] == "mcp"
    assert events[0]["success"] is True


@pytest.mark.asyncio
async def test_disabled_analytics_records_nothing_via_the_daemon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    buffer_dir = tmp_path / "analytics"
    monkeypatch.setattr("chunkhound.core.analytics.recorder._ANALYTICS_DIR", buffer_dir)

    db_path = tmp_path / ".chunkhound" / "test.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    config = Config(
        target_dir=tmp_path,
        database={"path": str(db_path), "provider": "duckdb"},
        indexing={},
    )
    instance = ChunkHoundDaemon(
        config,
        args=MagicMock(path=tmp_path),
        socket_path=str(tmp_path / "test.sock"),
        project_dir=tmp_path,
    )
    instance._initialization_complete.set()

    await instance._handle_tools_call(
        {"id": 1, "params": {"name": "daemon_status", "arguments": {}}}
    )

    assert not buffer_dir.exists()
