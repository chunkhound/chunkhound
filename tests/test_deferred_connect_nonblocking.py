"""Regression test: deferred DB connect does not block the asyncio loop.

The MCP daemon/proxy registration handshake depends on the daemon's event loop
remaining responsive during startup. DuckDB initialization can be slow
(database open, extension install, WAL recovery). If `provider.connect()` runs
on the event loop thread, the daemon can't accept/register clients and the
proxy times out waiting for the `registered` ack.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from chunkhound.core.config.config import Config
from chunkhound.database_factory import DatabaseServices
from chunkhound.mcp_server.base import MCPServerBase


class _TestServer(MCPServerBase):
    def _register_tools(self) -> None:  # pragma: no cover - not needed for test
        return

    async def run(self) -> None:  # pragma: no cover - not needed for test
        return


class _DummyProvider:
    def __init__(self, on_connect) -> None:
        self._on_connect = on_connect

    @property
    def is_connected(self) -> bool:
        return False

    def connect(self) -> None:
        self._on_connect()


@pytest.mark.asyncio
async def test_deferred_connect_runs_provider_connect_off_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Avoid inheriting host environment ChunkHound config (DB paths, etc).
    for key in list(os.environ):
        if key.startswith("CHUNKHOUND_"):
            monkeypatch.delenv(key, raising=False)

    config = Config(
        target_dir=tmp_path,
        database={"path": tmp_path / ".chunkhound" / "db", "provider": "duckdb"},
        indexing={"include": ["*.py"]},
    )

    server = _TestServer(config=config, args=SimpleNamespace(path=tmp_path))

    loop_thread_id = threading.get_ident()
    connect_thread_id: int | None = None

    def _on_connect() -> None:
        nonlocal connect_thread_id
        connect_thread_id = threading.get_ident()
        time.sleep(0.1)
        raise RuntimeError("intentional connect failure (test)")

    server.services = DatabaseServices(
        provider=_DummyProvider(_on_connect),
        indexing_coordinator=object(),
        search_service=object(),
        embedding_service=object(),
    )

    await server._deferred_connect_and_start(tmp_path)

    assert connect_thread_id is not None, "provider.connect() was not invoked"
    assert connect_thread_id != loop_thread_id, (
        "provider.connect() ran on the event loop thread; this can stall daemon IPC"
    )
