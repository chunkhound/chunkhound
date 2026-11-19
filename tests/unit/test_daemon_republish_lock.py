"""A serving daemon must stay discoverable if its lock file vanishes.

The lock is the only way a proxy finds a daemon (it carries the auth token).
A daemon that kept serving after its lock and registry entry disappeared made
every new proxy start a second daemon, which then crashed on the held DB.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.core.config.config import Config
from chunkhound.daemon import server as daemon_server
from chunkhound.daemon.server import ChunkHoundDaemon


@pytest.mark.asyncio
async def test_serving_daemon_republishes_deleted_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHUNKHOUND_DAEMON_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("CHUNKHOUND_DAEMON_REGISTRY_DIR", str(tmp_path / "registry"))
    monkeypatch.setattr(daemon_server, "_PUBLICATION_CHECK_INTERVAL", 0.05)

    project_dir = tmp_path / "project"
    db_path = project_dir / ".chunkhound" / "test.db"
    db_path.parent.mkdir(parents=True)
    config = Config(
        target_dir=project_dir,
        database={"path": str(db_path), "provider": "duckdb"},
        indexing={},
    )
    socket_path = "tcp:127.0.0.1:1"
    daemon = ChunkHoundDaemon(
        config,
        args=MagicMock(path=project_dir),
        socket_path=socket_path,
        project_dir=project_dir,
    )

    async def fake_initialize() -> None:
        daemon._start_startup_phase("initialize")
        daemon._initialized = True
        daemon._scan_target_path = daemon._normalize_requested_target_path(
            project_dir
        )
        daemon._startup_publish_complete.clear()
        daemon._deferred_start_task = asyncio.create_task(asyncio.sleep(0))
        daemon._startup_failure_message = None

    async def fake_barrier() -> None:
        daemon._start_startup_phase("startup_barrier")

    daemon.initialize = fake_initialize  # type: ignore[assignment]
    daemon.await_startup_barrier = fake_barrier  # type: ignore[assignment]

    mock_server = AsyncMock()
    mock_server.__aenter__ = AsyncMock(return_value=None)
    mock_server.__aexit__ = AsyncMock(return_value=None)

    discovery = daemon._discovery
    lock_path = discovery.get_lock_path()
    registry_path = discovery.get_registry_entry_path()

    async def wait_for(path: Path) -> None:
        for _ in range(100):
            if path.exists():
                return
            await asyncio.sleep(0.02)
        raise AssertionError(f"{path} did not appear")

    with patch(
        "chunkhound.daemon.server.ipc.create_server", new_callable=AsyncMock
    ) as create_server:
        create_server.return_value = (mock_server, socket_path)
        run_task = asyncio.create_task(daemon.run())
        try:
            await wait_for(lock_path)
            await wait_for(registry_path)
            lock_path.unlink()
            registry_path.unlink()

            await wait_for(lock_path)
            await wait_for(registry_path)
            lock = discovery.read_lock()
            assert lock is not None
            assert lock["pid"] == os.getpid()
            assert lock["auth_token"] == daemon._auth_token
            assert lock["socket_path"] == socket_path
        finally:
            daemon._shutdown_event.set()
            await asyncio.wait_for(run_task, timeout=15)

    assert not lock_path.exists()
