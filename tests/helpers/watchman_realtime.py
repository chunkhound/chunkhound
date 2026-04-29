from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import chunkhound.services.realtime_indexing_service as realtime_service_module
import pytest

from chunkhound.core.config.config import Config
from chunkhound.database_factory import create_services
from chunkhound.services.realtime_indexing_service import RealtimeIndexingService


def build_watchman_service(target_dir: Path) -> tuple[RealtimeIndexingService, object]:
    db_path = target_dir / ".chunkhound" / "test.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    config = Config(
        args=SimpleNamespace(path=target_dir),
        database={"path": str(db_path), "provider": "duckdb"},
        indexing={"realtime_backend": "watchman"},
    )
    services = create_services(db_path, config)
    services.provider.connect()
    return RealtimeIndexingService(services, config), services


_build_watchman_service = build_watchman_service


async def wait_for_logical_indexed(service_provider: object, file_path: Path) -> bool:
    deadline = asyncio.get_running_loop().time() + 5.0
    while asyncio.get_running_loop().time() < deadline:
        record = service_provider.get_file_by_path(str(file_path))
        if record is not None:
            return True
        await asyncio.sleep(0.1)
    return False


_wait_for_logical_indexed = wait_for_logical_indexed


async def wait_for_removed(service_provider: object, file_path: Path) -> bool:
    deadline = asyncio.get_running_loop().time() + 5.0
    while asyncio.get_running_loop().time() < deadline:
        record = service_provider.get_file_by_path(str(file_path))
        if record is None:
            return True
        await asyncio.sleep(0.1)
    return False


_wait_for_removed = wait_for_removed


async def wait_for_watchman_reconnect_state(
    service: RealtimeIndexingService,
    expected_state: str,
    *,
    timeout: float = 10.0,
) -> dict[str, object]:
    async def _poll() -> dict[str, object]:
        while True:
            stats = await service.get_health()
            reconnect = stats.get("watchman_reconnect")
            if isinstance(reconnect, dict) and reconnect.get("state") == expected_state:
                return stats
            await asyncio.sleep(0.05)

    return await asyncio.wait_for(_poll(), timeout=timeout)


_wait_for_watchman_reconnect_state = wait_for_watchman_reconnect_state


def active_watchman_disconnect_process(adapter: object) -> object:
    session = getattr(adapter, "_session", None)
    process = getattr(session, "_process", None)
    if process is not None:
        return process

    sidecar = getattr(adapter, "_sidecar", None)
    sidecar_process = getattr(sidecar, "_process", None)
    if sidecar_process is not None:
        return sidecar_process

    raise AssertionError("No active Watchman process available to trigger disconnect")


def active_session_close_handle(adapter: object) -> object:
    session = getattr(adapter, "_session", None)
    process = getattr(session, "_process", None)
    stdin = getattr(process, "stdin", None)
    if stdin is not None:
        return stdin

    writer = getattr(session, "_stream_writer", None)
    if writer is not None:
        return writer

    raise AssertionError("No active Watchman session close handle is available")


def prepend_poisoned_python_shims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shims_dir = tmp_path / "poisoned-python"
    shims_dir.mkdir()
    if os.name == "nt":
        for name in ("python.cmd", "python3.cmd"):
            (shims_dir / name).write_text(
                "@echo off\r\nexit /b 97\r\n", encoding="utf-8"
            )
    else:
        for name in ("python", "python3"):
            shim_path = shims_dir / name
            shim_path.write_text("#!/bin/sh\nexit 97\n", encoding="utf-8")
            shim_path.chmod(0o755)
    current_path = os.environ.get("PATH", "")
    monkeypatch.setenv(
        "PATH",
        str(shims_dir)
        if not current_path
        else f"{shims_dir}{os.pathsep}{current_path}",
    )


async def start_isolated_watchman_translation(
    service: RealtimeIndexingService, target_dir: Path
) -> object:
    adapter = realtime_service_module.WatchmanRealtimeAdapter(service)
    primary_filter = realtime_service_module.RealtimePathFilter(
        config=service.config,
        root_path=target_dir,
    )
    adapter._path_filter = primary_filter
    adapter._scope_path_filters = {str(target_dir): primary_filter}
    service.watch_path = target_dir
    service._service_state = "running"
    service._effective_backend = "watchman"
    service.monitoring_ready.set()
    service._monitoring_ready_at = service._utc_now()
    service.event_consumer_task = asyncio.create_task(service._consume_events())
    service.process_task = asyncio.create_task(service._process_loop())
    return adapter
