from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from chunkhound.core.config.config import Config
from chunkhound.database_factory import create_services
from chunkhound.services.realtime_indexing_service import RealtimeIndexingService
from tests.utils.windows_compat import wait_for_indexed


def _build_watchman_service(target_dir: Path) -> tuple[RealtimeIndexingService, object]:
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


def _subscription_pdu(*, name: str, exists: bool, is_new: bool) -> dict[str, object]:
    return {
        "subscription": "chunkhound-live-indexing",
        "clock": "c:0:1",
        "files": [
            {
                "name": name,
                "exists": exists,
                "new": is_new,
                "type": "f",
            }
        ],
    }


async def _wait_for_removed(service_provider: object, file_path: Path) -> bool:
    deadline = asyncio.get_running_loop().time() + 5.0
    while asyncio.get_running_loop().time() < deadline:
        record = service_provider.get_file_by_path(str(file_path))
        if record is None:
            return True
        await asyncio.sleep(0.1)
    return False


@pytest.mark.asyncio
async def test_watchman_subscription_pdu_indexes_created_file(tmp_path: Path) -> None:
    watch_dir = tmp_path / "watchman_project"
    watch_dir.mkdir(parents=True)
    service, services = _build_watchman_service(watch_dir)

    try:
        await service.start(watch_dir)
        queue = service.watchman_subscription_queue
        assert queue is not None

        file_path = watch_dir / "src" / "created.py"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("def created():\n    return 1\n", encoding="utf-8")

        queue.put_nowait(
            _subscription_pdu(name="src/created.py", exists=True, is_new=True)
        )

        assert await wait_for_indexed(services.provider, file_path)
    finally:
        await service.stop()
        services.provider.disconnect()


@pytest.mark.asyncio
async def test_watchman_subscription_pdu_deletes_indexed_file(tmp_path: Path) -> None:
    watch_dir = tmp_path / "watchman_project"
    watch_dir.mkdir(parents=True)
    service, services = _build_watchman_service(watch_dir)

    try:
        await service.start(watch_dir)
        queue = service.watchman_subscription_queue
        assert queue is not None

        file_path = watch_dir / "src" / "deleted.py"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("def deleted():\n    return 1\n", encoding="utf-8")

        await service.add_file(file_path, priority="priority")
        assert await wait_for_indexed(services.provider, file_path)

        file_path.unlink()
        queue.put_nowait(
            _subscription_pdu(name="src/deleted.py", exists=False, is_new=False)
        )

        assert await _wait_for_removed(services.provider, file_path)
    finally:
        await service.stop()
        services.provider.disconnect()


@pytest.mark.asyncio
async def test_watchman_relative_root_mapping_and_filtering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "w"
    target_dir = workspace / "p" / "a"
    target_dir.mkdir(parents=True)
    monkeypatch.setenv("CHUNKHOUND_FAKE_WATCHMAN_WATCH_ROOT", str(workspace.resolve()))
    monkeypatch.setenv("CHUNKHOUND_FAKE_WATCHMAN_RELATIVE_PATH", "p/a")

    service, services = _build_watchman_service(target_dir)

    try:
        await service.start(target_dir)
        queue = service.watchman_subscription_queue
        assert queue is not None

        included = target_dir / "src" / "mapped.py"
        included.parent.mkdir(parents=True, exist_ok=True)
        included.write_text("def mapped():\n    return 1\n", encoding="utf-8")

        excluded = target_dir / "src" / "ignored.xyz"
        excluded.write_text("ignored\n", encoding="utf-8")

        queue.put_nowait(
            {
                "subscription": "chunkhound-live-indexing",
                "clock": "c:0:2",
                "files": [
                    {
                        "name": "src/mapped.py",
                        "exists": True,
                        "new": True,
                        "type": "f",
                    },
                    {
                        "name": "src/ignored.xyz",
                        "exists": True,
                        "new": True,
                        "type": "f",
                    },
                ],
            }
        )

        assert await wait_for_indexed(services.provider, included)
        assert services.provider.get_file_by_path(str(excluded)) is None
    finally:
        await service.stop()
        services.provider.disconnect()
