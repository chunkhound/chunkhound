import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from chunkhound.providers.database.serial_executor import (
    DatabaseCompactionInProgressError,
)
from chunkhound.services.realtime.service import RealtimeIndexingService


@pytest.fixture
async def service():
    svc = RealtimeIndexingService(
        services=MagicMock(),
        config=MagicMock(),
    )
    yield svc
    await svc.stop()


async def _wait_for(predicate, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met before timeout")


@pytest.mark.asyncio
async def test_change_processing_retries_when_compaction_is_active(
    service: RealtimeIndexingService,
    tmp_path: Path,
) -> None:
    target_file = tmp_path / "busy_change.py"
    target_file.write_text("def busy_change():\n    return 1\n", encoding="utf-8")

    attempts = 0
    processed = asyncio.Event()

    async def flaky_process_file(
        file_path: Path, skip_embeddings: bool = False
    ) -> dict[str, int]:
        nonlocal attempts
        attempts += 1
        assert file_path == target_file
        assert skip_embeddings is True
        if attempts == 1:
            raise DatabaseCompactionInProgressError("busy")
        processed.set()
        return {"chunks": 1, "embeddings": 0}

    service.services = SimpleNamespace(
        indexing_coordinator=SimpleNamespace(process_file=flaky_process_file),
        provider=SimpleNamespace(flush=AsyncMock()),
    )
    service.add_file = AsyncMock(return_value=True)

    await service._enqueue_mutation(service._build_mutation("scan", target_file))

    process_task = asyncio.create_task(service._process_loop())
    service.process_task = process_task
    try:
        await asyncio.wait_for(processed.wait(), timeout=2.0)
        await _wait_for(lambda: attempts == 2)
    finally:
        process_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await process_task

    assert service.failed_files == set()
    assert service._last_error is None


@pytest.mark.asyncio
async def test_compaction_retry_budget_exhaustion_marks_failure(
    service: RealtimeIndexingService,
    tmp_path: Path,
) -> None:
    target_file = tmp_path / "busy_forever.py"
    target_file.write_text("def busy_forever():\n    return 1\n", encoding="utf-8")

    async def always_busy(
        file_path: Path, skip_embeddings: bool = False
    ) -> dict[str, int]:
        assert file_path == target_file
        assert skip_embeddings is True
        raise DatabaseCompactionInProgressError("busy")

    service._DELETE_CONFLICT_MAX_RETRIES = 1
    service._DELETE_CONFLICT_BASE_RETRY_DELAY_SECONDS = 0.0
    service.services = SimpleNamespace(
        indexing_coordinator=SimpleNamespace(process_file=always_busy),
        provider=SimpleNamespace(flush=AsyncMock()),
    )
    service.add_file = AsyncMock(return_value=True)

    await service._enqueue_mutation(service._build_mutation("scan", target_file))

    process_task = asyncio.create_task(service._process_loop())
    service.process_task = process_task
    try:
        await _wait_for(lambda: service._last_error is not None)
    finally:
        process_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await process_task

    assert str(target_file) in service.failed_files
    assert service._last_error is not None
    assert "retry budget exhausted" in service._last_error


@pytest.mark.asyncio
async def test_delete_processing_retries_when_compaction_is_active(
    service: RealtimeIndexingService,
    tmp_path: Path,
) -> None:
    target_file = tmp_path / "busy_delete.py"
    delete_calls = 0

    async def flaky_delete(file_path: str) -> None:
        nonlocal delete_calls
        delete_calls += 1
        assert Path(file_path) == target_file
        if delete_calls == 1:
            raise DatabaseCompactionInProgressError("busy")

    service.services = SimpleNamespace(
        provider=SimpleNamespace(delete_file_completely_async=flaky_delete),
    )

    await service._enqueue_mutation(service._build_mutation("delete", target_file))

    process_task = asyncio.create_task(service._process_loop())
    service.process_task = process_task
    try:
        await _wait_for(lambda: delete_calls == 2)
    finally:
        process_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await process_task

    assert service.failed_files == set()
    assert service._last_error is None
