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


def _watch_retry_budget_exhaustion(
    service: RealtimeIndexingService,
) -> asyncio.Event:
    """Signal exactly when realtime retry exhaustion is recorded."""
    exhausted = asyncio.Event()
    original_set_error = service._set_error

    def _set_error_and_signal(message: str) -> None:
        original_set_error(message)
        if "retry budget exhausted" in message:
            exhausted.set()

    service._set_error = _set_error_and_signal  # type: ignore[method-assign]
    return exhausted


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
        assert attempts >= 2  # retry must have fired after first compaction-busy
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

    exhausted = _watch_retry_budget_exhaustion(service)

    async def always_busy(
        file_path: Path, skip_embeddings: bool = False
    ) -> dict[str, int]:
        assert file_path == target_file
        assert skip_embeddings is True
        raise DatabaseCompactionInProgressError("busy")

    service._MAX_RETRY_BUDGET = 1
    service._RETRY_BASE_DELAY_SECONDS = 0.0
    service.services = SimpleNamespace(
        indexing_coordinator=SimpleNamespace(process_file=always_busy),
        provider=SimpleNamespace(flush=AsyncMock()),
    )
    service.add_file = AsyncMock(return_value=True)

    await service._enqueue_mutation(service._build_mutation("scan", target_file))

    process_task = asyncio.create_task(service._process_loop())
    service.process_task = process_task
    try:
        await asyncio.wait_for(exhausted.wait(), timeout=2.0)
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
    delete_completed = asyncio.Event()

    async def flaky_delete(file_path: str) -> None:
        nonlocal delete_calls
        delete_calls += 1
        assert Path(file_path) == target_file
        if delete_calls == 1:
            raise DatabaseCompactionInProgressError("busy")
        delete_completed.set()

    service.services = SimpleNamespace(
        provider=SimpleNamespace(delete_file_completely_async=flaky_delete),
    )

    await service._enqueue_mutation(service._build_mutation("delete", target_file))

    process_task = asyncio.create_task(service._process_loop())
    service.process_task = process_task
    try:
        await asyncio.wait_for(delete_completed.wait(), timeout=2.0)
    finally:
        process_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await process_task

    assert service.failed_files == set()
    assert service._last_error is None


@pytest.mark.asyncio
async def test_embed_processing_retries_when_compaction_is_active(
    service: RealtimeIndexingService,
    tmp_path: Path,
) -> None:
    target_file = tmp_path / "busy_embed.py"
    target_file.write_text("def busy_embed():\n    return 1\n", encoding="utf-8")

    attempts = 0
    completed = asyncio.Event()

    async def flaky_generate_missing_embeddings() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise DatabaseCompactionInProgressError("busy")
        completed.set()

    service.services = SimpleNamespace(
        indexing_coordinator=SimpleNamespace(
            generate_missing_embeddings=flaky_generate_missing_embeddings,
        ),
    )

    await service._enqueue_mutation(service._build_mutation("embed", target_file))

    process_task = asyncio.create_task(service._process_loop())
    service.process_task = process_task
    try:
        await asyncio.wait_for(completed.wait(), timeout=2.0)
        assert attempts >= 2  # retry must have fired after first compaction-busy
    finally:
        process_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await process_task

    assert service.failed_files == set()
    assert service._last_error is None


@pytest.mark.asyncio
async def test_embed_compaction_retry_budget_exhaustion_marks_failure(
    service: RealtimeIndexingService,
    tmp_path: Path,
) -> None:
    target_file = tmp_path / "busy_embed_forever.py"
    target_file.write_text(
        "def busy_embed_forever():\n    return 1\n", encoding="utf-8"
    )

    exhausted = _watch_retry_budget_exhaustion(service)

    async def always_busy_generate() -> None:
        raise DatabaseCompactionInProgressError("busy")

    service._MAX_RETRY_BUDGET = 1
    service._RETRY_BASE_DELAY_SECONDS = 0.0
    service.services = SimpleNamespace(
        indexing_coordinator=SimpleNamespace(
            generate_missing_embeddings=always_busy_generate,
        ),
    )

    await service._enqueue_mutation(service._build_mutation("embed", target_file))

    process_task = asyncio.create_task(service._process_loop())
    service.process_task = process_task
    try:
        await asyncio.wait_for(exhausted.wait(), timeout=2.0)
    finally:
        process_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await process_task

    assert str(target_file) in service.failed_files
    assert service._last_error is not None
    assert "retry budget exhausted" in service._last_error


@pytest.mark.asyncio
async def test_delete_batch_compaction_busy_schedules_retries(
    service: RealtimeIndexingService,
    tmp_path: Path,
) -> None:
    """_handle_compaction_busy_batch schedules retries for mutations within budget."""
    file_a = tmp_path / "batch_a.py"
    file_b = tmp_path / "batch_b.py"

    mutation_a = service._build_mutation("delete", file_a)
    mutation_b = service._build_mutation("delete", file_b)

    surviving, exhausted = service._handle_compaction_busy_batch(
        [mutation_a, mutation_b], "delete batch"
    )

    assert len(surviving) == 2
    assert len(exhausted) == 0
    assert service.failed_files == set()
    assert service._last_error is None


@pytest.mark.asyncio
async def test_delete_batch_compaction_retry_budget_exhaustion(
    service: RealtimeIndexingService,
    tmp_path: Path,
) -> None:
    """_handle_compaction_busy_batch marks failure when budget exhausted."""
    file_a = tmp_path / "exhaust_a.py"
    file_b = tmp_path / "exhaust_b.py"

    mutation_a = service._build_mutation("delete", file_a)
    mutation_b = service._build_mutation("delete", file_b)

    service._MAX_RETRY_BUDGET = 0

    surviving, exhausted = service._handle_compaction_busy_batch(
        [mutation_a, mutation_b], "delete batch"
    )

    assert len(surviving) == 0
    assert len(exhausted) == 2
    assert str(file_a) in service.failed_files
    assert str(file_b) in service.failed_files
    assert service._last_error is not None
    assert "retry budget exhausted" in service._last_error
