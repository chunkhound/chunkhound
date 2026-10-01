"""Contracts for async HNSW maintenance in the directory indexing service."""

import asyncio
import threading
from contextlib import suppress
from types import SimpleNamespace

import pytest

from chunkhound.services.directory_indexing_service import DirectoryIndexingService

_HNSW_OPERATIONS = ["drop_all_hnsw_indexes", "ensure_all_hnsw_indexes"]


def _make_service(operation_name: str, operation):
    database = SimpleNamespace(
        drop_all_hnsw_indexes=lambda: None,
        ensure_all_hnsw_indexes=lambda: None,
    )
    setattr(database, operation_name, operation)
    coordinator = SimpleNamespace(
        _db=database,
        resolve_rust_pipeline_decision=lambda log_reason=False: False,
    )
    return DirectoryIndexingService(coordinator, config=None)


async def _run_operation(service: DirectoryIndexingService, operation_name: str):
    if operation_name == "drop_all_hnsw_indexes":
        await service._drop_hnsw_indexes()
    else:
        await service._ensure_hnsw_indexes(used_rust_pipeline=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation_name", _HNSW_OPERATIONS)
async def test_hnsw_operations_allow_other_coroutines_to_progress(operation_name):
    """Each blocking database call leaves the event loop available to others."""
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    progress_during_operation = asyncio.Event()
    stop_monitor = asyncio.Event()

    def blocking_operation():
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test did not release the HNSW operation")
        finished.set()

    service = _make_service(operation_name, blocking_operation)

    async def monitor_loop_progress():
        while not stop_monitor.is_set():
            if started.is_set() and not release.is_set():
                progress_during_operation.set()
            await asyncio.sleep(0)

    monitor = asyncio.create_task(monitor_loop_progress())
    operation = asyncio.create_task(_run_operation(service, operation_name))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        await asyncio.wait_for(progress_during_operation.wait(), timeout=2)
    finally:
        release.set()
        await operation
        stop_monitor.set()
        with suppress(asyncio.CancelledError):
            await monitor

    assert finished.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation_name", _HNSW_OPERATIONS)
async def test_cancellation_waits_for_hnsw_worker_before_returning(operation_name):
    """Scan cancellation drains the non-cancellable thread before shutdown."""
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def blocking_operation():
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test did not release the HNSW operation")
        finished.set()

    service = _make_service(operation_name, blocking_operation)
    operation = asyncio.create_task(_run_operation(service, operation_name))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        operation.cancel()
        await asyncio.sleep(0.02)
        assert not operation.done(), "cancel must wait for the HNSW worker to finish"
    finally:
        release.set()

    with pytest.raises(asyncio.CancelledError):
        await operation
    assert finished.is_set()
