"""Contract tests: a parse worker that dies does not leave indexing stuck.

Observed on Windows: parse-pool workers died at start-up (an OSError releasing
their call-queue semaphore), the executor became broken, and the start-up
scan failed with BrokenProcessPool. The broken executor stayed the shared
pool, so every later parse in the process failed at once, and the failure
reached the server as a string, so nothing could tell it apart from a
permanent error.

These tests break the pool deterministically with a worker whose initializer
exits at once, which is how the observed workers died: before running a task.
"""

import multiprocessing
import os
import shutil
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import pytest
from loguru import logger

import chunkhound.pipeline_bridge as pipeline_bridge
from chunkhound.core.config.config import Config
from chunkhound.registry import configure_registry, create_indexing_coordinator
from chunkhound.services.directory_indexing_service import (
    DirectoryIndexingError,
    DirectoryIndexingService,
)
from tests.contracts.pipeline_harness import disconnect_registry_db

FIXTURE_FILE = (
    Path(__file__).resolve().parent.parent / "fixtures" / "pipeline" / "main.py"
)
_DEAD_WORKER_EXIT_CODE = 3


def _pool_whose_worker_dies() -> ProcessPoolExecutor:
    """A one-worker pool whose worker exits before it can run any task."""
    return ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=os._exit,
        initargs=(_DEAD_WORKER_EXIT_CODE,),
    )


@pytest.fixture
def broken_shared_pool(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Make the shared parse pool one whose worker dies; restore it afterwards."""
    previous = pipeline_bridge._parse_pool
    monkeypatch.setattr(pipeline_bridge, "_parse_pool", _pool_whose_worker_dies())
    yield
    replacement = pipeline_bridge._parse_pool
    pipeline_bridge._parse_pool = previous
    if replacement is not None and replacement is not previous:
        replacement.shutdown(cancel_futures=True)


def test_a_dead_worker_is_logged_and_the_next_batch_gets_new_workers(
    broken_shared_pool: None,
) -> None:
    errors: list[str] = []
    sink = logger.add(lambda message: errors.append(str(message)), level="ERROR")
    try:
        with pytest.raises(BrokenProcessPool):
            pipeline_bridge.parse_batch_callback([str(FIXTURE_FILE)])
    finally:
        logger.remove(sink)

    report = "\n".join(errors)
    assert "pid=" in report
    assert f"exitcode={_DEAD_WORKER_EXIT_CODE}" in report
    assert str(FIXTURE_FILE) in report

    ((language, chunks, error, _skip),) = pipeline_bridge.parse_batch_callback(
        [str(FIXTURE_FILE)]
    )
    assert error is None
    assert language == "python"
    assert chunks


@pytest.mark.asyncio
async def test_a_scan_that_lost_a_worker_fails_as_transient(
    broken_shared_pool: None, tmp_path: Path
) -> None:
    """The real indexing chain marks the failure as one a retry can fix."""
    project = tmp_path / "project"
    project.mkdir()
    shutil.copy(FIXTURE_FILE, project / FIXTURE_FILE.name)
    config = Config(
        target_dir=project,
        database={"provider": "duckdb", "path": str(tmp_path / "db")},
        embeddings_disabled=True,
    )
    configure_registry(config)
    service = DirectoryIndexingService(
        indexing_coordinator=create_indexing_coordinator(), config=config
    )
    try:
        with pytest.raises(DirectoryIndexingError) as failure:
            await service.process_directory(project, no_embeddings=True)
    finally:
        disconnect_registry_db()

    assert failure.value.transient is True
    assert "BrokenProcessPool" in str(failure.value)
