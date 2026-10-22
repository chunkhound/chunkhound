"""Directory indexing stays responsive while serialized HNSW DDL runs."""

import asyncio
import threading
from contextlib import suppress
from pathlib import Path

import pytest
from loguru import logger

from chunkhound.core.config.config import Config
from chunkhound.core.types.common import Language
from chunkhound.embeddings import EmbeddingManager
from chunkhound.parsers.parser_factory import create_parser_for_language
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.directory_indexing_service import DirectoryIndexingService
from chunkhound.services.indexing_coordinator import IndexingCoordinator
from tests.fixtures.fake_providers import FakeEmbeddingProvider

_OPERATIONS = ["drop_all_hnsw_indexes", "ensure_all_hnsw_indexes"]


class OfflineIndexingEmbeddings(FakeEmbeddingProvider):
    @property
    def base_url(self) -> None:
        return None


@pytest.fixture
async def indexed_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_environment
):
    monkeypatch.setenv("CHUNKHOUND_USE_RUST", "0")
    config = Config(
        database={"path": str(tmp_path / "index_test.duckdb")},
        indexing={"include": ["**/*.py"], "exclude": []},
    )
    config.target_dir = tmp_path
    db = DuckDBProvider(config.database.path, base_directory=tmp_path)
    db.config = config.database
    manager = EmbeddingManager()
    provider = OfflineIndexingEmbeddings(dims=16)
    manager.register_provider(provider, set_default=True)
    db.embedding_manager = manager
    db.connect()
    coordinator = IndexingCoordinator(
        db,
        tmp_path,
        provider,
        {Language.PYTHON: create_parser_for_language(Language.PYTHON)},
        config=config,
    )
    service = DirectoryIndexingService(coordinator, config)
    source = tmp_path / "mod.py"
    source.write_text("def first():\n    return 1\n", encoding="utf-8")
    warnings = []
    sink = logger.add(lambda message: warnings.append(str(message)), level="WARNING")
    try:
        initial = await service.process_directory(tmp_path)
        assert initial.files_processed == 1, initial
        assert initial.embeddings_generated > 0, "\n".join(warnings)
        assert db.get_existing_vector_indexes()
        source.write_text(
            "def first():\n    return 1\n\ndef second():\n    return 2\n",
            encoding="utf-8",
        )
        yield db, service, tmp_path
    finally:
        logger.remove(sink)
        if db.is_connected:
            await asyncio.to_thread(db.disconnect)


def _hold_operation(db, operation_name, monkeypatch):
    """Pause the real operation at the database boundary, not the service."""
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    finished = threading.Event()
    original = getattr(db, f"_executor_{operation_name}")

    def held(conn, state):
        loop.call_soon_threadsafe(started.set)
        try:
            if not release.wait(timeout=10):
                raise TimeoutError("test did not release HNSW DDL")
            return original(conn, state)
        finally:
            finished.set()

    monkeypatch.setattr(db, f"_executor_{operation_name}", held)
    return started, release, finished


@pytest.mark.parametrize("operation_name", _OPERATIONS)
async def test_indexing_allows_loop_progress_and_restores_vector_indexes(
    indexed_project, monkeypatch, operation_name
):
    db, service, root = indexed_project
    started, release, finished = _hold_operation(db, operation_name, monkeypatch)
    indexing = asyncio.create_task(service.process_directory(root))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        progress = asyncio.Event()
        asyncio.get_running_loop().call_soon(progress.set)
        await asyncio.wait_for(progress.wait(), timeout=2)
        assert not finished.is_set()
    finally:
        release.set()
        stats = await indexing

    assert stats.files_processed == 1
    assert stats.chunks_created > 0
    assert db.get_existing_vector_indexes()


@pytest.mark.parametrize("operation_name", _OPERATIONS)
async def test_scan_cancellation_returns_before_ddl_and_disconnect_waits_for_it(
    indexed_project, monkeypatch, operation_name
):
    db, service, root = indexed_project
    started, release, finished = _hold_operation(db, operation_name, monkeypatch)
    indexing = asyncio.create_task(service.process_directory(root))
    close_started = asyncio.Event()
    loop = asyncio.get_running_loop()
    close_observations = []
    original_close = db._executor_disconnect
    closing = None

    def observed_close(*args, **kwargs):
        close_observations.append(finished.is_set())
        return original_close(*args, **kwargs)

    def disconnect():
        loop.call_soon_threadsafe(close_started.set)
        db.disconnect()

    monkeypatch.setattr(db, "_executor_disconnect", observed_close)
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        indexing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(indexing, timeout=2)
        assert not finished.is_set(), "scan cancellation must not drain HNSW DDL"
        closing = asyncio.create_task(asyncio.to_thread(disconnect))
        await asyncio.wait_for(close_started.wait(), timeout=2)
    finally:
        release.set()
        with suppress(asyncio.CancelledError):
            await indexing
        if closing is not None:
            await closing

    assert close_observations == [True]
    assert not db.is_connected


@pytest.mark.parametrize("operation_name", _OPERATIONS)
async def test_indexing_reports_hnsw_database_failures(
    indexed_project, monkeypatch, operation_name
):
    db, service, root = indexed_project

    def failed(conn, state):
        raise RuntimeError("HNSW maintenance failed")

    monkeypatch.setattr(db, f"_executor_{operation_name}", failed)
    with pytest.raises(RuntimeError, match="HNSW maintenance failed"):
        await service.process_directory(root)
