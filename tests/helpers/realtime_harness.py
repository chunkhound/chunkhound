"""Shared realtime-service harness for search contract tests.

Owns service lifecycle so tests can assert only the public search contract.
Uses ``tmp_path`` for cleanup and keeps shutdown failures visible.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from chunkhound.core.config.config import Config
from chunkhound.database_factory import DatabaseServices, create_services
from chunkhound.embeddings import EmbeddingManager
from chunkhound.services.realtime.service import RealtimeIndexingService
from tests.helpers.embedding_config import create_embedding_manager_for_tests
from tests.helpers.search_contract import realtime_search_diagnostics
from tests.utils.windows_compat import realtime_backend_for_tests

DEFAULT_INCLUDE_PATTERNS = ("*.py", "*.js")
DEFAULT_EXCLUDE_PATTERNS = ("*.log",)
_STOP_TIMEOUT_SECONDS = 10.0
_FORCE_STOP_TIMEOUT_SECONDS = 3.0


@dataclass(slots=True)
class RealtimeHarness:
    """Real services and watcher state shared by one contract test."""

    services: DatabaseServices
    realtime_service: RealtimeIndexingService
    watch_dir: Path
    embedding_manager: EmbeddingManager


@asynccontextmanager
async def realtime_harness(
    tmp_path: Path,
    *,
    embedding_config: dict[str, Any] | None = None,
    include: tuple[str, ...] = DEFAULT_INCLUDE_PATTERNS,
    exclude: tuple[str, ...] = DEFAULT_EXCLUDE_PATTERNS,
    realtime_backend: str | None = None,
    create_watch_dir: bool = True,
    watch_dir_is_index_root: bool = False,
    indexing: dict[str, Any] | None = None,
) -> AsyncIterator[RealtimeHarness]:
    """Run a realtime service against a private workspace.

    ``tmp_path`` owns directory cleanup. Service and provider shutdown run in
    ``finally`` and propagate failures so broken lifecycle code cannot hide.
    ``watch_dir_is_index_root`` makes the watch directory the database base
    directory, so the watched root maps to the empty indexed prefix.
    """
    db_path = tmp_path / ".chunkhound" / "test.db"
    watch_dir = tmp_path / "project"
    if create_watch_dir:
        watch_dir.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    index_root = watch_dir if watch_dir_is_index_root else tmp_path

    indexing_defaults = {
        "include": list(include),
        "exclude": list(exclude),
        "realtime_backend": realtime_backend or realtime_backend_for_tests(),
    }
    if indexing:
        indexing_defaults.update(indexing)

    config = Config(
        args=SimpleNamespace(path=index_root),
        database={"path": str(db_path), "provider": "duckdb"},
        embedding=embedding_config,
        indexing=indexing_defaults,
    )
    embedding_manager = create_embedding_manager_for_tests(embedding_config)
    services = create_services(db_path, config, embedding_manager)
    services.provider.connect()
    realtime_service = RealtimeIndexingService(services, config)
    started = False

    try:
        await realtime_service.start(watch_dir)
        started = True
        with realtime_search_diagnostics(realtime_service):
            yield RealtimeHarness(
                services=services,
                realtime_service=realtime_service,
                watch_dir=watch_dir,
                embedding_manager=embedding_manager,
            )
    finally:
        try:
            if started:
                await _stop_realtime_service(realtime_service)
        finally:
            services.provider.close()


async def _stop_realtime_service(service: RealtimeIndexingService) -> None:
    """Stop the service within a bounded budget and surface a real failure."""
    stop_task = asyncio.create_task(service.stop())
    done, _ = await asyncio.wait({stop_task}, timeout=_STOP_TIMEOUT_SECONDS)

    if not done:
        observer = getattr(service, "observer", None)
        if observer is not None and observer.is_alive():
            observer.stop()
        done, _ = await asyncio.wait({stop_task}, timeout=_FORCE_STOP_TIMEOUT_SECONDS)

    if not done:
        stop_task.cancel()
        try:
            await stop_task
        except asyncio.CancelledError:
            pass
        raise RuntimeError(
            "RealtimeIndexingService.stop() did not finish within "
            f"{_STOP_TIMEOUT_SECONDS + _FORCE_STOP_TIMEOUT_SECONDS:.0f}s"
        )

    # Propagate any exception raised by stop().
    stop_task.result()
