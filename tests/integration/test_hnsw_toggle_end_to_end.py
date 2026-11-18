"""Canonical end-to-end contract for the DuckDB HNSW enable/disable toggle.

``duckdb_hnsw_enabled`` must control index existence and stay honored by both
real indexing pipelines (Rust and Python) and by a later compaction. Every
assertion here is user-visible state: the persisted index and the rows a
semantic search returns.
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from chunkhound.core.config.config import Config
from chunkhound.core.config.database_config import DatabaseConfig
from chunkhound.core.config.embedding_config import EmbeddingConfig
from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory
from chunkhound.core.types.common import Language
from chunkhound.embeddings import EmbeddingManager
from chunkhound.parsers.parser_factory import create_parser_for_language
from chunkhound.pipeline_bridge import run_rust_pipeline
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.directory_indexing_service import DirectoryIndexingService
from chunkhound.services.indexing_coordinator import IndexingCoordinator
from tests.fixtures.fake_providers import ConstantEmbeddingProvider
from tests.helpers.duckdb_vector_fixtures import has_hnsw_index, require_hnsw_index

try:
    from chunkhound_native import IndexingPipeline  # noqa: F401

    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False

pytestmark = pytest.mark.hnsw

requires_rust = pytest.mark.skipif(
    not _RUST_AVAILABLE, reason="Rust extension not built"
)

_DIMS = 8
_SOURCE_NAME = "module.py"
# One tiny file keeps the whole matrix fast; chunk count is read back from the
# database instead of hardcoded so parser changes cannot break the contract.
_SOURCE = "def first() -> int:\n    return 1\n\n\ndef second() -> int:\n    return 2\n"


def _rust_config(enabled: bool, embedding: EmbeddingConfig) -> SimpleNamespace:
    """Pipeline config for the native writer; compaction stays manual.

    Mirrors the harness in ``tests/unit/test_rust_hnsw_disabled_contract.py``.
    """
    return SimpleNamespace(
        indexing=SimpleNamespace(),
        embedding=embedding,
        database=SimpleNamespace(
            duckdb_hnsw_enabled=enabled,
            fragmentation_threshold_pct=None,
            max_disk_usage_mb=None,
        ),
    )


def _rust_embedding(
    monkeypatch: pytest.MonkeyPatch, embedding: ConstantEmbeddingProvider
) -> EmbeddingConfig:
    """Route the native writer's embedding callback to the offline provider."""
    monkeypatch.setattr(
        EmbeddingProviderFactory, "create_provider", staticmethod(lambda _: embedding)
    )
    # The fake name selects Rust's Python-provider callback, not native HTTP.
    return EmbeddingConfig.model_construct(
        provider=embedding.name, model=embedding.model, max_concurrent_batches=1
    )


def _run_rust(
    db_dir: Path, root: Path, enabled: bool, embedding: EmbeddingConfig
) -> None:
    """Force a real write run so index maintenance always has something to do."""
    result = asyncio.run(
        run_rust_pipeline(
            [(root / _SOURCE_NAME, None)],
            db_path=db_dir,
            project_root=root,
            force_reindex=True,
            config=_rust_config(enabled, embedding),
        )
    )
    assert not result["errors"], result


def _python_config(root: Path, db_dir: Path, enabled: bool) -> Config:
    return Config(
        target_dir=root,
        database={
            "path": str(db_dir / "chunks.db"),
            "duckdb_hnsw_enabled": enabled,
            "fragmentation_threshold_pct": None,
        },
        indexing={"include": ["**/*.py"], "exclude": []},
    )


def _run_python(
    db_dir: Path,
    root: Path,
    enabled: bool,
    monkeypatch: pytest.MonkeyPatch,
    embedding: ConstantEmbeddingProvider,
) -> None:
    """Drive the real Python indexing service (no Rust pipeline)."""
    monkeypatch.setenv("CHUNKHOUND_USE_RUST", "0")
    config = _python_config(root, db_dir, enabled)
    database = DuckDBProvider(config.database.path, base_directory=root)
    database.config = config.database
    manager = EmbeddingManager()
    manager.register_provider(embedding, set_default=True)
    database.embedding_manager = manager
    database.connect()
    try:
        coordinator = IndexingCoordinator(
            database,
            root,
            embedding,
            {Language.PYTHON: create_parser_for_language(Language.PYTHON)},
            config=config,
        )
        service = DirectoryIndexingService(coordinator, config)
        stats = asyncio.run(service.process_directory(root))
    finally:
        database.disconnect()
    assert stats.errors_encountered == [], stats.errors_encountered


@contextmanager
def _opened(
    db_dir: Path, root: Path, enabled: bool
) -> Generator[DuckDBProvider, None, None]:
    provider = DuckDBProvider(
        db_dir / "chunks.db",
        base_directory=root,
        config=DatabaseConfig(duckdb_hnsw_enabled=enabled),
    )
    provider.connect()
    try:
        yield provider
    finally:
        provider.disconnect(skip_checkpoint=True)


def _assert_index_state(db_dir: Path, root: Path, enabled: bool) -> None:
    with _opened(db_dir, root, enabled) as database:
        if enabled:
            require_hnsw_index(database, _DIMS)
        else:
            assert not has_hnsw_index(database, _DIMS)


def _assert_searchable(
    db_dir: Path,
    root: Path,
    enabled: bool,
    embedding: ConstantEmbeddingProvider,
) -> None:
    """All stored rows come back with perfect similarity in either mode."""
    query = asyncio.run(embedding.embed_single("any query"))
    with _opened(db_dir, root, enabled) as database:
        rows, _ = database.search_semantic(
            query, embedding.name, embedding.model, page_size=100
        )
        chunks = database.get_stats()["chunks"]

    assert chunks > 0
    assert len(rows) == chunks
    assert {Path(str(row["file_path"])).name for row in rows} == {_SOURCE_NAME}
    assert all(row["similarity"] == pytest.approx(1.0) for row in rows)


def _compact(db_dir: Path, root: Path, enabled: bool) -> None:
    with _opened(db_dir, root, enabled) as database:
        database.compact_database()


@pytest.mark.parametrize(
    "pipeline", [pytest.param("rust", marks=requires_rust), "python"]
)
@pytest.mark.parametrize("enabled", [True, False], ids=["enabled", "disabled"])
def test_hnsw_toggle_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pipeline: str,
    enabled: bool,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / _SOURCE_NAME).write_text(_SOURCE, encoding="utf-8")
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    embedding = ConstantEmbeddingProvider(dims=_DIMS)

    if pipeline == "rust":
        rust_embedding = _rust_embedding(monkeypatch, embedding)

        def run(state: bool) -> None:
            _run_rust(db_dir, root, state, rust_embedding)

    else:

        def run(state: bool) -> None:
            _run_python(db_dir, root, state, monkeypatch, embedding)

    # Scenario 1: a fresh DB gets an index iff the toggle is on.
    run(enabled)
    _assert_index_state(db_dir, root, enabled)
    _assert_searchable(db_dir, root, enabled, embedding)

    # Scenarios 2 and 3: cover both directions for every cell, whichever state
    # the fresh DB started in.
    if enabled:
        # Scenario 2: indexed DB -> disabled run removes the index and a later
        # compaction does not bring it back.
        run(False)
        _assert_index_state(db_dir, root, False)
        _assert_searchable(db_dir, root, False, embedding)
        _compact(db_dir, root, False)
        _assert_index_state(db_dir, root, False)

        # Scenario 3: index-free DB -> enabled run rebuilds it.
        run(True)
        _assert_index_state(db_dir, root, True)
        _assert_searchable(db_dir, root, True, embedding)
    else:
        # Scenario 3: index-free DB -> enabled run builds it.
        run(True)
        _assert_index_state(db_dir, root, True)
        _assert_searchable(db_dir, root, True, embedding)

        # Scenario 2: indexed DB -> disabled run removes it for good.
        run(False)
        _assert_index_state(db_dir, root, False)
        _assert_searchable(db_dir, root, False, embedding)
        _compact(db_dir, root, False)
        _assert_index_state(db_dir, root, False)

    # Scenario 4: reopening with the last run's config preserves the state.
    _assert_index_state(db_dir, root, enabled)
