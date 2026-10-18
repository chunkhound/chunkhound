"""Rust-path contract: ``duckdb_hnsw_enabled=false`` reaches the native writer.

The Python provider honors ``DatabaseConfig.duckdb_hnsw_enabled``, but the Rust
indexing pipeline owns the write path by default. This test drives the real
native pipeline through ``run_rust_pipeline`` and observes the database, so a
regression that stops forwarding the toggle (or stops honoring it in Rust)
fails here rather than silently resurrecting HNSW indexes.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chunkhound.core.config.config import Config
from chunkhound.core.config.database_config import DatabaseConfig
from chunkhound.core.config.embedding_config import EmbeddingConfig
from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory
from chunkhound.pipeline_bridge import run_rust_pipeline
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from tests.fixtures.fake_providers import FakeEmbeddingProvider
from tests.helpers.duckdb_vector_fixtures import (
    has_hnsw_index,
    insert_chunks,
    insert_embeddings,
    require_hnsw_index,
)

try:
    from chunkhound_native import IndexingPipeline  # noqa: F401

    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False

requires_rust = pytest.mark.skipif(
    not _RUST_AVAILABLE, reason="Rust extension not built"
)

_DIMS = 3
_SEED_VECTORS = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _indexing_config(hnsw_enabled: bool, embedding: Any = None) -> Any:
    """Disable compaction to keep real native runs focused on HNSW maintenance."""
    return SimpleNamespace(
        indexing=SimpleNamespace(),
        embedding=embedding,
        database=SimpleNamespace(
            duckdb_hnsw_enabled=hnsw_enabled,
            fragmentation_threshold_pct=None,
            max_disk_usage_mb=None,
        ),
    )


def _seed_chunks_db(db_dir: Path) -> None:
    """Create ``chunks.db`` with embeddings and a real HNSW index."""
    provider = DuckDBProvider(db_dir / "chunks.db", base_directory=db_dir)
    provider.connect()
    try:
        chunk_ids = insert_chunks(provider, "src/module.py", _DIMS)
        insert_embeddings(provider, chunk_ids, _SEED_VECTORS)
        provider.ensure_all_hnsw_indexes()
        require_hnsw_index(provider, _DIMS)
    finally:
        provider.disconnect(skip_checkpoint=True)


def _hnsw_present(db_dir: Path, *, hnsw_enabled: bool) -> bool:
    provider = DuckDBProvider(
        db_dir / "chunks.db",
        base_directory=db_dir,
        config=DatabaseConfig(duckdb_hnsw_enabled=hnsw_enabled),
    )
    provider.connect()
    try:
        return has_hnsw_index(provider, _DIMS)
    finally:
        provider.disconnect(skip_checkpoint=True)


@requires_rust
@pytest.mark.hnsw
@pytest.mark.parametrize(
    "do_cleanup", [True, False], ids=["orphan-cleanup", "nothing-to-write"]
)
def test_rust_indexing_disabled_drops_hnsw_index(
    tmp_path: Path, do_cleanup: bool
) -> None:
    """A disabled Rust run must clean up an index left by an earlier run.

    No files are handed to the pipeline. With ``do_cleanup=True`` the seeded
    row becomes an orphan DELETE; with ``do_cleanup=False`` the run changes
    nothing. The second case is the one where the "skip when nothing changed"
    optimization would otherwise leave the index in place, so it proves the
    disabled backend always runs the drop bracket.
    """
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    _seed_chunks_db(db_dir)

    asyncio.run(
        run_rust_pipeline(
            [],
            db_path=db_dir,
            project_root=tmp_path,
            skip_embeddings=True,
            do_cleanup=do_cleanup,
            config=_indexing_config(hnsw_enabled=False),
        )
    )

    assert not _hnsw_present(db_dir, hnsw_enabled=False), (
        "Rust indexing with duckdb_hnsw_enabled=false left an HNSW index behind"
    )


@requires_rust
@pytest.mark.hnsw
def test_real_config_disabled_toggle_reaches_rust_writer(tmp_path: Path) -> None:
    """A real Config must forward the disabled toggle to the native writer."""
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    _seed_chunks_db(db_dir)
    config = Config(
        target_dir=tmp_path,
        database={"duckdb_hnsw_enabled": False},
        indexing={"include": ["**/*.py"], "exclude": []},
    )

    asyncio.run(
        run_rust_pipeline(
            [],
            db_path=db_dir,
            project_root=tmp_path,
            skip_embeddings=True,
            do_cleanup=False,
            config=config,
        )
    )

    assert not _hnsw_present(db_dir, hnsw_enabled=False)


def _offline_embedding_config(monkeypatch: pytest.MonkeyPatch) -> EmbeddingConfig:
    """Replace only the external embedding boundary; parsing and storage stay real."""
    provider = FakeEmbeddingProvider(dims=_DIMS)
    monkeypatch.setattr(
        EmbeddingProviderFactory, "create_provider", staticmethod(lambda _: provider)
    )
    # The fake name selects Rust's Python-provider callback, not native HTTP.
    return EmbeddingConfig.model_construct(
        provider=provider.name, model=provider.model, max_concurrent_batches=1
    )


def _assert_persisted_without_hnsw(db_dir: Path) -> None:
    """Reopen the native-written database without enabling observer-side HNSW."""
    assert (db_dir / "chunks.db").is_file(), "Native indexing did not create a database"
    provider = DuckDBProvider(
        db_dir / "chunks.db",
        base_directory=db_dir.parent,
        config=DatabaseConfig(duckdb_hnsw_enabled=False),
    )
    provider.connect()
    try:
        stats = provider.get_stats()
        assert stats["files"] == 1, stats
        assert stats["chunks"] > 0, stats
        assert stats["embeddings"] > 0, stats
        assert not has_hnsw_index(provider, _DIMS), (
            "Native indexing created an HNSW index"
        )
    finally:
        provider.disconnect(skip_checkpoint=True)


@requires_rust
@pytest.mark.hnsw
async def test_rust_indexing_disabled_on_index_free_db_creates_no_hnsw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh native run must persist real embeddings without creating HNSW."""
    embedding = _offline_embedding_config(monkeypatch)
    source = tmp_path / "module.py"
    source.write_text(
        "def greeting(name):\n    return f'Hello, {name}!'\n", encoding="utf-8"
    )
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    result = await run_rust_pipeline(
        [(source, None)],
        db_path=db_dir,
        project_root=tmp_path,
        config=_indexing_config(hnsw_enabled=False, embedding=embedding),
    )
    assert not result["errors"], result
    _assert_persisted_without_hnsw(db_dir)


def _compaction_enabled_config() -> Any:
    """Allow auto-compaction so reopen path runs under disabled HNSW."""
    cfg = _indexing_config(hnsw_enabled=False)
    cfg.database.fragmentation_threshold_pct = 0.0
    return cfg


@requires_rust
@pytest.mark.hnsw
def test_rust_indexing_disabled_with_compaction_enabled_leaves_no_hnsw(
    tmp_path: Path,
) -> None:
    """Disabled run cleans HNSW even when compaction may run."""
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    _seed_chunks_db(db_dir)
    asyncio.run(
        run_rust_pipeline(
            [],
            db_path=db_dir,
            project_root=tmp_path,
            skip_embeddings=True,
            do_cleanup=True,
            config=_compaction_enabled_config(),
        )
    )
    assert not _hnsw_present(db_dir, hnsw_enabled=False)
