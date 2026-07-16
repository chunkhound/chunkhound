"""Phase 4: coordinator glue for large-index / LanceDB fragment pressure."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from chunkhound.core.models import Chunk
from chunkhound.core.types.common import ChunkType, Language
from chunkhound.services.indexing_coordinator import IndexingCoordinator


def _chunk(i: int = 0) -> Chunk:
    return Chunk(
        file_id=1,
        code=f"def f{i}():\n    return {i}\n",
        start_line=1,
        end_line=2,
        chunk_type=ChunkType.FUNCTION,
        language=Language.PYTHON,
        symbol=f"f{i}",
    )


def _coordinator(db: MagicMock) -> IndexingCoordinator:
    # Avoid real provider setup; construct with a minimal fake db.
    coord = IndexingCoordinator.__new__(IndexingCoordinator)
    coord._db = db
    coord.config = SimpleNamespace(
        indexing=SimpleNamespace(db_batch_size=0, exclude=["**/skip/**"]),
        embedding=SimpleNamespace(batch_size=100, max_concurrent_batches=1),
    )
    coord._embedding_provider = MagicMock(name="emb")
    coord.progress = None
    return coord


def test_apply_fragment_batch_cap_reduces_when_fragments_high():
    db = MagicMock()
    db.get_fragment_count.return_value = {"chunks": 250, "files": 10}
    coord = _coordinator(db)
    assert coord._apply_fragment_batch_cap(4000) == 2000


def test_determine_db_batch_size_applies_cap_after_config_base(monkeypatch):
    """Config base size is still reduced under fragment pressure."""
    db = MagicMock()
    db.get_fragment_count.return_value = {"chunks": 250}
    coord = _coordinator(db)
    coord.config.indexing.db_batch_size = 4000
    monkeypatch.delenv("CHUNKHOUND_DB_BATCH_SIZE", raising=False)
    assert coord._determine_db_batch_size([_chunk()]) == 2000


def test_apply_fragment_batch_cap_noop_when_no_fragment_api():
    db = MagicMock(spec=[])  # no get_fragment_count
    coord = _coordinator(db)
    assert coord._apply_fragment_batch_cap(4000) == 4000


def test_apply_fragment_batch_cap_noop_when_fragments_low():
    db = MagicMock()
    db.get_fragment_count.return_value = {"chunks": 10}
    coord = _coordinator(db)
    assert coord._apply_fragment_batch_cap(4000) == 4000


def test_maybe_optimize_after_store_batch_calls_optimize_when_needed():
    db = MagicMock()
    db.get_fragment_count.return_value = {"chunks": 200}
    db.should_optimize.return_value = True
    coord = _coordinator(db)
    coord._maybe_optimize_after_store_batch()
    db.should_optimize.assert_called_once_with(operation="post-batch-store")
    db.optimize_tables.assert_called_once()


def test_maybe_optimize_after_store_batch_skips_when_not_needed():
    db = MagicMock()
    db.get_fragment_count.return_value = {"chunks": 10}
    db.should_optimize.return_value = False
    coord = _coordinator(db)
    coord._maybe_optimize_after_store_batch()
    db.optimize_tables.assert_not_called()


def test_maybe_optimize_after_store_batch_skips_without_fragment_api():
    db = MagicMock(spec=["should_optimize", "optimize_tables"])
    db.should_optimize.return_value = True
    coord = _coordinator(db)
    coord._maybe_optimize_after_store_batch()
    db.should_optimize.assert_not_called()
    db.optimize_tables.assert_not_called()


@pytest.mark.asyncio
async def test_generate_missing_embeddings_defaults_exclude_from_config(monkeypatch):
    db = MagicMock()
    coord = _coordinator(db)
    coord.config.indexing.get_effective_config_excludes = MagicMock(
        return_value=["**/skip/**", "**/node_modules/**"]
    )
    captured: dict = {}

    class _FakeEmbService:
        def __init__(self, **kwargs):
            pass

        async def generate_missing_embeddings(self, exclude_patterns=None):
            captured["exclude_patterns"] = exclude_patterns
            return {"status": "complete", "generated": 0}

    monkeypatch.setattr(
        "chunkhound.services.embedding_service.EmbeddingService",
        _FakeEmbService,
    )
    result = await coord.generate_missing_embeddings()
    assert result["status"] == "complete"
    assert captured["exclude_patterns"] == ["**/skip/**", "**/node_modules/**"]
    coord.config.indexing.get_effective_config_excludes.assert_called_once()


@pytest.mark.asyncio
async def test_insert_chunks_in_sized_batches_splits_by_determined_size():
    db = MagicMock()
    db.get_fragment_count.return_value = {"chunks": 10}
    calls: list[int] = []

    async def _insert(batch):
        calls.append(len(batch))
        return [10 + i for i in range(len(batch))]

    db.insert_chunks_batch_async = _insert
    coord = _coordinator(db)
    coord._determine_db_batch_size = MagicMock(return_value=2)  # type: ignore[method-assign]
    chunks = [_chunk(i) for i in range(5)]
    ids = await coord._insert_chunks_in_sized_batches(chunks)
    assert len(ids) == 5
    assert calls == [2, 2, 1]
    assert coord._determine_db_batch_size.call_count == 1


def test_extract_cli_lancedb_fragment_threshold():
    from chunkhound.core.config.database_config import DatabaseConfig

    args = SimpleNamespace(
        db=None,
        database_path=None,
        database_provider=None,
        max_disk_usage_gb=None,
        read_only=False,
        fragmentation_threshold_pct=None,
        lancedb_optimize_fragment_threshold=50,
    )
    overrides = DatabaseConfig.extract_cli_overrides(args)
    assert overrides["lancedb_optimize_fragment_threshold"] == 50


def test_determine_db_batch_size_applies_fragment_cap(monkeypatch):
    db = MagicMock()
    db.get_fragment_count.return_value = {"chunks": 600}
    coord = _coordinator(db)
    # Force dynamic path (no env/config batch size).
    monkeypatch.delenv("CHUNKHOUND_DB_BATCH_SIZE", raising=False)
    size = coord._determine_db_batch_size([_chunk(i) for i in range(50)])
    # With fragments >= 500, base is reduced by // 4 with floor 500.
    assert size <= 5000
    assert size >= 500
