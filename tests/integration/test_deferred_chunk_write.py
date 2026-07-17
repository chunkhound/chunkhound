"""Contract tests for defer_chunk_write single-write path."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("lancedb")

from chunkhound.core.config.config import Config
from chunkhound.core.config.database_config import DatabaseConfig
from chunkhound.core.config.embedding_config import EmbeddingConfig
from chunkhound.core.config.indexing_config import IndexingConfig
from chunkhound.providers.database.lancedb_provider import LanceDBProvider
from chunkhound.services.indexing_coordinator import IndexingCoordinator
from tests.fixtures.fake_providers import FakeEmbeddingProvider


@pytest.mark.asyncio
async def test_deferred_write_indexes_new_file_with_vectors(
    tmp_path: Path,
) -> None:
    """New file under defer_chunk_write ends with zero missing embeddings."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").write_text("def hello():\n    return 1\n\ndef world():\n    return 2\n")

    db_dir = tmp_path / "db"
    cfg = Config(
        database=DatabaseConfig(
            path=db_dir,
            provider="lancedb",
            lancedb_optimize_fragment_threshold=1000,
        ),
        indexing=IndexingConfig(defer_chunk_write=True, cleanup=False),
        embedding=EmbeddingConfig(
            provider="openai", model="fake-embeddings", batch_size=50
        ),
    )
    db = LanceDBProvider(
        str(cfg.database.get_db_path()),
        base_directory=src,
        config=cfg.database,
    )
    db.connect()
    fake = FakeEmbeddingProvider(dims=32, batch_size=50)
    try:
        coord = IndexingCoordinator(
            database_provider=db,
            base_directory=src,
            embedding_provider=fake,  # type: ignore[arg-type]
            config=cfg,
        )
        result = await coord.process_directory(
            src, patterns=["**/*.py"], exclude_patterns=[]
        )
        assert result.get("status") == "success", result
        assert int(result.get("total_chunks", 0)) >= 1

        missing = db.get_chunks_without_embeddings_paginated(
            fake.name, fake.model, limit=50
        )
        assert missing == [], f"expected no missing embeds, got {len(missing)}"
    finally:
        db.disconnect()


@pytest.mark.asyncio
async def test_defer_respects_skip_embeddings_for_realtime(
    tmp_path: Path,
) -> None:
    """skip_embeddings=True must not embed even when defer_chunk_write is on."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "b.py").write_text("def a():\n    return 1\n")

    db_dir = tmp_path / "db"
    cfg = Config(
        database=DatabaseConfig(path=db_dir, provider="lancedb"),
        indexing=IndexingConfig(defer_chunk_write=True, cleanup=False),
        embedding=EmbeddingConfig(
            provider="openai", model="fake-embeddings", batch_size=50
        ),
    )
    db = LanceDBProvider(
        str(cfg.database.get_db_path()),
        base_directory=src,
        config=cfg.database,
    )
    db.connect()
    fake = FakeEmbeddingProvider(dims=32, batch_size=50)
    embed_calls = {"n": 0}
    original_embed = fake.embed_batch

    async def counting_embed(texts):  # type: ignore[no-untyped-def]
        embed_calls["n"] += 1
        return await original_embed(texts)

    fake.embed_batch = counting_embed  # type: ignore[method-assign]
    try:
        coord = IndexingCoordinator(
            database_provider=db,
            base_directory=src,
            embedding_provider=fake,  # type: ignore[arg-type]
            config=cfg,
        )
        result = await coord.process_file(src / "b.py", skip_embeddings=True)
        assert result.get("status") == "success", result
        assert embed_calls["n"] == 0, "skip_embeddings must not call embed API"
        missing = db.get_chunks_without_embeddings_paginated(
            fake.name, fake.model, limit=50
        )
        assert len(missing) >= 1, "chunks should remain for residual embed"
    finally:
        db.disconnect()


@pytest.mark.asyncio
async def test_classic_then_residual_clears_missing(tmp_path: Path) -> None:
    """Store without defer then generate_missing leaves no missing embeds."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "c.py").write_text("def z():\n    return 9\n")

    db_dir = tmp_path / "db"
    cfg = Config(
        database=DatabaseConfig(path=db_dir, provider="lancedb"),
        indexing=IndexingConfig(defer_chunk_write=False, cleanup=False),
        embedding=EmbeddingConfig(
            provider="openai", model="fake-embeddings", batch_size=50
        ),
    )
    db = LanceDBProvider(
        str(cfg.database.get_db_path()),
        base_directory=src,
        config=cfg.database,
    )
    db.connect()
    fake = FakeEmbeddingProvider(dims=32, batch_size=50)
    try:
        coord = IndexingCoordinator(
            database_provider=db,
            base_directory=src,
            embedding_provider=fake,  # type: ignore[arg-type]
            config=cfg,
        )
        await coord.process_directory(
            src, patterns=["**/*.py"], exclude_patterns=[]
        )
        # Without per-file embed, residual fills vectors
        emb = await coord.generate_missing_embeddings()
        assert emb.get("status") in ("success", "complete"), emb
        missing = db.get_chunks_without_embeddings_paginated(
            fake.name, fake.model, limit=50
        )
        assert missing == []
    finally:
        db.disconnect()


@pytest.mark.asyncio
async def test_cross_file_deferred_buffer_fewer_merge_inserts(
    tmp_path: Path,
) -> None:
    """Many small files should share merge_insert flushes (L2)."""
    from chunkhound.core.diagnostics.index_profile import IndexProfile

    src = tmp_path / "src"
    src.mkdir()
    for i in range(20):
        (src / f"f{i:02d}.py").write_text(f"def f{i}():\n    return {i}\n")

    db_dir = tmp_path / "db"
    cfg = Config(
        database=DatabaseConfig(
            path=db_dir,
            provider="lancedb",
            lancedb_optimize_fragment_threshold=10_000,
        ),
        # Small flush threshold so 20 tiny files still exercise buffering.
        indexing=IndexingConfig(
            defer_chunk_write=True,
            cleanup=False,
            db_batch_size=50,  # > chunks per file (~1-2) so multiple files per flush
        ),
        embedding=EmbeddingConfig(
            provider="openai", model="fake-embeddings", batch_size=50
        ),
    )
    db = LanceDBProvider(
        str(cfg.database.get_db_path()),
        base_directory=src,
        config=cfg.database,
    )
    profile = IndexProfile()
    db.set_index_profile(profile)
    db.connect()
    fake = FakeEmbeddingProvider(dims=32, batch_size=50)
    try:
        coord = IndexingCoordinator(
            database_provider=db,
            base_directory=src,
            embedding_provider=fake,  # type: ignore[arg-type]
            config=cfg,
        )
        result = await coord.process_directory(
            src, patterns=["**/*.py"], exclude_patterns=[]
        )
        assert result.get("status") == "success", result
        mi_calls = profile.db.merge_insert_calls
        # 20 files × 1 insert would be 20; with buffer we expect fewer chunk merges.
        # (file table merge_inserts also exist — count only chunk path via batches.)
        assert profile.db.chunk_insert_batches < 20, (
            f"expected cross-file flush, got chunk_insert_batches="
            f"{profile.db.chunk_insert_batches} merge_insert_calls={mi_calls}"
        )
        missing = db.get_chunks_without_embeddings_paginated(
            fake.name, fake.model, limit=50
        )
        assert missing == []
    finally:
        db.disconnect()


@pytest.mark.asyncio
async def test_insert_chunks_with_embeddings_batch_roundtrip(
    tmp_path: Path,
) -> None:
    from chunkhound.core.models import Chunk, File
    from chunkhound.core.types.common import ChunkType, Language

    cfg = DatabaseConfig(path=tmp_path, provider="lancedb")
    db = LanceDBProvider(
        str(cfg.get_db_path()), base_directory=tmp_path, config=cfg
    )
    db.connect()
    try:
        fid = int(
            db.insert_file(
                File(
                    path="t.py",
                    mtime=1.0,
                    language=Language.PYTHON,
                    size_bytes=10,
                )
            )
        )
        chunks = [
            Chunk(
                file_id=fid,
                code="def f():\n    pass\n",
                start_line=1,
                end_line=2,
                chunk_type=ChunkType.FUNCTION,
                language=Language.PYTHON,
                symbol="f",
            )
        ]
        vec = [0.1] * 32
        ids = db.insert_chunks_with_embeddings_batch(
            chunks, [vec], "fake", "fake-embeddings"
        )
        assert len(ids) == 1
        missing = db.get_chunks_without_embeddings_paginated(
            "fake", "fake-embeddings", limit=10
        )
        assert missing == []
    finally:
        db.disconnect()
