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
async def test_reindex_after_defer_does_not_duplicate_chunk_ids(
    tmp_path: Path,
) -> None:
    """Modify+reindex after cold defer must not leave duplicate content-hash ids.

    Deferred Lance writes use append; reindex must stay on classic merge path
    so updates do not stack duplicate rows for the same chunk id.
    """
    src = tmp_path / "src"
    src.mkdir()
    target = src / "m.py"
    target.write_text("def a():\n    return 1\n")

    db_dir = tmp_path / "db"
    cfg = Config(
        database=DatabaseConfig(
            path=db_dir,
            provider="lancedb",
            lancedb_optimize_fragment_threshold=10_000,
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
        r1 = await coord.process_directory(
            src, patterns=["**/*.py"], exclude_patterns=[]
        )
        assert r1.get("status") == "success", r1
        rows1 = db._chunks_table.search().to_list() if db._chunks_table else []
        ids1 = [int(r["id"]) for r in rows1]
        assert ids1, "expected chunks after first index"
        assert len(ids1) == len(set(ids1)), "duplicate ids after cold defer"

        # Change content so reindex is not skipped
        target.write_text("def a():\n    return 2\n\ndef b():\n    return 3\n")
        r2 = await coord.process_directory(
            src, patterns=["**/*.py"], exclude_patterns=[]
        )
        assert r2.get("status") == "success", r2
        emb = await coord.generate_missing_embeddings()
        assert emb.get("status") in ("success", "complete", "no_provider"), emb

        rows2 = db._chunks_table.search().to_list() if db._chunks_table else []
        ids2 = [int(r["id"]) for r in rows2]
        assert ids2, "expected chunks after reindex"
        assert len(ids2) == len(set(ids2)), (
            f"duplicate chunk ids after reindex: {len(ids2)} rows, "
            f"{len(set(ids2))} unique"
        )
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
async def test_multi_file_defer_leaves_no_missing_embeds(
    tmp_path: Path,
) -> None:
    """Many small files under defer_chunk_write leave zero missing embeds.

    Covers F1 cross-file append buffer (default flush) correctness — not
    call-count reduction.
    """
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
        indexing=IndexingConfig(
            defer_chunk_write=True,
            defer_flush_chunks=1000,
            cleanup=False,
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
        assert int(result.get("total_chunks", 0)) >= 20
        missing = db.get_chunks_without_embeddings_paginated(
            fake.name, fake.model, limit=50
        )
        assert missing == [], f"expected no missing embeds, got {len(missing)}"
    finally:
        db.disconnect()


@pytest.mark.asyncio
async def test_f1_cross_file_flush_fewer_chunk_batches_than_files(
    tmp_path: Path,
) -> None:
    """F1: many tiny new files should produce fewer deferred appends than files."""
    from chunkhound.core.diagnostics.index_profile import IndexProfile

    src = tmp_path / "src"
    src.mkdir()
    n_files = 40
    for i in range(n_files):
        (src / f"f{i:02d}.py").write_text(
            f"def f{i}():\n    return {i}\n\ndef g{i}():\n    return {i}\n"
        )

    db_dir = tmp_path / "db"
    cfg = Config(
        database=DatabaseConfig(
            path=db_dir,
            provider="lancedb",
            lancedb_optimize_fragment_threshold=10_000,
        ),
        indexing=IndexingConfig(
            defer_chunk_write=True,
            defer_flush_chunks=50,  # well above chunks-per-file, below total
            cleanup=False,
        ),
        embedding=EmbeddingConfig(
            provider="openai", model="fake-embeddings", batch_size=100
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
    fake = FakeEmbeddingProvider(dims=32, batch_size=100)
    try:
        coord = IndexingCoordinator(
            database_provider=db,
            base_directory=src,
            embedding_provider=fake,  # type: ignore[arg-type]
            config=cfg,
        )
        coord.attach_index_profile(profile)
        result = await coord.process_directory(
            src, patterns=["**/*.py"], exclude_patterns=[]
        )
        assert result.get("status") == "success", result
        missing = db.get_chunks_without_embeddings_paginated(
            fake.name, fake.model, limit=50
        )
        assert missing == []
        # Per-file L1 would be ~n_files chunk batches; F1 should be far fewer.
        batches = int(profile.db.chunk_insert_batches)
        assert batches < n_files, (
            f"expected F1 fewer chunk batches than files: "
            f"batches={batches} files={n_files}"
        )
        assert batches >= 1
    finally:
        db.disconnect()


@pytest.mark.asyncio
async def test_f1_flush_failure_does_not_wipe_appended_embeddings(
    tmp_path: Path,
) -> None:
    """Partial F1 flush failure must not classic-merge over already-appended rows."""
    from unittest.mock import AsyncMock

    src = tmp_path / "src"
    src.mkdir()
    # Enough chunks that two flush slices of 10 are needed mid-file.
    lines = "\n\n".join(f"def f{i}():\n    return {i}\n" for i in range(25))
    (src / "big.py").write_text(lines)

    db_dir = tmp_path / "db"
    cfg = Config(
        database=DatabaseConfig(
            path=db_dir,
            provider="lancedb",
            lancedb_optimize_fragment_threshold=10_000,
        ),
        indexing=IndexingConfig(
            defer_chunk_write=True,
            defer_flush_chunks=10,
            cleanup=False,
        ),
        embedding=EmbeddingConfig(
            provider="openai", model="fake-embeddings", batch_size=100
        ),
    )
    db = LanceDBProvider(
        str(cfg.database.get_db_path()),
        base_directory=src,
        config=cfg.database,
    )
    db.connect()
    fake = FakeEmbeddingProvider(dims=32, batch_size=100)
    real_insert = db.insert_chunks_with_embeddings_batch_async
    calls = {"n": 0}

    async def flaky_insert(chunks, embeddings, provider, model):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated append failure")
        return await real_insert(chunks, embeddings, provider, model)

    db.insert_chunks_with_embeddings_batch_async = AsyncMock(  # type: ignore[method-assign]
        side_effect=flaky_insert
    )
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
        # May succeed via classic residual on failed slice, or end flush retry.
        assert result.get("status") in ("success", "error"), result
        # Chunks that were append-written must not lose vectors (merge wipe).
        emb = await coord.generate_missing_embeddings()
        assert emb.get("status") in ("success", "complete"), emb
        missing = db.get_chunks_without_embeddings_paginated(
            fake.name, fake.model, limit=50
        )
        assert missing == [], f"still missing after residual: {len(missing)}"
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
