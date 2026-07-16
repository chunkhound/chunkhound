"""Phase 5: synthetic large-index soak (CI-scale, dual backend).

Not a multi-million-row soak — that belongs to the optional local script.
This proves the full store → stream-embed path stays correct at a few
thousand chunks without full-table metadata loads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from chunkhound.core.models import Chunk, File
from chunkhound.core.types.common import ChunkType, Language
from chunkhound.services.embedding_service import EmbeddingService
from tests.fixtures.fake_providers import FakeEmbeddingProvider

# CI-friendly size: large enough to multi-page, small enough for normal pytest.
SOAK_CHUNK_COUNT = 800
SOAK_PAGE_SIZE = 40


@pytest.fixture
def duckdb_provider(tmp_path: Path):
    from chunkhound.providers.database.duckdb_provider import DuckDBProvider

    provider = DuckDBProvider(tmp_path / "soak.db", base_directory=tmp_path)
    provider.connect()
    yield provider
    provider.disconnect()


def _seed_many_chunks(provider: Any, count: int) -> list[int]:
    """Insert many files/chunks to exercise multi-page keyset embed."""
    all_ids: list[int] = []
    # Spread across files so store paths see multi-file batches too.
    files = max(1, count // 50)
    per_file = count // files
    remainder = count % files
    for f in range(files):
        n = per_file + (1 if f < remainder else 0)
        file_id = int(
            provider.insert_file(
                File(
                    path=f"soak/module_{f:04d}.py",
                    mtime=1_700_000_000.0 + f,
                    language=Language.PYTHON,
                    size_bytes=256 * n,
                )
            )
        )
        chunks = [
            Chunk(
                file_id=file_id,
                code=f"def soak_{f}_{i}():\n    return {f * 1000 + i}\n",
                start_line=i * 3 + 1,
                end_line=i * 3 + 2,
                chunk_type=ChunkType.FUNCTION,
                language=Language.PYTHON,
                symbol=f"soak_{f}_{i}",
            )
            for i in range(n)
        ]
        all_ids.extend(int(x) for x in provider.insert_chunks_batch(chunks))
    assert len(all_ids) == count
    return all_ids


@pytest.mark.parametrize("provider_fixture", ["duckdb_provider", "lancedb_provider"])
@pytest.mark.asyncio
async def test_soak_stream_embed_all_chunks_without_full_table_load(
    provider_fixture: str, request: pytest.FixtureRequest
):
    provider = request.getfixturevalue(provider_fixture)
    chunk_ids = _seed_many_chunks(provider, SOAK_CHUNK_COUNT)

    original_all = provider.get_all_chunks_with_metadata
    all_calls = MagicMock(side_effect=original_all)
    provider.get_all_chunks_with_metadata = all_calls  # type: ignore[method-assign]

    page_calls = MagicMock(
        side_effect=provider.get_chunks_without_embeddings_paginated
    )
    provider.get_chunks_without_embeddings_paginated = page_calls  # type: ignore[method-assign]

    service = EmbeddingService(
        database_provider=provider,
        embedding_provider=FakeEmbeddingProvider(  # type: ignore[arg-type]
            dims=16, batch_size=SOAK_PAGE_SIZE
        ),
        embedding_batch_size=SOAK_PAGE_SIZE,
        db_batch_size=SOAK_PAGE_SIZE,
        max_concurrent_batches=2,
    )

    result = await service.generate_missing_embeddings()
    assert result["status"] == "success"
    assert result["generated"] == SOAK_CHUNK_COUNT
    all_calls.assert_not_called()
    # Multi-page keyset walk (800 chunks / page 40 ⇒ well above 2 pages).
    assert page_calls.call_count >= 2

    remaining = provider.get_chunks_without_embeddings_paginated(
        "fake", "fake-embeddings", limit=100
    )
    assert remaining == []

    # Spot-check presence for a sample of ids.
    sample = chunk_ids[:: max(1, len(chunk_ids) // 20)]
    existing = provider.get_existing_embeddings(sample, "fake", "fake-embeddings")
    assert set(sample).issubset(existing)


@pytest.mark.parametrize("provider_fixture", ["duckdb_provider", "lancedb_provider"])
@pytest.mark.asyncio
async def test_soak_exclude_patterns_leave_excluded_unembedded(
    provider_fixture: str, request: pytest.FixtureRequest
):
    provider = request.getfixturevalue(provider_fixture)
    # Keep + skip trees
    keep_id = int(
        provider.insert_file(
            File(
                path="keep/app.py",
                mtime=1.0,
                language=Language.PYTHON,
                size_bytes=100,
            )
        )
    )
    skip_id = int(
        provider.insert_file(
            File(
                path="vendor/lib.py",
                mtime=1.0,
                language=Language.PYTHON,
                size_bytes=100,
            )
        )
    )
    keep_chunks = [
        Chunk(
            file_id=keep_id,
            code=f"def keep_{i}():\n    return {i}\n",
            start_line=i * 2 + 1,
            end_line=i * 2 + 2,
            chunk_type=ChunkType.FUNCTION,
            language=Language.PYTHON,
            symbol=f"keep_{i}",
        )
        for i in range(30)
    ]
    skip_chunks = [
        Chunk(
            file_id=skip_id,
            code=f"def skip_{i}():\n    return {i}\n",
            start_line=i * 2 + 1,
            end_line=i * 2 + 2,
            chunk_type=ChunkType.FUNCTION,
            language=Language.PYTHON,
            symbol=f"skip_{i}",
        )
        for i in range(30)
    ]
    keep_ids = [int(x) for x in provider.insert_chunks_batch(keep_chunks)]
    skip_ids = [int(x) for x in provider.insert_chunks_batch(skip_chunks)]

    service = EmbeddingService(
        database_provider=provider,
        embedding_provider=FakeEmbeddingProvider(dims=8, batch_size=10),  # type: ignore[arg-type]
        embedding_batch_size=10,
        max_concurrent_batches=1,
    )
    result = await service.generate_missing_embeddings(
        exclude_patterns=["vendor/**"]
    )
    assert result["status"] == "success"
    assert result["generated"] == len(keep_ids)

    missing = provider.get_chunks_without_embeddings_paginated(
        "fake", "fake-embeddings", limit=200
    )
    missing_ids = {int(r["id"]) for r in missing}
    assert set(keep_ids).isdisjoint(missing_ids)
    assert set(skip_ids).issubset(missing_ids)
