"""Phase 2: streaming generate_missing_embeddings must not full-table-load."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from chunkhound.core.models import Chunk, File
from chunkhound.core.types.common import ChunkType, Language
from chunkhound.services.embedding_service import EmbeddingService
from tests.fixtures.fake_providers import FakeEmbeddingProvider


def _insert_file(provider: Any, path: str = "sample.py") -> int:
    return int(
        provider.insert_file(
            File(
                path=path,
                mtime=1_700_000_000.0,
                language=Language.PYTHON,
                size_bytes=128,
            )
        )
    )


def _insert_chunks(
    provider: Any, file_id: int, count: int, prefix: str = "fn"
) -> list[int]:
    chunks = [
        Chunk(
            file_id=file_id,
            code=f"def {prefix}_{i}():\n    return {i}\n",
            start_line=i * 3 + 1,
            end_line=i * 3 + 2,
            chunk_type=ChunkType.FUNCTION,
            language=Language.PYTHON,
            symbol=f"{prefix}_{i}",
        )
        for i in range(count)
    ]
    return [int(i) for i in provider.insert_chunks_batch(chunks)]


@pytest.fixture
def duckdb_provider(tmp_path: Path):
    from chunkhound.providers.database.duckdb_provider import DuckDBProvider

    provider = DuckDBProvider(tmp_path / "chunks.db", base_directory=tmp_path)
    provider.connect()
    yield provider
    provider.disconnect()


@pytest.mark.parametrize("provider_fixture", ["duckdb_provider", "lancedb_provider"])
@pytest.mark.asyncio
async def test_generate_missing_streams_without_full_table_load(
    provider_fixture: str, request: pytest.FixtureRequest
):
    provider = request.getfixturevalue(provider_fixture)
    file_id = _insert_file(provider)
    chunk_ids = _insert_chunks(provider, file_id, 7)

    # Spy: full-table load must not be used on the missing-embeddings path.
    original_all = provider.get_all_chunks_with_metadata
    all_calls = MagicMock(side_effect=original_all)
    provider.get_all_chunks_with_metadata = all_calls  # type: ignore[method-assign]

    embedder = FakeEmbeddingProvider(dims=8, batch_size=2)
    service = EmbeddingService(
        database_provider=provider,
        embedding_provider=embedder,  # type: ignore[arg-type]
        embedding_batch_size=2,
        db_batch_size=2,
        max_concurrent_batches=1,
    )

    result = await service.generate_missing_embeddings()
    assert result["status"] == "success"
    assert result["generated"] == 7
    all_calls.assert_not_called()

    # Residual empty after success
    remaining = provider.get_chunks_without_embeddings_paginated(
        "fake", "fake-embeddings", limit=100
    )
    assert remaining == []
    assert set(chunk_ids)  # sanity


@pytest.mark.parametrize("provider_fixture", ["duckdb_provider", "lancedb_provider"])
@pytest.mark.asyncio
async def test_generate_missing_respects_exclude_patterns_without_spin(
    provider_fixture: str, request: pytest.FixtureRequest
):
    provider = request.getfixturevalue(provider_fixture)
    keep_id = _insert_file(provider, "keep/a.py")
    skip_id = _insert_file(provider, "skip/b.py")
    keep_chunks = _insert_chunks(provider, keep_id, 2, prefix="keep")
    skip_chunks = _insert_chunks(provider, skip_id, 2, prefix="skip")

    embedder = FakeEmbeddingProvider(dims=8, batch_size=10)
    service = EmbeddingService(
        database_provider=provider,
        embedding_provider=embedder,  # type: ignore[arg-type]
        embedding_batch_size=10,
        max_concurrent_batches=1,
    )

    result = await service.generate_missing_embeddings(
        exclude_patterns=["skip/*"]
    )
    assert result["status"] == "success"
    assert result["generated"] == 2

    missing = provider.get_chunks_without_embeddings_paginated(
        "fake", "fake-embeddings", limit=100
    )
    missing_ids = {int(r["id"]) for r in missing}
    assert set(keep_chunks).isdisjoint(missing_ids)
    # Excluded chunks may still lack embeddings (not embedded).
    assert set(skip_chunks).issubset(missing_ids)


@pytest.mark.parametrize("provider_fixture", ["duckdb_provider", "lancedb_provider"])
@pytest.mark.asyncio
async def test_generate_missing_complete_when_nothing_to_do(
    provider_fixture: str, request: pytest.FixtureRequest
):
    provider = request.getfixturevalue(provider_fixture)
    embedder = FakeEmbeddingProvider(dims=8)
    service = EmbeddingService(
        database_provider=provider,
        embedding_provider=embedder,  # type: ignore[arg-type]
        max_concurrent_batches=1,
    )
    result = await service.generate_missing_embeddings()
    assert result["status"] == "complete"
    assert result["generated"] == 0
    assert "No embeddable chunks" in result.get("message", "")


class _FailingEmbeddingProvider(FakeEmbeddingProvider):
    """Fake provider that always raises on embed."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("simulated embed API failure")


class _PartialFailEmbeddingProvider(FakeEmbeddingProvider):
    """Succeed first API call, fail subsequent ones (partial page progress)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._calls = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self._calls += 1
        if self._calls == 1:
            return await super().embed(texts)
        raise RuntimeError("simulated partial embed failure")


@pytest.mark.parametrize("provider_fixture", ["duckdb_provider", "lancedb_provider"])
@pytest.mark.asyncio
async def test_generate_missing_failed_page_returns_error_without_skipping(
    provider_fixture: str, request: pytest.FixtureRequest
):
    provider = request.getfixturevalue(provider_fixture)
    file_id = _insert_file(provider)
    chunk_ids = _insert_chunks(provider, file_id, 3)

    service = EmbeddingService(
        database_provider=provider,
        embedding_provider=_FailingEmbeddingProvider(dims=8, batch_size=10),  # type: ignore[arg-type]
        embedding_batch_size=10,
        max_concurrent_batches=1,
    )
    result = await service.generate_missing_embeddings()
    assert result["status"] == "error"
    assert result["generated"] == 0

    missing = provider.get_chunks_without_embeddings_paginated(
        "fake", "fake-embeddings", limit=100
    )
    assert {int(r["id"]) for r in missing} == set(chunk_ids)


@pytest.mark.parametrize("provider_fixture", ["duckdb_provider", "lancedb_provider"])
@pytest.mark.asyncio
async def test_generate_missing_partial_failure_does_not_keyset_skip(
    provider_fixture: str, request: pytest.FixtureRequest
):
    """After exclude forces keyset mode, partial embed must not abandon missing IDs."""
    provider = request.getfixturevalue(provider_fixture)
    # Lower ids first so exclude-only pages are seen before keep pages when
    # walking by id is not guaranteed — still exercise multi-file exclude.
    skip_id = _insert_file(provider, "aaa_skip/x.py")
    keep_id = _insert_file(provider, "zzz_keep/y.py")
    _insert_chunks(provider, skip_id, 2, prefix="skip")
    keep_chunks = _insert_chunks(provider, keep_id, 4, prefix="keep")

    # batch_size=2 → first keep page may partially succeed then fail next batch
    embedder = _PartialFailEmbeddingProvider(dims=8, batch_size=2)
    service = EmbeddingService(
        database_provider=provider,
        embedding_provider=embedder,  # type: ignore[arg-type]
        embedding_batch_size=2,
        db_batch_size=2,
        max_concurrent_batches=1,
    )
    result = await service.generate_missing_embeddings(
        exclude_patterns=["aaa_skip/*"]
    )

    # Either completes via residual retries or errors — never success with
    # unembedded keep chunks left behind after advancing past them forever.
    missing = provider.get_chunks_without_embeddings_paginated(
        "fake", "fake-embeddings", limit=100
    )
    missing_keep = {int(r["id"]) for r in missing} & set(keep_chunks)

    if result["status"] == "success":
        assert missing_keep == set()
        assert result["generated"] >= len(keep_chunks)
    else:
        assert result["status"] == "error"
        # Failed IDs must still be queryable as missing (no silent skip).
        assert missing_keep


@pytest.mark.parametrize("provider_fixture", ["duckdb_provider", "lancedb_provider"])
@pytest.mark.asyncio
async def test_generate_missing_exclude_multi_page_keyset(
    provider_fixture: str, request: pytest.FixtureRequest
):
    """Small page size + excluded files first must still embed keep chunks."""
    provider = request.getfixturevalue(provider_fixture)
    skip_id = _insert_file(provider, "0_skip/a.py")
    keep_id = _insert_file(provider, "1_keep/b.py")
    _insert_chunks(provider, skip_id, 3, prefix="skip")
    keep_chunks = _insert_chunks(provider, keep_id, 3, prefix="keep")

    service = EmbeddingService(
        database_provider=provider,
        embedding_provider=FakeEmbeddingProvider(dims=8, batch_size=1),  # type: ignore[arg-type]
        embedding_batch_size=1,
        max_concurrent_batches=1,
    )
    result = await service.generate_missing_embeddings(
        exclude_patterns=["0_skip/*"]
    )
    assert result["status"] == "success"
    assert result["generated"] == 3

    missing = provider.get_chunks_without_embeddings_paginated(
        "fake", "fake-embeddings", limit=100
    )
    missing_ids = {int(r["id"]) for r in missing}
    assert set(keep_chunks).isdisjoint(missing_ids)
