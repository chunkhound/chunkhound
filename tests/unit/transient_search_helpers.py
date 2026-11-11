"""Shared fixtures for transient (diff/web) search service tests."""

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from unittest.mock import AsyncMock, MagicMock

from chunkhound.core.models.chunk import Chunk
from chunkhound.core.types.common import ChunkType, FileId, Language, LineNumber
from chunkhound.embeddings import LocalEmbeddingResult


def make_chunk(
    symbol: str,
    file_path: str = "src/foo.py",
    start_line: int = 1,
    end_line: int | None = None,
    code: str | None = None,
) -> Chunk:
    sl = max(1, start_line)
    el = end_line if end_line is not None else sl + 4
    el = max(el, sl)
    return Chunk(
        symbol=symbol,
        start_line=LineNumber(sl),
        end_line=LineNumber(el),
        code=code or f"def {symbol}(): pass",
        chunk_type=ChunkType.FUNCTION,
        file_id=FileId(1),
        language=Language.PYTHON,
        file_path=file_path,  # type: ignore[arg-type]
    )


def _page(results: list[dict]) -> tuple[list[dict], dict]:
    return (
        results,
        {
            "offset": 0,
            "page_size": 10,
            "has_more": False,
            "next_offset": None,
            "total": len(results),
        },
    )


def make_original(
    semantic_results: list[dict] | None = None,
    regex_results: list[dict] | None = None,
) -> MagicMock:
    """Return a fake database SearchService with canned single-page results."""
    original = MagicMock()
    sem = semantic_results or []
    reg = regex_results or []
    original.search_semantic = AsyncMock(return_value=_page(sem))
    original.search_regex = MagicMock(return_value=_page(reg))
    original.search_regex_async = AsyncMock(return_value=_page(reg))
    original.search_hybrid = AsyncMock(return_value=_page([]))
    original.get_chunk_context = MagicMock(return_value={"context": "data"})
    original.get_file_chunks = MagicMock(return_value=[])
    return original


def stream_chunks(
    chunks: Sequence[Chunk], batch_size: int = 100
) -> Callable[[], AsyncIterator[list[Chunk]]]:
    """Return a chunk-batch stream factory over ``chunks``."""

    async def stream() -> AsyncIterator[list[Chunk]]:
        for start in range(0, len(chunks), batch_size):
            await asyncio.sleep(0)
            yield list(chunks[start : start + batch_size])

    return stream


def keyed_embedding_manager(
    query_vector: list[float] | None,
    chunks: Sequence[Chunk] = (),
    chunk_vectors: Sequence[list[float]] = (),
) -> MagicMock:
    """Fake embedding manager: queries embed to ``query_vector``, each chunk
    embeds to its paired vector. ``query_vector=None`` yields no embeddings."""
    vectors = {chunk.code: vector for chunk, vector in zip(chunks, chunk_vectors)}
    dims = len(query_vector) if query_vector else 0

    async def embed(texts: list[str]) -> LocalEmbeddingResult:
        if query_vector is None:
            embeddings: list[list[float]] = []
        else:
            embeddings = [vectors.get(text, query_vector) for text in texts]
        return LocalEmbeddingResult(
            embeddings=embeddings,
            model="test-model",
            provider="test",
            dims=dims,
            embedding_space_identity="test-space",
        )

    manager = MagicMock()
    manager.embed_texts = AsyncMock(side_effect=embed)
    manager.embed_query_texts = manager.embed_texts
    return manager
