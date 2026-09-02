"""Compatibility facade for the streaming transient search service."""

from collections.abc import AsyncIterator
from typing import Any

from chunkhound.services.search_service_protocol import SearchServiceProtocol
from chunkhound.services.transient_search_service import TransientSearchService
from chunkhound.services.vector_cache import VectorCache

__all__ = ["DiffAwareSearchService", "SearchServiceProtocol"]


class DiffAwareSearchService(TransientSearchService):
    """Retain the former eager constructor for external callers."""

    def __init__(
        self,
        original: SearchServiceProtocol,
        diff_chunks: list[Any],
        diff_embeddings: list[list[float]],
        vector_source: str,
        embedding_manager: Any,
    ) -> None:
        count = min(len(diff_chunks), len(diff_embeddings))
        source_chunks = diff_chunks[:count]
        cache = VectorCache(max_entries=max(1, count))
        for chunk, vector in zip(source_chunks, diff_embeddings[:count]):
            cache.put(chunk.code, vector)

        async def chunk_stream() -> AsyncIterator[list[Any]]:
            if source_chunks:
                yield source_chunks

        super().__init__(
            original=original,
            chunk_stream=chunk_stream,
            vector_source=vector_source,
            embedding_manager=embedding_manager,
            vector_cache=cache,
        )
        self._diff_chunks = source_chunks
