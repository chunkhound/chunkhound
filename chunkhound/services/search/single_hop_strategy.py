"""Single-hop search strategy using standard HNSW vector similarity.

This strategy performs a single vector similarity search without expansion or reranking:
1. Generate embedding for query
2. Search vector index (HNSW) for similar chunks
3. Return top-k results based on similarity score

Best for:
- Fast, straightforward semantic search
- Providers without reranking capabilities
- Queries where top HNSW results are sufficient
"""

from typing import Any

from loguru import logger

from chunkhound.interfaces.database_provider import DatabaseProvider
from chunkhound.interfaces.embedding_provider import EmbeddingProvider


class SingleHopStrategy:
    """Standard single-hop semantic search using HNSW vector similarity."""

    def __init__(
        self,
        database_provider: DatabaseProvider,
        embedding_provider: EmbeddingProvider,
    ):
        """Initialize single-hop search strategy.

        Args:
            database_provider: Database provider for vector search
            embedding_provider: Embedding provider for query vectorization
        """
        self._db = database_provider
        self._embedding_provider = embedding_provider

    async def search(
        self,
        query: str,
        page_size: int,
        offset: int,
        threshold: float | None,
        provider: str,
        model: str,
        path_filter: str | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Perform standard single-hop semantic search.

        Args:
            query: Natural language search query
            page_size: Number of results per page
            offset: Starting position for pagination
            threshold: Optional similarity threshold to filter results
            provider: Embedding provider name
            model: Embedding model name
            path_filter: Optional relative path to limit search scope

        Returns:
            Tuple of (results, pagination_metadata)
        """
        # Generate query embedding
        query_results = await self._embedding_provider.embed([query])
        if not query_results:
            return [], {}

        query_vector = query_results[0]

        # Perform vector similarity search
        results, pagination = self._db.search_semantic(
            query_embedding=query_vector,
            provider=provider,
            model=model,
            page_size=page_size,
            offset=offset,
            threshold=threshold,
            path_filter=path_filter,
        )

        logger.info(f"Standard semantic search completed: {len(results)} results found")
        return results, pagination
