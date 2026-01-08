"""Search service for ChunkHound - handles semantic and regex search operations."""

import asyncio
from typing import Any

from loguru import logger

from chunkhound.core.types.common import ChunkId
from chunkhound.interfaces.database_provider import DatabaseProvider
from chunkhound.interfaces.embedding_provider import EmbeddingProvider

from .base_service import BaseService
from .search.context_retriever import ContextRetriever
from .search.multi_hop_strategy import MultiHopStrategy
from .search.result_enhancer import ResultEnhancer
from .search.single_hop_strategy import SingleHopStrategy


class SearchService(BaseService):
    """Service for performing semantic and regex searches across indexed code."""

    def __init__(
        self,
        database_provider: DatabaseProvider,
        embedding_provider: EmbeddingProvider | None = None,
        config: Any = None,
    ):
        """Initialize search service.

        Args:
            database_provider: Database provider for data access
            embedding_provider: Optional embedding provider for semantic search
            config: Optional configuration object for search strategies
        """
        super().__init__(database_provider)
        self._config = config
        self._embedding_provider = embedding_provider
        self._result_enhancer = ResultEnhancer()
        self._context_retriever = ContextRetriever(database_provider)

        # Initialize search strategies
        if embedding_provider:
            self._single_hop_strategy = SingleHopStrategy(
                database_provider, embedding_provider
            )
            self._multi_hop_strategy = MultiHopStrategy(
                database_provider,
                embedding_provider,
                self._single_hop_strategy.search,
                config=config,
            )
        else:
            self._single_hop_strategy = None
            self._multi_hop_strategy = None

    async def search_semantic(
        self,
        query: str,
        page_size: int = 10,
        offset: int = 0,
        threshold: float | None = None,
        provider: str | None = None,
        model: str | None = None,
        path_filter: str | None = None,
        force_strategy: str | None = None,
        time_limit: float | None = None,
        result_limit: int | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Perform semantic search using vector similarity.

        Automatically selects the best search strategy:
        - Multi-hop + reranking if provider supports reranking
        - Standard single-hop otherwise
        - Can be overridden with force_strategy parameter

        Args:
            query: Natural language search query
            page_size: Number of results per page
            offset: Starting position for pagination
            threshold: Optional similarity threshold to filter results
            provider: Optional specific embedding provider to use
            model: Optional specific model to use
            path_filter: Optional relative path to limit search scope
                (e.g., 'src/', 'tests/')
            force_strategy: Optional strategy override ('single_hop', 'multi_hop')
            time_limit: Optional time limit for multi-hop expansion (default: 5.0s)
            result_limit: Optional result limit for multi-hop expansion (default: 500, None = unlimited)

        Returns:
            Tuple of (results, pagination_metadata)
        """
        try:
            if not self._embedding_provider:
                raise ValueError(
                    "Embedding provider not configured for semantic search"
                )

            # Type narrowing for mypy
            embedding_provider = self._embedding_provider

            # Use provided provider/model or fall back to configured defaults
            search_provider = provider or embedding_provider.name
            search_model = model or embedding_provider.model

            # logger.debug(f"Search using provider='{search_provider}', model='{search_model}'")

            # Choose search strategy based on force_strategy or provider capabilities
            use_multi_hop = False

            if force_strategy == "multi_hop":
                use_multi_hop = True
            elif force_strategy == "single_hop":
                use_multi_hop = False
            else:
                # Auto-select based on provider capabilities
                use_multi_hop = (
                    hasattr(embedding_provider, "supports_reranking")
                    and embedding_provider.supports_reranking()
                )

            if use_multi_hop:
                # Ensure provider actually supports reranking for multi-hop
                if not (
                    hasattr(embedding_provider, "supports_reranking")
                    and embedding_provider.supports_reranking()
                ):
                    logger.warning(
                        "Multi-hop strategy requested but provider doesn't support reranking, falling back to single-hop"
                    )
                    use_multi_hop = False

            if use_multi_hop:
                logger.debug(f"Using multi-hop search with reranking for: '{query}'")
                assert self._multi_hop_strategy is not None
                results, pagination = await self._multi_hop_strategy.search(
                    query=query,
                    page_size=page_size,
                    offset=offset,
                    threshold=threshold,
                    provider=search_provider,
                    model=search_model,
                    path_filter=path_filter,
                    time_limit=time_limit,
                    result_limit=result_limit,
                )
            else:
                logger.debug(f"Using standard semantic search for: '{query}'")
                assert self._single_hop_strategy is not None
                results, pagination = await self._single_hop_strategy.search(
                    query=query,
                    page_size=page_size,
                    offset=offset,
                    threshold=threshold,
                    provider=search_provider,
                    model=search_model,
                    path_filter=path_filter,
                )

            # Enhance results with additional metadata
            enhanced_results = []
            for result in results:
                enhanced_result = self._result_enhancer.enhance_search_result(result)
                enhanced_results.append(enhanced_result)

            return enhanced_results, pagination

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise


    def search_regex(
        self,
        pattern: str,
        page_size: int = 10,
        offset: int = 0,
        path_filter: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Perform regex search on code content (synchronous).

        Args:
            pattern: Regular expression pattern to search for
            page_size: Number of results per page
            offset: Starting position for pagination
            path_filter: Optional relative path to limit search scope
                (e.g., 'src/', 'tests/')

        Returns:
            Tuple of (results, pagination_metadata)
        """
        try:
            logger.debug(f"Performing regex search for pattern: '{pattern}'")

            # Perform regex search
            results, pagination = self._db.search_regex(
                pattern=pattern,
                page_size=page_size,
                offset=offset,
                path_filter=path_filter,
            )

            # Enhance results with additional metadata
            enhanced_results = []
            for result in results:
                enhanced_result = self._result_enhancer.enhance_search_result(result)
                enhanced_results.append(enhanced_result)

            logger.info(
                f"Regex search completed: {len(enhanced_results)} results found"
            )
            return enhanced_results, pagination

        except Exception as e:
            logger.error(f"Regex search failed: {e}")
            raise

    async def search_regex_async(
        self,
        pattern: str,
        page_size: int = 10,
        offset: int = 0,
        path_filter: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Perform regex search on code content (asynchronous).

        This method uses async execution to avoid blocking the event loop,
        enabling better concurrency when performing multiple searches in parallel.

        Args:
            pattern: Regular expression pattern to search for
            page_size: Number of results per page
            offset: Starting position for pagination
            path_filter: Optional relative path to limit search scope
                (e.g., 'src/', 'tests/')

        Returns:
            Tuple of (results, pagination_metadata)
        """
        try:
            logger.debug(f"Performing async regex search for pattern: '{pattern}'")

            # Perform async regex search
            results, pagination = await self._db.search_regex_async(
                pattern=pattern,
                page_size=page_size,
                offset=offset,
                path_filter=path_filter,
            )

            # Enhance results with additional metadata
            enhanced_results = []
            for result in results:
                enhanced_result = self._result_enhancer.enhance_search_result(result)
                enhanced_results.append(enhanced_result)

            logger.info(
                f"Async regex search completed: {len(enhanced_results)} results found"
            )
            return enhanced_results, pagination

        except Exception as e:
            logger.error(f"Async regex search failed: {e}")
            raise

    async def search_hybrid(
        self,
        query: str,
        regex_pattern: str | None = None,
        page_size: int = 10,
        offset: int = 0,
        semantic_weight: float = 0.7,
        threshold: float | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Perform hybrid search combining semantic and regex results.

        Args:
            query: Natural language search query
            regex_pattern: Optional regex pattern to include in search
            page_size: Number of results per page
            offset: Starting position for pagination
            semantic_weight: Weight given to semantic results (0.0-1.0)
            threshold: Optional similarity threshold for semantic results

        Returns:
            Tuple of (results, pagination_metadata)
        """
        try:
            logger.debug(
                f"Performing hybrid search: query='{query}', pattern='{regex_pattern}'"
            )

            # Perform searches concurrently
            tasks = []

            # Semantic search
            if self._embedding_provider:
                semantic_task = asyncio.create_task(
                    self.search_semantic(
                        query,
                        page_size=page_size * 2,
                        offset=offset,
                        threshold=threshold,
                    )
                )
                tasks.append(("semantic", semantic_task))

            # Regex search
            if regex_pattern:

                async def get_regex_results() -> tuple[
                    list[dict[str, Any]], dict[str, Any]
                ]:
                    return self.search_regex(
                        regex_pattern, page_size=page_size * 2, offset=offset
                    )

                tasks.append(("regex", asyncio.create_task(get_regex_results())))

            # Wait for all searches to complete
            results_by_type = {}
            pagination_data = {}
            for search_type, task in tasks:
                results, pagination = await task
                results_by_type[search_type] = results
                pagination_data[search_type] = pagination

            # Combine and rank results
            combined_results = self._result_enhancer.combine_search_results(
                semantic_results=results_by_type.get("semantic", []),
                regex_results=results_by_type.get("regex", []),
                semantic_weight=semantic_weight,
                limit=page_size,
            )

            # Create combined pagination metadata
            combined_pagination = {
                "offset": offset,
                "page_size": page_size,
                "has_more": len(combined_results) == page_size,
                "next_offset": offset + page_size
                if len(combined_results) == page_size
                else None,
                "total": None,  # Cannot estimate for hybrid search
            }

            logger.info(
                f"Hybrid search completed: {len(combined_results)} results found"
            )
            return combined_results, combined_pagination

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise

    def get_chunk_context(
        self, chunk_id: ChunkId, context_lines: int = 5
    ) -> dict[str, Any]:
        """Get additional context around a specific chunk.

        Args:
            chunk_id: ID of the chunk to get context for
            context_lines: Number of lines before/after to include

        Returns:
            Dictionary with chunk details and surrounding context
        """
        return self._context_retriever.get_chunk_context(chunk_id, context_lines)

    def get_file_chunks(self, file_path: str) -> list[dict[str, Any]]:
        """Get all chunks for a specific file.

        Args:
            file_path: Path to the file

        Returns:
            List of chunks in the file ordered by line number
        """
        results = self._context_retriever.get_file_chunks(file_path)

        # Enhance results
        enhanced_results = []
        for result in results:
            enhanced_result = self._result_enhancer.enhance_search_result(result)
            enhanced_results.append(enhanced_result)

        return enhanced_results

