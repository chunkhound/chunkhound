"""Unified search orchestration - hybrid semantic + regex symbol search.

This module implements a multi-stage search strategy that combines:
1. Multi-hop semantic search with optional query expansion
2. Symbol extraction from semantic results
3. Parallel regex search for discovered symbols
4. Result unification at the chunk level

The unified search strategy is designed to provide comprehensive code discovery
by leveraging both semantic similarity and precise symbol matching, following
the algorithm outlined in the deep research specification.
"""

import asyncio
import re
from typing import Any

from loguru import logger

from chunkhound.database_factory import DatabaseServices
from chunkhound.embeddings import EmbeddingManager
from chunkhound.services.research.models import (
    MAX_SYMBOLS_TO_SEARCH,
    QUERY_EXPANSION_ENABLED,
    RELEVANCE_THRESHOLD,
    ResearchContext,
)


class UnifiedSearch:
    """Orchestrates unified semantic + symbol-based regex search."""

    def __init__(
        self,
        db_services: DatabaseServices,
        embedding_manager: EmbeddingManager,
    ):
        """Initialize unified search.

        Args:
            db_services: Database services bundle
            embedding_manager: Embedding manager for semantic search
        """
        self._db_services = db_services
        self._embedding_manager = embedding_manager

    async def unified_search(
        self,
        query: str,
        context: ResearchContext,
        expanded_queries: list[str] | None = None,
        emit_event_callback: Any = None,
        node_id: int | None = None,
        depth: int | None = None,
        path_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Perform unified semantic + symbol-based regex search (Steps 2-6).

        Algorithm steps:
        1. Multi-hop semantic search with internal reranking (Step 2)
        2. Extract symbols from semantic results (Step 3)
        3. Select top N symbols (Step 4) - already in relevance order from reranked results
        4. Regex search for top symbols (Step 5)
        5. Unify results at chunk level (Step 6)

        Note: Multi-hop semantic search already performs reranking internally,
        so symbols are extracted from already-reranked results and no additional
        reranking is needed.

        Args:
            query: Search query
            context: Research context with root query and ancestors
            expanded_queries: Optional list of expanded queries (if query expansion already done)
            emit_event_callback: Optional callback for emitting events
            node_id: Optional BFS node ID for event emission
            depth: Optional BFS depth for event emission

        Returns:
            List of unified chunks
        """
        search_service = self._db_services.search_service

        # Helper for event emission (if callback provided)
        async def emit_event(event_type: str, message: str, **kwargs):
            if emit_event_callback:
                await emit_event_callback(event_type, message, **kwargs)

        # Step 2: Multi-hop semantic search with reranking (optionally with query expansion)
        if QUERY_EXPANSION_ENABLED and expanded_queries:
            # Use provided expanded queries
            logger.debug("Step 2a: Using expanded queries for diverse semantic search")
            await emit_event(
                "query_expand", "Expanding query", node_id=node_id, depth=depth
            )

            logger.debug(
                f"Query expansion: 1 original + {len(expanded_queries) - 1} LLM-generated = {len(expanded_queries)} total: {expanded_queries}"
            )

            # Emit expanded queries event
            queries_preview = " | ".join(
                q[:40] + "..." if len(q) > 40 else q for q in expanded_queries[:3]
            )
            await emit_event(
                "query_expand_complete",
                f"Expanded to {len(expanded_queries)} queries",
                node_id=node_id,
                depth=depth,
                queries=len(expanded_queries),
            )

            # Run all semantic searches in parallel
            logger.debug(
                f"Step 2b: Running {len(expanded_queries)} parallel semantic searches"
            )
            search_tasks = [
                search_service.search_semantic(
                    query=expanded_q,
                    page_size=30,
                    threshold=RELEVANCE_THRESHOLD,
                    force_strategy="multi_hop",
                    path_filter=path_filter,
                )
                for expanded_q in expanded_queries
            ]
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

            # Unify results: deduplicate by chunk_id (same pattern as semantic+regex unification)
            semantic_map = {}
            for result in search_results:
                if isinstance(result, Exception):
                    logger.warning(
                        f"Semantic search failed during query expansion: {result}"
                    )
                    continue
                # Validate tuple structure before unpacking
                if not isinstance(result, tuple) or len(result) != 2:
                    logger.error(
                        f"Unexpected search result structure: {type(result)}, skipping"
                    )
                    continue
                results, _ = result
                for chunk in results:
                    chunk_id = chunk.get("chunk_id") or chunk.get("id")
                    if chunk_id and chunk_id not in semantic_map:
                        semantic_map[chunk_id] = chunk

            semantic_results = list(semantic_map.values())
            logger.debug(
                f"Unified {sum(len(r[0]) if not isinstance(r, Exception) else 0 for r in search_results)} results from {len(expanded_queries)} searches -> {len(semantic_results)} unique chunks"
            )

            # Emit search results event
            await emit_event(
                "search_semantic",
                f"Found {len(semantic_results)} chunks",
                node_id=node_id,
                depth=depth,
                chunks=len(semantic_results),
            )
        else:
            # Original single-query approach (fallback)
            logger.debug(
                f"Step 2: Running multi-hop semantic search for query: '{query}'"
            )
            await emit_event(
                "search_semantic",
                "Searching semantically",
                node_id=node_id,
                depth=depth,
            )

            semantic_results, _ = await search_service.search_semantic(
                query=query,
                page_size=30,
                threshold=RELEVANCE_THRESHOLD,
                force_strategy="multi_hop",
                path_filter=path_filter,
            )
            logger.debug(f"Semantic search returned {len(semantic_results)} chunks")

            # Emit search results event
            await emit_event(
                "search_semantic",
                f"Found {len(semantic_results)} chunks",
                node_id=node_id,
                depth=depth,
                chunks=len(semantic_results),
            )

        # Steps 3-5: Symbol extraction, reranking, and regex search
        regex_results = []
        if semantic_results:
            # Step 3: Extract symbols from semantic results
            logger.debug("Step 3: Extracting symbols from semantic results")
            await emit_event(
                "extract_symbols", "Extracting symbols", node_id=node_id, depth=depth
            )

            symbols = await self.extract_symbols_from_chunks(semantic_results)

            if symbols:
                # Step 4: Select top symbols (already in relevance order from reranked semantic results)
                logger.debug(
                    f"Step 4: Selecting top {MAX_SYMBOLS_TO_SEARCH} symbols from {len(symbols)} extracted symbols"
                )
                top_symbols = symbols[:MAX_SYMBOLS_TO_SEARCH]

                # Emit symbol extraction results
                symbols_preview = ", ".join(top_symbols[:5])
                if len(top_symbols) > 5:
                    symbols_preview += "..."
                await emit_event(
                    "extract_symbols_complete",
                    f"Extracted {len(symbols)} symbols, searching top {len(top_symbols)}",
                    node_id=node_id,
                    depth=depth,
                    symbols=len(symbols),
                )

                if top_symbols:
                    # Step 5: Regex search for top symbols
                    logger.debug(
                        f"Step 5: Running regex search for {len(top_symbols)} top symbols"
                    )
                    await emit_event(
                        "search_regex",
                        "Running regex search",
                        node_id=node_id,
                        depth=depth,
                    )

                    regex_results = await self.search_by_symbols(
                        top_symbols, path_filter=path_filter
                    )

                    # Emit regex search results
                    await emit_event(
                        "search_regex_complete",
                        f"Found {len(regex_results)} additional chunks",
                        node_id=node_id,
                        depth=depth,
                        chunks=len(regex_results),
                    )

        # Step 6: Unify results at chunk level (deduplicate by chunk_id)
        logger.debug(
            f"Step 6: Unifying {len(semantic_results)} semantic + {len(regex_results)} regex results"
        )
        unified_map = {}

        # Add semantic results first (they have relevance scores from multi-hop)
        for chunk in semantic_results:
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if chunk_id:
                unified_map[chunk_id] = chunk

        # Add regex results (only new chunks not already found)
        for chunk in regex_results:
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if chunk_id and chunk_id not in unified_map:
                unified_map[chunk_id] = chunk

        unified_chunks = list(unified_map.values())
        logger.debug(f"Unified to {len(unified_chunks)} unique chunks")

        # Note: Multi-hop semantic search already reranked results, no need to rerank again
        return unified_chunks

    async def extract_symbols_from_chunks(
        self, chunks: list[dict[str, Any]]
    ) -> list[str]:
        """Extract symbols from already-parsed chunks (language-agnostic).

        Leverages existing chunk data from UniversalParser which already extracted
        symbols for all 25+ supported languages. No re-parsing needed!

        Args:
            chunks: List of chunks from semantic search

        Returns:
            Deduplicated list of symbol names
        """
        symbols = set()

        for chunk in chunks:
            # Primary: Extract symbol name (function/class/method name)
            # This field is populated by UniversalParser for all languages
            if symbol := chunk.get("symbol"):
                if symbol and symbol.strip():
                    symbols.add(symbol.strip())

            # Secondary: Extract parameters as potential searchable symbols
            # Many functions/methods have meaningful parameter names
            metadata = chunk.get("metadata", {})
            if params := metadata.get("parameters"):
                if isinstance(params, list):
                    symbols.update(p.strip() for p in params if p and p.strip())

            # Tertiary: Extract from chunk_type-specific metadata
            # Some chunks have additional symbol information
            if chunk_type := metadata.get("kind"):
                # Skip generic types, focus on specific symbols
                if chunk_type not in ("block", "comment", "unknown"):
                    if name := chunk.get("name"):
                        symbols.add(name.strip())

        # Filter out common noise (single chars, numbers, common keywords)
        filtered_symbols = [
            s
            for s in symbols
            if len(s) > 1
            and not s.isdigit()
            and s.lower() not in {"self", "cls", "this"}
        ]

        logger.debug(
            f"Extracted {len(filtered_symbols)} symbols from {len(chunks)} chunks"
        )
        return filtered_symbols

    async def search_by_symbols(
        self, symbols: list[str], path_filter: str | None = None
    ) -> list[dict[str, Any]]:
        """Search codebase for top-ranked symbols using parallel async regex (Step 5).

        Uses async execution to avoid blocking the event loop, enabling better
        concurrency when searching for multiple symbols in parallel.

        Args:
            symbols: List of symbol names to search for

        Returns:
            List of chunks found via regex search
        """
        if not symbols:
            return []

        search_service = self._db_services.search_service

        async def search_symbol(symbol: str) -> list[dict[str, Any]]:
            """Search for a single symbol asynchronously."""
            try:
                # Escape special regex characters
                escaped = re.escape(symbol)
                # Match word boundaries to avoid partial matches
                # This works across all languages (identifier boundaries)
                pattern = rf"\b{escaped}\b"

                results, _ = await search_service.search_regex_async(
                    pattern=pattern,
                    page_size=10,  # Limit per symbol to avoid overwhelming results
                    offset=0,
                    path_filter=path_filter,
                )

                logger.debug(f"Found {len(results)} chunks for symbol '{symbol}'")
                return results

            except Exception as e:
                logger.warning(f"Regex search failed for symbol '{symbol}': {e}")
                return []

        # Run all symbol searches concurrently
        results_per_symbol = await asyncio.gather(*[search_symbol(s) for s in symbols])

        # Flatten results
        all_results = []
        for results in results_per_symbol:
            all_results.extend(results)

        logger.debug(
            f"Parallel symbol regex search complete: {len(all_results)} total chunks from {len(symbols)} symbols"
        )
        return all_results
