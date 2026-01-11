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

from chunkhound.core.config.research_config import ResearchConfig
from chunkhound.database_factory import DatabaseServices
from chunkhound.embeddings import EmbeddingManager
from chunkhound.services.research.shared.chunk_context_builder import get_chunk_text
from chunkhound.services.research.shared.chunk_dedup import get_chunk_id
from chunkhound.services.research.shared.models import (
    MAX_SYMBOLS_TO_SEARCH,
    QUERY_EXPANSION_ENABLED,
    REGEX_AUGMENTATION_RATIO,
    REGEX_MIN_RESULTS,
    ResearchContext,
)


class UnifiedSearch:
    """Orchestrates unified semantic + symbol-based regex search."""

    def __init__(
        self,
        db_services: DatabaseServices,
        embedding_manager: EmbeddingManager,
        config: ResearchConfig | None = None,
    ):
        """Initialize unified search.

        Args:
            db_services: Database services bundle
            embedding_manager: Embedding manager for semantic search
            config: Optional research configuration (for exhaustive mode control)
        """
        self._db_services = db_services
        self._embedding_manager = embedding_manager
        self._config = config

    async def unified_search(
        self,
        query: str,
        context: ResearchContext,
        expanded_queries: list[str] | None = None,
        rerank_queries: list[str] | None = None,
        emit_event_callback: Any = None,
        node_id: int | None = None,
        depth: int | None = None,
        path_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Perform unified semantic + symbol-based regex search (Steps 2-7).

        Algorithm steps:
        1. Multi-hop semantic search with internal reranking (Step 2)
        2. Extract symbols from semantic results (Step 3)
        3. Select top N symbols (Step 4) - already in relevance order from reranked results
        4. Regex search for top symbols (Step 5)
        5. Unify results at chunk level (Step 6)
        6. Unified rerank semantic + regex against ROOT query (Step 7)

        Step 7 ensures optimal ranking by reranking the combined semantic + regex
        results together against the ROOT query (not BFS queries), preventing
        diversity collapse and allowing highly relevant regex matches to outrank
        marginally relevant semantic matches.

        Args:
            query: Search query
            context: Research context with root query and ancestors
            expanded_queries: Optional list of expanded queries (if query expansion already done)
            rerank_queries: Optional list of queries for compound reranking (default: use root query only)
            emit_event_callback: Optional callback for emitting events
            node_id: Optional BFS node ID for event emission
            depth: Optional BFS depth for event emission
            path_filter: Optional path filter to limit search scope

        Returns:
            List of unified chunks
        """
        search_service = self._db_services.search_service

        # Helper for event emission (if callback provided)
        async def emit_event(event_type: str, message: str, **kwargs: Any) -> None:
            if emit_event_callback:
                await emit_event_callback(event_type, message, **kwargs)

        # Step 2: Multi-hop semantic search with reranking (optionally with query expansion)
        if QUERY_EXPANSION_ENABLED and expanded_queries:
            # Use provided expanded queries (expansion events emitted by caller)
            logger.debug("Step 2a: Using expanded queries for diverse semantic search")
            logger.debug(
                f"Query expansion: 1 original + {len(expanded_queries) - 1} LLM-generated = {len(expanded_queries)} total: {expanded_queries}"
            )

            # Run all semantic searches in parallel
            logger.debug(
                f"Step 2b: Running {len(expanded_queries)} parallel semantic searches"
            )
            # Determine time and result limits from config (if available)
            time_limit = self._config.get_effective_time_limit() if self._config else None
            result_limit = self._config.get_effective_result_limit() if self._config else None

            page_size = self._config.initial_page_size if self._config else 30
            search_tasks = [
                search_service.search_semantic(
                    query=expanded_q,
                    page_size=page_size,
                    threshold=None,  # No threshold filtering - elbow detection computes threshold
                    force_strategy="multi_hop",
                    path_filter=path_filter,
                    time_limit=time_limit,
                    result_limit=result_limit,
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
                    chunk_id = get_chunk_id(chunk)
                    if chunk_id and chunk_id not in semantic_map:
                        semantic_map[chunk_id] = chunk

            semantic_results = list(semantic_map.values())
            total_chunks = 0
            for r in search_results:
                if not isinstance(r, BaseException):
                    total_chunks += len(r[0])
            logger.debug(
                f"Unified {total_chunks} results from {len(expanded_queries)} searches -> {len(semantic_results)} unique chunks"
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

            # Determine time and result limits from config (if available)
            time_limit = self._config.get_effective_time_limit() if self._config else None
            result_limit = self._config.get_effective_result_limit() if self._config else None

            page_size = self._config.initial_page_size if self._config else 30
            semantic_results, _ = await search_service.search_semantic(
                query=query,
                page_size=page_size,
                threshold=None,  # No threshold filtering - elbow detection computes threshold
                force_strategy="multi_hop",
                path_filter=path_filter,
                time_limit=time_limit,
                result_limit=result_limit,
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
                max_symbols = self._config.max_symbols if self._config else MAX_SYMBOLS_TO_SEARCH
                logger.debug(
                    f"Step 4: Selecting top {max_symbols} symbols from {len(symbols)} extracted symbols"
                )
                top_symbols = symbols[:max_symbols]

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
                    # Compute dynamic target using config values with fallback
                    regex_min = self._config.regex_min_results if self._config else REGEX_MIN_RESULTS
                    regex_ratio = self._config.regex_augmentation_ratio if self._config else REGEX_AUGMENTATION_RATIO
                    target_count = max(
                        regex_min,
                        int(len(semantic_results) * regex_ratio),
                    )
                    target_per_symbol = max(1, target_count // len(top_symbols))

                    logger.debug(
                        f"Step 5: Running regex search for {len(top_symbols)} top "
                        f"symbols (target: {target_count} total, min={regex_min}, "
                        f"ratio={regex_ratio}, {target_per_symbol} per symbol)"
                    )
                    await emit_event(
                        "search_regex",
                        "Running regex search",
                        node_id=node_id,
                        depth=depth,
                    )

                    # Collect semantic chunk IDs for exclusion (before regex search)
                    semantic_chunk_ids = set()
                    for chunk in semantic_results:
                        chunk_id = get_chunk_id(chunk)
                        if chunk_id:
                            semantic_chunk_ids.add(chunk_id)

                    regex_results = await self.search_by_symbols(
                        top_symbols,
                        target_per_symbol=target_per_symbol,
                        path_filter=path_filter,
                        exclude_ids=semantic_chunk_ids,
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
            chunk_id = get_chunk_id(chunk)
            if chunk_id:
                unified_map[chunk_id] = chunk

        # Add regex results (only new chunks not already found)
        for chunk in regex_results:
            chunk_id = get_chunk_id(chunk)
            if chunk_id and chunk_id not in unified_map:
                unified_map[chunk_id] = chunk

        combined_pool = list(unified_map.values())
        logger.debug(f"Unified to {len(combined_pool)} unique chunks")

        # Step 7: Unified rerank against ROOT query (or compound queries)
        # Reranks semantic + regex results TOGETHER for optimal ranking
        # Uses ROOT query (not BFS queries) to prevent diversity collapse
        # Optionally uses compound reranking with multiple queries
        embedding_provider = self._embedding_manager.get_provider()
        if (
            hasattr(embedding_provider, "supports_reranking")
            and embedding_provider.supports_reranking()
            and len(combined_pool) > 1
        ):
            try:
                documents = [get_chunk_text(c) for c in combined_pool]

                # Compound reranking: rerank against each query and average scores
                if rerank_queries and len(rerank_queries) > 1:
                    logger.debug(
                        f"Step 7: Compound reranking against {len(rerank_queries)} queries"
                    )

                    # Rerank against each query
                    all_scores: list[dict[int, float]] = []
                    for rerank_query in rerank_queries:
                        rerank_results = await embedding_provider.rerank(
                            query=rerank_query,
                            documents=documents,
                        )

                        # Collect scores for this query
                        query_scores = {}
                        for rerank_result in rerank_results:
                            if 0 <= rerank_result.index < len(combined_pool):
                                query_scores[rerank_result.index] = rerank_result.score
                        all_scores.append(query_scores)

                    # Compute compound score as average across all queries
                    for idx in range(len(combined_pool)):
                        scores = [
                            query_scores.get(idx, 0.0) for query_scores in all_scores
                        ]
                        compound_score = sum(scores) / len(scores) if scores else 0.0
                        combined_pool[idx]["rerank_score"] = compound_score

                    logger.debug(
                        f"Step 7: Compound rerank complete - averaged scores from {len(rerank_queries)} queries"
                    )
                else:
                    # Single query reranking (default behavior)
                    rerank_query = (
                        rerank_queries[0] if rerank_queries else context.root_query
                    )
                    rerank_results = await embedding_provider.rerank(
                        query=rerank_query,
                        documents=documents,
                    )

                    # Apply reranking scores (same pattern as multi_hop_strategy.py)
                    for rerank_result in rerank_results:
                        if 0 <= rerank_result.index < len(combined_pool):
                            combined_pool[rerank_result.index]["rerank_score"] = (
                                rerank_result.score
                            )

                    logger.debug(
                        f"Step 7: Unified rerank complete - {len(combined_pool)} chunks reranked against root query"
                    )

                # Sort by rerank score descending
                combined_pool.sort(
                    key=lambda c: c.get("rerank_score", 0.0),
                    reverse=True,
                )

            except Exception as e:
                logger.warning(
                    f"Unified rerank failed, keeping semantic-priority order: {e}"
                )

        return combined_pool

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
        self,
        symbols: list[str],
        target_per_symbol: int = 10,
        path_filter: str | None = None,
        exclude_ids: set[int | str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search codebase for top-ranked symbols using parallel async regex (Step 5).

        Uses async execution to avoid blocking the event loop, enabling better
        concurrency when searching for multiple symbols in parallel.

        Args:
            symbols: List of symbol names to search for
            target_per_symbol: Target results per symbol (dynamic, semantic-based)
            path_filter: Optional path filter to limit search scope
            exclude_ids: Optional set of chunk IDs to exclude (semantic chunks)

        Returns:
            List of chunks found via regex search (excluding chunks in exclude_ids)
        """
        if not symbols:
            return []

        search_service = self._db_services.search_service
        exclude_ids = exclude_ids or set()

        async def search_symbol(symbol: str) -> list[dict[str, Any]]:
            """Search for a single symbol asynchronously with pagination.

            Implements internal pagination loop to ensure we find target_per_symbol
            UNDISCOVERED chunks (chunks not in exclude_ids), as per spec lines 192-216.
            """
            try:
                # Escape special regex characters
                escaped = re.escape(symbol)
                # Match word boundaries to avoid partial matches
                # This works across all languages (identifier boundaries)
                pattern = rf"\b{escaped}\b"

                # Internal pagination loop: keep fetching pages until we have enough undiscovered chunks
                results: list[dict[str, Any]] = []
                offset = 0
                # Use config value if available, fall back to 100 (spec default)
                scan_page_size = self._config.regex_scan_page_size if self._config else 100
                seen_chunk_ids = exclude_ids.copy()  # Track all seen chunk IDs (excluded + collected)
                # Safety limit to prevent infinite loops when exclusions are large
                max_pages = 20

                pages_fetched = 0
                while len(results) < target_per_symbol and pages_fetched < max_pages:
                    page, _ = await search_service.search_regex_async(
                        pattern=pattern,
                        page_size=scan_page_size,
                        offset=offset,
                        path_filter=path_filter,
                    )
                    if not page:
                        break  # No more results available from backend

                    # Filter out excluded chunk IDs and collect undiscovered chunks
                    for chunk in page:
                        chunk_id = get_chunk_id(chunk)
                        if chunk_id and chunk_id not in seen_chunk_ids:
                            results.append(chunk)
                            seen_chunk_ids.add(chunk_id)
                            if len(results) >= target_per_symbol:
                                break

                    offset += scan_page_size
                    pages_fetched += 1

                logger.debug(
                    f"Found {len(results)} undiscovered chunks for symbol '{symbol}' (target: {target_per_symbol})"
                )
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
            f"Parallel symbol regex search complete: {len(all_results)} chunks from {len(symbols)} symbols (after exclusion)"
        )
        return all_results

    async def expand_chunk_windows(
        self, chunks: list[dict], window_lines: int = 50
    ) -> list[dict]:
        """Expand retrieved chunks with neighboring context using line ranges.

        For each file represented in chunks, finds all chunks within window_lines
        of the retrieved chunks' line ranges.

        This method is idempotent - chunks already marked as expanded (via
        `_window_expanded` flag) will be preserved but not re-expanded.

        Args:
            chunks: Retrieved chunks to expand
            window_lines: Number of lines to expand before/after

        Returns:
            Original chunks plus neighboring chunks (deduplicated)
        """
        if not chunks:
            return chunks

        # Separate already-expanded chunks from chunks that need expansion
        already_expanded = []
        to_expand = []
        for c in chunks:
            if c.get("_window_expanded"):
                already_expanded.append(c)
            else:
                to_expand.append(c)

        # If all chunks are already expanded, return as-is
        if not to_expand:
            logger.debug(
                f"All {len(chunks)} chunks already expanded, skipping window expansion"
            )
            return chunks

        # Group unexpanded chunks by file_id
        by_file: dict[int, list[dict]] = {}
        for c in to_expand:
            file_id = c.get("file_id")
            if file_id is not None:
                by_file.setdefault(file_id, []).append(c)

        # Start with all original chunks (preserve already-expanded ones)
        expanded_chunks = list(chunks)
        existing_ids = {get_chunk_id(c) for c in expanded_chunks}

        for file_id, file_chunks in by_file.items():
            # Find line range to cover (with window expansion)
            min_line = min(c.get("start_line", 0) for c in file_chunks) - window_lines
            max_line = max(c.get("end_line", 0) for c in file_chunks) + window_lines

            # Use the new get_chunks_in_range method
            neighbors = await self._get_chunks_in_range(
                file_id=file_id,
                start_line=max(1, min_line),
                end_line=max_line
            )

            # Deduplicate by chunk_id
            for neighbor in neighbors:
                nid = get_chunk_id(neighbor)
                if nid not in existing_ids:
                    expanded_chunks.append(neighbor)
                    existing_ids.add(nid)

        # Mark all chunks as expanded (idempotent tracking)
        for chunk in expanded_chunks:
            chunk["_window_expanded"] = True

        newly_expanded = len(to_expand) + len(expanded_chunks) - len(chunks)
        logger.debug(
            f"Expanded {len(to_expand)} chunks to {len(expanded_chunks)} chunks "
            f"(+{len(expanded_chunks) - len(chunks)} neighbors, "
            f"{len(already_expanded)} already expanded)"
        )
        return expanded_chunks

    async def _get_chunks_in_range(
        self, file_id: int, start_line: int, end_line: int
    ) -> list[dict]:
        """Get chunks in a line range from database.

        Args:
            file_id: ID of the file to search within
            start_line: Start line of the range
            end_line: End line of the range

        Returns:
            List of chunk dictionaries overlapping the range
        """
        # Access the database provider through db_services
        provider = self._db_services.provider
        return provider.get_chunks_in_range(file_id, start_line, end_line)
