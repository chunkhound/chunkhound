"""Depth Exploration Service for Research (Phase 1.5).

This module implements depth exploration for the research algorithm:
1. Select top-K files by rerank score from Phase 1 coverage
2. Generate aspect-based exploration queries for each file via LLM
3. Execute unified search for each query (parallel, independent)
4. Global deduplication with Phase 1 chunks
5. Return expanded chunks with statistics

Key Invariants:
- ROOT query must be in ALL LLM prompts
- Exploration targets SAME files from different angles (not external deps)
- Queries are independent (no shared mutable state)
- Global dedup AFTER all explorations complete
- Threshold floor: effective_threshold >= phase1_threshold
"""

import asyncio
from typing import Any, cast

from loguru import logger

from chunkhound.core.config.research_config import ResearchConfig
from chunkhound.database_factory import DatabaseServices
from chunkhound.embeddings import EmbeddingManager
from chunkhound.llm_manager import LLMManager
from chunkhound.services.research.shared.chunk_dedup import (
    deduplicate_chunks,
    merge_chunk_lists,
)
from chunkhound.services.research.shared.elbow_detection import (
    compute_elbow_threshold,
)
from chunkhound.services.research.shared.failure_tracker import FailureMetrics
from chunkhound.services.research.shared.chunk_context_builder import (
    ChunkContextBuilder,
    get_chunk_text,
)
from chunkhound.services.research.shared.import_context import ImportContextService
from chunkhound.services.research.shared.import_resolution_helper import (
    resolve_and_fetch_imports,
)
from chunkhound.services.research.shared.models import (
    IMPORT_DEFAULT_SCORE,
    ResearchContext,
)
from chunkhound.services.research.shared.unified_search import UnifiedSearch
from chunkhound.services.research.shared.evidence_ledger import CONSTANTS_INSTRUCTION_SHORT


class DepthExplorationService:
    """Service for exploring existing coverage from multiple angles.

    Unlike gap detection (which finds missing external references), depth
    exploration generates queries to explore DIFFERENT ASPECTS of files
    already in coverage. This mimics follow-up question behavior
    while maintaining coverage-first philosophy.
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        embedding_manager: EmbeddingManager,
        db_services: DatabaseServices,
        config: ResearchConfig,
        import_resolver: Any | None = None,
        import_context_service: ImportContextService | None = None,
    ):
        """Initialize depth exploration service.

        Args:
            llm_manager: LLM manager for query generation
            embedding_manager: Embedding manager for unified search
            db_services: Database services bundle
            config: Research configuration
            import_resolver: Optional ImportResolverService for import resolution
            import_context_service: Optional ImportContextService for header injection
        """
        self._llm_manager = llm_manager
        self._embedding_manager = embedding_manager
        self._db_services = db_services
        self._config = config
        self._import_resolver = import_resolver
        self._import_context_service = import_context_service
        self._unified_search = UnifiedSearch(db_services, embedding_manager, config)

    async def explore_coverage_depth(
        self,
        root_query: str,
        covered_chunks: list[dict],
        phase1_threshold: float,
        path_filter: str | None = None,
        constants_context: str = "",
    ) -> tuple[list[dict], dict]:
        """Explore existing coverage from multiple angles (Phase 1.5 main entry).

        Args:
            root_query: Original research query
            covered_chunks: Raw chunk dicts from Phase 1
            phase1_threshold: Quality threshold from Phase 1 (floor for results)
            path_filter: Path filter to pass through to searches
            constants_context: Constants ledger context for LLM prompts

        Returns:
            Tuple of (expanded_chunks, stats) where:
                - expanded_chunks: Merged coverage + exploration chunks, deduplicated
                - stats: Statistics about exploration
        """
        if not covered_chunks:
            logger.warning("No covered chunks to explore")
            return covered_chunks, {
                "files_explored": 0,
                "queries_generated": 0,
                "chunks_added": 0,
            }

        logger.info(
            f"Phase 1.5: Depth exploration starting with {len(covered_chunks)} chunks"
        )

        # Step 1: Group chunks by file and select top-K files
        file_to_chunks = self._group_chunks_by_file(covered_chunks)
        top_files = self._select_top_files(
            file_to_chunks, self._config.max_exploration_files
        )
        logger.info(
            f"Step 1.5.1: Selected {len(top_files)} top files for exploration"
        )

        if not top_files:
            logger.info("No files selected for exploration")
            return covered_chunks, {
                "files_explored": 0,
                "queries_generated": 0,
                "chunks_added": 0,
            }

        # Step 2: Generate exploration queries for each file (parallel)
        exploration_queries, generation_metrics = await self._generate_all_exploration_queries(
            root_query, top_files, file_to_chunks, constants_context
        )
        total_queries = sum(len(queries) for queries in exploration_queries.values())
        logger.info(
            f"Step 1.5.2: Generated {total_queries} exploration queries "
            f"across {len(exploration_queries)} files "
            f"({generation_metrics.success_count}/{generation_metrics.total_operations} files succeeded)"
        )

        if not any(exploration_queries.values()):
            logger.info("No exploration queries generated")
            return covered_chunks, {
                "files_explored": len(top_files),
                "queries_generated": 0,
                "chunks_added": 0,
            }

        # Step 3: Execute unified search for each query (parallel, independent)
        exploration_results = await self._execute_exploration_queries(
            root_query, exploration_queries, phase1_threshold, path_filter
        )
        total_results = sum(len(r) for r in exploration_results)
        logger.info(
            f"Step 1.5.3: Exploration searches returned {total_results} chunks"
        )

        # Step 4: Global deduplication (SYNC POINT)
        unified_exploration_chunks = self._global_dedup(exploration_results)
        logger.info(
            f"Step 1.5.4: Global dedup: {total_results} -> "
            f"{len(unified_exploration_chunks)} unique exploration chunks"
        )

        # Step 5: Merge with coverage chunks
        expanded_chunks = self._merge_coverage(
            covered_chunks, unified_exploration_chunks
        )
        chunks_added = len(expanded_chunks) - len(covered_chunks)
        logger.info(
            f"Step 1.5.5: Final merge: {len(covered_chunks)} + "
            f"{len(unified_exploration_chunks)} -> {len(expanded_chunks)} total "
            f"({chunks_added} new chunks)"
        )

        # Step 6: Import resolution (if enabled)
        import_chunks_added = 0
        if self._config.import_resolution_enabled and self._import_resolver:
            import_chunks = await resolve_and_fetch_imports(
                chunks=expanded_chunks,
                import_resolver=self._import_resolver,
                db_services=self._db_services,
                config=self._config,
                path_filter=path_filter,
                default_score=IMPORT_DEFAULT_SCORE,
            )
            if import_chunks:
                expanded_chunks = self._merge_coverage(expanded_chunks, import_chunks)
                import_chunks_added = len(import_chunks)
                logger.info(
                    f"Step 1.5.6: Import resolution: added {import_chunks_added} chunks "
                    f"from {len({c.get('file_path') for c in import_chunks})} import files"
                )

        stats = {
            "files_explored": len(top_files),
            "queries_generated": total_queries,
            "exploration_chunks_found": total_results,
            "exploration_chunks_unique": len(unified_exploration_chunks),
            "chunks_added": chunks_added,
            "import_chunks_added": import_chunks_added,
            "total_chunks": len(expanded_chunks),
            "query_generation_failures": generation_metrics.to_dict(),
        }

        return expanded_chunks, stats

    def _group_chunks_by_file(
        self, chunks: list[dict]
    ) -> dict[str, list[dict]]:
        """Group chunks by file path.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Dictionary mapping file_path -> list of chunks
        """
        file_to_chunks: dict[str, list[dict]] = {}
        for chunk in chunks:
            file_path = chunk.get("file_path", "")
            if file_path:
                if file_path not in file_to_chunks:
                    file_to_chunks[file_path] = []
                file_to_chunks[file_path].append(chunk)
        return file_to_chunks

    def _select_top_files(
        self,
        file_to_chunks: dict[str, list[dict]],
        max_files: int,
    ) -> list[str]:
        """Select top-K files by average rerank score.

        Args:
            file_to_chunks: Mapping of file_path -> chunks
            max_files: Maximum files to select

        Returns:
            List of top file paths
        """
        # Calculate average score per file
        file_scores: dict[str, float] = {}
        for file_path, chunks in file_to_chunks.items():
            scores = [c.get("rerank_score", 0.0) for c in chunks]
            file_scores[file_path] = sum(scores) / len(scores) if scores else 0.0

        # Sort by score descending and take top-K
        sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
        return [file_path for file_path, _ in sorted_files[:max_files]]

    async def _generate_all_exploration_queries(
        self,
        root_query: str,
        top_files: list[str],
        file_to_chunks: dict[str, list[dict]],
        constants_context: str = "",
    ) -> tuple[dict[str, list[str]], FailureMetrics]:
        """Generate exploration queries for all top files in parallel.

        Args:
            root_query: Original research query
            top_files: List of file paths to explore
            file_to_chunks: Mapping of file_path -> chunks
            constants_context: Constants ledger context for LLM prompts

        Returns:
            Dictionary mapping file_path -> list of exploration queries
        """
        # Get concurrency limit from utility provider
        llm = self._llm_manager.get_utility_provider()
        max_concurrency = llm.get_synthesis_concurrency()
        semaphore = asyncio.Semaphore(max_concurrency)

        async def generate_for_file(file_path: str) -> tuple[str, list[str]]:
            async with semaphore:
                chunks = file_to_chunks.get(file_path, [])
                queries = await self._generate_exploration_queries(
                    root_query, chunks, file_path, constants_context
                )
                return file_path, queries

        # Run in parallel (bounded by semaphore)
        tasks = [generate_for_file(fp) for fp in top_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results with failure tracking
        exploration_queries: dict[str, list[str]] = {}
        generation_metrics = FailureMetrics(total_operations=len(top_files))

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                file_path = top_files[i]
                generation_metrics.add_failure(file_path, result)
                logger.warning(f"Query generation failed for {file_path}: {result}")
                continue
            file_path, queries = cast(tuple[str, list[str]], result)
            exploration_queries[file_path] = queries

        return exploration_queries, generation_metrics

    async def _generate_exploration_queries(
        self,
        root_query: str,
        file_chunks: list[dict],
        file_path: str,
        constants_context: str = "",
    ) -> list[str]:
        """Generate exploration queries for a specific file.

        CRITICAL: Includes ROOT query in prompt to maintain focus.

        Args:
            root_query: Original research query
            file_chunks: Chunks found in this file
            file_path: Path to the file
            constants_context: Constants ledger context for LLM prompts

        Returns:
            List of exploration queries (1-2 per file)
        """
        llm = self._llm_manager.get_utility_provider()
        num_queries = self._config.exploration_queries_per_file

        # Use shared ChunkContextBuilder for imports and chunk summaries
        builder = ChunkContextBuilder(
            import_context_service=self._import_context_service,
            llm_manager=self._llm_manager,
        )

        # Extract imports for file context (header injection)
        imports_context = ""
        if self._import_context_service and file_chunks:
            first_chunk = file_chunks[0]
            content = get_chunk_text(first_chunk)
            imports = self._import_context_service.get_file_imports(file_path, content)
            if imports:
                imports_context = "IMPORTS:\n" + "\n".join(imports) + "\n\n"

        # Build chunk summary using shared builder
        chunk_context = builder.build_chunk_summary(file_chunks, max_chunks=5)

        # JSON schema for structured output
        schema = {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": num_queries,
                }
            },
            "required": ["queries"],
        }

        # Build constants section if available
        # Note: Depth exploration uses inline instruction in prompt, not separate section
        constants_section = ""
        if constants_context:
            constants_section = f"\n{constants_context}\n"

        # Exploration query generation prompt
        prompt = f"""RESEARCH QUERY: {root_query}
{constants_section}
FILE: {file_path}
{imports_context}CHUNKS FOUND ({len(file_chunks)} total):
{chunk_context}

Generate {num_queries} specific queries to explore DIFFERENT ASPECTS of this file
that would help answer the ROOT QUERY.

Focus on:
1. Component interactions and data flow not yet covered
2. Implementation patterns or algorithms in this file
3. How this file relates to other parts of the system

IMPORTANT:
- Target aspects WITHIN THIS FILE, not external dependencies
- Each query should explore a different angle
- Queries should be specific enough to find new chunks
- {CONSTANTS_INSTRUCTION_SHORT}

Output JSON with queries array."""

        try:
            result = await llm.complete_structured(
                prompt=prompt,
                json_schema=schema,
                max_completion_tokens=512,
            )

            queries: list[str] = result.get("queries", [])
            logger.debug(
                f"Generated {len(queries)} exploration queries for {file_path}"
            )
            return queries

        except Exception as e:
            logger.warning(f"Exploration query generation failed for {file_path}: {e}")
            return []

    async def _execute_exploration_queries(
        self,
        root_query: str,
        exploration_queries: dict[str, list[str]],
        phase1_threshold: float,
        path_filter: str | None,
    ) -> list[list[dict]]:
        """Execute all exploration queries in parallel.

        Args:
            root_query: Original research query
            exploration_queries: Mapping of file_path -> queries
            phase1_threshold: Quality threshold floor
            path_filter: Path filter for searches

        Returns:
            List of result lists (one per query)
        """

        async def execute_single_query(
            query: str, source_file: str
        ) -> list[dict]:
            """Execute a single exploration query."""
            context = ResearchContext(root_query=root_query)

            # Run unified search with compound reranking (root + exploration query)
            chunks = await self._unified_search.unified_search(
                query=query,
                context=context,
                rerank_queries=[root_query, query],
                path_filter=path_filter,
            )

            # Apply window expansion if enabled
            if self._config.window_expansion_enabled:
                chunks = await self._unified_search.expand_chunk_windows(
                    chunks,
                    window_lines=self._config.window_expansion_lines
                )

            # Compute adaptive threshold
            if chunks:
                scores = [c.get("rerank_score", 0.0) for c in chunks]
                exploration_threshold = compute_elbow_threshold(scores)
            else:
                exploration_threshold = phase1_threshold

            # Apply threshold floor
            effective_threshold = max(phase1_threshold, exploration_threshold)

            # Filter chunks by threshold
            filtered = [
                c for c in chunks if c.get("rerank_score", 0.0) >= effective_threshold
            ]

            logger.debug(
                f"Exploration query for {source_file}: {len(chunks)} -> "
                f"{len(filtered)} chunks (threshold: {effective_threshold:.3f})"
            )
            return filtered

        # Flatten queries and execute in parallel
        tasks = []
        for file_path, queries in exploration_queries.items():
            for query in queries:
                tasks.append(execute_single_query(query, file_path))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        query_results: list[list[dict]] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Exploration query execution failed: {result}")
                query_results.append([])
            elif isinstance(result, list):
                query_results.append(result)

        return query_results

    def _global_dedup(self, query_results: list[list[dict]]) -> list[dict]:
        """Global deduplication across all exploration results.

        SYNC POINT: This happens ONLY after all queries complete.
        Conflict resolution: keep chunk with highest rerank_score.

        Args:
            query_results: List of result lists from exploration queries

        Returns:
            Deduplicated chunks
        """
        return deduplicate_chunks(query_results, log_prefix="Exploration dedup")

    def _merge_coverage(
        self, covered_chunks: list[dict], exploration_chunks: list[dict]
    ) -> list[dict]:
        """Merge coverage and exploration chunks.

        Args:
            covered_chunks: Chunks from Phase 1
            exploration_chunks: Chunks from exploration

        Returns:
            Merged and deduplicated chunks
        """
        return merge_chunk_lists(
            covered_chunks, exploration_chunks, log_prefix="Exploration merge"
        )
