"""Coverage-First Research Service for ChunkHound (v2).

This module implements the main orchestrator for the Coverage-First Research
algorithm, coordinating all four phases:

Phase 1: Full Coverage Retrieval
    - Query expansion (optional)
    - Multi-hop semantic search
    - Symbol extraction
    - Regex augmentation
    - Unified reranking

Phase 1.5: Depth Exploration (optional)
    - Select top-K files by score
    - Generate aspect-based exploration queries
    - Execute unified search for each query
    - Global deduplication with Phase 1 chunks

Phase 2: Gap Detection and Filling
    - Cluster coverage chunks
    - Shard by token budget
    - Parallel gap detection
    - Gap unification
    - Gap selection
    - Parallel gap filling (independent)
    - Global deduplication

Phase 3: Synthesis
    - Compound reranking (ROOT + gap queries)
    - Budget allocation
    - Boundary expansion
    - Compression loop (REQUIRED)
    - Final synthesis with citations

Key Invariants:
- ROOT query injected at EVERY LLM call
- path_filter propagated to ALL search operations
- Chunk deduplication by ID at all merge points
- Gap fills are independent (no shared mutable state)
- Token budget always respected
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from loguru import logger

from chunkhound.core.config.config import Config
from chunkhound.database_factory import DatabaseServices
from chunkhound.embeddings import EmbeddingManager
from chunkhound.llm_manager import LLMManager
from chunkhound.parsers.parser_factory import ParserFactory
from chunkhound.services.research.shared.elbow_detection import (
    compute_elbow_threshold,
)
from chunkhound.services.research.shared.import_context import ImportContextService
from chunkhound.services.research.shared.import_resolution_helper import (
    resolve_and_fetch_imports,
)
from chunkhound.services.research.shared.models import (
    IMPORT_DEFAULT_SCORE,
    ResearchContext,
)
from chunkhound.services.research.shared.query_expander import QueryExpander
from chunkhound.services.research.shared.unified_search import UnifiedSearch
from chunkhound.services.research.v2.coverage_synthesis import CoverageSynthesisEngine
from chunkhound.services.research.v2.depth_exploration import DepthExplorationService
from chunkhound.services.research.v2.gap_detection import GapDetectionService

# Note: We don't import v2 models here since services work with raw dicts
# CoverageResult and other models are for external API, not internal service communication

if TYPE_CHECKING:
    from chunkhound.api.cli.utils.tree_progress import TreeProgressDisplay


@dataclass
class _Phase1Result:
    """Internal result type for Phase 1 (uses raw dicts, not CoverageChunk objects)."""

    chunks: list[dict[str, Any]]  # Raw chunk dicts from unified search
    symbols: list[str]  # Extracted symbols
    phase1_threshold: float  # Dynamic threshold
    stats: dict[str, Any]  # Phase 1 statistics


class CoverageResearchService:
    """Service for performing deep research using coverage-first algorithm (v2).

    Implements the ResearchServiceProtocol with a three-phase approach:
    1. Coverage: Retrieve ALL relevant chunks via unified search
    2. Gap Detection: Identify and fill semantic gaps
    3. Synthesis: Combine chunks into answer within token budget
    """

    def __init__(
        self,
        database_services: DatabaseServices,
        embedding_manager: EmbeddingManager,
        llm_manager: LLMManager,
        config: Config,
        tool_name: str = "code_research",
        progress: "TreeProgressDisplay | None" = None,
        path_filter: str | None = None,
        import_resolver: Any | None = None,
    ):
        """Initialize coverage-first research service.

        Args:
            database_services: Database services bundle
            embedding_manager: Embedding manager for semantic search
            llm_manager: LLM manager for gap detection and synthesis
            config: Application configuration (contains ResearchConfig)
            tool_name: Name of the MCP tool (used in followup suggestions)
            progress: Optional TreeProgressDisplay for terminal UI (None for MCP)
            path_filter: Optional relative path to limit research scope
            import_resolver: Optional ImportResolverService for import resolution
        """
        self._db_services = database_services
        self._embedding_manager = embedding_manager
        self._llm_manager = llm_manager
        self._config = config.research  # ResearchConfig instance
        self._tool_name = tool_name
        self._progress = progress
        self._path_filter = path_filter
        self._import_resolver = import_resolver

        # Initialize ImportContextService for header injection
        parser_factory = ParserFactory()
        self._import_context_service = ImportContextService(parser_factory)

        # Initialize sub-services
        self._unified_search = UnifiedSearch(database_services, embedding_manager, self._config)
        self._query_expander = QueryExpander(llm_manager)
        self._depth_exploration_service = DepthExplorationService(
            llm_manager=llm_manager,
            embedding_manager=embedding_manager,
            db_services=database_services,
            config=self._config,
            import_resolver=import_resolver,
            import_context_service=self._import_context_service,
        )
        self._gap_detection_service = GapDetectionService(
            llm_manager=llm_manager,
            embedding_manager=embedding_manager,
            db_services=database_services,
            config=self._config,
            import_resolver=import_resolver,
            import_context_service=self._import_context_service,
        )
        self._synthesis_engine = CoverageSynthesisEngine(
            llm_manager=llm_manager,
            embedding_manager=embedding_manager,
            db_services=database_services,
            config=self._config,
            unified_search=self._unified_search,
            import_resolver=import_resolver,
            import_context_service=self._import_context_service,
            progress=progress,
        )

        logger.info(
            f"CoverageResearchService initialized with algorithm=v2, "
            f"path_filter={path_filter}, config={self._config}"
        )

    def _extract_imports_for_new_files(
        self,
        chunks: list[dict],
        existing_file_imports: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        """Extract imports for files not already in existing_file_imports.

        Called after each phase to accumulate imports as new files are discovered.
        Each file's imports are extracted exactly once when the first chunk from
        that file is encountered.

        Args:
            chunks: List of chunk dicts with 'file_path' and 'content' fields
            existing_file_imports: Dict of already-extracted imports to update

        Returns:
            Updated file_imports dict (same reference as input)
        """
        for chunk in chunks:
            file_path = chunk.get("file_path", "")
            if file_path and file_path not in existing_file_imports:
                content = chunk.get("content", "")
                imports = self._import_context_service.get_file_imports(
                    file_path, content
                )
                if imports:
                    existing_file_imports[file_path] = imports
        return existing_file_imports

    async def deep_research(self, query: str) -> dict[str, Any]:
        """Execute coverage-first research on a query (implements ResearchServiceProtocol).

        This is the main entry point that orchestrates all three phases of the
        coverage-first research algorithm.

        Args:
            query: Research query to investigate

        Returns:
            Dict with:
                - answer: Final synthesized response with citations
                - metadata: Comprehensive stats from all phases
                - gap_queries: Gap queries that were filled (optional)

        Phase timings are tracked and returned in metadata.phase_timings.
        """
        logger.info(f"Starting coverage-first research for query: '{query}'")
        start_time = time.time()

        # Track phase timings
        phase_timings: dict[str, float] = {}

        # Emit main start event
        await self._emit_event("main_start", f"Starting coverage-first research: {query[:60]}...")

        try:
            # Initialize file_imports dict - accumulates imports as new files discovered
            file_imports: dict[str, list[str]] = {}

            # Phase 1: Full Coverage Retrieval
            phase1_start = time.time()
            await self._emit_event("phase1_start", "Phase 1: Coverage retrieval")

            coverage_result = await self._phase1_coverage(query)

            # Extract imports for Phase 1 files (before Phase 1.5)
            file_imports = self._extract_imports_for_new_files(
                coverage_result.chunks, file_imports
            )
            logger.debug(f"Phase 1: Extracted imports for {len(file_imports)} files")

            phase_timings["phase1_ms"] = (time.time() - phase1_start) * 1000
            phase1_chunk_count = len(coverage_result.chunks)  # Store original count
            await self._emit_event(
                "phase1_complete",
                f"Phase 1 complete: {phase1_chunk_count} chunks",
                chunks=phase1_chunk_count,
            )

            # Phase 1.5: Depth Exploration (optional)
            depth_stats: dict[str, Any] = {}  # Initialize for metadata
            if self._config.depth_exploration_enabled:
                phase15_start = time.time()
                await self._emit_event("phase15_start", "Phase 1.5: Depth exploration")

                (
                    expanded_chunks,
                    depth_stats,
                ) = await self._depth_exploration_service.explore_coverage_depth(
                    root_query=query,
                    covered_chunks=coverage_result.chunks,
                    phase1_threshold=coverage_result.phase1_threshold,
                    path_filter=self._path_filter,
                )

                # Update coverage_result for Phase 2
                coverage_result.chunks = expanded_chunks

                # Extract imports for NEW files from depth exploration
                file_imports = self._extract_imports_for_new_files(
                    expanded_chunks, file_imports
                )
                logger.debug(f"Phase 1.5: file_imports now has {len(file_imports)} files")

                phase_timings["phase15_ms"] = (time.time() - phase15_start) * 1000
                chunks_added = depth_stats.get("chunks_added", 0)
                files_explored = depth_stats.get("files_explored", 0)
                queries_gen = depth_stats.get("queries_generated", 0)
                await self._emit_event(
                    "phase15_complete",
                    f"Phase 1.5 complete: {chunks_added} chunks added",
                    files_explored=files_explored,
                    queries_generated=queries_gen,
                    chunks_added=chunks_added,
                )

                logger.info(
                    f"Phase 1.5 complete: explored {files_explored} files, "
                    f"added {chunks_added} chunks"
                )

            # Phase 2: Gap Detection and Filling
            phase2_start = time.time()
            await self._emit_event("phase2_start", "Phase 2: Gap detection")

            all_chunks, gap_stats = await self._phase2_gap_detection(
                query, coverage_result
            )

            # Extract imports for NEW files from gap filling
            file_imports = self._extract_imports_for_new_files(
                all_chunks, file_imports
            )
            logger.debug(f"Phase 2: file_imports now has {len(file_imports)} files")

            phase_timings["phase2_ms"] = (time.time() - phase2_start) * 1000
            await self._emit_event(
                "phase2_complete",
                f"Phase 2 complete: {gap_stats.get('chunks_added', 0)} gap chunks added",
                gaps_detected=gap_stats.get("gaps_found", 0),
                gaps_filled=gap_stats.get("gaps_filled", 0),
                chunks_added=gap_stats.get("chunks_added", 0),
            )

            # Extract gap queries for compound context in Phase 3
            gap_queries = gap_stats.get("gap_queries", [])

            # Phase 3: Synthesis
            phase3_start = time.time()
            await self._emit_event("phase3_start", "Phase 3: Synthesis")

            answer, citations, synthesis_stats = await self._phase3_synthesis(
                query, all_chunks, gap_queries, file_imports
            )

            phase_timings["phase3_ms"] = (time.time() - phase3_start) * 1000
            await self._emit_event(
                "phase3_complete",
                f"Phase 3 complete: {synthesis_stats.get('final_tokens', 0)} tokens generated",
                tokens=synthesis_stats.get("final_tokens", 0),
            )

            # Calculate total time
            phase_timings["total_ms"] = (time.time() - start_time) * 1000

            # Emit main complete event
            await self._emit_event(
                "main_complete",
                f"Research complete: {phase_timings['total_ms'] / 1000:.1f}s",
            )

            # Build comprehensive metadata
            metadata = {
                "phase1_chunks": phase1_chunk_count,
                "phase15_chunks_added": depth_stats.get("chunks_added", 0),
                "phase2_chunks": len(all_chunks) - len(coverage_result.chunks),
                "gaps_detected": gap_stats.get("gaps_found", 0),
                "gaps_filled": gap_stats.get("gaps_filled", 0),
                "phase_timings": phase_timings,
                "token_budget": synthesis_stats.get("token_budget", {}),
                "depth_exploration": depth_stats if depth_stats else None,
                **gap_stats,  # Include full gap stats
                **synthesis_stats,  # Include full synthesis stats
            }

            logger.info(
                f"Coverage-first research complete: {phase_timings['total_ms']:.0f}ms total, "
                f"{len(all_chunks)} chunks, {len(gap_queries)} gap queries"
            )

            return {
                "answer": answer,
                "metadata": metadata,
                "gap_queries": gap_queries if gap_queries else None,
            }

        except Exception as e:
            logger.error(f"Coverage-first research failed: {e}", exc_info=True)
            await self._emit_event("main_error", f"Research failed: {e}")
            raise

    async def _phase1_coverage(self, query: str) -> _Phase1Result:
        """Phase 1: Full Coverage Retrieval.

        Steps:
        1. Query expansion (if enabled)
        2. Multi-hop semantic search
        3. Symbol extraction
        4. Regex augmentation
        5. Unified reranking
        6. Threshold computation

        Args:
            query: User research query

        Returns:
            _Phase1Result with chunks, symbols, threshold, and stats
        """
        logger.info("Phase 1: Starting full coverage retrieval")

        # Create research context
        context = ResearchContext(root_query=query)

        # Step 1.1: Query expansion (optional)
        expanded_queries: list[str] | None = None
        if self._config.query_expansion_enabled:
            await self._emit_event("query_expand", "Expanding query variants")

            expanded_queries = await self._query_expander.expand_query_with_llm(
                query, context
            )

            logger.info(
                f"Step 1.1: Query expansion: {len(expanded_queries)} queries generated"
            )
            await self._emit_event(
                "query_expand_complete",
                f"Expanded to {len(expanded_queries)} queries",
                queries=len(expanded_queries),
            )
        else:
            # Use original query only
            expanded_queries = [query]
            logger.info("Step 1.1: Query expansion disabled, using original query only")

        # Steps 1.2-1.7: Run unified search with expanded queries
        await self._emit_event("search_semantic", "Running unified search")

        chunks = await self._unified_search.unified_search(
            query=query,
            context=context,
            expanded_queries=expanded_queries,
            path_filter=self._path_filter,
        )

        logger.info(
            f"Steps 1.2-1.7: Unified search complete: {len(chunks)} chunks retrieved"
        )

        # Apply window expansion if enabled
        if self._config.window_expansion_enabled:
            chunks = await self._unified_search.expand_chunk_windows(
                chunks,
                window_lines=self._config.window_expansion_lines
            )
            logger.info(
                f"Step 1.8: Window expansion complete: {len(chunks)} chunks after expansion"
            )

        # Apply import resolution if enabled
        if self._config.import_resolution_enabled and self._import_resolver:
            import_chunks = await resolve_and_fetch_imports(
                chunks=chunks,
                import_resolver=self._import_resolver,
                db_services=self._db_services,
                config=self._config,
                path_filter=self._path_filter,
                default_score=IMPORT_DEFAULT_SCORE,
            )
            if import_chunks:
                # Merge import chunks (dedup by chunk_id)
                chunk_map = {c.get("chunk_id") or c.get("id"): c for c in chunks}
                for ic in import_chunks:
                    ic_id = ic.get("chunk_id") or ic.get("id")
                    if ic_id and ic_id not in chunk_map:
                        chunk_map[ic_id] = ic
                chunks = list(chunk_map.values())
                logger.info(
                    f"Step 1.9: Import resolution: added {len(import_chunks)} chunks "
                    f"from {len({c.get('file_path') for c in import_chunks})} import files"
                )

        # Extract symbols from chunks (already done by unified_search, but need for stats)
        symbols = await self._unified_search.extract_symbols_from_chunks(chunks)
        logger.info(f"Step 1.3: Extracted {len(symbols)} symbols from chunks")

        # Compute phase1 threshold via elbow detection
        phase1_threshold = compute_elbow_threshold(chunks)
        logger.info(f"Computed phase1_threshold: {phase1_threshold:.3f}")

        # Build stats
        stats = {
            "query_expansion_enabled": self._config.query_expansion_enabled,
            "num_expanded_queries": len(expanded_queries) if expanded_queries else 1,
            "semantic_chunks": len(chunks),
            "symbols_extracted": len(symbols),
            "phase1_threshold": phase1_threshold,
        }

        return _Phase1Result(
            chunks=chunks,
            symbols=symbols[:self._config.max_symbols],  # Limit to configured max
            phase1_threshold=phase1_threshold,
            stats=stats,
        )

    async def _phase2_gap_detection(
        self, query: str, coverage_result: _Phase1Result
    ) -> tuple[list[dict], dict]:
        """Phase 2: Gap Detection and Filling.

        Delegates to GapDetectionService for the full gap detection pipeline.

        Args:
            query: Original research query
            coverage_result: Results from Phase 1

        Returns:
            Tuple of (all_chunks, gap_stats) where:
                - all_chunks: Coverage + gap chunks, deduplicated
                - gap_stats: Statistics about gap detection and filling
        """
        logger.info("Phase 2: Starting gap detection and filling")

        await self._emit_event("gap_detect", "Detecting gaps in coverage")

        all_chunks, gap_stats = await self._gap_detection_service.detect_and_fill_gaps(
            root_query=query,
            covered_chunks=coverage_result.chunks,
            phase1_threshold=coverage_result.phase1_threshold,
            path_filter=self._path_filter,
        )

        # gap_stats already contains gap_queries from gap_detection_service (line 172 in gap_detection.py)
        logger.info(
            f"Phase 2 complete: {len(all_chunks)} total chunks "
            f"({gap_stats.get('gaps_filled', 0)} gaps filled)"
        )

        return all_chunks, gap_stats

    async def _phase3_synthesis(
        self,
        query: str,
        all_chunks: list[dict],
        gap_queries: list[str],
        file_imports: dict[str, list[str]],
    ) -> tuple[str, list[dict], dict]:
        """Phase 3: Synthesis.

        Delegates to CoverageSynthesisEngine for the full synthesis pipeline.

        Args:
            query: Original research query
            all_chunks: All chunks from Phase 2
            gap_queries: Gap queries that were filled
            file_imports: Pre-extracted imports per file (avoids re-extraction)

        Returns:
            Tuple of (answer, citations, stats)
        """
        logger.info("Phase 3: Starting synthesis")

        await self._emit_event("synthesis_rerank", "Reranking files")

        answer, citations, stats = await self._synthesis_engine.synthesize(
            root_query=query,
            all_chunks=all_chunks,
            gap_queries=gap_queries,
            target_tokens=self._config.target_tokens,
            file_imports=file_imports,
        )

        logger.info(
            f"Phase 3 complete: {stats.get('final_tokens', 0)} tokens generated"
        )

        return answer, citations, stats


    async def _emit_event(
        self,
        event_type: str,
        message: str,
        **metadata: Any,
    ) -> None:
        """Emit a progress event (compatible with TreeProgressDisplay).

        Args:
            event_type: Event type identifier
            message: Human-readable event description
            **metadata: Additional event data (chunks, files, tokens, etc.)
        """
        if not self._progress:
            return

        try:
            await self._progress.emit_event(
                event_type=event_type,
                message=message,
                metadata=metadata,
            )
        except Exception as e:
            logger.warning(f"Failed to emit progress event '{event_type}': {e}")

