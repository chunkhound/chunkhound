"""Wide coverage exploration strategy.

Combines depth exploration (Phase 1.5) and gap detection (Phase 2)
for comprehensive codebase coverage.
"""

from typing import Any

from loguru import logger

from chunkhound.core.config.research_config import ResearchConfig
from chunkhound.database_factory import DatabaseServices
from chunkhound.embeddings import EmbeddingManager
from chunkhound.llm_manager import LLMManager
from chunkhound.services.research.shared.exploration.elbow_filter import (
    filter_chunks_by_elbow,
)
from chunkhound.services.research.shared.file_reader import FileReader
from chunkhound.services.research.shared.import_context import ImportContextService


class WideCoverageStrategy:
    """Wide coverage exploration strategy (V2 algorithm).

    Implements ExplorationStrategy by composing DepthExplorationService
    (Phase 1.5) and GapDetectionService (Phase 2) into a single explore() call.

    The strategy:
    1. Depth exploration (if enabled): Explores existing coverage from
       multiple angles by generating aspect-based queries for top files
    2. Gap detection (always): Identifies and fills semantic gaps by
       clustering coverage, detecting missing references, and filling them
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
        """Initialize wide coverage strategy.

        Args:
            llm_manager: LLM manager for gap detection and unification
            embedding_manager: Embedding manager for semantic operations
            db_services: Database services bundle
            config: Research configuration (controls depth_exploration_enabled)
            import_resolver: Optional ImportResolverService for import resolution
            import_context_service: Optional ImportContextService for header injection
        """
        # Lazy imports to avoid circular dependency
        from chunkhound.services.research.shared.depth_exploration import (
            DepthExplorationService,
        )
        from chunkhound.services.research.shared.gap_detection import GapDetectionService

        self._config = config
        self._llm_manager = llm_manager
        self._db_services = db_services
        self._file_reader = FileReader(db_services)
        self._depth_exploration = DepthExplorationService(
            llm_manager=llm_manager,
            embedding_manager=embedding_manager,
            db_services=db_services,
            config=config,
            import_resolver=import_resolver,
            import_context_service=import_context_service,
        )
        self._gap_detection = GapDetectionService(
            llm_manager=llm_manager,
            embedding_manager=embedding_manager,
            db_services=db_services,
            config=config,
            import_resolver=import_resolver,
            import_context_service=import_context_service,
        )

    @property
    def name(self) -> str:
        """Strategy identifier."""
        return "wide_coverage"

    async def explore(
        self,
        root_query: str,
        initial_chunks: list[dict[str, Any]],
        phase1_threshold: float,
        path_filter: str | None = None,
        constants_context: str = "",
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
        """Execute wide coverage exploration.

        Runs depth exploration (if enabled) followed by gap detection
        to maximize coverage of relevant code.

        Args:
            root_query: Original research query (injected in all LLM prompts)
            initial_chunks: Chunks from Phase 1 coverage retrieval
            phase1_threshold: Quality threshold floor from Phase 1
            path_filter: Optional path filter for searches
            constants_context: Evidence ledger constants for LLM prompts

        Returns:
            Tuple of (expanded_chunks, exploration_stats, file_contents):
                - expanded_chunks: All chunks including initial + explored
                - exploration_stats: Combined stats from both phases
                - file_contents: Pre-read file contents (path -> content)
        """
        stats: dict[str, Any] = {}
        current_chunks = initial_chunks

        # Phase 1.5: Depth exploration (if enabled)
        if self._config.depth_exploration_enabled:
            logger.info(
                f"WideCoverageStrategy: Starting depth exploration "
                f"with {len(current_chunks)} chunks"
            )
            current_chunks, depth_stats = (
                await self._depth_exploration.explore_coverage_depth(
                    root_query=root_query,
                    covered_chunks=current_chunks,
                    phase1_threshold=phase1_threshold,
                    path_filter=path_filter,
                    constants_context=constants_context,
                )
            )
            stats["depth"] = depth_stats
            logger.info(
                f"WideCoverageStrategy: Depth exploration complete, "
                f"now have {len(current_chunks)} chunks"
            )

        # Phase 2: Gap detection (always runs)
        logger.info(
            f"WideCoverageStrategy: Starting gap detection "
            f"with {len(current_chunks)} chunks"
        )
        all_chunks, gap_stats = await self._gap_detection.detect_and_fill_gaps(
            root_query=root_query,
            covered_chunks=current_chunks,
            phase1_threshold=phase1_threshold,
            path_filter=path_filter,
            constants_context=constants_context,
        )
        stats["gap"] = gap_stats
        logger.info(
            f"WideCoverageStrategy: Gap detection complete, "
            f"now have {len(all_chunks)} chunks"
        )

        # Apply elbow-based filtering to final chunk set
        filtered_chunks, elbow_stats = filter_chunks_by_elbow(
            all_chunks, score_key="rerank_score"
        )
        stats["elbow_filter"] = elbow_stats
        logger.info(
            f"WideCoverageStrategy: Elbow filter kept "
            f"{len(filtered_chunks)}/{len(all_chunks)} chunks"
        )

        # Read files for filtered chunks (no token budget - elbow already filtered)
        logger.info(
            f"WideCoverageStrategy: Reading files for {len(filtered_chunks)} chunks"
        )
        file_contents = await self._file_reader.read_files_with_budget(
            chunks=filtered_chunks,
            llm_manager=self._llm_manager,
            max_tokens=None,  # Unlimited - elbow already filtered
        )
        logger.info(
            f"WideCoverageStrategy: Read {len(file_contents)} files"
        )

        # Aggregate top-level stats
        stats["chunks_before"] = len(initial_chunks)
        stats["chunks_after"] = len(filtered_chunks)
        stats["chunks_added"] = len(filtered_chunks) - len(initial_chunks)
        stats["files_read"] = len(file_contents)

        return filtered_chunks, stats, file_contents

    async def explore_raw(
        self,
        root_query: str,
        initial_chunks: list[dict[str, Any]],
        phase1_threshold: float,
        path_filter: str | None = None,
        constants_context: str = "",
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Execute wide coverage exploration without elbow filtering (for parallel mode).

        Returns raw chunks from depth exploration and gap detection without
        elbow detection or file reading. The caller (ParallelExplorationStrategy)
        handles unified elbow detection and file reading after merging results
        from multiple strategies.

        Args:
            root_query: Original research query
            initial_chunks: Chunks from Phase 1 coverage retrieval
            phase1_threshold: Quality threshold floor from Phase 1
            path_filter: Optional path filter for searches
            constants_context: Evidence ledger constants for LLM prompts

        Returns:
            Tuple of (raw_chunks, exploration_stats):
                - raw_chunks: All chunks from exploration without filtering
                - exploration_stats: Combined stats from depth and gap phases
        """
        stats: dict[str, Any] = {}
        current_chunks = initial_chunks

        # Phase 1.5: Depth exploration (if enabled)
        if self._config.depth_exploration_enabled:
            logger.info(
                f"WideCoverageStrategy.explore_raw: Starting depth exploration "
                f"with {len(current_chunks)} chunks"
            )
            current_chunks, depth_stats = (
                await self._depth_exploration.explore_coverage_depth(
                    root_query=root_query,
                    covered_chunks=current_chunks,
                    phase1_threshold=phase1_threshold,
                    path_filter=path_filter,
                    constants_context=constants_context,
                )
            )
            stats["depth"] = depth_stats
            logger.info(
                f"WideCoverageStrategy.explore_raw: Depth exploration complete, "
                f"now have {len(current_chunks)} chunks"
            )

        # Phase 2: Gap detection (always runs)
        logger.info(
            f"WideCoverageStrategy.explore_raw: Starting gap detection "
            f"with {len(current_chunks)} chunks"
        )
        all_chunks, gap_stats = await self._gap_detection.detect_and_fill_gaps(
            root_query=root_query,
            covered_chunks=current_chunks,
            phase1_threshold=phase1_threshold,
            path_filter=path_filter,
            constants_context=constants_context,
        )
        stats["gap"] = gap_stats
        logger.info(
            f"WideCoverageStrategy.explore_raw: Gap detection complete, "
            f"now have {len(all_chunks)} chunks"
        )

        # NO elbow filtering here
        # NO file reading here

        # Aggregate stats
        stats["chunks_before"] = len(initial_chunks)
        stats["chunks_after"] = len(all_chunks)
        stats["chunks_added"] = len(all_chunks) - len(initial_chunks)

        return all_chunks, stats
