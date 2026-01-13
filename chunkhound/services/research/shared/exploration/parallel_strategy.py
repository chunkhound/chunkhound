"""Parallel exploration strategy (v3 algorithm).

Runs BFS and WideCoverage exploration strategies concurrently,
merges results, deduplicates by chunk_id, and applies a single
unified elbow detection pass before file reading.
"""

import asyncio
from typing import TYPE_CHECKING, Any

from loguru import logger

from chunkhound.services.research.shared.chunk_dedup import get_chunk_id
from chunkhound.services.research.shared.exploration.elbow_filter import (
    filter_chunks_by_elbow,
    get_unified_score,
)

if TYPE_CHECKING:
    from chunkhound.llm_manager import LLMManager
    from chunkhound.services.research.shared.exploration.bfs_exploration_strategy import (  # noqa: E501
        BFSExplorationStrategy,
    )
    from chunkhound.services.research.shared.exploration.wide_coverage_strategy import (  # noqa: E501
        WideCoverageStrategy,
    )
    from chunkhound.services.research.shared.file_reader import FileReader


class ParallelExplorationStrategy:
    """Parallel exploration strategy (v3 algorithm).

    Runs BFS and WideCoverage exploration strategies concurrently with
    unified post-processing. This combines the breadth-first follow-up
    question approach of BFS with the depth exploration and gap detection
    of WideCoverage for maximum code discovery.

    The strategy:
    1. Runs BFS and WideCoverage strategies concurrently (both return raw chunks)
    2. Merges and deduplicates chunks by chunk_id (keeping highest score)
    3. Applies a single unified elbow detection pass on merged set
    4. Reads files for filtered chunks
    """

    def __init__(
        self,
        bfs_strategy: "BFSExplorationStrategy",
        wide_strategy: "WideCoverageStrategy",
        file_reader: "FileReader",
        llm_manager: "LLMManager",
    ):
        """Initialize parallel exploration strategy.

        Args:
            bfs_strategy: BFS exploration strategy instance
            wide_strategy: WideCoverage exploration strategy instance
            file_reader: FileReader for reading file contents after elbow detection
            llm_manager: LLM manager for token estimation during file reading
        """
        self._bfs = bfs_strategy
        self._wide = wide_strategy
        self._file_reader = file_reader
        self._llm_manager = llm_manager

    @property
    def name(self) -> str:
        """Strategy identifier."""
        return "parallel"

    async def explore_raw(
        self,
        root_query: str,
        initial_chunks: list[dict[str, Any]],
        phase1_threshold: float,
        path_filter: str | None = None,
        constants_context: str = "",
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Execute parallel exploration without post-processing.

        Runs BFS and WideCoverage explore_raw() concurrently, merges results.
        Returns raw merged chunks without elbow detection or file reading.

        Args:
            root_query: Original research query (injected in all LLM prompts)
            initial_chunks: Chunks from Phase 1 coverage retrieval
            phase1_threshold: Quality threshold floor from Phase 1
            path_filter: Optional path filter for searches
            constants_context: Evidence ledger constants for LLM prompts

        Returns:
            Tuple of (raw_chunks, exploration_stats):
                - raw_chunks: Merged and deduplicated chunks from both strategies
                - exploration_stats: Combined stats from both strategies
        """
        if not initial_chunks:
            logger.warning("ParallelExplorationStrategy: No initial chunks to explore")
            return [], {
                "bfs": {"nodes_explored": 0, "chunks_total": 0},
                "wide": {"chunks_before": 0, "chunks_after": 0},
                "merged_chunks": 0,
            }

        logger.info(
            f"ParallelExplorationStrategy: Starting parallel exploration "
            f"with {len(initial_chunks)} initial chunks"
        )

        # Run both strategies concurrently
        bfs_task = self._bfs.explore_raw(
            root_query=root_query,
            initial_chunks=initial_chunks,
            phase1_threshold=phase1_threshold,
            path_filter=path_filter,
            constants_context=constants_context,
        )
        wide_task = self._wide.explore_raw(
            root_query=root_query,
            initial_chunks=initial_chunks,
            phase1_threshold=phase1_threshold,
            path_filter=path_filter,
            constants_context=constants_context,
        )

        results = await asyncio.gather(bfs_task, wide_task, return_exceptions=True)

        # Handle exceptions from individual strategies
        bfs_result = results[0]
        wide_result = results[1]

        bfs_chunks: list[dict[str, Any]] = []
        bfs_stats: dict[str, Any] = {}
        wide_chunks: list[dict[str, Any]] = []
        wide_stats: dict[str, Any] = {}

        if isinstance(bfs_result, BaseException):
            logger.error(f"BFS exploration failed: {bfs_result}")
            bfs_stats = {
                "error": str(bfs_result), "nodes_explored": 0, "chunks_total": 0
            }
        else:
            bfs_chunks = bfs_result[0]
            bfs_stats = bfs_result[1]

        if isinstance(wide_result, BaseException):
            logger.error(f"WideCoverage exploration failed: {wide_result}")
            wide_stats = {
                "error": str(wide_result), "chunks_before": 0, "chunks_after": 0
            }
        else:
            wide_chunks = wide_result[0]
            wide_stats = wide_result[1]

        # Fail explicitly if both strategies failed
        if isinstance(bfs_result, BaseException) and isinstance(wide_result, BaseException):
            raise RuntimeError(
                f"Both exploration strategies failed. BFS: {bfs_result}, Wide: {wide_result}"
            )

        logger.info(
            f"ParallelExplorationStrategy: BFS found {len(bfs_chunks)} chunks, "
            f"WideCoverage found {len(wide_chunks)} chunks"
        )

        # Merge and dedupe chunks by chunk_id (keep highest score)
        merged_chunks = self._merge_and_dedupe(bfs_chunks, wide_chunks)
        logger.info(
            f"ParallelExplorationStrategy: Merged to {len(merged_chunks)} unique chunks"
        )

        return merged_chunks, {
            "bfs": bfs_stats,
            "wide": wide_stats,
            "merged_chunks": len(merged_chunks),
        }

    async def explore(
        self,
        root_query: str,
        initial_chunks: list[dict[str, Any]],
        phase1_threshold: float,
        path_filter: str | None = None,
        constants_context: str = "",
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
        """Execute parallel exploration with unified elbow detection.

        Delegates to explore_raw() for concurrent strategy execution and merging,
        then applies elbow detection and reads files.

        Args:
            root_query: Original research query (injected in all LLM prompts)
            initial_chunks: Chunks from Phase 1 coverage retrieval
            phase1_threshold: Quality threshold floor from Phase 1
            path_filter: Optional path filter for searches
            constants_context: Evidence ledger constants for LLM prompts

        Returns:
            Tuple of (expanded_chunks, exploration_stats, file_contents):
                - expanded_chunks: All chunks after merge, dedupe, and elbow filter
                - exploration_stats: Combined stats from both strategies
                - file_contents: Pre-read file contents (path -> content)
        """
        # Delegate to explore_raw for composition
        merged_chunks, raw_stats = await self.explore_raw(
            root_query=root_query,
            initial_chunks=initial_chunks,
            phase1_threshold=phase1_threshold,
            path_filter=path_filter,
            constants_context=constants_context,
        )

        if not merged_chunks:
            return [], {
                **raw_stats,
                "elbow_filter": {"method": "passthrough", "reason": "empty_input"},
                "chunks_before": len(initial_chunks),
                "chunks_after": 0,
                "chunks_added": -len(initial_chunks),
                "files_read": 0,
            }, {}

        # Apply unified elbow detection (score_key=None uses get_unified_score)
        filtered_chunks, elbow_stats = filter_chunks_by_elbow(
            merged_chunks, score_key=None
        )
        logger.info(
            f"ParallelExplorationStrategy: Elbow filter kept "
            f"{len(filtered_chunks)}/{len(merged_chunks)} chunks"
        )

        # Read files for filtered chunks (no token budget - elbow already filtered)
        logger.info(
            f"ParallelExplorationStrategy: Reading files for "
            f"{len(filtered_chunks)} chunks"
        )
        file_contents = await self._file_reader.read_files_with_budget(
            chunks=filtered_chunks,
            llm_manager=self._llm_manager,
            max_tokens=None,  # Unlimited - elbow already filtered
        )
        logger.info(
            f"ParallelExplorationStrategy: Read {len(file_contents)} files"
        )

        # Aggregate stats
        stats: dict[str, Any] = {
            **raw_stats,
            "elbow_filter": elbow_stats,
            "chunks_before": len(initial_chunks),
            "chunks_after": len(filtered_chunks),
            "chunks_added": len(filtered_chunks) - len(initial_chunks),
            "files_read": len(file_contents),
        }

        return filtered_chunks, stats, file_contents

    def _merge_and_dedupe(
        self,
        bfs_chunks: list[dict[str, Any]],
        wide_chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge chunks from both strategies, dedupe by chunk_id keeping highest score.

        Args:
            bfs_chunks: Chunks from BFS exploration
            wide_chunks: Chunks from WideCoverage exploration

        Returns:
            List of unique chunks, with highest-scored version on conflicts
        """
        chunk_map: dict[int | str, dict[str, Any]] = {}

        for chunk in bfs_chunks + wide_chunks:
            chunk_id = get_chunk_id(chunk)
            if not chunk_id:
                # Skip chunks without IDs (shouldn't happen, but be defensive)
                logger.debug("Skipping chunk without ID during merge")
                continue

            existing = chunk_map.get(chunk_id)
            if existing is None:
                chunk_map[chunk_id] = chunk
            else:
                # Keep chunk with higher score
                existing_score = get_unified_score(existing)
                new_score = get_unified_score(chunk)
                if new_score > existing_score:
                    chunk_map[chunk_id] = chunk

        return list(chunk_map.values())
