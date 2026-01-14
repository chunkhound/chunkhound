"""Exploration strategy protocol for research services.

This module defines the common interface for exploration strategies,
enabling v1 BFS and v2 wide coverage algorithms to be swapped in either
research pipeline.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ExplorationStrategy(Protocol):
    """Strategy for exploring codebase beyond initial coverage.

    Both v1 BFS exploration and v2 wide coverage (depth + gap detection)
    implement this protocol, making them interchangeable in research pipelines.
    """

    @property
    def name(self) -> str:
        """Strategy identifier (e.g., 'wide_coverage', 'bfs')."""
        ...

    async def explore(
        self,
        root_query: str,
        initial_chunks: list[dict[str, Any]],
        phase1_threshold: float,
        path_filter: str | None = None,
        constants_context: str = "",
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
        """Execute exploration strategy.

        Args:
            root_query: Original research query (injected in all LLM prompts)
            initial_chunks: Chunks from Phase 1 coverage retrieval
            phase1_threshold: Quality threshold floor from Phase 1
            path_filter: Optional path filter for searches
            constants_context: Evidence ledger constants for LLM prompts

        Returns:
            Tuple of (expanded_chunks, exploration_stats, file_contents):
                - expanded_chunks: All chunks including initial + explored
                - exploration_stats: Statistics about exploration process
                - file_contents: Pre-read file contents (path -> content)
        """
        ...

    async def explore_raw(
        self,
        root_query: str,
        initial_chunks: list[dict[str, Any]],
        phase1_threshold: float,
        path_filter: str | None = None,
        constants_context: str = "",
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Execute exploration without post-processing (for parallel composition).

        Used by ParallelExplorationStrategy to run multiple strategies concurrently,
        then apply unified elbow detection and file reading on merged results.

        Args:
            root_query: Original research query (injected in all LLM prompts)
            initial_chunks: Chunks from Phase 1 coverage retrieval
            phase1_threshold: Quality threshold floor from Phase 1
            path_filter: Optional path filter for searches
            constants_context: Evidence ledger constants for LLM prompts

        Returns:
            Tuple of (raw_chunks, exploration_stats):
                - raw_chunks: All chunks without elbow filtering
                - exploration_stats: Statistics about exploration process
        """
        ...
