"""Exploration strategies for research services.

This module provides swappable exploration strategies that can be used
in either v1 BFS or v2 coverage-first research pipelines.

Available Strategies:
    ExplorationStrategy: Protocol defining the common interface
    BFSExplorationStrategy: V1 algorithm (BFS tree with follow-up questions)
    WideCoverageStrategy: V2 algorithm (depth exploration + gap detection)
    ParallelExplorationStrategy: v3 algorithm (concurrent BFS + WideCoverage)

Example usage:
    from chunkhound.services.research.shared.exploration import (
        ExplorationStrategy,
        WideCoverageStrategy,
        BFSExplorationStrategy,
        ParallelExplorationStrategy,
    )

    # Create strategy
    strategy = WideCoverageStrategy(
        llm_manager=llm_manager,
        embedding_manager=embedding_manager,
        db_services=db_services,
        config=research_config,
    )

    # Use in research pipeline
    expanded_chunks, stats, file_contents = await strategy.explore(
        root_query=query,
        initial_chunks=phase1_chunks,
        phase1_threshold=threshold,
    )
"""

from chunkhound.services.research.shared.exploration.bfs_exploration_strategy import (
    BFSExplorationStrategy,
)
from chunkhound.services.research.shared.exploration.elbow_filter import (
    filter_chunks_by_elbow,
    get_unified_score,
)
from chunkhound.services.research.shared.exploration.parallel_strategy import (
    ParallelExplorationStrategy,
)
from chunkhound.services.research.shared.exploration.protocol import (
    ExplorationStrategy,
)
from chunkhound.services.research.shared.exploration.wide_coverage_strategy import (
    WideCoverageStrategy,
)

__all__ = [
    "BFSExplorationStrategy",
    "ExplorationStrategy",
    "ParallelExplorationStrategy",
    "WideCoverageStrategy",
    "filter_chunks_by_elbow",
    "get_unified_score",
]
