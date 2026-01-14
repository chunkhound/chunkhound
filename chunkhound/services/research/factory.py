"""Factory for creating research services based on configuration."""

from collections.abc import Callable
from typing import TYPE_CHECKING

from chunkhound.core.config.config import Config
from chunkhound.services.research.protocol import ResearchServiceProtocol

if TYPE_CHECKING:
    from chunkhound.api.cli.utils.tree_progress import TreeProgressDisplay
    from chunkhound.database_factory import DatabaseServices
    from chunkhound.embeddings import EmbeddingManager
    from chunkhound.llm_manager import LLMManager
    from chunkhound.services.research.shared.import_resolver import (
        ImportResolverService,
    )


def _create_import_resolver(config: Config) -> "ImportResolverService | None":
    """Create import resolver if enabled in config."""
    if not config.research.import_resolution_enabled:
        return None
    from chunkhound.parsers.parser_factory import ParserFactory
    from chunkhound.services.research.shared.import_resolver import (
        ImportResolverService,
    )

    return ImportResolverService(ParserFactory())


def _create_v1_service(
    config: Config,
    db_services: "DatabaseServices",
    embedding_manager: "EmbeddingManager",
    llm_manager: "LLMManager",
    tool_name: str,
    progress: "TreeProgressDisplay | None",
    path_filter: str | None,
) -> ResearchServiceProtocol:
    """Create v1 research service with BFS exploration strategy."""
    from chunkhound.services.research.shared.exploration import (
        BFSExplorationStrategy,
    )
    from chunkhound.services.research.v1.pluggable_research_service import (
        PluggableResearchService,
    )

    exploration_strategy = BFSExplorationStrategy(
        llm_manager=llm_manager,
        embedding_manager=embedding_manager,
        db_services=db_services,
        config=config.research,
    )

    return PluggableResearchService(
        database_services=db_services,
        embedding_manager=embedding_manager,
        llm_manager=llm_manager,
        exploration_strategy=exploration_strategy,
        tool_name=tool_name,
        progress=progress,
        path_filter=path_filter,
        config=config.research,
    )


def _create_v2_service(
    config: Config,
    db_services: "DatabaseServices",
    embedding_manager: "EmbeddingManager",
    llm_manager: "LLMManager",
    tool_name: str,
    progress: "TreeProgressDisplay | None",
    path_filter: str | None,
) -> ResearchServiceProtocol:
    """Create v2 research service with wide coverage exploration strategy."""
    from chunkhound.services.research.shared.exploration import (
        WideCoverageStrategy,
    )
    from chunkhound.services.research.v1.pluggable_research_service import (
        PluggableResearchService,
    )

    import_resolver = _create_import_resolver(config)

    exploration_strategy = WideCoverageStrategy(
        llm_manager=llm_manager,
        embedding_manager=embedding_manager,
        db_services=db_services,
        config=config.research,
        import_resolver=import_resolver,
    )

    return PluggableResearchService(
        database_services=db_services,
        embedding_manager=embedding_manager,
        llm_manager=llm_manager,
        exploration_strategy=exploration_strategy,
        tool_name=tool_name,
        progress=progress,
        path_filter=path_filter,
        config=config.research,
    )


def _create_v3_service(
    config: Config,
    db_services: "DatabaseServices",
    embedding_manager: "EmbeddingManager",
    llm_manager: "LLMManager",
    tool_name: str,
    progress: "TreeProgressDisplay | None",
    path_filter: str | None,
) -> ResearchServiceProtocol:
    """Create v3 research service with parallel BFS + wide coverage exploration."""
    from chunkhound.services.research.shared.exploration import (
        BFSExplorationStrategy,
        ParallelExplorationStrategy,
        WideCoverageStrategy,
    )
    from chunkhound.services.research.shared.file_reader import FileReader
    from chunkhound.services.research.v1.pluggable_research_service import (
        PluggableResearchService,
    )

    import_resolver = _create_import_resolver(config)

    bfs_strategy = BFSExplorationStrategy(
        llm_manager=llm_manager,
        embedding_manager=embedding_manager,
        db_services=db_services,
        config=config.research,
    )

    wide_strategy = WideCoverageStrategy(
        llm_manager=llm_manager,
        embedding_manager=embedding_manager,
        db_services=db_services,
        config=config.research,
        import_resolver=import_resolver,
    )

    parallel_strategy = ParallelExplorationStrategy(
        bfs_strategy=bfs_strategy,
        wide_strategy=wide_strategy,
        file_reader=FileReader(db_services),
        llm_manager=llm_manager,
    )

    return PluggableResearchService(
        database_services=db_services,
        embedding_manager=embedding_manager,
        llm_manager=llm_manager,
        exploration_strategy=parallel_strategy,
        tool_name=tool_name,
        progress=progress,
        path_filter=path_filter,
        config=config.research,
    )


# Type alias for creator functions
_CreatorFunc = Callable[
    [
        Config,
        "DatabaseServices",
        "EmbeddingManager",
        "LLMManager",
        str,
        "TreeProgressDisplay | None",
        str | None,
    ],
    ResearchServiceProtocol,
]

# Registry mapping algorithm names to their creator functions
STRATEGY_REGISTRY: dict[str, _CreatorFunc] = {
    "v1": _create_v1_service,
    "v2": _create_v2_service,
    "v3": _create_v3_service,
}


class ResearchServiceFactory:
    """Factory for creating research services."""

    @staticmethod
    def create(
        config: Config,
        db_services: "DatabaseServices",
        embedding_manager: "EmbeddingManager",
        llm_manager: "LLMManager",
        tool_name: str = "code_research",
        progress: "TreeProgressDisplay | None" = None,
        path_filter: str | None = None,
    ) -> ResearchServiceProtocol:
        """Create a research service based on config.

        Args:
            config: Application configuration
            db_services: Database services
            embedding_manager: Embedding manager
            llm_manager: LLM manager
            tool_name: Name of the MCP tool (used in followup suggestions)
            progress: Optional TreeProgressDisplay for terminal UI (None for MCP)
            path_filter: Optional relative path to limit research scope

        Returns:
            Research service instance (v1, v2, or v3 based on config.research.algorithm)
        """
        algorithm = config.research.algorithm
        creator = STRATEGY_REGISTRY.get(algorithm, STRATEGY_REGISTRY["v1"])
        return creator(
            config, db_services, embedding_manager, llm_manager,
            tool_name, progress, path_filter
        )
