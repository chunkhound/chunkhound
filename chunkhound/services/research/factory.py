"""Factory for creating research services based on configuration."""

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

        if algorithm == "v2":
            # v2 = v1 synthesis + wide coverage exploration
            from chunkhound.services.research.shared.exploration import (
                WideCoverageStrategy,
            )
            from chunkhound.services.research.v1.pluggable_research_service import (
                PluggableResearchService,
            )

            import_resolver = _create_import_resolver(config)

            # Create wide coverage exploration strategy (v2 algorithm)
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
        elif algorithm == "v3":
            # v3 = parallel BFS + WideCoverage with unified elbow detection
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

            # Create BFS exploration strategy
            bfs_strategy = BFSExplorationStrategy(
                llm_manager=llm_manager,
                embedding_manager=embedding_manager,
                db_services=db_services,
                config=config.research,
            )

            # Create wide coverage exploration strategy
            wide_strategy = WideCoverageStrategy(
                llm_manager=llm_manager,
                embedding_manager=embedding_manager,
                db_services=db_services,
                config=config.research,
                import_resolver=import_resolver,
            )

            # Create parallel strategy wrapping both
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
        else:
            # Default to v1 with BFS exploration strategy
            from chunkhound.services.research.shared.exploration import (
                BFSExplorationStrategy,
            )
            from chunkhound.services.research.v1.pluggable_research_service import (
                PluggableResearchService,
            )

            # Create BFS exploration strategy
            v1_exploration_strategy = BFSExplorationStrategy(
                llm_manager=llm_manager,
                embedding_manager=embedding_manager,
                db_services=db_services,
            )

            return PluggableResearchService(
                database_services=db_services,
                embedding_manager=embedding_manager,
                llm_manager=llm_manager,
                exploration_strategy=v1_exploration_strategy,
                tool_name=tool_name,
                progress=progress,
                path_filter=path_filter,
                config=config.research,
            )
