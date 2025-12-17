"""Factory for creating research services based on configuration."""

from typing import TYPE_CHECKING

from chunkhound.core.config.config import Config
from chunkhound.services.research.protocol import ResearchServiceProtocol

if TYPE_CHECKING:
    from chunkhound.api.cli.utils.tree_progress import TreeProgressDisplay
    from chunkhound.database_factory import DatabaseServices
    from chunkhound.embeddings import EmbeddingManager
    from chunkhound.llm_manager import LLMManager


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
            Research service instance (v1 or v2 based on config.research.algorithm)
        """
        algorithm = config.research.algorithm

        if algorithm == "v2":
            # Lazy import to avoid circular imports
            from chunkhound.parsers.parser_factory import ParserFactory
            from chunkhound.services.research.shared.import_resolver import (
                ImportResolverService,
            )
            from chunkhound.services.research.v2.coverage_research_service import (
                CoverageResearchService,
            )

            # Create import resolver if import resolution is enabled
            import_resolver = None
            if config.research.import_resolution_enabled:
                parser_factory = ParserFactory()
                import_resolver = ImportResolverService(parser_factory)

            return CoverageResearchService(
                database_services=db_services,
                embedding_manager=embedding_manager,
                llm_manager=llm_manager,
                config=config,
                tool_name=tool_name,
                progress=progress,
                path_filter=path_filter,
                import_resolver=import_resolver,
            )
        else:
            # Default to v1 (BFS)
            from chunkhound.services.research.v1.bfs_research_service import (
                BFSResearchService,
            )

            return BFSResearchService(
                database_services=db_services,
                embedding_manager=embedding_manager,
                llm_manager=llm_manager,
                tool_name=tool_name,
                progress=progress,
                path_filter=path_filter,
            )
