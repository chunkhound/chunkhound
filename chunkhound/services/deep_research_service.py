"""Deep Research Service for ChunkHound - Backwards compatibility wrapper.

This module provides backwards compatibility for the DeepResearchService class.
The actual implementation has been moved to chunkhound.services.research.v1.pluggable_research_service.

Use PluggableResearchService directly from chunkhound.services.research.v1 for new code.
"""

from typing import TYPE_CHECKING, Any

from chunkhound.core.config.config import Config
from chunkhound.services.research.factory import ResearchServiceFactory
from chunkhound.services.research.v1.pluggable_research_service import (
    PluggableResearchService,
)

if TYPE_CHECKING:
    from chunkhound.api.cli.utils.tree_progress import TreeProgressDisplay
    from chunkhound.database_factory import DatabaseServices
    from chunkhound.embeddings import EmbeddingManager
    from chunkhound.llm_manager import LLMManager


async def run_deep_research(
    *,
    services: "DatabaseServices",
    embedding_manager: "EmbeddingManager",
    llm_manager: "LLMManager",
    query: str,
    tool_name: str = "code_research",
    progress: "TreeProgressDisplay | None" = None,
    path: str | None = None,
) -> dict[str, Any]:
    """Run deep research with the same preflight validations as the MCP tool.

    This is a convenience wrapper around the factory-based research service.
    """
    if not llm_manager or not llm_manager.is_configured():
        raise Exception(
            "LLM not configured. Configure an LLM provider via:\n"
            "1. Create .chunkhound.json with llm configuration, OR\n"
            "2. Set CHUNKHOUND_LLM_API_KEY environment variable"
        )

    if not embedding_manager or not embedding_manager.list_providers():
        raise Exception(
            "No embedding providers available. Code research requires reranking "
            "support."
        )

    embedding_provider = embedding_manager.get_provider()
    if not (
        hasattr(embedding_provider, "supports_reranking")
        and embedding_provider.supports_reranking()
    ):
        raise Exception(
            "Code research requires a provider with reranking support. "
            "Configure a rerank_model in your embedding configuration."
        )

    # Create config from environment
    config = Config.from_environment()

    # Create research service using factory (v1/v2/v3 based on config)
    research_service = ResearchServiceFactory.create(
        config=config,
        db_services=services,
        embedding_manager=embedding_manager,
        llm_manager=llm_manager,
        tool_name=tool_name,
        progress=progress,
        path_filter=path,
    )

    return await research_service.deep_research(query)


# Backwards compatibility alias
BFSResearchService = PluggableResearchService

# Re-export constants for backwards compatibility (tests access these)
from chunkhound.services.research.shared.models import (
    ENABLE_ADAPTIVE_BUDGETS,
    FOLLOWUP_OUTPUT_TOKENS_MAX,
    FOLLOWUP_OUTPUT_TOKENS_MIN,
    MAX_FOLLOWUP_QUESTIONS,
    MAX_SYMBOLS_TO_SEARCH,
    NODE_SIMILARITY_THRESHOLD,
    NUM_LLM_EXPANDED_QUERIES,
    QUERY_EXPANSION_ENABLED,
    RELEVANCE_THRESHOLD,
)

# Backwards compatibility alias
DeepResearchService = BFSResearchService

__all__ = [
    "DeepResearchService",
    "BFSResearchService",
    "run_deep_research",
    # Constants (backwards compatibility)
    "RELEVANCE_THRESHOLD",
    "NODE_SIMILARITY_THRESHOLD",
    "MAX_FOLLOWUP_QUESTIONS",
    "MAX_SYMBOLS_TO_SEARCH",
    "QUERY_EXPANSION_ENABLED",
    "NUM_LLM_EXPANDED_QUERIES",
    "ENABLE_ADAPTIVE_BUDGETS",
    "FOLLOWUP_OUTPUT_TOKENS_MIN",
    "FOLLOWUP_OUTPUT_TOKENS_MAX",
]
