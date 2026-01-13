"""Protocol for research services - both v1 (BFS) and v2 (coverage-first)."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ResearchServiceProtocol(Protocol):
    """Protocol for research services (v1/v2/v3 implementations).

    Implementations should accept these constructor parameters:
        database_services: Database services bundle
        embedding_manager: Embedding manager for semantic search
        llm_manager: LLM manager for generating questions and synthesis
        config: Application configuration (optional, v2/v3 require this)
        import_resolver: Import resolver service (optional, v2/v3-only)
        tool_name: Name of the MCP tool (used in followup suggestions)
        progress: Optional TreeProgressDisplay for terminal UI (None for MCP)
        path_filter: Optional relative path to limit research scope

    Note: Python Protocols don't enforce __init__ signatures. Use
    ResearchServiceFactory.create() to instantiate implementations.
    """

    async def deep_research(self, query: str) -> dict[str, Any]:
        """Execute research on a query.

        Args:
            query: Research query to investigate

        Returns:
            Dict with answer, citations, stats, and optional fields
        """
        ...
