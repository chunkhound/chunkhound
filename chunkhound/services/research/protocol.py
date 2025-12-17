"""Protocol for research services - both v1 (BFS) and v2 (coverage-first)."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from chunkhound.api.cli.utils.tree_progress import TreeProgressDisplay


@dataclass
class ResearchResult:
    """Result from research service."""

    answer: str
    citations: list[dict[str, Any]]
    stats: dict[str, Any]
    # Optional fields for v2
    gap_queries: list[str] | None = None
    phase_timings: dict[str, float] | None = None


class ResearchServiceProtocol(Protocol):
    """Protocol for research services (v1 and v2 implementations)."""

    def __init__(
        self,
        database_services: Any,
        embedding_manager: Any,
        llm_manager: Any,
        config: Any = None,
        import_resolver: Any | None = None,
        tool_name: str = "code_research",
        progress: "TreeProgressDisplay | None" = None,
        path_filter: str | None = None,
    ) -> None:
        """Initialize research service.

        Args:
            database_services: Database services bundle
            embedding_manager: Embedding manager for semantic search
            llm_manager: LLM manager for generating questions and synthesis
            config: Application configuration (optional, v2 requires this)
            import_resolver: Import resolver service (optional, v2-only)
            tool_name: Name of the MCP tool (used in followup suggestions)
            progress: Optional TreeProgressDisplay for terminal UI (None for MCP)
            path_filter: Optional relative path to limit research scope
        """
        ...

    async def deep_research(self, query: str) -> dict[str, Any]:
        """Execute research on a query.

        Args:
            query: Research query to investigate

        Returns:
            Dict with answer, citations, stats, and optional fields
        """
        ...
