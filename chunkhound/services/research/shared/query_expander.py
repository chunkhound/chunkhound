"""Query expansion for semantic search in deep research.

This module handles query expansion and transformation for semantic search,
optimizing queries for embedding model position bias and generating diverse
search variations using LLM.
"""

from loguru import logger

from chunkhound.llm_manager import LLMManager
from chunkhound.services import prompts
from chunkhound.services.research.shared.models import (
    NUM_LLM_EXPANDED_QUERIES,
    QUERY_EXPANSION_TOKENS,
    ResearchContext,
)


class QueryExpander:
    """Handles query expansion and transformation for semantic search.

    This class provides two key functionalities:
    1. Building search queries that optimize for embedding model position bias
    2. Expanding queries using LLM to generate semantically diverse variations

    The query building strategy places the current query first, followed by
    minimal parent context, as embedding models weight the beginning 15-50%
    more heavily.

    Query expansion uses LLM to generate alternative phrasings and perspectives,
    improving recall by casting a wider semantic net.
    """

    def __init__(self, llm_manager: LLMManager):
        """Initialize query expander.

        Args:
            llm_manager: LLM manager for query expansion operations
        """
        self._llm_manager = llm_manager

    def build_search_query(self, query: str, context: ResearchContext) -> str:
        """Build search query combining input with BFS context.

        Evidence-based design (per research on embedding model position bias):
        - Current query FIRST (embedding models weight beginning 15-50% more heavily)
        - Minimal parent context (last 1-2 ancestors for disambiguation)
        - Clear separator to distinguish query from context
        - Root query implicitly preserved through ancestor chain

        Args:
            query: Current query
            context: Research context with ancestors

        Returns:
            Combined search query optimized for semantic search
        """
        if not context.ancestors:
            # Root node: just the query itself
            return query

        # For child nodes: prioritize current query, add minimal parent context
        # Take last 1-2 ancestors (not more to avoid redundancy)
        parent_context = (
            context.ancestors[-2:]
            if len(context.ancestors) >= 2
            else context.ancestors[-1:]
        )
        context_str = " → ".join(parent_context)

        # Current query FIRST (position bias optimization), then context
        return f"{query} | Context: {context_str}"

    async def expand_query_with_llm(
        self, query: str, context: ResearchContext
    ) -> list[str]:
        """Expand query into multiple diverse semantic search queries.

        Uses LLM to generate different perspectives on the same question,
        improving recall by casting a wider semantic net.

        Args:
            query: Current query to expand
            context: Research context with root query and ancestors

        Returns:
            List of expanded queries (defaults to [query] if expansion fails)
        """
        llm = self._llm_manager.get_utility_provider()

        # Define JSON schema for structured output
        schema = {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": f"Array of exactly {NUM_LLM_EXPANDED_QUERIES} expanded search queries (semantically complete sentences)",
                }
            },
            "required": ["queries"],
            "additionalProperties": False,
        }

        # Simplified system prompt per GPT-5-Nano best practices
        system = prompts.QUERY_EXPANSION_SYSTEM

        # Build context string
        context_str = ""
        if context.ancestors:
            ancestor_path = " → ".join(context.ancestors[-2:])
            context_str = f"\nPrior: {ancestor_path}"

        # Optimized prompt for semantic diversity
        prompt = prompts.QUERY_EXPANSION_USER.format(
            query=query,
            context_root_query=context.root_query,
            context_str=context_str,
            num_queries=NUM_LLM_EXPANDED_QUERIES,
        )

        logger.debug(
            f"Query expansion budget: {QUERY_EXPANSION_TOKENS:,} tokens (model: {llm.model})"
        )

        try:
            result = await llm.complete_structured(
                prompt=prompt,
                json_schema=schema,
                system=system,
                max_completion_tokens=QUERY_EXPANSION_TOKENS,
            )

            expanded = result.get("queries", [])

            # Validation: expect exactly 2 queries from LLM
            if not expanded or len(expanded) < NUM_LLM_EXPANDED_QUERIES:
                logger.warning(
                    f"LLM returned {len(expanded) if expanded else 0} queries, expected {NUM_LLM_EXPANDED_QUERIES}, using original query only"
                )
                return [query]

            # Filter empty strings
            expanded = [q.strip() for q in expanded if q and q.strip()]

            # PREPEND ORIGINAL QUERY (new logic)
            # Original query goes first for position bias in embedding models
            final_queries = [query] + expanded[:NUM_LLM_EXPANDED_QUERIES]

            logger.debug(
                f"Expanded query into {len(final_queries)} variations: {final_queries}"
            )
            return final_queries

        except Exception as e:
            logger.warning(f"Query expansion failed: {e}, using original query only")
            return [query]
