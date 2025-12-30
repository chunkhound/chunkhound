"""Deep Research Service for ChunkHound - Backwards compatibility wrapper.

This module provides backwards compatibility for the DeepResearchService class.
The actual implementation has been moved to chunkhound.services.research.v1.pluggable_research_service.

Use PluggableResearchService directly from chunkhound.services.research.v1 for new code.
"""

from chunkhound.services.research.v1.pluggable_research_service import (
    PluggableResearchService,
)

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
