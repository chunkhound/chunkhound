"""Research module - Deep research functionality split into focused components.

Backwards compatibility layer: Re-exports shared components and v1 implementation.
"""

# Re-export factory and protocol (primary interface)
from chunkhound.services.research.factory import ResearchServiceFactory
from chunkhound.services.research.protocol import (
    ResearchResult,
    ResearchServiceProtocol,
)

# Re-export shared components for backwards compatibility
from chunkhound.services.research.shared import (
    BFSNode,
    BudgetCalculator,
    ContextManager,
    QueryExpander,
    ResearchContext,
)

# Re-export constants from shared.models for backwards compatibility
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

# Re-export v1 implementation for backwards compatibility
from chunkhound.services.research.v1 import (
    BFSResearchService,
    QualityValidator,
    QuestionGenerator,
    SynthesisEngine,
)

# Re-export v2 implementation
from chunkhound.services.research.v2 import (
    CoverageResearchService,
    CoverageSynthesisEngine,
    GapCandidate,
    GapDetectionService,
    UnifiedGap,
)

__all__ = [
    # Factory and protocol (primary interface)
    "ResearchServiceFactory",
    "ResearchServiceProtocol",
    "ResearchResult",
    # Shared components
    "BudgetCalculator",
    "ContextManager",
    "BFSNode",
    "ResearchContext",
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
    # v1 implementation
    "BFSResearchService",
    "QualityValidator",
    "QuestionGenerator",
    "QueryExpander",
    "SynthesisEngine",
    # v2 implementation
    "CoverageResearchService",
    "GapDetectionService",
    "CoverageSynthesisEngine",
    "GapCandidate",
    "UnifiedGap",
]
