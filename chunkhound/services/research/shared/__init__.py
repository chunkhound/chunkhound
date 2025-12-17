"""Shared reusable components for research services.

This module contains components used across multiple research implementations
(v1 BFS and future versions). Components that are v1-specific remain in the
parent directory.
"""

# Core models
# Services
from chunkhound.services.research.shared.budget_calculator import BudgetCalculator
from chunkhound.services.research.shared.chunk_dedup import (
    deduplicate_chunks,
    get_chunk_id,
    merge_chunk_lists,
)
from chunkhound.services.research.shared.citation_manager import CitationManager
from chunkhound.services.research.shared.context_manager import ContextManager
from chunkhound.services.research.shared.elbow_detection import find_elbow_kneedle
from chunkhound.services.research.shared.file_reader import FileReader
from chunkhound.services.research.shared.import_context import ImportContextService
from chunkhound.services.research.shared.import_resolution_helper import (
    resolve_and_fetch_imports,
)
from chunkhound.services.research.shared.import_resolver import ImportResolverService
from chunkhound.services.research.shared.models import (
    _CITATION_PATTERN,
    _CITATION_SEQUENCE_PATTERN,
    CHUNKS_TO_LOC_ESTIMATE,
    CLUSTER_OUTPUT_TOKEN_BUDGET,
    ENABLE_ADAPTIVE_BUDGETS,
    ENABLE_SMART_BOUNDARIES,
    EXTRA_CONTEXT_TOKENS,
    FILE_CONTENT_TOKENS_MAX,
    FILE_CONTENT_TOKENS_MIN,
    FOLLOWUP_OUTPUT_TOKENS_MAX,
    FOLLOWUP_OUTPUT_TOKENS_MIN,
    IMPORT_DEFAULT_SCORE,
    IMPORT_SYNTHESIS_SCORE,
    INTERNAL_MAX_TOKENS,
    INTERNAL_ROOT_TARGET,
    LEAF_ANSWER_TOKENS_BASE,
    LEAF_ANSWER_TOKENS_BONUS,
    LLM_INPUT_TOKENS_MAX,
    LLM_INPUT_TOKENS_MIN,
    LOC_THRESHOLD_MEDIUM,
    LOC_THRESHOLD_SMALL,
    LOC_THRESHOLD_TINY,
    MAX_BOUNDARY_EXPANSION_LINES,
    MAX_CHUNKS_PER_FILE_REPR,
    MAX_FILE_CONTENT_TOKENS,
    MAX_FOLLOWUP_QUESTIONS,
    MAX_LEAF_ANSWER_TOKENS,
    MAX_LLM_INPUT_TOKENS,
    MAX_SYMBOLS_TO_SEARCH,
    MAX_SYNTHESIS_TOKENS,
    MAX_TOKENS_PER_CLUSTER,
    MAX_TOKENS_PER_FILE_REPR,
    NODE_SIMILARITY_THRESHOLD,
    NUM_LLM_EXPANDED_QUERIES,
    OUTPUT_TOKENS_WITH_REASONING,
    QUERY_EXPANSION_ENABLED,
    QUERY_EXPANSION_TOKENS,
    QUESTION_FILTERING_TOKENS,
    QUESTION_SYNTHESIS_TOKENS,
    RELEVANCE_THRESHOLD,
    REQUIRE_CITATIONS,
    SINGLE_PASS_MAX_TOKENS,
    SINGLE_PASS_OVERHEAD_TOKENS,
    SINGLE_PASS_TIMEOUT_SECONDS,
    SYNTHESIS_INPUT_TOKENS_LARGE,
    SYNTHESIS_INPUT_TOKENS_MEDIUM,
    SYNTHESIS_INPUT_TOKENS_SMALL,
    SYNTHESIS_INPUT_TOKENS_TINY,
    TARGET_OUTPUT_TOKENS,
    TOKEN_BUDGET_PER_FILE,
    BFSNode,
    ResearchContext,
)
from chunkhound.services.research.shared.query_expander import QueryExpander
from chunkhound.services.research.shared.unified_search import UnifiedSearch

__all__ = [
    # Models
    "BFSNode",
    "ResearchContext",
    # Services
    "BudgetCalculator",
    "CitationManager",
    "ContextManager",
    "FileReader",
    "ImportContextService",
    "ImportResolverService",
    "QueryExpander",
    "UnifiedSearch",
    # Utilities
    "find_elbow_kneedle",
    "resolve_and_fetch_imports",
    # Chunk deduplication
    "get_chunk_id",
    "deduplicate_chunks",
    "merge_chunk_lists",
    # Constants - Search
    "RELEVANCE_THRESHOLD",
    "NODE_SIMILARITY_THRESHOLD",
    "MAX_FOLLOWUP_QUESTIONS",
    "MAX_SYMBOLS_TO_SEARCH",
    "QUERY_EXPANSION_ENABLED",
    "NUM_LLM_EXPANDED_QUERIES",
    # Constants - Adaptive budgets
    "ENABLE_ADAPTIVE_BUDGETS",
    "FILE_CONTENT_TOKENS_MIN",
    "FILE_CONTENT_TOKENS_MAX",
    "LLM_INPUT_TOKENS_MIN",
    "LLM_INPUT_TOKENS_MAX",
    "LEAF_ANSWER_TOKENS_BASE",
    "LEAF_ANSWER_TOKENS_BONUS",
    "INTERNAL_ROOT_TARGET",
    "INTERNAL_MAX_TOKENS",
    "FOLLOWUP_OUTPUT_TOKENS_MIN",
    "FOLLOWUP_OUTPUT_TOKENS_MAX",
    "QUERY_EXPANSION_TOKENS",
    "QUESTION_SYNTHESIS_TOKENS",
    "QUESTION_FILTERING_TOKENS",
    # Constants - Legacy
    "TOKEN_BUDGET_PER_FILE",
    "EXTRA_CONTEXT_TOKENS",
    "MAX_FILE_CONTENT_TOKENS",
    "MAX_LLM_INPUT_TOKENS",
    "MAX_LEAF_ANSWER_TOKENS",
    "MAX_SYNTHESIS_TOKENS",
    # Constants - Single-pass synthesis
    "SINGLE_PASS_MAX_TOKENS",
    "OUTPUT_TOKENS_WITH_REASONING",
    "SINGLE_PASS_OVERHEAD_TOKENS",
    "SINGLE_PASS_TIMEOUT_SECONDS",
    "TARGET_OUTPUT_TOKENS",
    # Constants - Repository sizing
    "CHUNKS_TO_LOC_ESTIMATE",
    "LOC_THRESHOLD_TINY",
    "LOC_THRESHOLD_SMALL",
    "LOC_THRESHOLD_MEDIUM",
    "SYNTHESIS_INPUT_TOKENS_TINY",
    "SYNTHESIS_INPUT_TOKENS_SMALL",
    "SYNTHESIS_INPUT_TOKENS_MEDIUM",
    "SYNTHESIS_INPUT_TOKENS_LARGE",
    # Constants - Citations
    "REQUIRE_CITATIONS",
    "_CITATION_PATTERN",
    "_CITATION_SEQUENCE_PATTERN",
    # Constants - Map-reduce
    "MAX_TOKENS_PER_CLUSTER",
    "CLUSTER_OUTPUT_TOKEN_BUDGET",
    # Constants - Smart boundaries
    "ENABLE_SMART_BOUNDARIES",
    "MAX_BOUNDARY_EXPANSION_LINES",
    # Constants - File reranking
    "MAX_CHUNKS_PER_FILE_REPR",
    "MAX_TOKENS_PER_FILE_REPR",
    # Constants - Import resolution
    "IMPORT_DEFAULT_SCORE",
    "IMPORT_SYNTHESIS_SCORE",
]
