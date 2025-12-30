"""Shared reusable components for research services.

This module contains components used across multiple research implementations
(v1 BFS and future versions). Components that are v1-specific remain in the
parent directory.
"""

# Core models
# Services
from chunkhound.services.research.shared.budget_calculator import BudgetCalculator
from chunkhound.services.research.shared.chunk_context_builder import (
    ChunkContextBuilder,
)
from chunkhound.services.research.shared.chunk_dedup import (
    deduplicate_chunks,
    get_chunk_id,
    merge_chunk_lists,
)
from chunkhound.services.research.shared.citation_manager import CitationManager
from chunkhound.services.research.shared.context_manager import ContextManager
from chunkhound.services.research.shared.depth_exploration import (
    DepthExplorationService,
)
from chunkhound.services.research.shared.elbow_detection import find_elbow_kneedle
from chunkhound.services.research.shared.evidence_ledger import (
    CONSTANTS_INSTRUCTION_FULL,
    CONSTANTS_INSTRUCTION_SHORT,
    FACT_EXTRACTION_SYSTEM,
    FACT_EXTRACTION_USER,
    FACTS_MAP_INSTRUCTION,
    FACTS_REDUCE_INSTRUCTION,
    ConfidenceLevel,
    ConstantEntry,
    EntityLink,
    EvidenceLedger,
    EvidenceType,
    FactConflict,
    FactEntry,
    FactExtractor,
)
from chunkhound.services.research.shared.file_reader import FileReader
from chunkhound.services.research.shared.gap_detection import GapDetectionService
from chunkhound.services.research.shared.gap_models import GapCandidate, UnifiedGap
from chunkhound.services.research.shared.import_context import ImportContextService
from chunkhound.services.research.shared.import_resolution_helper import (
    resolve_and_fetch_imports,
)
from chunkhound.services.research.shared.import_resolver import ImportResolverService
from chunkhound.services.research.shared.models import (
    _CITATION_PATTERN,
    _CITATION_SEQUENCE_PATTERN,
    CLUSTER_OUTPUT_TOKEN_BUDGET,
    ENABLE_ADAPTIVE_BUDGETS,
    ENABLE_SMART_BOUNDARIES,
    EXTRA_CONTEXT_TOKENS,
    FACT_EXTRACTION_TOKENS,
    FACTS_LEDGER_MAX_ENTRIES,
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
    MAX_BOUNDARY_EXPANSION_LINES,
    MAX_CHUNKS_PER_FILE_REPR,
    MAX_FACTS_PER_CLUSTER,
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
    TARGET_OUTPUT_TOKENS,
    TOKEN_BUDGET_PER_FILE,
    BFSNode,
    ResearchContext,
    build_output_guidance,
)
from chunkhound.services.research.shared.query_expander import QueryExpander
from chunkhound.services.research.shared.unified_search import UnifiedSearch

__all__ = [
    # Models
    "BFSNode",
    "GapCandidate",
    "ResearchContext",
    "UnifiedGap",
    # Services
    "BudgetCalculator",
    "ChunkContextBuilder",
    "CitationManager",
    "ContextManager",
    "DepthExplorationService",
    "FileReader",
    "GapDetectionService",
    "ImportContextService",
    "ImportResolverService",
    "QueryExpander",
    "UnifiedSearch",
    # Utilities
    "build_output_guidance",
    "find_elbow_kneedle",
    "resolve_and_fetch_imports",
    # Chunk deduplication
    "get_chunk_id",
    "deduplicate_chunks",
    "merge_chunk_lists",
    # Evidence ledger (unified constants + facts)
    "ConfidenceLevel",
    "ConstantEntry",
    "EntityLink",
    "EvidenceLedger",
    "EvidenceType",
    "FactConflict",
    "FactEntry",
    "FactExtractor",
    "CONSTANTS_INSTRUCTION_FULL",
    "CONSTANTS_INSTRUCTION_SHORT",
    "FACT_EXTRACTION_SYSTEM",
    "FACT_EXTRACTION_USER",
    "FACTS_MAP_INSTRUCTION",
    "FACTS_REDUCE_INSTRUCTION",
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
    # NOTE: Repository sizing constants (CHUNKS_TO_LOC_ESTIMATE, LOC_THRESHOLD_*, SYNTHESIS_INPUT_TOKENS_*)
    # have been removed. Elbow detection now determines relevance cutoffs based on score distributions.
    # Constants - Citations
    "REQUIRE_CITATIONS",
    "_CITATION_PATTERN",
    "_CITATION_SEQUENCE_PATTERN",
    # Constants - Map-reduce
    "MAX_TOKENS_PER_CLUSTER",
    "CLUSTER_OUTPUT_TOKEN_BUDGET",
    # Constants - Fact extraction
    "FACT_EXTRACTION_TOKENS",
    "MAX_FACTS_PER_CLUSTER",
    "FACTS_LEDGER_MAX_ENTRIES",
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
