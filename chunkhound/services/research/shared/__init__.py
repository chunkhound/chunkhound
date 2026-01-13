"""Shared reusable components for research services.

This module contains components used across multiple research implementations
(v1 BFS and future versions). Components that are v1-specific remain in the
parent directory.

Uses lazy imports to defer import costs until symbols are actually accessed.
"""

import importlib
from typing import Any

# Mapping of symbol names to (module_path, attribute_name)
# When attribute_name is None, it means the symbol name matches the attribute name
_LAZY_IMPORTS: dict[str, tuple[str, str | None]] = {
    # Services
    "BudgetCalculator": (
        "chunkhound.services.research.shared.budget_calculator",
        None,
    ),
    "ChunkContextBuilder": (
        "chunkhound.services.research.shared.chunk_context_builder",
        None,
    ),
    "CitationManager": (
        "chunkhound.services.research.shared.citation_manager",
        None,
    ),
    "ContextManager": (
        "chunkhound.services.research.shared.context_manager",
        None,
    ),
    "DepthExplorationService": (
        "chunkhound.services.research.shared.depth_exploration",
        None,
    ),
    "FileReader": (
        "chunkhound.services.research.shared.file_reader",
        None,
    ),
    "GapDetectionService": (
        "chunkhound.services.research.shared.gap_detection",
        None,
    ),
    "ImportContextService": (
        "chunkhound.services.research.shared.import_context",
        None,
    ),
    "ImportResolverService": (
        "chunkhound.services.research.shared.import_resolver",
        None,
    ),
    "QueryExpander": (
        "chunkhound.services.research.shared.query_expander",
        None,
    ),
    "UnifiedSearch": (
        "chunkhound.services.research.shared.unified_search",
        None,
    ),
    # Chunk deduplication
    "deduplicate_chunks": (
        "chunkhound.services.research.shared.chunk_dedup",
        None,
    ),
    "get_chunk_id": (
        "chunkhound.services.research.shared.chunk_dedup",
        None,
    ),
    "merge_chunk_lists": (
        "chunkhound.services.research.shared.chunk_dedup",
        None,
    ),
    # Elbow detection
    "find_elbow_kneedle": (
        "chunkhound.services.research.shared.elbow_detection",
        None,
    ),
    # Evidence ledger
    "CONSTANTS_INSTRUCTION_FULL": (
        "chunkhound.services.research.shared.evidence_ledger",
        None,
    ),
    "CONSTANTS_INSTRUCTION_SHORT": (
        "chunkhound.services.research.shared.evidence_ledger",
        None,
    ),
    "FACT_EXTRACTION_SYSTEM": (
        "chunkhound.services.research.shared.evidence_ledger",
        None,
    ),
    "FACT_EXTRACTION_USER": (
        "chunkhound.services.research.shared.evidence_ledger",
        None,
    ),
    "FACTS_MAP_INSTRUCTION": (
        "chunkhound.services.research.shared.evidence_ledger",
        None,
    ),
    "FACTS_REDUCE_INSTRUCTION": (
        "chunkhound.services.research.shared.evidence_ledger",
        None,
    ),
    "ConfidenceLevel": (
        "chunkhound.services.research.shared.evidence_ledger",
        None,
    ),
    "ConstantEntry": (
        "chunkhound.services.research.shared.evidence_ledger",
        None,
    ),
    "EntityLink": (
        "chunkhound.services.research.shared.evidence_ledger",
        None,
    ),
    "EvidenceLedger": (
        "chunkhound.services.research.shared.evidence_ledger",
        None,
    ),
    "EvidenceType": (
        "chunkhound.services.research.shared.evidence_ledger",
        None,
    ),
    "FactConflict": (
        "chunkhound.services.research.shared.evidence_ledger",
        None,
    ),
    "FactEntry": (
        "chunkhound.services.research.shared.evidence_ledger",
        None,
    ),
    "FactExtractor": (
        "chunkhound.services.research.shared.evidence_ledger",
        None,
    ),
    # Gap models
    "GapCandidate": (
        "chunkhound.services.research.shared.gap_models",
        None,
    ),
    "UnifiedGap": (
        "chunkhound.services.research.shared.gap_models",
        None,
    ),
    # Import resolution helper
    "resolve_and_fetch_imports": (
        "chunkhound.services.research.shared.import_resolution_helper",
        None,
    ),
    # Models - all symbols from the models module
    "_CITATION_PATTERN": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "_CITATION_SEQUENCE_PATTERN": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "CLUSTER_OUTPUT_TOKEN_BUDGET": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "ENABLE_ADAPTIVE_BUDGETS": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "ENABLE_SMART_BOUNDARIES": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "EXTRA_CONTEXT_TOKENS": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "FACT_EXTRACTION_TOKENS": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "FACTS_LEDGER_MAX_ENTRIES": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "FILE_CONTENT_TOKENS_MAX": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "FILE_CONTENT_TOKENS_MIN": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "FOLLOWUP_OUTPUT_TOKENS_MAX": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "FOLLOWUP_OUTPUT_TOKENS_MIN": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "IMPORT_DEFAULT_SCORE": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "IMPORT_SYNTHESIS_SCORE": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "INTERNAL_MAX_TOKENS": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "INTERNAL_ROOT_TARGET": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "LEAF_ANSWER_TOKENS_BASE": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "LEAF_ANSWER_TOKENS_BONUS": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "LLM_INPUT_TOKENS_MAX": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "LLM_INPUT_TOKENS_MIN": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "MAX_BOUNDARY_EXPANSION_LINES": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "MAX_CHUNKS_PER_FILE_REPR": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "MAX_FACTS_PER_CLUSTER": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "MAX_FILE_CONTENT_TOKENS": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "MAX_FOLLOWUP_QUESTIONS": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "MAX_LEAF_ANSWER_TOKENS": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "MAX_LLM_INPUT_TOKENS": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "MAX_SYMBOLS_TO_SEARCH": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "MAX_SYNTHESIS_TOKENS": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "MAX_TOKENS_PER_CLUSTER": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "MAX_TOKENS_PER_FILE_REPR": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "NODE_SIMILARITY_THRESHOLD": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "NUM_LLM_EXPANDED_QUERIES": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "OUTPUT_TOKENS_WITH_REASONING": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "QUERY_EXPANSION_ENABLED": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "QUERY_EXPANSION_TOKENS": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "QUESTION_FILTERING_TOKENS": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "QUESTION_SYNTHESIS_TOKENS": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "RELEVANCE_THRESHOLD": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "REQUIRE_CITATIONS": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "SINGLE_PASS_MAX_TOKENS": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "SINGLE_PASS_OVERHEAD_TOKENS": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "SINGLE_PASS_TIMEOUT_SECONDS": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "TARGET_OUTPUT_TOKENS": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "TOKEN_BUDGET_PER_FILE": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "BFSNode": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "ResearchContext": (
        "chunkhound.services.research.shared.models",
        None,
    ),
    "build_output_guidance": (
        "chunkhound.services.research.shared.models",
        None,
    ),
}


def __getattr__(name: str) -> Any:
    """Lazy import handler for module attributes."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        attr = getattr(module, attr_name if attr_name else name)
        # Cache in globals for future access
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return list of available attributes for IDE support."""
    return list(__all__)


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
    # NOTE: Repository sizing constants (CHUNKS_TO_LOC_ESTIMATE, LOC_THRESHOLD_*,
    # SYNTHESIS_INPUT_TOKENS_*) have been removed. Elbow detection now determines
    # relevance cutoffs based on score distributions.
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
