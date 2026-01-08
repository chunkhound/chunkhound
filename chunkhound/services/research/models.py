"""Backwards compatibility shim for research models.

This module re-exports all components from shared.models to maintain backwards
compatibility with code that imports from chunkhound.services.research.models.

New code should import directly from chunkhound.services.research.shared.models.
"""

from chunkhound.services.research.shared.models import (
    ENABLE_ADAPTIVE_BUDGETS,
    EXTRA_CONTEXT_TOKENS,
    FILE_CONTENT_TOKENS_MAX,
    FILE_CONTENT_TOKENS_MIN,
    FOLLOWUP_OUTPUT_TOKENS_MAX,
    FOLLOWUP_OUTPUT_TOKENS_MIN,
    LEAF_ANSWER_TOKENS_BASE,
    LEAF_ANSWER_TOKENS_BONUS,
    LLM_INPUT_TOKENS_MAX,
    LLM_INPUT_TOKENS_MIN,
    MAX_FOLLOWUP_QUESTIONS,
    MAX_LEAF_ANSWER_TOKENS,
    MAX_LLM_INPUT_TOKENS,
    MAX_SYMBOLS_TO_SEARCH,
    MAX_SYNTHESIS_TOKENS,
    NODE_SIMILARITY_THRESHOLD,
    NUM_LLM_EXPANDED_QUERIES,
    QUERY_EXPANSION_ENABLED,
    RELEVANCE_THRESHOLD,
    TOKEN_BUDGET_PER_FILE,
    BFSNode,
    ResearchContext,
)

__all__ = [
    # Data models
    "BFSNode",
    "ResearchContext",
    # Constants
    "RELEVANCE_THRESHOLD",
    "NODE_SIMILARITY_THRESHOLD",
    "MAX_FOLLOWUP_QUESTIONS",
    "MAX_SYMBOLS_TO_SEARCH",
    "QUERY_EXPANSION_ENABLED",
    "NUM_LLM_EXPANDED_QUERIES",
    "ENABLE_ADAPTIVE_BUDGETS",
    "FILE_CONTENT_TOKENS_MIN",
    "FILE_CONTENT_TOKENS_MAX",
    "LLM_INPUT_TOKENS_MIN",
    "LLM_INPUT_TOKENS_MAX",
    "LEAF_ANSWER_TOKENS_BASE",
    "LEAF_ANSWER_TOKENS_BONUS",
    "FOLLOWUP_OUTPUT_TOKENS_MIN",
    "FOLLOWUP_OUTPUT_TOKENS_MAX",
    "TOKEN_BUDGET_PER_FILE",
    "EXTRA_CONTEXT_TOKENS",
    "MAX_LEAF_ANSWER_TOKENS",
    "MAX_LLM_INPUT_TOKENS",
    "MAX_SYNTHESIS_TOKENS",
]
