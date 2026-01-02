"""Data models and constants for deep research service.

This module contains shared data structures and configuration constants
used by the deep research service for BFS-based semantic exploration.
"""

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

# Constants
RELEVANCE_THRESHOLD = 0.5  # Lower threshold for better recall, reranking will filter
NODE_SIMILARITY_THRESHOLD = (
    0.2  # Reserved for future similarity-based deduplication (currently uses LLM)
)
MAX_FOLLOWUP_QUESTIONS = 3
MAX_SYMBOLS_TO_SEARCH = 5  # Top N symbols to search via regex (from spec)
QUERY_EXPANSION_ENABLED = True  # Enable LLM-powered query expansion for better recall
NUM_LLM_EXPANDED_QUERIES = 2  # LLM generates 2 queries, we prepend original = 3 total

# Regex augmentation strategy (industry standard: 30% keyword + 70% semantic)
REGEX_AUGMENTATION_RATIO = 0.3  # Regex results = 30% of semantic count
REGEX_MIN_RESULTS = 20  # Min regex results (ensures value when semantic low)

# Import resolution default scores
# Import-resolved chunks don't have rerank scores, so we assign defaults.
# Lower values = more conservative ranking (won't outrank actual search results)
IMPORT_DEFAULT_SCORE = 0.3  # Phases 1, 1.5, 2: discovered early, higher priority
IMPORT_SYNTHESIS_SCORE = 0.2  # Phase 3: late discovery imports, lower priority

# Adaptive token budgets (depth-dependent)
ENABLE_ADAPTIVE_BUDGETS = True  # Enable depth-based adaptive budgets

# File content budget range (input: what LLM sees for code)
FILE_CONTENT_TOKENS_MIN = 10_000  # Root nodes (synthesizing, need less raw code)
FILE_CONTENT_TOKENS_MAX = 50_000  # Leaf nodes (analyzing, need full implementations)

# LLM total input budget range (query + context + code)
LLM_INPUT_TOKENS_MIN = 15_000  # Root nodes
LLM_INPUT_TOKENS_MAX = 60_000  # Leaf nodes

# Leaf answer output budget (what LLM generates at leaves)
# NOTE: Reduced from 30k to balance cost vs quality. If you observe:
#   - Frequent "Missing: [detail]" statements
#   - Theoretical placeholders ("provide exact values")
#   - Incomplete analysis of complex components
# Consider increasing these values. Quality validation warnings will indicate budget pressure.
LEAF_ANSWER_TOKENS_BASE = (
    18_000  # Base budget for leaf nodes (was 30k, reduced for cost)
)
LEAF_ANSWER_TOKENS_BONUS = (
    3_000  # Additional tokens for deeper leaves (was 5k, reduced for cost)
)

# Internal synthesis output budget (what LLM generates at internal nodes)
# NOTE: Reduced from 17.5k/32k to balance cost vs quality. If root synthesis appears rushed or
# omits critical architectural details, consider increasing INTERNAL_ROOT_TARGET.
INTERNAL_ROOT_TARGET = 11_000  # Root synthesis target (was 17.5k, reduced for cost)
INTERNAL_MAX_TOKENS = (
    19_000  # Maximum for deep internal nodes (was 32k, reduced for cost)
)

# Follow-up question generation output budget (what LLM generates for follow-up questions)
# NOTE: High budgets needed for reasoning models (o1/o3/GPT-5) which use internal "thinking" tokens
# WHY: Reasoning models consume 5-15k tokens for internal reasoning before producing 100-500 tokens of output
# The actual generated questions are concise, but the model needs reasoning budget to evaluate relevance
FOLLOWUP_OUTPUT_TOKENS_MIN = (
    8_000  # Root/shallow nodes: simpler questions, less reasoning needed
)
FOLLOWUP_OUTPUT_TOKENS_MAX = (
    15_000  # Deep nodes: complex synthesis requires more reasoning depth
)

# Utility operation output budgets (for reasoning models like o1/o3/GPT-5)
# These operations use utility provider and don't vary by depth
# WHY: Each utility operation produces small output but requires reasoning budget for quality
QUERY_EXPANSION_TOKENS = (
    10_000  # Generate 2 queries (~200 output + ~8k reasoning to ensure diversity)
)
QUESTION_SYNTHESIS_TOKENS = (
    15_000  # Synthesize to 1-3 questions (~500 output + ~12k reasoning for quality)
)
QUESTION_FILTERING_TOKENS = (
    5_000  # Filter by relevance (~50 output + ~4k reasoning for accuracy)
)

# Legacy constants (used when ENABLE_ADAPTIVE_BUDGETS = False)
TOKEN_BUDGET_PER_FILE = 4000
EXTRA_CONTEXT_TOKENS = 1000
MAX_FILE_CONTENT_TOKENS = 3000
MAX_LLM_INPUT_TOKENS = 5000
MAX_LEAF_ANSWER_TOKENS = 400
MAX_SYNTHESIS_TOKENS = 600

# Single-pass synthesis constants (new architecture)
SINGLE_PASS_MAX_TOKENS = (
    150_000  # Total budget for single-pass synthesis (input + output)
)
OUTPUT_TOKENS_WITH_REASONING = 30_000  # Fixed output budget for reasoning models (18k output + 12k reasoning buffer)
SINGLE_PASS_OVERHEAD_TOKENS = 5_000  # Prompt template and overhead
SINGLE_PASS_TIMEOUT_SECONDS = 600  # 10 minutes timeout for large synthesis calls
# Available for code/chunks: Scales dynamically with repo size (30k-150k input tokens)

# Target output length (controlled via prompt instructions, not API token limits)
# WHY: OUTPUT_TOKENS_WITH_REASONING is FIXED at 30k for all queries (reasoning models need this)
# This allows reasoning models to use thinking tokens while producing appropriately sized output
# NOTE: Only INPUT budget scales dynamically based on repository size, output is fixed
TARGET_OUTPUT_TOKENS = 15_000  # Default target for standard research outputs

# NOTE: Elbow Detection Replaces Repo-Size Scaling (2024-12)
# The v1 synthesis engine now uses Kneedle algorithm elbow detection to determine
# the relevance cutoff for chunks, rather than scaling input budgets based on
# repository size (LOC). This provides data-driven cutoffs based on actual
# score distributions rather than arbitrary size-based thresholds.
# See: chunkhound/services/research/shared/elbow_detection.py

# Output control
REQUIRE_CITATIONS = True  # Validate file:line format

# Map-reduce synthesis constants
MAX_TOKENS_PER_CLUSTER = 30_000  # Token budget per cluster for parallel synthesis
CLUSTER_OUTPUT_TOKEN_BUDGET = 15_000  # Fallback/minimum output budget per cluster (elbow detection may override)

# Fact extraction
FACT_EXTRACTION_TOKENS = 8_000  # Output budget per cluster
MAX_FACTS_PER_CLUSTER = 30  # Limit per cluster
FACTS_LEDGER_MAX_ENTRIES = 200  # Max in final ledger
MAX_FACT_STATEMENT_CHARS = 100  # Hard limit enforced at extraction

# Tiered formatting thresholds (fact count)
# Controls progressive compression as fact count increases
FACTS_TIER_VERBOSE = 20  # 0-20: Full verbose format
FACTS_TIER_COMPACT = 50  # 21-50: Compact single-line
FACTS_TIER_INDEXED = 100  # 51-100: Compact with file index
# 101+: Summary by category only

# Pre-compiled regex patterns for citation processing
_CITATION_PATTERN = re.compile(r"\[\d+\]")  # Matches [N] citations
_CITATION_SEQUENCE_PATTERN = re.compile(r"(?:\[\d+\])+")  # Matches sequences like [1][2][3]

# Smart boundary detection for context-aware file reading
ENABLE_SMART_BOUNDARIES = True  # Expand to natural code boundaries (functions/classes)
MAX_BOUNDARY_EXPANSION_LINES = 300  # Maximum lines to expand for complete functions

# File-level reranking for synthesis budget allocation
# Prevents file diversity collapse where deep BFS exploration causes score accumulation in few files
MAX_CHUNKS_PER_FILE_REPR = (
    5  # Top chunks to include in file representative document for reranking
)
MAX_TOKENS_PER_FILE_REPR = 2000  # Token limit for file representative document


@dataclass
class BFSNode:
    """Node in the BFS research graph."""

    query: str
    parent: "BFSNode | None" = None
    depth: int = 0
    children: list["BFSNode"] = field(default_factory=list)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    file_contents: dict[str, str] = field(
        default_factory=dict
    )  # Full file contents for synthesis
    answer: str | None = None
    node_id: int = 0
    unanswered_aspects: list[str] = field(
        default_factory=list
    )  # Questions we couldn't answer
    token_budgets: dict[str, int] = field(
        default_factory=dict
    )  # Adaptive token budgets for this node
    task_id: int | None = None  # Progress task ID for TUI display

    # Termination tracking
    is_terminated_leaf: bool = False  # True if terminated due to no new information
    new_chunk_count: int = 0  # Count of truly new chunks
    duplicate_chunk_count: int = 0  # Count of duplicate chunks


@dataclass
class ResearchContext:
    """Context for research traversal."""

    root_query: str
    ancestors: list[str] = field(default_factory=list)
    traversal_path: list[str] = field(default_factory=list)


def build_output_guidance(target_tokens: int = TARGET_OUTPUT_TOKENS) -> str:
    """Build consistent output guidance for synthesis prompts.

    Args:
        target_tokens: Target output token count (default: TARGET_OUTPUT_TOKENS)

    Returns:
        Output guidance string for LLM prompts
    """
    return f"Target output: ~{target_tokens:,} tokens (includes reasoning)."
