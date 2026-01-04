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

# Synthesis budget calculation (repository size scaling)
CHUNKS_TO_LOC_ESTIMATE = 20  # Rough estimation: 1 chunk ≈ 20 lines of code
LOC_THRESHOLD_TINY = 10_000  # Very small repos
LOC_THRESHOLD_SMALL = 100_000  # Small repos
LOC_THRESHOLD_MEDIUM = 1_000_000  # Medium repos
# Large repos: >= 1M LOC

SYNTHESIS_INPUT_TOKENS_TINY = 30_000  # Very small repos (< 10k LOC)
SYNTHESIS_INPUT_TOKENS_SMALL = 50_000  # Small repos (< 100k LOC)
SYNTHESIS_INPUT_TOKENS_MEDIUM = 80_000  # Medium repos (< 1M LOC)
SYNTHESIS_INPUT_TOKENS_LARGE = 150_000  # Large repos (>= 1M LOC)

# Output control
REQUIRE_CITATIONS = True  # Validate file:line format

# Map-reduce synthesis constants
MAX_TOKENS_PER_CLUSTER = 30_000  # Token budget per cluster for parallel synthesis
CLUSTER_OUTPUT_TOKEN_BUDGET = 15_000  # Max output tokens per cluster summary

# Pre-compiled regex patterns for citation processing
_CITATION_PATTERN = re.compile(r"\[\d+\]")  # Matches [N] citations
_CITATION_SEQUENCE_PATTERN = re.compile(
    r"(?:\[\d+\])+"
)  # Matches sequences like [1][2][3]

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
