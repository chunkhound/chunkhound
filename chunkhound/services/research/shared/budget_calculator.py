"""Token budget calculation for deep research synthesis.

This module handles adaptive token budget allocation based on repository size and
node position in the research tree. Budgets scale dynamically to balance quality
with cost across different codebase sizes and research depths.

Architecture:
- Synthesis budgets: Scale INPUT tokens based on repo size (30k-150k)
- Adaptive budgets: Scale both INPUT and OUTPUT tokens based on tree depth
- Reasoning model support: Fixed 30k output budget for thinking + generation

Usage:
    calculator = BudgetCalculator()

    # For synthesis (single-pass research)
    budgets = calculator.calculate_synthesis_budgets(repo_stats)

    # For adaptive BFS research
    budgets = calculator.get_adaptive_token_budgets(depth=2, max_depth=5, is_leaf=True)
"""

from typing import Any

from loguru import logger

# Repository size thresholds
CHUNKS_TO_LOC_ESTIMATE = 20  # Rough estimation: 1 chunk ≈ 20 lines of code
LOC_THRESHOLD_TINY = 10_000  # Very small repos
LOC_THRESHOLD_SMALL = 100_000  # Small repos
LOC_THRESHOLD_MEDIUM = 1_000_000  # Medium repos
# Large repos: >= 1M LOC

# Synthesis input token budgets (scale with repo size)
SYNTHESIS_INPUT_TOKENS_TINY = 30_000  # Very small repos (< 10k LOC)
SYNTHESIS_INPUT_TOKENS_SMALL = 50_000  # Small repos (< 100k LOC)
SYNTHESIS_INPUT_TOKENS_MEDIUM = 80_000  # Medium repos (< 1M LOC)
SYNTHESIS_INPUT_TOKENS_LARGE = 150_000  # Large repos (>= 1M LOC)

# Fixed output and overhead budgets
OUTPUT_TOKENS_WITH_REASONING = 30_000  # Fixed output budget for reasoning models (18k output + 12k reasoning buffer)
SINGLE_PASS_OVERHEAD_TOKENS = 5_000  # Prompt template and overhead

# Adaptive budget control
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

# Legacy constants (used when ENABLE_ADAPTIVE_BUDGETS = False)
MAX_FILE_CONTENT_TOKENS = 3000
MAX_LLM_INPUT_TOKENS = 5000
MAX_LEAF_ANSWER_TOKENS = 400
MAX_SYNTHESIS_TOKENS = 600


class BudgetCalculator:
    """Calculate token budgets for deep research operations.

    This class provides methods to calculate adaptive token budgets based on:
    - Repository size (for synthesis operations)
    - Tree depth and node position (for BFS research)

    All calculations are stateless and can be called independently.
    """

    def calculate_synthesis_budgets(self, repo_stats: dict[str, Any]) -> dict[str, int]:
        """Calculate synthesis token budgets based on repository size.

        Output budget is FIXED at 30k tokens for reasoning models (includes thinking + output).
        Only INPUT budget scales with repo size from small repos (~65k total) to large repos
        (~185k total) using piecewise linear brackets with diminishing returns.

        Args:
            repo_stats: Repository statistics from get_stats() including chunk count

        Returns:
            Dictionary with input_tokens, output_tokens, overhead_tokens, total_tokens
        """
        total_chunks = repo_stats.get("chunks", 0)

        # Estimate LOC from chunk count
        estimated_loc = total_chunks * CHUNKS_TO_LOC_ESTIMATE

        # Scale INPUT budget based on repository size (piecewise linear brackets with diminishing returns)
        if estimated_loc < LOC_THRESHOLD_TINY:
            # Very small repos: minimal input context
            input_tokens = SYNTHESIS_INPUT_TOKENS_TINY
        elif estimated_loc < LOC_THRESHOLD_SMALL:
            # Small repos: moderate input context
            input_tokens = SYNTHESIS_INPUT_TOKENS_SMALL
        elif estimated_loc < LOC_THRESHOLD_MEDIUM:
            # Medium repos: standard input context
            input_tokens = SYNTHESIS_INPUT_TOKENS_MEDIUM
        else:
            # Large repos (>= 1M LOC): maximum input context
            input_tokens = SYNTHESIS_INPUT_TOKENS_LARGE

        overhead_tokens = SINGLE_PASS_OVERHEAD_TOKENS
        total_tokens = input_tokens + OUTPUT_TOKENS_WITH_REASONING + overhead_tokens

        logger.debug(
            f"Synthesis budgets for ~{estimated_loc:,} LOC: "
            f"input={input_tokens:,}, output={OUTPUT_TOKENS_WITH_REASONING:,}, total={total_tokens:,}"
        )

        return {
            "input_tokens": input_tokens,
            "output_tokens": OUTPUT_TOKENS_WITH_REASONING,
            "overhead_tokens": overhead_tokens,
            "total_tokens": total_tokens,
        }

    def get_adaptive_token_budgets(
        self, depth: int, max_depth: int, is_leaf: bool
    ) -> dict[str, int]:
        """Calculate adaptive token budgets based on node depth and tree position.

        Strategy (LLM×MapReduce Pyramid):
        - Leaves: Dense implementation details (10-12k tokens) - focused analysis
        - Internal nodes: Progressive compression toward root
        - Root: Concise synthesis (5-8k tokens target) - practical overview

        The deeper the node during expansion, the more detail needed.
        As we collapse upward during synthesis, we compress while maintaining quality.

        Args:
            depth: Current node depth (0 = root)
            max_depth: Maximum depth for this codebase (3-7 typically)
            is_leaf: Whether this is a leaf node

        Returns:
            Dictionary with adaptive token budgets for this node
        """
        if not ENABLE_ADAPTIVE_BUDGETS:
            # Fallback to legacy fixed budgets
            return {
                "file_content_tokens": MAX_FILE_CONTENT_TOKENS,
                "llm_input_tokens": MAX_LLM_INPUT_TOKENS,
                "answer_tokens": MAX_LEAF_ANSWER_TOKENS
                if is_leaf
                else MAX_SYNTHESIS_TOKENS,
            }

        # Normalize depth: 0.0 at root, 1.0 at max_depth
        depth_ratio = depth / max(max_depth, 1)

        # INPUT BUDGETS (what LLM sees - file content and total input)
        # ==============================================================

        # File content budget: Scales with depth (10k → 50k tokens)
        # Root needs LESS raw code (synthesizing), leaves need MORE (analyzing)
        file_content_tokens = int(
            FILE_CONTENT_TOKENS_MIN
            + (FILE_CONTENT_TOKENS_MAX - FILE_CONTENT_TOKENS_MIN) * depth_ratio
        )

        # LLM total input budget (query + context + code): 15k → 60k tokens
        llm_input_tokens = int(
            LLM_INPUT_TOKENS_MIN
            + (LLM_INPUT_TOKENS_MAX - LLM_INPUT_TOKENS_MIN) * depth_ratio
        )

        # OUTPUT BUDGETS (what LLM generates)
        # ====================================

        if is_leaf:
            # LEAVES: Dense, focused detail (10-12k tokens)
            # Scale slightly with depth to handle variable max_depth (3-7)
            answer_tokens = int(
                LEAF_ANSWER_TOKENS_BASE + LEAF_ANSWER_TOKENS_BONUS * depth_ratio
            )
        else:
            # INTERNAL NODES: Compress as we go UP the tree
            # Root (depth 0) gets concise output (5k)
            # Deeper internal nodes get more budget before compressing
            answer_tokens = int(
                INTERNAL_ROOT_TARGET
                + (INTERNAL_MAX_TOKENS - INTERNAL_ROOT_TARGET) * depth_ratio
            )

        # Follow-up question generation budget: Scales with depth (3k → 8k)
        # Deeper nodes have more context to analyze, need more output tokens
        followup_output_tokens = int(
            FOLLOWUP_OUTPUT_TOKENS_MIN
            + (FOLLOWUP_OUTPUT_TOKENS_MAX - FOLLOWUP_OUTPUT_TOKENS_MIN) * depth_ratio
        )

        logger.debug(
            f"Adaptive budgets for depth {depth}/{max_depth} ({'leaf' if is_leaf else 'internal'}): "
            f"file={file_content_tokens:,}, input={llm_input_tokens:,}, output={answer_tokens:,}, "
            f"followup={followup_output_tokens:,}"
        )

        return {
            "file_content_tokens": file_content_tokens,
            "llm_input_tokens": llm_input_tokens,
            "answer_tokens": answer_tokens,
            "followup_output_tokens": followup_output_tokens,
        }
