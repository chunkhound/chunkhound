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

import os
from typing import Any

from loguru import logger

from chunkhound.services.research.shared.models import (
    ENABLE_ADAPTIVE_BUDGETS,
    FILE_CONTENT_TOKENS_MAX,
    FILE_CONTENT_TOKENS_MIN,
    FOLLOWUP_OUTPUT_TOKENS_MAX,
    FOLLOWUP_OUTPUT_TOKENS_MIN,
    INTERNAL_MAX_TOKENS,
    INTERNAL_ROOT_TARGET,
    LEAF_ANSWER_TOKENS_BASE,
    LEAF_ANSWER_TOKENS_BONUS,
    LLM_INPUT_TOKENS_MAX,
    LLM_INPUT_TOKENS_MIN,
    MAX_FILE_CONTENT_TOKENS,
    MAX_LEAF_ANSWER_TOKENS,
    MAX_LLM_INPUT_TOKENS,
    MAX_SYNTHESIS_TOKENS,
    OUTPUT_TOKENS_WITH_REASONING,
)

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

# Fixed overhead budget (unique to budget_calculator)
SINGLE_PASS_OVERHEAD_TOKENS = 5_000  # Prompt template and overhead


class BudgetCalculator:
    """Calculate token budgets for deep research operations.

    This class provides methods to calculate adaptive token budgets based on:
    - Repository size (for synthesis operations)
    - Tree depth and node position (for BFS research)

    All calculations are stateless and can be called independently.
    """

    def calculate_synthesis_budgets(self, repo_stats: dict[str, Any]) -> dict[str, int]:
        """Calculate synthesis token budgets based on repository size.

        Output budget is FIXED at 30k tokens for reasoning models (includes
        thinking + output). Only INPUT budget scales with repo size from small
        repos (~65k total) to large repos (~185k total) using piecewise linear
        brackets with diminishing returns.

        Args:
            repo_stats: Repository statistics from get_stats() including chunk count

        Returns:
            Dictionary with input_tokens, output_tokens, overhead_tokens, total_tokens
        """
        total_chunks = repo_stats.get("chunks", 0)

        # Estimate LOC from chunk count
        estimated_loc = total_chunks * CHUNKS_TO_LOC_ESTIMATE

        # Scale INPUT budget based on repository size (piecewise linear)
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

        # Optional hard cap for synthesis input tokens, primarily to keep
        # external LLM transports like codex exec within argv/stdin limits
        # when used in higher-level tools (e.g., code_mapper).
        env_cap = os.getenv("CHUNKHOUND_SYNTHESIS_INPUT_TOKENS_MAX")
        if env_cap:
            try:
                cap_value = int(env_cap)
                if cap_value > 0:
                    input_tokens = min(input_tokens, cap_value)
            except ValueError:
                # Ignore invalid override and keep calculated budget
                pass

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
