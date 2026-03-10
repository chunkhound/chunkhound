"""
Research configuration for ChunkHound code research algorithm.

This module provides a type-safe, validated configuration system for the
coverage-first research algorithm with support for multiple configuration
sources (environment variables, config files, CLI arguments).
"""

import argparse
import os
from typing import Any, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Default values (kept in sync with chunkhound.services.research.shared.models)
# These are duplicated here to avoid circular imports between config and services.
# If you change these values, update the corresponding constants in models.py.
_DEFAULT_RELEVANCE_THRESHOLD = 0.5
_DEFAULT_MAX_SYMBOLS = 5
_DEFAULT_REGEX_AUGMENTATION_RATIO = 0.3
_DEFAULT_REGEX_MIN_RESULTS = 20
_DEFAULT_MAX_BOUNDARY_EXPANSION_LINES = 300
_DEFAULT_MAX_CHUNKS_PER_FILE_REPR = 5
_DEFAULT_MAX_TOKENS_PER_FILE_REPR = 2000
_DEFAULT_QUERY_EXPANSION_ENABLED = True
_DEFAULT_NUM_EXPANDED_QUERIES = 2


class ResearchConfig(BaseSettings):
    """
    Research algorithm configuration for ChunkHound code research.

    Configuration Sources (in order of precedence):
    1. CLI arguments
    2. Environment variables (CHUNKHOUND_RESEARCH_*)
    3. Config files
    4. Default values

    Environment Variables:
        CHUNKHOUND_RESEARCH_ALGORITHM=v2
        CHUNKHOUND_RESEARCH_QUERY_EXPANSION_ENABLED=true
        CHUNKHOUND_RESEARCH_NUM_EXPANDED_QUERIES=2
        CHUNKHOUND_RESEARCH_EXHAUSTIVE_MODE=false
    """

    model_config = SettingsConfigDict(
        env_prefix="CHUNKHOUND_RESEARCH_",
        env_nested_delimiter="__",
        case_sensitive=False,
        validate_default=True,
        extra="ignore",  # Ignore unknown fields for forward compatibility
    )

    # Feature Flag
    algorithm: Literal["v1", "v2", "v3"] = Field(
        default="v3",
        description=(
            "Research algorithm version (v1=BFS exploration, v2=hybrid v1 "
            "synthesis + wide coverage exploration, v3=parallel BFS + wide "
            "coverage). Note: v3 (default) runs parallel exploration and uses "
            "more LLM tokens than v1/v2."
        ),
    )

    # Phase 1: Coverage Parameters
    query_expansion_enabled: bool = Field(
        default=_DEFAULT_QUERY_EXPANSION_ENABLED,
        description="Enable LLM-based query expansion for broader coverage",
    )

    num_expanded_queries: int = Field(
        default=_DEFAULT_NUM_EXPANDED_QUERIES,
        ge=1,
        le=5,
        description="Number of additional queries to generate from root query",
    )

    initial_page_size: int = Field(
        default=30,
        ge=10,
        le=100,
        description="Results per vector query in multi-hop search",
    )

    relevance_threshold: float = Field(
        default=_DEFAULT_RELEVANCE_THRESHOLD,
        ge=0.3,
        le=0.8,
        description="Minimum rerank score for chunk inclusion",
    )

    max_symbols: int = Field(
        default=_DEFAULT_MAX_SYMBOLS,
        ge=1,
        le=20,
        description="Maximum symbols to extract for regex search augmentation",
    )

    regex_augmentation_ratio: float = Field(
        default=_DEFAULT_REGEX_AUGMENTATION_RATIO,
        ge=0.1,
        le=1.0,
        description=(
            "Regex target as fraction of semantic count (industry standard: 0.3)"
        ),
    )

    regex_min_results: int = Field(
        default=_DEFAULT_REGEX_MIN_RESULTS,
        ge=10,
        le=100,
        description="Minimum regex results regardless of augmentation ratio",
    )

    regex_scan_page_size: int = Field(
        default=100,
        ge=50,
        le=200,
        description="Internal pagination batch size for regex exclusion scanning",
    )

    multi_hop_time_limit: float = Field(
        default=5.0,
        ge=1.0,
        le=15.0,
        description="Maximum duration for multi-hop expansion (seconds)",
    )

    multi_hop_result_limit: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="Maximum chunks accumulated during multi-hop expansion",
    )

    multi_hop_min_candidates: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Minimum candidates above threshold to continue expansion",
    )

    multi_hop_score_degradation: float = Field(
        default=0.15,
        ge=0.05,
        le=0.5,
        description="Maximum score drop in top-5 before terminating expansion",
    )

    multi_hop_min_relevance: float = Field(
        default=0.3,
        ge=0.1,
        le=0.8,
        description="Quality floor for expansion candidates",
    )

    # Phase 1.5: Depth Exploration Parameters
    depth_exploration_enabled: bool = Field(
        default=True,
        description="Enable depth exploration to find more chunks in discovered files",
    )

    max_exploration_files: int = Field(
        default=5,
        ge=1,
        le=15,
        description="Maximum files to explore for additional aspects (top-K by score)",
    )

    exploration_queries_per_file: int = Field(
        default=2,
        ge=1,
        le=3,
        description="Number of aspect-based queries to generate per file",
    )

    # Phase 2: Gap Detection Parameters
    min_gaps: int = Field(
        default=1,
        ge=0,
        le=5,
        description="Minimum gaps to process after selection",
    )

    max_gaps: int = Field(
        default=10,
        ge=5,
        le=30,
        description="Maximum gaps to fill after selection",
    )

    gap_similarity_threshold: float = Field(
        default=0.25,
        ge=0.1,
        le=0.5,
        description="Cosine distance threshold for clustering similar gaps",
    )

    shard_budget: int = Field(
        default=40_000,
        ge=20_000,
        le=60_000,
        description="Token budget per gap detection shard for LLM processing",
    )

    min_cluster_size: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Minimum cluster size for HDBSCAN clustering",
    )

    # Phase 3: Synthesis Parameters
    target_tokens: int = Field(
        default=20_000,
        ge=10_000,
        le=100_000,
        description="Output token budget for final synthesis",
    )

    max_compression_iterations: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum compression loop iterations before error",
    )

    max_boundary_expansion_lines: int = Field(
        default=_DEFAULT_MAX_BOUNDARY_EXPANSION_LINES,
        ge=50,
        le=500,
        description="Maximum lines to expand for complete functions/classes",
    )

    max_chunks_per_file_repr: int = Field(
        default=_DEFAULT_MAX_CHUNKS_PER_FILE_REPR,
        ge=1,
        le=10,
        description="Top chunks per file for representative document creation",
    )

    max_tokens_per_file_repr: int = Field(
        default=_DEFAULT_MAX_TOKENS_PER_FILE_REPR,
        ge=500,
        le=5000,
        description="Token limit per file representative document",
    )

    context_window: int = Field(
        default=150_000,
        ge=50_000,
        le=200_000,
        description="Maximum tokens for LLM context window",
    )

    compression_max_depth: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Maximum recursion depth for hierarchical compression",
    )

    final_synthesis_threshold: int = Field(
        default=75_000,
        ge=30_000,
        le=200_000,
        description="Maximum tokens for final synthesis LLM call",
    )

    # Context Enhancement Parameters
    window_expansion_enabled: bool = Field(
        default=True,
        description="Enable neighboring chunk expansion for context",
    )

    window_expansion_lines: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Lines to expand before/after retrieved chunks",
    )

    import_resolution_enabled: bool = Field(
        default=True,
        description="Automatically fetch source files for imports in retrieved chunks",
    )

    import_resolution_max_files: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of import source files to fetch per synthesis",
    )

    # Exhaustive Mode
    exhaustive_mode: bool = Field(
        default=False,
        description="Enable exhaustive retrieval (600s timeout, no result limit)",
    )

    exhaustive_time_limit: float = Field(
        default=600.0,
        ge=60.0,
        le=1800.0,
        description="Safety timeout for exhaustive mode (seconds)",
    )

    def get_effective_time_limit(self) -> float:
        """
        Get the effective time limit based on exhaustive mode setting.

        Returns:
            Time limit in seconds
        """
        if self.exhaustive_mode:
            return self.exhaustive_time_limit
        return self.multi_hop_time_limit

    def get_effective_result_limit(self) -> int | None:
        """
        Get the effective result limit based on exhaustive mode setting.

        Returns:
            Result limit or None if exhaustive mode (no limit)
        """
        return None if self.exhaustive_mode else self.multi_hop_result_limit

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add research-related CLI arguments."""
        parser.add_argument(
            "--research-algorithm",
            choices=["v1", "v2", "v3"],
            help="Research algorithm version (v1=BFS exploration, v2=hybrid v1 synthesis + wide coverage exploration, v3=parallel BFS + wide coverage)",
        )
        parser.add_argument(
            "--exhaustive-mode",
            action="store_true",
            default=None,
            help="Enable exhaustive retrieval (600s timeout, no result limit)",
        )
        parser.add_argument(
            "--multi-hop-time-limit",
            type=float,
            help="Maximum duration for multi-hop expansion in seconds (default: 5.0)",
        )
        parser.add_argument(
            "--multi-hop-result-limit",
            type=int,
            help="Maximum chunks accumulated during multi-hop expansion (default: 500)",
        )

    @classmethod
    def load_from_env(cls) -> dict[str, Any]:
        """Load research config from environment variables."""
        config: dict[str, Any] = {}

        if algorithm := os.getenv("CHUNKHOUND_RESEARCH_ALGORITHM"):
            config["algorithm"] = algorithm.strip().lower()

        if query_exp := os.getenv("CHUNKHOUND_RESEARCH_QUERY_EXPANSION_ENABLED"):
            config["query_expansion_enabled"] = query_exp.lower() in (
                "true",
                "1",
                "yes",
            )

        if num_queries := os.getenv("CHUNKHOUND_RESEARCH_NUM_EXPANDED_QUERIES"):
            config["num_expanded_queries"] = int(num_queries)

        if page_size := os.getenv("CHUNKHOUND_RESEARCH_INITIAL_PAGE_SIZE"):
            config["initial_page_size"] = int(page_size)

        if threshold := os.getenv("CHUNKHOUND_RESEARCH_RELEVANCE_THRESHOLD"):
            config["relevance_threshold"] = float(threshold)

        if max_syms := os.getenv("CHUNKHOUND_RESEARCH_MAX_SYMBOLS"):
            config["max_symbols"] = int(max_syms)

        if regex_ratio := os.getenv("CHUNKHOUND_RESEARCH_REGEX_AUGMENTATION_RATIO"):
            config["regex_augmentation_ratio"] = float(regex_ratio)

        if regex_min := os.getenv("CHUNKHOUND_RESEARCH_REGEX_MIN_RESULTS"):
            config["regex_min_results"] = int(regex_min)

        if scan_size := os.getenv("CHUNKHOUND_RESEARCH_REGEX_SCAN_PAGE_SIZE"):
            config["regex_scan_page_size"] = int(scan_size)

        if time_limit := os.getenv("CHUNKHOUND_RESEARCH_MULTI_HOP_TIME_LIMIT"):
            config["multi_hop_time_limit"] = float(time_limit)

        if result_limit := os.getenv("CHUNKHOUND_RESEARCH_MULTI_HOP_RESULT_LIMIT"):
            config["multi_hop_result_limit"] = int(result_limit)

        if min_cand := os.getenv("CHUNKHOUND_RESEARCH_MULTI_HOP_MIN_CANDIDATES"):
            config["multi_hop_min_candidates"] = int(min_cand)

        if score_deg := os.getenv("CHUNKHOUND_RESEARCH_MULTI_HOP_SCORE_DEGRADATION"):
            config["multi_hop_score_degradation"] = float(score_deg)

        if min_rel := os.getenv("CHUNKHOUND_RESEARCH_MULTI_HOP_MIN_RELEVANCE"):
            config["multi_hop_min_relevance"] = float(min_rel)

        # Phase 1.5: Depth Exploration
        if depth_exp := os.getenv("CHUNKHOUND_RESEARCH_DEPTH_EXPLORATION_ENABLED"):
            config["depth_exploration_enabled"] = depth_exp.lower() in (
                "true",
                "1",
                "yes",
            )

        if max_exp_files := os.getenv("CHUNKHOUND_RESEARCH_MAX_EXPLORATION_FILES"):
            config["max_exploration_files"] = int(max_exp_files)

        if exp_queries := os.getenv("CHUNKHOUND_RESEARCH_EXPLORATION_QUERIES_PER_FILE"):
            config["exploration_queries_per_file"] = int(exp_queries)

        if min_gaps := os.getenv("CHUNKHOUND_RESEARCH_MIN_GAPS"):
            config["min_gaps"] = int(min_gaps)

        if max_gaps := os.getenv("CHUNKHOUND_RESEARCH_MAX_GAPS"):
            config["max_gaps"] = int(max_gaps)

        if gap_thresh := os.getenv("CHUNKHOUND_RESEARCH_GAP_SIMILARITY_THRESHOLD"):
            config["gap_similarity_threshold"] = float(gap_thresh)

        if shard_budget := os.getenv("CHUNKHOUND_RESEARCH_SHARD_BUDGET"):
            config["shard_budget"] = int(shard_budget)

        if min_cluster := os.getenv("CHUNKHOUND_RESEARCH_MIN_CLUSTER_SIZE"):
            config["min_cluster_size"] = int(min_cluster)

        if target_tok := os.getenv("CHUNKHOUND_RESEARCH_TARGET_TOKENS"):
            config["target_tokens"] = int(target_tok)

        if max_iter := os.getenv("CHUNKHOUND_RESEARCH_MAX_COMPRESSION_ITERATIONS"):
            config["max_compression_iterations"] = int(max_iter)

        if max_expand := os.getenv("CHUNKHOUND_RESEARCH_MAX_BOUNDARY_EXPANSION_LINES"):
            config["max_boundary_expansion_lines"] = int(max_expand)

        if max_chunks := os.getenv("CHUNKHOUND_RESEARCH_MAX_CHUNKS_PER_FILE_REPR"):
            config["max_chunks_per_file_repr"] = int(max_chunks)

        if max_tok_repr := os.getenv("CHUNKHOUND_RESEARCH_MAX_TOKENS_PER_FILE_REPR"):
            config["max_tokens_per_file_repr"] = int(max_tok_repr)

        if ctx_window := os.getenv("CHUNKHOUND_RESEARCH_CONTEXT_WINDOW"):
            config["context_window"] = int(ctx_window)

        if comp_depth := os.getenv("CHUNKHOUND_RESEARCH_COMPRESSION_MAX_DEPTH"):
            config["compression_max_depth"] = int(comp_depth)

        if synth_thresh := os.getenv("CHUNKHOUND_RESEARCH_FINAL_SYNTHESIS_THRESHOLD"):
            config["final_synthesis_threshold"] = int(synth_thresh)

        # Context Enhancement
        if window_exp := os.getenv("CHUNKHOUND_RESEARCH_WINDOW_EXPANSION_ENABLED"):
            config["window_expansion_enabled"] = window_exp.lower() in (
                "true",
                "1",
                "yes",
            )

        if window_lines := os.getenv("CHUNKHOUND_RESEARCH_WINDOW_EXPANSION_LINES"):
            config["window_expansion_lines"] = int(window_lines)

        if import_res := os.getenv("CHUNKHOUND_RESEARCH_IMPORT_RESOLUTION_ENABLED"):
            config["import_resolution_enabled"] = import_res.lower() in (
                "true",
                "1",
                "yes",
            )

        if import_max := os.getenv("CHUNKHOUND_RESEARCH_IMPORT_RESOLUTION_MAX_FILES"):
            config["import_resolution_max_files"] = int(import_max)

        if exhaustive := os.getenv("CHUNKHOUND_RESEARCH_EXHAUSTIVE_MODE"):
            config["exhaustive_mode"] = exhaustive.lower() in ("true", "1", "yes")

        if exh_limit := os.getenv("CHUNKHOUND_RESEARCH_EXHAUSTIVE_TIME_LIMIT"):
            config["exhaustive_time_limit"] = float(exh_limit)

        return config

    @classmethod
    def extract_cli_overrides(cls, args: Any) -> dict[str, Any]:
        """Extract research config from CLI arguments."""
        overrides = {}

        if hasattr(args, "research_algorithm") and args.research_algorithm:
            overrides["algorithm"] = args.research_algorithm

        if hasattr(args, "exhaustive_mode") and args.exhaustive_mode is not None:
            overrides["exhaustive_mode"] = args.exhaustive_mode

        if (
            hasattr(args, "multi_hop_time_limit")
            and args.multi_hop_time_limit is not None
        ):
            overrides["multi_hop_time_limit"] = args.multi_hop_time_limit

        if (
            hasattr(args, "multi_hop_result_limit")
            and args.multi_hop_result_limit is not None
        ):
            overrides["multi_hop_result_limit"] = args.multi_hop_result_limit

        return overrides

    def __repr__(self) -> str:
        """String representation with key settings."""
        return (
            f"ResearchConfig("
            f"algorithm={self.algorithm}, "
            f"exhaustive_mode={self.exhaustive_mode}, "
            f"query_expansion_enabled={self.query_expansion_enabled}, "
            f"target_tokens={self.target_tokens})"
        )
