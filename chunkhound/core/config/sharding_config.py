"""Sharding configuration for ChunkHound.

This module provides configuration for database sharding behavior including
thresholds for splitting, merging, compaction, and quality control.
"""

from typing import Literal

from pydantic import BaseModel, Field


class ShardingConfig(BaseModel):
    """Configuration for database sharding behavior.

    Controls when shards are split, merged, and compacted based on
    size thresholds and quality metrics.
    """

    split_threshold: int = Field(
        default=100_000,
        ge=50,
        description="Number of chunks that triggers shard splitting",
    )

    merge_threshold: int = Field(
        default=10_000,
        ge=10,
        description="Number of chunks below which shards are merged",
    )

    merge_target_count: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Candidate target shards for per-vector routing during merge",
    )

    compaction_threshold: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Fragmentation ratio that triggers shard compaction",
    )

    incremental_sync_threshold: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Change ratio threshold for incremental vs full sync",
    )

    quality_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Minimum quality score for shard index",
    )

    quality_sample_size: int = Field(
        default=1000,
        ge=10,
        description="Number of samples used for quality evaluation",
    )

    shard_similarity_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for shard clustering decisions",
    )

    max_concurrent_shards: int = Field(
        default=4,
        ge=1,
        description="Maximum number of shards to query in parallel",
    )

    default_quantization: Literal["f32", "f16", "i8"] = Field(
        default="i8",
        description="Default vector quantization format: f32, f16, or i8",
    )

    overfetch_multiplier: int = Field(
        default=5,
        ge=2,
        le=10,
        description="Multiplier for k during search to provide reranking candidates",
    )

    hnsw_connectivity: int = Field(
        default=16,
        ge=8,
        le=64,
        description="Maximum connections per node in HNSW graph (M parameter)",
    )

    hnsw_expansion_add: int = Field(
        default=128,
        ge=64,
        le=512,
        description="Expansion factor during index construction (efConstruction)",
    )

    hnsw_expansion_search: int = Field(
        default=64,
        ge=32,
        le=256,
        description="Expansion factor during search (ef)",
    )

    quality_check_mode: Literal["immediate", "deferred"] = Field(
        default="immediate",
        description="When to run quality checks: immediate or deferred (bulk mode)",
    )

    quality_check_interval: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Batches between quality checks when mode is deferred",
    )

    def __repr__(self) -> str:
        """String representation of sharding configuration."""
        return (
            f"ShardingConfig("
            f"split_threshold={self.split_threshold}, "
            f"merge_threshold={self.merge_threshold}, "
            f"default_quantization={self.default_quantization!r})"
        )
