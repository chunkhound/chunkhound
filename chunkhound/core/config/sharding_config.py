"""Sharding configuration for ChunkHound.

This module provides configuration for database sharding behavior including
thresholds for splitting, merging, compaction, and quality control.
"""

from typing import Literal

from pydantic import BaseModel, Field

# Bytes per dimension for each quantization type
QUANTIZATION_BYTES: dict[str, int] = {"i8": 1, "f16": 2, "f32": 4, "f64": 8}


def bytes_per_vector(dims: int, quantization: str, connectivity: int) -> int:
    """Calculate bytes per vector in HNSW index including graph overhead.

    Args:
        dims: Vector dimensionality
        quantization: Quantization type (i8, f16, f32, f64)
        connectivity: HNSW M parameter (edges per node)

    Returns:
        Estimated bytes per vector in memory
    """
    vector_bytes = dims * QUANTIZATION_BYTES.get(quantization, 1)
    # HNSW overhead: connectivity edges (uint32 each) + metadata
    hnsw_overhead = (connectivity * 8) + 16
    return vector_bytes + hnsw_overhead


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

    rebuild_batch_size: int = Field(
        default=10_000,
        ge=1000,
        le=100_000,
        description="Batch size for streaming embeddings during index rebuild",
    )

    enable_aggressive_gc: bool = Field(
        default=False,
        description=(
            "Enable aggressive garbage collection after each shard operation "
            "(useful for memory-constrained environments)"
        ),
    )

    incremental_sync_threshold: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Change ratio threshold for incremental vs full sync",
    )

    quality_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum quality score for shard index (i8-tuned)",
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
        description="Maximum number of shards to query in parallel (overridden by memory_budget_bytes)",
    )

    memory_budget_bytes: int = Field(
        default=2 * 1024**3,  # 2GB
        ge=256 * 1024**2,  # min 256MB
        description="Memory budget for concurrent shard loading in bytes",
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

    background_rebuild_enabled: bool = Field(
        default=True,
        description="Enable non-blocking background shard rebuilds",
    )

    def __repr__(self) -> str:
        """String representation of sharding configuration."""
        return (
            f"ShardingConfig("
            f"split_threshold={self.split_threshold}, "
            f"merge_threshold={self.merge_threshold}, "
            f"default_quantization={self.default_quantization!r})"
        )
