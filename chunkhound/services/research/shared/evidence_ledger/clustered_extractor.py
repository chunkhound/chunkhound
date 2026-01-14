"""Clustered fact extraction - clusters files before LLM extraction.

This module provides a unified fact extraction utility that:
1. Clusters files using HDBSCAN with token bounds for natural semantic groupings
2. Extracts facts from each cluster in parallel with proportional allocation
3. Returns both the evidence ledger AND cluster groups for downstream reuse

The cluster groups can be reused for synthesis (map-reduce) without redundant
clustering, improving performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

from chunkhound.services.clustering_service import ClusterGroup, ClusteringService

from .extractor import FactExtractor
from .ledger import EvidenceLedger

if TYPE_CHECKING:
    from chunkhound.interfaces.embedding_provider import EmbeddingProvider
    from chunkhound.interfaces.llm_provider import LLMProvider

# Token bounds for HDBSCAN clustering
MIN_TOKENS_PER_CLUSTER = 15_000
MAX_TOKENS_PER_CLUSTER = 50_000

# Proportional fact allocation: 25 facts per 100k tokens
FACTS_PER_100K_TOKENS = 25
MIN_FACTS_PER_CLUSTER = 3


@dataclass
class ClusteredExtractionResult:
    """Result of clustered fact extraction, including clusters for reuse.

    Attributes:
        evidence_ledger: Merged EvidenceLedger with facts from all clusters
        cluster_groups: HDBSCAN cluster groups (reusable for synthesis)
        cluster_metadata: Clustering statistics (num_clusters, avg_tokens_per_cluster, etc.)
    """

    evidence_ledger: EvidenceLedger
    cluster_groups: list[ClusterGroup]
    cluster_metadata: dict[str, Any]


async def extract_facts_with_clustering(
    files: dict[str, str],
    root_query: str,
    llm_provider: LLMProvider,
    embedding_provider: EmbeddingProvider,
    max_concurrency: int = 4,
    min_tokens_per_cluster: int = MIN_TOKENS_PER_CLUSTER,
    max_tokens_per_cluster: int = MAX_TOKENS_PER_CLUSTER,
) -> ClusteredExtractionResult:
    """Extract facts from files using HDBSCAN bounded clustering.

    Clusters files via HDBSCAN with token bounds, then extracts facts from each
    cluster in parallel with proportional fact allocation. Returns both the
    evidence ledger AND the cluster groups, enabling reuse in downstream synthesis.

    This prevents prompt size overflow for large file sets by ensuring each
    LLM call only sees files from a single cluster, staying within context limits.
    Fact allocation scales with cluster size (25 facts per 100k tokens).

    Args:
        files: Dict mapping file_path -> content
        root_query: Research query for context
        llm_provider: LLM provider for fact extraction (utility model)
        embedding_provider: Embedding provider for clustering
        max_concurrency: Maximum parallel LLM calls
        min_tokens_per_cluster: Minimum tokens per cluster (HDBSCAN bound)
        max_tokens_per_cluster: Maximum tokens per cluster (HDBSCAN bound)

    Returns:
        ClusteredExtractionResult with:
        - evidence_ledger: Merged facts from all clusters
        - cluster_groups: HDBSCAN cluster groups (reuse for synthesis)
        - cluster_metadata: Clustering stats (num_clusters, etc.)
    """
    if not files:
        return ClusteredExtractionResult(
            evidence_ledger=EvidenceLedger(),
            cluster_groups=[],
            cluster_metadata={"num_clusters": 0},
        )

    clustering_service = ClusteringService(
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
    )

    # Use HDBSCAN with token bounds for natural semantic groupings
    cluster_groups, metadata = await clustering_service.cluster_files_hdbscan_bounded(
        files,
        min_tokens_per_cluster=min_tokens_per_cluster,
        max_tokens_per_cluster=max_tokens_per_cluster,
    )

    logger.info(
        f"Clustered {len(files)} files into {metadata['num_clusters']} HDBSCAN groups "
        f"(bounds: [{min_tokens_per_cluster:,}, {max_tokens_per_cluster:,}]) for fact extraction"
    )

    # Convert ClusterGroup objects to extraction format with proportional fact allocation
    clusters_for_extraction = [
        (
            cluster.cluster_id,
            cluster.files_content,
            max(MIN_FACTS_PER_CLUSTER, int(cluster.total_tokens * FACTS_PER_100K_TOKENS / 100_000)),
        )
        for cluster in cluster_groups
    ]

    # Extract facts from all clusters
    extractor = FactExtractor(llm_provider)
    evidence_ledger = await extractor.extract_from_clusters(
        clusters=clusters_for_extraction,
        root_query=root_query,
        max_concurrency=max_concurrency,
    )

    return ClusteredExtractionResult(
        evidence_ledger=evidence_ledger,
        cluster_groups=cluster_groups,
        cluster_metadata=metadata,
    )
