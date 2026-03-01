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

def cluster_files_by_system_token_pack(
    files: dict[str, str],
    *,
    file_to_system_id: dict[str, int | None],
    llm_provider: LLMProvider,
    min_tokens_per_cluster: int = MIN_TOKENS_PER_CLUSTER,
    max_tokens_per_cluster: int = MAX_TOKENS_PER_CLUSTER,
) -> tuple[list[ClusterGroup], dict[str, Any]]:
    """Cluster files by snapshot system id with deterministic token packing.

    This provides a snapshot-aware clustering path that does not require embeddings.
    Clusters are initially grouped by system_id, then split/merged deterministically
    under token bounds.

    Notes:
    - Single files exceeding max_tokens_per_cluster are returned as single-file
      overflow clusters (best-effort).
    - Underfilled clusters are merged/packed deterministically when possible, but a
      final underfilled cluster may remain when unavoidable.
    """
    if not files:
        raise ValueError("Cannot cluster empty files dictionary")

    file_tokens: dict[str, int] = {
        fp: int(llm_provider.estimate_tokens(content)) for fp, content in files.items()
    }
    total_tokens = int(sum(file_tokens.values()))

    # Deterministic system grouping based on file paths present in `files`.
    system_to_files: dict[int | None, list[str]] = {}
    for fp in sorted(files):
        system_id = file_to_system_id.get(fp)
        if system_id is not None:
            try:
                system_id = int(system_id)
            except Exception:
                system_id = None
        system_to_files.setdefault(system_id, []).append(fp)

    bins: list[dict[str, Any]] = []
    next_order = 0
    num_overflow_bins = 0
    num_splits = 0

    def system_sort_key(system_id: int | None) -> tuple[int, int]:
        if system_id is None:
            return (1, 0)
        return (0, int(system_id))

    for system_id in sorted(system_to_files.keys(), key=system_sort_key):
        fps = system_to_files[system_id]
        overflow_files = [fp for fp in fps if file_tokens[fp] > max_tokens_per_cluster]
        normal_files = [fp for fp in fps if file_tokens[fp] <= max_tokens_per_cluster]

        for fp in sorted(overflow_files):
            bins.append(
                {
                    "order": next_order,
                    "file_paths": [fp],
                    "tokens": int(file_tokens[fp]),
                    "overflow": True,
                }
            )
            next_order += 1
            num_overflow_bins += 1

        items = sorted(normal_files, key=lambda fp: (-file_tokens[fp], fp))
        system_bins: list[dict[str, Any]] = []
        for fp in items:
            tokens = int(file_tokens[fp])
            placed = False
            for b in system_bins:
                if int(b["tokens"]) + tokens <= max_tokens_per_cluster:
                    b["file_paths"].append(fp)
                    b["tokens"] = int(b["tokens"]) + tokens
                    placed = True
                    break
            if not placed:
                system_bins.append(
                    {
                        "order": next_order,
                        "file_paths": [fp],
                        "tokens": tokens,
                        "overflow": False,
                    }
                )
                next_order += 1

        if len(system_bins) > 1:
            num_splits += 1

        for b in system_bins:
            b["file_paths"].sort()
            bins.append(b)

    # Pass 1: merge underfilled bins into existing bins (best-fit, deterministic).
    num_merges = 0
    active_bins = [b for b in bins if b.get("file_paths")]
    active_bins.sort(key=lambda b: int(b["order"]))

    def is_small(b: dict[str, Any]) -> bool:
        return (not bool(b.get("overflow"))) and int(b["tokens"]) < min_tokens_per_cluster

    small_bins = [b for b in active_bins if is_small(b)]
    small_bins.sort(key=lambda b: (int(b["tokens"]), int(b["order"])))

    for b in small_bins:
        if not b.get("file_paths"):
            continue

        candidates: list[tuple[int, int, dict[str, Any]]] = []
        b_tokens = int(b["tokens"])
        for tgt in active_bins:
            if tgt is b:
                continue
            if not tgt.get("file_paths"):
                continue
            if bool(tgt.get("overflow")):
                continue
            tgt_tokens = int(tgt["tokens"])
            if tgt_tokens + b_tokens <= max_tokens_per_cluster:
                remaining = max_tokens_per_cluster - (tgt_tokens + b_tokens)
                candidates.append((int(remaining), int(tgt["order"]), tgt))

        if not candidates:
            continue

        candidates.sort(key=lambda t: (t[0], t[1]))
        target = candidates[0][2]
        target["file_paths"].extend(b["file_paths"])
        target["file_paths"].sort()
        target["tokens"] = int(target["tokens"]) + b_tokens

        b["file_paths"] = []
        b["tokens"] = 0
        num_merges += 1

    # Pass 2: pack remaining small bins together (bin-level FFD, deterministic).
    remaining_bins = [b for b in active_bins if b.get("file_paths")]
    remaining_small = [b for b in remaining_bins if is_small(b)]
    keep_bins = [b for b in remaining_bins if b not in remaining_small]

    remaining_small.sort(key=lambda b: (-int(b["tokens"]), int(b["order"])))
    packed_bins: list[dict[str, Any]] = []
    for b in remaining_small:
        b_tokens = int(b["tokens"])
        placed = False
        for pb in packed_bins:
            if int(pb["tokens"]) + b_tokens <= max_tokens_per_cluster:
                pb["file_paths"].extend(b["file_paths"])
                pb["file_paths"].sort()
                pb["tokens"] = int(pb["tokens"]) + b_tokens
                placed = True
                break
        if not placed:
            packed_bins.append(
                {
                    "order": int(b["order"]),
                    "file_paths": list(b["file_paths"]),
                    "tokens": b_tokens,
                    "overflow": False,
                }
            )

    final_bins = keep_bins + packed_bins
    final_bins.sort(
        key=lambda b: (
            -int(b["tokens"]),
            str((b.get("file_paths") or [""])[0]),
            int(b["order"]),
        )
    )

    cluster_groups: list[ClusterGroup] = []
    for cluster_id, b in enumerate(final_bins):
        file_paths = list(b["file_paths"])
        files_content = {fp: files[fp] for fp in file_paths}
        cluster_groups.append(
            ClusterGroup(
                cluster_id=int(cluster_id),
                file_paths=file_paths,
                files_content=files_content,
                total_tokens=int(b["tokens"]),
            )
        )

    metadata = {
        "num_clusters": len(cluster_groups),
        "total_files": len(files),
        "total_tokens": total_tokens,
        "avg_tokens_per_cluster": int(total_tokens / max(1, len(cluster_groups))),
        "min_tokens_per_cluster": int(min_tokens_per_cluster),
        "max_tokens_per_cluster": int(max_tokens_per_cluster),
        "num_overflow_bins": int(num_overflow_bins),
        "num_splits": int(num_splits),
        "num_merges": int(num_merges),
    }
    return cluster_groups, metadata


async def extract_facts_with_system_clustering(
    files: dict[str, str],
    root_query: str,
    llm_provider: LLMProvider,
    *,
    file_to_system_id: dict[str, int | None],
    max_concurrency: int = 4,
    min_tokens_per_cluster: int = MIN_TOKENS_PER_CLUSTER,
    max_tokens_per_cluster: int = MAX_TOKENS_PER_CLUSTER,
) -> ClusteredExtractionResult:
    """Extract facts using snapshot system clustering + deterministic token packing."""
    if not files:
        return ClusteredExtractionResult(
            evidence_ledger=EvidenceLedger(),
            cluster_groups=[],
            cluster_metadata={"num_clusters": 0},
        )

    cluster_groups, metadata = cluster_files_by_system_token_pack(
        files,
        file_to_system_id=file_to_system_id,
        llm_provider=llm_provider,
        min_tokens_per_cluster=min_tokens_per_cluster,
        max_tokens_per_cluster=max_tokens_per_cluster,
    )

    logger.info(
        f"System token-pack clustered {len(files)} files into {metadata['num_clusters']} groups "
        f"(bounds: [{min_tokens_per_cluster:,}, {max_tokens_per_cluster:,}]) for fact extraction"
    )

    clusters_for_extraction = [
        (
            cluster.cluster_id,
            cluster.files_content,
            max(
                MIN_FACTS_PER_CLUSTER,
                int(cluster.total_tokens * FACTS_PER_100K_TOKENS / 100_000),
            ),
        )
        for cluster in cluster_groups
    ]

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
