"""Clustering service for grouping sources in map-reduce synthesis.

Uses two-phase HDBSCAN clustering on embeddings to group files into token-bounded
clusters for parallel synthesis operations:
1. Phase 1: HDBSCAN discovers natural semantic clusters
2. Phase 2: Greedy grouping merges clusters to approach token budget
"""

from dataclasses import dataclass

import hdbscan  # type: ignore[import-untyped]
import numpy as np
from loguru import logger
from sklearn.metrics import pairwise_distances  # type: ignore[import-not-found]

from chunkhound.interfaces.embedding_provider import EmbeddingProvider
from chunkhound.interfaces.llm_provider import LLMProvider


@dataclass
class ClusterGroup:
    """A cluster of files for synthesis."""

    cluster_id: int
    file_paths: list[str]
    files_content: dict[str, str]  # file_path -> content
    total_tokens: int


class ClusteringService:
    """Service for clustering files into token-bounded groups using HDBSCAN."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
        max_tokens_per_cluster: int = 30000,
        min_cluster_size: int = 3,
    ):
        """Initialize clustering service.

        Args:
            embedding_provider: Provider for generating embeddings
            llm_provider: Provider for token estimation
            max_tokens_per_cluster: Maximum tokens allowed per cluster (hard limit)
            min_cluster_size: Minimum files for HDBSCAN natural clusters
        """
        self._embedding_provider = embedding_provider
        self._llm_provider = llm_provider
        self._max_tokens_per_cluster = max_tokens_per_cluster
        self._min_cluster_size = min_cluster_size

    async def cluster_files(
        self, files: dict[str, str]
    ) -> tuple[list[ClusterGroup], dict[str, int]]:
        """Cluster files into token-bounded groups using two-phase HDBSCAN.

        Args:
            files: Dictionary mapping file_path -> file_content

        Returns:
            Tuple of (cluster_groups, metadata) where metadata contains:
                - num_native_clusters: Natural clusters from HDBSCAN Phase 1
                - num_outliers: Noise points detected by HDBSCAN
                - num_clusters: Final clusters after Phase 2 grouping
                - total_files: Total number of files
                - total_tokens: Total tokens across all files
                - avg_tokens_per_cluster: Average tokens per final cluster

        Raises:
            ValueError: If files dict is empty
        """
        if not files:
            raise ValueError("Cannot cluster empty files dictionary")

        # Calculate total tokens
        total_tokens = sum(
            self._llm_provider.estimate_tokens(content) for content in files.values()
        )

        logger.info(
            f"Clustering {len(files)} files ({total_tokens:,} tokens) "
            f"with two-phase HDBSCAN (max {self._max_tokens_per_cluster:,} tokens/cluster)"
        )

        # Special case: single file or all fit in one cluster
        if len(files) == 1 or total_tokens <= self._max_tokens_per_cluster:
            logger.info("Single cluster sufficient - will use single-pass synthesis")
            cluster_group = ClusterGroup(
                cluster_id=0,
                file_paths=list(files.keys()),
                files_content=files,
                total_tokens=total_tokens,
            )
            metadata = {
                "num_native_clusters": 1,
                "num_outliers": 0,
                "num_clusters": 1,
                "total_files": len(files),
                "total_tokens": total_tokens,
                "avg_tokens_per_cluster": total_tokens,
            }
            return [cluster_group], metadata

        # Generate embeddings for each file
        file_paths = list(files.keys())
        file_contents = [files[fp] for fp in file_paths]

        logger.debug(f"Generating embeddings for {len(file_contents)} files")
        embeddings = await self._embedding_provider.embed(file_contents)

        # Phase 1: HDBSCAN discovery of natural clusters
        logger.debug("Phase 1: Discovering natural clusters with HDBSCAN")
        embeddings_array = np.array(embeddings)
        labels, phase1_meta = self._discover_natural_clusters(
            embeddings_array
        )

        # Partition files by cluster labels
        native_clusters: dict[int, list[str]] = {}
        for file_path, cluster_id in zip(file_paths, labels):
            native_clusters.setdefault(int(cluster_id), []).append(file_path)

        logger.info(
            f"Phase 1 complete: {phase1_meta['num_native_clusters']} clusters, "
            f"{phase1_meta['num_outliers']} outliers"
        )

        # Phase 2: Greedy grouping to token budget
        logger.debug("Phase 2: Grouping clusters to token budget")
        final_clusters, cluster_id_to_final_idx = self._group_clusters_to_budget(
            native_clusters, files, labels, embeddings_array
        )

        # Build cluster groups with token counts
        cluster_groups: list[ClusterGroup] = []
        for cluster_id, cluster_file_paths in enumerate(final_clusters):
            cluster_files_content = {fp: files[fp] for fp in cluster_file_paths}
            cluster_tokens = sum(
                self._llm_provider.estimate_tokens(content)
                for content in cluster_files_content.values()
            )

            cluster_group = ClusterGroup(
                cluster_id=cluster_id,
                file_paths=cluster_file_paths,
                files_content=cluster_files_content,
                total_tokens=cluster_tokens,
            )
            cluster_groups.append(cluster_group)

            logger.debug(
                f"Cluster {cluster_id}: {len(cluster_file_paths)} files, "
                f"{cluster_tokens:,} tokens"
            )

        avg_tokens = total_tokens / len(cluster_groups) if cluster_groups else 0
        metadata = {
            "num_native_clusters": phase1_meta["num_native_clusters"],
            "num_outliers": phase1_meta["num_outliers"],
            "num_clusters": len(cluster_groups),
            "total_files": len(files),
            "total_tokens": total_tokens,
            "avg_tokens_per_cluster": int(avg_tokens),
        }

        logger.info(
            f"Phase 2 complete: {len(cluster_groups)} final clusters, "
            f"avg {int(avg_tokens):,} tokens/cluster"
        )

        return cluster_groups, metadata

    def _discover_natural_clusters(
        self, embeddings: np.ndarray
    ) -> tuple[np.ndarray, dict[str, int]]:
        """Phase 1: Use HDBSCAN to discover natural semantic clusters.

        Args:
            embeddings: Array of embedding vectors (n_samples, n_features)

        Returns:
            Tuple of (labels, metadata) where:
                - labels: Array of cluster IDs (-1 for outliers)
                - metadata: Dict with num_native_clusters and num_outliers
        """
        # Adjust min_cluster_size if too large for dataset
        # HDBSCAN requires minimum 2 points to form a cluster
        min_cluster_size = min(self._min_cluster_size, len(embeddings) - 1)
        min_cluster_size = max(2, min_cluster_size)

        # HDBSCAN parameter selection:
        # - min_cluster_size: User-configurable (default: 3)
        # - min_samples: Set to 1 to produce fine-grained clusters for Phase 2 merging
        #   (HDBSCAN default is min_cluster_size, but that creates fewer, larger clusters
        #   which can exceed token budget. We need many small clusters to merge optimally)
        # - metric: 'euclidean' is standard for low-dim embeddings
        # - cluster_selection_method: 'eom' (Excess of Mass) finds stable clusters
        #   (alternative 'leaf' would produce even more fine-grained clusters)
        # - allow_single_cluster: True handles edge case of very small/homogeneous repos
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=1,
            metric="euclidean",
            cluster_selection_method="eom",
            allow_single_cluster=True,
        )

        try:
            labels = clusterer.fit_predict(embeddings)
        except Exception as e:
            logger.warning(
                f"HDBSCAN clustering failed: {e}. "
                f"Falling back to single cluster for {len(embeddings)} files"
            )
            # Fallback: assign all files to cluster 0
            labels = np.zeros(len(embeddings), dtype=int)
            metadata = {
                "num_native_clusters": 1,
                "num_outliers": 0,
            }
            return labels, metadata

        # Count native clusters (excluding outliers with label=-1)
        unique_labels = set(labels)
        num_native_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        num_outliers = int(np.sum(labels == -1))

        logger.debug(
            f"HDBSCAN found {num_native_clusters} natural clusters, "
            f"{num_outliers} outliers"
        )

        metadata = {
            "num_native_clusters": num_native_clusters,
            "num_outliers": num_outliers,
        }

        return labels, metadata

    def _group_clusters_to_budget(
        self,
        native_clusters: dict[int, list[str]],
        files: dict[str, str],
        labels: np.ndarray,
        embeddings: np.ndarray,
    ) -> tuple[list[list[str]], dict[int, int]]:
        """Phase 2: Greedily merge clusters to approach token budget.

        Strategy:
        1. Calculate token counts for each native cluster
        2. Build distance matrix between cluster centroids
        3. Iteratively merge closest clusters while staying under budget
        4. Handle outliers by creating separate cluster

        Args:
            native_clusters: Dict mapping cluster_id -> file_paths
            files: Dict mapping file_path -> content (for token counting)
            labels: Array of cluster labels from HDBSCAN
            embeddings: Array of embedding vectors

        Returns:
            Tuple of (final_clusters, cluster_id_to_final_idx) where:
                - final_clusters: List of file path lists
                - cluster_id_to_final_idx: Maps native cluster_id to final cluster index
        """
        # Step 1: Calculate token counts per native cluster
        cluster_tokens: dict[int, int] = {}
        for cluster_id, file_paths in native_clusters.items():
            cluster_tokens[cluster_id] = sum(
                self._llm_provider.estimate_tokens(files[fp]) for fp in file_paths
            )

        # Separate outliers (cluster_id=-1) for special handling
        outlier_cluster = native_clusters.pop(-1, [])
        if -1 in cluster_tokens:
            del cluster_tokens[-1]

        # Handle edge case: no native clusters, only outliers
        if not native_clusters:
            logger.warning("No native clusters found, using outliers as single cluster")
            return ([outlier_cluster] if outlier_cluster else [], {})

        # Step 2: Calculate cluster centroids from embeddings
        cluster_ids = sorted([cid for cid in native_clusters.keys()])

        # Calculate centroid for each cluster as mean of member embeddings
        centroids = []
        for cluster_id in cluster_ids:
            mask = labels == cluster_id
            cluster_embeddings = embeddings[mask]
            if len(cluster_embeddings) > 0:
                centroid = cluster_embeddings.mean(axis=0)
            else:
                # Fallback: zero vector (shouldn't happen)
                centroid = np.zeros(embeddings.shape[1])
            centroids.append(centroid)

        distances = pairwise_distances(np.array(centroids), metric="euclidean")

        # Step 3: Greedy merging with token budget constraint
        # Track which clusters have been merged (map cluster_id -> merged_group_id)
        merged_groups: dict[int, int] = {cid: cid for cid in cluster_ids}
        group_to_clusters: dict[int, list[int]] = {
            cid: [cid] for cid in cluster_ids
        }  # Maps group_id to list of native cluster_ids in that group
        group_tokens: dict[int, int] = cluster_tokens.copy()  # group_id -> token_count

        while True:
            # Find closest pair of unmerged groups that can be merged
            best_pair = self._find_best_merge_candidate(
                cluster_ids, merged_groups, group_tokens, distances
            )

            # No more valid merges
            if best_pair is None:
                break

            # Merge the closest pair
            ci, cj = best_pair
            group_i = merged_groups[ci]
            group_j = merged_groups[cj]

            # Merge group_j into group_i
            group_to_clusters[group_i].extend(group_to_clusters[group_j])
            group_tokens[group_i] += group_tokens[group_j]

            # Update all members of group_j to point to group_i
            for cluster_id in group_to_clusters[group_j]:
                merged_groups[cluster_id] = group_i

            # Remove group_j
            del group_to_clusters[group_j]
            del group_tokens[group_j]

        # Step 4: Build final cluster file lists and mapping
        final_clusters: list[list[str]] = []
        cluster_id_to_final_idx: dict[int, int] = {}

        # Add merged groups and build mapping
        for final_idx, group_id in enumerate(sorted(group_to_clusters.keys())):
            cluster_files: list[str] = []
            for cluster_id in group_to_clusters[group_id]:
                cluster_files.extend(native_clusters[cluster_id])
                # Map each native cluster_id to this final cluster index
                cluster_id_to_final_idx[cluster_id] = final_idx
            final_clusters.append(cluster_files)

        # Merge outliers into nearest clusters (respecting token budget)
        if outlier_cluster:
            outliers_merged = self._merge_outliers_to_nearest_clusters(
                outlier_cluster,
                final_clusters,
                files,
                labels,
                embeddings,
                np.array(centroids),
                cluster_id_to_final_idx,
                cluster_ids,
            )
            logger.debug(
                f"Merged {outliers_merged} outlier files into nearest clusters, "
                f"{len(outlier_cluster) - outliers_merged} remain as separate cluster"
            )

            # If some outliers couldn't be merged (budget constraints), add as separate cluster
            if len(outlier_cluster) > 0:
                final_clusters.append(outlier_cluster)

        return final_clusters, cluster_id_to_final_idx

    def _find_best_merge_candidate(
        self,
        cluster_ids: list[int],
        merged_groups: dict[int, int],
        group_tokens: dict[int, int],
        distances: np.ndarray,
    ) -> tuple[int, int] | None:
        """Find the closest pair of clusters that can be merged within token budget.

        Args:
            cluster_ids: List of native cluster IDs
            merged_groups: Map of cluster_id -> current_group_id
            group_tokens: Map of group_id -> token_count
            distances: Distance matrix between cluster centroids

        Returns:
            Tuple of (cluster_id_i, cluster_id_j) if valid merge found, else None
        """
        best_pair: tuple[int, int] | None = None
        best_distance = float("inf")

        for i, ci in enumerate(cluster_ids):
            for j in range(i + 1, len(cluster_ids)):
                cj = cluster_ids[j]

                # Skip if already in same group
                if merged_groups[ci] == merged_groups[cj]:
                    continue

                # Calculate merged token count
                group_i = merged_groups[ci]
                group_j = merged_groups[cj]
                merged_tokens = group_tokens[group_i] + group_tokens[group_j]

                # Check if merge respects hard budget limit
                if merged_tokens <= self._max_tokens_per_cluster:
                    if distances[i][j] < best_distance:
                        best_distance = distances[i][j]
                        best_pair = (ci, cj)

        return best_pair

    def _merge_outliers_to_nearest_clusters(
        self,
        outlier_files: list[str],
        final_clusters: list[list[str]],
        files: dict[str, str],
        labels: np.ndarray,
        embeddings: np.ndarray,
        centroids: np.ndarray,
        cluster_id_to_final_idx: dict[int, int],
        cluster_ids: list[int],
    ) -> int:
        """Merge outlier files into their nearest clusters while respecting token budget.

        Args:
            outlier_files: List of file paths marked as outliers (modified in-place)
            final_clusters: List of final clusters (modified in-place)
            files: Dictionary mapping file_path -> content
            labels: Original HDBSCAN labels array
            embeddings: Array of all embedding vectors
            centroids: Array of cluster centroids (indexed by position in sorted cluster_ids)
            cluster_id_to_final_idx: Maps native cluster_id to final_clusters index
            cluster_ids: Sorted list of native cluster IDs (matches centroids order)

        Returns:
            Number of outliers successfully merged
        """
        if not final_clusters:
            return 0

        # Build map from file_path to embedding index
        file_paths = list(files.keys())
        path_to_idx = {fp: i for i, fp in enumerate(file_paths)}

        outliers_merged = 0
        remaining_outliers = []

        for outlier_file in outlier_files:
            # Get outlier embedding
            if outlier_file not in path_to_idx:
                remaining_outliers.append(outlier_file)
                continue

            outlier_idx = path_to_idx[outlier_file]
            outlier_embedding = embeddings[outlier_idx]
            outlier_tokens = self._llm_provider.estimate_tokens(files[outlier_file])

            # Find nearest cluster that can accommodate this outlier
            distances_to_centroids = np.linalg.norm(
                centroids - outlier_embedding, axis=1
            )
            sorted_centroid_indices = np.argsort(distances_to_centroids)

            merged = False
            for centroid_idx in sorted_centroid_indices:
                # Map centroid index to native cluster_id, then to final cluster index
                native_cluster_id = cluster_ids[centroid_idx]
                final_cluster_idx = cluster_id_to_final_idx[native_cluster_id]

                # Calculate current cluster token count
                cluster_files = final_clusters[final_cluster_idx]
                cluster_tokens = sum(
                    self._llm_provider.estimate_tokens(files[fp])
                    for fp in cluster_files
                )

                # Check if adding outlier respects budget
                if cluster_tokens + outlier_tokens <= self._max_tokens_per_cluster:
                    final_clusters[final_cluster_idx].append(outlier_file)
                    outliers_merged += 1
                    merged = True
                    break

            if not merged:
                remaining_outliers.append(outlier_file)

        # Update outlier_files list to only contain unmerged outliers
        outlier_files.clear()
        outlier_files.extend(remaining_outliers)

        return outliers_merged
