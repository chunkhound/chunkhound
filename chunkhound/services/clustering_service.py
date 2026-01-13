"""Clustering service for grouping sources in map-reduce synthesis.

Uses HDBSCAN for natural semantic clustering (first pass) and k-means
for budget-based clustering (subsequent passes) to group files into
token-bounded clusters for parallel synthesis operations.
"""

from dataclasses import dataclass

import hdbscan
import numpy as np
from loguru import logger
from sklearn.cluster import KMeans  # type: ignore[import-untyped]

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
    """Service for clustering files using k-means or HDBSCAN algorithms."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
    ):
        """Initialize clustering service.

        Args:
            embedding_provider: Provider for generating embeddings
            llm_provider: Provider for token estimation
        """
        self._embedding_provider = embedding_provider
        self._llm_provider = llm_provider

    async def cluster_files(
        self, files: dict[str, str], n_clusters: int
    ) -> tuple[list[ClusterGroup], dict[str, int]]:
        """Cluster files into exactly n_clusters using k-means.

        Args:
            files: Dictionary mapping file_path -> file_content
            n_clusters: Exact number of clusters to produce

        Returns:
            Tuple of (cluster_groups, metadata) where metadata contains:
                - num_clusters: Number of clusters (equals n_clusters)
                - total_files: Total number of files
                - total_tokens: Total tokens across all files
                - avg_tokens_per_cluster: Average tokens per cluster

        Raises:
            ValueError: If files dict is empty or n_clusters < 1
        """
        if not files:
            raise ValueError("Cannot cluster empty files dictionary")
        if n_clusters < 1:
            raise ValueError("n_clusters must be at least 1")

        # Clamp n_clusters to number of files
        n_clusters = min(n_clusters, len(files))

        # Calculate total tokens
        total_tokens = sum(
            self._llm_provider.estimate_tokens(content) for content in files.values()
        )

        logger.info(
            f"K-means clustering {len(files)} files ({total_tokens:,} tokens) "
            f"into {n_clusters} clusters"
        )

        # Special case: single cluster requested or single file
        if n_clusters == 1 or len(files) == 1:
            logger.info("Single cluster - will produce single output")
            cluster_group = ClusterGroup(
                cluster_id=0,
                file_paths=list(files.keys()),
                files_content=files,
                total_tokens=total_tokens,
            )
            metadata = {
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
        embeddings = await self._embedding_provider.embed_batch(file_contents)
        embeddings_array = np.array(embeddings)

        # K-means clustering
        logger.debug(f"Running k-means with n_clusters={n_clusters}")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings_array)

        # Build cluster groups
        cluster_to_files: dict[int, list[str]] = {}
        for file_path, cluster_id in zip(file_paths, labels):
            cluster_to_files.setdefault(int(cluster_id), []).append(file_path)

        cluster_groups: list[ClusterGroup] = []
        for cluster_id in sorted(cluster_to_files.keys()):
            cluster_file_paths = cluster_to_files[cluster_id]
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
            "num_clusters": len(cluster_groups),
            "total_files": len(files),
            "total_tokens": total_tokens,
            "avg_tokens_per_cluster": int(avg_tokens),
        }

        logger.info(
            f"K-means complete: {len(cluster_groups)} clusters, "
            f"avg {int(avg_tokens):,} tokens/cluster"
        )

        return cluster_groups, metadata

    async def cluster_files_hdbscan(
        self,
        files: dict[str, str],
        min_cluster_size: int = 2,
    ) -> tuple[list[ClusterGroup], dict[str, int]]:
        """Cluster files using HDBSCAN for natural semantic grouping.

        HDBSCAN discovers natural clusters based on density, without requiring
        a predetermined number of clusters. Outliers are reassigned to the
        nearest cluster centroid (not dropped).

        Args:
            files: Dictionary mapping file_path -> file_content
            min_cluster_size: Minimum size for HDBSCAN clusters (default: 2)

        Returns:
            Tuple of (cluster_groups, metadata) where metadata contains:
                - num_clusters: Number of clusters after outlier reassignment
                - num_native_clusters: Original HDBSCAN clusters (before outliers)
                - num_outliers: Count of noise points reassigned
                - total_files: Total number of files
                - total_tokens: Total tokens across all files
                - avg_tokens_per_cluster: Average tokens per cluster

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
            f"HDBSCAN clustering {len(files)} files ({total_tokens:,} tokens)"
        )

        # Special case: single file
        if len(files) == 1:
            logger.info("Single file - will produce single cluster")
            cluster_group = ClusterGroup(
                cluster_id=0,
                file_paths=list(files.keys()),
                files_content=files,
                total_tokens=total_tokens,
            )
            metadata = {
                "num_clusters": 1,
                "num_native_clusters": 1,
                "num_outliers": 0,
                "total_files": 1,
                "total_tokens": total_tokens,
                "avg_tokens_per_cluster": total_tokens,
            }
            return [cluster_group], metadata

        # Generate embeddings for each file
        file_paths = list(files.keys())
        file_contents = [files[fp] for fp in file_paths]

        logger.debug(f"Generating embeddings for {len(file_contents)} files")
        embeddings = await self._embedding_provider.embed_batch(file_contents)
        embeddings_array = np.array(embeddings)

        # HDBSCAN clustering
        effective_min_cluster_size = min(min_cluster_size, len(embeddings_array) - 1)
        effective_min_cluster_size = max(2, effective_min_cluster_size)

        logger.debug(
            f"Running HDBSCAN with min_cluster_size={effective_min_cluster_size}"
        )

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=effective_min_cluster_size,
            min_samples=1,
            metric="euclidean",
            cluster_selection_method="eom",
            allow_single_cluster=True,
        )

        try:
            labels = clusterer.fit_predict(embeddings_array)
        except Exception as e:
            logger.warning(f"HDBSCAN clustering failed: {e}, using single cluster")
            labels = np.zeros(len(file_paths), dtype=int)

        # Count native clusters and outliers before reassignment
        unique_labels = set(labels)
        num_native_clusters = len([l for l in unique_labels if l >= 0])
        num_outliers = int(np.sum(labels == -1))

        # Reassign outliers to nearest cluster
        labels = self._reassign_outliers_to_nearest(labels, embeddings_array)

        # Build cluster groups
        cluster_to_files: dict[int, list[str]] = {}
        for file_path, cluster_id in zip(file_paths, labels):
            cluster_to_files.setdefault(int(cluster_id), []).append(file_path)

        cluster_groups: list[ClusterGroup] = []
        for cluster_id in sorted(cluster_to_files.keys()):
            cluster_file_paths = cluster_to_files[cluster_id]
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
            "num_clusters": len(cluster_groups),
            "num_native_clusters": num_native_clusters,
            "num_outliers": num_outliers,
            "total_files": len(files),
            "total_tokens": total_tokens,
            "avg_tokens_per_cluster": int(avg_tokens),
        }

        logger.info(
            f"HDBSCAN complete: {num_native_clusters} native clusters, "
            f"{num_outliers} outliers reassigned, "
            f"{len(cluster_groups)} final clusters"
        )

        return cluster_groups, metadata

    def _reassign_outliers_to_nearest(
        self,
        labels: np.ndarray,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """Reassign outliers (label=-1) to nearest cluster centroid.

        Uses Euclidean distance to find the nearest cluster for each outlier.
        If all points are outliers, assigns them all to a single cluster.

        Args:
            labels: Cluster labels from HDBSCAN (-1 for outliers)
            embeddings: Embedding vectors for each file

        Returns:
            Modified labels array with no -1 values
        """
        outlier_mask = labels == -1
        if not outlier_mask.any():
            return labels

        # Make a copy to avoid modifying the original
        labels = labels.copy()

        valid_labels = set(labels[~outlier_mask])
        if not valid_labels:
            # All points are outliers - create single cluster
            logger.debug("All points are outliers, creating single cluster")
            return np.zeros_like(labels)

        # Compute centroids for each valid cluster
        centroids: dict[int, np.ndarray] = {}
        for label in valid_labels:
            cluster_embeddings = embeddings[labels == label]
            centroids[label] = cluster_embeddings.mean(axis=0)

        # Reassign each outlier to nearest centroid
        outlier_indices = np.where(outlier_mask)[0]
        for i in outlier_indices:
            distances = {
                label: float(np.linalg.norm(embeddings[i] - centroid))
                for label, centroid in centroids.items()
            }
            nearest_label = min(distances, key=distances.get)  # type: ignore[arg-type]
            labels[i] = nearest_label

        logger.debug(f"Reassigned {len(outlier_indices)} outliers to nearest clusters")

        return labels

    async def cluster_files_hdbscan_bounded(
        self,
        files: dict[str, str],
        min_cluster_size: int = 2,
        min_tokens_per_cluster: int = 15_000,
        max_tokens_per_cluster: int = 50_000,
    ) -> tuple[list[ClusterGroup], dict[str, int]]:
        """Cluster files using HDBSCAN with token bounds enforcement.

        Performs HDBSCAN clustering then enforces token bounds by:
        - Splitting clusters exceeding max_tokens_per_cluster using k-means
        - Merging clusters below min_tokens_per_cluster into nearest neighbors

        Args:
            files: Dictionary mapping file_path -> file_content
            min_cluster_size: Minimum size for HDBSCAN clusters (default: 2)
            min_tokens_per_cluster: Minimum tokens per cluster (default: 15,000)
            max_tokens_per_cluster: Maximum tokens per cluster (default: 50,000)

        Returns:
            Tuple of (cluster_groups, metadata) where metadata contains:
                - num_clusters: Final cluster count after bounds enforcement
                - num_native_clusters: HDBSCAN clusters before bounds enforcement
                - num_outliers: Count of noise points from HDBSCAN
                - num_splits: Number of split operations performed
                - num_merges: Number of merge operations performed
                - total_files: Total number of files
                - total_tokens: Total tokens across all files
                - avg_tokens_per_cluster: Average tokens per cluster

        Raises:
            ValueError: If files dict is empty

        Note:
            Single files exceeding max_tokens_per_cluster cannot be split further.
            Such files are returned as-is in their own cluster, potentially
            exceeding the configured bound. A warning is logged when this occurs.
        """
        if not files:
            raise ValueError("Cannot cluster empty files dictionary")

        # Calculate total tokens and per-file tokens
        file_tokens: dict[str, int] = {
            fp: self._llm_provider.estimate_tokens(content)
            for fp, content in files.items()
        }
        total_tokens = sum(file_tokens.values())

        logger.info(
            f"HDBSCAN bounded clustering {len(files)} files ({total_tokens:,} tokens), "
            f"bounds: [{min_tokens_per_cluster:,}, {max_tokens_per_cluster:,}]"
        )

        # Special case: single file
        if len(files) == 1:
            logger.info("Single file - will produce single cluster")
            cluster_group = ClusterGroup(
                cluster_id=0,
                file_paths=list(files.keys()),
                files_content=files,
                total_tokens=total_tokens,
            )
            metadata = {
                "num_clusters": 1,
                "num_native_clusters": 1,
                "num_outliers": 0,
                "num_splits": 0,
                "num_merges": 0,
                "total_files": 1,
                "total_tokens": total_tokens,
                "avg_tokens_per_cluster": total_tokens,
            }
            return [cluster_group], metadata

        # Generate embeddings for each file (once, reused for all operations)
        file_paths = list(files.keys())
        file_contents = [files[fp] for fp in file_paths]

        logger.debug(f"Generating embeddings for {len(file_contents)} files")
        embeddings = await self._embedding_provider.embed_batch(file_contents)
        embeddings_array = np.array(embeddings)

        # Build file_path -> embedding mapping for later operations
        file_embeddings: dict[str, np.ndarray] = {
            fp: embeddings_array[i] for i, fp in enumerate(file_paths)
        }

        # HDBSCAN clustering
        effective_min_cluster_size = min(min_cluster_size, len(embeddings_array) - 1)
        effective_min_cluster_size = max(2, effective_min_cluster_size)

        logger.debug(
            f"Running HDBSCAN with min_cluster_size={effective_min_cluster_size}"
        )

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=effective_min_cluster_size,
            min_samples=1,
            metric="euclidean",
            cluster_selection_method="eom",
            allow_single_cluster=True,
        )

        try:
            labels = clusterer.fit_predict(embeddings_array)
        except Exception as e:
            logger.warning(f"HDBSCAN clustering failed: {e}, using single cluster")
            labels = np.zeros(len(file_paths), dtype=int)

        # Count native clusters and outliers before reassignment
        unique_labels = set(labels)
        num_native_clusters = len([label for label in unique_labels if label >= 0])
        num_outliers = int(np.sum(labels == -1))

        # Reassign outliers to nearest cluster
        labels = self._reassign_outliers_to_nearest(labels, embeddings_array)

        # Build initial cluster groups with file paths and tokens
        cluster_to_files: dict[int, list[str]] = {}
        for file_path, cluster_id in zip(file_paths, labels):
            cluster_to_files.setdefault(int(cluster_id), []).append(file_path)

        # Phase 1: SPLIT oversized clusters (recursive to guarantee bounds)
        num_splits = 0

        def split_cluster_recursively(
            file_paths_to_split: list[str],
        ) -> list[list[str]]:
            """Split files into clusters respecting max_tokens recursively."""
            cluster_tokens = sum(file_tokens[fp] for fp in file_paths_to_split)

            # Base case: fits in budget or single file (can't split further)
            fits_budget = cluster_tokens <= max_tokens_per_cluster
            if not fits_budget and len(file_paths_to_split) == 1:
                logger.warning(
                    f"Single file exceeds max_tokens_per_cluster "
                    f"({cluster_tokens} > {max_tokens_per_cluster}): "
                    f"{file_paths_to_split[0]}"
                )
            if fits_budget or len(file_paths_to_split) <= 1:
                return [file_paths_to_split]

            # K-means split into 2 clusters
            embeddings = np.array(
                [file_embeddings[fp] for fp in file_paths_to_split]
            )
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            split_labels = kmeans.fit_predict(embeddings)

            cluster_0 = [
                fp for fp, lbl in zip(file_paths_to_split, split_labels)
                if lbl == 0
            ]
            cluster_1 = [
                fp for fp, lbl in zip(file_paths_to_split, split_labels)
                if lbl == 1
            ]

            # Guard: k-means may return all files in one cluster (identical embeddings)
            # Use deterministic fallback to guarantee splitting for bounded prompts
            if not cluster_0 or not cluster_1:
                logger.warning(
                    f"K-means could not split {len(file_paths_to_split)} files "
                    "(identical embeddings?), using token-balanced fallback"
                )
                # Deterministic fallback: greedy bin-packing by token count
                # Sort by tokens descending for better balance
                sorted_files = sorted(
                    file_paths_to_split,
                    key=lambda fp: file_tokens[fp],
                    reverse=True,
                )
                bin_0: list[str] = []
                bin_1: list[str] = []
                bin_0_tokens = 0
                bin_1_tokens = 0
                for fp in sorted_files:
                    tokens = file_tokens[fp]
                    if bin_0_tokens <= bin_1_tokens:
                        bin_0.append(fp)
                        bin_0_tokens += tokens
                    else:
                        bin_1.append(fp)
                        bin_1_tokens += tokens
                cluster_0, cluster_1 = bin_0, bin_1

            # Recursively split any oversized subclusters
            result: list[list[str]] = []
            for subcluster in [cluster_0, cluster_1]:
                result.extend(split_cluster_recursively(subcluster))
            return result

        # Apply recursive splitting to all oversized clusters
        new_cluster_to_files: dict[int, list[str]] = {}
        next_cluster_id = 0

        for cluster_id, cluster_file_paths in cluster_to_files.items():
            cluster_tokens = sum(file_tokens[fp] for fp in cluster_file_paths)

            if cluster_tokens > max_tokens_per_cluster and len(cluster_file_paths) > 1:
                logger.debug(
                    f"Splitting cluster {cluster_id} ({cluster_tokens:,} tokens, "
                    f"{len(cluster_file_paths)} files) recursively"
                )
                subclusters = split_cluster_recursively(cluster_file_paths)
                for subcluster in subclusters:
                    new_cluster_to_files[next_cluster_id] = subcluster
                    next_cluster_id += 1
                num_splits += 1
            else:
                # Keep cluster as-is
                new_cluster_to_files[next_cluster_id] = cluster_file_paths
                next_cluster_id += 1

        cluster_to_files = new_cluster_to_files

        # Phase 2: MERGE undersized clusters
        num_merges = 0
        num_unmergeable = 0
        unmergeable_clusters: set[int] = set()

        def get_cluster_tokens(cluster_id: int) -> int:
            return sum(file_tokens[fp] for fp in cluster_to_files[cluster_id])

        def compute_centroid(cluster_id: int) -> np.ndarray:
            cluster_embeddings = np.array(
                [file_embeddings[fp] for fp in cluster_to_files[cluster_id]]
            )
            centroid: np.ndarray = cluster_embeddings.mean(axis=0)
            return centroid

        def find_valid_merge_target(
            smallest_id: int,
            smallest_tokens: int,
            cluster_tokens_map: dict[int, int],
        ) -> int | None:
            """Find nearest cluster that won't exceed max_tokens after merge."""
            smallest_centroid = compute_centroid(smallest_id)

            # Find candidates that won't exceed max_tokens after merge
            candidates: list[tuple[int, float]] = []
            for cid in cluster_tokens_map:
                if cid == smallest_id:
                    continue
                merged_tokens = smallest_tokens + cluster_tokens_map[cid]
                if merged_tokens <= max_tokens_per_cluster:
                    other_centroid = compute_centroid(cid)
                    dist = float(np.linalg.norm(smallest_centroid - other_centroid))
                    candidates.append((cid, dist))

            if not candidates:
                return None  # No valid target exists

            # Return nearest valid target
            candidates.sort(key=lambda x: x[1])
            return candidates[0][0]

        while len(cluster_to_files) > 1:
            # Find smallest cluster (excluding already-marked unmergeable)
            cluster_tokens_map = {
                cid: get_cluster_tokens(cid)
                for cid in cluster_to_files
                if cid not in unmergeable_clusters
            }

            if not cluster_tokens_map:
                break  # All remaining clusters are unmergeable

            smallest_id = min(cluster_tokens_map, key=cluster_tokens_map.get)  # type: ignore[arg-type]
            smallest_tokens = cluster_tokens_map[smallest_id]

            if smallest_tokens >= min_tokens_per_cluster:
                break  # All clusters meet minimum threshold

            # Find nearest cluster that respects max_tokens_per_cluster
            # Need full token map for merge validation
            full_cluster_tokens_map = {
                cid: get_cluster_tokens(cid) for cid in cluster_to_files
            }
            target_id = find_valid_merge_target(
                smallest_id, smallest_tokens, full_cluster_tokens_map
            )

            if target_id is None:
                # No valid target - keep cluster as-is
                logger.warning(
                    f"Cluster {smallest_id} ({smallest_tokens:,} tokens) cannot merge "
                    f"without exceeding max ({max_tokens_per_cluster:,}). Keeping."
                )
                unmergeable_clusters.add(smallest_id)
                num_unmergeable += 1
                continue

            logger.debug(
                f"Merging cluster {smallest_id} ({smallest_tokens:,} tokens) "
                f"into cluster {target_id}"
            )

            # Merge smallest into target
            cluster_to_files[target_id].extend(cluster_to_files[smallest_id])
            del cluster_to_files[smallest_id]
            num_merges += 1

        # Phase 3: Renumber clusters sequentially
        final_cluster_groups: list[ClusterGroup] = []
        for new_id, old_id in enumerate(sorted(cluster_to_files.keys())):
            cluster_file_paths = cluster_to_files[old_id]
            cluster_files_content = {fp: files[fp] for fp in cluster_file_paths}
            cluster_tokens = sum(file_tokens[fp] for fp in cluster_file_paths)

            cluster_group = ClusterGroup(
                cluster_id=new_id,
                file_paths=cluster_file_paths,
                files_content=cluster_files_content,
                total_tokens=cluster_tokens,
            )
            final_cluster_groups.append(cluster_group)

            logger.debug(
                f"Cluster {new_id}: {len(cluster_file_paths)} files, "
                f"{cluster_tokens:,} tokens"
            )

        avg_tokens = (
            total_tokens / len(final_cluster_groups) if final_cluster_groups else 0
        )
        metadata = {
            "num_clusters": len(final_cluster_groups),
            "num_native_clusters": num_native_clusters,
            "num_outliers": num_outliers,
            "num_splits": num_splits,
            "num_merges": num_merges,
            "num_unmergeable": num_unmergeable,
            "total_files": len(files),
            "total_tokens": total_tokens,
            "avg_tokens_per_cluster": int(avg_tokens),
        }

        logger.info(
            f"HDBSCAN bounded complete: {num_native_clusters} native clusters, "
            f"{num_outliers} outliers reassigned, {num_splits} splits, "
            f"{num_merges} merges, {num_unmergeable} unmergeable, "
            f"{len(final_cluster_groups)} final clusters"
        )

        return final_cluster_groups, metadata
