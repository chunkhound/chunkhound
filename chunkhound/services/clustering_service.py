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
        embeddings = await self._embedding_provider.embed(file_contents)
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
        embeddings = await self._embedding_provider.embed(file_contents)
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
