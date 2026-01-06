"""USearch wrapper module for HNSW index operations.

Encapsulates USearch index creation, viewing, clustering, multi-search,
quality measurement, and medoid computation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from usearch.eval import self_recall
from usearch.index import Index, Indexes, MetricKind, ScalarKind


@dataclass
class SearchResult:
    """Single search result from USearch."""

    key: int
    distance: float


@dataclass
class Clustering:
    """Clustering result from hierarchical k-means."""

    labels: np.ndarray
    centroids: np.ndarray
    cluster_sizes: np.ndarray


# Default HNSW parameters per spec
DEFAULT_CONNECTIVITY = 16
DEFAULT_EXPANSION_ADD = 128
DEFAULT_EXPANSION_SEARCH = 64


def _get_scalar_kind(quantization: str) -> ScalarKind:
    """Convert quantization string to USearch ScalarKind."""
    mapping = {
        "i8": ScalarKind.I8,
        "f16": ScalarKind.F16,
        "f32": ScalarKind.F32,
        "f64": ScalarKind.F64,
    }
    if quantization not in mapping:
        raise ValueError(
            f"Unsupported quantization '{quantization}'. "
            f"Supported: {list(mapping.keys())}"
        )
    return mapping[quantization]


def create(
    dims: int,
    quantization: Literal["i8", "f16", "f32", "f64"] = "i8",
    connectivity: int = DEFAULT_CONNECTIVITY,
    expansion_add: int = DEFAULT_EXPANSION_ADD,
    expansion_search: int = DEFAULT_EXPANSION_SEARCH,
) -> Index:
    """Create a new USearch HNSW index.

    Args:
        dims: Number of dimensions for vectors
        quantization: Storage type for vectors (i8, f16, f32, f64)
        connectivity: Maximum connections per node in HNSW graph
        expansion_add: Expansion factor during index construction
        expansion_search: Expansion factor during search

    Returns:
        Newly created USearch Index
    """
    return Index(
        ndim=dims,
        metric=MetricKind.Cos,
        dtype=_get_scalar_kind(quantization),
        connectivity=connectivity,
        expansion_add=expansion_add,
        expansion_search=expansion_search,
    )


def open_view(path: Path | str) -> Index:
    """Open an existing USearch index as a memory-mapped view.

    This loads the index without copying it entirely into RAM,
    enabling efficient access to large indexes.

    Args:
        path: Path to the .usearch index file

    Returns:
        Memory-mapped view of the index

    Raises:
        FileNotFoundError: If index file does not exist
        RuntimeError: If index cannot be loaded
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Index file not found: {path}")
    index = Index.restore(str(path), view=True)
    if index is None:
        raise RuntimeError(f"Failed to load index from {path}")
    return index


def open_writable(path: Path | str) -> Index:
    """Open an existing USearch index for modification.

    Unlike open_view(), this loads the index fully into RAM,
    allowing add/remove operations.

    Args:
        path: Path to the .usearch index file

    Returns:
        Writable USearch Index

    Raises:
        FileNotFoundError: If index file does not exist
        RuntimeError: If index cannot be loaded
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Index file not found: {path}")
    index = Index.restore(str(path), view=False)
    if index is None:
        raise RuntimeError(f"Failed to load index from {path}")
    return index


def cluster(
    index: Index,
    min_count: int,
    max_count: int,
) -> Clustering:
    """Cluster index vectors using USearch's native HNSW-based clustering.

    Uses USearch's native index.cluster() method which leverages the HNSW graph
    structure for efficient clustering. This is memory-mapped and significantly
    faster than loading all vectors into RAM for custom k-means.

    Falls back to custom k-means for small indexes where native clustering
    fails (USearch requires sufficient graph connectivity for clustering).

    Args:
        index: USearch index containing vectors to cluster
        min_count: Minimum vectors per cluster
        max_count: Maximum vectors per cluster

    Returns:
        Clustering result with labels, centroids, and cluster sizes

    Raises:
        ValueError: If index is empty or constraints cannot be satisfied
    """
    if len(index) == 0:
        raise ValueError("Cannot cluster empty index")

    n_vectors = len(index)

    # Try native USearch clustering first (uses HNSW graph structure)
    try:
        native_clustering = index.cluster(
            min_count=min_count, max_count=max_count, threads=4
        )
        centroid_keys, cluster_sizes = native_clustering.centroids_popularity

        # Build key-to-position mapping (keys are embedding IDs, not 0..N-1)
        keys = list(index.keys)
        key_to_pos = {int(k): i for i, k in enumerate(keys)}

        # Build labels array from native clustering
        labels = np.zeros(n_vectors, dtype=np.int64)
        for cluster_idx, centroid_key in enumerate(centroid_keys):
            members = native_clustering.members_of(centroid_key)
            for member in members:
                pos = key_to_pos.get(int(member))
                if pos is not None:
                    labels[pos] = cluster_idx

        # Get centroid vectors (native clustering uses medoids - actual data points)
        centroids = np.zeros((len(centroid_keys), index.ndim), dtype=np.float32)
        for i, key in enumerate(centroid_keys):
            centroids[i] = index[int(key)]

        return Clustering(
            labels=labels,
            centroids=centroids,
            cluster_sizes=np.array(cluster_sizes, dtype=np.int64),
        )

    except RuntimeError:
        # Native clustering fails for small indexes ("Index too small to cluster!")
        # Fall back to custom k-means implementation
        return _kmeans_cluster(index, min_count, max_count)


def _kmeans_cluster(
    index: Index,
    min_count: int,
    max_count: int,
) -> Clustering:
    """Fallback k-means clustering for small indexes.

    Used when USearch native clustering fails due to insufficient index size
    (native clustering requires adequate HNSW graph connectivity).
    """
    n_vectors = len(index)

    # Determine number of clusters based on constraints
    # Use ceiling division to ensure clusters don't exceed max_count
    # Minimum of 2 clusters required for splitting capability
    n_clusters = (n_vectors + max_count - 1) // max_count
    n_clusters = min(n_clusters, n_vectors // max(min_count, 1))
    n_clusters = max(2, n_clusters)

    # Load vectors into RAM for k-means
    vectors = np.zeros((n_vectors, index.ndim), dtype=np.float32)
    for i in range(n_vectors):
        vectors[i] = index[i]

    # K-means++ initialization and clustering
    centroids = _kmeans_plusplus_init(vectors, n_clusters)
    labels, centroids = _kmeans(vectors, centroids, max_iter=100)
    cluster_sizes = np.bincount(labels, minlength=n_clusters)

    return Clustering(
        labels=labels,
        centroids=centroids,
        cluster_sizes=cluster_sizes,
    )


def _kmeans_plusplus_init(vectors: np.ndarray, k: int) -> np.ndarray:
    """Initialize centroids using k-means++ algorithm."""
    n_samples = len(vectors)
    centroids = np.zeros((k, vectors.shape[1]), dtype=np.float32)

    # First centroid: random
    idx = np.random.randint(n_samples)
    centroids[0] = vectors[idx]

    # Remaining centroids: proportional to squared distance
    for i in range(1, k):
        # Compute distances to nearest centroid
        dists = np.min(
            np.linalg.norm(vectors[:, np.newaxis] - centroids[:i], axis=2), axis=1
        )
        dists_sq = dists**2

        # Sample proportional to squared distance
        sum_sq = dists_sq.sum()
        if sum_sq == 0:
            # All points equidistant from centroids - uniform random selection
            probs = np.ones(n_samples) / n_samples
        else:
            probs = dists_sq / sum_sq
        idx = np.random.choice(n_samples, p=probs)
        centroids[i] = vectors[idx]

    return centroids


def _kmeans(
    vectors: np.ndarray, centroids: np.ndarray, max_iter: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Run k-means clustering."""
    for _ in range(max_iter):
        # Assign points to nearest centroid
        dists = np.linalg.norm(vectors[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(dists, axis=1)

        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for k in range(len(centroids)):
            mask = labels == k
            if mask.sum() > 0:
                new_centroids[k] = vectors[mask].mean(axis=0)
            else:
                # Empty cluster: reinitialize to random point
                new_centroids[k] = vectors[np.random.randint(len(vectors))]

        # Check convergence
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return labels, centroids


def multi_search(
    paths: list[Path | str],
    query: np.ndarray,
    k: int,
) -> list[SearchResult]:
    """Search across multiple USearch indexes in parallel.

    Uses USearch's Indexes class for efficient multi-index search.

    Args:
        paths: List of paths to .usearch index files
        query: Query vector
        k: Number of nearest neighbors to return

    Returns:
        List of SearchResult sorted by distance (ascending)
    """
    if not paths:
        return []

    # Open indexes as views for efficient access
    # Note: usearch stubs incorrectly expect PathLike, but implementation needs str
    path_strings = [str(p) for p in paths]
    multi_index = Indexes(paths=path_strings, view=True, threads=4)  # type: ignore[arg-type]

    # Search across all indexes
    matches = multi_index.search(query, k)

    # Convert to SearchResult list
    results = []
    for i in range(len(matches.keys)):
        results.append(
            SearchResult(
                key=int(matches.keys[i]),
                distance=float(matches.distances[i]),
            )
        )

    return sorted(results, key=lambda r: r.distance)


def measure_quality(
    path: Path | str,
    sample_size: int = 1000,
) -> float:
    """Measure search quality via self-recall (can each vector find itself?).

    Uses USearch's native self_recall which is optimized for mmap indexes.

    Args:
        path: Path to .usearch index file
        sample_size: Number of vectors to sample for quality check

    Returns:
        Recall as float between 0.0 and 1.0
    """
    index = open_view(path)
    n_vectors = len(index)

    if n_vectors == 0:
        return 1.0  # Empty index is trivially perfect

    sample = min(sample_size, n_vectors)
    stats = self_recall(index, sample=sample, count=10)

    if stats.count_queries == 0:
        return 1.0

    return stats.count_matches / stats.count_queries


def get_medoid(index: Index) -> tuple[int, np.ndarray]:
    """Find the medoid of an index (most central vector).

    The medoid is the vector with minimum average distance to all other vectors.

    Args:
        index: USearch index

    Returns:
        Tuple of (medoid_key, medoid_vector)

    Raises:
        ValueError: If index is empty
    """
    n_vectors = len(index)

    if n_vectors == 0:
        raise ValueError("Cannot find medoid of empty index")

    # Get all keys and their vectors (use actual keys, not positional indices)
    keys = list(index.keys)
    vectors = np.zeros((n_vectors, index.ndim), dtype=np.float32)
    for i, key in enumerate(keys):
        vectors[i] = index[key]

    # Compute pairwise distances and find medoid
    # For large indexes, use sampling to approximate
    if n_vectors > 10000:
        # Sample-based approximation
        sample_size = min(1000, n_vectors)
        sample_indices = np.random.choice(n_vectors, size=sample_size, replace=False)
        sample_vectors = vectors[sample_indices]

        # Compute average distance to sample for all vectors
        avg_distances = np.zeros(n_vectors)
        for i, vec in enumerate(vectors):
            distances = np.linalg.norm(sample_vectors - vec, axis=1)
            avg_distances[i] = distances.mean()
    else:
        # Exact computation for small indexes
        avg_distances = np.zeros(n_vectors)
        for i in range(n_vectors):
            distances = np.linalg.norm(vectors - vectors[i], axis=1)
            avg_distances[i] = distances.mean()

    medoid_idx = int(np.argmin(avg_distances))
    return int(keys[medoid_idx]), vectors[medoid_idx]
