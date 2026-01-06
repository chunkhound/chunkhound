"""Synthetic embedding generator for offline testing.

Provides deterministic, reproducible embedding vectors for testing
without external API dependencies. Supports:
- Hash-seeded deterministic generation (same text -> same vector)
- Clustered vector generation with controlled separation
- Orthonormal vector generation for exact recall testing
- Brute-force ground truth search for validation
"""

import numpy as np


class SyntheticEmbeddingGenerator:
    """Generate synthetic embeddings for offline testing.

    Features:
    - Deterministic hash-seeded vectors: same text produces same vector
    - Clustered vectors with controllable separation for clustering tests
    - Orthonormal vectors via QR decomposition for recall validation
    """

    def __init__(self, dims: int = 1536, seed: int = 42):
        """Initialize generator.

        Args:
            dims: Embedding dimensions (default: 1536, matches OpenAI)
            seed: Random seed for reproducibility
        """
        self.dims = dims
        self.rng = np.random.default_rng(seed)

    def generate_hash_seeded(self, text: str) -> np.ndarray:
        """Generate deterministic vector from text hash.

        Same text always produces same vector across runs.

        Args:
            text: Input text to embed

        Returns:
            Normalized unit vector of shape (dims,)
        """
        seed = hash(text) % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.random(self.dims)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def generate_clustered(
        self,
        num_clusters: int,
        per_cluster: int,
        separation: float = 0.5,
    ) -> list[tuple[np.ndarray, int]]:
        """Generate vectors in distinct clusters.

        Creates C clusters with N points each, separated by minimum distance.

        Args:
            num_clusters: Number of clusters (C)
            per_cluster: Vectors per cluster (N)
            separation: Minimum cosine distance between cluster centroids

        Returns:
            List of (vector, cluster_id) pairs
        """
        # Generate cluster centroids with minimum separation
        centroids = self._generate_separated_centroids(num_clusters, separation)

        results: list[tuple[np.ndarray, int]] = []

        for cluster_id, centroid in enumerate(centroids):
            for _ in range(per_cluster):
                # Add small noise to centroid
                noise = self.rng.normal(0, 0.1, self.dims)
                vec = centroid + noise
                # Normalize to unit length
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                results.append((vec, cluster_id))

        return results

    def _generate_separated_centroids(
        self, num_clusters: int, separation: float
    ) -> list[np.ndarray]:
        """Generate cluster centroids with minimum separation.

        Uses rejection sampling to ensure centroids have at least
        the specified cosine distance from each other.

        Args:
            num_clusters: Number of centroids to generate
            separation: Minimum cosine distance (1 - cosine_similarity)

        Returns:
            List of normalized centroid vectors
        """
        centroids: list[np.ndarray] = []
        max_attempts = 1000

        for _ in range(num_clusters):
            for attempt in range(max_attempts):
                candidate = self.rng.random(self.dims)
                candidate = candidate / np.linalg.norm(candidate)

                # Check separation from existing centroids
                is_separated = True
                for existing in centroids:
                    cosine_sim = float(np.dot(candidate, existing))
                    distance = 1.0 - cosine_sim
                    if distance < separation:
                        is_separated = False
                        break

                if is_separated:
                    centroids.append(candidate)
                    break
            else:
                # Fallback: accept suboptimal centroid after max attempts
                candidate = self.rng.random(self.dims)
                centroids.append(candidate / np.linalg.norm(candidate))

        return centroids

    def generate_orthogonal(self, k: int) -> list[np.ndarray]:
        """Generate k orthonormal vectors using QR decomposition.

        Useful for testing recall where exact neighbors are mathematically known.
        For orthonormal vectors, each vector's nearest neighbor is well-defined.

        Args:
            k: Number of orthonormal vectors (must be <= dims)

        Returns:
            List of k orthonormal vectors

        Raises:
            ValueError: If k > dims
        """
        if k > self.dims:
            msg = f"Cannot generate {k} orthonormal vectors in {self.dims} dimensions"
            raise ValueError(msg)

        # Generate random matrix and apply QR decomposition
        random_matrix = self.rng.random((self.dims, k))
        q, _ = np.linalg.qr(random_matrix)

        # Extract k orthonormal columns
        return [q[:, i].copy() for i in range(k)]


def brute_force_search(
    query: np.ndarray,
    vectors: list[np.ndarray],
    k: int,
) -> list[int]:
    """Compute exact k-nearest neighbors via brute force.

    Ground truth for validating approximate search algorithms.
    Uses cosine distance (1 - cosine_similarity).

    Args:
        query: Query vector
        vectors: List of candidate vectors
        k: Number of nearest neighbors to return

    Returns:
        Indices of top k nearest vectors, sorted by distance (ascending)
    """
    if not vectors:
        return []

    # Compute cosine distances
    distances: list[tuple[float, int]] = []
    query_norm = np.linalg.norm(query)

    for idx, vec in enumerate(vectors):
        vec_norm = np.linalg.norm(vec)
        if query_norm > 0 and vec_norm > 0:
            cosine_sim = float(np.dot(query, vec) / (query_norm * vec_norm))
        else:
            cosine_sim = 0.0
        distance = 1.0 - cosine_sim
        distances.append((distance, idx))

    # Sort by distance ascending
    distances.sort(key=lambda x: x[0])

    # Return top k indices
    return [idx for _, idx in distances[:k]]
