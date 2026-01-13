"""USearch wrapper module for HNSW index operations.

Encapsulates USearch index creation, viewing, multi-search,
quality measurement, and medoid computation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
from usearch.eval import self_recall
from usearch.index import Index, Indexes, MetricKind, ScalarKind


@dataclass
class SearchResult:
    """Single search result from USearch."""

    key: int
    distance: float


# Default HNSW parameters per spec
DEFAULT_CONNECTIVITY = 16
DEFAULT_EXPANSION_ADD = 128
DEFAULT_EXPANSION_SEARCH = 64


def _prepare_query_for_cosine(query: np.ndarray) -> np.ndarray:
    """Prepare query vector for cosine similarity search on quantized indexes.

    For i8 quantization with cosine metric, usearch internally:
    1. Normalizes vectors to unit length
    2. Scales to [-127, 127] range

    The Indexes (multi-index) class may not apply this transformation
    consistently, so we pre-normalize and ensure proper memory layout.

    Args:
        query: Query vector (any float dtype)

    Returns:
        C-contiguous float32 array, unit normalized
    """
    # Ensure C-contiguous float32 (required by usearch SIMD)
    query = np.ascontiguousarray(query, dtype=np.float32)

    # Unit normalize for cosine metric alignment
    norm = np.linalg.norm(query)
    if norm > 0:
        query = query / norm

    return query


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


def multi_search(
    paths: list[Path | str],
    query: np.ndarray,
    k: int,
) -> list[SearchResult]:
    """Search across multiple USearch indexes in parallel.

    Uses USearch's Indexes class for efficient multi-index search.

    Args:
        paths: List of paths to .usearch index files
        query: Query vector (will be normalized for cosine similarity)
        k: Number of nearest neighbors to return

    Returns:
        List of SearchResult sorted by distance (ascending)
    """
    if not paths:
        return []

    # Prepare query for quantized cosine search
    query = _prepare_query_for_cosine(query)

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
    """Find approximate medoid via sample-mean nearest neighbor search.

    Uses USearch's efficient key slicing and HNSW search to find an
    approximate medoid in O(sample_size + log n) time, avoiding the
    O(n) iteration over all keys that blocks on large indexes.

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

    # Sample vectors efficiently - keys[:n] uses get_keys_in_slice() which is O(n)
    # not the O(n) iterator that calls get_key_at_offset() for each key
    sample_size = min(100, n_vectors)
    sample_keys = cast(np.ndarray, index.keys[:sample_size])

    # Get sample vectors and compute mean
    sample_vecs = np.array([index[k] for k in sample_keys], dtype=np.float32)
    mean_vec = sample_vecs.mean(axis=0)

    # Normalize mean vector for cosine search on quantized index
    mean_vec = _prepare_query_for_cosine(mean_vec)

    # Search for nearest actual vector to mean - O(log n) via HNSW
    results = index.search(mean_vec, count=1)
    medoid_key = int(results.keys[0])
    medoid_vec = cast(np.ndarray, index[medoid_key])

    return medoid_key, medoid_vec
