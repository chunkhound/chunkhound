"""Elbow detection utilities for threshold computation.

Implements the Kneedle algorithm (Satopaa et al. 2011) for finding elbow points
in score curves. Used for adaptive threshold computation in research phases.
"""

import numpy as np
from loguru import logger


def find_elbow_kneedle(sorted_scores: list[float]) -> int | None:
    """Find elbow point in score curve using simplified Kneedle algorithm.

    Implementation based on Kneedle algorithm (Satopaa et al. 2011):
    1. Normalize scores to [0,1]
    2. Draw line from first to last point
    3. Find point with maximum perpendicular distance to line
    4. That's the elbow/knee point

    Args:
        sorted_scores: Scores sorted DESCENDING (highest to lowest)

    Returns:
        Index of elbow point (0-based array index), or None if no clear elbow detected.
        Return value can be used to threshold: scores[:elbow_idx+1] are above elbow.

    Examples:
        >>> scores = [0.95, 0.92, 0.88, 0.45, 0.42, 0.40]  # Clear drop at index 2
        >>> find_elbow_kneedle(scores)
        2  # Select first 3 items (indices 0, 1, 2)

        >>> scores = [0.5, 0.5, 0.5, 0.5]  # All identical
        >>> find_elbow_kneedle(scores)
        None  # No elbow

        >>> scores = [0.9, 0.8]  # Too few points
        >>> find_elbow_kneedle(scores)
        None  # Need at least 3 points
    """
    if len(sorted_scores) < 3:
        logger.debug("Kneedle: Too few points (<3), cannot detect elbow")
        return None  # Need at least 3 points for elbow

    # Extract scores
    scores = np.array(sorted_scores)

    # Normalize scores to [0, 1]
    min_score = scores.min()
    max_score = scores.max()
    if max_score == min_score:
        logger.debug("Kneedle: All scores identical, no elbow")
        return None  # All scores identical, no elbow

    normalized_scores = (scores - min_score) / (max_score - min_score)

    # X-axis: normalized positions [0, 1]
    x = np.linspace(0, 1, len(normalized_scores))

    # Draw line from first point to last point
    # Line equation: y = mx + b
    x1, y1 = x[0], normalized_scores[0]
    x2, y2 = x[-1], normalized_scores[-1]

    # Handle vertical line case (shouldn't happen with normalized x)
    if x2 == x1:
        logger.debug("Kneedle: Vertical line case, no elbow")
        return None

    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1

    # Compute perpendicular distance from each point to line
    # Formula: |mx - y + b| / sqrt(m^2 + 1)
    numerator = np.abs(m * x - normalized_scores + b)
    denominator = np.sqrt(m**2 + 1)
    distances = numerator / denominator

    # Find point with maximum distance (that's the elbow)
    elbow_idx = int(np.argmax(distances))

    # Validate elbow is significant (distance > 1% of normalized range)
    if distances[elbow_idx] < 0.01:
        logger.debug(
            f"Kneedle: Elbow not significant (distance={distances[elbow_idx]:.4f} < 0.01)"
        )
        return None  # Elbow not significant enough

    logger.debug(
        f"Kneedle: Found elbow at index {elbow_idx} "
        f"(distance={distances[elbow_idx]:.4f}, score={sorted_scores[elbow_idx]:.3f})"
    )

    # Return 0-based index (for array slicing: scores[:elbow_idx+1])
    return elbow_idx


def compute_elbow_threshold(chunks_or_scores: list[dict] | list[float]) -> float:
    """Compute elbow threshold from chunks or scores using Kneedle algorithm.

    Uses the Kneedle algorithm (Satopaa et al. 2011) to detect the elbow point
    in the score distribution. Falls back to median if Kneedle fails to find
    a significant elbow.

    Args:
        chunks_or_scores: Either:
            - List of chunks (dicts with 'rerank_score' key)
            - List of raw float scores

    Returns:
        Threshold value (score at elbow point, or median if no elbow)

    Examples:
        >>> chunks = [{'rerank_score': 0.95}, {'rerank_score': 0.88}]
        >>> compute_elbow_threshold(chunks)
        0.88

        >>> scores = [0.95, 0.88, 0.45, 0.42]
        >>> compute_elbow_threshold(scores)
        0.45
    """
    # Handle empty input
    if not chunks_or_scores:
        return 0.5  # Default threshold

    # Extract scores from chunks or use raw scores
    if isinstance(chunks_or_scores[0], dict):
        # Type narrowing: if first element is dict, all are dicts
        chunk_list: list[dict] = chunks_or_scores  # type: ignore[assignment]
        scores = [c.get("rerank_score", 0.0) for c in chunk_list]
    else:
        # Type narrowing: if first element is not dict, all are floats
        scores = list(chunks_or_scores)

    if not scores:
        return 0.5

    sorted_scores = sorted(scores, reverse=True)

    # Try Kneedle algorithm first
    elbow_idx = find_elbow_kneedle(sorted_scores)
    if elbow_idx is not None and elbow_idx < len(sorted_scores):
        threshold = float(sorted_scores[elbow_idx])
        logger.debug(
            f"Elbow threshold: {threshold:.3f} (Kneedle at index {elbow_idx} "
            f"of {len(scores)} scores)"
        )
        return threshold

    # Fallback to median if Kneedle fails
    median_idx = len(sorted_scores) // 2
    threshold = float(sorted_scores[median_idx])
    logger.debug(
        f"Elbow threshold: {threshold:.3f} (median fallback, "
        f"Kneedle found no significant elbow in {len(scores)} scores)"
    )
    return threshold
