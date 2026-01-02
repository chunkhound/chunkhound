"""Elbow-based filtering utility for exploration strategies.

This module provides a unified elbow-based filtering function that can be used
by both BFS and WideCoverage strategies to filter chunks before returning results.
"""

from typing import Any

from loguru import logger

from chunkhound.services.research.shared.elbow_detection import find_elbow_kneedle


def get_unified_score(chunk: dict[str, Any]) -> float:
    """Get score for elbow detection, preferring rerank_score over score.

    This provides a unified scoring approach for chunks from different
    exploration strategies:
    - BFS uses "score" from semantic search
    - WideCoverage uses "rerank_score" from reranking

    Args:
        chunk: Chunk dictionary with score fields

    Returns:
        Float score value, preferring rerank_score if available
    """
    rerank = chunk.get("rerank_score")
    if rerank is not None:
        return float(rerank)
    score = chunk.get("score")
    if score is not None:
        return float(score)
    return 0.0


def filter_chunks_by_elbow(
    chunks: list[dict[str, Any]],
    score_key: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Filter chunks using elbow detection on score distribution.

    Uses the Kneedle algorithm to find the natural breakpoint in chunk scores,
    keeping only chunks at or above the elbow point.

    Args:
        chunks: List of chunk dicts with scores
        score_key: Key to extract score from chunk dict, or None to use
            get_unified_score() which prefers rerank_score over score.
            - BFS strategy uses "score"
            - Wide coverage uses "rerank_score"
            - Parallel strategy uses None (unified scoring)

    Returns:
        Tuple of (filtered_chunks, filter_stats):
            - filtered_chunks: Chunks at or above elbow point (sorted by score desc)
            - filter_stats: Dict with filtering metadata:
                - method: "elbow" | "passthrough"
                - reason: Why this method was used
                - original_count: Input chunk count
                - filtered_count: Output chunk count
                - cutoff_score: Score at elbow point (if elbow found)

    Fallback behavior:
        - < 3 chunks: Return all (elbow needs 3+ points)
        - No elbow detected: Return all (uniform/gradual scores)
        - Empty input: Return empty list
    """
    # Handle empty input
    if not chunks:
        return [], {
            "method": "passthrough",
            "reason": "empty_input",
            "original_count": 0,
            "filtered_count": 0,
        }

    # Handle too few chunks for elbow detection
    if len(chunks) < 3:
        return chunks, {
            "method": "passthrough",
            "reason": "too_few_chunks",
            "original_count": len(chunks),
            "filtered_count": len(chunks),
        }

    # Sort chunks by score (highest first)
    # Use get_unified_score when score_key is None for unified scoring
    if score_key is None:
        sorted_chunks = sorted(chunks, key=get_unified_score, reverse=True)
        scores = [get_unified_score(c) for c in sorted_chunks]
        score_key_display = "unified"
    else:
        sorted_chunks = sorted(
            chunks, key=lambda c: c.get(score_key, 0.0), reverse=True
        )
        scores = [c.get(score_key, 0.0) for c in sorted_chunks]
        score_key_display = score_key

    # Find elbow point (0-based index)
    elbow_idx = find_elbow_kneedle(scores)

    if elbow_idx is None:
        # No clear elbow - return all chunks sorted
        logger.debug(
            f"Elbow filter: No elbow detected in {len(chunks)} chunks, keeping all"
        )
        return sorted_chunks, {
            "method": "passthrough",
            "reason": "no_elbow_detected",
            "original_count": len(chunks),
            "filtered_count": len(chunks),
        }

    # Filter to chunks at or above elbow (include elbow point)
    filtered_chunks = sorted_chunks[: elbow_idx + 1]
    cutoff_score = scores[elbow_idx]

    logger.info(
        f"Elbow filter: keeping {len(filtered_chunks)}/{len(chunks)} chunks "
        f"(cutoff score: {cutoff_score:.3f}, score_key: {score_key_display})"
    )

    return filtered_chunks, {
        "method": "elbow",
        "reason": "elbow_detected",
        "original_count": len(chunks),
        "filtered_count": len(filtered_chunks),
        "cutoff_score": cutoff_score,
        "elbow_index": elbow_idx,
    }
