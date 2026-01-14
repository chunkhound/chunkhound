"""Shared chunk deduplication utilities for research services.

This module provides common chunk deduplication logic used across multiple
research phases (Phase 1.5 depth exploration, Phase 2 gap detection, Phase 3
synthesis). The pattern is: deduplicate by chunk_id, keeping highest rerank_score.
"""

from loguru import logger

# Import from core utils (canonical location) and re-export for backwards compatibility
from chunkhound.core.utils.chunk_utils import get_chunk_id


def deduplicate_chunks(
    chunk_lists: list[list[dict]],
    score_field: str = "rerank_score",
    default_score: float = 0.0,
    log_prefix: str = "Global dedup",
) -> list[dict]:
    """Deduplicate chunks across multiple lists.

    This is used at SYNC POINTS after parallel operations complete
    (e.g., after all gap fills, after all exploration queries).
    Conflict resolution: keeps chunk with highest score.

    Args:
        chunk_lists: List of chunk lists to deduplicate
        score_field: Field name containing the score (default: rerank_score)
        default_score: Default score if field is missing (default: 0.0)
        log_prefix: Prefix for debug log message

    Returns:
        Deduplicated list of chunks (highest score wins)
    """
    chunk_map: dict[int | str, dict] = {}

    for chunk_list in chunk_lists:
        for chunk in chunk_list:
            chunk_id = get_chunk_id(chunk)
            if not chunk_id:
                logger.warning(f"Chunk missing ID: {chunk.get('file_path')}")
                continue

            existing = chunk_map.get(chunk_id)
            if existing is None:
                chunk_map[chunk_id] = chunk
            else:
                existing_score = existing.get(score_field, default_score)
                new_score = chunk.get(score_field, default_score)
                if new_score > existing_score:
                    chunk_map[chunk_id] = chunk

    deduplicated = list(chunk_map.values())
    total_input = sum(len(cl) for cl in chunk_lists)
    logger.debug(f"{log_prefix}: {total_input} → {len(deduplicated)} unique")

    return deduplicated


def merge_chunk_lists(
    base_chunks: list[dict],
    new_chunks: list[dict],
    score_field: str = "rerank_score",
    default_score: float = 0.0,
    log_prefix: str = "Merge",
) -> list[dict]:
    """Merge two chunk lists with deduplication.

    Base chunks are added first, new chunks may overwrite if higher score.
    This is used for merging Phase N coverage with Phase N+1 results.

    Args:
        base_chunks: Primary chunk list (e.g., Phase 1 coverage)
        new_chunks: Secondary chunk list (e.g., gap-filled chunks)
        score_field: Field name containing the score
        default_score: Default score if field is missing
        log_prefix: Prefix for debug log message

    Returns:
        Merged and deduplicated list (highest score wins)
    """
    chunk_map: dict[int | str, dict] = {}

    # Add base chunks first
    for chunk in base_chunks:
        chunk_id = get_chunk_id(chunk)
        if chunk_id:
            chunk_map[chunk_id] = chunk

    # Add new chunks (may overwrite if higher score)
    for chunk in new_chunks:
        chunk_id = get_chunk_id(chunk)
        if not chunk_id:
            continue

        existing = chunk_map.get(chunk_id)
        if existing is None:
            chunk_map[chunk_id] = chunk
        else:
            existing_score = existing.get(score_field, default_score)
            new_score = chunk.get(score_field, default_score)
            if new_score > existing_score:
                chunk_map[chunk_id] = chunk

    merged = list(chunk_map.values())
    logger.debug(
        f"{log_prefix}: {len(base_chunks)} + {len(new_chunks)} → {len(merged)} total"
    )

    return merged
