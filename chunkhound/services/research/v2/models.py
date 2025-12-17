"""Data models for Coverage-First Research (v2).

This module contains data structures for the v2 coverage-first research algorithm,
which uses complete coverage retrieval with gap filling instead of BFS exploration.

Key concepts:
- GapCandidate: Potential gap identified by LLM during analysis
- UnifiedGap: Merged gap from multiple shards with voting
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GapCandidate:
    """Gap candidate identified by LLM during shard analysis.

    Represents a potential missing piece of coverage identified by analyzing
    a subset (shard) of the retrieved chunks.
    """

    query: str  # Search query to fill this gap
    rationale: str  # Why this gap is important
    confidence: float  # Confidence score (0.0-1.0)
    source_shard: int  # Which shard identified this gap


@dataclass
class UnifiedGap:
    """Unified gap after clustering and merging similar gaps.

    Represents multiple gap candidates from different shards that were
    determined to be semantically similar and merged into one query.
    """

    query: str  # Merged/refined query
    sources: list[GapCandidate]  # Original gap candidates
    vote_count: int  # Number of shards that found this gap
    avg_confidence: float = 0.0  # Average confidence across sources
    score: float = 0.0  # Combined score for gap selection
