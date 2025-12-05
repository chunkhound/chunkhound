"""Evaluation helpers for ChunkHound tools.

This package contains shared building blocks for evaluation harnesses:

- ``language_samples``: synthetic corpora generators per language
- ``metrics``: retrieval metric data structures and aggregations
- ``search``: orchestration and CLI entry point for search evaluation
"""

from __future__ import annotations

from .language_samples import QueryDefinition  # noqa: F401
from .metrics import (  # noqa: F401
    AggregateMetrics,
    EvalResult,
    QueryMetrics,
    aggregate_metrics,
    build_json_payload,
    format_human_summary,
)

__all__ = [
    "AggregateMetrics",
    "EvalResult",
    "QueryDefinition",
    "QueryMetrics",
    "aggregate_metrics",
    "build_json_payload",
    "format_human_summary",
]

