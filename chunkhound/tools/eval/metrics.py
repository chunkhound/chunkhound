"""Retrieval metrics for search evaluation tools.

This module defines reusable data structures and aggregation helpers for:

- Per-query metrics (precision/recall@k, hit counts)
- Aggregated metrics (latency statistics, hit-rate, nDCG@K, MRR)
- Human-readable and JSON summaries of evaluation runs
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any

from chunkhound.core.types.common import Language


@dataclass
class QueryMetrics:
    """Per-query evaluation metrics for multiple k values."""

    query_id: str
    language: Language
    pattern: str
    search_type: str
    latency_ms: float
    total_results: int
    first_relevant_rank: int | None
    metrics_by_k: dict[int, dict[str, float]]


@dataclass
class AggregateMetrics:
    """Aggregated metrics across queries."""

    metrics_by_k: dict[int, dict[str, float]]
    latency_stats_ms: dict[str, float]
    mrr: float = 0.0


@dataclass
class EvalResult:
    """Complete evaluation result."""

    mode: str
    search_mode: str
    languages: list[Language]
    ks: list[int]
    per_query: list[QueryMetrics] = field(default_factory=list)
    per_language: dict[str, AggregateMetrics] = field(default_factory=dict)
    global_metrics: AggregateMetrics | None = None


def aggregate_metrics(
    per_query: list[QueryMetrics],
    ks: list[int],
) -> AggregateMetrics:
    """Aggregate metrics across a set of queries."""
    if not per_query:
        return AggregateMetrics(
            metrics_by_k={
                k: {"recall": 0.0, "precision": 0.0, "hit_rate": 0.0, "ndcg": 0.0}
                for k in ks
            },
            latency_stats_ms={"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0},
            mrr=0.0,
        )

    # Latency stats
    latencies = [q.latency_ms for q in per_query]
    latencies_sorted = sorted(latencies)
    mean_latency = statistics.mean(latencies)
    p50 = statistics.median(latencies_sorted)
    p95_index = int(0.95 * (len(latencies_sorted) - 1))
    p95 = latencies_sorted[p95_index]
    max_latency = max(latencies_sorted)

    # Mean reciprocal rank (MRR)
    reciprocals: list[float] = []
    for q in per_query:
        rank = q.first_relevant_rank
        if rank is not None and rank > 0:
            reciprocals.append(1.0 / float(rank))
    mrr = statistics.mean(reciprocals) if reciprocals else 0.0

    metrics_by_k: dict[int, dict[str, float]] = {}

    for k in ks:
        recalls: list[float] = []
        precisions: list[float] = []
        hits_flags: list[float] = []
        ndcgs: list[float] = []
        for q in per_query:
            m = q.metrics_by_k.get(k)
            if not m:
                continue
            recalls.append(m.get("recall", 0.0))
            precisions.append(m.get("precision", 0.0))
            hit_count = m.get("hit_count", 0.0)
            hits_flags.append(1.0 if hit_count > 0 else 0.0)

            # nDCG@k for binary relevance with at most one relevant document per query.
            rank = q.first_relevant_rank
            if rank is not None and rank <= k:
                dcg = 1.0 / math.log2(float(rank) + 1.0)
                idcg = 1.0 / math.log2(2.0)
                ndcg = dcg / idcg if idcg > 0.0 else 0.0
            else:
                ndcg = 0.0
            ndcgs.append(ndcg)

        if recalls:
            avg_recall = statistics.mean(recalls)
            avg_precision = statistics.mean(precisions)
            hit_rate = statistics.mean(hits_flags)
            avg_ndcg = statistics.mean(ndcgs) if ndcgs else 0.0
        else:
            avg_recall = 0.0
            avg_precision = 0.0
            hit_rate = 0.0
            avg_ndcg = 0.0

        metrics_by_k[k] = {
            "recall": avg_recall,
            "precision": avg_precision,
            "hit_rate": hit_rate,
            "ndcg": avg_ndcg,
        }

    latency_stats: dict[str, float] = {
        "mean": mean_latency,
        "p50": p50,
        "p95": p95,
        "max": max_latency,
    }

    return AggregateMetrics(
        metrics_by_k=metrics_by_k,
        latency_stats_ms=latency_stats,
        mrr=mrr,
    )


def format_human_summary(result: EvalResult) -> None:
    """Print a concise human-readable summary of evaluation metrics."""
    if result.global_metrics is None:
        print("No metrics computed.")
        return

    languages_str = ", ".join(lang.value for lang in result.languages)
    print(f"Mode: {result.mode}, search={result.search_mode}")
    print(f"Languages: {languages_str}")
    print(f"Queries: {len(result.per_query)}")

    print("\nGlobal metrics:")
    for k in sorted(result.ks):
        m = result.global_metrics.metrics_by_k.get(k, {})
        recall = m.get("recall", 0.0)
        precision = m.get("precision", 0.0)
        hit_rate = m.get("hit_rate", 0.0)
        ndcg = m.get("ndcg", 0.0)
        print(
            f"  k={k:2d}: recall={recall:.3f}, "
            f"precision={precision:.3f}, hit-rate={hit_rate:.3f}, ndcg={ndcg:.3f}"
        )

    lat = result.global_metrics.latency_stats_ms
    print(
        f"\nLatency (ms): mean={lat['mean']:.1f}, "
        f"p50={lat['p50']:.1f}, p95={lat['p95']:.1f}, max={lat['max']:.1f}"
    )

    print(f"\nMRR: {result.global_metrics.mrr:.3f}")

    print("\nPer-language metrics:")
    for language in sorted(result.languages, key=lambda l: l.value):
        lang_metrics = result.per_language.get(language.value)
        if not lang_metrics:
            continue
        line_parts = [f"  {language.value}:"]
        for k in sorted(result.ks):
            m = lang_metrics.metrics_by_k.get(k, {})
            recall = m.get("recall", 0.0)
            hit_rate = m.get("hit_rate", 0.0)
            ndcg = m.get("ndcg", 0.0)
            line_parts.append(f"k={k}: r={recall:.2f}, h={hit_rate:.2f}, n={ndcg:.2f}")
        print(" ".join(line_parts))


def build_json_payload(result: EvalResult) -> dict[str, Any]:
    """Convert EvalResult to JSON-serializable payload."""
    global_metrics = (
        {
            "metrics_by_k": result.global_metrics.metrics_by_k,
            "latency_ms": result.global_metrics.latency_stats_ms,
            "mrr": result.global_metrics.mrr,
        }
        if result.global_metrics
        else None
    )

    per_language: dict[str, Any] = {}
    for lang, metrics in result.per_language.items():
        per_language[lang] = {
            "metrics_by_k": metrics.metrics_by_k,
            "latency_ms": metrics.latency_stats_ms,
            "mrr": metrics.mrr,
        }

    per_query_payload: list[dict[str, Any]] = []
    for q in result.per_query:
        per_query_payload.append(
            {
                "id": q.query_id,
                "language": q.language.value,
                "pattern": q.pattern,
                "search_type": q.search_type,
                "latency_ms": q.latency_ms,
                "total_results": q.total_results,
                "first_relevant_rank": q.first_relevant_rank,
                "metrics_by_k": q.metrics_by_k,
            }
        )

    return {
        "mode": result.mode,
        "search_mode": result.search_mode,
        "languages": [lang.value for lang in result.languages],
        "ks": result.ks,
        "global": global_metrics,
        "per_language": per_language,
        "per_query": per_query_payload,
    }


