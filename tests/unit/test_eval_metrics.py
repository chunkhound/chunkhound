from __future__ import annotations

import pytest

from chunkhound.core.types.common import Language
from chunkhound.tools.eval.metrics import (
    AggregateMetrics,
    EvalResult,
    QueryMetrics,
    aggregate_metrics,
    build_json_payload,
)


def _make_query_metrics(
    query_id: str,
    first_relevant_rank: int | None,
    latency_ms: float,
    metrics_by_k: dict[int, dict[str, float]],
) -> QueryMetrics:
    return QueryMetrics(
        query_id=query_id,
        language=Language.PYTHON,
        pattern="token",
        search_type="regex",
        latency_ms=latency_ms,
        total_results=10,
        first_relevant_rank=first_relevant_rank,
        metrics_by_k=metrics_by_k,
    )


def test_aggregate_metrics_computes_latency_and_mrr() -> None:
    ks = [1, 5]

    q1 = _make_query_metrics(
        "q1",
        first_relevant_rank=1,
        latency_ms=100.0,
        metrics_by_k={
            1: {"recall": 1.0, "precision": 1.0, "hit_count": 1.0},
            5: {"recall": 1.0, "precision": 0.2, "hit_count": 1.0},
        },
    )
    q2 = _make_query_metrics(
        "q2",
        first_relevant_rank=3,
        latency_ms=300.0,
        metrics_by_k={
            1: {"recall": 0.0, "precision": 0.0, "hit_count": 0.0},
            5: {"recall": 1.0, "precision": 0.2, "hit_count": 1.0},
        },
    )

    agg = aggregate_metrics([q1, q2], ks)

    m1 = agg.metrics_by_k[1]
    assert m1["recall"] == pytest.approx(0.5)
    assert m1["precision"] == pytest.approx(0.5)
    assert m1["hit_rate"] == pytest.approx(0.5)

    m5 = agg.metrics_by_k[5]
    assert m5["recall"] == pytest.approx(1.0)
    assert m5["precision"] == pytest.approx(0.2)
    assert m5["hit_rate"] == pytest.approx(1.0)

    expected_mrr = (1.0 + (1.0 / 3.0)) / 2.0
    assert agg.mrr == pytest.approx(expected_mrr)

    lat = agg.latency_stats_ms
    assert lat["mean"] == pytest.approx(200.0)
    assert lat["p50"] == pytest.approx(200.0)
    assert lat["max"] == pytest.approx(300.0)


def test_aggregate_metrics_empty_queries_returns_zeroes() -> None:
    ks = [1, 5]
    agg = aggregate_metrics([], ks)

    assert isinstance(agg, AggregateMetrics)
    for k in ks:
        metrics = agg.metrics_by_k[k]
        assert metrics["recall"] == 0.0
        assert metrics["precision"] == 0.0
        assert metrics["hit_rate"] == 0.0
        assert metrics["ndcg"] == 0.0

    assert agg.latency_stats_ms["mean"] == 0.0
    assert agg.mrr == 0.0


def test_build_json_payload_structure() -> None:
    ks = [1]
    q = _make_query_metrics(
        "q1",
        first_relevant_rank=1,
        latency_ms=42.0,
        metrics_by_k={1: {"recall": 1.0, "precision": 1.0, "hit_count": 1.0}},
    )
    agg = aggregate_metrics([q], ks)

    result = EvalResult(
        mode="mixed",
        search_mode="regex",
        languages=[Language.PYTHON],
        ks=ks,
        per_query=[q],
        per_language={Language.PYTHON.value: agg},
        global_metrics=agg,
    )

    payload = build_json_payload(result)

    assert payload["mode"] == "mixed"
    assert payload["search_mode"] == "regex"
    assert payload["languages"] == [Language.PYTHON.value]
    assert payload["ks"] == ks
    assert "global" in payload
    assert "per_language" in payload
    assert "per_query" in payload
    assert payload["per_query"][0]["id"] == "q1"

