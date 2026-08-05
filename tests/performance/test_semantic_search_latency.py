"""Performance regression guards for DuckDB semantic search.

Opt-in: set ``CHUNKHOUND_ENABLE_PERF_TESTS=1``. Wall-clock budgets are
machine-dependent, so these never gate the default suite.

Overfetch efficiency is read from the structured metrics the provider binds to
its overfetch debug log — the same fields operators use to diagnose an ANN beam
that is being wasted on filtered-out rows.
"""

from __future__ import annotations

import math
import os
import statistics
import time
from collections.abc import Generator
from pathlib import Path

import pytest
from loguru import logger

from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from tests.helpers.duckdb_vector_fixtures import (
    MODEL,
    PROVIDER,
    connected_provider,
    insert_chunks,
    insert_embeddings,
    require_hnsw_index,
)

_ENABLE_ENV = "CHUNKHOUND_ENABLE_PERF_TESTS"
_CORPUS_SIZE = 1_000
_DIMS = 64
_QUERY_COUNT = 50
_P95_BUDGET_SECONDS = 0.5
_MAX_MEAN_OVERFETCH_ITERATIONS = 3.0
_TARGET_PATH = "perf/target.py"
_NOISE_PATH = "perf/noise.py"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.hnsw,
    pytest.mark.skipif(
        os.getenv(_ENABLE_ENV) != "1",
        reason=f"set {_ENABLE_ENV}=1 to run semantic-search performance guards",
    ),
]


def _unit_vector(seed: int) -> list[float]:
    """Deterministic unit vector; avoids RNG drift between runs."""
    values = [math.sin((seed + 1) * (axis + 1) * 0.137) for axis in range(_DIMS)]
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values]


@pytest.fixture(scope="module")
def seeded_provider(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[DuckDBProvider, None, None]:
    """One indexed corpus shared by every measurement in this module."""
    tmp_path: Path = tmp_path_factory.mktemp("perf")
    with connected_provider(tmp_path) as provider:
        _seed_corpus(provider)
        yield provider


def _seed_corpus(provider: DuckDBProvider) -> None:
    """Index a two-file corpus so path filtering forces real overfetch."""
    target_size = _CORPUS_SIZE // 10
    noise_ids = insert_chunks(provider, _NOISE_PATH, _CORPUS_SIZE - target_size)
    target_ids = insert_chunks(provider, _TARGET_PATH, target_size)
    insert_embeddings(
        provider, noise_ids, [_unit_vector(index) for index in range(len(noise_ids))]
    )
    insert_embeddings(
        provider,
        target_ids,
        [_unit_vector(10_000 + index) for index in range(len(target_ids))],
    )
    require_hnsw_index(provider, _DIMS)


def _search_latency(provider: DuckDBProvider, seed: int, **kwargs: object) -> float:
    """Return wall-clock seconds for one semantic search."""
    start = time.perf_counter()
    provider.search_semantic(_unit_vector(seed), PROVIDER, MODEL, **kwargs)  # type: ignore[arg-type]
    return time.perf_counter() - start


@pytest.fixture
def overfetch_metrics() -> Generator[list[dict[str, object]], None, None]:
    """Collect the structured overfetch metrics emitted during a test."""
    captured: list[dict[str, object]] = []

    def sink(message: object) -> None:
        extra = message.record["extra"]  # type: ignore[attr-defined]
        if "filter_selectivity" in extra:
            captured.append(dict(extra))

    sink_id = logger.add(sink, level="DEBUG")
    try:
        yield captured
    finally:
        logger.remove(sink_id)


def test_semantic_search_p95_latency_within_budget(
    seeded_provider: DuckDBProvider,
) -> None:
    """P95 semantic search latency stays under the interactive budget."""
    # Warm the index/connection and prove the corpus is actually searchable —
    # an empty corpus would make the latency numbers meaningless.
    warm_results, _ = seeded_provider.search_semantic(
        _unit_vector(0), PROVIDER, MODEL, page_size=10
    )
    assert len(warm_results) == 10
    latencies = sorted(
        _search_latency(seeded_provider, seed, page_size=10)
        for seed in range(_QUERY_COUNT)
    )

    p95 = latencies[math.ceil(0.95 * len(latencies)) - 1]

    assert p95 < _P95_BUDGET_SECONDS, (
        f"P95={p95 * 1000:.1f}ms exceeds {_P95_BUDGET_SECONDS * 1000:.0f}ms "
        f"(median={statistics.median(latencies) * 1000:.1f}ms)"
    )


def test_overfetch_converges_in_few_iterations(
    seeded_provider: DuckDBProvider, overfetch_metrics: list[dict[str, object]]
) -> None:
    """Filtered and unfiltered searches both converge without runaway widening."""
    for seed in range(20):
        seeded_provider.search_semantic(
            _unit_vector(seed),
            PROVIDER,
            MODEL,
            page_size=10,
            path_filter=_TARGET_PATH if seed % 2 else None,
        )

    iterations = [int(metric["iterations"]) for metric in overfetch_metrics]  # type: ignore[call-overload]

    assert iterations, "no overfetch metrics were emitted"
    mean_iterations = statistics.fmean(iterations)
    assert mean_iterations < _MAX_MEAN_OVERFETCH_ITERATIONS, (
        f"mean overfetch iterations={mean_iterations:.2f} (max={max(iterations)})"
    )


def test_candidate_filter_selectivity_is_reported(
    seeded_provider: DuckDBProvider, overfetch_metrics: list[dict[str, object]]
) -> None:
    """Every overfetch reports a usable selectivity ratio for diagnostics."""
    seeded_provider.search_semantic(
        _unit_vector(1), PROVIDER, MODEL, page_size=10, path_filter=_TARGET_PATH
    )

    selectivity = [
        float(metric["filter_selectivity"])  # type: ignore[arg-type]
        for metric in overfetch_metrics
        if metric["filter_selectivity"] is not None
    ]

    assert selectivity, "no selectivity metrics were emitted"
    assert all(0.0 <= value <= 1.0 for value in selectivity)
