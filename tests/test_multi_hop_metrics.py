"""Small judged benchmark for deterministic multi-hop retrieval."""

from typing import cast

import pytest

from chunkhound.interfaces.database_provider import DatabaseProvider
from chunkhound.interfaces.embedding_provider import EmbeddingProvider
from chunkhound.services.search_service import SearchService
from tests.fixtures.multi_hop_synthetic import build_multi_hop_scenario

QUERIES = [
    "security validation mechanisms",
    "database connection auth",
    "credential provider secret",
]


def _recall_at_k(results: list[dict], relevant_ids: set[int], k: int) -> float:
    if not relevant_ids:
        return 0.0
    returned = {result["chunk_id"] for result in results[:k]}
    return len(returned & relevant_ids) / len(relevant_ids)


def _mrr(results: list[dict], relevant_ids: set[int]) -> float:
    for rank, result in enumerate(results, start=1):
        if result["chunk_id"] in relevant_ids:
            return 1.0 / rank
    return 0.0


@pytest.mark.fast
@pytest.mark.asyncio
async def test_multi_hop_improves_recall_and_mrr_on_synthetic_benchmark() -> None:
    """Multi-hop should outperform single-hop on predefined bridging scenarios."""
    scenario = build_multi_hop_scenario()
    search_service = SearchService(
        cast(DatabaseProvider, scenario.db),
        cast(EmbeddingProvider, scenario.provider),
    )

    single_recall_total = 0.0
    multi_recall_total = 0.0
    single_mrr_total = 0.0
    multi_mrr_total = 0.0

    for query in QUERIES:
        relevant_ids = scenario.qrels[query]
        single_results, _ = await search_service.search_semantic(
            query,
            page_size=5,
            force_strategy="single_hop",
        )
        multi_results, _ = await search_service.search_semantic(query, page_size=5)

        single_recall_total += _recall_at_k(single_results, relevant_ids, 5)
        multi_recall_total += _recall_at_k(multi_results, relevant_ids, 5)
        single_mrr_total += _mrr(single_results, relevant_ids)
        multi_mrr_total += _mrr(multi_results, relevant_ids)

    query_count = len(QUERIES)
    single_recall = single_recall_total / query_count
    multi_recall = multi_recall_total / query_count
    single_mrr = single_mrr_total / query_count
    multi_mrr = multi_mrr_total / query_count

    assert single_recall == 0.5
    assert multi_recall == 1.0
    assert single_mrr == pytest.approx((0.0 + 1.0 + 1.0) / 3)
    assert multi_mrr == 1.0
    assert multi_recall > single_recall
    assert multi_mrr > single_mrr
