"""Synthesis engine fallback behavior when rerank results are invalid."""

import pytest

from chunkhound.interfaces.embedding_provider import RerankResult
from chunkhound.services.clustering_service import ClusterGroup
from chunkhound.services.research import SynthesisEngine
from chunkhound.services.research.shared.models import ResearchContext
from tests.fixtures.fake_providers import FakeEmbeddingProvider
from tests.unit.research.conftest import FakeParent


class _OutOfBoundsEmbeddingProvider(FakeEmbeddingProvider):
    async def rerank(self, query: str, documents: list[str], top_k=None):  # noqa: ANN001
        return [RerankResult(index=len(documents) + 5, score=1.0)]


@pytest.mark.asyncio
async def test_rerank_out_of_bounds_falls_back(llm_manager):
    embedding_provider = _OutOfBoundsEmbeddingProvider()
    parent = FakeParent(embedding_provider)
    engine = SynthesisEngine(
        llm_manager, database_services=object(), parent_service=parent
    )

    chunks = [
        {
            "file_path": "a.py",
            "content": "print('hi')",
            "score": 1.0,
            "start_line": 1,
            "end_line": 1,
        }
    ]
    files = {"a.py": "print('hi')"}
    budgets = {"input_tokens": 1000, "output_tokens": 100}

    (
        _prioritized,
        budgeted_files,
        _info,
    ) = await engine._manage_token_budget_for_synthesis(
        chunks=chunks,
        files=files,
        root_query="test query",
        synthesis_budgets=budgets,
    )

    assert "a.py" in budgeted_files


@pytest.mark.asyncio
async def test_map_synthesis_uses_output_budget_for_cluster_allocation(
    capturing_llm_manager,
):
    llm_manager, fake_provider = capturing_llm_manager
    parent = FakeParent(FakeEmbeddingProvider())
    engine = SynthesisEngine(
        llm_manager, database_services=object(), parent_service=parent
    )

    cluster = ClusterGroup(
        cluster_id=0,
        file_paths=["a.py"],
        files_content={"a.py": "print('hi')"},
        total_tokens=20_000,
    )
    chunks = [
        {
            "file_path": "a.py",
            "content": "print('hi')",
            "start_line": 1,
            "end_line": 1,
        }
    ]

    await engine._map_synthesis_on_cluster(
        cluster=cluster,
        chunks=chunks,
        context=ResearchContext(root_query="synthesis test"),
        synthesis_budgets={"output_tokens": 30_000},
        total_input_tokens=100_000,
    )

    assert len(fake_provider.calls) == 1
    call = fake_provider.calls[0]
    assert call["max_completion_tokens"] == 6000
    assert "Target output: ~6,000 tokens" in call["system"]
