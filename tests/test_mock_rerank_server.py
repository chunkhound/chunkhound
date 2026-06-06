"""Deterministic contract tests for the mock rerank server."""

import httpx
import pytest

from tests.fixtures.rerank_server_manager import RerankServerManager
from tests.rerank_server import MockRerankResult, MockRerankScenario


@pytest.mark.asyncio
async def test_mock_rerank_server_health_reports_identity() -> None:
    """Health should prove the deterministic mock is the process under test."""
    scenario = MockRerankScenario(
        name="health-count",
        query="unused",
        documents=["unused"],
        results=[MockRerankResult(index=0, score=1.0)],
    )

    async with RerankServerManager(scenarios=[scenario]) as manager:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{manager.base_url}/health")

        assert response.status_code == 200
        assert response.json() == {
            "healthy": True,
            "service": "mock-rerank-server",
            "scenario_count": 1,
        }


@pytest.mark.asyncio
async def test_mock_rerank_server_matches_exact_cohere_request_and_records_it() -> None:
    """Cohere requests should match exact fixtures and keep request capture."""
    query = "python function"
    documents = [
        "def add(a, b): return a + b",
        "The weather is nice today",
        "class Calculator: pass",
    ]
    scenario = MockRerankScenario(
        name="cohere-match",
        query=query,
        documents=documents,
        results=[
            MockRerankResult(index=0, score=0.99),
            MockRerankResult(index=2, score=0.41),
            MockRerankResult(index=1, score=0.05),
        ],
        response_format="cohere",
    )

    async with RerankServerManager(scenarios=[scenario]) as manager:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.post(
                f"{manager.base_url}/rerank",
                json={
                    "model": "test-model",
                    "query": query,
                    "documents": documents,
                    "top_n": 2,
                },
            )

        assert response.status_code == 200
        assert response.json() == {
            "results": [
                {"index": 0, "relevance_score": 0.99},
                {"index": 2, "relevance_score": 0.41},
            ]
        }
        assert manager.requests == [
            {
                "model": "test-model",
                "query": query,
                "documents": documents,
                "top_n": 2,
            }
        ]


@pytest.mark.asyncio
async def test_mock_rerank_server_supports_tei_and_bare_array_formats() -> None:
    """The mock should cover both wrapped and bare-array TEI contracts."""
    scenarios = [
        MockRerankScenario(
            name="tei",
            query="wrapped",
            documents=["a", "b"],
            results=[MockRerankResult(index=1, score=0.88)],
            response_format="tei",
        ),
        MockRerankScenario(
            name="tei-bare",
            query="bare",
            documents=["x", "y"],
            results=[MockRerankResult(index=0, score=0.77)],
            response_format="tei-bare",
        ),
    ]

    async with RerankServerManager(scenarios=scenarios) as manager:
        async with httpx.AsyncClient(timeout=2.0) as client:
            wrapped = await client.post(
                f"{manager.base_url}/rerank",
                json={"query": "wrapped", "texts": ["a", "b"]},
            )
            bare = await client.post(
                f"{manager.base_url}/rerank",
                json={"query": "bare", "texts": ["x", "y"]},
            )

        assert wrapped.json() == {"results": [{"index": 1, "score": 0.88}]}
        assert bare.json() == [{"index": 0, "score": 0.77}]


@pytest.mark.asyncio
async def test_rerank_server_manager_stops_cleanly() -> None:
    """Manager should expose a live server only inside its async context."""
    manager = RerankServerManager()

    assert not await manager.is_running()
    async with manager:
        assert await manager.is_running()
    assert not await manager.is_running()


@pytest.mark.asyncio
async def test_mock_rerank_server_rejects_unmatched_requests() -> None:
    """Exact matching keeps tests deterministic by failing unexpected requests."""
    scenario = MockRerankScenario(
        name="known-request",
        query="expected",
        documents=["doc"],
        results=[MockRerankResult(index=0, score=1.0)],
    )

    async with RerankServerManager(scenarios=[scenario]) as manager:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.post(
                f"{manager.base_url}/rerank",
                json={
                    "model": "test-model",
                    "query": "unexpected",
                    "documents": ["doc"],
                },
            )

        assert response.status_code == 400
        assert response.json()["error"] == "No mock rerank scenario matched request"
