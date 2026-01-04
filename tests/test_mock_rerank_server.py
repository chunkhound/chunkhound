"""
Test the mock reranking server functionality.

This test verifies that the mock reranking server works correctly
and integrates properly with the test infrastructure.
"""

import httpx
import pytest


@pytest.mark.asyncio
async def test_mock_rerank_server_health():
    """Test that the mock rerank server health endpoint works."""
    # The server should be auto-started by fixtures if needed
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get("http://localhost:8001/health")

            # If we get a response, server is running (mock or external)
            assert response.status_code == 200
            data = response.json()

            # Check response structure
            assert isinstance(data, dict)
            # Could be our mock server or external server
            # Our mock returns {"healthy": True, "service": "mock-rerank-server"}
            # External might return different structure

    except (httpx.RequestError, httpx.TimeoutException):
        # Server not running - this is okay, test will be skipped
        pytest.skip("No reranking server available")


@pytest.mark.asyncio
async def test_mock_rerank_server_reranking():
    """Test that the mock rerank server can rerank documents."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            # First check if server is running
            health_response = await client.get("http://localhost:8001/health")
            if health_response.status_code != 200:
                pytest.skip("Rerank server not healthy")

            # Test reranking
            rerank_request = {
                "model": "test-model",
                "query": "python function",
                "documents": [
                    "def add(a, b): return a + b",  # Should rank high
                    "The weather is nice today",  # Should rank low
                    "class Calculator: pass",  # Should rank medium
                ],
                "top_n": 2,
            }

            response = await client.post(
                "http://localhost:8001/rerank", json=rerank_request
            )

            assert response.status_code == 200
            data = response.json()

            # Check response structure
            assert "results" in data
            assert isinstance(data["results"], list)

            # Check we got at most top_n results
            assert len(data["results"]) <= 2

            # Check result structure
            for result in data["results"]:
                assert "index" in result
                assert "relevance_score" in result
                assert isinstance(result["index"], int)
                assert isinstance(result["relevance_score"], (int, float))
                assert 0 <= result["index"] < len(rerank_request["documents"])
                assert 0 <= result["relevance_score"] <= 1

            # Results should be sorted by score descending
            scores = [r["relevance_score"] for r in data["results"]]
            assert scores == sorted(scores, reverse=True)

    except (httpx.RequestError, httpx.TimeoutException):
        pytest.skip("No reranking server available")


@pytest.mark.asyncio
async def test_rerank_server_manager_lifecycle():
    """Test the rerank server manager can start and stop servers."""
    from tests.fixtures.rerank_server_manager import RerankServerManager

    manager = RerankServerManager(port=8002)  # Use different port to avoid conflicts

    # Server should not be running initially on this port
    assert not await manager.is_running()

    # Start server
    async with manager:
        # Server should be running
        assert await manager.is_running()

        # Test health endpoint
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get("http://127.0.0.1:8002/health")
            assert response.status_code == 200

    # Server should be stopped after context exit
    assert not await manager.is_running()
