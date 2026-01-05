#!/usr/bin/env python3
"""
Mock reranking server for testing multi-hop semantic search.

This lightweight server provides a /rerank endpoint compatible with both:
- Cohere format (documents field, relevance_score)
- TEI format (texts field, score)

Auto-detects format from request for testing purposes without requiring
heavy dependencies like vLLM or TEI servers.
"""

import asyncio
import json
import sys

import httpx
from aiohttp import web
from loguru import logger

# Configure logger for testing
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")


class MockRerankServer:
    """Mock reranking server with Cohere and TEI compatible API using aiohttp."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8001):
        self.host = host
        self.port = port
        self.app = None
        self.runner = None
        self.site = None

    async def health_handler(self, request: web.Request) -> web.Response:
        """Handle health check requests."""
        return web.json_response({"healthy": True, "service": "mock-rerank-server"})

    async def rerank_handler(self, request: web.Request) -> web.Response:
        """Handle reranking requests with Cohere and TEI compatible API.

        Auto-detects format from request:
        - Cohere: uses 'documents' field, requires 'model'
        - TEI: uses 'texts' field, model optional
        """
        try:
            # Parse request body
            body = await request.json()

            # Extract parameters and detect format
            query = body.get("query", "")
            model = body.get("model", "mock-reranker")

            # Detect format: TEI uses "texts", Cohere uses "documents"
            is_tei = "texts" in body
            documents = body.get("texts" if is_tei else "documents", [])
            top_n = body.get("top_n")

            format_name = "TEI" if is_tei else "Cohere"
            logger.debug(
                f"Detected {format_name} format request with {len(documents)} documents"
            )

            # Calculate mock relevance scores
            results = []
            for idx, doc in enumerate(documents):
                score = self._calculate_relevance(query, doc)

                # Use format-appropriate field name
                if is_tei:
                    results.append({"index": idx, "score": score})
                else:
                    results.append({"index": idx, "relevance_score": score})

            # Sort by score descending (works for both field names)
            score_field = "score" if is_tei else "relevance_score"
            results.sort(key=lambda x: x[score_field], reverse=True)

            # Apply top_n if specified
            if top_n is not None and top_n > 0:
                results = results[:top_n]

            logger.debug(
                f"Reranked {len(documents)} documents, returning {len(results)} results ({format_name} format)"
            )

            return web.json_response(
                {"results": results, "model": model, "meta": {"api_version": "v1"}}
            )

        except Exception as e:
            logger.error(f"Error in rerank handler: {e}")
            return web.json_response({"error": str(e)}, status=400)

    def _calculate_relevance(self, query: str, document: str) -> float:
        """
        Calculate mock relevance score using simple heuristics.

        This is for testing only - uses basic text similarity.
        """
        if not query or not document:
            return 0.0

        # Convert to lowercase for comparison
        query_lower = query.lower()
        doc_lower = document.lower()

        # Simple scoring heuristics
        score = 0.0

        # 1. Exact query match
        if query_lower in doc_lower:
            score += 0.5

        # 2. Word overlap (Jaccard similarity)
        query_words = set(query_lower.split())
        doc_words = set(doc_lower.split())

        if query_words and doc_words:
            intersection = query_words & doc_words
            union = query_words | doc_words
            if union:
                jaccard = len(intersection) / len(union)
                score += jaccard * 0.3

        # 3. Keyword matching for common programming terms
        programming_keywords = {
            "function",
            "class",
            "method",
            "def",
            "import",
            "return",
            "async",
            "await",
            "api",
            "endpoint",
        }

        query_has_keywords = bool(query_words & programming_keywords)
        doc_has_keywords = bool(doc_words & programming_keywords)

        if query_has_keywords and doc_has_keywords:
            score += 0.2

        # Ensure score is between 0 and 1
        return min(max(score, 0.0), 1.0)

    async def start(self) -> None:
        """Start the mock server."""
        logger.info(f"Starting mock rerank server on {self.host}:{self.port}")

        # Create aiohttp application
        self.app = web.Application()
        self.app.router.add_get("/health", self.health_handler)
        self.app.router.add_post("/rerank", self.rerank_handler)

        # Start server
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()

        logger.info(f"Mock rerank server listening on http://{self.host}:{self.port}")
        logger.info("Endpoints: /health, /rerank")

    async def stop(self) -> None:
        """Stop the mock server."""
        if self.site:
            logger.info("Stopping mock rerank server")
            await self.site.stop()
            self.site = None
        if self.runner:
            await self.runner.cleanup()
            self.runner = None
        self.app = None

    async def serve_forever(self) -> None:
        """Run the server until interrupted."""
        if not self.runner:
            await self.start()

        # Keep running until stopped
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            await self.stop()


async def test_server():
    """Test the mock server with a sample request."""
    # Start server
    server = MockRerankServer()
    await server.start()

    try:
        # Give server time to start
        await asyncio.sleep(0.1)

        # Test health endpoint
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8001/health")
            print(f"Health check: {response.status_code} - {response.json()}")

            # Test rerank endpoint
            rerank_request = {
                "model": "test-model",
                "query": "python function definition",
                "documents": [
                    "def calculate_sum(a, b): return a + b",
                    "import numpy as np",
                    "class Calculator: pass",
                    "function add(x, y) { return x + y; }",
                ],
            }

            response = await client.post(
                "http://localhost:8001/rerank", json=rerank_request
            )
            print(f"Rerank response: {response.status_code}")
            print(json.dumps(response.json(), indent=2))

    finally:
        await server.stop()


def main():
    """Run the mock rerank server."""
    import argparse

    parser = argparse.ArgumentParser(description="Mock reranking server for testing")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--test", action="store_true", help="Run test mode")

    args = parser.parse_args()

    if args.test:
        # Run test mode
        asyncio.run(test_server())
    else:
        # Run server
        server = MockRerankServer(host=args.host, port=args.port)
        try:
            asyncio.run(server.serve_forever())
        except KeyboardInterrupt:
            logger.info("Server interrupted by user")


if __name__ == "__main__":
    main()
