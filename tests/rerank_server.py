#!/usr/bin/env python3
"""Deterministic mock rerank server for HTTP-level rerank contract tests."""

import asyncio
import sys
from dataclasses import dataclass, field
from typing import Any

from aiohttp import web
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")


@dataclass
class MockRerankResult:
    """One rerank row returned by the mock service."""

    index: int
    score: float


@dataclass
class MockRerankScenario:
    """Deterministic rerank scenario matched by exact query + document list."""

    query: str
    documents: list[str]
    results: list[MockRerankResult]
    response_format: str = "cohere"
    name: str = "scenario"
    status_code: int = 200
    error: str | None = None


class MockRerankServer:
    """Small deterministic rerank server with explicit scenario-driven responses."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8001,
        *,
        scenarios: list[MockRerankScenario] | None = None,
    ):
        self.host = host
        self.port = port
        self.scenarios = scenarios or []
        self.requests: list[dict[str, Any]] = []
        self.app: web.Application | None = None
        self.runner: web.AppRunner | None = None
        self.site: web.TCPSite | None = None

    async def health_handler(self, request: web.Request) -> web.Response:
        """Return server identity so tests can verify the expected mock is running."""
        return web.json_response(
            {
                "healthy": True,
                "service": "mock-rerank-server",
                "scenario_count": len(self.scenarios),
            }
        )

    async def rerank_handler(self, request: web.Request) -> web.Response:
        """Return the predeclared response for the matching scenario."""
        try:
            body = await request.json()
            self.requests.append(body)

            has_texts = "texts" in body
            has_documents = "documents" in body
            if has_texts and has_documents:
                return web.json_response(
                    {
                        "error": "Mixed rerank payload shape is invalid",
                        "error_type": "MockValidationError",
                    },
                    status=400,
                )

            request_format = "tei" if has_texts else "cohere"
            documents = body.get("texts" if request_format == "tei" else "documents", [])
            query = body.get("query", "")
            top_n = body.get("top_n")

            scenario = self._match_scenario(
                query=query,
                documents=documents,
                request_format=request_format,
            )
            if scenario is None:
                return web.json_response(
                    {
                        "error": "No mock rerank scenario matched request",
                        "query": query,
                        "documents": documents,
                        "request_format": request_format,
                    },
                    status=400,
                )

            if scenario.error is not None:
                return web.json_response(
                    {"error": scenario.error, "error_type": "MockScenarioError"},
                    status=scenario.status_code,
                )

            results = list(scenario.results)
            if top_n is not None and top_n > 0:
                results = results[:top_n]

            payload = self._build_response_payload(scenario, results)
            return web.json_response(payload, status=scenario.status_code)

        except Exception as exc:
            logger.error(f"Error in rerank handler: {exc}")
            return web.json_response({"error": str(exc)}, status=400)

    def _match_scenario(
        self, *, query: str, documents: list[str], request_format: str
    ) -> MockRerankScenario | None:
        for scenario in self.scenarios:
            if scenario.query == query and scenario.documents == documents:
                # Strip -bare suffix for format matching (tei-bare → tei).
                # Use removesuffix (not rstrip) — rstrip treats its argument as
                # a character set and would strip 'e', 'r', 'e' from 'cohere'.
                scenario_format = scenario.response_format
                if scenario_format.endswith("-bare"):
                    scenario_format = scenario_format[:-5]
                if scenario_format == request_format:
                    return scenario
        return None

    def _build_response_payload(
        self, scenario: MockRerankScenario, results: list[MockRerankResult]
    ) -> dict[str, Any] | list[dict[str, Any]]:
        if scenario.response_format == "tei-bare":
            return [
                {"index": result.index, "score": result.score} for result in results
            ]

        if scenario.response_format == "tei":
            return {
                "results": [
                    {"index": result.index, "score": result.score}
                    for result in results
                ]
            }

        if scenario.response_format == "cohere":
            return {
                "results": [
                    {"index": result.index, "relevance_score": result.score}
                    for result in results
                ]
            }

        raise ValueError(f"Unsupported mock response format: {scenario.response_format}")

    async def start(self) -> None:
        """Start the aiohttp server."""
        logger.info(f"Starting mock rerank server on {self.host}:{self.port}")
        self.app = web.Application()
        self.app.router.add_get("/health", self.health_handler)
        self.app.router.add_post("/rerank", self.rerank_handler)
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()

    async def stop(self) -> None:
        """Stop the aiohttp server."""
        if self.site is not None:
            await self.site.stop()
            self.site = None
        if self.runner is not None:
            await self.runner.cleanup()
            self.runner = None
        self.app = None
        self.requests.clear()

    async def serve_forever(self) -> None:
        """Run the server until interrupted."""
        if self.runner is None:
            await self.start()
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            await self.stop()


def main() -> None:
    """Run an empty deterministic server for manual debugging."""
    import argparse

    parser = argparse.ArgumentParser(description="Deterministic mock rerank server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    server = MockRerankServer(host=args.host, port=args.port)
    try:
        asyncio.run(server.serve_forever())
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")


if __name__ == "__main__":
    main()
