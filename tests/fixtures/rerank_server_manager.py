"""In-process mock rerank server manager for deterministic HTTP contract tests."""

from typing import Any

import httpx

from tests.rerank_server import MockRerankScenario, MockRerankServer


class RerankServerManager:
    """Own a deterministic mock rerank server for one test."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        scenarios: list[MockRerankScenario] | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.server = MockRerankServer(host=host, port=port, scenarios=scenarios or [])

    @property
    def requests(self) -> list[dict[str, Any]]:
        return self.server.requests

    def set_scenarios(self, scenarios: list[MockRerankScenario]) -> None:
        self.server.scenarios = scenarios

    async def start(self) -> None:
        if self.server.runner is not None:
            return

        await self.server.start()
        sockets = getattr(getattr(self.server.site, "_server", None), "sockets", None)
        if not sockets:
            raise RuntimeError("Mock rerank server did not expose a bound socket")

        self.port = int(sockets[0].getsockname()[1])
        self.base_url = f"http://{self.host}:{self.port}"

    async def stop(self) -> None:
        await self.server.stop()

    async def is_running(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=1.0) as client:
                response = await client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except (httpx.RequestError, httpx.TimeoutException):
            return False

    async def __aenter__(self) -> "RerankServerManager":
        await self.start()
        return self

    async def __aexit__(
        self, exc_type: object, exc_val: object, exc_tb: object
    ) -> None:
        await self.stop()


async def ensure_rerank_server_running(
    host: str = "127.0.0.1",
    port: int = 0,
    *,
    scenarios: list[MockRerankScenario] | None = None,
) -> RerankServerManager:
    """Start and return a deterministic in-process mock rerank server."""
    manager = RerankServerManager(host=host, port=port, scenarios=scenarios)
    await manager.start()
    return manager
