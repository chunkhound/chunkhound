"""Minimal JSON-RPC client for the MCP Streamable HTTP transport.

Analogous to ``SubprocessJsonRpcClient``, but talks to a running HTTP MCP
server over the network instead of bridging a stdio subprocess.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from .jsonrpc_envelope import (
    JsonRpcTimeoutError,
    SubprocessJsonRpcError,
    build_notification,
    build_request,
    unwrap_result,
)

_ACCEPT_HEADER = "application/json, text/event-stream"


def _parse_sse_data(text: str) -> dict[str, Any]:
    """Extract the JSON payload from an SSE-framed ``data: ...`` response body."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("data:"):
            return json.loads(stripped[len("data:") :].strip())
    raise SubprocessJsonRpcError(f"No SSE data line found in response body: {text!r}")


class HttpMcpClient:
    """JSON-RPC client for the MCP Streamable HTTP transport.

    Captures the ``Mcp-Session-Id`` returned by ``initialize`` and forwards
    it on every subsequent request, matching how a persistent client (e.g.
    Claude Code) behaves against the stateful session manager.
    """

    def __init__(self, base_url: str, auth_token: str | None = None):
        self._base_url = base_url.rstrip("/")
        self._mcp_url = f"{self._base_url}/mcp"
        headers = {}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        self._client = httpx.AsyncClient(headers=headers)
        self._session_id: str | None = None
        self._next_request_id = 1
        self._closed = False

    async def initialize(self, timeout: float = 10.0) -> dict[str, Any]:
        """Send the ``initialize`` request and capture the session ID."""
        return await self.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
            timeout=timeout,
        )

    async def send_request(
        self, method: str, params: dict[str, Any] | None = None, timeout: float = 5.0
    ) -> dict[str, Any]:
        """Send a JSON-RPC request and return its result object."""
        request_id = self._next_request_id
        self._next_request_id += 1

        payload = build_request(method, params, request_id)
        response = await self._post(payload, timeout=timeout)

        if response.status_code != 200:
            raise SubprocessJsonRpcError(
                f"HTTP {response.status_code} from {method}: {response.text}"
            )

        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            body = _parse_sse_data(response.text)
        else:
            body = response.json()
        return unwrap_result(body, method)

    async def send_notification(
        self, method: str, params: dict[str, Any] | None = None
    ) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        await self._post(build_notification(method, params), timeout=5.0)

    async def get(self, path: str, timeout: float = 5.0) -> httpx.Response:
        """GET an arbitrary path on the server (e.g. ``/health``)."""
        return await self._client.get(f"{self._base_url}{path}", timeout=timeout)

    async def terminate_session(self, timeout: float = 5.0) -> httpx.Response:
        """Explicitly tear down the current session via ``DELETE /mcp``.

        Requires a prior ``initialize()`` call to have captured a session ID.
        """
        if self._session_id is None:
            raise SubprocessJsonRpcError(
                "terminate_session() called before a session ID was captured "
                "(call initialize() first)"
            )
        return await self._client.delete(
            self._mcp_url,
            headers={"Mcp-Session-Id": self._session_id},
            timeout=timeout,
        )

    async def _post(self, payload: dict[str, Any], timeout: float) -> httpx.Response:
        headers = {"Accept": _ACCEPT_HEADER, "Content-Type": "application/json"}
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        try:
            response = await self._client.post(
                self._mcp_url, json=payload, headers=headers, timeout=timeout
            )
        except httpx.TimeoutException as exc:
            raise JsonRpcTimeoutError(
                f"Request {payload.get('method')} timed out after {timeout}s"
            ) from exc

        session_id = response.headers.get("Mcp-Session-Id")
        if session_id:
            self._session_id = session_id
        return response

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._closed:
            return
        self._closed = True
        await self._client.aclose()
