"""Drive an MCPServerBase through its public serve_connection() seam.

Tests use these helpers instead of reaching into the SDK's private request
handler table, so they exercise the same path as a real client.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import mcp.types as types

from chunkhound.mcp_server.base import MCPServerBase

_INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 0,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-11-25",
        "capabilities": {},
        "clientInfo": {"name": "test-client", "version": "0.0.0"},
    },
}


async def _request(
    server: MCPServerBase, method: str, params: dict[str, Any], timeout: float
) -> dict[str, Any]:
    incoming: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    outgoing: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def read_frame() -> dict[str, Any] | None:
        return await incoming.get()

    async def write_frame(frame: dict[str, Any]) -> None:
        await outgoing.put(frame)

    session = asyncio.create_task(server.serve_connection(read_frame, write_frame))
    try:
        await incoming.put(_INITIALIZE)
        await asyncio.wait_for(outgoing.get(), timeout)
        await incoming.put({"jsonrpc": "2.0", "method": "notifications/initialized"})
        await incoming.put(
            {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        )
        while True:
            frame = await asyncio.wait_for(outgoing.get(), timeout)
            if frame.get("id") == 1:
                return frame
    finally:
        await incoming.put(None)
        session.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await session


async def call_tool(
    server: MCPServerBase,
    name: str,
    arguments: dict[str, Any],
    timeout: float = 10.0,
) -> types.CallToolResult:
    frame = await _request(
        server, "tools/call", {"name": name, "arguments": arguments}, timeout
    )
    assert "result" in frame, frame
    return types.CallToolResult.model_validate(frame["result"])


async def list_tools(
    server: MCPServerBase, timeout: float = 10.0
) -> types.ListToolsResult:
    frame = await _request(server, "tools/list", {}, timeout)
    assert "result" in frame, frame
    return types.ListToolsResult.model_validate(frame["result"])
