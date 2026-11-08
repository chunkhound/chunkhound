"""Unit coverage for MCPServerBase.serve_connection() against a fake,
in-process duplex-frame transport -- no daemon, no IPC, no stdio involved.
"""

import asyncio
import contextlib
from typing import Any
from unittest.mock import MagicMock

import pytest

from chunkhound.mcp_server.stdio import StdioMCPServer


async def _drive(server: StdioMCPServer, frames: list[dict]) -> list[dict]:
    incoming: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    outgoing: list[dict] = []

    async def read_frame() -> dict[str, Any] | None:
        return await incoming.get()

    async def write_frame(frame: dict) -> None:
        outgoing.append(frame)

    session_task = asyncio.create_task(server.serve_connection(read_frame, write_frame))
    try:
        for frame in frames:
            await incoming.put(frame)
        for _ in range(50):
            if len(outgoing) >= len(frames):
                break
            await asyncio.sleep(0.05)
        return outgoing
    finally:
        await incoming.put(None)
        session_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await session_task


@pytest.mark.asyncio
async def test_serve_connection_answers_legacy_initialize() -> None:
    server = StdioMCPServer(config=MagicMock())
    responses = await _drive(
        server,
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "test-client", "version": "0.0.0"},
                },
            }
        ],
    )
    assert responses and responses[0]["id"] == 1 and "result" in responses[0]


@pytest.mark.asyncio
async def test_serve_connection_answers_modern_per_request_envelope() -> None:
    """A bare `_meta`-bearing request with no prior `initialize` must still
    get a valid modern-era response (resultType/ttlMs/cacheScope)."""
    from mcp_types import (
        CLIENT_CAPABILITIES_META_KEY,
        CLIENT_INFO_META_KEY,
        PROTOCOL_VERSION_META_KEY,
    )

    server = StdioMCPServer(config=MagicMock())
    # on_list_tools waits (up to 5s) for initialization; _drive gives up sooner.
    server._initialization_complete.set()
    responses = await _drive(
        server,
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {
                    "_meta": {
                        PROTOCOL_VERSION_META_KEY: "2026-07-28",
                        CLIENT_INFO_META_KEY: {
                            "name": "modern-client",
                            "version": "0.0.0",
                        },
                        CLIENT_CAPABILITIES_META_KEY: {},
                    }
                },
            }
        ],
    )
    assert responses and "result" in responses[0]
    assert responses[0]["result"]["resultType"] == "complete"
    assert "ttlMs" in responses[0]["result"]
    assert "cacheScope" in responses[0]["result"]


@pytest.mark.asyncio
async def test_serve_connection_skips_invalid_jsonrpc_shape_and_keeps_serving() -> None:
    """A dict that fails JSON-RPC shape validation is skipped (as the SDK's
    stdio_server does); the session keeps answering later requests."""
    server = StdioMCPServer(config=MagicMock())
    incoming: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    outgoing: asyncio.Queue[dict] = asyncio.Queue()

    async def read_frame() -> dict[str, Any] | None:
        return await incoming.get()

    async def write_frame(frame: dict) -> None:
        await outgoing.put(frame)

    session_task = asyncio.create_task(server.serve_connection(read_frame, write_frame))
    try:
        await incoming.put({"not": "a valid jsonrpc message"})
        await incoming.put(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "test-client", "version": "0.0.0"},
                },
            }
        )
        response = await asyncio.wait_for(outgoing.get(), timeout=5.0)
        assert response["id"] == 1 and "result" in response
        assert not session_task.done()
    finally:
        await incoming.put(None)
        await asyncio.wait_for(session_task, timeout=5.0)  # EOF ends session


@pytest.mark.asyncio
async def test_tools_call_rejects_arguments_violating_input_schema() -> None:
    server = StdioMCPServer(config=MagicMock())
    init = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "0.0.0"},
        },
    }
    responses = await _drive(
        server,
        [
            init,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "search", "arguments": {"type": "regex"}},
            },
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "search",
                    "arguments": {"type": "bogus", "query": "x"},
                },
            },
        ],
    )
    by_id = {r["id"]: r for r in responses}
    for call_id in (2, 3):
        result = by_id[call_id]["result"]
        assert result["isError"] is True
        assert result["content"][0]["text"].startswith("Input validation error:")
