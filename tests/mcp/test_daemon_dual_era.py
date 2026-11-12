"""Contract test: the daemon serves both the legacy handshake era and the
modern 2026-07-28 per-request-envelope era on the same IPC socket, per
client connection.

Uses ``daemon_harness`` (tests/mcp/conftest.py): a real ``ChunkHoundDaemon``
bound to a real IPC socket, with only DB/embedding/realtime startup stubbed
out (protocol dispatch under test here does not touch the database).
"""

from __future__ import annotations

import pytest
from mcp_types import (
    CLIENT_CAPABILITIES_META_KEY,
    CLIENT_INFO_META_KEY,
    PROTOCOL_VERSION_META_KEY,
)


@pytest.mark.asyncio
async def test_daemon_serves_legacy_initialize_handshake(daemon_harness) -> None:
    reader, writer = await daemon_harness.connect_and_register()
    response = await daemon_harness.send_and_recv(
        reader,
        writer,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "pi", "version": "0.0.0"},
            },
        },
    )
    assert "result" in response
    assert response["result"]["serverInfo"]["name"] == "ChunkHound Code Search"

    tools_response = await daemon_harness.send_and_recv(
        reader,
        writer,
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    )
    assert any(t["name"] == "search" for t in tools_response["result"]["tools"])


@pytest.mark.asyncio
async def test_daemon_serves_modern_per_request_envelope(daemon_harness) -> None:
    reader, writer = await daemon_harness.connect_and_register()
    response = await daemon_harness.send_and_recv(
        reader,
        writer,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {
                "_meta": {
                    PROTOCOL_VERSION_META_KEY: "2026-07-28",
                    CLIENT_INFO_META_KEY: {"name": "modern-client", "version": "0.0.0"},
                    CLIENT_CAPABILITIES_META_KEY: {},
                }
            },
        },
    )
    assert "result" in response
    assert response["result"]["resultType"] == "complete"
    assert "ttlMs" in response["result"]
    assert "cacheScope" in response["result"]


@pytest.mark.asyncio
async def test_daemon_serves_two_concurrent_requests_on_one_connection(
    daemon_harness,
) -> None:
    """Two in-flight requests on the same connection may complete out of
    request order (each is dispatched as its own async task);
    clients must correlate by id, not arrival order — both must still
    arrive correctly."""
    reader, writer = await daemon_harness.connect_and_register()
    await daemon_harness.send(
        writer, {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    )
    await daemon_harness.send(
        writer, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    )
    responses = {
        (await daemon_harness.recv(reader))["id"],
        (await daemon_harness.recv(reader))["id"],
    }
    assert responses == {1, 2}


@pytest.mark.asyncio
async def test_daemon_skips_malformed_frames_and_keeps_serving(daemon_harness) -> None:
    """A non-object frame (batch array) or an object that is not valid
    JSON-RPC must not end the client's session; later requests still work."""
    reader, writer = await daemon_harness.connect_and_register()
    await daemon_harness.send(writer, [{"jsonrpc": "2.0", "id": 9, "method": "ping"}])
    await daemon_harness.send(writer, {"not": "a valid jsonrpc message"})
    response = await daemon_harness.send_and_recv(
        reader,
        writer,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "pi", "version": "0.0.0"},
            },
        },
    )
    assert response["id"] == 1 and "result" in response
