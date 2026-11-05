"""Shared fixtures for tests/mcp/ — real daemon, real IPC transport.

``daemon_harness`` spins up a real ``ChunkHoundDaemon`` bound to a real IPC
socket (the same ``chunkhound.daemon.ipc`` transport used in production) and
exposes raw register/send/recv helpers so tests can drive the daemon's
JSON-RPC protocol directly, without a ``chunkhound mcp`` proxy subprocess in
the loop.

Database/service initialization is stubbed out (``daemon.initialize()`` is
replaced with a no-op that just marks initialization complete) because these
tests only exercise protocol dispatch (tools/list, the legacy/modern era
split, concurrent requests) via ``MCPServerBase.serve_connection()`` — none
of which require a real DuckDB connection or embedding/LLM providers.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from chunkhound.core.config.config import Config
from chunkhound.daemon import ipc
from chunkhound.daemon.server import ChunkHoundDaemon


@dataclass
class DaemonHarness:
    """Thin raw-frame client for a real ``ChunkHoundDaemon`` IPC connection."""

    daemon: ChunkHoundDaemon
    address: str
    _writers: list[asyncio.StreamWriter]

    async def connect_and_register(
        self,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Open a new IPC connection and complete the register handshake."""
        reader, writer = await ipc.create_client(self.address)
        self._writers.append(writer)
        ipc.write_frame(writer, {"type": "register", "pid": os.getpid()})
        await writer.drain()
        ack = await asyncio.wait_for(ipc.read_frame(reader), timeout=5.0)
        assert isinstance(ack, dict) and ack.get("type") == "registered", ack
        return reader, writer

    async def send(self, writer: asyncio.StreamWriter, message: Any) -> None:
        ipc.write_frame(writer, message)
        await writer.drain()

    async def recv(
        self, reader: asyncio.StreamReader, timeout: float = 5.0
    ) -> dict[str, Any]:
        frame = await asyncio.wait_for(ipc.read_frame(reader), timeout=timeout)
        assert isinstance(frame, dict), frame
        return frame

    async def send_and_recv(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        message: dict[str, Any],
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        await self.send(writer, message)
        return await self.recv(reader, timeout=timeout)

    async def aclose(self) -> None:
        for writer in self._writers:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass


@pytest.fixture
async def daemon_harness(tmp_path: Path) -> AsyncIterator[DaemonHarness]:
    """Real ``ChunkHoundDaemon`` bound to a real IPC socket; no DB required."""
    project_dir = tmp_path
    db_path = project_dir / ".chunkhound" / "test.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    config = Config(
        target_dir=project_dir,
        database={"path": str(db_path), "provider": "duckdb"},
        indexing={},
    )
    socket_path = str(project_dir / f"daemon-{uuid.uuid4().hex}.sock")

    daemon = ChunkHoundDaemon(
        config,
        args=MagicMock(path=project_dir),
        socket_path=socket_path,
        project_dir=project_dir,
    )
    # Skip real DB/embedding/realtime startup — only protocol dispatch
    # (tools/list, legacy/modern era handling) is under test here.
    daemon._initialized = True
    daemon._initialization_complete.set()

    server, actual_address = await ipc.create_server(socket_path, daemon._handle_client)

    harness = DaemonHarness(daemon=daemon, address=actual_address, _writers=[])
    try:
        yield harness
    finally:
        await harness.aclose()
        server.close()
        try:
            await server.wait_closed()
        except Exception:
            pass
