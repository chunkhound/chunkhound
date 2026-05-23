"""Tests for daemon IPC helper contracts."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.daemon.ipc import authenticated_ping


class FakeWriter:
    def __init__(self) -> None:
        self.closed = False
        self.waited_closed = False

    def close(self) -> None:
        self.closed = True

    def write(self, data: bytes) -> None:
        pass

    async def wait_closed(self) -> None:
        self.waited_closed = True

    async def drain(self) -> None:
        pass


@pytest.mark.asyncio
async def test_authenticated_ping_returns_false_when_connect_times_out() -> None:
    async def never_connect(address: str):
        await asyncio.sleep(1)

    with patch("chunkhound.daemon.ipc.create_client", side_effect=never_connect):
        assert await authenticated_ping("tcp:127.0.0.1:1", "tok", timeout=0.01) is False


@pytest.mark.asyncio
async def test_authenticated_ping_returns_false_for_malformed_address() -> None:
    assert await authenticated_ping("tcp:127.0.0.1:notaport", "tok") is False


@pytest.mark.asyncio
async def test_authenticated_ping_closes_writer_when_read_times_out() -> None:
    writer = FakeWriter()

    async def never_read(reader):
        await asyncio.sleep(1)

    with (
        patch(
            "chunkhound.daemon.ipc.create_client",
            new=AsyncMock(return_value=(MagicMock(), writer)),
        ),
        patch("chunkhound.daemon.ipc.read_frame", side_effect=never_read),
    ):
        assert await authenticated_ping("tcp:127.0.0.1:1", "tok", timeout=0.01) is False

    assert writer.closed is True
    assert writer.waited_closed is True


@pytest.mark.asyncio
async def test_authenticated_ping_returns_false_for_malformed_register_ack() -> None:
    writer = FakeWriter()

    with (
        patch(
            "chunkhound.daemon.ipc.create_client",
            new=AsyncMock(return_value=(MagicMock(), writer)),
        ),
        patch(
            "chunkhound.daemon.ipc.read_frame",
            new=AsyncMock(return_value={"type": "nope"}),
        ),
    ):
        assert await authenticated_ping("tcp:127.0.0.1:1", "tok", timeout=0.1) is False

    assert writer.closed is True
    assert writer.waited_closed is True


@pytest.mark.asyncio
async def test_authenticated_ping_returns_false_for_malformed_frame() -> None:
    writer = FakeWriter()

    with (
        patch(
            "chunkhound.daemon.ipc.create_client",
            new=AsyncMock(return_value=(MagicMock(), writer)),
        ),
        patch(
            "chunkhound.daemon.ipc.read_frame",
            new=AsyncMock(side_effect=ValueError("bad frame")),
        ),
    ):
        assert await authenticated_ping("tcp:127.0.0.1:1", "tok", timeout=0.1) is False

    assert writer.closed is True
    assert writer.waited_closed is True
