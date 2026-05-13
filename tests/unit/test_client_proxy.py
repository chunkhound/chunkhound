from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import chunkhound.daemon.client_proxy as client_proxy_module
from chunkhound.daemon.client_proxy import ClientProxy, _SocketForwardResult


class _FakeWriter:
    def __init__(self) -> None:
        self.closed = False
        self.wait_closed_called = False

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        self.wait_closed_called = True


def _make_proxy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ClientProxy, Mock, _FakeWriter]:
    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    proxy = ClientProxy(project_dir, SimpleNamespace())

    discovery = Mock()
    discovery.find_or_start_daemon = AsyncMock(return_value="tcp:127.0.0.1:9000")
    discovery.read_lock.return_value = {"auth_token": "token"}
    discovery.get_daemon_log_path.return_value = (
        project_dir / ".chunkhound" / "daemon.log"
    )
    discovery.format_startup_failure.return_value = "formatted startup failure"
    proxy._discovery = discovery

    writer = _FakeWriter()
    monkeypatch.setattr(
        client_proxy_module.ipc,
        "create_client",
        AsyncMock(return_value=(asyncio.StreamReader(), writer)),
    )
    monkeypatch.setattr(
        client_proxy_module.ipc,
        "write_frame",
        lambda _writer, _frame: None,
    )
    monkeypatch.setattr(
        client_proxy_module.ipc,
        "read_frame",
        AsyncMock(return_value={"type": "registered"}),
    )
    return proxy, discovery, writer


@pytest.mark.asyncio
async def test_run_prefers_formatted_startup_failure_over_transport_reset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy, discovery, writer = _make_proxy(tmp_path, monkeypatch)
    proxy._forward_stdin_to_socket = AsyncMock(
        side_effect=ConnectionResetError("socket reset during startup")
    )
    proxy._forward_socket_to_stdout = AsyncMock(
        return_value=_SocketForwardResult(message_count=0, clean_close=True)
    )

    with pytest.raises(RuntimeError, match="formatted startup failure"):
        await proxy.run()

    discovery.format_startup_failure.assert_called_once()
    assert writer.closed
    assert writer.wait_closed_called


@pytest.mark.asyncio
async def test_run_cancels_pending_stdout_task_before_raising_stdin_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy, _, writer = _make_proxy(tmp_path, monkeypatch)
    stdout_cancelled = asyncio.Event()

    async def blocked_stdout(
        _reader: asyncio.StreamReader,
    ) -> _SocketForwardResult:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            stdout_cancelled.set()
            raise

    proxy._forward_stdin_to_socket = AsyncMock(side_effect=ValueError("stdin exploded"))
    proxy._forward_socket_to_stdout = blocked_stdout

    with pytest.raises(ValueError, match="stdin exploded"):
        await proxy.run()

    assert stdout_cancelled.is_set()
    assert writer.closed
    assert writer.wait_closed_called


@pytest.mark.asyncio
async def test_run_allows_clean_stdout_shutdown_after_first_mcp_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy, discovery, writer = _make_proxy(tmp_path, monkeypatch)
    stdin_cancelled = asyncio.Event()

    async def blocked_stdin(_writer: asyncio.StreamWriter) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            stdin_cancelled.set()
            raise

    proxy._forward_stdin_to_socket = blocked_stdin
    proxy._forward_socket_to_stdout = AsyncMock(
        return_value=_SocketForwardResult(message_count=1, clean_close=True)
    )

    await proxy.run()

    discovery.format_startup_failure.assert_not_called()
    assert stdin_cancelled.is_set()
    assert writer.closed
    assert writer.wait_closed_called


@pytest.mark.asyncio
async def test_forward_socket_to_stdout_returns_clean_close_on_eof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy = ClientProxy(Path("."), SimpleNamespace())
    incomplete_read = asyncio.IncompleteReadError(partial=b"", expected=1)
    monkeypatch.setattr(
        client_proxy_module.ipc,
        "read_frame",
        AsyncMock(side_effect=incomplete_read),
    )

    result = await proxy._forward_socket_to_stdout(asyncio.StreamReader())

    assert result == _SocketForwardResult(message_count=0, clean_close=True)


@pytest.mark.asyncio
async def test_forward_socket_to_stdout_propagates_non_eof_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy = ClientProxy(Path("."), SimpleNamespace())
    monkeypatch.setattr(
        client_proxy_module.ipc,
        "read_frame",
        AsyncMock(side_effect=ValueError("bad frame")),
    )

    with pytest.raises(ValueError, match="bad frame"):
        await proxy._forward_socket_to_stdout(asyncio.StreamReader())


@pytest.mark.asyncio
async def test_run_prefers_stdout_read_error_over_startup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy, discovery, writer = _make_proxy(tmp_path, monkeypatch)
    stdin_cancelled = asyncio.Event()

    async def blocked_stdin(_writer: asyncio.StreamWriter) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            stdin_cancelled.set()
            raise

    async def broken_stdout(_reader: asyncio.StreamReader) -> _SocketForwardResult:
        raise ValueError("bad frame")

    proxy._forward_stdin_to_socket = blocked_stdin
    proxy._forward_socket_to_stdout = broken_stdout

    with pytest.raises(ValueError, match="bad frame"):
        await proxy.run()

    discovery.format_startup_failure.assert_not_called()
    assert stdin_cancelled.is_set()
    assert writer.closed
    assert writer.wait_closed_called
