from __future__ import annotations

import asyncio
import io
import sys
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
        return_value=_SocketForwardResult(message_count=0)
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
            await asyncio.sleep(3600)
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
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            stdin_cancelled.set()
            raise

    proxy._forward_stdin_to_socket = blocked_stdin
    proxy._forward_socket_to_stdout = AsyncMock(
        return_value=_SocketForwardResult(message_count=1)
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

    assert result == _SocketForwardResult(message_count=0)


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
async def test_forward_stdin_windows_forwards_json_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy = ClientProxy(Path("."), SimpleNamespace())
    chunks = [b"not-json\n", b'{"method":"initialize"}\n', b""]
    frames = []

    stdin = SimpleNamespace(buffer=SimpleNamespace(fileno=lambda: 123))
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(
        ClientProxy, "_windows_stdin_handle", staticmethod(lambda _fd: 456)
    )
    monkeypatch.setattr(
        ClientProxy,
        "_windows_pipe_bytes_available",
        staticmethod(lambda _handle: len(chunks[0]) if chunks[0] else None),
    )
    monkeypatch.setattr(
        client_proxy_module.os,
        "read",
        lambda _fd, _size: chunks.pop(0),
    )
    monkeypatch.setattr(
        client_proxy_module.ipc,
        "write_frame",
        lambda _writer, frame: frames.append(frame),
    )

    await asyncio.wait_for(proxy._forward_stdin_windows(_FakeWriter()), timeout=1.0)

    assert frames == [{"method": "initialize"}]


@pytest.mark.asyncio
async def test_forward_stdin_windows_cancellation_does_not_wait_for_stdin_eof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy = ClientProxy(Path("."), SimpleNamespace())
    checked_pipe = asyncio.Event()

    stdin = SimpleNamespace(buffer=SimpleNamespace(fileno=lambda: 123))
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(
        ClientProxy, "_windows_stdin_handle", staticmethod(lambda _fd: 456)
    )

    def no_data(_handle: int) -> int:
        checked_pipe.set()
        return 0

    monkeypatch.setattr(
        ClientProxy, "_windows_pipe_bytes_available", staticmethod(no_data)
    )

    task = asyncio.create_task(proxy._forward_stdin_windows(_FakeWriter()))
    await asyncio.wait_for(checked_pipe.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1.0)


@pytest.mark.asyncio
async def test_run_prefers_stdout_read_error_over_startup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy, discovery, writer = _make_proxy(tmp_path, monkeypatch)
    stdin_cancelled = asyncio.Event()

    async def blocked_stdin(_writer: asyncio.StreamWriter) -> None:
        try:
            await asyncio.sleep(3600)
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


@pytest.mark.asyncio
async def test_connect_or_startup_failure_raises_runtime_error_on_oserror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OSError from create_client is wrapped in RuntimeError with formatted failure."""
    proxy, discovery, _ = _make_proxy(tmp_path, monkeypatch)
    monkeypatch.setattr(
        client_proxy_module.ipc,
        "create_client",
        AsyncMock(side_effect=OSError("connection refused")),
    )

    with pytest.raises(RuntimeError, match="formatted startup failure"):
        await proxy._connect_or_startup_failure("tcp:127.0.0.1:9999")

    discovery.format_startup_failure.assert_called_once()


@pytest.mark.asyncio
async def test_connect_or_startup_failure_passes_through_on_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path: connect_or_startup_failure returns reader/writer on success."""
    proxy = ClientProxy(tmp_path / "repo", SimpleNamespace())
    proxy._discovery = Mock()
    proxy._discovery.format_startup_failure.return_value = "formatted"
    proxy._discovery.get_daemon_log_path.return_value = tmp_path / "daemon.log"

    reader = asyncio.StreamReader()
    writer = _FakeWriter()
    monkeypatch.setattr(
        client_proxy_module.ipc,
        "create_client",
        AsyncMock(return_value=(reader, writer)),
    )

    result_reader, result_writer = await proxy._connect_or_startup_failure(
        "tcp:127.0.0.1:9000"
    )

    assert result_reader is reader
    assert result_writer is writer


@pytest.mark.asyncio
async def test_run_raises_runtime_error_when_lock_disappears_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """lock=None after connect raises RuntimeError with formatted startup failure."""
    proxy, discovery, writer = _make_proxy(tmp_path, monkeypatch)
    discovery.read_lock.return_value = None

    proxy._forward_stdin_to_socket = AsyncMock()
    proxy._forward_socket_to_stdout = AsyncMock(
        return_value=_SocketForwardResult(message_count=0)
    )

    with pytest.raises(RuntimeError, match="formatted startup failure"):
        await proxy.run()

    discovery.format_startup_failure.assert_called_once()
    assert writer.closed
    assert writer.wait_closed_called


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "handshake_error",
    [
        asyncio.IncompleteReadError(partial=b"", expected=1),
        asyncio.TimeoutError(),
        ConnectionResetError("ack lost"),
    ],
)
async def test_run_wraps_registration_handshake_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    handshake_error: BaseException,
) -> None:
    proxy, discovery, writer = _make_proxy(tmp_path, monkeypatch)
    monkeypatch.setattr(
        client_proxy_module.ipc,
        "read_frame",
        AsyncMock(side_effect=handshake_error),
    )

    with pytest.raises(RuntimeError, match="formatted startup failure"):
        await proxy.run()

    discovery.format_startup_failure.assert_called_once()
    assert writer.closed
    assert writer.wait_closed_called


@pytest.mark.asyncio
async def test_registration_ack_timeout_reports_alive_daemon_not_died(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy, discovery, writer = _make_proxy(tmp_path, monkeypatch)
    discovery.format_startup_failure.side_effect = (
        lambda *, prefix, log_path=None, returncode=None: prefix
    )
    discovery.ping_daemon = AsyncMock(return_value=True)
    monkeypatch.setattr(
        client_proxy_module.ipc,
        "read_frame",
        AsyncMock(side_effect=asyncio.TimeoutError()),
    )

    with pytest.raises(RuntimeError) as exc_info:
        await proxy.run()

    message = str(exc_info.value)
    assert "registration ack timed out" in message
    assert "daemon still reachable" in message
    assert "daemon died" not in message
    assert writer.closed
    assert writer.wait_closed_called


@pytest.mark.asyncio
async def test_run_wraps_registration_write_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy, discovery, writer = _make_proxy(tmp_path, monkeypatch)
    monkeypatch.setattr(
        client_proxy_module.ipc,
        "write_frame",
        Mock(side_effect=BrokenPipeError("write failed")),
    )

    with pytest.raises(RuntimeError, match="formatted startup failure"):
        await proxy.run()

    discovery.format_startup_failure.assert_called_once()
    assert writer.closed
    assert writer.wait_closed_called


# ---------------------------------------------------------------------------
# Orphan watchdog
# ---------------------------------------------------------------------------


def test_client_proxy_captures_initial_parent_pid_on_unix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    getppid = Mock(return_value=2000)
    monkeypatch.setattr(client_proxy_module.os, "getppid", getppid)

    proxy = ClientProxy(tmp_path / "repo", SimpleNamespace())

    assert proxy._initial_ppid == 2000
    getppid.assert_called_once_with()


def test_is_orphaned_detects_pid_1_reparent_on_unix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(client_proxy_module.os, "getppid", Mock(side_effect=[2000, 1]))
    proxy = ClientProxy(tmp_path / "repo", SimpleNamespace())

    assert proxy._is_orphaned() is True


def test_is_orphaned_detects_non_pid_1_parent_change_on_unix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        client_proxy_module.os,
        "getppid",
        Mock(side_effect=[2000, 3000]),
    )
    proxy = ClientProxy(tmp_path / "repo", SimpleNamespace())

    assert proxy._is_orphaned() is True


def test_is_orphaned_ignores_live_parent_on_unix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        client_proxy_module.os,
        "getppid",
        Mock(side_effect=[2000, 2000]),
    )
    proxy = ClientProxy(tmp_path / "repo", SimpleNamespace())

    assert proxy._is_orphaned() is False


def test_is_orphaned_allows_pid_1_when_initial_parent_was_pid_1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        client_proxy_module.os,
        "getppid",
        Mock(side_effect=[1, 1]),
    )
    proxy = ClientProxy(tmp_path / "repo", SimpleNamespace())

    assert proxy._is_orphaned() is False


@pytest.mark.asyncio
async def test_orphan_watchdog_triggers_shutdown_when_orphaned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When _is_orphaned returns True, the watchdog triggers clean shutdown."""
    proxy, discovery, writer = _make_proxy(tmp_path, monkeypatch)
    monkeypatch.setattr(client_proxy_module.ClientProxy, "_ORPHAN_POLL_INTERVAL", 0.0)
    monkeypatch.setattr(proxy, "_is_orphaned", lambda: True)

    stdin_cancelled = asyncio.Event()

    async def blocked_stdin(_writer: asyncio.StreamWriter) -> None:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            stdin_cancelled.set()
            raise

    async def blocked_stdout(_reader: asyncio.StreamReader) -> _SocketForwardResult:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            raise

    proxy._forward_stdin_to_socket = blocked_stdin
    proxy._forward_socket_to_stdout = blocked_stdout

    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr)

    await proxy.run()

    assert stdin_cancelled.is_set()
    assert "Parent process died" in stderr.getvalue()
    assert writer.closed
    assert writer.wait_closed_called


@pytest.mark.asyncio
async def test_run_observes_shutdown_event_when_watchdog_signals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run exits when the watchdog sets the shutdown event."""
    proxy, _, writer = _make_proxy(tmp_path, monkeypatch)

    stdin_cancelled = asyncio.Event()
    stdout_cancelled = asyncio.Event()
    watchdog_cancelled = asyncio.Event()

    async def blocked_stdin(_writer: asyncio.StreamWriter) -> None:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            stdin_cancelled.set()
            raise

    async def blocked_stdout(_reader: asyncio.StreamReader) -> _SocketForwardResult:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            stdout_cancelled.set()
            raise

    async def event_only_watchdog() -> None:
        proxy._shutdown_event.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            watchdog_cancelled.set()
            raise

    proxy._orphan_watchdog = event_only_watchdog
    proxy._forward_stdin_to_socket = blocked_stdin
    proxy._forward_socket_to_stdout = blocked_stdout

    await asyncio.wait_for(proxy.run(), timeout=1.0)

    assert stdin_cancelled.is_set()
    assert stdout_cancelled.is_set()
    assert watchdog_cancelled.is_set()
    assert writer.closed
    assert writer.wait_closed_called


@pytest.mark.asyncio
async def test_orphan_watchdog_canceled_on_clean_shutdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Watchdog is cancelled when stdin closes first (normal operation)."""
    proxy, _, writer = _make_proxy(tmp_path, monkeypatch)
    monkeypatch.setattr(proxy, "_is_orphaned", lambda: False)

    watchdog_cancelled = asyncio.Event()

    async def cancellable_watchdog() -> None:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            watchdog_cancelled.set()
            raise

    proxy._orphan_watchdog = cancellable_watchdog
    proxy._forward_stdin_to_socket = AsyncMock()
    proxy._forward_socket_to_stdout = AsyncMock(
        return_value=_SocketForwardResult(message_count=1)
    )

    await proxy.run()

    assert watchdog_cancelled.is_set()
    assert writer.closed
    assert writer.wait_closed_called


def test_is_orphaned_returns_false_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_is_orphaned returns False on Windows regardless of PPID."""
    monkeypatch.setattr(sys, "platform", "win32")
    getppid = Mock(return_value=1)
    monkeypatch.setattr(client_proxy_module.os, "getppid", getppid)

    proxy = ClientProxy(Path("."), SimpleNamespace())

    assert proxy._initial_ppid is None
    assert proxy._is_orphaned() is False
    getppid.assert_not_called()
