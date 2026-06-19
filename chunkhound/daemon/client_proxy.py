"""Stdio ↔ IPC bridge — the lightweight proxy run by each Claude instance.

Each ``chunkhound mcp`` invocation (in daemon mode) becomes a ClientProxy that:
1. Discovers or starts the daemon.
2. Connects to the daemon via the IPC transport (Unix socket or TCP loopback).
3. Performs the registration handshake.
4. Bidirectionally forwards stdin ↔ IPC and IPC ↔ stdout.

The proxy actively encodes/decodes: it bridges two different transports:
    stdio (JSON-RPC newline-delimited) ↔ IPC (length-prefixed msgpack/JSON frames)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import ipc
from .discovery import DaemonDiscovery

_WINDOWS_STDIN_POLL_INTERVAL = 0.01
_WINDOWS_STDIN_READ_CHUNK_SIZE = 65536
_WINDOWS_PIPE_CLOSED_ERRORS = {6, 38, 109}  # invalid handle, EOF, broken pipe


@dataclass(frozen=True)
class _SocketForwardResult:
    message_count: int


class ClientProxy:
    """Bridge between Claude's stdio and the ChunkHound daemon IPC socket."""

    _ORPHAN_POLL_INTERVAL = 5.0

    def __init__(self, project_dir: Path, args: Any) -> None:
        self._project_dir = project_dir.resolve()
        self._args = args
        self._discovery = DaemonDiscovery(self._project_dir)
        self._initial_ppid = None if sys.platform == "win32" else os.getppid()

    async def _connect_or_startup_failure(
        self, address: str
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Connect to *address* or surface a daemon connection failure.

        The daemon may exit between ``find_or_start_daemon()`` returning
        and this connection attempt (e.g. startup barrier failure or
        normal shutdown after the last client disconnected).  Catch
        ``OSError`` and format a caller-visible diagnostic error.
        """
        try:
            return await ipc.create_client(address)
        except OSError:
            raise RuntimeError(
                self._discovery.format_startup_failure(
                    prefix=(
                        "ChunkHound daemon IPC connection failed \u2014 daemon "
                        "may have shut down before accepting"
                    ),
                    log_path=self._discovery.get_daemon_log_path(),
                )
            )

    def _startup_failure_error(self, prefix: str) -> RuntimeError:
        """Build the standard caller-visible startup failure wrapper."""
        return RuntimeError(
            self._discovery.format_startup_failure(
                prefix=prefix,
                log_path=self._discovery.get_daemon_log_path(),
            )
        )

    async def _probe_daemon_liveness(self) -> bool | None:
        """Best-effort daemon liveness probe for registration diagnostics."""
        try:
            return await self._discovery.ping_daemon()
        except Exception:
            return None

    async def _register_with_daemon(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Complete the auth-token registration handshake."""
        lock = self._discovery.read_lock()
        if lock is None:
            raise self._startup_failure_error(
                "ChunkHound daemon IPC lock file removed before registration — "
                "daemon may have shut down"
            )
        auth_token = lock.get("auth_token")

        reg_frame: dict = {"type": "register", "pid": os.getpid()}
        if auth_token is not None:
            reg_frame["auth_token"] = auth_token

        try:
            ipc.write_frame(writer, reg_frame)
            await writer.drain()
        except (
            OSError,
            EOFError,
            asyncio.IncompleteReadError,
            BrokenPipeError,
            ConnectionAbortedError,
            ConnectionResetError,
        ) as error:
            raise self._startup_failure_error(
                "ChunkHound daemon died during registration handshake"
            ) from error

        try:
            ack = await asyncio.wait_for(ipc.read_frame(reader), timeout=10.0)
        except asyncio.TimeoutError as error:
            daemon_alive = await self._probe_daemon_liveness()
            if daemon_alive is True:
                prefix = (
                    "ChunkHound daemon registration ack timed out after 10.0s; "
                    "daemon still reachable"
                )
            elif daemon_alive is False:
                prefix = (
                    "ChunkHound daemon registration ack timed out after 10.0s; "
                    "daemon no longer reachable"
                )
            else:
                prefix = (
                    "ChunkHound daemon registration ack timed out after 10.0s; "
                    "daemon reachability could not be confirmed"
                )
            raise self._startup_failure_error(prefix) from error
        except (
            OSError,
            EOFError,
            asyncio.IncompleteReadError,
            BrokenPipeError,
            ConnectionAbortedError,
            ConnectionResetError,
        ) as error:
            raise self._startup_failure_error(
                "ChunkHound daemon connection closed before registration ack"
            ) from error

        if not isinstance(ack, dict) or ack.get("type") != "registered":
            raise RuntimeError(f"Unexpected registration response from daemon: {ack}")

    def _is_orphaned(self) -> bool:
        """Return True if the original parent process no longer owns this proxy.

        On Unix, an orphaned process may be reparented to PID 1 (init/launchd)
        or to a child subreaper. Treat any parent PID change from startup as
        orphaning. On Windows, the existing pipe-close detection is sufficient.
        """
        if sys.platform == "win32":
            return False
        current_ppid = os.getppid()
        return current_ppid != self._initial_ppid

    async def _orphan_watchdog(self) -> None:
        """Poll parent PID; signal shutdown when orphaned.

        Sets the shutdown event and completes cleanly when the parent dies —
        the caller's asyncio.wait (FIRST_COMPLETED) will then cancel
        stdin/stdout forwarding and trigger normal cleanup.
        """
        while True:
            await asyncio.sleep(self._ORPHAN_POLL_INTERVAL)
            if self._is_orphaned():
                print(
                    "Parent process died — shutting down orphaned proxy",
                    file=sys.stderr,
                )
                self._shutdown_event.set()
                return

    async def run(self) -> None:
        """Connect to the daemon and relay messages until stdin closes."""
        address = await self._discovery.find_or_start_daemon(self._args)

        # Between find_or_start_daemon() and the registration handshake, the
        # daemon may have started graceful shutdown and removed published
        # artifacts (ASAP cleanup). Catch errors and surface a caller-visible
        # startup failure rather than a raw OSError or EOF.
        reader, writer = await self._connect_or_startup_failure(address)

        try:
            await self._register_with_daemon(reader, writer)

            # Bidirectional forwarding with orphan watchdog
            # Use wait() with FIRST_COMPLETED so when stdin closes or the parent
            # dies, we immediately close the socket connection rather than waiting
            # for both tasks.
            # This is critical on Windows where proc.terminate() may not cleanly
            # close stdin, leaving the stdin reader blocked.
            self._shutdown_event = asyncio.Event()
            stdin_task = asyncio.create_task(self._forward_stdin_to_socket(writer))
            stdout_task = asyncio.create_task(self._forward_socket_to_stdout(reader))
            watchdog_task = asyncio.create_task(self._orphan_watchdog())
            shutdown_task = asyncio.create_task(self._shutdown_event.wait())

            _, pending = await asyncio.wait(
                {stdin_task, stdout_task, watchdog_task, shutdown_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

            stdout_error = self._task_error(stdout_task)
            stdin_error = self._task_error(stdin_task)
            stdout_result = self._task_result(stdout_task)
            closed_before_mcp = (
                stdout_error is None
                and isinstance(stdout_result, _SocketForwardResult)
                and stdout_result.message_count == 0
            )
            if closed_before_mcp and (
                stdin_error is None or self._is_ipc_shutdown_error(stdin_error)
            ):
                raise RuntimeError(
                    self._discovery.format_startup_failure(
                        prefix=(
                            "ChunkHound daemon closed the IPC connection before "
                            "serving any MCP traffic"
                        ),
                        log_path=self._discovery.get_daemon_log_path(),
                    )
                )
            if stdout_error is not None:
                raise stdout_error
            if stdin_error is not None:
                raise stdin_error
        finally:
            # Nudge stdin toward EOF during normal cleanup without making
            # process exit depend on the peer closing its pipe first.
            try:
                sys.stdin.buffer.close()
            except Exception:
                pass
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    @staticmethod
    def _task_error(task: asyncio.Task[Any]) -> BaseException | None:
        """Return the finished task error without surfacing cancellations."""
        if not task.done() or task.cancelled():
            return None
        return task.exception()

    @staticmethod
    def _task_result(task: asyncio.Task[Any]) -> Any | None:
        """Return the finished task result when it completed successfully."""
        if not task.done() or task.cancelled():
            return None
        if task.exception() is not None:
            return None
        return task.result()

    @staticmethod
    def _is_ipc_shutdown_error(error: BaseException) -> bool:
        """Treat transport resets as daemon-closure noise during startup failure."""
        return isinstance(
            error,
            (BrokenPipeError, ConnectionAbortedError, ConnectionResetError),
        )

    async def _forward_stdin_to_socket(self, writer: asyncio.StreamWriter) -> None:
        """Read JSON lines from stdin, parse, and write as IPC frames to the socket.

        Windows polls the inherited synchronous pipe before reading because
        ProactorEventLoop requires overlapped-I/O handles. Unix uses
        connect_read_pipe() for true async I/O.
        """
        if sys.platform == "win32":
            await self._forward_stdin_windows(writer)
        else:
            await self._forward_stdin_async(writer)

    async def _forward_stdin_async(self, writer: asyncio.StreamWriter) -> None:
        """Unix: async stdin reading via connect_read_pipe."""
        loop = asyncio.get_running_loop()
        stdin = asyncio.StreamReader()
        transport, _ = await loop.connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(stdin), sys.stdin.buffer
        )
        try:
            while True:
                line = await stdin.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line.decode())
                except json.JSONDecodeError:
                    continue
                ipc.write_frame(writer, msg)
                await writer.drain()
        finally:
            transport.close()

    async def _forward_stdin_windows(self, writer: asyncio.StreamWriter) -> None:
        """Windows: poll synchronous stdin so cancellation never waits for EOF."""
        fd = sys.stdin.buffer.fileno()
        handle = self._windows_stdin_handle(fd)
        buffer = bytearray()

        while True:
            available = self._windows_pipe_bytes_available(handle)
            if available is None:
                if buffer:
                    await self._write_stdin_line(writer, bytes(buffer))
                break
            if available == 0:
                await asyncio.sleep(_WINDOWS_STDIN_POLL_INTERVAL)
                continue

            chunk = os.read(fd, min(available, _WINDOWS_STDIN_READ_CHUNK_SIZE))
            if not chunk:
                if buffer:
                    await self._write_stdin_line(writer, bytes(buffer))
                break

            buffer.extend(chunk)
            while True:
                newline = buffer.find(b"\n")
                if newline < 0:
                    break
                line = bytes(buffer[: newline + 1])
                del buffer[: newline + 1]
                await self._write_stdin_line(writer, line)

    @staticmethod
    def _windows_stdin_handle(fd: int) -> int:
        """Return the Windows OS handle for stdin fd."""
        import importlib

        msvcrt = importlib.import_module("msvcrt")
        return int(getattr(msvcrt, "get_osfhandle")(fd))

    @staticmethod
    def _windows_pipe_bytes_available(handle: int) -> int | None:
        """Return available pipe bytes, or None when the pipe has closed."""
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        available = ctypes.c_ulong()
        ok = kernel32.PeekNamedPipe(
            ctypes.c_void_p(handle),
            None,
            0,
            None,
            ctypes.byref(available),
            None,
        )
        if ok:
            return int(available.value)
        error = int(getattr(ctypes, "get_last_error")())
        if error in _WINDOWS_PIPE_CLOSED_ERRORS:
            return None
        raise OSError(error, "PeekNamedPipe failed")

    async def _write_stdin_line(
        self,
        writer: asyncio.StreamWriter,
        line: bytes,
    ) -> None:
        """Parse one JSON-RPC stdin line and forward valid messages."""
        try:
            msg = json.loads(line.decode())
        except json.JSONDecodeError:
            return
        ipc.write_frame(writer, msg)
        await writer.drain()

    async def _forward_socket_to_stdout(
        self, reader: asyncio.StreamReader
    ) -> _SocketForwardResult:
        """Read IPC frames from socket, serialize as JSON lines, write to stdout."""
        message_count = 0
        while True:
            try:
                msg = await ipc.read_frame(reader)
            except asyncio.IncompleteReadError:
                return _SocketForwardResult(
                    message_count=message_count,
                )
            line = json.dumps(msg) + "\n"
            sys.stdout.buffer.write(line.encode())
            sys.stdout.buffer.flush()
            message_count += 1
