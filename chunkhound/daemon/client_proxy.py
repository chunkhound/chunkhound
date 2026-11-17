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

from chunkhound.utils import stdio_lines

from . import ipc
from .discovery import DaemonDiscovery


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
        """Read JSON lines from stdin and write them as IPC frames to the socket."""
        async for msg in stdio_lines.iter_json_frames():
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
            if not stdio_lines.is_object_frame(msg):
                continue
            line = json.dumps(msg) + "\n"
            stdio_lines.write_stdout_line(line.encode())
            message_count += 1
