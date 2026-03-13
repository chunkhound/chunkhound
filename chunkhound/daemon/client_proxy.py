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
from pathlib import Path
from typing import Any

from . import ipc
from .discovery import DaemonDiscovery


class ClientProxy:
    """Bridge between Claude's stdio and the ChunkHound daemon IPC socket."""

    def __init__(self, project_dir: Path, args: Any) -> None:
        self._project_dir = project_dir.resolve()
        self._args = args
        self._discovery = DaemonDiscovery(self._project_dir)

    async def run(self) -> None:
        """Connect to the daemon and relay messages until stdin closes."""
        address = await self._discovery.find_or_start_daemon(self._args)

        reader, writer = await ipc.create_client(address)
        try:
            # Read auth token from lock file (written by the daemon)
            lock = self._discovery.read_lock()
            auth_token = lock.get("auth_token") if lock else None

            # Registration handshake
            reg_frame: dict = {"type": "register", "pid": os.getpid()}
            if auth_token is not None:
                reg_frame["auth_token"] = auth_token
            ipc.write_frame(writer, reg_frame)
            await writer.drain()

            ack = await asyncio.wait_for(ipc.read_frame(reader), timeout=10.0)
            if not isinstance(ack, dict) or ack.get("type") != "registered":
                raise RuntimeError(
                    f"Unexpected registration response from daemon: {ack}"
                )

            # Bidirectional forwarding
            # Use wait() with FIRST_COMPLETED so when stdin closes, we immediately
            # close the socket connection rather than waiting for both tasks.
            # This is critical on Windows where proc.terminate() may not cleanly
            # close stdin, leaving the stdin reader blocked.
            stdin_task = asyncio.create_task(self._forward_stdin_to_socket(writer))
            stdout_task = asyncio.create_task(self._forward_socket_to_stdout(reader))

            done, pending = await asyncio.wait(
                {stdin_task, stdout_task}, return_when=asyncio.FIRST_COMPLETED
            )

            # Retrieve exceptions from completed tasks to prevent
            # "Task exception was never retrieved" noise on stderr.
            for task in done:
                if not task.cancelled():
                    exc = task.exception()
                    if exc is not None:
                        msg = f"[chunkhound] client_proxy task error: {exc!r}\n"
                        sys.stderr.write(msg)
                        sys.stderr.flush()

            # Cancel remaining tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _forward_stdin_to_socket(self, writer: asyncio.StreamWriter) -> None:
        """Read JSON lines from stdin, parse, and write as IPC frames to the socket.

        Windows uses a thread-executor approach because ProactorEventLoop's
        connect_read_pipe() requires overlapped-I/O handles, but inherited stdin
        pipes are synchronous from the child's perspective and fail silently.
        Unix uses connect_read_pipe() for true async I/O.
        """
        if sys.platform == "win32":
            await self._forward_stdin_threaded(writer)
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

    async def _forward_stdin_threaded(self, writer: asyncio.StreamWriter) -> None:
        """Windows: thread-based stdin reading via run_in_executor.

        Blocking readline() in a thread pool avoids the IOCP handle
        requirement that makes connect_read_pipe() fail on inherited pipes.
        """
        loop = asyncio.get_running_loop()
        while True:
            line = await loop.run_in_executor(None, sys.stdin.buffer.readline)
            if not line:
                break
            try:
                msg = json.loads(line.decode())
            except json.JSONDecodeError:
                continue
            ipc.write_frame(writer, msg)
            await writer.drain()

    async def _forward_socket_to_stdout(self, reader: asyncio.StreamReader) -> None:
        """Read IPC frames from socket, serialize as JSON lines, write to stdout."""
        while True:
            try:
                msg = await ipc.read_frame(reader)
            except asyncio.IncompleteReadError:
                break
            except Exception as e:
                sys.stderr.write(f"[chunkhound] IPC read error: {e!r}\n")
                sys.stderr.flush()
                break
            line = json.dumps(msg) + "\n"
            sys.stdout.buffer.write(line.encode())
            sys.stdout.buffer.flush()
