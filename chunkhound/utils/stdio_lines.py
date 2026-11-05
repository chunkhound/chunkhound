"""Cross-platform stdin line reader / stdout line writer.

Shared by ClientProxy (stdio<->IPC relay) and StdioMCPServer (--no-daemon
direct SDK serving) so the Windows-safe stdin handling exists in one place.
Zero chunkhound domain logic, zero mcp imports — pure OS I/O.
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import AsyncIterator

_WINDOWS_STDIN_POLL_INTERVAL = 0.01
_WINDOWS_STDIN_READ_CHUNK_SIZE = 65536
_WINDOWS_PIPE_CLOSED_ERRORS = {6, 38, 109}  # invalid handle, EOF, broken pipe


async def iter_stdin_lines() -> AsyncIterator[bytes]:
    """Yield raw lines (including trailing \\n) from stdin until EOF."""
    if sys.platform == "win32":
        async for line in _iter_stdin_lines_windows():
            yield line
    else:
        async for line in _iter_stdin_lines_unix():
            yield line


async def _iter_stdin_lines_unix() -> AsyncIterator[bytes]:
    """Unix: true async I/O via connect_read_pipe."""
    loop = asyncio.get_running_loop()
    stdin = asyncio.StreamReader()
    transport, _ = await loop.connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(stdin), sys.stdin.buffer
    )
    try:
        while True:
            line = await stdin.readline()
            if not line:
                return
            yield line
    finally:
        transport.close()


async def _iter_stdin_lines_windows() -> AsyncIterator[bytes]:
    """Windows: poll the synchronous pipe so cancellation never waits for EOF."""
    fd = sys.stdin.buffer.fileno()
    handle = _windows_stdin_handle(fd)
    buffer = bytearray()

    while True:
        available = _windows_pipe_bytes_available(handle)
        if available is None:
            if buffer:
                yield bytes(buffer)
            return
        if available == 0:
            await asyncio.sleep(_WINDOWS_STDIN_POLL_INTERVAL)
            continue

        chunk = os.read(fd, min(available, _WINDOWS_STDIN_READ_CHUNK_SIZE))
        if not chunk:
            if buffer:
                yield bytes(buffer)
            return

        buffer.extend(chunk)
        while True:
            newline = buffer.find(b"\n")
            if newline < 0:
                break
            yield bytes(buffer[: newline + 1])
            del buffer[: newline + 1]


def _windows_stdin_handle(fd: int) -> int:
    import importlib

    msvcrt = importlib.import_module("msvcrt")
    return int(getattr(msvcrt, "get_osfhandle")(fd))


def _windows_pipe_bytes_available(handle: int) -> int | None:
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


def write_stdout_line(line: bytes) -> None:
    """Write one line (caller includes trailing \\n) to stdout and flush."""
    sys.stdout.buffer.write(line)
    sys.stdout.buffer.flush()
