"""IPC transport abstraction — selects Unix sockets or TCP loopback by platform.

Public API:
    create_server(address, cb) -> (server, actual_address)
    create_client(address)     -> (reader, writer)
    is_connectable(address)    -> bool
    read_frame(reader)         -> Any
    write_frame(writer, obj)   -> None

Address format:
    Unix/macOS : /tmp/chunkhound-XXXXXXXX.sock  (filesystem path)
    Windows    : tcp:127.0.0.1:<port>           (TCP loopback)
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable, Coroutine
from typing import Any

from .codec import read_frame, write_frame  # noqa: F401 — re-exported

_WINDOWS = sys.platform == "win32"


def _parse_tcp_address(address: str) -> tuple[str, int]:
    """Parse 'tcp:host:port' into (host, port)."""
    parts = address.split(":", 2)
    return parts[1], int(parts[2])


async def create_server(
    address: str,
    client_connected_cb: Callable[
        [asyncio.StreamReader, asyncio.StreamWriter],
        Coroutine[Any, Any, None],
    ],
) -> tuple[asyncio.AbstractServer, str]:
    """Start the IPC server and return (server, actual_address).

    On Unix, *address* is a socket path and returned unchanged.
    On Windows, *address* is 'tcp:127.0.0.1:0' and the actual port is
    resolved and returned as 'tcp:127.0.0.1:<port>'.
    """
    if _WINDOWS:
        from .windows_pipe import create_server as _ws_create
        host, port = _parse_tcp_address(address)
        server, actual_port = await _ws_create(host, port, client_connected_cb)
        return server, f"tcp:{host}:{actual_port}"
    else:
        from .unix_socket import create_server as _us_create
        server = await _us_create(address, client_connected_cb)
        return server, address


async def create_client(
    address: str,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Connect to the IPC server at *address*."""
    if _WINDOWS:
        from .windows_pipe import create_client as _wp_create
        host, port = _parse_tcp_address(address)
        return await _wp_create(host, port)
    else:
        from .unix_socket import create_client as _us_create
        return await _us_create(address)


async def is_connectable(address: str) -> bool:
    """Return True if *address* is reachable."""
    if _WINDOWS:
        from .windows_pipe import is_connectable as _wp_conn
        host, port = _parse_tcp_address(address)
        return await _wp_conn(host, port)
    else:
        from .unix_socket import is_connectable as _us_conn
        return await _us_conn(address)
