"""TCP loopback IPC transport for Windows.

Named pipes require pywin32 which is not a dependency.  TCP on 127.0.0.1 is
the accepted local-only alternative for Windows dev tooling.  The port is
OS-assigned (bind to port 0) and written into the lock file.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any


async def create_server(
    host: str,
    port: int,
    client_connected_cb: Callable[
        [asyncio.StreamReader, asyncio.StreamWriter],
        Coroutine[Any, Any, None],
    ],
) -> tuple[asyncio.AbstractServer, int]:
    """Start a TCP server on *host*:*port* (use 0 to let OS assign).

    Returns:
        (server, assigned_port)
    """
    server = await asyncio.start_server(client_connected_cb, host=host, port=port)
    socks = server.sockets
    assigned_port = socks[0].getsockname()[1] if socks else port
    return server, assigned_port


async def create_client(
    host: str,
    port: int,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Connect to a TCP server at *host*:*port*."""
    return await asyncio.open_connection(host, port)


async def is_connectable(host: str, port: int) -> bool:
    """Return True if TCP *host*:*port* accepts connections."""
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=1.0
        )
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        return True
    except Exception:
        return False
