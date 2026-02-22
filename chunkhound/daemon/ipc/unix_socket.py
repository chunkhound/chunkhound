"""Unix domain socket IPC transport (Linux/macOS)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any


async def create_server(
    path: str,
    client_connected_cb: Callable[
        [asyncio.StreamReader, asyncio.StreamWriter],
        Coroutine[Any, Any, None],
    ],
) -> asyncio.AbstractServer:
    """Start a Unix domain socket server at *path*."""
    return await asyncio.start_unix_server(client_connected_cb, path=path)


async def create_client(
    path: str,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Connect to a Unix domain socket server at *path*."""
    return await asyncio.open_unix_connection(path)


async def is_connectable(path: str) -> bool:
    """Return True if the socket at *path* accepts connections."""
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(path), timeout=1.0
        )
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        return True
    except Exception:
        return False
