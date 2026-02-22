"""Client session tracking and lifecycle management for the daemon."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field

_PID_POLL_INTERVAL = 60.0  # seconds between hung-process checks


@dataclass
class ClientSession:
    """Represents a single connected MCP proxy client."""

    client_id: str
    pid: int
    writer: asyncio.StreamWriter
    connected_at: float = field(default_factory=time.time)


class ClientManager:
    """Manages active client sessions and triggers shutdown when empty.

    Args:
        on_empty: Sync callback invoked when the last client disconnects.
    """

    def __init__(self, on_empty: Callable[[], None]) -> None:
        self._sessions: dict[str, ClientSession] = {}
        self._on_empty = on_empty
        self._lock = asyncio.Lock()

    def register(
        self, client_id: str, pid: int, writer: asyncio.StreamWriter
    ) -> ClientSession:
        """Register a new client and return its session object."""
        session = ClientSession(client_id=client_id, pid=pid, writer=writer)
        self._sessions[client_id] = session
        return session

    def remove(self, client_id: str) -> None:
        """Remove a client session and trigger shutdown if no clients remain."""
        self._sessions.pop(client_id, None)
        if not self._sessions:
            self._on_empty()

    def count(self) -> int:
        """Return the number of currently connected clients."""
        return len(self._sessions)

    def _pid_alive(self, pid: int) -> bool:
        if sys.platform == "win32":
            import psutil
            return psutil.pid_exists(pid)
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    async def poll_pids(self) -> None:
        """Background task: evict clients whose proxy process has died.

        Runs on a 60-second interval until cancelled.
        """
        while True:
            await asyncio.sleep(_PID_POLL_INTERVAL)
            dead = [
                cid
                for cid, sess in list(self._sessions.items())
                if not self._pid_alive(sess.pid)
            ]
            for cid in dead:
                sess = self._sessions.get(cid)
                if sess is None:
                    continue
                try:
                    sess.writer.close()
                    await sess.writer.wait_closed()
                except Exception:
                    pass
                self.remove(cid)
