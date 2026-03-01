"""Daemon discovery via lock file and IPC transport.

Handles:
- Lock file creation/reading in <project>/.chunkhound/daemon.lock
- IPC address derivation from project directory hash (platform-specific)
- Daemon liveness checking (PID + connectivity)
- Starting a new daemon subprocess when none is running
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .process import pid_alive

# Lock file relative to project directory
_LOCK_FILE_REL = ".chunkhound/daemon.lock"
# Starter lock — prevents two proxies from both spawning a daemon simultaneously
_STARTER_LOCK_REL = ".chunkhound/daemon.starter.lock"
# Socket directory (Linux/macOS)
_SOCKET_DIR = "/tmp"
# Startup polling interval and timeout
_STARTUP_POLL_INTERVAL = 0.1
_STARTUP_TIMEOUT = 30.0


class DaemonDiscovery:
    """Locate or start the daemon for a given project directory."""

    def __init__(self, project_dir: Path) -> None:
        self._project_dir = project_dir.resolve()

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def get_lock_path(self) -> Path:
        """Return the absolute path of the lock file."""
        return self._project_dir / _LOCK_FILE_REL

    def get_starter_lock_path(self) -> Path:
        """Return the absolute path of the starter lock file."""
        return self._project_dir / _STARTER_LOCK_REL

    def get_ipc_address(self) -> str:
        """Derive a unique IPC address from the project directory.

        Linux/macOS: Unix socket path in /tmp.
        Windows: TCP loopback address with port 0 (OS-assigned at bind time).
        """
        digest = hashlib.sha256(str(self._project_dir).encode()).hexdigest()[:8]
        if sys.platform == "win32":
            return "tcp:127.0.0.1:0"
        return os.path.join(_SOCKET_DIR, f"chunkhound-{digest}.sock")

    def get_socket_path(self) -> str:
        """Compatibility alias for get_ipc_address()."""
        return self.get_ipc_address()

    # ------------------------------------------------------------------
    # Lock file I/O
    # ------------------------------------------------------------------

    def read_lock(self) -> dict[str, Any] | None:
        """Read and parse the lock file.

        Returns:
            Dict with keys ``pid``, ``socket_path``, ``started_at``,
            ``auth_token``, or ``None`` if the file does not exist or is corrupt.
        """
        lock_path = self.get_lock_path()
        try:
            with open(lock_path) as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
                return None
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

    def write_lock(
        self, pid: int, socket_path: str, auth_token: str | None = None
    ) -> None:
        """Write the lock file atomically.

        If *auth_token* is provided it is written as-is; otherwise a fresh
        token is generated.  On POSIX the file is chmod'd to 0o600 so only
        the owning user can read the auth token.
        """
        lock_path = self.get_lock_path()
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "pid": pid,
            "socket_path": socket_path,
            "started_at": time.time(),
            "auth_token": (
                auth_token if auth_token is not None else secrets.token_hex(32)
            ),
        }
        tmp_path = lock_path.with_suffix(".lock.tmp")
        with open(tmp_path, "w") as f:
            json.dump(data, f)
        tmp_path.replace(lock_path)  # replace() is atomic and overwrites on Windows
        if sys.platform != "win32":
            try:
                os.chmod(lock_path, 0o600)
            except OSError:
                pass

    def remove_lock(self) -> None:
        """Remove the lock file, ignoring errors if it does not exist."""
        try:
            self.get_lock_path().unlink()
        except FileNotFoundError:
            pass

    # ------------------------------------------------------------------
    # Starter lock (prevents duplicate-daemon race on concurrent proxy start)
    # ------------------------------------------------------------------

    def _acquire_starter_lock(self) -> bool:
        """Try to atomically acquire the starter lock.

        Uses ``open(..., 'x')`` which maps to ``O_CREAT|O_EXCL`` — atomic on
        POSIX and Windows NTFS.  If the lock file already exists, checks whether
        the stored PID is still alive; if it is dead (stale lock), removes the
        file and retries once.

        Returns:
            True if this process acquired the lock, False if another live
            process already holds it.
        """
        starter_path = self.get_starter_lock_path()
        starter_path.parent.mkdir(parents=True, exist_ok=True)

        for attempt in range(2):
            try:
                with open(starter_path, "x") as f:
                    json.dump({"pid": os.getpid()}, f)
                return True
            except FileExistsError:
                # Check whether the holder is still alive
                try:
                    with open(starter_path) as f:
                        holder = json.load(f)
                    holder_pid = holder.get("pid", 0)
                    if isinstance(holder_pid, int) and pid_alive(holder_pid):
                        return False  # Another live process holds the lock
                    # Stale — remove and retry once
                    starter_path.unlink(missing_ok=True)
                except (FileNotFoundError, json.JSONDecodeError, OSError):
                    # Race: file disappeared between exists-check and open
                    if attempt == 0:
                        continue
                    return False

        return False

    def _release_starter_lock(self) -> None:
        """Release the starter lock if this process holds it."""
        starter_path = self.get_starter_lock_path()
        try:
            with open(starter_path) as f:
                holder = json.load(f)
            if holder.get("pid") == os.getpid():
                starter_path.unlink(missing_ok=True)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass

    # ------------------------------------------------------------------
    # Liveness checks
    # ------------------------------------------------------------------

    async def _socket_connectable(self, address: str) -> bool:
        """Try to open a short-lived connection to the IPC address.

        Returns True if the server accepts connections.
        """
        from . import ipc
        return await ipc.is_connectable(address)

    def is_daemon_alive(self) -> bool:
        """Synchronous liveness check: PID alive AND address reachable (file or TCP).

        Full socket connectivity is verified asynchronously in
        ``find_or_start_daemon``.
        """
        lock = self.read_lock()
        if lock is None:
            return False
        pid = lock.get("pid")
        socket_path = lock.get("socket_path", "")
        if not isinstance(pid, int) or not pid_alive(pid):
            return False
        # On Unix verify the socket file exists; on Windows trust the PID check
        if sys.platform != "win32" and not str(socket_path).startswith("tcp:"):
            return os.path.exists(socket_path)
        return True

    # ------------------------------------------------------------------
    # Daemon startup
    # ------------------------------------------------------------------

    def _start_daemon_subprocess(self, args: Any) -> None:
        """Launch the daemon as a detached subprocess.

        The new process runs ``chunkhound _daemon`` with the same configuration
        flags as the current invocation.  It is started in a new session so it
        outlives the proxy process.

        Args:
            args: Original CLI ``argparse.Namespace`` from the proxy invocation.
        """
        socket_path = self.get_ipc_address()

        cmd = [
            sys.executable, "-m", "chunkhound.api.cli.main",
            "_daemon",
            "--project-dir", str(self._project_dir),
            "--socket-path", socket_path,
        ]

        # Forward relevant config flags if present
        flag_map = {
            "config": "--config",
            "db_path": "--db-path",
            "embedding_provider": "--embedding-provider",
            "embedding_model": "--embedding-model",
            "embedding_api_key": "--embedding-api-key",
            "embedding_base_url": "--embedding-base-url",
            "embedding_batch_size": "--embedding-batch-size",
            "llm_provider": "--llm-provider",
        }
        for attr, flag in flag_map.items():
            val = getattr(args, attr, None)
            if val is not None:
                cmd += [flag, str(val)]

        if getattr(args, "debug", False):
            cmd.append("--debug")

        env = os.environ.copy()
        env["CHUNKHOUND_DAEMON_MODE"] = "true"

        # Route daemon stdout/stderr to a log file so startup failures are
        # diagnosable (especially on Windows where the IPC transport may fail).
        log_path = self._project_dir / ".chunkhound" / "daemon.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as log_file:
            subprocess.Popen(
                cmd,
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=log_file,
                env=env,
                cwd=str(self._project_dir),
            )

    async def find_or_start_daemon(self, args: Any) -> str:
        """Return the IPC address of the running daemon, starting one if needed.

        Uses an atomic starter lock to prevent two proxies from simultaneously
        spawning duplicate daemons (TOCTOU race).  The proxy that acquires the
        lock is the sole daemon starter; others skip straight to polling.

        On Windows the port is OS-assigned at bind time and stored in the lock
        file, so we always read the lock file to obtain the actual address
        rather than using the pre-computed address.

        Args:
            args: CLI ``argparse.Namespace`` — forwarded to subprocess if daemon
                  needs to be started.

        Returns:
            The IPC address that the daemon is listening on.

        Raises:
            RuntimeError: If the daemon does not become reachable within
                          ``_STARTUP_TIMEOUT`` seconds.
        """
        initial_address = self.get_ipc_address()

        lock = self.read_lock()
        if lock is not None:
            pid = lock.get("pid")
            sp_str = str(lock.get("socket_path", initial_address))
            if isinstance(pid, int) and pid_alive(pid):
                # Daemon process is live — wait for its address to become
                # connectable (it may still be initializing).
                deadline = time.monotonic() + _STARTUP_TIMEOUT
                while time.monotonic() < deadline:
                    if await self._socket_connectable(sp_str):
                        return sp_str
                    await asyncio.sleep(_STARTUP_POLL_INTERVAL)
                raise RuntimeError(
                    f"ChunkHound daemon (pid={pid}) did not become reachable "
                    f"within {_STARTUP_TIMEOUT}s (address: {sp_str})"
                )
            # Stale lock (PID dead) — remove before starting fresh daemon
            self.remove_lock()
            # Remove stale Unix socket file if it lingers
            if sys.platform != "win32" and not initial_address.startswith("tcp:"):
                try:
                    os.unlink(initial_address)
                except FileNotFoundError:
                    pass

        # No live daemon — race to become the sole daemon starter.
        if self._acquire_starter_lock():
            try:
                self._start_daemon_subprocess(args)
            finally:
                # Release after daemon is running (or failed to start); poll
                # loop below handles the "not yet connectable" case.
                # We keep the lock until the daemon wrote its lock file so that
                # concurrent proxies that lose the race don't prematurely try to
                # spawn a second daemon — they'll poll in the section below.
                pass
            # Poll until connectable; read the lock file to get the actual address
            # (critical on Windows where the port is only known after daemon binds)
            deadline = time.monotonic() + _STARTUP_TIMEOUT
            try:
                while time.monotonic() < deadline:
                    lock = self.read_lock()
                    if lock is not None:
                        actual_address = str(lock.get("socket_path", initial_address))
                        if await self._socket_connectable(actual_address):
                            return actual_address
                    await asyncio.sleep(_STARTUP_POLL_INTERVAL)
            finally:
                self._release_starter_lock()

            raise RuntimeError(
                f"ChunkHound daemon did not start within {_STARTUP_TIMEOUT}s "
                f"(address: {initial_address})"
            )
        else:
            # Another proxy is starting the daemon — poll until it's ready.
            deadline = time.monotonic() + _STARTUP_TIMEOUT
            while time.monotonic() < deadline:
                lock = self.read_lock()
                if lock is not None:
                    pid = lock.get("pid")
                    actual_address = str(lock.get("socket_path", initial_address))
                    if isinstance(pid, int) and pid_alive(pid):
                        if await self._socket_connectable(actual_address):
                            return actual_address
                await asyncio.sleep(_STARTUP_POLL_INTERVAL)

            raise RuntimeError(
                f"ChunkHound daemon did not become reachable within "
                f"{_STARTUP_TIMEOUT}s (address: {initial_address})"
            )
