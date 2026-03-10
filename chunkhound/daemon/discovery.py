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
from dataclasses import dataclass
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


@dataclass(slots=True)
class DaemonStartupHandle:
    """Process handle and diagnostics surface for a daemon startup attempt."""

    process: subprocess.Popen[Any]
    log_path: Path


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

    def get_daemon_log_path(self) -> Path:
        """Return the daemon log path used for startup diagnostics."""
        return self._project_dir / ".chunkhound" / "daemon.log"

    def _read_json_file(self, path: Path) -> dict[str, Any] | None:
        try:
            with open(path) as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
                return None
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

    # ------------------------------------------------------------------
    # Lock file I/O
    # ------------------------------------------------------------------

    def read_lock(self) -> dict[str, Any] | None:
        """Read and parse the lock file.

        Returns:
            Dict with keys ``pid``, ``socket_path``, ``started_at``,
            ``auth_token``, or ``None`` if the file does not exist or is corrupt.
        """
        return self._read_json_file(self.get_lock_path())

    def read_starter_lock(self) -> dict[str, Any] | None:
        """Read and parse the starter lock file."""
        return self._read_json_file(self.get_starter_lock_path())

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

    def _tail_daemon_log(
        self, log_path: Path | None = None, *, max_lines: int = 20
    ) -> str | None:
        """Return the recent daemon log tail, if one is available."""
        path = log_path or self.get_daemon_log_path()
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return None
        if not text:
            return None
        return "\n".join(text.splitlines()[-max_lines:])

    def _format_startup_failure(
        self,
        *,
        prefix: str,
        log_path: Path | None = None,
        returncode: int | None = None,
    ) -> str:
        """Build a fast-fail startup error with optional daemon log context."""
        message = prefix
        if returncode is not None:
            message = f"{message} (exit code {returncode})"
        log_tail = self._tail_daemon_log(log_path)
        if log_tail:
            return f"{message}\nRecent daemon log output:\n{log_tail}"
        return message

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

    @staticmethod
    def _build_forwarded_args(args: Any) -> list[str]:
        """Rebuild daemon-compatible argv tokens from the parsed args namespace.

        Both ``mcp`` and ``_daemon`` parsers register identical config flags
        via ``add_config_arguments()``.  By introspecting the daemon parser's
        own action list we forward every flag — including those added in future
        — without maintaining a hand-written map that can fall out of sync.

        Args:
            args: Parsed ``argparse.Namespace`` from the proxy (mcp) invocation.

        Returns:
            List of flag tokens ready to append to the daemon subprocess cmd.
        """
        import argparse as _ap

        from chunkhound.api.cli.parsers.daemon_parser import add_daemon_subparser

        # Build a temporary daemon parser solely for introspection.
        _tmp = _ap.ArgumentParser()
        daemon_parser = add_daemon_subparser(_tmp.add_subparsers())

        # These dests are daemon-specific positional/required args that have
        # no equivalent in the mcp parser and are already handled explicitly.
        _skip_dests = {"project_dir", "socket_path", "help"}

        forwarded: list[str] = []
        for action in daemon_parser._actions:
            if not action.option_strings:
                continue  # positional — skip
            dest = action.dest
            if dest in _skip_dests:
                continue
            val = getattr(args, dest, None)
            if val is None:
                continue
            flag = action.option_strings[0]
            if action.const is True:
                # store_true: only add the flag when the value is True
                if val:
                    forwarded.append(flag)
            elif action.const is False:
                # store_false: only add the flag when the value is False
                if not val:
                    forwarded.append(flag)
            elif isinstance(val, list):
                # append action (e.g. --include / --exclude)
                for item in val:
                    forwarded.extend([flag, str(item)])
            else:
                # Regular store action — forward only when explicitly set
                # (i.e. different from the action's declared default).
                if val != action.default:
                    forwarded.extend([flag, str(val)])

        return forwarded

    def _start_daemon_subprocess(self, args: Any) -> DaemonStartupHandle:
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

        # Forward all config flags by introspecting the daemon parser's own
        # action definitions.  Both `mcp` and `_daemon` register the same
        # config arguments via add_config_arguments(), so iterating the daemon
        # parser ensures every current and future flag is forwarded without
        # maintaining a hand-written flag_map.
        cmd.extend(self._build_forwarded_args(args))

        env = os.environ.copy()
        env["CHUNKHOUND_DAEMON_MODE"] = "true"

        # Route daemon stdout/stderr to a log file so startup failures are
        # diagnosable (especially on Windows where the IPC transport may fail).
        log_path = self.get_daemon_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as log_file:
            process = subprocess.Popen(
                cmd,
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=log_file,
                env=env,
                cwd=str(self._project_dir),
            )
        return DaemonStartupHandle(process=process, log_path=log_path)

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
                    if not pid_alive(pid):
                        raise RuntimeError(
                            self._format_startup_failure(
                                prefix=(
                                    "ChunkHound daemon exited before it became "
                                    f"reachable (pid={pid}, address: {sp_str})"
                                )
                            )
                        )
                    await asyncio.sleep(_STARTUP_POLL_INTERVAL)
                raise RuntimeError(
                    self._format_startup_failure(
                        prefix=(
                            f"ChunkHound daemon (pid={pid}) did not become "
                            f"reachable within {_STARTUP_TIMEOUT}s "
                            f"(address: {sp_str})"
                        )
                    )
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
            startup: DaemonStartupHandle
            try:
                startup = self._start_daemon_subprocess(args)
            finally:
                # The starter lock is NOT released here; it is held until the polling
                # loop's finally block below (line ~387) so concurrent proxies don't
                # race to spawn a second daemon. If _start_daemon_subprocess() raises,
                # the lock leaks until the outer finally fires or stale-PID cleanup
                # on the next invocation — acceptable, as the lock is process-scoped.
                pass
            # Poll until connectable; read the lock file to get the actual address
            # (critical on Windows where the port is only known after daemon binds)
            deadline = time.monotonic() + _STARTUP_TIMEOUT
            try:
                while time.monotonic() < deadline:
                    returncode = startup.process.poll()
                    if returncode is not None:
                        raise RuntimeError(
                            self._format_startup_failure(
                                prefix=(
                                    "ChunkHound daemon exited before it became "
                                    "reachable"
                                ),
                                log_path=startup.log_path,
                                returncode=returncode,
                            )
                        )
                    lock = self.read_lock()
                    if lock is not None:
                        actual_address = str(lock.get("socket_path", initial_address))
                        if await self._socket_connectable(actual_address):
                            return actual_address
                    await asyncio.sleep(_STARTUP_POLL_INTERVAL)
            finally:
                self._release_starter_lock()

            raise RuntimeError(
                self._format_startup_failure(
                    prefix=(
                        f"ChunkHound daemon did not start within {_STARTUP_TIMEOUT}s "
                        f"(address: {initial_address})"
                    ),
                    log_path=startup.log_path,
                )
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
                starter_lock = self.read_starter_lock()
                starter_lock_path = self.get_starter_lock_path()
                if starter_lock is None:
                    if not starter_lock_path.exists():
                        raise RuntimeError(
                            self._format_startup_failure(
                                prefix=(
                                    "ChunkHound daemon startup ended before it "
                                    "became reachable"
                                )
                            )
                        )
                else:
                    holder_pid = starter_lock.get("pid")
                    if isinstance(holder_pid, int) and not pid_alive(holder_pid):
                        starter_lock_path.unlink(missing_ok=True)
                        raise RuntimeError(
                            self._format_startup_failure(
                                prefix=(
                                    "ChunkHound daemon startup ended before it "
                                    "became reachable"
                                )
                            )
                        )
                await asyncio.sleep(_STARTUP_POLL_INTERVAL)

            raise RuntimeError(
                self._format_startup_failure(
                    prefix=(
                        "ChunkHound daemon did not become reachable within "
                        f"{_STARTUP_TIMEOUT}s (address: {initial_address})"
                    )
                )
            )
