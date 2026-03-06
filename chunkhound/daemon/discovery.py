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
# User-scoped registry of running daemons, keyed by canonical project root hash
_REGISTRY_DIR_REL = ".chunkhound/daemon-registry"
# Socket directory (Linux/macOS)
_SOCKET_DIR = "/tmp"
# Startup polling interval and timeout
_STARTUP_POLL_INTERVAL = 0.1
_STARTUP_TIMEOUT = 30.0


def _canonical_project_dir(project_dir: Path) -> Path:
    """Return the canonical project root used for daemon identity."""
    return project_dir.resolve()


def _normalized_project_dir(project_dir: Path) -> Path:
    """Return the comparison-safe project root for overlap checks."""
    canonical = _canonical_project_dir(project_dir)
    if sys.platform == "win32":
        return Path(os.path.normcase(str(canonical)))
    return canonical


def _project_dir_identity(project_dir: Path) -> str:
    """Return the stable string identity used for hashing and comparisons."""
    return str(_normalized_project_dir(project_dir))


def _roots_overlap(root_a: Path, root_b: Path) -> bool:
    """Return True if two canonical roots are identical or nested."""
    left = _normalized_project_dir(root_a)
    right = _normalized_project_dir(root_b)
    if left == right:
        return True
    try:
        right.relative_to(left)
        return True
    except ValueError:
        pass
    try:
        left.relative_to(right)
        return True
    except ValueError:
        return False


def _write_json_atomically(
    path: Path,
    data: dict[str, Any],
    *,
    private: bool = False,
) -> None:
    """Write JSON to *path* atomically using a sibling temp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f)
    tmp_path.replace(path)
    if private and sys.platform != "win32":
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass


class DaemonDiscovery:
    """Locate or start the daemon for a given project directory."""

    def __init__(self, project_dir: Path) -> None:
        self._project_dir = _canonical_project_dir(project_dir)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def get_lock_path(self) -> Path:
        """Return the absolute path of the lock file."""
        return self._project_dir / _LOCK_FILE_REL

    def get_starter_lock_path(self) -> Path:
        """Return the absolute path of the starter lock file."""
        return self._project_dir / _STARTER_LOCK_REL

    def get_registry_dir(self) -> Path:
        """Return the user-scoped daemon registry directory."""
        return Path.home() / _REGISTRY_DIR_REL

    def get_registry_entry_path(self) -> Path:
        """Return the registry entry path for this canonical project root."""
        digest = hashlib.sha256(
            _project_dir_identity(self._project_dir).encode()
        ).hexdigest()[:16]
        return self.get_registry_dir() / f"{digest}.json"

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
            ``auth_token``, ``project_dir``, or ``None`` if the file does
            not exist or is corrupt.
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
        data = {
            "pid": pid,
            "socket_path": socket_path,
            "started_at": time.time(),
            "project_dir": str(self._project_dir),
            "auth_token": (
                auth_token if auth_token is not None else secrets.token_hex(32)
            ),
        }
        _write_json_atomically(lock_path, data, private=True)

    def remove_lock(self) -> None:
        """Remove the lock file, ignoring errors if it does not exist."""
        try:
            self.get_lock_path().unlink()
        except FileNotFoundError:
            pass

    def write_registry_entry(self, pid: int, socket_path: str) -> None:
        """Publish this daemon in the user-scoped registry."""
        data = {
            "project_dir": str(self._project_dir),
            "pid": pid,
            "socket_path": socket_path,
            "lock_path": str(self.get_lock_path()),
            "started_at": time.time(),
        }
        _write_json_atomically(self.get_registry_entry_path(), data)

    def remove_registry_entry(self) -> None:
        """Remove this daemon's registry entry if present."""
        try:
            self.get_registry_entry_path().unlink()
        except FileNotFoundError:
            pass

    def _remove_registry_entry_file(self, entry_path: Path) -> None:
        """Best-effort removal for a stale registry entry."""
        try:
            entry_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass

    def _read_registry_entry(self, entry_path: Path) -> dict[str, Any] | None:
        """Read a registry entry, deleting malformed files opportunistically."""
        try:
            with open(entry_path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
        self._remove_registry_entry_file(entry_path)
        return None

    def _validated_registry_entry(self, entry_path: Path) -> dict[str, Any] | None:
        """Return authoritative live-daemon metadata for a registry entry."""
        entry = self._read_registry_entry(entry_path)
        if entry is None:
            return None

        project_dir_raw = entry.get("project_dir")
        pid = entry.get("pid")
        lock_path_raw = entry.get("lock_path")
        if not isinstance(project_dir_raw, str) or not isinstance(pid, int):
            self._remove_registry_entry_file(entry_path)
            return None

        root = _canonical_project_dir(Path(project_dir_raw))
        expected_lock_path = root / _LOCK_FILE_REL
        if (
            not isinstance(lock_path_raw, str)
            or Path(lock_path_raw) != expected_lock_path
        ):
            self._remove_registry_entry_file(entry_path)
            return None

        if not pid_alive(pid):
            self._remove_registry_entry_file(entry_path)
            return None

        other_discovery = DaemonDiscovery(root)
        lock = other_discovery.read_lock()
        if lock is None:
            self._remove_registry_entry_file(entry_path)
            return None

        lock_pid = lock.get("pid")
        if not isinstance(lock_pid, int) or lock_pid != pid:
            self._remove_registry_entry_file(entry_path)
            return None

        lock_project_dir = lock.get("project_dir")
        if isinstance(lock_project_dir, str):
            if _canonical_project_dir(Path(lock_project_dir)) != root:
                self._remove_registry_entry_file(entry_path)
                return None

        return {
            "project_dir": str(root),
            "pid": pid,
            "socket_path": str(lock.get("socket_path", entry.get("socket_path", ""))),
            "lock_path": str(expected_lock_path),
            "started_at": float(lock.get("started_at", entry.get("started_at", 0.0))),
        }

    def find_conflicting_daemon(self) -> dict[str, Any] | None:
        """Return overlapping live-daemon metadata for a different root, if any."""
        registry_dir = self.get_registry_dir()
        if not registry_dir.exists():
            return None

        for entry_path in registry_dir.glob("*.json"):
            entry = self._validated_registry_entry(entry_path)
            if entry is None:
                continue
            other_root = _canonical_project_dir(Path(str(entry["project_dir"])))
            if _normalized_project_dir(other_root) == _normalized_project_dir(
                self._project_dir
            ):
                continue
            if _roots_overlap(self._project_dir, other_root):
                return entry
        return None

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
            sys.executable,
            "-m",
            "chunkhound.api.cli.main",
            "_daemon",
            "--project-dir",
            str(self._project_dir),
            "--socket-path",
            socket_path,
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

        conflict = self.find_conflicting_daemon()
        if conflict is not None:
            conflict_root = str(conflict["project_dir"])
            conflict_pid = conflict.get("pid")
            pid_suffix = (
                f" (pid {conflict_pid})" if isinstance(conflict_pid, int) else ""
            )
            raise RuntimeError(
                f"Cannot start ChunkHound daemon for '{self._project_dir}' because "
                f"a daemon is already running for overlapping root '{conflict_root}'"
                f"{pid_suffix}. Overlapping daemon roots are not supported."
            )

        # No live daemon — race to become the sole daemon starter.
        if self._acquire_starter_lock():
            try:
                self._start_daemon_subprocess(args)
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
