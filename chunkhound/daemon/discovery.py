"""Daemon discovery via lock file and IPC transport.

Handles:
- Runtime-scoped lock file creation/reading for a canonical project root
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
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .process import pid_alive

# Runtime-scoped lock files keyed by canonical project root hash
_LOCKS_DIR_NAME = "daemon-locks"
# Runtime-scoped starter locks — prevent duplicate spawns within one environment
_STARTER_LOCKS_DIR_NAME = "daemon-starter-locks"
# User-scoped runtime directory override (used by tests and debugging)
_RUNTIME_DIR_ENV = "CHUNKHOUND_DAEMON_RUNTIME_DIR"
# User-scoped registry of running daemons, keyed by canonical project root hash
_REGISTRY_DIR_NAME = "daemon-registry"
# User-scoped startup lock — serializes overlap checks across project roots
_GLOBAL_STARTUP_LOCK_NAME = "daemon.global.startup.lock"
# Socket directory (Linux/macOS)
_SOCKET_DIR = "/tmp"
# Startup polling interval and timeout
_STARTUP_POLL_INTERVAL = 0.1
_STARTUP_TIMEOUT = 30.0
_WINDOWS_REPLACE_RETRIES = 20
_WINDOWS_REPLACE_RETRY_DELAY = 0.01


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


def _project_dir_hash(project_dir: Path, *, length: int) -> str:
    """Return a stable hash of the canonical project identity."""
    return hashlib.sha256(_project_dir_identity(project_dir).encode()).hexdigest()[
        :length
    ]


def _runtime_owner_tag() -> str:
    """Return a filesystem-safe tag for user-scoped runtime state."""
    if hasattr(os, "getuid"):
        return str(os.getuid())

    username = os.environ.get("USERNAME") or os.environ.get("USER") or "user"
    safe = "".join(c if c.isalnum() or c in "-._" else "-" for c in username)
    return safe or "user"


def _default_runtime_dir() -> Path:
    """Return the user-scoped runtime directory for daemon metadata."""
    override = os.environ.get(_RUNTIME_DIR_ENV)
    if override:
        return Path(override).expanduser()

    suffix = f"chunkhound-{_runtime_owner_tag()}"
    if sys.platform != "win32":
        runtime_root = os.environ.get("XDG_RUNTIME_DIR")
        if runtime_root:
            return Path(runtime_root) / suffix

    return Path(tempfile.gettempdir()) / suffix


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
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        for attempt in range(_WINDOWS_REPLACE_RETRIES):
            try:
                tmp_path.replace(path)
                break
            except PermissionError:
                if sys.platform != "win32" or attempt >= _WINDOWS_REPLACE_RETRIES - 1:
                    raise
                time.sleep(_WINDOWS_REPLACE_RETRY_DELAY)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass
        raise
    if private and sys.platform != "win32":
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass


@dataclass(slots=True)
class DaemonStartupHandle:
    """Process handle and diagnostics surface for a daemon startup attempt."""

    process: subprocess.Popen[Any]
    log_path: Path


class DaemonDiscovery:
    """Locate or start the daemon for a given project directory."""

    def __init__(self, project_dir: Path) -> None:
        self._project_dir = _canonical_project_dir(project_dir)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def get_lock_path(self) -> Path:
        """Return the absolute path of the lock file."""
        return self.get_runtime_dir() / _LOCKS_DIR_NAME / (
            f"{_project_dir_hash(self._project_dir, length=16)}.json"
        )

    def get_starter_lock_path(self) -> Path:
        """Return the absolute path of the starter lock file."""
        return self.get_runtime_dir() / _STARTER_LOCKS_DIR_NAME / (
            f"{_project_dir_hash(self._project_dir, length=16)}.json"
        )

    def get_runtime_dir(self) -> Path:
        """Return the user-scoped runtime directory for daemon metadata."""
        return _default_runtime_dir()

    def get_registry_dir(self) -> Path:
        """Return the user-scoped daemon registry directory."""
        return self.get_runtime_dir() / _REGISTRY_DIR_NAME

    def get_global_startup_lock_path(self) -> Path:
        """Return the user-scoped global startup lock path."""
        return self.get_runtime_dir() / _GLOBAL_STARTUP_LOCK_NAME

    def get_registry_entry_path(self) -> Path:
        """Return the registry entry path for this canonical project root."""
        return self.get_registry_dir() / (
            f"{_project_dir_hash(self._project_dir, length=16)}.json"
        )

    def get_ipc_address(self) -> str:
        """Derive a unique IPC address from the project directory.

        Linux/macOS: Unix socket path in /tmp.
        Windows: TCP loopback address with port 0 (OS-assigned at bind time).
        """
        digest = _project_dir_hash(self._project_dir, length=8)
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
            ``auth_token``, ``project_dir``, or ``None`` if the file does
            not exist or is corrupt.
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

    def _overlap_error(self, conflict: dict[str, Any]) -> RuntimeError:
        """Build the overlap error raised when a conflicting daemon is live."""
        conflict_root = str(conflict["project_dir"])
        conflict_pid = conflict.get("pid")
        pid_suffix = f" (pid {conflict_pid})" if isinstance(conflict_pid, int) else ""
        return RuntimeError(
            f"Cannot start ChunkHound daemon for '{self._project_dir}' because "
            f"a daemon is already running for overlapping root '{conflict_root}'"
            f"{pid_suffix}. Overlapping daemon roots are not supported."
        )

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
        expected_lock_path = DaemonDiscovery(root).get_lock_path()
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
    # Startup locks (prevent duplicate-daemon races across and within roots)
    # ------------------------------------------------------------------

    def _acquire_pid_lock(self, lock_path: Path) -> bool:
        """Try to atomically acquire a PID lock."""
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        for attempt in range(2):
            try:
                with open(lock_path, "x") as f:
                    json.dump({"pid": os.getpid()}, f)
                return True
            except FileExistsError:
                try:
                    with open(lock_path) as f:
                        holder = json.load(f)
                    holder_pid = holder.get("pid", 0)
                    if isinstance(holder_pid, int) and pid_alive(holder_pid):
                        return False
                    lock_path.unlink(missing_ok=True)
                except (FileNotFoundError, json.JSONDecodeError, OSError):
                    if attempt == 0:
                        continue
                    return False

        return False

    def _release_pid_lock(self, lock_path: Path) -> None:
        """Release a PID lock if this process owns it."""
        try:
            with open(lock_path) as f:
                holder = json.load(f)
            if holder.get("pid") == os.getpid():
                lock_path.unlink(missing_ok=True)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass

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
        return self._acquire_pid_lock(self.get_starter_lock_path())

    def _release_starter_lock(self) -> None:
        """Release the starter lock if this process holds it."""
        self._release_pid_lock(self.get_starter_lock_path())

    def _acquire_global_startup_lock(self) -> bool:
        """Try to acquire the user-scoped global startup lock."""
        return self._acquire_pid_lock(self.get_global_startup_lock_path())

    def _release_global_startup_lock(self) -> None:
        """Release the user-scoped global startup lock."""
        self._release_pid_lock(self.get_global_startup_lock_path())

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

    def _remove_stale_lock_artifacts(self, initial_address: str) -> None:
        """Remove stale daemon metadata before attempting a fresh start."""
        self.remove_lock()
        if sys.platform != "win32" and not initial_address.startswith("tcp:"):
            try:
                os.unlink(initial_address)
            except FileNotFoundError:
                pass

    async def _reuse_live_daemon(
        self,
        initial_address: str,
        timeout: float,
    ) -> str | None:
        """Return a live daemon address from the lock, or clean up stale state."""
        lock = self.read_lock()
        if lock is None:
            return None

        pid = lock.get("pid")
        actual_address = str(lock.get("socket_path", initial_address))
        if isinstance(pid, int) and pid_alive(pid):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if await self._socket_connectable(actual_address):
                    return actual_address
                if not pid_alive(pid):
                    raise RuntimeError(
                        self._format_startup_failure(
                            prefix=(
                                "ChunkHound daemon exited before it became reachable "
                                f"(pid={pid}, address: {actual_address})"
                            )
                        )
                    )
                remaining = deadline - time.monotonic()
                await asyncio.sleep(min(_STARTUP_POLL_INTERVAL, max(remaining, 0.0)))
            raise RuntimeError(
                self._format_startup_failure(
                    prefix=(
                        f"ChunkHound daemon (pid={pid}) did not become reachable "
                        f"within {timeout}s (address: {actual_address})"
                    )
                )
            )

        self._remove_stale_lock_artifacts(initial_address)
        return None

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

        If a daemon is already starting (lock file with live PID exists but
        not yet connectable), waits for it rather than starting a second
        daemon.  This prevents duplicate-daemon races when two proxies start
        simultaneously.
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
        existing_address = await self._reuse_live_daemon(
            initial_address,
            _STARTUP_TIMEOUT,
        )
        if existing_address is not None:
            return existing_address

        deadline = time.monotonic() + _STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if not self._acquire_global_startup_lock():
                conflict = self.find_conflicting_daemon()
                if conflict is not None:
                    raise self._overlap_error(conflict)

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break

                existing_address = await self._reuse_live_daemon(
                    initial_address,
                    remaining,
                )
                if existing_address is not None:
                    return existing_address

                await asyncio.sleep(min(_STARTUP_POLL_INTERVAL, remaining))
                continue

            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break

                existing_address = await self._reuse_live_daemon(
                    initial_address,
                    remaining,
                )
                if existing_address is not None:
                    return existing_address

                conflict = self.find_conflicting_daemon()
                if conflict is not None:
                    raise self._overlap_error(conflict)

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break

                # No live daemon — race to become the sole daemon starter.
                # The global startup lock is acquired before the per-root
                # starter lock and released after it, so no live process can
                # hold the starter lock while we already hold the global one.
                if not self._acquire_starter_lock():
                    raise AssertionError(
                        "starter lock unavailable while global startup lock is held"
                    )

                try:
                    startup = self._start_daemon_subprocess(args)
                    poll_deadline = time.monotonic() + remaining
                    while time.monotonic() < poll_deadline:
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
                            actual_address = str(
                                lock.get("socket_path", initial_address)
                            )
                            if await self._socket_connectable(actual_address):
                                return actual_address
                        sleep_for = poll_deadline - time.monotonic()
                        await asyncio.sleep(
                            min(_STARTUP_POLL_INTERVAL, max(sleep_for, 0.0))
                        )

                    raise RuntimeError(
                        self._format_startup_failure(
                            prefix=(
                                f"ChunkHound daemon did not start within "
                                f"{_STARTUP_TIMEOUT}s (address: {initial_address})"
                            ),
                            log_path=startup.log_path,
                        )
                    )
                finally:
                    self._release_starter_lock()
            finally:
                self._release_global_startup_lock()

        raise RuntimeError(
            self._format_startup_failure(
                prefix=(
                    f"ChunkHound daemon did not become reachable within "
                    f"{_STARTUP_TIMEOUT}s (address: {initial_address})"
                )
            )
        )
