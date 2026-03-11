from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

from chunkhound.daemon.process import pid_alive
from chunkhound.watchman.session import build_watchman_base_command
from chunkhound.watchman_runtime.loader import (
    PackagedWatchmanRuntime,
    materialize_watchman_binary,
    resolve_packaged_watchman_runtime,
)


def _watchman_socket_path_limit_bytes() -> int:
    return PrivateWatchmanSidecar._UNIX_SOCKET_PATH_MAX_BYTES


@dataclass(frozen=True)
class WatchmanSidecarPaths:
    """ChunkHound-owned paths for the private Watchman sidecar."""

    project_root: Path
    root: Path
    runtime_root: Path
    project_socket_path: Path
    socket_path: Path
    statefile_path: Path
    logfile_path: Path
    metadata_path: Path

    @classmethod
    def for_target_dir(cls, target_dir: Path) -> WatchmanSidecarPaths:
        project_root = target_dir.expanduser().resolve()
        root = project_root / ".chunkhound" / "watchman"
        project_socket_path = root / "sock"
        return cls(
            project_root=project_root,
            root=root,
            runtime_root=root / "runtime",
            project_socket_path=project_socket_path,
            socket_path=cls._resolve_socket_path(
                project_root=project_root,
                project_socket_path=project_socket_path,
            ),
            statefile_path=root / "state",
            logfile_path=root / "watchman.log",
            metadata_path=root / "metadata.json",
        )

    @property
    def using_socket_fallback(self) -> bool:
        return self.socket_path != self.project_socket_path

    def managed_socket_paths(self) -> tuple[Path, ...]:
        if self.socket_path == self.project_socket_path:
            return (self.socket_path,)
        return (self.socket_path, self.project_socket_path)

    @staticmethod
    def _resolve_socket_path(
        *, project_root: Path, project_socket_path: Path
    ) -> Path:
        if os.name == "nt":
            return project_socket_path

        limit = _watchman_socket_path_limit_bytes()
        if len(os.fsencode(str(project_socket_path))) < limit:
            return project_socket_path

        digest = hashlib.sha256(str(project_root).encode("utf-8")).hexdigest()[:16]
        fallback_socket_path = (
            Path(tempfile.gettempdir()) / "chunkhound-watchman" / digest / "sock"
        )
        if len(os.fsencode(str(fallback_socket_path))) >= limit:
            raise RuntimeError(
                "Watchman private socket path is too long for this platform even "
                "after deterministic fallback: "
                f"{fallback_socket_path}"
            )
        return fallback_socket_path


@dataclass(frozen=True)
class WatchmanSidecarMetadata:
    """ChunkHound-owned metadata for a private Watchman sidecar."""

    pid: int
    started_at: str
    process_start_time_epoch: float | None
    runtime_version: str
    socket_path: str
    statefile_path: str
    logfile_path: str
    binary_path: str

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> WatchmanSidecarMetadata:
        pid = payload.get("pid")
        started_at = payload.get("started_at")
        process_start_time_epoch_payload = payload.get("process_start_time_epoch")
        runtime_version = payload.get("runtime_version")
        socket_path = payload.get("socket_path")
        statefile_path = payload.get("statefile_path")
        logfile_path = payload.get("logfile_path")
        binary_path = payload.get("binary_path")

        if not isinstance(pid, int) or pid <= 0:
            raise ValueError("metadata pid must be a positive integer")
        if not isinstance(started_at, str) or not started_at.strip():
            raise ValueError("metadata field 'started_at' must be a non-empty string")

        process_start_time_epoch: float | None
        if process_start_time_epoch_payload is None:
            process_start_time_epoch = None
        elif isinstance(process_start_time_epoch_payload, (int, float)):
            process_start_time_epoch = float(process_start_time_epoch_payload)
            if process_start_time_epoch <= 0:
                raise ValueError(
                    "metadata field 'process_start_time_epoch' must be positive"
                )
        else:
            raise ValueError(
                "metadata field 'process_start_time_epoch' must be numeric when present"
            )

        values = {
            "runtime_version": runtime_version,
            "socket_path": socket_path,
            "statefile_path": statefile_path,
            "logfile_path": logfile_path,
            "binary_path": binary_path,
        }
        for key, value in values.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"metadata field {key!r} must be a non-empty string"
                )

        return cls(
            pid=pid,
            started_at=started_at,
            process_start_time_epoch=process_start_time_epoch,
            runtime_version=runtime_version,
            socket_path=socket_path,
            statefile_path=statefile_path,
            logfile_path=logfile_path,
            binary_path=binary_path,
        )


def _iso_from_epoch(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(epoch_seconds, timezone.utc).replace(
        microsecond=0
    ).isoformat()


def _terminate_process_tree_sync(pid: int, timeout: float) -> None:
    """Terminate a process tree, escalating to kill if needed."""

    if pid <= 0:
        return
    if pid == os.getpid():
        raise RuntimeError(
            "Refusing to terminate the current ChunkHound process "
            "as Watchman stale state"
        )

    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    try:
        processes = root.children(recursive=True)
    except psutil.NoSuchProcess:
        return
    except psutil.AccessDenied:
        processes = []
    processes.append(root)

    for process in processes:
        try:
            process.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    _, alive = psutil.wait_procs(processes, timeout=timeout)

    for process in alive:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    _, stubborn = psutil.wait_procs(alive, timeout=max(1.0, timeout / 2.0))
    if stubborn:
        stubborn_pids = ", ".join(str(process.pid) for process in stubborn)
        raise RuntimeError(
            "Watchman sidecar did not exit after terminate/kill escalation: "
            f"{stubborn_pids}"
        )


class PrivateWatchmanSidecar:
    """Manage a ChunkHound-owned private Watchman process."""

    _READY_TIMEOUT_SECONDS = 5.0
    _PROCESS_EXIT_TIMEOUT_SECONDS = 5.0
    _READY_POLL_INTERVAL_SECONDS = 0.05
    _PROCESS_START_TIME_EPSILON_SECONDS = 0.1
    _UNIX_SOCKET_PATH_MAX_BYTES = 104 if sys.platform == "darwin" else 108

    def __init__(
        self, target_dir: Path, debug_sink: Callable[[str], None] | None = None
    ) -> None:
        self.paths = WatchmanSidecarPaths.for_target_dir(target_dir)
        self._debug_sink = debug_sink
        self._process: subprocess.Popen | None = None
        self._process_start_time_epoch: float | None = None
        self._metadata: WatchmanSidecarMetadata | None = None
        self._runtime: PackagedWatchmanRuntime | None = None

    def _debug(self, message: str) -> None:
        try:
            if self._debug_sink is not None:
                self._debug_sink(f"watchman: {message}")
        except Exception:
            pass

    def read_metadata(self) -> WatchmanSidecarMetadata | None:
        if not self.paths.metadata_path.is_file():
            return None
        try:
            payload = json.loads(self.paths.metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self._debug(f"ignoring unreadable sidecar metadata: {error}")
            return None
        if not isinstance(payload, dict):
            self._debug("ignoring non-object sidecar metadata payload")
            return None
        try:
            return WatchmanSidecarMetadata.from_payload(payload)
        except ValueError as error:
            self._debug(f"ignoring malformed sidecar metadata: {error}")
            return None

    def get_health(self) -> dict[str, Any]:
        metadata = self._metadata or self.read_metadata()
        if metadata is None and self._process is not None:
            pid: int | None = self._process.pid
        else:
            pid = metadata.pid if metadata is not None else None

        running = False
        if self._process is not None:
            running = self._process.poll() is None
        elif pid is not None:
            running = pid_alive(pid)

        runtime_version = None
        binary_path = None
        started_at = None
        process_start_time_epoch = self._process_start_time_epoch
        if metadata is not None:
            runtime_version = metadata.runtime_version
            binary_path = metadata.binary_path
            started_at = metadata.started_at
            process_start_time_epoch = metadata.process_start_time_epoch
        elif self._runtime is not None:
            runtime_version = self._runtime.runtime_version
            if process_start_time_epoch is not None:
                started_at = _iso_from_epoch(process_start_time_epoch)

        return {
            "watchman_pid": pid,
            "watchman_started_at": started_at,
            "watchman_process_start_time_epoch": process_start_time_epoch,
            "watchman_runtime_version": runtime_version,
            "watchman_binary_path": binary_path,
            "watchman_socket_path": str(self.paths.socket_path),
            "watchman_statefile_path": str(self.paths.statefile_path),
            "watchman_logfile_path": str(self.paths.logfile_path),
            "watchman_metadata_path": str(self.paths.metadata_path),
            "watchman_alive": running,
        }

    async def cleanup_stale_state(self) -> str | None:
        self.paths.root.mkdir(parents=True, exist_ok=True)
        metadata = self.read_metadata()

        if metadata is None:
            if any(
                path.exists()
                for path in (
                    *self.paths.managed_socket_paths(),
                    self.paths.statefile_path,
                    self.paths.logfile_path,
                )
            ):
                self._debug("cleaning orphaned Watchman artifacts without metadata")
                self._remove_owned_artifacts(remove_log=True)
                return "orphaned_artifacts"
            return None

        owned_pid = await asyncio.to_thread(
            self._resolve_owned_metadata_pid,
            metadata,
            "startup cleanup",
        )
        if owned_pid is None:
            self._debug(f"removing stale dead Watchman sidecar pid={metadata.pid}")
            self._remove_owned_artifacts(remove_log=True)
            return "removed_stale_sidecar"

        self._debug(f"terminating stale live Watchman sidecar pid={owned_pid}")
        await asyncio.to_thread(
            _terminate_process_tree_sync,
            owned_pid,
            self._PROCESS_EXIT_TIMEOUT_SECONDS,
        )
        self._remove_owned_artifacts(remove_log=True)
        return "replaced_live_sidecar"

    async def start(self) -> WatchmanSidecarMetadata:
        if self._process is not None and self._process.poll() is None:
            await self.stop()

        self.paths.root.mkdir(parents=True, exist_ok=True)
        cleanup_reason = await self.cleanup_stale_state()
        if cleanup_reason is not None:
            self._debug(f"startup cleanup completed: {cleanup_reason}")

        self._runtime = resolve_packaged_watchman_runtime()
        binary_path = materialize_watchman_binary(
            destination_root=self.paths.runtime_root
        )
        self._validate_socket_path()
        if self.paths.using_socket_fallback:
            self._debug(
                "using deterministic short Watchman socket fallback "
                f"{self.paths.socket_path} instead of {self.paths.project_socket_path}"
            )

        command = [
            *build_watchman_base_command(binary_path),
            "--foreground",
            "--sockname",
            str(self.paths.socket_path),
            "--statefile",
            str(self.paths.statefile_path),
            "--logfile",
            str(self.paths.logfile_path),
            "--no-save-state",
        ]

        self._debug(
            "starting private Watchman sidecar with "
            f"binary={binary_path} socket={self.paths.socket_path}"
        )

        log_handle = self.paths.logfile_path.open("ab")
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=log_handle,
                cwd=self.paths.project_root,
            )
        finally:
            log_handle.close()

        try:
            await self._wait_for_ready()
        except BaseException:
            await self.stop()
            raise

        process_start_time_epoch = self._read_process_start_time_epoch(
            self._process.pid
        )
        if process_start_time_epoch is None:
            raise RuntimeError(
                "Watchman sidecar exited before its process identity could be recorded"
            )
        self._process_start_time_epoch = process_start_time_epoch

        metadata = WatchmanSidecarMetadata(
            pid=self._process.pid,
            started_at=_iso_from_epoch(process_start_time_epoch),
            process_start_time_epoch=process_start_time_epoch,
            runtime_version=self._runtime.runtime_version,
            socket_path=str(self.paths.socket_path),
            statefile_path=str(self.paths.statefile_path),
            logfile_path=str(self.paths.logfile_path),
            binary_path=str(binary_path),
        )
        self._write_metadata(metadata)
        self._metadata = metadata
        return metadata

    async def stop(self, *, remove_log: bool = False) -> None:
        metadata = self._metadata or self.read_metadata()
        pid = None

        if self._process is not None and self._process.poll() is None:
            pid = self._process.pid
        elif metadata is not None:
            pid = await asyncio.to_thread(
                self._resolve_owned_metadata_pid,
                metadata,
                "shutdown",
            )

        if pid is not None:
            self._debug(f"stopping private Watchman sidecar pid={pid}")
            await asyncio.to_thread(
                _terminate_process_tree_sync,
                pid,
                self._PROCESS_EXIT_TIMEOUT_SECONDS,
            )

        self._process = None
        self._process_start_time_epoch = None
        self._metadata = None
        self._remove_owned_artifacts(remove_log=remove_log)

    def _validate_socket_path(self) -> None:
        if os.name == "nt":
            return
        encoded_path = os.fsencode(str(self.paths.socket_path))
        if len(encoded_path) >= self._UNIX_SOCKET_PATH_MAX_BYTES:
            raise RuntimeError(
                "Watchman private socket path is too long for this platform: "
                f"{self.paths.socket_path}"
            )

    async def _wait_for_ready(self) -> None:
        deadline = asyncio.get_running_loop().time() + self._READY_TIMEOUT_SECONDS
        while True:
            if self._process is None:
                raise RuntimeError("Watchman sidecar process was not started")

            returncode = self._process.poll()
            if returncode is not None:
                raise RuntimeError(
                    "Watchman sidecar exited before it became ready "
                    f"(exit code {returncode})"
                )

            if (
                self.paths.socket_path.exists()
                and self.paths.statefile_path.exists()
                and self.paths.logfile_path.exists()
            ):
                return

            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError(
                    "Watchman sidecar did not create its private socket before timeout"
                )

            await asyncio.sleep(self._READY_POLL_INTERVAL_SECONDS)

    def _read_process_start_time_epoch(self, pid: int) -> float | None:
        try:
            return psutil.Process(pid).create_time()
        except psutil.NoSuchProcess:
            return None
        except (psutil.AccessDenied, psutil.Error) as error:
            raise RuntimeError(
                f"Unable to inspect live process {pid} for Watchman ownership: {error}"
            ) from error

    def _resolve_owned_metadata_pid(
        self, metadata: WatchmanSidecarMetadata, context: str
    ) -> int | None:
        live_process_start_time = self._read_process_start_time_epoch(metadata.pid)
        if live_process_start_time is None:
            return None

        if metadata.process_start_time_epoch is None:
            raise RuntimeError(
                "Refusing to terminate live process for Watchman "
                f"{context} because metadata does not record process_start_time_epoch"
            )

        if (
            abs(live_process_start_time - metadata.process_start_time_epoch)
            > self._PROCESS_START_TIME_EPSILON_SECONDS
        ):
            raise RuntimeError(
                "Refusing to terminate live process for Watchman "
                f"{context} because metadata start time does not match pid "
                f"{metadata.pid}"
            )

        return metadata.pid

    def _write_metadata(self, metadata: WatchmanSidecarMetadata) -> None:
        self.paths.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.paths.metadata_path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps(asdict(metadata), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temp_path, self.paths.metadata_path)
        if os.name != "nt":
            os.chmod(self.paths.metadata_path, stat.S_IRUSR | stat.S_IWUSR)

    def _remove_owned_artifacts(self, *, remove_log: bool) -> None:
        paths = [
            self.paths.metadata_path,
            self.paths.statefile_path,
            *self.paths.managed_socket_paths(),
        ]
        if remove_log:
            paths.append(self.paths.logfile_path)

        for path in paths:
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            except IsADirectoryError:
                continue


__all__ = [
    "PrivateWatchmanSidecar",
    "WatchmanSidecarMetadata",
    "WatchmanSidecarPaths",
]
