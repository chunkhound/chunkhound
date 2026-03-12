from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from chunkhound.daemon.process import pid_alive
from chunkhound.watchman.sidecar import (
    PrivateWatchmanSidecar,
    WatchmanSidecarMetadata,
    WatchmanSidecarPaths,
)
from chunkhound.watchman_runtime.loader import materialize_watchman_binary


def _host_sidecar_command(
    *,
    binary_path: Path,
    socket_path: Path,
    statefile_path: Path,
    logfile_path: Path,
) -> list[str]:
    command = [
        str(binary_path),
        "--foreground",
        "--sockname",
        str(socket_path),
        "--statefile",
        str(statefile_path),
        "--logfile",
        str(logfile_path),
        "--no-save-state",
    ]
    if os.name == "nt":
        return ["cmd.exe", "/c", *command]
    return command


def _prepend_poisoned_python_shims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shims_dir = tmp_path / "poisoned-python"
    shims_dir.mkdir()
    if os.name == "nt":
        for name in ("python.cmd", "python3.cmd"):
            (shims_dir / name).write_text(
                "@echo off\r\nexit /b 97\r\n", encoding="utf-8"
            )
    else:
        for name in ("python", "python3"):
            shim_path = shims_dir / name
            shim_path.write_text("#!/bin/sh\nexit 97\n", encoding="utf-8")
            shim_path.chmod(0o755)
    current_path = os.environ.get("PATH", "")
    monkeypatch.setenv(
        "PATH",
        str(shims_dir)
        if not current_path
        else f"{shims_dir}{os.pathsep}{current_path}",
    )


def _wait_for_sidecar_files(
    *,
    process: subprocess.Popen,
    socket_path: Path,
    statefile_path: Path,
    logfile_path: Path,
) -> None:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if socket_path.exists() and statefile_path.exists() and logfile_path.exists():
            return
        if process.poll() is not None:
            raise AssertionError(
                "packaged Watchman runtime exited before it created sidecar files "
                f"(rc={process.returncode})"
            )
        time.sleep(0.05)
    raise AssertionError("timed out waiting for packaged Watchman sidecar files")


def _stop_process(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2.0)


@pytest.mark.asyncio
async def test_private_watchman_sidecar_start_writes_metadata_and_artifacts(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sidecar = PrivateWatchmanSidecar(repo_root)

    metadata = await sidecar.start()

    assert metadata.pid > 0
    assert metadata.process_start_time_epoch is not None
    assert sidecar.paths.metadata_path.is_file()
    assert sidecar.paths.socket_path.exists()
    assert sidecar.paths.statefile_path.exists()
    assert sidecar.paths.logfile_path.exists()

    health = sidecar.get_health()
    assert health["watchman_pid"] == metadata.pid
    assert (
        health["watchman_process_start_time_epoch"] == metadata.process_start_time_epoch
    )
    assert health["watchman_alive"] is True

    await sidecar.stop()


@pytest.mark.asyncio
async def test_private_watchman_sidecar_start_ignores_poisoned_python_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _prepend_poisoned_python_shims(tmp_path, monkeypatch)
    sidecar = PrivateWatchmanSidecar(repo_root)

    metadata = await sidecar.start()

    assert metadata.pid > 0
    assert Path(metadata.binary_path).is_file()
    assert sidecar.paths.socket_path.exists()

    await sidecar.stop()


@pytest.mark.asyncio
async def test_private_watchman_sidecar_stop_cleans_state_and_keeps_runtime_cache(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sidecar = PrivateWatchmanSidecar(repo_root)

    metadata = await sidecar.start()
    binary_path = Path(metadata.binary_path)

    await sidecar.stop()

    assert not sidecar.paths.metadata_path.exists()
    assert not sidecar.paths.socket_path.exists()
    assert not sidecar.paths.statefile_path.exists()
    assert sidecar.paths.logfile_path.exists()
    assert binary_path.is_file()


@pytest.mark.asyncio
async def test_private_watchman_sidecar_replaces_stale_dead_metadata(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sidecar = PrivateWatchmanSidecar(repo_root)

    dead_process = subprocess.Popen([sys.executable, "-c", "pass"])
    dead_process.wait(timeout=5.0)

    sidecar.paths.root.mkdir(parents=True, exist_ok=True)
    sidecar.paths.socket_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar.paths.socket_path.write_text("stale socket\n", encoding="utf-8")
    sidecar.paths.statefile_path.write_text("stale state\n", encoding="utf-8")
    sidecar.paths.logfile_path.write_text("stale log\n", encoding="utf-8")
    sidecar._write_metadata(
        WatchmanSidecarMetadata(
            pid=dead_process.pid,
            started_at="2026-03-10T00:00:00+00:00",
            process_start_time_epoch=1.0,
            runtime_version="stale-runtime",
            socket_path=str(sidecar.paths.socket_path),
            statefile_path=str(sidecar.paths.statefile_path),
            logfile_path=str(sidecar.paths.logfile_path),
            binary_path="stale-binary",
        )
    )

    metadata = await sidecar.start()

    assert metadata.pid != dead_process.pid
    assert pid_alive(metadata.pid)

    await sidecar.stop()


@pytest.mark.asyncio
async def test_private_watchman_sidecar_replaces_owned_stale_live_process(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sidecar = PrivateWatchmanSidecar(repo_root)

    binary_path = materialize_watchman_binary(
        destination_root=sidecar.paths.runtime_root
    )
    stale_process = subprocess.Popen(
        _host_sidecar_command(
            binary_path=binary_path,
            socket_path=sidecar.paths.socket_path,
            statefile_path=sidecar.paths.statefile_path,
            logfile_path=sidecar.paths.logfile_path,
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=sidecar.paths.project_root,
    )
    _wait_for_sidecar_files(
        process=stale_process,
        socket_path=sidecar.paths.socket_path,
        statefile_path=sidecar.paths.statefile_path,
        logfile_path=sidecar.paths.logfile_path,
    )
    stale_start_time = psutil.Process(stale_process.pid).create_time()
    sidecar._write_metadata(
        WatchmanSidecarMetadata(
            pid=stale_process.pid,
            started_at="2026-03-10T00:00:00+00:00",
            process_start_time_epoch=stale_start_time,
            runtime_version="watchman-placeholder-2026-03",
            socket_path=str(sidecar.paths.socket_path),
            statefile_path=str(sidecar.paths.statefile_path),
            logfile_path=str(sidecar.paths.logfile_path),
            binary_path=str(binary_path),
        )
    )

    metadata = await sidecar.start()

    assert metadata.pid != stale_process.pid
    assert not pid_alive(stale_process.pid)
    assert pid_alive(metadata.pid)

    await sidecar.stop()


@pytest.mark.asyncio
async def test_private_watchman_sidecar_refuses_live_process_with_legacy_metadata(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sidecar = PrivateWatchmanSidecar(repo_root)

    binary_path = materialize_watchman_binary(
        destination_root=sidecar.paths.runtime_root
    )
    live_process = subprocess.Popen(
        _host_sidecar_command(
            binary_path=binary_path,
            socket_path=sidecar.paths.socket_path,
            statefile_path=sidecar.paths.statefile_path,
            logfile_path=sidecar.paths.logfile_path,
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=sidecar.paths.project_root,
    )
    try:
        _wait_for_sidecar_files(
            process=live_process,
            socket_path=sidecar.paths.socket_path,
            statefile_path=sidecar.paths.statefile_path,
            logfile_path=sidecar.paths.logfile_path,
        )
        sidecar.paths.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar.paths.metadata_path.write_text(
            json.dumps(
                {
                    "pid": live_process.pid,
                    "started_at": "2026-03-10T00:00:00+00:00",
                    "runtime_version": "watchman-placeholder-2026-03",
                    "socket_path": str(sidecar.paths.socket_path),
                    "statefile_path": str(sidecar.paths.statefile_path),
                    "logfile_path": str(sidecar.paths.logfile_path),
                    "binary_path": str(binary_path),
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(
            RuntimeError, match="does not record process_start_time_epoch"
        ):
            await sidecar.start()

        assert pid_alive(live_process.pid)
    finally:
        if live_process.poll() is None:
            _stop_process(live_process)


@pytest.mark.asyncio
async def test_private_watchman_sidecar_refuses_shutdown_for_mismatched_live_process(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sidecar = PrivateWatchmanSidecar(repo_root)

    binary_path = materialize_watchman_binary(
        destination_root=sidecar.paths.runtime_root
    )
    live_process = subprocess.Popen(
        _host_sidecar_command(
            binary_path=binary_path,
            socket_path=sidecar.paths.socket_path,
            statefile_path=sidecar.paths.statefile_path,
            logfile_path=sidecar.paths.logfile_path,
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=sidecar.paths.project_root,
    )
    try:
        _wait_for_sidecar_files(
            process=live_process,
            socket_path=sidecar.paths.socket_path,
            statefile_path=sidecar.paths.statefile_path,
            logfile_path=sidecar.paths.logfile_path,
        )
        sidecar._write_metadata(
            WatchmanSidecarMetadata(
                pid=live_process.pid,
                started_at="2026-03-10T00:00:00+00:00",
                process_start_time_epoch=1.0,
                runtime_version="watchman-placeholder-2026-03",
                socket_path=str(sidecar.paths.socket_path),
                statefile_path=str(sidecar.paths.statefile_path),
                logfile_path=str(sidecar.paths.logfile_path),
                binary_path=str(binary_path),
            )
        )

        with pytest.raises(RuntimeError, match="start time does not match"):
            await sidecar.stop()

        assert pid_alive(live_process.pid)
    finally:
        if live_process.poll() is None:
            _stop_process(live_process)
        sidecar._remove_owned_artifacts(remove_log=True)


@pytest.mark.asyncio
async def test_private_watchman_sidecar_start_failure_leaves_no_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sidecar = PrivateWatchmanSidecar(repo_root)
    monkeypatch.setenv("CHUNKHOUND_FAKE_WATCHMAN_FAIL_BEFORE_READY", "1")

    with pytest.raises(RuntimeError, match="exited before it became ready"):
        await sidecar.start()

    assert not sidecar.paths.metadata_path.exists()
    assert not sidecar.paths.socket_path.exists()
    assert not sidecar.paths.statefile_path.exists()
    assert sidecar.paths.logfile_path.exists()


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.name == "nt", reason="unix socket path validation does not apply on Windows"
)
async def test_private_watchman_sidecar_uses_short_socket_fallback_for_overlong_path(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    project_socket_path = repo_root.resolve() / ".chunkhound" / "watchman" / "sock"
    monkeypatch.setattr(
        PrivateWatchmanSidecar,
        "_UNIX_SOCKET_PATH_MAX_BYTES",
        len(os.fsencode(str(project_socket_path))),
        raising=False,
    )

    expected_socket_path = WatchmanSidecarPaths._resolve_socket_path(
        project_root=repo_root.resolve(),
        project_socket_path=project_socket_path,
    )
    sidecar = PrivateWatchmanSidecar(repo_root)

    metadata = await sidecar.start()

    assert sidecar.paths.using_socket_fallback is True
    assert sidecar.paths.project_socket_path == project_socket_path
    assert sidecar.paths.socket_path == expected_socket_path
    assert metadata.socket_path == str(expected_socket_path)
    assert sidecar.paths.metadata_path.exists()
    assert sidecar.paths.socket_path.exists()
    assert not sidecar.paths.project_socket_path.exists()

    await sidecar.stop()

    assert not sidecar.paths.metadata_path.exists()
    assert not sidecar.paths.socket_path.exists()
    assert not sidecar.paths.project_socket_path.exists()
