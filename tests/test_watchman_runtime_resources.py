from __future__ import annotations

import os
import stat
import subprocess
import time
from pathlib import Path

import pytest

from chunkhound.watchman_runtime.loader import (
    UnsupportedWatchmanRuntimePlatformError,
    materialize_watchman_binary,
    resolve_packaged_watchman_runtime,
)


@pytest.mark.parametrize(
    ("system_name", "machine_name", "platform_tag", "binary_path"),
    [
        ("Linux", "amd64", "linux-x86_64", "bin/watchman"),
        ("Darwin", "arm64", "macos-arm64", "bin/watchman"),
        ("Darwin", "x86_64", "macos-x86_64", "bin/watchman"),
        ("Windows", "amd64", "windows-x86_64", "bin/watchman.cmd"),
    ],
)
def test_resolve_packaged_watchman_runtime_declared_slots(
    system_name: str,
    machine_name: str,
    platform_tag: str,
    binary_path: str,
) -> None:
    runtime = resolve_packaged_watchman_runtime(
        system_name=system_name, machine_name=machine_name
    )

    assert runtime.platform_tag == platform_tag
    assert runtime.relative_binary_path.as_posix() == binary_path
    assert runtime.probe_args == ("--version",)
    assert "platform-specific" in runtime.packaging_decision
    assert runtime.source_size > 0


def test_resolve_packaged_watchman_runtime_rejects_unknown_platform() -> None:
    with pytest.raises(UnsupportedWatchmanRuntimePlatformError):
        resolve_packaged_watchman_runtime(system_name="Linux", machine_name="ppc64le")


def _host_probe_command(*, binary_path: Path, probe_args: tuple[str, ...]) -> list[str]:
    if os.name == "nt":
        return ["cmd.exe", "/c", str(binary_path), *probe_args]
    return [str(binary_path), *probe_args]


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
                "materialized Watchman runtime exited before it created sidecar files "
                f"(rc={process.returncode})"
            )
        time.sleep(0.05)
    raise AssertionError("timed out waiting for packaged Watchman sidecar files")


def _stop_sidecar_process(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2.0)


def test_materialize_watchman_binary_writes_executable_for_host(tmp_path: Path) -> None:
    runtime = resolve_packaged_watchman_runtime()

    binary_path = materialize_watchman_binary(destination_root=tmp_path)

    assert binary_path == (
        tmp_path
        / runtime.platform_tag
        / runtime.runtime_version
        / Path(*runtime.relative_binary_path.parts)
    )
    assert binary_path.is_file()
    if os.name != "nt":
        assert binary_path.stat().st_mode & stat.S_IXUSR

    result = subprocess.run(
        _host_probe_command(binary_path=binary_path, probe_args=runtime.probe_args),
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0
    assert "watchman" in result.stdout.lower()
    assert "placeholder" in result.stdout.lower()


def test_materialized_watchman_binary_supports_private_sidecar_flags(
    tmp_path: Path,
) -> None:
    binary_path = materialize_watchman_binary(destination_root=tmp_path)
    sidecar_root = tmp_path / "sidecar"
    socket_path = sidecar_root / "sock"
    statefile_path = sidecar_root / "state"
    logfile_path = sidecar_root / "watchman.log"

    process = subprocess.Popen(
        _host_sidecar_command(
            binary_path=binary_path,
            socket_path=socket_path,
            statefile_path=statefile_path,
            logfile_path=logfile_path,
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_sidecar_files(
            process=process,
            socket_path=socket_path,
            statefile_path=statefile_path,
            logfile_path=logfile_path,
        )
        assert process.poll() is None
        assert "fake watchman start" in logfile_path.read_text(encoding="utf-8")
    finally:
        if process.poll() is None:
            _stop_sidecar_process(process)


def test_materialize_watchman_binary_rewrites_corrupt_payload(tmp_path: Path) -> None:
    binary_path = materialize_watchman_binary(destination_root=tmp_path)
    binary_path.write_text("corrupt\n", encoding="utf-8")

    repaired_path = materialize_watchman_binary(destination_root=tmp_path)

    assert repaired_path == binary_path
    assert "corrupt" not in repaired_path.read_text(encoding="utf-8")
