from __future__ import annotations

import json
import os
import queue
import stat
import subprocess
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import TextIO

import pytest

from chunkhound.watchman_runtime import loader as watchman_runtime_loader_module
from chunkhound.watchman_runtime.loader import (
    UnsupportedWatchmanRuntimePlatformError,
    build_watchman_runtime_command_prefix,
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
    assert runtime.launch_mode == "python_bridge"
    assert runtime.probe_args == ("--version",)
    assert "platform-specific" in runtime.packaging_decision
    assert runtime.source_size > 0


@pytest.mark.parametrize(
    ("system_name", "machine_name", "binary_path"),
    [
        ("Linux", "amd64", Path("/tmp/watchman")),
        ("Windows", "amd64", Path("C:/tmp/watchman.cmd")),
    ],
)
def test_build_watchman_runtime_command_prefix_uses_current_interpreter(
    system_name: str,
    machine_name: str,
    binary_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = resolve_packaged_watchman_runtime(
        system_name=system_name, machine_name=machine_name
    )
    monkeypatch.setattr(
        watchman_runtime_loader_module.sys,
        "executable",
        "/tmp/chunkhound-python",
        raising=False,
    )

    command = build_watchman_runtime_command_prefix(
        runtime=runtime,
        binary_path=binary_path,
    )

    assert command == [
        "/tmp/chunkhound-python",
        "-m",
        "chunkhound.watchman_runtime.bridge",
    ]


def test_build_watchman_runtime_command_prefix_uses_binary_for_native_launches() -> (
    None
):
    runtime = resolve_packaged_watchman_runtime()
    native_runtime = replace(runtime, launch_mode="native_binary")
    binary_path = Path("/tmp/native-watchman")

    command = build_watchman_runtime_command_prefix(
        runtime=native_runtime,
        binary_path=binary_path,
    )

    assert command == [str(binary_path)]


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


def _host_client_command(*, binary_path: Path, socket_path: Path) -> list[str]:
    command = [
        str(binary_path),
        "--sockname",
        str(socket_path),
        "--no-spawn",
        "--no-pretty",
        "--persistent",
        "--server-encoding",
        "json",
        "--output-encoding",
        "json",
        "--json-command",
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


def _stop_process(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2.0)


def _read_json_line(stream: TextIO) -> dict[str, object]:
    line = stream.readline()
    if not line:
        raise AssertionError("expected a JSON response from the Watchman client")
    payload = json.loads(line)
    assert isinstance(payload, dict)
    return payload


class _JsonLineReader:
    _EOF = object()

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._queue: queue.Queue[object] = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            line = self._stream.readline()
            if not line:
                self._queue.put(self._EOF)
                return
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                self._queue.put(error)
                return
            self._queue.put(payload)

    def read(self, *, timeout: float) -> dict[str, object]:
        try:
            payload = self._queue.get(timeout=timeout)
        except queue.Empty as error:
            raise AssertionError(
                "timed out waiting for Watchman JSON output"
            ) from error
        if payload is self._EOF:
            raise AssertionError("Watchman client exited before emitting JSON output")
        if isinstance(payload, json.JSONDecodeError):
            raise AssertionError(
                f"failed to decode Watchman JSON output: {payload}"
            ) from payload
        assert isinstance(payload, dict)
        return payload


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
    assert runtime.runtime_version in result.stdout


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
        assert "watchman runtime sidecar start" in logfile_path.read_text(
            encoding="utf-8"
        )
    finally:
        if process.poll() is None:
            _stop_sidecar_process(process)


def test_materialized_watchman_binary_supports_persistent_client_session(
    tmp_path: Path,
) -> None:
    binary_path = materialize_watchman_binary(destination_root=tmp_path)
    sidecar_root = tmp_path / "sidecar"
    socket_path = sidecar_root / "sock"
    statefile_path = sidecar_root / "state"
    logfile_path = sidecar_root / "watchman.log"
    project_root = tmp_path / "repo"
    project_root.mkdir()

    sidecar = subprocess.Popen(
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
    client: subprocess.Popen[str] | None = None
    try:
        _wait_for_sidecar_files(
            process=sidecar,
            socket_path=socket_path,
            statefile_path=statefile_path,
            logfile_path=logfile_path,
        )
        client = subprocess.Popen(
            _host_client_command(binary_path=binary_path, socket_path=socket_path),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert client.stdin is not None
        assert client.stdout is not None
        reader = _JsonLineReader(client.stdout)

        client.stdin.write(
            json.dumps(
                ["version", {"required": ["cmd-watch-project", "relative_root"]}]
            )
            + "\n"
        )
        client.stdin.flush()
        version_response = reader.read(timeout=5.0)
        assert version_response["capabilities"] == {
            "cmd-watch-project": True,
            "relative_root": True,
        }

        client.stdin.write(json.dumps(["watch-project", str(project_root)]) + "\n")
        client.stdin.flush()
        watch_project_response = reader.read(timeout=5.0)
        assert watch_project_response["watch"] == str(project_root)
        assert "relative_path" not in watch_project_response

        client.stdin.write(
            json.dumps(
                [
                    "subscribe",
                    str(project_root),
                    "chunkhound-live-indexing",
                    {"fields": ["name", "exists", "new", "type"]},
                ]
            )
            + "\n"
        )
        client.stdin.flush()
        subscribe_response = reader.read(timeout=5.0)
        assert subscribe_response["subscribe"] == "chunkhound-live-indexing"

        live_file = project_root / "src" / "runtime_live.py"
        live_file.parent.mkdir(parents=True, exist_ok=True)
        live_file.write_text(
            "def runtime_live_symbol():\n    return 1\n", encoding="utf-8"
        )

        deadline = time.monotonic() + 10.0
        live_payload: dict[str, object] | None = None
        while time.monotonic() < deadline:
            payload = reader.read(timeout=1.0)
            files = payload.get("files")
            if not isinstance(files, list):
                continue
            if payload.get("subscription") != "chunkhound-live-indexing":
                continue
            if any(
                isinstance(item, dict)
                and item.get("name") == "src/runtime_live.py"
                and item.get("exists") is True
                and item.get("type") == "f"
                for item in files
            ):
                live_payload = payload
                break

        assert live_payload is not None
    finally:
        if client is not None:
            if client.stdin is not None:
                client.stdin.close()
            if client.poll() is None:
                _stop_process(client)
        if sidecar.poll() is None:
            _stop_sidecar_process(sidecar)


def test_materialize_watchman_binary_rewrites_corrupt_payload(tmp_path: Path) -> None:
    binary_path = materialize_watchman_binary(destination_root=tmp_path)
    binary_path.write_text("corrupt\n", encoding="utf-8")

    repaired_path = materialize_watchman_binary(destination_root=tmp_path)

    assert repaired_path == binary_path
    assert "corrupt" not in repaired_path.read_text(encoding="utf-8")
