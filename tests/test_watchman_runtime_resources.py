from __future__ import annotations

import hashlib
import io
import json
import os
import queue
import stat
import subprocess
import tarfile
import threading
import time
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import TextIO

import psutil
import pytest

from chunkhound.watchman_runtime import loader as watchman_runtime_loader_module
from chunkhound.watchman_runtime.loader import (
    UnsupportedWatchmanRuntimePlatformError,
    build_watchman_client_command,
    build_watchman_probe_command,
    build_watchman_runtime_command_prefix,
    build_watchman_runtime_environment,
    build_watchman_sidecar_command,
    listener_path_is_filesystem,
    materialize_watchman_binary,
    resolve_packaged_watchman_runtime,
)

pytestmark = pytest.mark.requires_native_watchman


def _build_ar_member(name: str, payload: bytes) -> bytes:
    header = (
        f"{name}/".ljust(16)
        + f"{0:<12}"
        + f"{0:<6}"
        + f"{0:<6}"
        + f"{100644:<8}"
        + f"{len(payload):<10}"
        + "`\n"
    ).encode("ascii")
    padding = b"\n" if len(payload) % 2 else b""
    return header + payload + padding


def _build_deb_archive(*, data_member_name: str, tar_payload: bytes) -> bytes:
    return (
        b"!<arch>\n"
        + _build_ar_member("debian-binary", b"2.0\n")
        + _build_ar_member(data_member_name, tar_payload)
    )


@pytest.mark.parametrize(
    (
        "system_name",
        "machine_name",
        "platform_tag",
        "binary_path",
        "support_paths",
        "listener_transport",
        "env_path_entries",
        "source_archive_count",
    ),
    [
        (
            "Linux",
            "amd64",
            "linux-x86_64",
            "bin/watchman",
            (
                PurePosixPath("lib/libboost_context.so.1.74.0"),
                PurePosixPath("lib/libdouble-conversion.so.3"),
                PurePosixPath("lib/libevent-2.1.so.7"),
                PurePosixPath("lib/libgflags.so.2.2"),
                PurePosixPath("lib/liblz4.so.1"),
                PurePosixPath("lib/libsnappy.so.1"),
                PurePosixPath("lib/libxxhash.so.0"),
                PurePosixPath("lib/libzstd.so.1"),
            ),
            "unix_socket",
            {"LD_LIBRARY_PATH": (PurePosixPath("lib"),)},
            9,
        ),
        (
            "Windows",
            "AMD64",
            "windows-x86_64",
            "bin/watchman.exe",
            (
                PurePosixPath("bin/eledo-pty-bridge.exe"),
                PurePosixPath("bin/gflags.dll"),
                PurePosixPath("bin/glog.dll"),
                PurePosixPath("bin/libcrypto-3.dll"),
                PurePosixPath("bin/watchman-diag.exe"),
                PurePosixPath("bin/watchman-make.exe"),
                PurePosixPath("bin/watchman-replicate-subscription.exe"),
                PurePosixPath("bin/watchman-wait.exe"),
                PurePosixPath("bin/watchmanctl.exe"),
            ),
            "named_pipe",
            {"PATH": (PurePosixPath("bin"),)},
            1,
        ),
    ],
)
def test_resolve_packaged_watchman_runtime_declared_slots(
    system_name: str,
    machine_name: str,
    platform_tag: str,
    binary_path: str,
    support_paths: tuple[PurePosixPath, ...],
    listener_transport: str,
    env_path_entries: dict[str, tuple[PurePosixPath, ...]],
    source_archive_count: int,
) -> None:
    runtime = resolve_packaged_watchman_runtime(
        system_name=system_name, machine_name=machine_name
    )

    assert runtime.platform_tag == platform_tag
    assert runtime.relative_binary_path.as_posix() == binary_path
    assert runtime.relative_support_paths == support_paths
    assert runtime.launch_mode == "native_binary"
    assert runtime.listener_transport == listener_transport
    assert runtime.probe_args == ("--version",)
    assert runtime.env_path_entries == env_path_entries
    assert len(runtime.source_archives) == source_archive_count
    assert "platform-specific" in runtime.packaging_decision
    assert runtime.source_size > 0


def test_build_watchman_runtime_command_prefix_uses_current_interpreter_for_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = replace(resolve_packaged_watchman_runtime(), launch_mode="python_bridge")
    binary_path = Path("/tmp/watchman")
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


def _runtime_env(*, binary_path: Path) -> dict[str, str]:
    runtime = resolve_packaged_watchman_runtime()
    return build_watchman_runtime_environment(runtime=runtime, binary_path=binary_path)


def _wait_for_sidecar_files(
    *,
    runtime,
    process: subprocess.Popen,
    socket_path: str | Path,
    pidfile_path: Path,
    logfile_path: Path,
) -> None:
    deadline = time.monotonic() + (
        5.0 if listener_path_is_filesystem(runtime) else 15.0
    )
    while time.monotonic() < deadline:
        listener_ready = True
        if listener_path_is_filesystem(runtime):
            listener_ready = Path(socket_path).exists()
        artifacts_ready = True
        if listener_path_is_filesystem(runtime):
            artifacts_ready = pidfile_path.exists() and logfile_path.exists()
        if listener_ready and artifacts_ready:
            return
        if process.poll() is not None:
            raise AssertionError(
                "materialized Watchman runtime exited before it created sidecar files "
                f"(rc={process.returncode})"
            )
        time.sleep(0.05)
    raise AssertionError("timed out waiting for packaged Watchman sidecar files")


def _wait_for_named_pipe_ready(
    *,
    runtime,
    binary_path: Path,
    socket_path: str | Path,
    pidfile_path: Path,
    statefile_path: Path,
    logfile_path: Path,
    env: dict[str, str],
    timeout_seconds: float = 15.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            _run_one_shot_command(
                runtime=runtime,
                binary_path=binary_path,
                socket_path=socket_path,
                pidfile_path=pidfile_path,
                statefile_path=statefile_path,
                logfile_path=logfile_path,
                command=["version"],
                env=env,
            )
            return
        except AssertionError:
            time.sleep(0.1)
    raise AssertionError("timed out waiting for packaged Watchman named pipe readiness")


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


def _run_one_shot_command(
    *,
    runtime,
    binary_path: Path,
    socket_path: str | Path,
    pidfile_path: Path,
    statefile_path: Path,
    logfile_path: Path,
    command: list[object],
    env: dict[str, str],
) -> dict[str, object]:
    result = subprocess.run(
        build_watchman_client_command(
            runtime=runtime,
            binary_path=binary_path,
            socket_path=socket_path,
            statefile_path=statefile_path,
            logfile_path=logfile_path,
            pidfile_path=pidfile_path,
            persistent=False,
        ),
        input=json.dumps(command) + "\n",
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            "watchman one-shot command failed: "
            f"cmd={command[0]!r} rc={result.returncode} stderr={result.stderr!r}"
        )
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict)
    return payload


def _runtime_listener_path(*, runtime, sidecar_root: Path, suffix: str) -> str | Path:
    if listener_path_is_filesystem(runtime):
        return sidecar_root / suffix
    return rf"\\.\pipe\chunkhound-watchman-{suffix}-{os.getpid()}"


def test_build_watchman_command_uses_named_pipe_for_windows_native_runtime() -> None:
    runtime = resolve_packaged_watchman_runtime(
        system_name="Windows",
        machine_name="AMD64",
    )
    binary_path = Path(r"C:\runtime\watchman.exe")
    named_pipe = r"\\.\pipe\chunkhound-watchman-test"

    sidecar_command = build_watchman_sidecar_command(
        runtime=runtime,
        binary_path=binary_path,
        socket_path=named_pipe,
        statefile_path=Path(r"C:\runtime\watchman.state"),
        logfile_path=Path(r"C:\runtime\watchman.log"),
        pidfile_path=Path(r"C:\runtime\watchman.pid"),
    )
    client_command = build_watchman_client_command(
        runtime=runtime,
        binary_path=binary_path,
        socket_path=named_pipe,
        statefile_path=Path(r"C:\runtime\watchman.state"),
        logfile_path=Path(r"C:\runtime\watchman.log"),
        pidfile_path=Path(r"C:\runtime\watchman.pid"),
        persistent=False,
    )

    assert "--named-pipe-path" in sidecar_command
    assert "--unix-listener-path" not in sidecar_command
    assert "--named-pipe-path" in client_command
    assert "--unix-listener-path" not in client_command


def test_read_deb_member_bytes_reads_expected_member_without_extractall(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tar_payload = io.BytesIO()
    with tarfile.open(fileobj=tar_payload, mode="w:gz") as handle:
        escaped_info = tarfile.TarInfo("../escape")
        escaped_payload = b"nope\n"
        escaped_info.size = len(escaped_payload)
        handle.addfile(escaped_info, io.BytesIO(escaped_payload))

        watchman_info = tarfile.TarInfo("./usr/local/bin/watchman")
        watchman_payload = b"#!/bin/sh\necho native-watchman\n"
        watchman_info.mode = 0o755
        watchman_info.size = len(watchman_payload)
        handle.addfile(watchman_info, io.BytesIO(watchman_payload))

    archive_path = tmp_path / "watchman.deb"
    archive_path.write_bytes(
        _build_deb_archive(
            data_member_name="data.tar.gz",
            tar_payload=tar_payload.getvalue(),
        )
    )

    def _fail_extractall(self, path=".", members=None, *, numeric_owner=False):
        raise AssertionError("deb hydration should not call tarfile.extractall()")

    monkeypatch.setattr(tarfile.TarFile, "extractall", _fail_extractall)

    payload = watchman_runtime_loader_module._read_deb_member_bytes(
        archive_path,
        source_root_prefix=PurePosixPath("usr/local"),
        source_relative_path=PurePosixPath("bin/watchman"),
    )

    assert payload == b"#!/bin/sh\necho native-watchman\n"


def test_read_deb_member_bytes_reads_symlinked_target_member(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tar_payload = io.BytesIO()
    with tarfile.open(fileobj=tar_payload, mode="w:gz") as handle:
        watchman_real = tarfile.TarInfo("usr/local/bin/watchman-real")
        watchman_payload = b"#!/bin/sh\necho native-watchman\n"
        watchman_real.mode = 0o755
        watchman_real.size = len(watchman_payload)
        handle.addfile(watchman_real, io.BytesIO(watchman_payload))

        watchman_link = tarfile.TarInfo("usr/local/bin/watchman")
        watchman_link.type = tarfile.SYMTYPE
        watchman_link.linkname = "watchman-real"
        handle.addfile(watchman_link)

    archive_path = tmp_path / "watchman.deb"
    archive_path.write_bytes(
        _build_deb_archive(
            data_member_name="data.tar.gz",
            tar_payload=tar_payload.getvalue(),
        )
    )

    def _fail_extractall(self, path=".", members=None, *, numeric_owner=False):
        raise AssertionError("deb hydration should not call tarfile.extractall()")

    monkeypatch.setattr(tarfile.TarFile, "extractall", _fail_extractall)

    payload = watchman_runtime_loader_module._read_deb_member_bytes(
        archive_path,
        source_root_prefix=PurePosixPath("usr/local"),
        source_relative_path=PurePosixPath("bin/watchman"),
    )

    assert payload == b"#!/bin/sh\necho native-watchman\n"


def test_read_deb_member_bytes_rejects_unsafe_link_target(
    tmp_path: Path,
) -> None:
    tar_payload = io.BytesIO()
    with tarfile.open(fileobj=tar_payload, mode="w:gz") as handle:
        watchman_link = tarfile.TarInfo("usr/local/bin/watchman")
        watchman_link.type = tarfile.SYMTYPE
        watchman_link.linkname = "../../../../escape"
        handle.addfile(watchman_link)

    archive_path = tmp_path / "watchman.deb"
    archive_path.write_bytes(
        _build_deb_archive(
            data_member_name="data.tar.gz",
            tar_payload=tar_payload.getvalue(),
        )
    )

    with pytest.raises(RuntimeError, match="link target is unsafe"):
        watchman_runtime_loader_module._read_deb_member_bytes(
            archive_path,
            source_root_prefix=PurePosixPath("usr/local"),
            source_relative_path=PurePosixPath("bin/watchman"),
        )


def test_materialize_watchman_binary_writes_windows_payload_tree(
    tmp_path: Path,
) -> None:
    runtime = resolve_packaged_watchman_runtime(
        system_name="Windows",
        machine_name="AMD64",
    )

    binary_path = materialize_watchman_binary(
        destination_root=tmp_path,
        system_name="Windows",
        machine_name="AMD64",
    )

    assert binary_path == (
        tmp_path
        / runtime.platform_tag
        / runtime.runtime_version
        / Path(*runtime.relative_binary_path.parts)
    )
    assert binary_path.is_file()
    for relative_support_path in runtime.relative_support_paths:
        support_path = runtime.materialized_root(binary_path) / Path(
            *relative_support_path.parts
        )
        assert support_path.is_file()


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
    for relative_support_path in runtime.relative_support_paths:
        support_path = runtime.materialized_root(binary_path) / Path(
            *relative_support_path.parts
        )
        assert support_path.is_file()

    result = subprocess.run(
        build_watchman_probe_command(runtime=runtime, binary_path=binary_path),
        capture_output=True,
        check=False,
        env=_runtime_env(binary_path=binary_path),
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == runtime.runtime_version


def test_materialized_watchman_binary_supports_private_sidecar_flags(
    tmp_path: Path,
) -> None:
    runtime = resolve_packaged_watchman_runtime()
    binary_path = materialize_watchman_binary(destination_root=tmp_path)
    runtime_env = _runtime_env(binary_path=binary_path)
    sidecar_root = tmp_path / "sidecar"
    sidecar_root.mkdir(parents=True, exist_ok=True)
    socket_path = _runtime_listener_path(
        runtime=runtime,
        sidecar_root=sidecar_root,
        suffix="flags",
    )
    pidfile_path = sidecar_root / "pid"
    statefile_path = sidecar_root / "state"
    logfile_path = sidecar_root / "watchman.log"

    process = subprocess.Popen(
        build_watchman_sidecar_command(
            runtime=runtime,
            binary_path=binary_path,
            socket_path=socket_path,
            statefile_path=statefile_path,
            logfile_path=logfile_path,
            pidfile_path=pidfile_path,
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=runtime_env,
    )
    try:
        _wait_for_sidecar_files(
            runtime=runtime,
            process=process,
            socket_path=socket_path,
            pidfile_path=pidfile_path,
            logfile_path=logfile_path,
        )
        _wait_for_named_pipe_ready(
            runtime=runtime,
            binary_path=binary_path,
            socket_path=socket_path,
            pidfile_path=pidfile_path,
            statefile_path=statefile_path,
            logfile_path=logfile_path,
            env=runtime_env,
        )
        assert process.poll() is None
        cmdline = psutil.Process(process.pid).cmdline()
        assert cmdline
        assert cmdline[0] == str(binary_path)
        assert "chunkhound.watchman_runtime.bridge" not in " ".join(cmdline)
    finally:
        if process.poll() is None:
            _stop_sidecar_process(process)


def test_materialized_watchman_binary_supports_persistent_client_session(
    tmp_path: Path,
) -> None:
    runtime = resolve_packaged_watchman_runtime()
    binary_path = materialize_watchman_binary(destination_root=tmp_path)
    sidecar_root = tmp_path / "sidecar"
    sidecar_root.mkdir(parents=True, exist_ok=True)
    socket_path = _runtime_listener_path(
        runtime=runtime,
        sidecar_root=sidecar_root,
        suffix="session",
    )
    pidfile_path = sidecar_root / "pid"
    statefile_path = sidecar_root / "state"
    logfile_path = sidecar_root / "watchman.log"
    project_root = tmp_path / "repo"
    project_root.mkdir()

    sidecar = subprocess.Popen(
        build_watchman_sidecar_command(
            runtime=runtime,
            binary_path=binary_path,
            socket_path=socket_path,
            statefile_path=statefile_path,
            logfile_path=logfile_path,
            pidfile_path=pidfile_path,
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=_runtime_env(binary_path=binary_path),
    )
    client: subprocess.Popen[str] | None = None
    try:
        _wait_for_sidecar_files(
            runtime=runtime,
            process=sidecar,
            socket_path=socket_path,
            pidfile_path=pidfile_path,
            logfile_path=logfile_path,
        )
        runtime_env = _runtime_env(binary_path=binary_path)

        version_response = _run_one_shot_command(
            runtime=runtime,
            binary_path=binary_path,
            socket_path=socket_path,
            pidfile_path=pidfile_path,
            statefile_path=statefile_path,
            logfile_path=logfile_path,
            command=["version", {"required": ["cmd-watch-project", "relative_root"]}],
            env=runtime_env,
        )
        capabilities = version_response.get("capabilities")
        assert isinstance(capabilities, dict)
        assert capabilities.get("cmd-watch-project") is True
        assert capabilities.get("relative_root") is True

        watch_project_response = _run_one_shot_command(
            runtime=runtime,
            binary_path=binary_path,
            socket_path=socket_path,
            pidfile_path=pidfile_path,
            statefile_path=statefile_path,
            logfile_path=logfile_path,
            command=["watch-project", str(project_root)],
            env=runtime_env,
        )
        assert Path(str(watch_project_response["watch"])).resolve() == project_root
        assert "relative_path" not in watch_project_response

        client = subprocess.Popen(
            build_watchman_client_command(
                runtime=runtime,
                binary_path=binary_path,
                socket_path=socket_path,
                statefile_path=statefile_path,
                logfile_path=logfile_path,
                pidfile_path=pidfile_path,
            ),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=runtime_env,
            text=True,
        )
        assert client.stdin is not None
        assert client.stdout is not None
        reader = _JsonLineReader(client.stdout)

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
    runtime = resolve_packaged_watchman_runtime()
    binary_path = materialize_watchman_binary(destination_root=tmp_path)
    binary_path.write_bytes(b"corrupt\n")

    repaired_path = materialize_watchman_binary(destination_root=tmp_path)

    assert repaired_path == binary_path
    assert (
        hashlib.sha256(repaired_path.read_bytes()).hexdigest()
        == runtime.source_digest
    )
