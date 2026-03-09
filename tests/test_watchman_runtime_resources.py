from __future__ import annotations

import os
import stat
import subprocess
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


def test_materialize_watchman_binary_rewrites_corrupt_payload(tmp_path: Path) -> None:
    binary_path = materialize_watchman_binary(destination_root=tmp_path)
    binary_path.write_text("corrupt\n", encoding="utf-8")

    repaired_path = materialize_watchman_binary(destination_root=tmp_path)

    assert repaired_path == binary_path
    assert "corrupt" not in repaired_path.read_text(encoding="utf-8")
