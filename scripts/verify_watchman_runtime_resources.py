from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

_REQUIRED_WHEEL_PATHS: tuple[str, ...] = (
    "chunkhound/watchman_runtime/__init__.py",
    "chunkhound/watchman_runtime/README.md",
    "chunkhound/watchman_runtime/platforms/linux-x86_64/manifest.json",
    "chunkhound/watchman_runtime/platforms/linux-x86_64/bin/watchman",
    "chunkhound/watchman_runtime/platforms/macos-arm64/manifest.json",
    "chunkhound/watchman_runtime/platforms/macos-arm64/bin/watchman",
    "chunkhound/watchman_runtime/platforms/macos-x86_64/manifest.json",
    "chunkhound/watchman_runtime/platforms/macos-x86_64/bin/watchman",
    "chunkhound/watchman_runtime/platforms/windows-x86_64/manifest.json",
    "chunkhound/watchman_runtime/platforms/windows-x86_64/bin/watchman.cmd",
    "chunkhound/watchman_runtime/platforms/windows-x86_64/bin/watchman.ps1",
)


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


def _verify_wheel_has_platform_only_tag(wheel_path: Path) -> None:
    if "-py3-none-" not in wheel_path.name or wheel_path.name.endswith("any.whl"):
        raise RuntimeError(
            "Watchman runtime wheels must use an explicit py3-none-platform tag: "
            f"{wheel_path}"
        )


def _verify_wheel_contents(wheel_path: Path) -> None:
    with zipfile.ZipFile(wheel_path) as zf:
        names = set(zf.namelist())
    missing = [path for path in _REQUIRED_WHEEL_PATHS if path not in names]
    if missing:
        missing_rendered = "\n".join(f"- {item}" for item in missing)
        raise RuntimeError(
            "Wheel is missing required Watchman runtime resources: "
            f"{wheel_path}\n{missing_rendered}"
        )


def _verify_runtime_reads(*, wheel_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="chunkhound-watchman-wheel-verify-") as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(wheel_path) as zf:
            zf.extractall(root)

        code = "\n".join(
            [
                "import os",
                "import json",
                "import subprocess",
                "import sys",
                "import time",
                "from pathlib import Path",
                "",
                (
                    "from chunkhound.watchman_runtime.loader import "
                    "materialize_watchman_binary, "
                    "resolve_packaged_watchman_runtime"
                ),
                "",
                "runtime = resolve_packaged_watchman_runtime()",
                (
                    "binary_path = materialize_watchman_binary("
                    "destination_root=Path('materialized'))"
                ),
                "assert binary_path.is_file()",
                "if os.name != 'nt':",
                "    assert os.access(binary_path, os.X_OK)",
                "if os.name == 'nt':",
                (
                    "    command = ['cmd.exe', '/c', str(binary_path), "
                    "*runtime.probe_args]"
                ),
                "else:",
                "    command = [str(binary_path), *runtime.probe_args]",
                (
                "result = subprocess.run(command, check=True, capture_output=True, "
                "text=True)"
                ),
                "assert 'watchman' in result.stdout.lower()",
                "assert 'placeholder' in result.stdout.lower()",
                "sidecar_root = Path('sidecar')",
                "socket_path = sidecar_root / 'sock'",
                "statefile_path = sidecar_root / 'state'",
                "logfile_path = sidecar_root / 'watchman.log'",
                "sidecar_command = [",
                "    str(binary_path),",
                "    '--foreground',",
                "    '--sockname',",
                "    str(socket_path),",
                "    '--statefile',",
                "    str(statefile_path),",
                "    '--logfile',",
                "    str(logfile_path),",
                "    '--no-save-state',",
                "]",
                "if os.name == 'nt':",
                "    sidecar_command = ['cmd.exe', '/c', *sidecar_command]",
                (
                    "sidecar = subprocess.Popen("
                    "sidecar_command, stdin=subprocess.DEVNULL, "
                    "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)"
                ),
                "deadline = time.monotonic() + 5.0",
                "while time.monotonic() < deadline:",
                (
                    "    if socket_path.exists() and statefile_path.exists() "
                    "and logfile_path.exists():"
                ),
                "        break",
                "    if sidecar.poll() is not None:",
                (
                    "        raise AssertionError("
                    "f'packaged runtime exited early: {sidecar.returncode}')"
                ),
                "    time.sleep(0.05)",
                "assert socket_path.exists()",
                "assert statefile_path.exists()",
                "assert logfile_path.exists()",
                "assert sidecar.poll() is None",
                "client_command = [",
                "    str(binary_path),",
                "    '--sockname',",
                "    str(socket_path),",
                "    '--no-spawn',",
                "    '--no-pretty',",
                "    '--persistent',",
                "    '--server-encoding',",
                "    'json',",
                "    '--output-encoding',",
                "    'json',",
                "    '--json-command',",
                "]",
                "if os.name == 'nt':",
                "    client_command = ['cmd.exe', '/c', *client_command]",
                (
                    "client = subprocess.Popen("
                    "client_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, "
                    "stderr=subprocess.PIPE, text=True)"
                ),
                "assert client.stdin is not None",
                "assert client.stdout is not None",
                (
                    "client.stdin.write(json.dumps(['version', {'required': "
                    "['cmd-watch-project', 'relative_root']}]) + '\\n')"
                ),
                "client.stdin.flush()",
                "version_response = json.loads(client.stdout.readline())",
                (
                    "assert version_response['capabilities'] == {"
                    "'cmd-watch-project': True, 'relative_root': True}"
                ),
                (
                    "project_root = Path('project'); "
                    "project_root.mkdir(exist_ok=True)"
                ),
                (
                    "client.stdin.write(json.dumps(['watch-project', "
                    "str(project_root.resolve())]) + '\\n')"
                ),
                "client.stdin.flush()",
                "watch_project = json.loads(client.stdout.readline())",
                "assert watch_project['watch'] == str(project_root.resolve())",
                (
                    "client.stdin.write(json.dumps(['subscribe', "
                    "str(project_root.resolve()), 'chunkhound-live-indexing', "
                    "{'fields': ['name', 'exists', 'new', 'type']}]) + '\\n')"
                ),
                "client.stdin.flush()",
                "subscribe_response = json.loads(client.stdout.readline())",
                (
                    "assert subscribe_response['subscribe'] == "
                    "'chunkhound-live-indexing'"
                ),
                "client.stdin.close()",
                "try:",
                "    client.wait(timeout=5.0)",
                "except subprocess.TimeoutExpired:",
                "    client.terminate()",
                "    client.wait(timeout=2.0)",
                "sidecar.terminate()",
                "try:",
                "    sidecar.wait(timeout=5.0)",
                "except subprocess.TimeoutExpired:",
                "    sidecar.kill()",
                "    sidecar.wait(timeout=2.0)",
                "sys.exit(0)",
            ]
        )

        env = os.environ.copy()
        env["PYTHONPATH"] = str(root)

        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(root),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Wheel runtime resource verification failed.\n"
                f"wheel={wheel_path}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}\n"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify Watchman runtime payloads exist in a built wheel and can be "
            "materialized into a runnable binary path."
        )
    )
    parser.add_argument(
        "wheels",
        nargs="+",
        type=Path,
        help="Path(s) to .whl file(s) to verify.",
    )
    args = parser.parse_args(argv)

    wheel_paths: list[Path] = []
    for raw in args.wheels:
        if raw.is_file() and raw.suffix == ".whl":
            wheel_paths.append(raw)
            continue
        raise FileNotFoundError(f"Wheel not found: {raw}")

    for wheel_path in wheel_paths:
        _verify_wheel_has_platform_only_tag(wheel_path)
        _verify_wheel_contents(wheel_path)
        _verify_runtime_reads(wheel_path=wheel_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
