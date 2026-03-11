from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import psutil

_REQUIRED_WHEEL_PATHS: tuple[str, ...] = (
    "chunkhound/watchman_runtime/__init__.py",
    "chunkhound/watchman_runtime/bridge.py",
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
_LIVE_MUTATION_TIMEOUT_ENV = (
    "CHUNKHOUND_WATCHMAN_RUNTIME_VERIFY_LIVE_TIMEOUT_SECONDS"
)


def _terminate_process_tree(pid: int) -> None:
    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    try:
        processes = root.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        processes = []
    processes.append(root)

    for process in processes:
        try:
            process.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    _, alive = psutil.wait_procs(processes, timeout=2.0)
    for process in alive:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    psutil.wait_procs(alive, timeout=2.0)


def _terminate_processes_using_root(root: Path) -> None:
    root_str = str(root)
    current_pid = os.getpid()
    candidates: list[int] = []

    for process in psutil.process_iter(["pid", "cwd", "cmdline"]):
        pid = process.info.get("pid")
        if not isinstance(pid, int) or pid == current_pid:
            continue

        try:
            cwd = process.info.get("cwd")
            cmdline = process.info.get("cmdline") or []
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

        if isinstance(cwd, str) and cwd.startswith(root_str):
            candidates.append(pid)
            continue
        if any(isinstance(arg, str) and root_str in arg for arg in cmdline):
            candidates.append(pid)

    for pid in candidates:
        _terminate_process_tree(pid)


def _remove_tree_with_retries(
    root: Path, *, attempts: int = 5, base_delay_seconds: float = 0.2
) -> None:
    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            shutil.rmtree(root)
            return
        except FileNotFoundError:
            return
        except OSError as error:
            last_error = error
            if os.name == "nt":
                _terminate_processes_using_root(root)
            if attempt == attempts - 1:
                raise
            time.sleep(base_delay_seconds * (attempt + 1))

    if last_error is not None:
        raise last_error


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
    root = Path(tempfile.mkdtemp(prefix="chunkhound-watchman-wheel-verify-"))
    try:
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
                "assert runtime.runtime_version in result.stdout",
                "",
                "def _stop_process(process, *, close_stdin=False):",
                "    if process is None:",
                "        return",
                "    if close_stdin and process.stdin is not None:",
                "        try:",
                "            process.stdin.close()",
                "        except OSError:",
                "            pass",
                "    try:",
                "        process.wait(timeout=5.0)",
                "    except subprocess.TimeoutExpired:",
                "        process.terminate()",
                "        try:",
                "            process.wait(timeout=2.0)",
                "        except subprocess.TimeoutExpired:",
                "            process.kill()",
                "            process.wait(timeout=2.0)",
                "",
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
                "sidecar = None",
                "client = None",
                "reader_thread = None",
                "try:",
                (
                    "    sidecar = subprocess.Popen("
                    "sidecar_command, stdin=subprocess.DEVNULL, "
                    "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)"
                ),
                "    deadline = time.monotonic() + 5.0",
                "    while time.monotonic() < deadline:",
                (
                    "        if socket_path.exists() and statefile_path.exists() "
                    "and logfile_path.exists():"
                ),
                "            break",
                "        if sidecar.poll() is not None:",
                (
                    "            raise AssertionError("
                    "f'packaged runtime exited early: {sidecar.returncode}')"
                ),
                "        time.sleep(0.05)",
                "    assert socket_path.exists()",
                "    assert statefile_path.exists()",
                "    assert logfile_path.exists()",
                "    assert sidecar.poll() is None",
                (
                    "    assert 'watchman runtime sidecar start' in "
                    "logfile_path.read_text(encoding='utf-8')"
                ),
                "    client_command = [",
                "        str(binary_path),",
                "        '--sockname',",
                "        str(socket_path),",
                "        '--no-spawn',",
                "        '--no-pretty',",
                "        '--persistent',",
                "        '--server-encoding',",
                "        'json',",
                "        '--output-encoding',",
                "        'json',",
                "        '--json-command',",
                "    ]",
                "    if os.name == 'nt':",
                "        client_command = ['cmd.exe', '/c', *client_command]",
                (
                    "    client = subprocess.Popen("
                    "client_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, "
                    "stderr=subprocess.PIPE, text=True)"
                ),
                "    assert client.stdin is not None",
                "    assert client.stdout is not None",
                "    import queue",
                "    import threading",
                "    responses = queue.Queue()",
                "    EOF = object()",
                "    def _reader():",
                "        while True:",
                "            line = client.stdout.readline()",
                "            if not line:",
                "                responses.put(EOF)",
                "                return",
                "            responses.put(json.loads(line))",
                "    reader_thread = threading.Thread(target=_reader)",
                "    reader_thread.start()",
                (
                    "    client.stdin.write(json.dumps(['version', {'required': "
                    "['cmd-watch-project', 'relative_root']}]) + '\\n')"
                ),
                "    client.stdin.flush()",
                "    version_response = responses.get(timeout=5.0)",
                (
                    "    assert version_response['capabilities'] == {"
                    "'cmd-watch-project': True, 'relative_root': True}"
                ),
                (
                    "    project_root = Path('project'); "
                    "project_root.mkdir(exist_ok=True)"
                ),
                (
                    "    client.stdin.write(json.dumps(['watch-project', "
                    "str(project_root.resolve())]) + '\\n')"
                ),
                "    client.stdin.flush()",
                "    watch_project = responses.get(timeout=5.0)",
                "    assert watch_project['watch'] == str(project_root.resolve())",
                (
                    "    client.stdin.write(json.dumps(['subscribe', "
                    "str(project_root.resolve()), 'chunkhound-live-indexing', "
                    "{'fields': ['name', 'exists', 'new', 'type']}]) + '\\n')"
                ),
                "    client.stdin.flush()",
                "    subscribe_response = responses.get(timeout=5.0)",
                (
                    "    assert subscribe_response['subscribe'] == "
                    "'chunkhound-live-indexing'"
                ),
                (
                    "    live_file = (project_root / 'src' / "
                    "'installed_runtime_live.py'); "
                    "live_file.parent.mkdir(parents=True, exist_ok=True)"
                ),
                (
                    "    live_file.write_text("
                    "'def installed_runtime_live_symbol():\\n    return 1\\n', "
                    "encoding='utf-8')"
                ),
                (
                    "    live_timeout = float(os.environ.get("
                    f"'{_LIVE_MUTATION_TIMEOUT_ENV}', '10.0'))"
                ),
                "    deadline = time.monotonic() + live_timeout",
                "    live_payload = None",
                "    while time.monotonic() < deadline:",
                "        try:",
                "            payload = responses.get(timeout=1.0)",
                "        except queue.Empty:",
                "            continue",
                "        if payload is EOF:",
                (
                    "            raise AssertionError("
                    "'watchman client exited before live mutation delivery'"
                    ")"
                ),
                "        if payload.get('subscription') != 'chunkhound-live-indexing':",
                "            continue",
                "        files = payload.get('files')",
                "        if not isinstance(files, list):",
                "            continue",
                "        if any(",
                "            isinstance(item, dict)",
                "            and item.get('name') == 'src/installed_runtime_live.py'",
                "            and item.get('exists') is True",
                "            and item.get('type') == 'f'",
                "            for item in files",
                "        ):",
                "            live_payload = payload",
                "            break",
                "    if live_payload is None:",
                (
                    "        raise AssertionError("
                    "'timed out waiting for live subscription payload'"
                    ")"
                ),
                "finally:",
                "    if client is not None:",
                "        _stop_process(client, close_stdin=True)",
                "        if reader_thread is not None:",
                "            reader_thread.join(timeout=5.0)",
                "            assert not reader_thread.is_alive(), (",
                "                'watchman client reader thread did not exit'",
                "            )",
                "        if client.stdout is not None:",
                "            client.stdout.close()",
                "        if client.stderr is not None:",
                "            client.stderr.close()",
                "    if sidecar is not None:",
                "        _stop_process(sidecar)",
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
    finally:
        _remove_tree_with_retries(root)


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
