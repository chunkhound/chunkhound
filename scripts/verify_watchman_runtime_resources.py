from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import psutil

import hatch_build
from scripts import watchman_verifier_cleanup

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WATCHMAN_RUNTIME_ROOT = _REPO_ROOT / "chunkhound" / "watchman_runtime"
_LIVE_MUTATION_TIMEOUT_ENV = (
    "CHUNKHOUND_WATCHMAN_RUNTIME_VERIFY_LIVE_TIMEOUT_SECONDS"
)
_RUNTIME_SLOT_PREFIX = "chunkhound/watchman_runtime/platforms/"
_RUNTIME_MANIFEST_SUFFIX = "/manifest.json"


def _supported_runtime_platforms() -> tuple[str, ...]:
    return tuple(sorted(hatch_build._load_supported_watchman_platforms()))


def _runtime_manifest_path(platform_tag: str) -> Path:
    return _WATCHMAN_RUNTIME_ROOT / "platforms" / platform_tag / "manifest.json"


def _manifest_wheel_platform_tags(platform_tag: str) -> tuple[str, ...]:
    manifest_path = _runtime_manifest_path(platform_tag)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    wheel_platform_tags = manifest.get("wheel_platform_tags")
    if not isinstance(wheel_platform_tags, list) or not wheel_platform_tags:
        raise RuntimeError(
            "Watchman runtime manifest must declare non-empty wheel_platform_tags: "
            f"{manifest_path}"
        )
    parsed: list[str] = []
    for item in wheel_platform_tags:
        if not isinstance(item, str) or not item.strip():
            raise RuntimeError(
                "Watchman runtime manifest wheel_platform_tags must contain "
                f"non-empty strings: {manifest_path}"
            )
        parsed.append(item.strip())
    return tuple(parsed)


def _required_wheel_paths_for_platforms(
    platform_tags: tuple[str, ...],
) -> tuple[str, ...]:
    required = [
        "chunkhound/watchman_runtime/__init__.py",
        "chunkhound/watchman_runtime/README.md",
    ]
    for platform_tag in platform_tags:
        manifest_path = _runtime_manifest_path(platform_tag)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        required.append(
            f"chunkhound/watchman_runtime/platforms/{platform_tag}/manifest.json"
        )
        binary_rel = manifest["binary"]
        required.append(
            f"chunkhound/watchman_runtime/platforms/{platform_tag}/{binary_rel}"
        )
        for support_rel in manifest.get("support_files", []):
            required.append(
                f"chunkhound/watchman_runtime/platforms/{platform_tag}/{support_rel}"
            )
    return tuple(required)


def _wheel_platform_tags_from_filename(wheel_path: Path) -> tuple[str, ...]:
    platform_fragment = wheel_path.name[: -len(".whl")].rsplit("-py3-none-", 1)[-1]
    return tuple(tag for tag in platform_fragment.split(".") if tag)


def _runtime_platforms_in_wheel(
    *, wheel_path: Path, names: set[str]
) -> tuple[str, ...]:
    discovered = tuple(
        sorted(
            path[len(_RUNTIME_SLOT_PREFIX) : -len(_RUNTIME_MANIFEST_SUFFIX)]
            for path in names
            if path.startswith(_RUNTIME_SLOT_PREFIX)
            and path.endswith(_RUNTIME_MANIFEST_SUFFIX)
        )
    )
    if not discovered:
        raise RuntimeError(
            "Wheel is missing a required Watchman runtime manifest: "
            f"{wheel_path}"
        )

    supported = set(_supported_runtime_platforms())
    unexpected = sorted(
        platform for platform in discovered if platform not in supported
    )
    if unexpected:
        unexpected_rendered = ", ".join(unexpected)
        raise RuntimeError(
            "Wheel contains unsupported Watchman runtime platform manifests: "
            f"{wheel_path} ({unexpected_rendered})"
        )

    if len(discovered) != 1:
        rendered = ", ".join(discovered)
        raise RuntimeError(
            "Watchman runtime wheels must contain exactly one packaged runtime "
            f"platform slot: {wheel_path} ({rendered})"
        )

    allowed_wheel_tags = set(_manifest_wheel_platform_tags(discovered[0]))
    wheel_platform_tags = set(_wheel_platform_tags_from_filename(wheel_path))
    if not wheel_platform_tags <= allowed_wheel_tags:
        raise RuntimeError(
            "Wheel runtime slot does not match wheel tag: "
            f"{wheel_path} (slot={discovered[0]!r}, "
            f"actual={sorted(wheel_platform_tags)!r}, "
            f"allowed={sorted(allowed_wheel_tags)!r})"
        )
    return discovered


def _runtime_platform_for_wheel(wheel_path: Path) -> str:
    _verify_wheel_has_platform_only_tag(wheel_path)
    with zipfile.ZipFile(wheel_path) as zf:
        names = set(zf.namelist())
    runtime_platforms = _runtime_platforms_in_wheel(wheel_path=wheel_path, names=names)
    return runtime_platforms[0]


def _verify_supported_wheel_matrix(wheel_paths: list[Path]) -> dict[str, Path]:
    supported_platforms = set(_supported_runtime_platforms())
    discovered: dict[str, Path] = {}

    for wheel_path in wheel_paths:
        runtime_platform = _runtime_platform_for_wheel(wheel_path)
        previous = discovered.get(runtime_platform)
        if previous is not None:
            raise RuntimeError(
                "Found multiple wheels for the same supported Watchman runtime "
                f"platform {runtime_platform!r}: {previous} and {wheel_path}"
            )
        discovered[runtime_platform] = wheel_path

    missing = sorted(supported_platforms - set(discovered))
    if missing:
        rendered = ", ".join(missing)
        raise RuntimeError(
            "Missing required supported Watchman runtime wheel(s): "
            f"{rendered}. Supplied wheels: "
            f"{', '.join(str(path) for path in wheel_paths)}"
        )

    extra = sorted(set(discovered) - supported_platforms)
    if extra:
        rendered = ", ".join(extra)
        raise RuntimeError(
            "Supplied wheel set contains unsupported Watchman runtime platform(s): "
            f"{rendered}"
        )

    if len(discovered) != len(wheel_paths):
        raise RuntimeError(
            "Supplied wheel set contains duplicate or unexpected wheel artifacts."
        )

    return discovered


def _terminate_process_tree(pid: int) -> None:
    watchman_verifier_cleanup.terminate_process_tree(pid, psutil_module=psutil)


def _terminate_processes_using_root(root: Path) -> None:
    watchman_verifier_cleanup.terminate_processes_using_root(
        root,
        os_module=os,
        psutil_module=psutil,
        process_terminator=_terminate_process_tree,
    )


def _remove_tree_with_retries(
    root: Path, *, attempts: int = 5, base_delay_seconds: float = 0.2
) -> None:
    watchman_verifier_cleanup.remove_tree_with_retries(
        root,
        attempts=attempts,
        base_delay_seconds=base_delay_seconds,
        os_module=os,
        shutil_module=shutil,
        time_module=time,
        process_root_terminator=_terminate_processes_using_root,
    )


def _verify_wheel_has_platform_only_tag(wheel_path: Path) -> None:
    if "-py3-none-" not in wheel_path.name or wheel_path.name.endswith("any.whl"):
        raise RuntimeError(
            "Watchman runtime wheels must use an explicit py3-none-platform tag: "
            f"{wheel_path}"
        )


def _verify_wheel_contents(wheel_path: Path) -> str:
    with zipfile.ZipFile(wheel_path) as zf:
        names = set(zf.namelist())
    runtime_platforms = _runtime_platforms_in_wheel(wheel_path=wheel_path, names=names)
    required_wheel_paths = _required_wheel_paths_for_platforms(runtime_platforms)
    missing = [path for path in required_wheel_paths if path not in names]
    if missing:
        missing_rendered = "\n".join(f"- {item}" for item in missing)
        raise RuntimeError(
            "Wheel is missing required Watchman runtime resources: "
            f"{wheel_path}\n{missing_rendered}"
        )
    unexpected = sorted(
        path
        for path in names
        if path.startswith(_RUNTIME_SLOT_PREFIX)
        and not path.endswith("/")
        and path not in required_wheel_paths
    )
    if unexpected:
        unexpected_rendered = "\n".join(f"- {item}" for item in unexpected)
        raise RuntimeError(
            "Wheel contains unexpected Watchman runtime resources: "
            f"{wheel_path}\n{unexpected_rendered}"
        )
    return runtime_platforms[0]


def _should_verify_runtime_reads(runtime_platform: str) -> bool:
    return runtime_platform == hatch_build._host_watchman_platform()


def _verify_runtime_reads(*, wheel_path: Path, runtime_platform: str) -> None:
    system_name, machine_name = runtime_platform.split("-", maxsplit=1)
    root = Path(tempfile.mkdtemp(prefix="chunkhound-watchman-wheel-verify-"))
    try:
        with zipfile.ZipFile(wheel_path) as zf:
            zf.extractall(root)

        code = "\n".join(
            [
                "import os",
                "import json",
                "import psutil",
                "import subprocess",
                "import sys",
                "import time",
                "from pathlib import Path",
                "",
                f"SYSTEM_NAME = {system_name!r}",
                f"MACHINE_NAME = {machine_name!r}",
                "",
                (
                    "from chunkhound.watchman_runtime.loader import "
                    "build_watchman_client_command, "
                    "build_watchman_probe_command, "
                    "build_watchman_runtime_environment, "
                    "build_watchman_sidecar_command, "
                    "listener_path_is_filesystem, "
                    "materialize_watchman_binary, "
                    "resolve_packaged_watchman_runtime"
                ),
                "",
                (
                    "runtime = resolve_packaged_watchman_runtime("
                    "system_name=SYSTEM_NAME, machine_name=MACHINE_NAME)"
                ),
                (
                    "binary_path = materialize_watchman_binary("
                    "destination_root=Path('materialized'), "
                    "system_name=SYSTEM_NAME, machine_name=MACHINE_NAME)"
                ),
                "assert binary_path.is_file()",
                "if os.name != 'nt':",
                "    assert os.access(binary_path, os.X_OK)",
                (
                    "runtime_env = build_watchman_runtime_environment("
                    "runtime=runtime, binary_path=binary_path)"
                ),
                (
                    "result = subprocess.run("
                    "build_watchman_probe_command(runtime=runtime, "
                    "binary_path=binary_path), "
                    "check=True, capture_output=True, text=True, env=runtime_env)"
                ),
                "assert result.stdout.strip() == runtime.runtime_version",
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
                "def _run_one_shot(command):",
                "    result = subprocess.run(",
                "        build_watchman_client_command(",
                "            runtime=runtime,",
                "            binary_path=binary_path,",
                "            socket_path=socket_path,",
                "            statefile_path=statefile_path,",
                "            logfile_path=logfile_path,",
                "            pidfile_path=pidfile_path,",
                "            persistent=False,",
                "        ),",
                "        input=json.dumps(command) + '\\n',",
                "        capture_output=True,",
                "        check=False,",
                "        text=True,",
                "        env=runtime_env,",
                "    )",
                "    assert result.returncode == 0, result.stderr",
                "    payload = json.loads(result.stdout)",
                "    assert isinstance(payload, dict)",
                "    return payload",
                "",
                "def _wait_for_named_pipe_ready(timeout=15.0):",
                "    deadline = time.monotonic() + timeout",
                "    while time.monotonic() < deadline:",
                "        try:",
                "            _run_one_shot(['version'])",
                "            return",
                "        except AssertionError:",
                "            time.sleep(0.1)",
                (
                    "    raise AssertionError("
                    "'timed out waiting for named-pipe version readiness'"
                    ")"
                ),
                "",
                "sidecar_root = Path('sidecar').resolve()",
                "sidecar_root.mkdir(parents=True, exist_ok=True)",
                "socket_path = (",
                "    sidecar_root / 'sock'",
                "    if listener_path_is_filesystem(runtime)",
                "    else (",
                "        rf'\\\\.\\pipe\\chunkhound-watchman-wheel-verify-'",
                "        + str(os.getpid())",
                "    )",
                ")",
                "pidfile_path = sidecar_root / 'pid'",
                "statefile_path = sidecar_root / 'state'",
                "logfile_path = sidecar_root / 'watchman.log'",
                (
                    "sidecar_command = build_watchman_sidecar_command("
                    "runtime=runtime, binary_path=binary_path, "
                    "socket_path=socket_path, statefile_path=statefile_path, "
                    "logfile_path=logfile_path, pidfile_path=pidfile_path)"
                ),
                "sidecar = None",
                "client = None",
                "reader_thread = None",
                "try:",
                (
                    "    sidecar = subprocess.Popen("
                    "sidecar_command, stdin=subprocess.DEVNULL, "
                    "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, "
                    "env=runtime_env)"
                ),
                (
                    "    deadline = time.monotonic() + ("
                    "5.0 if listener_path_is_filesystem(runtime) else 15.0)"
                ),
                "    while time.monotonic() < deadline:",
                "        listener_ready = True",
                "        if listener_path_is_filesystem(runtime):",
                "            listener_ready = Path(socket_path).exists()",
                (
                    "        if (listener_ready and ("
                    "not listener_path_is_filesystem(runtime) or "
                    "(pidfile_path.exists() and logfile_path.exists()))):"
                ),
                "            try:",
                "                _run_one_shot(['version'])",
                "                break",
                "            except AssertionError:",
                "                pass",
                "        if sidecar.poll() is not None:",
                (
                    "            raise AssertionError("
                    "f'packaged runtime exited early: {sidecar.returncode}')"
                ),
                "        time.sleep(0.05)",
                "    if listener_path_is_filesystem(runtime):",
                "        assert Path(socket_path).exists()",
                "        assert pidfile_path.exists()",
                "        assert logfile_path.exists()",
                "    else:",
                "        _wait_for_named_pipe_ready()",
                "    assert sidecar.poll() is None",
                "    cmdline = psutil.Process(sidecar.pid).cmdline()",
                "    assert cmdline and cmdline[0] == str(binary_path)",
                (
                    "    assert 'chunkhound.watchman_runtime.bridge' not in "
                    "' '.join(cmdline)"
                ),
                (
                    "    version_response = _run_one_shot("
                    "['version', {'required': ['cmd-watch-project', 'relative_root']}]"
                    ")"
                ),
                "    capabilities = version_response.get('capabilities')",
                "    assert isinstance(capabilities, dict)",
                "    assert capabilities.get('cmd-watch-project') is True",
                "    assert capabilities.get('relative_root') is True",
                (
                    "    project_root = Path('project').resolve(); "
                    "project_root.mkdir(exist_ok=True)"
                ),
                (
                    "    watch_project = _run_one_shot("
                    "['watch-project', str(project_root.resolve())])"
                ),
                (
                    "    assert Path(str(watch_project['watch'])).resolve() == "
                    "project_root.resolve()"
                ),
                (
                    "    client_command = build_watchman_client_command("
                    "runtime=runtime, binary_path=binary_path, "
                    "socket_path=socket_path, statefile_path=statefile_path, "
                    "logfile_path=logfile_path, pidfile_path=pidfile_path)"
                ),
                (
                    "    client = subprocess.Popen("
                    "client_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, "
                    "stderr=subprocess.PIPE, text=True, env=runtime_env)"
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
    parser.add_argument(
        "--require-supported-matrix",
        action="store_true",
        help=(
            "Require the supplied wheel set to contain exactly one wheel for "
            "each supported packaged Watchman runtime platform."
        ),
    )
    args = parser.parse_args(argv)

    wheel_paths: list[Path] = []
    for raw in args.wheels:
        if raw.is_file() and raw.suffix == ".whl":
            wheel_paths.append(raw)
            continue
        raise FileNotFoundError(f"Wheel not found: {raw}")

    if args.require_supported_matrix:
        _verify_supported_wheel_matrix(wheel_paths)

    for wheel_path in wheel_paths:
        _verify_wheel_has_platform_only_tag(wheel_path)
        runtime_platform = _verify_wheel_contents(wheel_path)
        if _should_verify_runtime_reads(runtime_platform):
            _verify_runtime_reads(
                wheel_path=wheel_path,
                runtime_platform=runtime_platform,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
