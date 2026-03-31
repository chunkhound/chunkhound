from __future__ import annotations

import argparse
import asyncio
import json
import ntpath
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import psutil

import hatch_build
from chunkhound.daemon.discovery import DaemonDiscovery
from scripts import verify_watchman_runtime_resources as runtime_verifier
from scripts import watchman_verifier_cleanup

_MCP_INIT_PARAMS = {
    "protocolVersion": "2024-11-05",
    "clientInfo": {"name": "watchman-wheel-e2e", "version": "0.0.1"},
    "capabilities": {},
}
_READY_TIMEOUT_SECONDS = 60.0
_SEARCH_TIMEOUT_SECONDS = 30.0
_SOURCE_FALLBACK_FAILURE_TIMEOUT_SECONDS = 20.0
_DETERMINISTIC_FAILURE_URL_BASE = "https://127.0.0.1:9/chunkhound-watchman-fail"
_SOURCE_FALLBACK_PRETEND_VERSION = "0.0.0"


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


class SubprocessJsonRpcClient:
    def __init__(self, process: asyncio.subprocess.Process) -> None:
        if process.stdin is None or process.stdout is None:
            raise ValueError("Process must expose stdin/stdout pipes")
        self._process = process
        self._reader_task: asyncio.Task[None] | None = None
        self._pending_requests: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._next_request_id = 1
        self._request_lock = asyncio.Lock()
        self._closed = False

    async def start(self) -> None:
        if self._reader_task is not None:
            raise RuntimeError("JSON-RPC client already started")
        self._reader_task = asyncio.create_task(self._read_responses())

    async def send_request(
        self, method: str, params: dict[str, Any] | None = None, timeout: float = 5.0
    ) -> dict[str, Any]:
        if self._reader_task is None:
            raise RuntimeError("JSON-RPC client not started")
        if self._closed:
            raise RuntimeError("JSON-RPC client already closed")

        async with self._request_lock:
            request_id = self._next_request_id
            self._next_request_id += 1

        future: asyncio.Future[dict[str, Any]] = asyncio.Future()
        self._pending_requests[request_id] = future
        request: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params is not None:
            request["params"] = params

        assert self._process.stdin is not None
        self._process.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
        try:
            await self._process.stdin.drain()
            response = await asyncio.wait_for(future, timeout=timeout)
        except Exception:
            self._pending_requests.pop(request_id, None)
            raise
        finally:
            self._pending_requests.pop(request_id, None)

        if "error" in response:
            raise RuntimeError(f"JSON-RPC error response: {response['error']}")
        result = response.get("result")
        if not isinstance(result, dict):
            raise RuntimeError(
                f"JSON-RPC result payload missing or invalid: {response}"
            )
        return result

    async def send_notification(
        self, method: str, params: dict[str, Any] | None = None
    ) -> None:
        if self._reader_task is None:
            raise RuntimeError("JSON-RPC client not started")
        if self._closed:
            raise RuntimeError("JSON-RPC client already closed")

        notification: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            notification["params"] = params

        assert self._process.stdin is not None
        self._process.stdin.write((json.dumps(notification) + "\n").encode("utf-8"))
        await self._process.stdin.drain()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()

        if self._process.stdin is not None:
            try:
                self._process.stdin.close()
                await self._process.stdin.wait_closed()
            except Exception:
                pass

        if self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()

        if self._reader_task is not None:
            try:
                await asyncio.wait_for(self._reader_task, timeout=2.0)
            except Exception:
                self._reader_task.cancel()
                try:
                    await self._reader_task
                except Exception:
                    pass
            self._reader_task = None

    async def _read_responses(self) -> None:
        assert self._process.stdout is not None
        try:
            while True:
                raw_line = await self._process.stdout.readline()
                if not raw_line:
                    self._fail_pending(
                        RuntimeError(
                            "JSON-RPC subprocess terminated unexpectedly "
                            f"(rc={self._process.returncode})"
                        )
                    )
                    return

                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    continue
                request_id = payload.get("id")
                if not isinstance(request_id, int):
                    continue
                future = self._pending_requests.get(request_id)
                if future is not None and not future.done():
                    future.set_result(payload)
        except asyncio.CancelledError:
            pass
        except Exception as error:
            self._fail_pending(error)

    def _fail_pending(self, error: Exception) -> None:
        for future in self._pending_requests.values():
            if not future.done():
                future.set_exception(error)
        self._pending_requests.clear()


def _python_path(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _chunkhound_path(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "chunkhound.exe"
    return venv_dir / "bin" / "chunkhound"


def _mcp_command_args(project_dir: Path) -> tuple[str, ...]:
    return ("mcp", "--no-embeddings", str(project_dir))


def _source_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _lock_path_for_runtime(project_dir: Path, runtime_dir: Path) -> Path:
    previous = os.environ.get("CHUNKHOUND_DAEMON_RUNTIME_DIR")
    os.environ["CHUNKHOUND_DAEMON_RUNTIME_DIR"] = str(runtime_dir)
    try:
        return DaemonDiscovery(project_dir).get_lock_path()
    finally:
        if previous is None:
            os.environ.pop("CHUNKHOUND_DAEMON_RUNTIME_DIR", None)
        else:
            os.environ["CHUNKHOUND_DAEMON_RUNTIME_DIR"] = previous


def _host_compatible_wheels(wheel_paths: list[Path]) -> list[Path]:
    compatible: list[Path] = []
    for wheel_path in wheel_paths:
        runtime_platform = runtime_verifier._runtime_platform_for_wheel(wheel_path)
        if runtime_verifier._should_verify_runtime_reads(runtime_platform):
            compatible.append(wheel_path)
    return compatible


def _utf8_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base_env or os.environ)
    if os.name == "nt":
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")
    return env


def _filtered_path_entries(
    original_path: str, *, blocked_roots: list[Path], preferred_entry: str
) -> str:
    filtered: list[str] = [preferred_entry]
    seen = {preferred_entry}
    for raw_entry in original_path.split(os.pathsep):
        if not raw_entry or raw_entry in seen:
            continue
        try:
            resolved = Path(raw_entry).resolve(strict=False)
        except OSError:
            resolved = Path(raw_entry)
        if any(resolved.is_relative_to(root) for root in blocked_roots):
            continue
        filtered.append(raw_entry)
        seen.add(raw_entry)
    return os.pathsep.join(filtered)


async def _create_subprocess_exec_safe(
    *args: str,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        *args,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        env=_utf8_env(env),
        cwd=cwd,
    )


def _write_project(project_dir: Path) -> tuple[Path, str]:
    db_path = (project_dir / ".chunkhound" / "test.db").resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "database": {"path": str(db_path), "provider": "duckdb"},
        "indexing": {
            "include": ["*.py"],
            "realtime_backend": "watchman",
        },
    }
    (project_dir / ".chunkhound.json").write_text(json.dumps(config), encoding="utf-8")
    (project_dir / "seed.py").write_text(
        "def seed_symbol_for_watchman_runtime():\n    return 1\n",
        encoding="utf-8",
    )

    live_symbol = f"watchman_live_installed_{int(time.time() * 1000)}"
    return project_dir / "src" / "installed_watchman_live.py", live_symbol


def _mcp_env(venv_dir: Path) -> dict[str, str]:
    env = _utf8_env()
    original_virtual_env = env.get("VIRTUAL_ENV")
    for key in list(env.keys()):
        if key.startswith("CHUNKHOUND_"):
            del env[key]
    for key in ("PYTHONHOME", "PYTHONPATH", "__PYVENV_LAUNCHER__"):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["VIRTUAL_ENV"] = str(venv_dir)
    venv_bin = str(_python_path(venv_dir).parent)
    blocked_roots: list[Path] = []
    for candidate in (original_virtual_env, sys.prefix):
        if not candidate:
            continue
        try:
            root = Path(candidate).resolve(strict=False)
        except OSError:
            continue
        if root == venv_dir.resolve(strict=False):
            continue
        blocked_roots.append(root)
    env["PATH"] = _filtered_path_entries(
        env.get("PATH", ""),
        blocked_roots=blocked_roots,
        preferred_entry=venv_bin,
    )
    env["CHUNKHOUND_MCP_MODE"] = "1"
    return env


def _clean_room_env(
    venv_dir: Path,
    *,
    runtime_dir: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict[str, str]:
    env = _mcp_env(venv_dir)
    if runtime_dir is not None:
        env["CHUNKHOUND_DAEMON_RUNTIME_DIR"] = str(runtime_dir)
    if extra_env:
        env.update(extra_env)
    return env


def _editable_install_env() -> dict[str, str]:
    env = _utf8_env()
    env["SETUPTOOLS_SCM_PRETEND_VERSION"] = _SOURCE_FALLBACK_PRETEND_VERSION
    env["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CHUNKHOUND"] = (
        _SOURCE_FALLBACK_PRETEND_VERSION
    )
    return env


def _assert_sidecar_uses_installed_runtime(
    realtime: dict[str, Any], *, venv_dir: Path
) -> None:
    watchman_pid = realtime.get("watchman_pid")
    if not isinstance(watchman_pid, int) or watchman_pid <= 0:
        raise RuntimeError(f"Invalid Watchman sidecar pid in daemon status: {realtime}")

    process = _resolve_native_watchman_process(
        watchman_pid,
        expected_binary_path=realtime.get("watchman_binary_path"),
    )

    expected_bin_dir = _python_path(venv_dir).parent
    process_env = process.environ()
    if process_env.get("VIRTUAL_ENV") != str(venv_dir):
        raise RuntimeError(
            "Watchman sidecar inherited the wrong virtualenv environment: "
            f"{process_env.get('VIRTUAL_ENV')!r}"
        )
    path_entries = process_env.get("PATH", "").split(os.pathsep)
    normalized_entries = [
        os.path.normcase(os.path.normpath(entry))
        for entry in path_entries
        if entry
    ]
    expected_venv_entry = os.path.normcase(os.path.normpath(str(expected_bin_dir)))
    expected_runtime_dir = realtime.get("watchman_binary_path")
    runtime_entry: str | None = None
    if isinstance(expected_runtime_dir, str) and expected_runtime_dir:
        runtime_entry = os.path.normcase(
            os.path.normpath(ntpath.dirname(expected_runtime_dir))
        )

    if not normalized_entries:
        raise RuntimeError(
            "Watchman sidecar PATH was not pinned to the installed-wheel "
            f"virtualenv: {process_env.get('PATH')!r}"
        )
    if runtime_entry is not None and normalized_entries[0] == runtime_entry:
        if len(normalized_entries) < 2 or normalized_entries[1] != expected_venv_entry:
            raise RuntimeError(
                "Watchman sidecar PATH did not keep the installed-wheel virtualenv "
                "immediately after the materialized runtime directory: "
                f"{process_env.get('PATH')!r}"
            )
        return
    if normalized_entries[0] != expected_venv_entry:
        raise RuntimeError(
            "Watchman sidecar PATH was not pinned to the installed-wheel "
            f"virtualenv: {process_env.get('PATH')!r}"
        )


def _resolve_native_watchman_process(
    watchman_pid: int, *, expected_binary_path: object
) -> psutil.Process:
    process = psutil.Process(watchman_pid)
    cmdline = process.cmdline()
    if not isinstance(expected_binary_path, str) or not expected_binary_path:
        raise RuntimeError(
            "Watchman daemon status did not report a native binary path: "
            f"{expected_binary_path!r}"
        )
    if any(arg == "chunkhound.watchman_runtime.bridge" for arg in cmdline):
        raise RuntimeError(
            "Watchman sidecar fell back to the packaged runtime bridge instead of "
            f"the native daemon: {cmdline}"
        )
    normalized_expected = os.path.normcase(os.path.normpath(expected_binary_path))
    if cmdline and (
        os.path.normcase(os.path.normpath(cmdline[0])) == normalized_expected
    ):
        return process

    raise RuntimeError(
        "Watchman sidecar did not launch the expected native daemon binary: "
        f"expected={expected_binary_path!r} actual={cmdline}"
    )


def _parse_tool_json(result: dict[str, Any]) -> dict[str, Any]:
    content = result.get("content", [])
    if not isinstance(content, list) or not content:
        raise RuntimeError(f"Unexpected MCP tool result: {result}")
    text = content[0].get("text")
    if not isinstance(text, str):
        raise RuntimeError(f"Missing text payload in MCP tool result: {result}")
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Expected JSON object tool payload, got: {parsed!r}")
    return parsed


def _flatten_tool_text(result: dict[str, Any]) -> str:
    content = result.get("content", [])
    if not isinstance(content, list):
        return ""
    rendered: list[str] = []
    for item in content:
        if isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str):
                rendered.append(text)
    return "\n".join(rendered)


def _has_nested_watchman_health(realtime: dict[str, Any]) -> bool:
    subscription_names = realtime.get("watchman_subscription_names")
    watchman_scopes = realtime.get("watchman_scopes")
    loss_of_sync = realtime.get("watchman_loss_of_sync")
    return (
        isinstance(subscription_names, list)
        and len(subscription_names) >= 1
        and isinstance(watchman_scopes, list)
        and len(watchman_scopes) >= 1
        and isinstance(loss_of_sync, dict)
    )


async def _wait_for_ready(client: SubprocessJsonRpcClient) -> dict[str, Any]:
    deadline = time.monotonic() + _READY_TIMEOUT_SECONDS
    last_status: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        result = await client.send_request(
            "tools/call",
            {"name": "daemon_status", "arguments": {}},
            timeout=15.0,
        )
        last_status = _parse_tool_json(result)
        realtime = last_status.get("scan_progress", {}).get("realtime", {})
        if (
            last_status.get("status") == "ready"
            and realtime.get("watchman_connection_state") == "connected"
            and realtime.get("watchman_subscription_count") == 1
            and _has_nested_watchman_health(realtime)
        ):
            return last_status
        await asyncio.sleep(0.5)

    raise RuntimeError(f"Timed out waiting for ready Watchman daemon: {last_status}")


async def _wait_for_search_hit(
    client: SubprocessJsonRpcClient, *, query: str
) -> dict[str, Any]:
    deadline = time.monotonic() + _SEARCH_TIMEOUT_SECONDS
    last_result: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        last_result = await client.send_request(
            "tools/call",
            {"name": "search", "arguments": {"query": query, "type": "regex"}},
            timeout=15.0,
        )
        if query in _flatten_tool_text(last_result):
            return last_result
        await asyncio.sleep(0.5)

    raise RuntimeError(
        "Timed out waiting for live mutation to become searchable. "
        f"Last result: {_flatten_tool_text(last_result or {})}"
    )


async def _wait_for_daemon_status(
    client: SubprocessJsonRpcClient,
    *,
    predicate,
    description: str,
    timeout: float = _READY_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_status: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        result = await client.send_request(
            "tools/call",
            {"name": "daemon_status", "arguments": {}},
            timeout=15.0,
        )
        last_status = _parse_tool_json(result)
        if predicate(last_status):
            return last_status
        await asyncio.sleep(0.5)

    raise RuntimeError(
        f"Timed out waiting for daemon status {description}: {last_status}"
    )


def _source_tree_copy_ignore(_root: str, names: list[str]) -> set[str]:
    ignored = {
        ".cache",
        ".chunkhound",
        ".git",
        ".venv",
        ".ruff_cache",
        ".uv-cache",
        ".uv_cache",
        ".uvcache",
        ".pytest_cache",
        ".mypy_cache",
        "__pycache__",
        "dist",
        "build",
    }
    return {name for name in names if name in ignored}


def _rewrite_watchman_source_urls(source_root: Path) -> None:
    manifests = [
        source_root
        / "chunkhound"
        / "watchman_runtime"
        / "platforms"
        / platform_tag
        / "manifest.json"
        for platform_tag in sorted(hatch_build._load_supported_watchman_platforms())
    ]
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        platform_tag = manifest.get("platform", "unknown")
        if isinstance(manifest.get("source_url"), str):
            manifest["source_url"] = (
                f"{_DETERMINISTIC_FAILURE_URL_BASE}/{platform_tag}/runtime"
            )
        support_sources = manifest.get("support_file_sources")
        if isinstance(support_sources, dict):
            for index, descriptor in enumerate(support_sources.values()):
                if isinstance(descriptor, dict) and isinstance(
                    descriptor.get("source_url"), str
                ):
                    descriptor["source_url"] = (
                        f"{_DETERMINISTIC_FAILURE_URL_BASE}/"
                        f"{platform_tag}/support-{index}"
                    )
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )


def _write_project_config(project_dir: Path, *, realtime_backend: str | None) -> None:
    db_path = (project_dir / ".chunkhound" / "test.db").resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    indexing: dict[str, object] = {"include": ["*.py"]}
    if realtime_backend is not None:
        indexing["realtime_backend"] = realtime_backend
    config = {
        "database": {"path": str(db_path), "provider": "duckdb"},
        "indexing": indexing,
    }
    (project_dir / ".chunkhound.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    (project_dir / "seed.py").write_text(
        "def seed_symbol_for_source_fallback():\n    return 1\n",
        encoding="utf-8",
    )


async def _start_proxy_client(
    *,
    chunkhound_exe: Path,
    project_dir: Path,
    env: dict[str, str],
) -> tuple[asyncio.subprocess.Process, SubprocessJsonRpcClient]:
    proc = await _create_subprocess_exec_safe(
        str(chunkhound_exe),
        *_mcp_command_args(project_dir),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        cwd=str(project_dir),
    )
    client = SubprocessJsonRpcClient(proc)
    await client.start()
    await client.send_request("initialize", _MCP_INIT_PARAMS, timeout=30.0)
    await client.send_notification("notifications/initialized", {})
    return proc, client


async def _run_proxy_to_failure(
    *,
    chunkhound_exe: Path,
    project_dir: Path,
    env: dict[str, str],
    timeout: float = _SOURCE_FALLBACK_FAILURE_TIMEOUT_SECONDS,
) -> tuple[int, str, str]:
    proc = await _create_subprocess_exec_safe(
        str(chunkhound_exe),
        *_mcp_command_args(project_dir),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        cwd=str(project_dir),
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    return proc.returncode or 0, stdout.decode(), stderr.decode()


async def _verify_source_fallback(source_root: Path) -> None:
    root = Path(tempfile.mkdtemp(prefix="chunkhound-watchman-source-verify-"))
    try:
        editable_source_root = root / "editable-source"
        shutil.copytree(
            source_root,
            editable_source_root,
            ignore=_source_tree_copy_ignore,
        )
        _rewrite_watchman_source_urls(editable_source_root)

        venv_dir = root / "venv"
        runtime_dir = root / "runtime"
        cache_dir = root / "cache"
        default_project_dir = root / "project-default"
        watchman_project_dir = root / "project-watchman"
        default_project_dir.mkdir(parents=True, exist_ok=True)
        watchman_project_dir.mkdir(parents=True, exist_ok=True)
        _write_project_config(default_project_dir, realtime_backend=None)
        _write_project_config(watchman_project_dir, realtime_backend="watchman")

        subprocess.run(
            ["uv", "venv", str(venv_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        python_path = _python_path(venv_dir)
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python_path),
                "-e",
                str(editable_source_root),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=_editable_install_env(),
        )
        chunkhound_exe = _chunkhound_path(venv_dir)
        if not chunkhound_exe.is_file():
            raise FileNotFoundError(
                f"Installed chunkhound executable not found: {chunkhound_exe}"
            )

        default_env = _clean_room_env(
            venv_dir,
            runtime_dir=runtime_dir / "default",
            extra_env={
                "CHUNKHOUND_WATCHMAN_RUNTIME_CACHE_DIR": str(cache_dir / "default"),
            },
        )
        proc, client = await _start_proxy_client(
            chunkhound_exe=chunkhound_exe,
            project_dir=default_project_dir,
            env=default_env,
        )
        default_stderr = ""
        try:
            status = await _wait_for_daemon_status(
                client,
                predicate=lambda payload: payload.get("status") == "ready"
                and payload.get("scan_progress", {})
                .get("realtime", {})
                .get("configured_backend")
                == "watchdog"
                and payload.get("scan_progress", {})
                .get("realtime", {})
                .get("effective_backend")
                == "watchdog",
                description="watchdog fallback readiness",
            )
            realtime = status["scan_progress"]["realtime"]
            if realtime.get("configured_backend") != "watchdog":
                raise RuntimeError(
                    f"Source fallback configured backend was not watchdog: {realtime}"
                )
            if realtime.get("effective_backend") != "watchdog":
                raise RuntimeError(
                    f"Source fallback effective backend was not watchdog: {realtime}"
                )
        finally:
            await client.close()
            if proc.stderr is not None:
                default_stderr = (await proc.stderr.read()).decode(
                    "utf-8",
                    errors="replace",
                )
        if default_stderr.strip():
            print(default_stderr, file=os.sys.stderr)

        failure_runtime_dir = runtime_dir / "watchman"
        failure_env = _clean_room_env(
            venv_dir,
            runtime_dir=failure_runtime_dir,
            extra_env={
                "CHUNKHOUND_WATCHMAN_RUNTIME_CACHE_DIR": str(cache_dir / "watchman"),
                "CHUNKHOUND_WATCHMAN_RUNTIME_DOWNLOAD_TIMEOUT_SECONDS": "1",
                "CHUNKHOUND_WATCHMAN_RUNTIME_DOWNLOAD_RETRIES": "1",
            },
        )
        returncode, _stdout, stderr_text = await _run_proxy_to_failure(
            chunkhound_exe=chunkhound_exe,
            project_dir=watchman_project_dir,
            env=failure_env,
        )
        if returncode == 0:
            raise RuntimeError(
                "Explicit source-install watchman startup unexpectedly succeeded"
            )
        if "Recent daemon log output" not in stderr_text:
            raise RuntimeError(
                "Explicit source-install watchman failure did not include "
                f"daemon log context: {stderr_text}"
            )

        daemon_log_path = watchman_project_dir / ".chunkhound" / "daemon.log"
        if not daemon_log_path.exists():
            raise RuntimeError(
                "Explicit source-install watchman failure did not write daemon.log"
            )
        daemon_log_text = daemon_log_path.read_text(
            encoding="utf-8",
            errors="replace",
        )
        if (
            "Watchman runtime archive download failed" not in daemon_log_text
            and "Watchman" not in daemon_log_text
        ):
            raise RuntimeError(
                "Explicit source-install watchman failure did not surface "
                f"Watchman startup evidence: {daemon_log_text}"
            )

        lock_path = _lock_path_for_runtime(watchman_project_dir, failure_runtime_dir)
        if lock_path.exists():
            raise RuntimeError(
                "Explicit source-install watchman failure published a daemon lock"
            )
    finally:
        _terminate_processes_using_root(root)
        _remove_tree_with_retries(root)


async def _verify_wheel(wheel_path: Path) -> None:
    root = Path(tempfile.mkdtemp(prefix="chunkhound-watchman-live-wheel-verify-"))
    try:
        venv_dir = root / "venv"
        project_dir = root / "project"
        project_dir.mkdir(parents=True, exist_ok=True)
        live_file, live_symbol = _write_project(project_dir)

        subprocess.run(
            ["uv", "venv", str(venv_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        python_path = _python_path(venv_dir)
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python_path), str(wheel_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        chunkhound_exe = _chunkhound_path(venv_dir)
        if not chunkhound_exe.is_file():
            raise FileNotFoundError(
                f"Installed chunkhound executable not found: {chunkhound_exe}"
            )

        proc = await _create_subprocess_exec_safe(
            str(chunkhound_exe),
            *_mcp_command_args(project_dir),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_mcp_env(venv_dir),
            cwd=str(project_dir),
        )
        client = SubprocessJsonRpcClient(proc)
        stderr_text = ""
        await client.start()
        try:
            await client.send_request("initialize", _MCP_INIT_PARAMS, timeout=30.0)
            await client.send_notification("notifications/initialized", {})

            ready_status = await _wait_for_ready(client)
            realtime = ready_status["scan_progress"]["realtime"]
            if realtime.get("watchman_sidecar_state") != "running":
                raise RuntimeError(f"Unexpected Watchman sidecar state: {realtime}")
            if not _has_nested_watchman_health(realtime):
                raise RuntimeError(
                    "Ready Watchman daemon_status payload was missing nested health "
                    f"structure: {realtime}"
                )
            _assert_sidecar_uses_installed_runtime(realtime, venv_dir=venv_dir)

            live_file.parent.mkdir(parents=True, exist_ok=True)
            live_file.write_text(
                f"def {live_symbol}():\n    return 'live'\n",
                encoding="utf-8",
            )

            await _wait_for_search_hit(client, query=live_symbol)

            final_status = _parse_tool_json(
                await client.send_request(
                    "tools/call",
                    {"name": "daemon_status", "arguments": {}},
                    timeout=15.0,
                )
            )
            final_realtime = final_status["scan_progress"]["realtime"]
            if not _has_nested_watchman_health(final_realtime):
                raise RuntimeError(
                    "Final Watchman daemon_status payload was missing nested health "
                    f"structure: {final_realtime}"
                )
            if int(final_realtime.get("watchman_subscription_pdu_count", 0)) < 1:
                raise RuntimeError(
                    "Live mutation became searchable, but no subscription PDUs were "
                    f"recorded: {final_realtime}"
                )
        finally:
            await client.close()
            if proc.stderr is not None:
                stderr_text = (await proc.stderr.read()).decode(
                    "utf-8", errors="replace"
                )
        if stderr_text.strip():
            print(stderr_text, file=os.sys.stderr)
    finally:
        _terminate_processes_using_root(root)
        _remove_tree_with_retries(root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify installed-wheel Watchman live indexing and the documented "
            "source/editable fallback contract."
        )
    )
    parser.add_argument(
        "wheels",
        nargs="*",
        type=Path,
        help="Path(s) to .whl file(s) to verify for installed-wheel live indexing.",
    )
    parser.add_argument(
        "--require-supported-matrix",
        action="store_true",
        help=(
            "Require the supplied wheel set to contain exactly one wheel for "
            "each supported packaged Watchman runtime platform."
        ),
    )
    parser.add_argument(
        "--verify-source-fallback",
        action="store_true",
        help=(
            "Verify that a source/editable install defaults to watchdog and "
            "fails fast when watchman is explicitly requested."
        ),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=_source_root(),
        help=(
            "Source tree root to use for the source/editable fallback proof "
            "(defaults to this repository root)."
        ),
    )
    args = parser.parse_args(argv)

    if not args.wheels and not args.verify_source_fallback:
        parser.error("Provide wheel paths and/or --verify-source-fallback.")

    wheel_paths = list(args.wheels)
    for wheel_path in wheel_paths:
        if not wheel_path.is_file() or wheel_path.suffix != ".whl":
            raise FileNotFoundError(f"Wheel not found: {wheel_path}")

    if args.require_supported_matrix:
        if not wheel_paths:
            parser.error("--require-supported-matrix requires at least one wheel.")
        runtime_verifier._verify_supported_wheel_matrix(wheel_paths)

    for wheel_path in _host_compatible_wheels(wheel_paths):
        asyncio.run(_verify_wheel(wheel_path))

    if args.verify_source_fallback:
        asyncio.run(_verify_source_fallback(args.source_root))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
