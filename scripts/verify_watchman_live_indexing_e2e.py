from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import psutil

_MCP_INIT_PARAMS = {
    "protocolVersion": "2024-11-05",
    "clientInfo": {"name": "watchman-wheel-e2e", "version": "0.0.1"},
    "capabilities": {},
}
_READY_TIMEOUT_SECONDS = 60.0
_SEARCH_TIMEOUT_SECONDS = 30.0


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


def _assert_sidecar_uses_installed_runtime(
    realtime: dict[str, Any], *, venv_dir: Path
) -> None:
    watchman_pid = realtime.get("watchman_pid")
    if not isinstance(watchman_pid, int) or watchman_pid <= 0:
        raise RuntimeError(f"Invalid Watchman sidecar pid in daemon status: {realtime}")

    process = _resolve_bridge_process(watchman_pid)

    expected_bin_dir = _python_path(venv_dir).parent
    process_env = process.environ()
    if process_env.get("VIRTUAL_ENV") != str(venv_dir):
        raise RuntimeError(
            "Watchman sidecar inherited the wrong virtualenv environment: "
            f"{process_env.get('VIRTUAL_ENV')!r}"
        )
    path_entries = process_env.get("PATH", "").split(os.pathsep)
    if not path_entries or path_entries[0] != str(expected_bin_dir):
        raise RuntimeError(
            "Watchman sidecar PATH was not pinned to the installed-wheel "
            f"virtualenv: {process_env.get('PATH')!r}"
        )


def _resolve_bridge_process(watchman_pid: int) -> psutil.Process:
    process = psutil.Process(watchman_pid)
    cmdline = process.cmdline()
    if len(cmdline) >= 3 and cmdline[1:3] == [
        "-m",
        "chunkhound.watchman_runtime.bridge",
    ]:
        return process

    if os.name == "nt" and len(cmdline) >= 3 and cmdline[0].lower().endswith("cmd.exe"):
        for child in process.children(recursive=True):
            try:
                child_cmdline = child.cmdline()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            if len(child_cmdline) >= 3 and child_cmdline[1:3] == [
                "-m",
                "chunkhound.watchman_runtime.bridge",
            ]:
                return child

    raise RuntimeError(
        "Watchman sidecar did not launch the packaged runtime bridge: "
        f"{cmdline}"
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
            "mcp",
            str(project_dir),
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
            "Install built wheel(s) into clean temporary environments and prove "
            "that a Watchman-backed live mutation becomes searchable."
        )
    )
    parser.add_argument(
        "wheels",
        nargs="+",
        type=Path,
        help="Path(s) to .whl file(s) to verify.",
    )
    args = parser.parse_args(argv)

    for wheel_path in args.wheels:
        if not wheel_path.is_file() or wheel_path.suffix != ".whl":
            raise FileNotFoundError(f"Wheel not found: {wheel_path}")

    for wheel_path in args.wheels:
        asyncio.run(_verify_wheel(wheel_path))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
