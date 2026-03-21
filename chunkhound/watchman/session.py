from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from ..watchman_runtime.loader import (
    build_watchman_client_command,
    build_watchman_runtime_command_prefix,
    build_watchman_runtime_environment,
    resolve_packaged_watchman_runtime,
)
from .scope import (
    WatchmanScopePlan,
    WatchmanSubscriptionScope,
    build_watchman_scope_plan,
    discover_nested_linux_mount_roots,
    discover_nested_windows_junction_scopes,
)

_REQUIRED_CAPABILITIES: tuple[str, ...] = ("cmd-watch-project", "relative_root")
_DEFAULT_SUBSCRIPTION_NAME = "chunkhound-live-indexing"


def build_watchman_base_command(
    *,
    binary_path: Path,
    socket_path: str | Path,
    statefile_path: Path,
    logfile_path: Path,
    pidfile_path: Path,
) -> list[str]:
    """Build the client command for a packaged Watchman executable."""

    runtime = resolve_packaged_watchman_runtime()
    return build_watchman_client_command(
        runtime=runtime,
        binary_path=binary_path,
        socket_path=socket_path,
        statefile_path=statefile_path,
        logfile_path=logfile_path,
        pidfile_path=pidfile_path,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class WatchmanSessionSetup:
    scope_plan: WatchmanScopePlan
    subscription_name: str
    subscription_names: tuple[str, ...]
    capabilities: dict[str, bool]


class WatchmanCliSession:
    """Persistent JSON CLI bridge to a private Watchman sidecar."""

    _COMMAND_TIMEOUT_SECONDS = 5.0
    _PROCESS_EXIT_TIMEOUT_SECONDS = 5.0
    _SUBSCRIPTION_NAME_MAX_LENGTH = 128
    _SUBSCRIPTION_NAME_HASH_LENGTH = 12
    _SUBSCRIPTION_QUEUE_MAXSIZE = 1000

    def __init__(
        self,
        *,
        binary_path: Path,
        socket_path: str | Path,
        statefile_path: Path,
        logfile_path: Path,
        pidfile_path: Path,
        project_root: Path,
        debug_sink: Callable[[str], None] | None = None,
        command_prefix: Sequence[str] | None = None,
        command_env: dict[str, str] | None = None,
        subscription_overflow_handler: (
            Callable[[dict[str, object], int, int], None] | None
        ) = None,
    ) -> None:
        self._binary_path = binary_path
        self._socket_path = str(socket_path)
        self._statefile_path = statefile_path
        self._logfile_path = logfile_path
        self._pidfile_path = pidfile_path
        self._project_root = project_root
        self._debug_sink = debug_sink
        self._command_prefix = tuple(command_prefix) if command_prefix else None
        self._command_env = dict(command_env) if command_env else None
        self._subscription_overflow_handler = subscription_overflow_handler
        self.subscription_queue: asyncio.Queue[dict[str, object]] = asyncio.Queue(
            maxsize=self._SUBSCRIPTION_QUEUE_MAXSIZE
        )
        self._stream_reader: asyncio.StreamReader | None = None
        self._stream_writer: asyncio.StreamWriter | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._process_wait_task: asyncio.Task[None] | None = None
        self._pending_reply: asyncio.Future[dict[str, object]] | None = None
        self._command_lock = asyncio.Lock()
        self._scope_plan: WatchmanScopePlan | None = None
        self._subscription_name: str | None = None
        self._subscription_names: tuple[str, ...] = ()
        self._subscription_scopes: dict[str, WatchmanSubscriptionScope] = {}
        self._capabilities: dict[str, bool] = {}
        self._last_warning: str | None = None
        self._last_warning_at: str | None = None
        self._last_error: str | None = None
        self._last_error_at: str | None = None
        self._last_response_at: str | None = None
        self._last_subscription_at: str | None = None
        self._command_count = 0
        self._subscription_pdu_count = 0
        self._subscription_pdu_dropped = 0
        self._stop_requested = False
        self._unexpected_exit_future: asyncio.Future[str | None] | None = None

    @property
    def scope_plan(self) -> WatchmanScopePlan | None:
        return self._scope_plan

    @property
    def subscription_name(self) -> str | None:
        return self._subscription_name

    @property
    def subscription_names(self) -> tuple[str, ...]:
        return self._subscription_names

    def scope_for_subscription(
        self, subscription_name: str | None
    ) -> WatchmanSubscriptionScope | None:
        if not isinstance(subscription_name, str) or not subscription_name:
            return None
        return self._subscription_scopes.get(subscription_name)

    def supports_prepared_session_startup(self) -> bool:
        if self._command_prefix is not None:
            return False
        runtime = resolve_packaged_watchman_runtime()
        return runtime.listener_transport == "unix_socket"

    async def prepare(self) -> dict[str, bool]:
        if self._process is not None and self._process.returncode is None:
            await self.stop()

        self._reset_state()
        self._unexpected_exit_future = asyncio.get_running_loop().create_future()

        try:
            if self._use_direct_socket_session():
                self._debug(
                    "starting direct Watchman socket session "
                    f"socket={self._socket_path}"
                )
                reader, writer = await asyncio.open_unix_connection(
                    path=self._socket_path
                )
                self._stream_reader = reader
                self._stream_writer = writer
                self._reader_task = asyncio.create_task(self._reader_loop())
                version_response = await self._send_command(
                    ["version", {"required": list(_REQUIRED_CAPABILITIES)}]
                )
                capabilities = self._require_capabilities(version_response)
                self._capabilities = dict(capabilities)
                return dict(capabilities)

            version_response = await self._run_one_shot_command(
                ["version", {"required": list(_REQUIRED_CAPABILITIES)}]
            )
            capabilities = self._require_capabilities(version_response)

            command = self._build_command()
            self._debug(
                "starting Watchman CLI session with "
                f"binary={self._binary_path} socket={self._socket_path}"
            )
            self._process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(self._project_root),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._build_command_env(),
            )
            self._process_wait_task = asyncio.create_task(self._wait_for_process_exit())
            self._reader_task = asyncio.create_task(self._reader_loop())
            self._stderr_task = asyncio.create_task(self._stderr_loop())
            self._capabilities = dict(capabilities)
            return dict(capabilities)
        except Exception:
            await self.stop()
            raise

    async def watch_project(self, target_path: Path) -> dict[str, object]:
        self._ensure_process_running()
        return await self._send_command(["watch-project", str(target_path.resolve())])

    async def startup_watch_project_once(
        self, target_path: Path
    ) -> dict[str, object]:
        return await self._run_one_shot_command(
            ["watch-project", str(target_path.resolve())]
        )

    async def watch_roots(self, roots: Sequence[Path]) -> tuple[Path, ...]:
        self._ensure_process_running()
        watched_roots: list[Path] = []
        seen_roots: set[Path] = set()
        for root in roots:
            if root in seen_roots:
                continue
            seen_roots.add(root)
            await self._send_command(["watch", str(root)])
            watched_roots.append(root)
        return tuple(watched_roots)

    async def startup_watch_roots_once(self, roots: Sequence[Path]) -> tuple[Path, ...]:
        watched_roots: list[Path] = []
        seen_roots: set[Path] = set()
        for root in roots:
            if root in seen_roots:
                continue
            seen_roots.add(root)
            await self._run_one_shot_command(["watch", str(root)])
            watched_roots.append(root)
        return tuple(watched_roots)

    async def subscribe_scopes(
        self,
        *,
        target_path: Path,
        scope_plan: WatchmanScopePlan,
        subscription_name: str | None = None,
    ) -> WatchmanSessionSetup:
        self._ensure_process_running()
        if self._subscription_names:
            raise RuntimeError("Watchman session already has active subscriptions")

        resolved_subscription_name = subscription_name or _DEFAULT_SUBSCRIPTION_NAME
        resolved_subscription_names: list[str] = []
        resolved_subscription_scopes: dict[str, WatchmanSubscriptionScope] = {}
        target_root = target_path.resolve()
        for scope_index, scope in enumerate(scope_plan.scopes):
            scoped_subscription_name = self._subscription_name_for_scope(
                base_name=resolved_subscription_name,
                target_path=target_root,
                scope=scope,
                scope_index=scope_index,
            )
            subscribe_response = await self._send_command(
                self._build_subscribe_command(
                    scope=scope,
                    subscription_name=scoped_subscription_name,
                )
            )
            subscribed_name = subscribe_response.get("subscribe")
            if subscribed_name != scoped_subscription_name:
                raise RuntimeError(
                    "Watchman subscribe response did not confirm the expected "
                    f"subscription name: {subscribed_name!r}"
                )
            resolved_subscription_names.append(scoped_subscription_name)
            resolved_subscription_scopes[scoped_subscription_name] = scope

        self._scope_plan = scope_plan
        self._subscription_name = resolved_subscription_name
        self._subscription_names = tuple(resolved_subscription_names)
        self._subscription_scopes = dict(resolved_subscription_scopes)
        return WatchmanSessionSetup(
            scope_plan=scope_plan,
            subscription_name=resolved_subscription_name,
            subscription_names=tuple(resolved_subscription_names),
            capabilities=dict(self._capabilities),
        )

    async def start(
        self,
        *,
        target_path: Path,
        subscription_name: str | None = None,
        scope_plan: WatchmanScopePlan | None = None,
        nested_mount_roots: Sequence[Path] | None = None,
        additional_scopes: Sequence[WatchmanSubscriptionScope] | None = None,
    ) -> WatchmanSessionSetup:
        await self.prepare()
        try:
            resolved_scope_plan = scope_plan
            if resolved_scope_plan is None:
                resolved_nested_mount_roots = (
                    discover_nested_linux_mount_roots(target_path)
                    if nested_mount_roots is None
                    else tuple(nested_mount_roots)
                )
                resolved_additional_scopes = (
                    discover_nested_windows_junction_scopes(target_path)
                    if additional_scopes is None
                    else tuple(additional_scopes)
                )
                startup_watch_roots = [
                    *resolved_nested_mount_roots,
                    *(scope.watch_root for scope in resolved_additional_scopes),
                ]
                if self.supports_prepared_session_startup():
                    watch_project_response = await self.watch_project(target_path)
                    await self.watch_roots(startup_watch_roots)
                else:
                    watch_project_response = await self.startup_watch_project_once(
                        target_path
                    )
                    await self.startup_watch_roots_once(startup_watch_roots)
                resolved_scope_plan = build_watchman_scope_plan(
                    target_path,
                    watch_project_response,
                    nested_mount_roots=resolved_nested_mount_roots,
                    additional_scopes=resolved_additional_scopes,
                )

            return await self.subscribe_scopes(
                target_path=resolved_scope_plan.primary_scope.requested_path,
                scope_plan=resolved_scope_plan,
                subscription_name=subscription_name,
            )
        except Exception:
            await self.stop()
            raise

    async def stop(self) -> None:
        self._stop_requested = True
        self._scope_plan = None
        self._subscription_name = None
        self._subscription_names = ()
        self._subscription_scopes = {}
        self._capabilities = {}
        self._clear_subscription_queue()
        self._fail_pending_reply(RuntimeError("Watchman session stopped"))
        self._resolve_unexpected_exit(None)

        writer = self._stream_writer
        process = self._process
        reader_task = self._reader_task
        stderr_task = self._stderr_task
        process_wait_task = self._process_wait_task
        cleanup_complete = False

        try:
            if writer is not None:
                writer.close()
                try:
                    await writer.wait_closed()
                except (BrokenPipeError, ConnectionResetError):
                    pass

            if process is not None and process.stdin is not None:
                try:
                    process.stdin.close()
                    await process.stdin.wait_closed()
                except (BrokenPipeError, ConnectionResetError, AttributeError):
                    pass

            if process is not None and process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(
                        process.wait(), timeout=self._PROCESS_EXIT_TIMEOUT_SECONDS
                    )
                except asyncio.TimeoutError:
                    process.kill()
                    await asyncio.wait_for(
                        process.wait(), timeout=self._PROCESS_EXIT_TIMEOUT_SECONDS
                    )

            await self._await_background_task(process_wait_task)
            await self._await_background_task(reader_task)
            await self._await_background_task(stderr_task)
            cleanup_complete = True
        finally:
            # If shutdown is cancelled mid-flight, keep the live handles on the
            # session object so a follow-up stop() call can still terminate the
            # same process and await the same tasks safely.
            if cleanup_complete:
                if self._stream_reader is not None:
                    self._stream_reader = None
                if self._stream_writer is writer:
                    self._stream_writer = None
                if self._process is process:
                    self._process = None
                if self._reader_task is reader_task:
                    self._reader_task = None
                if self._stderr_task is stderr_task:
                    self._stderr_task = None
                if self._process_wait_task is process_wait_task:
                    self._process_wait_task = None

    async def wait_for_unexpected_exit(self) -> str | None:
        future = self._unexpected_exit_future
        if future is None:
            return None
        return await asyncio.shield(future)

    def get_health(self) -> dict[str, Any]:
        process_alive = self._process is not None and self._process.returncode is None
        direct_session_alive = (
            self._stream_writer is not None and not self._stop_requested
        )
        session_alive = process_alive or direct_session_alive
        primary_scope = self._scope_plan.primary_scope if self._scope_plan else None
        watchman_scopes: list[dict[str, object]] = []
        for subscription_name in self._subscription_names:
            scope = self._subscription_scopes.get(subscription_name)
            if scope is None:
                continue
            watchman_scopes.append(
                {
                    "subscription_name": subscription_name,
                    "scope_kind": scope.scope_kind,
                    "requested_path": str(scope.requested_path),
                    "watch_root": str(scope.watch_root),
                    "relative_root": scope.relative_root,
                }
            )
        return {
            "watchman_session_alive": session_alive,
            "watchman_session_pid": self._process.pid if self._process else None,
            "watchman_session_last_warning": self._last_warning,
            "watchman_session_last_warning_at": self._last_warning_at,
            "watchman_session_last_error": self._last_error,
            "watchman_session_last_error_at": self._last_error_at,
            "watchman_session_last_response_at": self._last_response_at,
            "watchman_subscription_last_received_at": self._last_subscription_at,
            "watchman_session_command_count": self._command_count,
            "watchman_subscription_queue_size": self.subscription_queue.qsize(),
            "watchman_subscription_queue_maxsize": self.subscription_queue.maxsize,
            "watchman_subscription_pdu_count": self._subscription_pdu_count,
            "watchman_subscription_pdu_dropped": self._subscription_pdu_dropped,
            "watchman_subscription_name": self._subscription_name,
            "watchman_subscription_names": list(self._subscription_names),
            "watchman_watch_root": (
                str(primary_scope.watch_root) if primary_scope else None
            ),
            "watchman_relative_root": (
                primary_scope.relative_root if primary_scope else None
            ),
            "watchman_scopes": watchman_scopes,
            "watchman_session_capabilities": dict(self._capabilities),
        }

    def _build_command(self, *, persistent: bool = True) -> list[str]:
        runtime = resolve_packaged_watchman_runtime()
        command = build_watchman_client_command(
            runtime=runtime,
            binary_path=self._binary_path,
            socket_path=self._socket_path,
            statefile_path=self._statefile_path,
            logfile_path=self._logfile_path,
            pidfile_path=self._pidfile_path,
            persistent=persistent,
        )
        if self._command_prefix is None:
            return command
        default_prefix = build_watchman_runtime_command_prefix(
            runtime=runtime,
            binary_path=self._binary_path,
        )
        return [*self._command_prefix, *command[len(default_prefix) :]]

    def _build_command_env(self) -> dict[str, str]:
        if self._command_env is not None:
            return dict(self._command_env)
        runtime = resolve_packaged_watchman_runtime()
        return build_watchman_runtime_environment(
            runtime=runtime,
            binary_path=self._binary_path,
        )

    def _build_subscribe_command(
        self,
        *,
        scope: WatchmanSubscriptionScope,
        subscription_name: str,
    ) -> list[object]:
        payload: dict[str, object] = {
            "expression": ["allof", ["type", "f"]],
            "fields": ["name", "exists", "new", "type"],
            "empty_on_fresh_instance": True,
        }
        if scope.relative_root is not None:
            payload["relative_root"] = scope.relative_root
        return ["subscribe", str(scope.watch_root), subscription_name, payload]

    def _ensure_process_running(self) -> None:
        if self._stream_writer is not None and not self._stop_requested:
            return
        process = self._process
        if process is None or process.returncode is not None or process.stdin is None:
            raise RuntimeError("Watchman session process is not running")

    async def _send_command(self, command: list[object]) -> dict[str, object]:
        self._ensure_process_running()
        payload = json.dumps(command, separators=(",", ":")).encode("utf-8") + b"\n"
        async with self._command_lock:
            loop = asyncio.get_running_loop()
            future: asyncio.Future[dict[str, object]] = loop.create_future()
            self._pending_reply = future
            try:
                writer = self._stream_writer
                if writer is not None:
                    writer.write(payload)
                    await writer.drain()
                else:
                    process = self._process
                    assert process is not None
                    assert process.stdin is not None
                    process.stdin.write(payload)
                    await process.stdin.drain()
                response = await asyncio.wait_for(
                    asyncio.shield(future), timeout=self._COMMAND_TIMEOUT_SECONDS
                )
            finally:
                if self._pending_reply is future:
                    self._pending_reply = None

        self._command_count += 1
        self._last_response_at = _utc_now()

        warning = response.get("warning")
        if isinstance(warning, str) and warning:
            self._record_warning(warning)

        error = response.get("error")
        if isinstance(error, str) and error:
            self._record_error(error)
            raise RuntimeError(error)

        return response

    async def _run_one_shot_command(self, command: list[object]) -> dict[str, object]:
        process = await asyncio.create_subprocess_exec(
            *self._build_command(persistent=False),
            cwd=str(self._project_root),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._build_command_env(),
        )
        try:
            if (
                process.stdin is None
                or process.stdout is None
                or process.stderr is None
            ):
                raise RuntimeError(
                    "Watchman one-shot client did not expose stdio pipes"
                )

            process.stdin.write(
                json.dumps(command, separators=(",", ":")).encode("utf-8") + b"\n"
            )
            await process.stdin.drain()
            process.stdin.close()
            await process.stdin.wait_closed()

            deadline = asyncio.get_running_loop().time() + self._COMMAND_TIMEOUT_SECONDS
            while True:
                timeout = deadline - asyncio.get_running_loop().time()
                if timeout <= 0:
                    raise RuntimeError(
                        f"Timed out waiting for Watchman response to {command[0]!r}"
                    )
                raw_line = await asyncio.wait_for(
                    process.stdout.readline(),
                    timeout=timeout,
                )
                if not raw_line:
                    stderr_text = (await process.stderr.read()).decode(
                        "utf-8", errors="replace"
                    )
                    raise RuntimeError(
                        "Watchman one-shot client exited before returning a response "
                        f"for {command[0]!r}: {stderr_text.strip()}"
                    )
                payload = json.loads(raw_line.decode("utf-8"))
                if not isinstance(payload, dict):
                    continue
                if isinstance(payload.get("log"), str):
                    self._record_warning(f"watchman log: {payload['log']}")
                    continue
                error = payload.get("error")
                if isinstance(error, str) and error:
                    self._record_error(error)
                    raise RuntimeError(error)
                self._last_response_at = _utc_now()
                self._command_count += 1
                return payload
        finally:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(
                        process.wait(), timeout=self._PROCESS_EXIT_TIMEOUT_SECONDS
                    )
                except asyncio.TimeoutError:
                    process.kill()
                    await asyncio.wait_for(
                        process.wait(), timeout=self._PROCESS_EXIT_TIMEOUT_SECONDS
                    )

    async def _reader_loop(self) -> None:
        reader = self._stream_reader
        if reader is None:
            process = self._process
            if process is None or process.stdout is None:
                return
            reader = process.stdout

        try:
            while True:
                raw_line = await reader.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                self._debug(f"watchman session stdout: {line}")
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as error:
                    self._record_error(f"Invalid Watchman JSON output: {error}")
                    continue

                if not isinstance(payload, dict):
                    self._record_warning(
                        "Ignoring unexpected non-object Watchman payload"
                    )
                    continue

                if "subscription" in payload:
                    self._queue_subscription_pdu(payload)
                    continue

                if "log" in payload:
                    log_message = payload.get("log")
                    if isinstance(log_message, str) and log_message:
                        self._record_warning(f"watchman log: {log_message}")
                    continue

                reply = self._pending_reply
                if reply is not None and not reply.done():
                    reply.set_result(payload)
                    continue

                self._record_warning(
                    "Received unexpected Watchman response without a pending command"
                )
        except asyncio.CancelledError:
            pass
        except Exception as error:
            self._record_error(f"Watchman stdout reader failed: {error}")
            self._fail_pending_reply(error)
        else:
            if self._stream_writer is not None and not self._stop_requested:
                message = "Watchman session connection closed unexpectedly"
                self._record_error(message)
                self._fail_pending_reply(RuntimeError(message))
                self._resolve_unexpected_exit(message)

    async def _stderr_loop(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return

        while True:
            raw_line = await process.stderr.readline()
            if not raw_line:
                break
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            self._debug(f"watchman session stderr: {line}")
            self._record_warning(f"watchman stderr: {line}")

    def _queue_subscription_pdu(self, payload: dict[str, object]) -> None:
        self._last_subscription_at = _utc_now()
        warning = payload.get("warning")
        if isinstance(warning, str) and warning:
            self._record_warning(warning)

        error = payload.get("error")
        if isinstance(error, str) and error:
            self._record_error(error)

        try:
            self.subscription_queue.put_nowait(payload)
        except asyncio.QueueFull:
            self._subscription_pdu_dropped += 1
            self._record_warning(
                "Watchman subscription queue full; dropped a raw subscription PDU"
            )
            if self._subscription_overflow_handler is not None:
                try:
                    self._subscription_overflow_handler(
                        payload,
                        self._subscription_pdu_dropped,
                        self.subscription_queue.maxsize,
                    )
                except Exception as error:
                    self._record_error(
                        f"watchman subscription overflow handler failed: {error}"
                    )
            return
        self._subscription_pdu_count += 1

    def _require_capabilities(self, response: dict[str, object]) -> dict[str, bool]:
        raw_capabilities = response.get("capabilities")
        if not isinstance(raw_capabilities, dict):
            raise RuntimeError(
                "Watchman version response did not include a capabilities object"
            )
        missing_capability = os.environ.get(
            "CHUNKHOUND_FAKE_WATCHMAN_MISSING_CAPABILITY"
        )
        if missing_capability:
            raw_capabilities = dict(raw_capabilities)
            raw_capabilities[missing_capability] = False

        capabilities: dict[str, bool] = {}
        for capability in _REQUIRED_CAPABILITIES:
            value = raw_capabilities.get(capability)
            if value is not True:
                raise RuntimeError(
                    "Watchman session requires capability "
                    f"{capability!r}, but the sidecar reported {value!r}"
                )
            capabilities[capability] = True
        return capabilities

    def _fail_pending_reply(self, error: Exception) -> None:
        reply = self._pending_reply
        if reply is not None and not reply.done():
            reply.set_exception(error)
        self._pending_reply = None

    def _reset_state(self) -> None:
        self._stop_requested = False
        self._stream_reader = None
        self._stream_writer = None
        self._scope_plan = None
        self._subscription_name = None
        self._subscription_names = ()
        self._subscription_scopes = {}
        self._capabilities = {}
        self._last_warning = None
        self._last_warning_at = None
        self._last_error = None
        self._last_error_at = None
        self._last_response_at = None
        self._last_subscription_at = None
        self._command_count = 0
        self._subscription_pdu_count = 0
        self._subscription_pdu_dropped = 0
        self._clear_subscription_queue()
        self._process_wait_task = None
        self._unexpected_exit_future = None

    def _clear_subscription_queue(self) -> None:
        while True:
            try:
                self.subscription_queue.get_nowait()
            except asyncio.QueueEmpty:
                return

    def _debug(self, message: str) -> None:
        try:
            if self._debug_sink is not None:
                self._debug_sink(f"watchman-session: {message}")
        except Exception:
            pass

    def _record_warning(self, message: str) -> None:
        self._last_warning = message
        self._last_warning_at = _utc_now()
        self._debug(f"warning: {message}")

    def _record_error(self, message: str) -> None:
        self._last_error = message
        self._last_error_at = _utc_now()
        self._debug(f"error: {message}")

    def _resolve_unexpected_exit(self, message: str | None) -> None:
        future = self._unexpected_exit_future
        if future is not None and not future.done():
            future.set_result(message)

    def _use_direct_socket_session(self) -> bool:
        return self.supports_prepared_session_startup()

    async def _wait_for_process_exit(self) -> None:
        process = self._process
        if process is None:
            return

        try:
            returncode = await process.wait()
        except asyncio.CancelledError:
            return

        if self._stop_requested:
            return

        message = f"Watchman session exited unexpectedly (rc={returncode})"
        self._record_error(message)
        self._fail_pending_reply(RuntimeError(message))
        self._resolve_unexpected_exit(message)

    async def _await_background_task(self, task: asyncio.Task[None] | None) -> None:
        if task is None:
            return
        if task.done():
            try:
                await task
            except asyncio.CancelledError:
                pass
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    def _subscription_name_for_scope(
        self,
        *,
        base_name: str,
        target_path: Path,
        scope: WatchmanSubscriptionScope,
        scope_index: int,
    ) -> str:
        if scope.scope_kind == "primary" or scope_index == 0:
            return base_name
        try:
            suffix_source = scope.requested_path.relative_to(target_path).as_posix()
        except ValueError:
            suffix_source = scope.requested_path.as_posix()
        suffix = self._sanitize_subscription_suffix(suffix_source)
        if not suffix:
            suffix = f"scope-{scope_index}"
        return self._bound_subscription_name(f"{base_name}--{suffix}")

    @classmethod
    def _bound_subscription_name(cls, value: str) -> str:
        if len(value) <= cls._SUBSCRIPTION_NAME_MAX_LENGTH:
            return value
        hash_suffix = hashlib.sha256(value.encode("utf-8")).hexdigest()[
            : cls._SUBSCRIPTION_NAME_HASH_LENGTH
        ]
        prefix_limit = cls._SUBSCRIPTION_NAME_MAX_LENGTH - len(hash_suffix) - 1
        prefix = value[:prefix_limit].rstrip("-")
        if not prefix:
            prefix = value[:prefix_limit]
        return f"{prefix}-{hash_suffix}"

    @staticmethod
    def _sanitize_subscription_suffix(value: str) -> str:
        candidate = value.replace("\\", "/").strip("/")
        if not candidate:
            return ""
        parts: list[str] = []
        for chunk in PurePosixPath(candidate).parts:
            normalized_chunk = "".join(
                character.lower() if character.isalnum() else "-"
                for character in chunk
            ).strip("-")
            if normalized_chunk:
                parts.append(normalized_chunk)
        return "-".join(parts)
