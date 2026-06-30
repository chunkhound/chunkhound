"""ChunkHound daemon server — single DuckDB owner, multi-client IPC server.

Extends MCPServerBase for service lifecycle and adds a JSON-RPC 2.0 IPC
server that serves multiple MCP proxy clients concurrently.

IPC handshake (length-prefixed frames):
  Client → {"type":"register","pid":<proxy_pid>}
  Daemon → {"type":"registered","client_id":"<uuid>"}
  [subsequent frames: raw MCP JSON-RPC 2.0 messages]
"""

from __future__ import annotations

import asyncio
import os
import secrets
import sys
import uuid
from pathlib import Path
from typing import Any

from chunkhound.core.config.config import Config
from chunkhound.mcp_server.base import MCPServerBase
from chunkhound.mcp_server.common import handle_tool_call, tool_call_failed
from chunkhound.version import __version__

from . import ipc
from .client_manager import ClientManager
from .discovery import DaemonDiscovery

_DEFAULT_SHUTDOWN_DELAY = 0.0


class ChunkHoundDaemon(MCPServerBase):
    """Daemon that owns the sole DuckDB connection and serves multiple clients.

    All MCP protocol handling is implemented in pure JSON-RPC 2.0 without
    depending on the MCP SDK so that the daemon can safely import independently.
    """

    def __init__(
        self,
        config: Config,
        args: Any,
        socket_path: str,
        project_dir: Path,
    ) -> None:
        super().__init__(config, args=args)
        self._socket_path = socket_path
        self._project_dir = project_dir
        self._discovery = DaemonDiscovery(project_dir)
        self._shutdown_event = asyncio.Event()
        self._initialization_complete = asyncio.Event()
        self._pid_poll_task: asyncio.Task | None = None
        self._delayed_shutdown_task: asyncio.Task[None] | None = None
        self._client_manager = ClientManager(on_empty=self._on_all_clients_gone)
        # True only after we successfully bound the socket and published our lock.
        self._lock_written = False
        self._auth_token: str | None = None  # Set before accepting connections
        delay_str = os.environ.get(
            "CHUNKHOUND_DAEMON_SHUTDOWN_DELAY", str(_DEFAULT_SHUTDOWN_DELAY)
        )
        try:
            self._shutdown_delay = float(delay_str)
        except ValueError:
            self._shutdown_delay = _DEFAULT_SHUTDOWN_DELAY

    # ------------------------------------------------------------------
    # MCPServerBase abstract requirements
    # ------------------------------------------------------------------

    def _register_tools(self) -> None:
        """No-op: daemon dispatches tools via JSON-RPC directly."""
        pass

    def _realtime_startup_mode(self) -> str:
        """Report daemon-mode startup timing for the public status surface."""
        return "daemon"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Start the IPC server and run until all clients disconnect."""
        try:
            # Register SIGTERM / SIGINT handlers so the daemon shuts down
            # gracefully (runs _graceful_shutdown) instead of being killed hard.
            import signal as _signal

            def _on_signal() -> None:
                self.debug_log("Signal received — initiating graceful shutdown")
                self._shutdown_event.set()

            loop = asyncio.get_running_loop()
            try:
                loop.add_signal_handler(_signal.SIGTERM, _on_signal)
                loop.add_signal_handler(_signal.SIGINT, _on_signal)
            except (NotImplementedError, RuntimeError):
                # Signal handlers not supported on this platform/loop
                pass

            # Initialise services (DB, embeddings, realtime indexing)
            await self.initialize()

            # --- DAEMON PUBLISH FIRST (before sidecar startup barrier) ---
            self._start_startup_phase("daemon_publish")
            try:
                # Generate auth token BEFORE accepting connections so every client
                # sees a non-None token in _handle_client from the very first frame.
                auth_token = secrets.token_hex(32)
                self._auth_token = auth_token

                # Ensure the runtime-scoped Unix socket directory exists before bind.
                if sys.platform != "win32" and not self._socket_path.startswith("tcp:"):
                    Path(self._socket_path).parent.mkdir(parents=True, exist_ok=True)

                # Start the IPC server using the authoritative transport address.
                server, actual_address = await ipc.create_server(
                    self._socket_path, self._handle_client
                )
                self._socket_path = actual_address

                # Write lock file so proxies can discover us.
                self._discovery.write_lock(
                    os.getpid(), self._socket_path, auth_token=auth_token
                )

                # Signal initialization complete BEFORE post-write validation and
                # registry entry, so tool calls never wait for the event after
                # the daemon becomes discoverable via the lock file.
                self._initialization_complete.set()

                # Post-write validation: verify our PID is the one recorded; if
                # not, another daemon won the startup race.
                written_lock = self._discovery.read_lock()
                if written_lock is None or written_lock.get("pid") != os.getpid():
                    message = (
                        "Daemon publish failed: another daemon won the startup race"
                    )
                    self.debug_log(
                        "Lock file PID mismatch after write — another daemon won "
                        "the race; shutting down"
                    )
                    self._set_startup_failure(
                        message,
                        phase_name="daemon_publish",
                    )
                    self._resolve_startup_publish_complete()
                    self._shutdown_event.set()
                    return

                self._lock_written = True
                self._mark_startup_exposure_ready()
                try:
                    # The socket may already be connectable by the time we publish
                    # this entry. That is acceptable because the registry is only
                    # an index: overlapping startups still re-check overlap state
                    # under the global startup lock before launching a new daemon.
                    self._discovery.write_registry_entry(os.getpid(), self._socket_path)
                except Exception as e:
                    self.debug_log(f"Registry publish failed (non-fatal): {e}")
                self.debug_log(
                    "Lock file written "
                    f"(pid={os.getpid()}, address={self._socket_path})"
                )
                self._complete_startup_phase("daemon_publish")
                self._resolve_startup_publish_complete()
            except Exception as error:
                self._set_startup_failure(
                    f"Daemon publish failed: {error}",
                    phase_name="daemon_publish",
                )
                self._resolve_startup_publish_complete()
                self._shutdown_event.set()
                return

            # --- THEN deferred startup barrier (sidecar readiness) ---
            # _initialization_complete was already set right after write_lock(),
            # before any client could discover this daemon.
            await self.await_startup_barrier()
            self._complete_startup()

            # Start PID poll background task
            self._pid_poll_task = asyncio.create_task(self._client_manager.poll_pids())

            self.debug_log(f"Listening on {self._socket_path}")

            async with server:
                await self._shutdown_event.wait()

            self.debug_log("Shutdown event received, tearing down")

        except Exception as e:
            self.debug_log(f"Daemon run() error: {e}")
            import traceback

            traceback.print_exc(file=sys.stderr)
            raise
        finally:
            await self._graceful_shutdown()

    def _on_all_clients_gone(self) -> None:
        """Called by ClientManager when the last client disconnects."""
        if (
            self._delayed_shutdown_task is not None
            and not self._delayed_shutdown_task.done()
        ):
            self.debug_log("Last client disconnected — shutdown already scheduled")
            return

        self.debug_log("Last client disconnected — scheduling shutdown")
        task = asyncio.create_task(self._delayed_shutdown())
        self._delayed_shutdown_task = task
        task.add_done_callback(self._clear_delayed_shutdown_task)

    def _clear_delayed_shutdown_task(self, task: asyncio.Task[None]) -> None:
        if self._delayed_shutdown_task is task:
            self._delayed_shutdown_task = None

    async def _delayed_shutdown(self) -> None:
        """Optionally wait shutdown_delay seconds before triggering shutdown."""
        if self._shutdown_delay > 0:
            self.debug_log(f"Shutdown delay: waiting {self._shutdown_delay}s")
            await asyncio.sleep(self._shutdown_delay)
        # Re-check in case a new client connected during the delay
        if self._client_manager.count() == 0:
            self._shutdown_event.set()

    # ------------------------------------------------------------------
    # Client connection handling
    # ------------------------------------------------------------------

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a single proxy client connection."""
        client_id: str | None = None
        try:
            # --- Registration handshake ---
            try:
                reg = await asyncio.wait_for(ipc.read_frame(reader), timeout=10.0)
            except asyncio.IncompleteReadError:
                return
            if not isinstance(reg, dict) or reg.get("type") != "register":
                return

            # Auth token check — reject unknown clients silently to avoid
            # leaking information about the expected token value.
            if self._auth_token is not None and (
                reg.get("auth_token") != self._auth_token
            ):
                return

            raw_pid = reg.get("pid", 0)
            try:
                pid: int = int(raw_pid)
            except (TypeError, ValueError):
                return  # Reject malformed registration frame
            client_id = str(uuid.uuid4())
            self._client_manager.register(client_id, pid, writer)
            self.debug_log(f"Client registered: id={client_id} pid={pid}")

            # Acknowledge
            ipc.write_frame(writer, {"type": "registered", "client_id": client_id})
            await writer.drain()

            # --- MCP JSON-RPC message loop ---
            while True:
                try:
                    msg = await ipc.read_frame(reader)
                except asyncio.IncompleteReadError:
                    break
                except Exception as e:
                    self.debug_log(f"IPC read error: {e!r}")
                    break

                if not isinstance(msg, dict):
                    continue

                response = await self._dispatch_mcp(msg, client_id)
                if response is not None:
                    ipc.write_frame(writer, response)
                    await writer.drain()

        except asyncio.TimeoutError:
            self.debug_log("Client registration timed out")
        except Exception as e:
            self.debug_log(f"Client handler error: {e}")
        finally:
            if client_id is not None:
                self._client_manager.remove(client_id)
                self.debug_log(f"Client disconnected: {client_id}")
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # MCP JSON-RPC dispatch
    # ------------------------------------------------------------------

    async def _dispatch_mcp(
        self, msg: dict[str, Any], client_id: str
    ) -> dict[str, Any] | None:
        """Route a JSON-RPC 2.0 message to the correct handler.

        Returns a JSON-RPC response dict, or None for notifications.
        """
        method: str = msg.get("method", "")
        req_id = msg.get("id")

        # Notifications have no "id" — fire-and-forget
        if req_id is None and method.startswith("notifications/"):
            return None

        try:
            if method == "initialize":
                return await self._handle_initialize(msg)
            elif method == "tools/list":
                return await self._handle_tools_list(msg)
            elif method == "tools/call":
                return await self._handle_tools_call(msg)
            elif method == "ping":
                return {"jsonrpc": "2.0", "id": req_id, "result": {}}
            elif method.startswith("notifications/"):
                return None
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}",
                    },
                }
        except Exception as e:
            self.debug_log(f"Dispatch error for {method}: {e}")
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32603, "message": str(e)},
            }

    async def _handle_initialize(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Respond to the MCP initialize request."""
        return {
            "jsonrpc": "2.0",
            "id": msg.get("id"),
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                    "name": "ChunkHound Code Search",
                    "version": __version__,
                },
                "capabilities": {"tools": {}},
            },
        }

    async def _handle_tools_list(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Respond to the tools/list request with available tool schemas."""
        try:
            await asyncio.wait_for(self._initialization_complete.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            pass

        tools = self._build_filtered_tool_dicts()
        return {
            "jsonrpc": "2.0",
            "id": msg.get("id"),
            "result": {"tools": tools},
        }

    async def _handle_tools_call(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool and return its result."""
        params = msg.get("params", {})
        tool_name: str = params.get("name", "")
        arguments: dict[str, Any] = params.get("arguments", {})

        text_contents = await handle_tool_call(
            tool_name=tool_name,
            arguments=arguments,
            services=self.services,
            embedding_manager=self.embedding_manager,
            initialization_complete=self._initialization_complete,
            debug_mode=self.debug_mode,
            scan_progress=self._scan_progress,
            llm_manager=self.llm_manager,
            config=self.config,
            ensure_services=self.ensure_tool_services,
        )

        content = [{"type": tc.type, "text": tc.text} for tc in text_contents]

        return {
            "jsonrpc": "2.0",
            "id": msg.get("id"),
            "result": {"content": content, "isError": tool_call_failed(text_contents)},
        }

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def _graceful_shutdown(self) -> None:
        """Stop background tasks, clean up services, remove published artifacts.

        Only the daemon that won publication may remove shared artifacts. Race
        losers must not delete the winner's lock, socket path, or registry entry.
        """
        # Close active client connections immediately so proxies get EOF and
        # exit rather than deadlocking until the daemon process fully exits.
        try:
            await self._client_manager.close_all()
        except Exception as e:
            self.debug_log(f"Client close error (non-fatal): {e}")

        if self._pid_poll_task is not None and not self._pid_poll_task.done():
            self._pid_poll_task.cancel()
            try:
                await self._pid_poll_task
            except asyncio.CancelledError:
                pass

        delayed_shutdown_task = self._delayed_shutdown_task
        if delayed_shutdown_task is not None:
            if not delayed_shutdown_task.done():
                delayed_shutdown_task.cancel()
            try:
                await delayed_shutdown_task
            except asyncio.CancelledError:
                pass
            finally:
                if self._delayed_shutdown_task is delayed_shutdown_task:
                    self._delayed_shutdown_task = None

        try:
            await asyncio.wait_for(self.cleanup(), timeout=10.0)
        except (asyncio.TimeoutError, Exception) as e:
            self.debug_log(f"Cleanup error (non-fatal): {e}")

        if self._lock_written:
            try:
                self._discovery.remove_lock()
                self.debug_log("Removed IPC lock file")
            except Exception as e:
                self.debug_log(f"Lock removal failed (non-fatal): {e}")

            if sys.platform != "win32" and not self._socket_path.startswith("tcp:"):
                try:
                    os.unlink(self._socket_path)
                    self.debug_log("Removed IPC socket")
                except FileNotFoundError:
                    pass
            else:
                self.debug_log("Skipping socket cleanup — TCP transport")

            try:
                self._discovery.remove_registry_entry()
            except Exception as e:
                self.debug_log(f"Registry cleanup failed (non-fatal): {e}")

        self.debug_log("Daemon shutdown complete")
