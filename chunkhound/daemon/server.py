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
import sys
import uuid
from pathlib import Path
from typing import Any

from chunkhound.core.config.config import Config
from chunkhound.mcp_server.base import MCPServerBase
from chunkhound.mcp_server.common import handle_tool_call
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
        self._client_manager = ClientManager(on_empty=self._on_all_clients_gone)
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

            loop = asyncio.get_event_loop()
            try:
                loop.add_signal_handler(_signal.SIGTERM, _on_signal)
                loop.add_signal_handler(_signal.SIGINT, _on_signal)
            except (NotImplementedError, RuntimeError):
                # Signal handlers not supported on this platform/loop
                pass

            # Initialise services (DB, embeddings, realtime indexing)
            await self.initialize()
            self._initialization_complete.set()
            self.debug_log("Daemon initialised")

            # Start IPC server; on Windows actual address differs (port 0 → real port)
            server, actual_address = await ipc.create_server(
                self._socket_path, self._handle_client
            )
            self._socket_path = actual_address

            # Write lock file so proxies can discover us
            self._discovery.write_lock(os.getpid(), self._socket_path)
            self.debug_log(
                f"Lock file written (pid={os.getpid()}, address={self._socket_path})"
            )

            # Start PID poll background task
            self._pid_poll_task = asyncio.create_task(
                self._client_manager.poll_pids()
            )

            self.debug_log(f"Listening on {self._socket_path}")

            async with server:
                await self._shutdown_event.wait()

            self.debug_log("Shutdown event received, tearing down")

        except Exception as e:
            self.debug_log(f"Daemon run() error: {e}")
            import traceback
            traceback.print_exc(file=sys.stderr)
        finally:
            await self._graceful_shutdown()

    def _on_all_clients_gone(self) -> None:
        """Called by ClientManager when the last client disconnects."""
        self.debug_log("Last client disconnected — scheduling shutdown")
        asyncio.create_task(self._delayed_shutdown())

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

            pid: int = reg.get("pid", 0)
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
                except (asyncio.IncompleteReadError, Exception):
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
            await asyncio.wait_for(
                self._initialization_complete.wait(), timeout=5.0
            )
        except asyncio.TimeoutError:
            pass

        tools = self._build_available_tools_as_dicts()
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
            services=self.ensure_services(),
            embedding_manager=self.embedding_manager,
            initialization_complete=self._initialization_complete,
            debug_mode=self.debug_mode,
            scan_progress=self._scan_progress,
            llm_manager=self.llm_manager,
            config=self.config,
        )

        content = [
            {"type": tc.type, "text": tc.text}
            for tc in text_contents
        ]

        return {
            "jsonrpc": "2.0",
            "id": msg.get("id"),
            "result": {"content": content, "isError": False},
        }

    # ------------------------------------------------------------------
    # Tool schema building (mirrors StdioMCPServer.build_available_tools)
    # ------------------------------------------------------------------

    def _build_available_tools_as_dicts(self) -> list[dict[str, Any]]:
        """Build a JSON-serialisable list of available tool schemas."""
        return self._build_filtered_tool_dicts()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def _graceful_shutdown(self) -> None:
        """Stop background tasks, clean up services, remove lock file.

        Always removes the lock file and socket, even if cleanup fails.
        """
        if self._pid_poll_task is not None and not self._pid_poll_task.done():
            self._pid_poll_task.cancel()
            try:
                await self._pid_poll_task
            except asyncio.CancelledError:
                pass

        try:
            await asyncio.wait_for(self.cleanup(), timeout=10.0)
        except (asyncio.TimeoutError, Exception) as e:
            self.debug_log(f"Cleanup error (non-fatal): {e}")

        # Always remove lock file and socket regardless of cleanup outcome
        self._discovery.remove_lock()

        # Remove socket file on Unix; TCP loopback needs no cleanup
        if sys.platform != "win32" and not self._socket_path.startswith("tcp:"):
            try:
                os.unlink(self._socket_path)
            except FileNotFoundError:
                pass

        self.debug_log("Daemon shutdown complete")
