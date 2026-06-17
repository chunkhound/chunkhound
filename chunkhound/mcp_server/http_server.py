"""HTTP MCP server using aiohttp + MCP Streamable HTTP transport.

Returns 200 OK + text/event-stream for tool calls with periodic SSE
keepalive comments so Claude Code's HTTP client never sees an idle
connection regardless of how long the tool takes.

NEVER USE print() in this module — it breaks the stdio MCP server if
this module is imported alongside it. Use sys.stderr or the debug log.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

from aiohttp import web

from chunkhound.core.config.config import Config
from chunkhound.mcp_server.base import MCPServerBase
from chunkhound.mcp_server.common import handle_tool_call
from chunkhound.version import __version__


class HttpMCPServer(MCPServerBase):
    """MCP server that speaks Streamable HTTP (POST /mcp).

    Tool calls are answered with an inline SSE stream so the POST
    connection is never idle — avoiding Claude Code's HTTP timeout.
    """

    def __init__(
        self,
        config: Config,
        host: str = "0.0.0.0",
        port: int = 5173,
        args: Any = None,
    ) -> None:
        super().__init__(config, args=args)
        self.host = host
        self.port = port
        self._tool_lock = asyncio.Lock()
        self._initialization_complete = asyncio.Event()

    def _register_tools(self) -> None:
        pass  # Routing handled in HTTP layer; no MCP SDK Server object used.

    async def run(self) -> None:
        """Start the aiohttp application and block until cancelled."""
        app = web.Application()
        app.router.add_post("/mcp", self._handle_post)
        app.router.add_get("/mcp", self._handle_get)
        app.router.add_delete("/mcp", self._handle_delete)
        app.on_startup.append(self._on_startup)
        app.on_cleanup.append(self._on_cleanup)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()

        sys.stderr.write(
            f"[chunkhound] HTTP MCP server listening on {self.host}:{self.port}\n"
        )
        sys.stderr.flush()

        try:
            await asyncio.get_event_loop().create_future()  # run forever
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        finally:
            await runner.cleanup()

    async def _on_startup(self, app: web.Application) -> None:
        await self.initialize()
        self._initialization_complete.set()
        sys.stderr.write("[chunkhound] HTTP MCP server ready\n")
        sys.stderr.flush()

    async def _on_cleanup(self, app: web.Application) -> None:
        await self.cleanup()

    # ------------------------------------------------------------------ #
    # Request handlers                                                     #
    # ------------------------------------------------------------------ #

    async def _handle_post(
        self, request: web.Request
    ) -> web.Response | web.StreamResponse:
        try:
            body = await request.json()
        except Exception:
            return web.Response(status=400, text="Invalid JSON")

        method = body.get("method", "")
        req_id = body.get("id")

        if method == "initialize":
            return self._init_response(req_id)
        if method in ("notifications/initialized", "notifications/cancelled"):
            return web.Response(status=202)
        if method == "tools/list":
            return self._tools_list_response(req_id)
        if method == "tools/call":
            return await self._tool_call_stream(request, body, req_id)
        if method == "ping":
            return web.json_response({"jsonrpc": "2.0", "id": req_id, "result": {}})
        return web.json_response({
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        })

    async def _handle_get(self, request: web.Request) -> web.Response:
        # We answer POST inline — no separate GET SSE channel needed.
        return web.Response(status=405, text="Use POST /mcp")

    async def _handle_delete(self, request: web.Request) -> web.Response:
        return web.Response(status=200)

    # ------------------------------------------------------------------ #
    # Response builders                                                    #
    # ------------------------------------------------------------------ #

    def _init_response(self, req_id: Any) -> web.Response:
        return web.json_response({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                    "name": "ChunkHound Code Search",
                    "version": __version__,
                },
                "capabilities": {"tools": {}},
            },
        })

    def _tools_list_response(self, req_id: Any) -> web.Response:
        tool_dicts = self._build_filtered_tool_dicts()
        return web.json_response({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": t["name"],
                        "description": t["description"],
                        "inputSchema": t["inputSchema"],
                    }
                    for t in tool_dicts
                ]
            },
        })

    async def _tool_call_stream(
        self, request: web.Request, body: dict, req_id: Any
    ) -> web.StreamResponse:
        """Stream tool result as SSE in the POST response body.

        Sends ': ping' comments every 15 s so the TCP connection is never
        idle — Claude Code's idle timeout resets on every received byte.
        """
        params = body.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Access-Control-Allow-Origin": "*",
            },
        )
        await response.prepare(request)
        # Immediate byte so the client knows the stream is live.
        await response.write(b": connected\n\n")

        async def _run() -> list:
            async with self._tool_lock:
                services = await self.ensure_services()
                return await handle_tool_call(
                    tool_name=tool_name,
                    arguments=arguments,
                    services=services,
                    embedding_manager=self.embedding_manager,
                    initialization_complete=self._initialization_complete,
                    debug_mode=self.debug_mode,
                    scan_progress=self._scan_progress,
                    llm_manager=self.llm_manager,
                    config=self.config,
                )

        task = asyncio.create_task(_run())

        while not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=15.0)
            except asyncio.TimeoutError:
                try:
                    await response.write(b": ping\n\n")
                except Exception:
                    task.cancel()
                    return response

        try:
            content = task.result()
        except Exception as exc:
            err_payload = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": str(exc)},
            }
            await response.write(
                f"event: message\ndata: {json.dumps(err_payload)}\n\n".encode()
            )
            return response
        else:
            content_dicts = [{"type": c.type, "text": c.text} for c in content]

        result_payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"content": content_dicts},
        }
        await response.write(
            f"event: message\ndata: {json.dumps(result_payload)}\n\n".encode()
        )
        return response


async def main(args: Any = None) -> None:
    """Async entry point — called from CLI or standalone."""
    import argparse

    from chunkhound.api.cli.utils.config_factory import create_validated_config
    from chunkhound.mcp_server.common import add_common_mcp_arguments

    if args is None:
        parser = argparse.ArgumentParser(description="ChunkHound HTTP MCP server")
        add_common_mcp_arguments(parser)
        parser.add_argument("--port", type=int, default=5173)
        parser.add_argument("--host", default="0.0.0.0")
        args = parser.parse_args()

    os.environ["CHUNKHOUND_MCP_MODE"] = "1"

    try:
        import numpy  # noqa: F401
    except ImportError:
        pass

    config, validation_errors = create_validated_config(args, "mcp")

    if validation_errors:
        for err in validation_errors:
            sys.stderr.write(f"[chunkhound] config warning: {err}\n")
        sys.stderr.flush()

    port = int(getattr(args, "port", 5173))
    host = str(getattr(args, "host", "0.0.0.0"))

    server = HttpMCPServer(config, host=host, port=port, args=args)
    await server.run()


def main_sync() -> None:
    """Synchronous entry point for pyproject.toml scripts."""
    asyncio.run(main())
