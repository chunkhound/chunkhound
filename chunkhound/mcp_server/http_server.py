"""HTTP MCP server implementation using the base class pattern.

This module implements the Streamable HTTP transport for MCP, built on top
of the official MCP SDK's ``StreamableHTTPSessionManager``. It shares tool
dispatch and service lifecycle management with the stdio transport via
``MCPServerBase``.

NOTE: Unlike stdio.py, this module intentionally does NOT silence Python
logging or loguru. Stdio needs silence because log output on stdout would
corrupt the JSON-RPC stream; HTTP responses are framed by the ASGI
transport, so normal logging is safe here. Do not "helpfully" re-add
logging.disable()/loguru silencing to this module.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from loguru import logger

from chunkhound.core.config.config import Config

from .base import _MCP_AVAILABLE, MCPServerBase
from .status import derive_daemon_status

if _MCP_AVAILABLE:
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
else:  # pragma: no cover - optional dependency path
    StreamableHTTPSessionManager = None  # type: ignore[assignment,misc]

if TYPE_CHECKING:  # type-checkers only; avoid runtime hard deps at import
    from starlette.applications import Starlette
    from starlette.types import Receive, Scope, Send


class _StreamableHttpEndpoint:
    """Adapts the session manager's ``handle_request`` as a raw ASGI app.

    Starlette's ``Route`` treats a bound method as a request/response
    function (via ``inspect.ismethod``), which would break SSE streaming.
    Wrapping it in a plain callable object keeps Route's raw-ASGI dispatch.
    Also: unlike ``Mount``, which only matches ``/mcp/*`` and 307-redirects
    a bare ``/mcp`` to ``/mcp/``, ``Route`` matches the exact ``/mcp`` path
    that MCP clients (and the issue's own curl example) are configured with.

    Session teardown: this class is a plain callable (not a function/method),
    so Starlette's ``Route`` never restricts it to specific HTTP methods —
    ``DELETE /mcp`` reaches ``handle_request`` exactly like ``GET``/``POST``.
    The MCP SDK's stateful ``StreamableHTTPSessionManager`` already handles
    ``DELETE`` itself: given a valid ``Mcp-Session-Id`` header, it closes that
    session's streams and marks it terminated, so subsequent requests with
    the same session ID get 404 instead of resuming it. No extra routing is
    needed here for a client to explicitly tear down its session. Note the
    SDK keeps the terminated transport's dict entry (not just its streams)
    for the life of the process — a server that serves many short-lived
    stateful clients will accumulate small terminated-session entries over
    time; this is an upstream SDK characteristic, not something this module
    works around.
    """

    def __init__(self, session_manager: Any):
        self._session_manager = session_manager

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self._session_manager.handle_request(scope, receive, send)


class _BearerAuthMiddleware:
    """Raw ASGI middleware enforcing bearer-token auth on the HTTP transport.

    Deliberately NOT Starlette's ``BaseHTTPMiddleware``, which buffers the
    entire response body and would break SSE streaming. ``/health`` is
    always exempt, and every request passes through unchecked when no token
    is configured.

    Registered as a Starlette ``Middleware`` entry (see ``_build_app``)
    ordered AFTER ``CORSMiddleware`` rather than wrapped outside the whole
    app: browsers send CORS preflight ``OPTIONS`` requests without an
    ``Authorization`` header, so if this ran before ``CORSMiddleware`` it
    would 401 every preflight (with no CORS headers on the response) and
    break browser-based clients whenever both ``--cors`` and
    ``--auth-token`` are set. ``CORSMiddleware`` answers preflight requests
    itself and never forwards them further in, so this middleware only ever
    sees real requests.
    """

    def __init__(self, app: Any, token: str | None):
        self.app = app
        self.token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or self.token is None
            or scope.get("path") == "/health"
        ):
            await self.app(scope, receive, send)
            return

        from starlette.responses import JSONResponse

        headers = dict(scope.get("headers") or [])
        auth_header = headers.get(b"authorization", b"").decode("latin-1")
        expected = f"Bearer {self.token}"
        if not secrets.compare_digest(auth_header, expected):
            response = JSONResponse(
                {"error": "Unauthorized"},
                status_code=401,
                headers={"WWW-Authenticate": 'Bearer realm="ChunkHound MCP"'},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


class HttpMCPServer(MCPServerBase):
    """MCP server implementation for the Streamable HTTP transport."""

    def __init__(self, config: Config, args: Any = None):
        """Initialize HTTP MCP server.

        Args:
            config: Validated configuration object
            args: Original CLI arguments for direct path access
        """
        super().__init__(config, args=args)

        self.host: str = config.mcp.host
        self.port: int = config.mcp.port
        self.auth_token: str | None = config.mcp.auth_token
        self.cors: bool = config.mcp.cors

        if _MCP_AVAILABLE:
            self._session_manager = StreamableHTTPSessionManager(
                app=self.server,  # type: ignore[arg-type]
                stateless=False,
            )
        else:
            self._session_manager = None  # type: ignore[assignment]

    def _register_tools(self) -> None:
        """Register tool handlers with the HTTP server."""
        self._register_common_tool_handlers()

    def _build_app(self) -> Any:
        """Build the Starlette ASGI app: health check, MCP endpoint, auth."""
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route

        async def health(request: Any) -> JSONResponse:
            # Deliberately unauthenticated (see _BearerAuthMiddleware) so
            # load balancers/readiness probes can reach it without a token.
            # derive_daemon_status can include scan_error/last_error text
            # (e.g. file paths from a failed scan) — an accepted tradeoff
            # for operational visibility, not an oversight.
            return JSONResponse(derive_daemon_status(self._scan_progress))

        from starlette.middleware import Middleware

        routes = [
            Route("/health", health, methods=["GET"]),
            Route("/mcp", _StreamableHttpEndpoint(self._session_manager)),
        ]

        # Order matters: CORSMiddleware must be OUTERMOST (listed first) so it
        # can answer preflight OPTIONS requests directly, before they ever
        # reach the bearer-auth middleware. See _BearerAuthMiddleware's
        # docstring for why the reverse order breaks browser clients.
        middleware = []
        if self.cors:
            from starlette.middleware.cors import CORSMiddleware

            # allow_credentials is deliberately omitted: this transport uses
            # bearer-token auth (sent explicitly via the Authorization header
            # by the client code), never cookies, so there is nothing for
            # "credentials" to carry. Wildcard allow_origins is also mutually
            # exclusive with allow_credentials=True per the CORS spec — do
            # not "helpfully" add it alongside allow_origins=["*"].
            middleware.append(
                Middleware(
                    CORSMiddleware,
                    allow_origins=["*"],
                    allow_methods=["*"],
                    allow_headers=["*"],
                )
            )
        middleware.append(Middleware(_BearerAuthMiddleware, token=self.auth_token))

        return Starlette(
            routes=routes,
            middleware=middleware,
            lifespan=self._lifespan,
        )

    @asynccontextmanager
    async def _lifespan(self, app: Starlette) -> AsyncIterator[None]:
        """Manage server lifecycle: initialize services, run session manager."""
        try:
            await self.initialize()
            self._initialization_complete.set()
            self.debug_log("HTTP server initialization complete")

            async with self._session_manager.run():
                yield
        finally:
            await self.cleanup()

    async def run(self) -> None:
        """Run the HTTP server via uvicorn."""
        if not _MCP_AVAILABLE:
            raise RuntimeError("HTTP MCP transport requires the 'mcp' package")

        import uvicorn

        app = self._build_app()
        uvicorn_config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="info" if self.debug_mode else "warning",
        )
        server = uvicorn.Server(uvicorn_config)
        # await server.serve() (not server.run()) — server.run() would try to
        # start a second event loop nested inside the one we're already
        # running under asyncio.run(). Uvicorn installs its own
        # SIGINT/SIGTERM/SIGBREAK handlers when run on the main thread.
        await server.serve()


async def main(args: Any = None, config: Config | None = None) -> None:
    """Main entry point for the MCP HTTP server.

    Args:
        args: Pre-parsed arguments. If None, will parse from sys.argv.
        config: Pre-validated configuration. If provided (e.g. by
            ``mcp_command``, which already validated it once), validation is
            not repeated; otherwise a fresh one is built and validated here.
    """
    import argparse

    from chunkhound.api.cli.utils.config_factory import create_validated_config
    from chunkhound.core.config.mcp_config import MCPConfig
    from chunkhound.mcp_server.common import add_common_mcp_arguments

    if args is None:
        # Direct invocation - parse arguments
        parser = argparse.ArgumentParser(
            description="ChunkHound MCP HTTP server",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        add_common_mcp_arguments(parser)
        MCPConfig.add_cli_arguments(parser)
        args = parser.parse_args()

    # Mark process as MCP mode so downstream code avoids interactive prompts
    os.environ["CHUNKHOUND_MCP_MODE"] = "1"

    if config is None:
        # Create and validate configuration
        config, validation_errors = create_validated_config(args, "mcp")

        if validation_errors:
            msg = "; ".join(str(e) for e in validation_errors)
            logger.error(f"ChunkHound MCP HTTP server configuration errors: {msg}")
            sys.exit(1)

    try:
        server = HttpMCPServer(config, args=args)
        await server.run()
    except Exception as e:
        logger.error(f"ChunkHound MCP HTTP server failed to start: {e}")
        sys.exit(1)


def main_sync() -> None:
    """Synchronous wrapper for CLI entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
