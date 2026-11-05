"""HTTP MCP server implementation using the base class pattern.

Implements the Streamable HTTP transport for MCP. Gets a fully wired
``StreamableHTTPSessionManager`` from ``MCPServerBase.build_streamable_http_app()``,
sharing tool dispatch and service lifecycle with the other transports. This module just adds the
Starlette app, health endpoint, bearer auth, and CORS -- no direct ``mcp``
import needed.

NOTE: Unlike stdio.py, this module does NOT silence logging/loguru. Stdio
needs silence to keep stdout clean for JSON-RPC; HTTP is ASGI-framed, so
normal logging is safe. Don't re-add silencing here.
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

if TYPE_CHECKING:  # avoid hard runtime deps
    from starlette.applications import Starlette
    from starlette.types import Receive, Scope, Send


class _StreamableHttpEndpoint:
    """Adapts the session manager's ``handle_request`` as a raw ASGI app.

    A plain callable object, not a bound method, so Starlette's ``Route``
    keeps raw-ASGI dispatch (needed for SSE) and doesn't restrict it to
    specific HTTP methods. ``Route`` (not ``Mount``) is used so ``/mcp``
    matches exactly, no ``/mcp/`` redirect.

    ``DELETE /mcp`` reaches ``handle_request`` like any other method, and the
    SDK's session manager already handles session teardown for it. Note: the
    SDK keeps terminated sessions' dict entries for the process lifetime
    (upstream characteristic, not handled here).
    """

    def __init__(self, session_manager: Any):
        self._session_manager = session_manager

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self._session_manager.handle_request(scope, receive, send)


class _BearerAuthMiddleware:
    """Raw ASGI middleware enforcing bearer-token auth on the HTTP transport.

    Not Starlette's ``BaseHTTPMiddleware`` (buffers the body, breaks SSE).
    ``/health`` is always exempt; requests pass through unchecked when no
    token is configured.

    Registered after ``CORSMiddleware`` so CORS preflight ``OPTIONS``
    requests (sent without an ``Authorization`` header) get answered before
    reaching auth -- otherwise browser clients would get 401'd on preflight.
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

        self._session_manager = (
            self.build_streamable_http_app(stateless=False) if _MCP_AVAILABLE else None
        )

    def _register_tools(self) -> None:
        """Register tool handlers with the HTTP server."""
        self._register_common_tool_handlers()

    def _build_app(self) -> Any:
        """Build the Starlette ASGI app: health check, MCP endpoint, auth."""
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route

        async def health(request: Any) -> JSONResponse:
            # Unauthenticated so probes/load balancers can reach it (see
            # _BearerAuthMiddleware). May include scan_error/last_error text.
            return JSONResponse(derive_daemon_status(self._current_scan_progress()))

        from starlette.middleware import Middleware

        routes = [
            Route("/health", health, methods=["GET"]),
            Route("/mcp", _StreamableHttpEndpoint(self._session_manager)),
        ]

        # CORSMiddleware must come first/outermost so it answers preflight
        # OPTIONS before bearer-auth sees it (see _BearerAuthMiddleware).
        middleware = []
        if self.cors:
            from starlette.middleware.cors import CORSMiddleware

            # allow_credentials omitted: auth is a bearer header, not
            # cookies, and it's mutually exclusive with allow_origins=["*"].
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

            # Only None when _MCP_AVAILABLE is False, in which case run()
            # raises before _lifespan is reached (see run() below).
            async with self._session_manager.run():  # type: ignore[union-attr]
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
        # serve(), not run() — run() would nest a second event loop inside
        # the one started by asyncio.run().
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
        # parse CLI args
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
        config, validation_errors = await create_validated_config(args, "mcp")

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
