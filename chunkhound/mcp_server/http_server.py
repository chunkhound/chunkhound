"""HTTP MCP server implementation using the base class pattern.

This module implements the HTTP transport for MCP using FastMCP,
inheriting common initialization and lifecycle management from MCPServerBase.

Unlike stdio, HTTP servers can use lazy initialization and don't require
global state management.
"""

from typing import Any
import os

# Lazy import FastMCP at runtime to avoid hard import dependency at module import time
FastMCP = None  # type: ignore
from chunkhound.core.config.config import Config

from .base import MCPServerBase
from .common import parse_mcp_arguments
from .tools import TOOL_REGISTRY, execute_tool


class HttpMCPServer(MCPServerBase):
    """MCP server implementation for HTTP protocol using FastMCP.

    Uses lazy initialization pattern - services are initialized on first
    request rather than at startup. This is more suitable for HTTP's
    request/response model.
    """

    def __init__(self, config: Config, port: int = 5173, host: str = "0.0.0.0"):
        """Initialize HTTP MCP server.

        Args:
            config: Validated configuration object
            port: Port to listen on (default: 5173)
            host: Host to bind to (default: "0.0.0.0")
        """
        super().__init__(config)
        self.port = port
        self.host = host

        # Create FastMCP instance lazily to avoid import errors during smoke import
        global FastMCP  # noqa: PLW0603
        if FastMCP is None:  # type: ignore
            from fastmcp import FastMCP as _FastMCP  # noqa: WPS433
            FastMCP = _FastMCP  # type: ignore
        self.app: Any = FastMCP("ChunkHound Code Search")  # type: ignore

        # Register tools with the server
        self._register_tools()

    def _register_tools(self) -> None:
        """Register all tools from the registry with FastMCP.

        Uses TOOL_REGISTRY for descriptions to ensure consistency with stdio mode.
        """
        # FastMCP requires explicit function signatures, not **kwargs
        # So we create specific handler functions for each tool

        @self.app.tool(description=TOOL_REGISTRY["get_stats"].description)
        async def get_stats() -> dict[str, Any]:
            await self.initialize()
            return await execute_tool(
                tool_name="get_stats",
                services=self.ensure_services(),
                embedding_manager=self.embedding_manager,
                arguments={},
                scan_progress=self._scan_progress,
                llm_manager=self.llm_manager,
            )

        @self.app.tool(description=TOOL_REGISTRY["health_check"].description)
        async def health_check() -> dict[str, Any]:
            await self.initialize()
            return await execute_tool(
                tool_name="health_check",
                services=self.ensure_services(),
                embedding_manager=self.embedding_manager,
                arguments={},
                llm_manager=self.llm_manager,
            )

        @self.app.tool(description=TOOL_REGISTRY["search_regex"].description)
        async def search_regex(
            pattern: str,
            page_size: int = 10,
            offset: int = 0,
            max_response_tokens: int = 20000,
            path: str | None = None,
        ) -> dict[str, Any]:
            await self.initialize()

            # Build arguments dict
            args = {
                "pattern": pattern,
                "page_size": page_size,
                "offset": offset,
                "max_response_tokens": max_response_tokens,
            }
            if path is not None:
                args["path"] = path

            return await execute_tool(
                tool_name="search_regex",
                services=self.ensure_services(),
                embedding_manager=self.embedding_manager,
                arguments=parse_mcp_arguments(args),
                llm_manager=self.llm_manager,
            )

        @self.app.tool(description=TOOL_REGISTRY["search_semantic"].description)
        async def search_semantic(
            query: str,
            page_size: int = 10,
            offset: int = 0,
            max_response_tokens: int = 20000,
            path: str | None = None,
            provider: str | None = None,
            model: str | None = None,
            threshold: float | None = None,
        ) -> dict[str, Any]:
            await self.initialize()

            # Build arguments dict
            args = {
                "query": query,
                "page_size": page_size,
                "offset": offset,
                "max_response_tokens": max_response_tokens,
            }
            if path is not None:
                args["path"] = path
            if provider is not None:
                args["provider"] = provider
            if model is not None:
                args["model"] = model
            if threshold is not None:
                args["threshold"] = threshold

            return await execute_tool(
                tool_name="search_semantic",
                services=self.ensure_services(),
                embedding_manager=self.embedding_manager,
                arguments=parse_mcp_arguments(args),
                llm_manager=self.llm_manager,
            )

        @self.app.tool(description=TOOL_REGISTRY["code_research"].description)
        async def code_research(query: str, path: str | None = None) -> dict[str, Any]:
            await self.initialize()

            # Build arguments dict (path is optional scope prefix)
            args: dict[str, Any] = {"query": query}
            if path is not None:
                args["path"] = path

            result = await execute_tool(
                tool_name="code_research",
                services=self.ensure_services(),
                embedding_manager=self.embedding_manager,
                arguments=parse_mcp_arguments(args),
                llm_manager=self.llm_manager,
            )
            # code_research returns raw markdown string, wrap it for FastMCP
            if isinstance(result, str):
                return {"answer": result}
            return dict(result) if hasattr(result, "__dict__") else result

        # Patch tool schemas after all decorators have executed
        # This replaces FastMCP's auto-generated schemas with TOOL_REGISTRY schemas
        self._patch_tool_schema("get_stats")
        self._patch_tool_schema("health_check")
        self._patch_tool_schema("search_regex")
        self._patch_tool_schema("search_semantic")
        self._patch_tool_schema("code_research")

    def _patch_tool_schema(self, tool_name: str) -> None:
        """Replace FastMCP's auto-generated schema with TOOL_REGISTRY schema.

        Args:
            tool_name: Name of tool to patch
        """
        try:
            tool = self.app._tool_manager._tools[tool_name]
            tool.parameters = TOOL_REGISTRY[tool_name].parameters
        except (AttributeError, KeyError) as e:
            self.debug_log(f"Warning: Could not patch schema for {tool_name}: {e}")

    async def run(self) -> None:
        """Run the HTTP server.

        FastMCP handles most of the lifecycle management, so this is simpler
        than the stdio implementation.
        """
        try:
            self.debug_log(f"Starting HTTP server on {self.host}:{self.port}")

            # Run the FastMCP server in HTTP mode
            await self.app.run_http_async(port=self.port, host=self.host)

        except KeyboardInterrupt:
            self.debug_log("Server interrupted by user")
        except Exception as e:
            self.debug_log(f"Server error: {e}")
            if self.debug_mode:
                import traceback

                traceback.print_exc()
        finally:
            # Cleanup resources
            await self.cleanup()
            self.debug_log("Server shutdown complete")


async def main() -> None:
    """Main entry point for HTTP server"""
    import argparse
    import sys

    from chunkhound.api.cli.utils.config_factory import create_validated_config
    from chunkhound.mcp_server.common import add_common_mcp_arguments

    parser = argparse.ArgumentParser(
        description="ChunkHound MCP HTTP server (FastMCP 2.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Add common MCP arguments
    add_common_mcp_arguments(parser)

    # HTTP-specific arguments
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5173, help="Port to bind to")

    args = parser.parse_args()

    # Mark process as MCP mode so downstream code avoids interactive prompts
    os.environ["CHUNKHOUND_MCP_MODE"] = "1"

    # Create and validate configuration
    config, validation_errors = create_validated_config(args, "mcp")

    if validation_errors:
        for error in validation_errors:
            print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

    # Create and run the HTTP server
    server = HttpMCPServer(config, port=args.port, host=args.host)
    await server.run()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
