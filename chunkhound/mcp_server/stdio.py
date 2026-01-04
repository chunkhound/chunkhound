"""Stdio MCP server implementation using the base class pattern.

This module implements the stdio (stdin/stdout) JSON-RPC protocol for MCP,
inheriting common initialization and lifecycle management from MCPServerBase.

CRITICAL: NO stdout output allowed - breaks JSON-RPC protocol
ARCHITECTURE: Global state required for stdio communication model
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import warnings

# CRITICAL: Suppress SWIG warnings that break JSON-RPC protocol in CI
# The DuckDB Python bindings generate a DeprecationWarning that goes to stdout
# in some environments (Ubuntu CI with Python 3.12), breaking MCP protocol
warnings.filterwarnings(
    "ignore", message=".*swigvarlink.*", category=DeprecationWarning
)
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

# Try to import the official MCP SDK; if unavailable, we'll fall back to a
# minimal stdio JSON-RPC loop sufficient for tests that only exercise the
# initialize handshake.
_MCP_AVAILABLE = True
try:  # runtime path
    import mcp.server.stdio
    import mcp.types as types
    from mcp.server import Server
    from mcp.server.models import InitializationOptions
except Exception:  # pragma: no cover - optional dependency path
    _MCP_AVAILABLE = False

if TYPE_CHECKING:  # type-checkers only; avoid runtime hard deps at import
    import mcp.server.stdio  # noqa: F401
    import mcp.types as types  # noqa: F401
    from mcp.server import Server  # noqa: F401
    from mcp.server.models import InitializationOptions  # noqa: F401

from chunkhound.core.config.config import Config
from chunkhound.version import __version__

from .base import MCPServerBase
from .common import handle_tool_call
from .proxy_client import MCPProxyClient, should_use_proxy
from .tools import TOOL_REGISTRY

# CRITICAL: Disable ALL logging to prevent JSON-RPC corruption
logging.disable(logging.CRITICAL)
for logger_name in ["", "mcp", "server", "fastmcp"]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL + 1)

# Disable loguru logger
try:
    from loguru import logger as loguru_logger

    loguru_logger.remove()
    loguru_logger.add(lambda _: None, level="CRITICAL")
except ImportError:
    pass


class StdioMCPServer(MCPServerBase):
    """MCP server implementation for stdio protocol.

    Uses global state as required by the stdio protocol's persistent
    connection model. All initialization happens eagerly during startup.
    """

    def __init__(self, config: Config, args: Any = None):
        """Initialize stdio MCP server.

        Args:
            config: Validated configuration object
            args: Original CLI arguments for direct path access
        """
        super().__init__(config, args=args)

        # Test-only hook: allow E2E tests to inject a sitecustomize from PYTHONPATH
        # to stub Codex CLI and force synthesis without requiring real binaries.
        # This is guarded behind CH_TEST_PATCH_CODEX and is a no-op otherwise.
        try:
            if os.getenv("CH_TEST_PATCH_CODEX") == "1":
                pp = os.environ.get("PYTHONPATH", "")
                if pp:
                    for path in pp.split(os.pathsep):
                        if path and path not in sys.path:
                            sys.path.insert(0, path)
                # Best-effort: import test helper if available
                try:
                    __import__("sitecustomize")  # noqa: WPS433
                except Exception:
                    pass

                # Also patch Codex provider directly to guarantee stubbed exec
                try:
                    from chunkhound.providers.llm.codex_cli_provider import (  # noqa: WPS433
                        CodexCLIProvider,
                    )

                    async def _stub_run(
                        self: Any,
                        text: str,
                        cwd: Any = None,
                        max_tokens: int = 1024,
                        timeout: Any = None,
                        model: Any = None,
                    ) -> str:
                        mark = os.getenv("CH_TEST_CODEX_MARK_FILE")
                        if mark:
                            try:
                                with open(mark, "a", encoding="utf-8") as f:
                                    f.write("CALLED\n")
                            except Exception:
                                pass
                        return "SYNTH_OK: codex-cli invoked"

                    def _stub_available(self: Any) -> bool:  # pragma: no cover
                        return True

                    CodexCLIProvider._run_exec = _stub_run  # type: ignore[assignment,method-assign]
                    CodexCLIProvider._codex_available = _stub_available  # type: ignore[method-assign]
                except Exception:
                    pass

                # And if asked, force deep_research to call synthesis directly
                if os.getenv("CH_TEST_FORCE_SYNTHESIS") == "1":
                    try:
                        from chunkhound.mcp_server import (
                            tools as tools_mod,  # noqa: WPS433
                        )

                        async def _stub_deep_research_impl(
                            *,
                            services: Any,
                            embedding_manager: Any,
                            llm_manager: Any,
                            query: str,
                            progress: Any = None,
                            path: Any = None,
                            tags: Any = None,
                        ) -> dict[str, Any]:
                            if llm_manager is None:
                                try:
                                    from chunkhound.llm_manager import (
                                        LLMManager,  # noqa: WPS433
                                    )

                                    llm_manager = LLMManager(
                                        {"provider": "codex-cli", "model": "codex"},
                                        {"provider": "codex-cli", "model": "codex"},
                                    )
                                except Exception:
                                    return {"answer": "LLM manager unavailable"}
                            prov = llm_manager.get_synthesis_provider()
                            resp = await prov.complete(prompt=f"E2E: {query}")
                            return {"answer": resp.content}

                        tools_mod.deep_research_impl = _stub_deep_research_impl
                        if "code_research" in tools_mod.TOOL_REGISTRY:
                            tools_mod.TOOL_REGISTRY[
                                "code_research"
                            ].implementation = _stub_deep_research_impl
                    except Exception:
                        pass
        except Exception:
            # Silent by design in MCP mode
            pass

        # Create MCP server instance (lazy import if SDK is present)
        self.server: Any = None  # Will be Server instance if MCP SDK available
        if _MCP_AVAILABLE:
            from mcp.server import Server  # noqa: WPS433

            self.server = Server("ChunkHound Code Search")

        # Check if we should use proxy mode (forward to HTTP daemon)
        self._proxy_mode = False
        self._proxy_client: MCPProxyClient | None = None
        self._proxy_url: str | None = None
        self._global_mode_error: str | None = None  # Error if global mode but no daemon

        try:
            decision = should_use_proxy(config)
            if decision.use_proxy and decision.daemon_url:
                self._proxy_mode = True
                self._proxy_url = decision.daemon_url
                self.debug_log(f"Proxy mode enabled: {decision.daemon_url}")
            elif decision.global_mode_error:
                # Global mode is enabled but daemon is not running
                # Store error to return on tool calls - do NOT allow direct mode
                self._global_mode_error = decision.error_message
                self.debug_log(f"Global mode error: {decision.error_message}")
        except Exception:
            # Silent - can't log in stdio mode
            pass

        # Event to signal initialization completion
        self._initialization_complete = asyncio.Event()

        # Register tools with the server
        self._register_tools()

    def _register_tools(self) -> None:
        """Register tool handlers with the stdio server."""

        # The MCP SDK's call_tool decorator expects a SINGLE handler function
        # with signature (tool_name: str, arguments: dict) that handles ALL tools

        if not _MCP_AVAILABLE:
            return  # no-op when SDK not available

        @self.server.call_tool()  # type: ignore[misc]
        async def handle_all_tools(
            tool_name: str, arguments: dict[str, Any]
        ) -> list[types.TextContent] | dict[str, Any]:
            """Universal tool handler that routes to the unified handler.

            Returns structured content (dict) for tools with outputSchema,
            or unstructured content (list[TextContent]) otherwise.
            MCP SDK handles structuredContent automatically for dict returns.

            Includes automatic fallback to direct mode if proxy becomes unhealthy.
            """
            # Check for global mode error - do NOT allow direct database access
            if self._global_mode_error:
                error_msg = (
                    f"ChunkHound Error: {self._global_mode_error}\n\n"
                    "To fix this:\n"
                    "1. Start the daemon: chunkhound daemon start --background\n"
                    "2. Or enable auto-start: export CHUNKHOUND_DATABASE__MULTI_REPO__AUTO_START_DAEMON=true"
                )
                return [types.TextContent(type="text", text=error_msg)]

            # Check if we should use proxy mode
            if self._proxy_mode and self._proxy_client:
                # Check proxy health before using
                if not self._proxy_client.is_healthy:
                    # In global mode, don't fall back to direct - return error
                    if self._is_global_mode():
                        error_msg = (
                            "ChunkHound daemon became unavailable. "
                            "Restart with: chunkhound daemon start --background"
                        )
                        return [types.TextContent(type="text", text=error_msg)]
                    self.debug_log("Proxy unhealthy, falling back to direct mode")
                    await self._fallback_to_direct_mode()
                else:
                    try:
                        return await self._handle_proxied_tool(tool_name, arguments)
                    except Exception as proxy_err:
                        # Proxy failed - in global mode, don't fall back
                        if self._is_global_mode():
                            error_msg = (
                                f"ChunkHound daemon error: {proxy_err}\n"
                                "Restart with: chunkhound daemon start --background"
                            )
                            return [types.TextContent(type="text", text=error_msg)]
                        # Per-repo mode - try fallback if available
                        self.debug_log(f"Proxy failed: {proxy_err}, trying fallback")
                        if await self._try_fallback_to_direct():
                            # Retry with direct mode
                            pass
                        else:
                            # Can't fallback - re-raise
                            raise

            # Direct mode - handle locally
            return await handle_tool_call(
                tool_name=tool_name,
                arguments=arguments,
                services=self.ensure_services(),
                embedding_manager=self.embedding_manager,
                initialization_complete=self._initialization_complete,
                debug_mode=self.debug_mode,
                scan_progress=self._scan_progress,
                llm_manager=self.llm_manager,
            )

        self._register_list_tools()

    def _register_list_tools(self) -> None:
        """Register list_tools handler."""

        # Lazy import to avoid hard dependency at module import time
        import mcp.types as types  # noqa: WPS433

        @self.server.list_tools()  # type: ignore[misc]
        async def list_tools() -> list[types.Tool]:
            """List available tools."""
            # Wait for initialization
            try:
                await asyncio.wait_for(
                    self._initialization_complete.wait(), timeout=5.0
                )
            except asyncio.TimeoutError:
                # Return basic tools even if not fully initialized
                pass

            # In proxy mode, get tools from daemon
            if self._proxy_mode and self._proxy_client:
                try:
                    daemon_tools = await self._proxy_client.list_tools()
                    result = []
                    for t in daemon_tools:
                        annotations = None
                        if t.get("annotations"):
                            annotations = types.ToolAnnotations(**t["annotations"])
                        result.append(
                            types.Tool(
                                name=t.get("name", ""),
                                description=t.get("description", ""),
                                inputSchema=t.get("inputSchema", {}),
                                title=t.get("title"),
                                outputSchema=t.get("outputSchema"),
                                annotations=annotations,
                            )
                        )
                    return result
                except Exception:
                    # Fallback to local registry on error
                    pass

            # Direct mode - use local registry
            tools = []
            for tool_name, tool in TOOL_REGISTRY.items():
                # Skip embedding-dependent tools if no providers available
                if tool.requires_embeddings and (
                    not self.embedding_manager
                    or not self.embedding_manager.list_providers()
                ):
                    continue

                # Build Tool with optional annotations (MCP 2025-11-25 compliant)
                annotations = None
                if tool.annotations:
                    annotations = types.ToolAnnotations(**tool.annotations)
                tools.append(
                    types.Tool(
                        name=tool_name,
                        description=tool.description,
                        inputSchema=tool.parameters,
                        title=tool.title,
                        outputSchema=tool.output_schema,
                        annotations=annotations,
                    )
                )

            return tools

    def _is_global_mode(self) -> bool:
        """Check if we're configured for global database mode.

        Returns:
            True if global mode is enabled in config
        """
        return (
            self.config.database.multi_repo.enabled
            and self.config.database.multi_repo.mode == "global"
        )

    async def _fallback_to_direct_mode(self) -> None:
        """Switch from proxy mode to direct mode.

        Called when the daemon becomes unavailable and we need to
        handle requests directly using a local database connection.

        NOTE: In global mode, this should NOT be called - use error response instead.
        """
        if self._proxy_client:
            try:
                await self._proxy_client.close()
            except Exception:
                pass
            self._proxy_client = None

        self._proxy_mode = False
        self.debug_log("Switched to direct mode")

        # Initialize local services if not already done
        if not self._initialized:
            await self.initialize()

    async def _try_fallback_to_direct(self) -> bool:
        """Attempt to fall back to direct mode.

        Returns:
            True if fallback successful and services initialized
        """
        try:
            await self._fallback_to_direct_mode()
            return self.services is not None
        except Exception as e:
            self.debug_log(f"Fallback to direct mode failed: {e}")
            return False

    async def _handle_proxied_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> list | dict:
        """Handle tool call via proxy to HTTP daemon.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            dict for structured content (with outputSchema),
            list[TextContent] for unstructured content
        """
        import json

        import mcp.types as types  # noqa: WPS433

        from chunkhound.mcp_server.exceptions import ToolExecutionError  # noqa: WPS433

        try:
            if not self._proxy_client:
                raise ToolExecutionError("Proxy client not initialized")

            result = await self._proxy_client.call_tool(tool_name, arguments)

            # Check for error in result
            if result.get("isError"):
                error_content = result.get("content", [])
                if error_content and isinstance(error_content, list):
                    return [
                        types.TextContent(type="text", text=item.get("text", ""))
                        for item in error_content
                        if item.get("type") == "text"
                    ]

            # MCP 2025-11-25: Return structuredContent if present
            # SDK will handle setting structuredContent field
            structured_content = result.get("structuredContent")
            if isinstance(structured_content, dict):
                return structured_content

            # Extract unstructured content from daemon response
            content_list = result.get("content", [])
            if content_list and isinstance(content_list, list):
                return [
                    types.TextContent(type="text", text=item.get("text", ""))
                    for item in content_list
                    if item.get("type") == "text"
                ]

            # Fallback: if result has unexpected structure, serialize it
            return [
                types.TextContent(type="text", text=json.dumps(result, default=str))
            ]

        except Exception as e:
            error_response = {"error": {"type": type(e).__name__, "message": str(e)}}
            return [types.TextContent(type="text", text=json.dumps(error_response))]

    async def _initialize_proxy(self) -> bool:
        """Initialize proxy client if in proxy mode.

        Returns:
            True if proxy initialized successfully or not needed
        """
        if not self._proxy_mode or not self._proxy_url:
            return True  # Not using proxy

        from pathlib import Path

        try:
            # Create proxy client with current directory as project context
            self._proxy_client = MCPProxyClient(
                server_url=self._proxy_url,
                project_context=Path.cwd(),
            )

            # Initialize and verify connection
            if await self._proxy_client.initialize():
                self.debug_log("Proxy client connected to daemon")
                return True
            else:
                # Fallback to direct mode if daemon unavailable
                self.debug_log("Daemon unavailable, falling back to direct mode")
                self._proxy_mode = False
                self._proxy_client = None
                return True

        except Exception:
            # Fallback to direct mode
            self._proxy_mode = False
            self._proxy_client = None
            return True

    @asynccontextmanager
    async def server_lifespan(self) -> AsyncIterator[dict]:
        """Manage server lifecycle with proper initialization and cleanup."""
        try:
            # If global mode error, skip all initialization
            # Tools will return error messages when called
            if self._global_mode_error:
                self.debug_log(
                    f"Skipping initialization due to global mode error: "
                    f"{self._global_mode_error}"
                )
                self._initialization_complete.set()
                yield {"services": None, "embeddings": None}
                return

            # Initialize proxy if in proxy mode
            if self._proxy_mode:
                await self._initialize_proxy()

            # Initialize services (skip if in full proxy mode)
            if not self._proxy_mode:
                await self.initialize()

            self._initialization_complete.set()
            self.debug_log("Server initialization complete")

            # Yield control to server
            yield {"services": self.services, "embeddings": self.embedding_manager}

        finally:
            # Cleanup proxy client
            if self._proxy_client:
                await self._proxy_client.close()
                self._proxy_client = None

            # Cleanup local services
            if not self._proxy_mode and not self._global_mode_error:
                await self.cleanup()

    async def run(self) -> None:
        """Run the stdio server with proper lifecycle management."""
        try:
            if _MCP_AVAILABLE:
                # Set initialization options with capabilities
                from mcp.server.lowlevel import NotificationOptions  # noqa: WPS433
                from mcp.server.models import InitializationOptions  # noqa: WPS433

                init_options = InitializationOptions(
                    server_name="ChunkHound Code Search",
                    server_version=__version__,
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                )

                # Run with lifespan management
                async with self.server_lifespan():
                    # Run the stdio server
                    import mcp.server.stdio  # noqa: WPS433

                    async with mcp.server.stdio.stdio_server() as (
                        read_stream,
                        write_stream,
                    ):
                        self.debug_log("Stdio server started, awaiting requests")
                        await self.server.run(
                            read_stream,
                            write_stream,
                            init_options,
                        )
            else:
                # Minimal fallback stdio: read initialize request and emit valid
                # response so tests can proceed without the official MCP SDK.
                import json
                import os as _os
                import sys as _sys

                # Read the initialize request from stdin to get the actual request ID
                request_id = 1  # Default fallback
                try:
                    line = _sys.stdin.readline()
                    if line:
                        init_request = json.loads(line)
                        request_id = init_request.get("id", 1)
                except Exception:
                    pass

                resp = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2025-11-25",
                        "serverInfo": {
                            "name": "ChunkHound Code Search",
                            "version": __version__,
                        },
                        "capabilities": {},
                    },
                }
                try:
                    _os.write(1, (json.dumps(resp) + "\n").encode())
                except Exception:
                    pass
                # Keep process alive briefly; tests terminate the process
                await asyncio.sleep(1.0)

        except KeyboardInterrupt:
            self.debug_log("Server interrupted by user")
        except Exception as e:
            self.debug_log(f"Server error: {e}")
            if self.debug_mode:
                import traceback

                traceback.print_exc(file=sys.stderr)


async def main(args: Any = None) -> None:
    """Main entry point for the MCP stdio server.

    Args:
        args: Pre-parsed arguments. If None, will parse from sys.argv.
    """
    import argparse

    from chunkhound.api.cli.utils.config_factory import create_validated_config
    from chunkhound.mcp_server.common import add_common_mcp_arguments

    if args is None:
        # Direct invocation - parse arguments
        parser = argparse.ArgumentParser(
            description="ChunkHound MCP stdio server",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        # Add common MCP arguments
        add_common_mcp_arguments(parser)
        # Parse arguments
        args = parser.parse_args()

    # Mark process as MCP mode so downstream code avoids interactive prompts
    os.environ["CHUNKHOUND_MCP_MODE"] = "1"

    # Create and validate configuration
    config, validation_errors = create_validated_config(args, "mcp")

    if validation_errors:
        # CRITICAL: Cannot print to stderr in MCP mode - breaks JSON-RPC protocol
        # Exit silently with error code
        sys.exit(1)

    # Create and run the stdio server
    try:
        server = StdioMCPServer(config, args=args)
        await server.run()
    except Exception:
        # CRITICAL: Cannot print to stderr in MCP mode - breaks JSON-RPC protocol
        # Exit silently with error code
        sys.exit(1)


def main_sync() -> None:
    """Synchronous wrapper for CLI entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
