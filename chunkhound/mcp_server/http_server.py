"""HTTP MCP server implementation for global multi-repository mode.

This module implements the HTTP/SSE transport for MCP, enabling:
- Multiple concurrent client connections
- Global database mode with cross-project search
- Centralized file watching for all indexed projects
- Health checks and management endpoints

The HTTP server is designed to run as a daemon, owning the database write lock
and handling all file monitoring and indexing operations centrally.

Architecture:
    HTTP Server (daemon)
    ├── ProjectRegistry: tracks all indexed projects
    ├── WatcherManager: monitors file changes across all projects
    ├── IndexingCoordinator: processes file changes
    └── MCP Tools: serves search queries to clients

Usage:
    # Start daemon
    chunkhound daemon start --port 5173

    # Or run directly
    chunkhound mcp http --port 5173
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

# MCP SDK imports
try:
    import mcp.types as types
    from mcp.server import Server
    from mcp.server.lowlevel import NotificationOptions
    from mcp.server.models import InitializationOptions
    from mcp.server.sse import SseServerTransport

    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False

# Starlette for HTTP server
try:
    import uvicorn
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.middleware.cors import CORSMiddleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Route

    _HTTP_AVAILABLE = True
except ImportError:
    _HTTP_AVAILABLE = False

if TYPE_CHECKING:
    from chunkhound.services.index_job_manager import IndexJobManager
    from chunkhound.services.project_registry import ProjectRegistry
    from chunkhound.services.watcher_manager import WatcherManager

from chunkhound.core.config.config import Config
from chunkhound.version import __version__

from .base import MCPServerBase
from .common import (
    MAX_REQUEST_BODY_SIZE,
    SUPPORTED_PROTOCOL_VERSIONS,
    handle_tool_call,
)
from .tools import TOOL_REGISTRY

# Allowed hostnames for local-only daemon
_ALLOWED_HOSTS = {"localhost", "127.0.0.1"}
_ALLOWED_SCHEMES = {"http", "https"}


def _validate_origin(request: Request) -> JSONResponse | None:
    """Validate Origin header if present.

    Returns JSONResponse with 403 if Origin is invalid, None if OK.

    Uses proper URL parsing to prevent bypass via crafted origins like
    'http://127.0.0.1:1234.attacker.com' which would pass prefix matching.
    """
    origin = request.headers.get("Origin")
    if origin is None:
        return None  # No Origin header is OK

    # Reject empty origin
    if not origin:
        return JSONResponse(
            {"error": "Invalid origin"},
            status_code=400,
        )

    try:
        from urllib.parse import urlparse

        parsed = urlparse(origin)

        # Validate scheme
        if parsed.scheme not in _ALLOWED_SCHEMES:
            return JSONResponse(
                {"error": "Origin not allowed"},
                status_code=403,
            )

        # Validate hostname (parsed.hostname handles port stripping)
        if parsed.hostname not in _ALLOWED_HOSTS:
            return JSONResponse(
                {"error": "Origin not allowed"},
                status_code=403,
            )

        # Validate port is numeric (catches 'http://127.0.0.1:8080.attacker.com')
        # Accessing .port raises ValueError if port contains non-numeric chars
        _ = parsed.port

        return None

    except ValueError:
        # Invalid port (e.g., '8080.attacker.com')
        return JSONResponse(
            {"error": "Invalid origin"},
            status_code=400,
        )
    except Exception:
        # Other malformed origin
        return JSONResponse(
            {"error": "Invalid origin"},
            status_code=400,
        )


def _validate_protocol_version(request: Request) -> JSONResponse | None:
    """Validate MCP-Protocol-Version header if present.

    Returns JSONResponse with 400 if version unsupported, None if OK.
    """
    version = request.headers.get("MCP-Protocol-Version")
    if version is None:
        return None  # No version header is OK (backwards compatibility)

    if version not in SUPPORTED_PROTOCOL_VERSIONS:
        return JSONResponse(
            {"error": f"Unsupported MCP protocol version: {version}"},
            status_code=400,
        )
    return None


def _validate_body_size(request: Request) -> JSONResponse | None:
    """Validate request body size to prevent memory exhaustion.

    Returns JSONResponse with 413 if body too large, None if OK.
    """
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            size = int(content_length)
            if size > MAX_REQUEST_BODY_SIZE:
                return JSONResponse(
                    {
                        "error": f"Request body too large: {size} bytes "
                        f"(max {MAX_REQUEST_BODY_SIZE})"
                    },
                    status_code=413,
                )
        except ValueError:
            pass  # Invalid content-length, let request.json() handle it
    return None


class HTTPMCPServer(MCPServerBase):
    """MCP server implementation for HTTP/SSE transport.

    Designed for global database mode where multiple clients connect to
    a single server instance that owns file watching and indexing.
    """

    # Default values (used when no config or explicit args provided)
    DEFAULT_HOST = "127.0.0.1"
    DEFAULT_PORT = 5173

    def __init__(
        self,
        config: Config,
        host: str | None = None,
        port: int | None = None,
        args: Any = None,
    ):
        """Initialize HTTP MCP server.

        Args:
            config: Validated configuration object
            host: Host to bind to (default from config or 127.0.0.1)
            port: Port to listen on (default from config or 5173)
            args: Original CLI arguments
        """
        super().__init__(config, args=args)

        # Use explicit args > config > defaults
        config_host = config.database.multi_repo.daemon_host
        config_port = config.database.multi_repo.daemon_port
        self.host = host or config_host
        self.port = port or config_port

        # Server components
        self.app: Starlette | None = None
        self.sse_transport: SseServerTransport | None = None

        # Multi-repo components (initialized in start())
        self.project_registry: ProjectRegistry | None = None
        self.watcher_manager: WatcherManager | None = None
        self.index_job_manager: IndexJobManager | None = None

        # MCP server instance
        if _MCP_AVAILABLE:
            self.server: Server = Server("ChunkHound Code Search")
        else:
            self.server = None  # type: ignore

        # Event to signal initialization completion
        self._initialization_complete = asyncio.Event()

        # Server state
        self._started_at: float | None = None
        self._shutdown_event = asyncio.Event()

        # Active client sessions (protected by lock for async safety)
        self._active_sessions: dict[str, dict[str, Any]] = {}
        self._sessions_lock = asyncio.Lock()

        # Session cleanup configuration
        self._session_max_age_seconds = 24 * 60 * 60  # 24 hours
        self._session_cleanup_interval = 60 * 60  # 1 hour
        self._session_cleanup_task: asyncio.Task | None = None

        # Register MCP tools
        self._register_tools()

    def _register_tools(self) -> None:
        """Register tool handlers with the MCP server."""
        if not _MCP_AVAILABLE or self.server is None:
            return

        @self.server.call_tool()
        async def handle_all_tools(
            tool_name: str, arguments: dict[str, Any]
        ) -> list[types.TextContent] | dict[str, Any]:
            """Universal tool handler that routes to the unified handler.

            Returns structured content (dict) for tools with outputSchema,
            or unstructured content (list[TextContent]) otherwise.
            """
            return await handle_tool_call(
                tool_name=tool_name,
                arguments=arguments,
                services=self.ensure_services(),
                embedding_manager=self.embedding_manager,
                initialization_complete=self._initialization_complete,
                debug_mode=self.debug_mode,
                scan_progress=self._scan_progress,
                llm_manager=self.llm_manager,
                # Pass project registry for multi-project search resolution
                project_registry=self.project_registry,
            )

        @self.server.list_tools()
        async def list_tools() -> list[types.Tool]:
            """List available tools."""
            try:
                await asyncio.wait_for(
                    self._initialization_complete.wait(), timeout=5.0
                )
            except asyncio.TimeoutError:
                pass

            tools = []
            for tool_name, tool in TOOL_REGISTRY.items():
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

    async def _initialize_multi_repo(self) -> None:
        """Initialize multi-repository support components."""
        if not self.config.database.multi_repo.enabled:
            self.debug_log("Multi-repo mode not enabled, skipping")
            return

        if self.config.database.multi_repo.mode != "global":
            self.debug_log("Not in global mode, skipping multi-repo init")
            return

        self.debug_log("Initializing multi-repository support...")

        # Import here to avoid circular imports
        from chunkhound.services.index_job_manager import IndexJobManager
        from chunkhound.services.project_registry import ProjectRegistry
        from chunkhound.services.watcher_manager import WatcherManager

        # Initialize project registry
        if self.services and self.services.provider:
            self.project_registry = ProjectRegistry(self.services.provider)

            # Initialize watcher manager with config
            self.watcher_manager = WatcherManager(
                indexing_coordinator=self.services.indexing_coordinator,
                debug_sink=self.debug_log,
                config=self.config,
            )

            # Connect registry and watcher manager
            self.project_registry.set_watcher_manager(self.watcher_manager)
            self.watcher_manager.set_project_registry(self.project_registry)

            # Start watcher manager background processor
            await self.watcher_manager.start()

            # Start watchers for all indexed projects
            results = self.project_registry.start_all_watchers()
            for name, success in results.items():
                status = "started" if success else "FAILED"
                self.debug_log(f"Watcher for {name}: {status}")

            # Initialize index job manager for async indexing
            self.index_job_manager = IndexJobManager(
                indexing_coordinator=self.services.indexing_coordinator,
                config=self.config,
                project_registry=self.project_registry,
                watcher_manager=self.watcher_manager,
                debug_sink=self.debug_log,
            )
            await self.index_job_manager.start()

            count = self.project_registry.get_project_count()
            self.debug_log(f"Multi-repo initialized: {count} projects")

    async def _handle_sse(self, request: Request) -> Response:
        """Handle SSE connection for MCP protocol."""
        if not _MCP_AVAILABLE or self.server is None:
            return JSONResponse(
                {"error": "MCP SDK not available"},
                status_code=503,
            )

        # Extract project context from header
        project_context = request.headers.get("X-ChunkHound-Project")
        client_id = request.headers.get("X-Client-ID", str(id(request)))

        self.debug_log(
            f"SSE connection from client {client_id}, project: {project_context}"
        )

        try:
            # Create SSE transport for this connection
            async with SseServerTransport("/messages").connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                # Track session only after successful connection setup
                async with self._sessions_lock:
                    self._active_sessions[client_id] = {
                        "project": project_context,
                        "connected_at": time.time(),
                    }
                try:
                    read_stream, write_stream = streams

                    init_options = InitializationOptions(
                        server_name="ChunkHound Code Search",
                        server_version=__version__,
                        capabilities=self.server.get_capabilities(
                            notification_options=NotificationOptions(),
                            experimental_capabilities={},
                        ),
                    )

                    await self.server.run(
                        read_stream,
                        write_stream,
                        init_options,
                    )
                finally:
                    # Clean up session
                    async with self._sessions_lock:
                        self._active_sessions.pop(client_id, None)
        finally:
            self.debug_log(f"SSE connection closed for client {client_id}")

        return Response()

    async def _handle_messages(self, request: Request) -> Response:
        """Handle MCP message POST requests."""
        if not _MCP_AVAILABLE:
            return JSONResponse(
                {"error": "MCP SDK not available"},
                status_code=503,
            )

        # SSE transport handles messages via the connection
        return JSONResponse(
            {"error": "Use SSE endpoint for MCP messages"},
            status_code=400,
        )

    async def _handle_health(self, request: Request) -> JSONResponse:
        """Handle health check endpoint."""
        uptime = time.time() - self._started_at if self._started_at else 0

        # Get project stats
        projects_count = 0
        watchers_active = 0
        pending_events = 0
        if self.project_registry:
            projects_count = self.project_registry.get_project_count()
        if self.watcher_manager:
            watchers_active = len(self.watcher_manager.get_all_status())
            pending_events = self.watcher_manager.get_pending_count()

        # Get database stats
        db_stats = {}
        if self.services and self.services.provider:
            try:
                stats = self.services.provider.get_stats()
                db_stats = {
                    "files": stats.get("files", 0),
                    "chunks": stats.get("chunks", 0),
                    "embeddings": stats.get("embeddings", 0),
                }
            except Exception:
                pass

        # Get memory usage
        memory_mb = 0.0
        try:
            import psutil

            process = psutil.Process()
            memory_mb = round(process.memory_info().rss / 1024 / 1024, 2)
        except ImportError:
            pass
        except Exception:
            pass

        # Get indexing job stats
        jobs_active = 0
        jobs_info: list[dict] = []
        if self.index_job_manager:
            active_jobs = self.index_job_manager.get_active_jobs()
            jobs_active = len(active_jobs)
            jobs_info = [
                {
                    "job_id": j.job_id,
                    "project": j.project_name or str(j.project_path),
                    "status": j.status.value,
                    "phase": j.phase,
                }
                for j in active_jobs
            ]

        return JSONResponse(
            {
                "status": "healthy",
                "version": __version__,
                "uptime_seconds": round(uptime, 2),
                "projects_indexed": projects_count,
                "watchers_active": watchers_active,
                "pending_events": pending_events,
                "memory_mb": memory_mb,
                "active_sessions": len(self._active_sessions),
                "database": db_stats,
                "scan_progress": self._scan_progress,
                "indexing_jobs": {
                    "active": jobs_active,
                    "jobs": jobs_info,
                },
            }
        )

    async def _handle_stats(self, request: Request) -> JSONResponse:
        """Handle database stats endpoint.

        Query parameters:
            path: Optional path prefix to filter stats by project

        Returns:
            Database statistics (files, chunks, embeddings) optionally filtered by path
        """
        if not self.services or not self.services.provider:
            return JSONResponse(
                {"error": "Server not initialized"},
                status_code=503,
            )

        path = request.query_params.get("path")

        try:
            if path:
                # Get stats for specific project path
                stats = self.services.provider.get_stats_for_path(path)
            else:
                # Get global stats
                stats = self.services.provider.get_stats()

            return JSONResponse(
                {
                    "files": stats.get("files", 0),
                    "chunks": stats.get("chunks", 0),
                    "embeddings": stats.get("embeddings", 0),
                    "path": path,
                }
            )

        except Exception as e:
            self.debug_log(f"Failed to get stats: {e}")
            return JSONResponse(
                {"error": f"Failed to get stats: {e}"},
                status_code=500,
            )

    async def _handle_projects_list(self, request: Request) -> JSONResponse:
        """Handle projects list endpoint."""
        if not self.project_registry:
            return JSONResponse(
                {"error": "Multi-repo mode not enabled"},
                status_code=400,
            )

        projects = self.project_registry.list_projects()
        return JSONResponse(
            {
                "projects": [p.to_dict() for p in projects],
            }
        )

    async def _handle_project_index(self, request: Request) -> JSONResponse:
        """Handle project index request.

        Supports two modes:
        - Synchronous (default): Blocks until indexing completes
        - Async (async=true): Returns job ID immediately, indexes in background
        """
        # Validate Origin header (security: prevents DNS rebinding attacks)
        origin_error = _validate_origin(request)
        if origin_error:
            return origin_error

        if not self.services:
            return JSONResponse(
                {"error": "Server not initialized"},
                status_code=503,
            )

        # Check body size before parsing
        if error := _validate_body_size(request):
            return error

        try:
            body = await request.json()
            path = body.get("path")
            name = body.get("name")
            tags = body.get("tags", [])
            async_mode = body.get("async", False)

            if not path:
                return JSONResponse(
                    {"error": "path is required"},
                    status_code=400,
                )

            project_path = Path(path).resolve()

            if not project_path.exists():
                return JSONResponse(
                    {"error": f"Path does not exist: {path}"},
                    status_code=400,
                )

            # Async mode: create background job and return immediately
            if async_mode:
                if not self.index_job_manager:
                    return JSONResponse(
                        {
                            "error": "Async indexing not available (multi-repo mode required)"
                        },
                        status_code=400,
                    )

                try:
                    job = await self.index_job_manager.create_job(
                        project_path, project_name=name, tags=tags
                    )
                    return JSONResponse(
                        {
                            "status": "accepted",
                            "job_id": job.job_id,
                            "path": str(project_path),
                            "message": "Indexing started in background",
                        },
                        status_code=202,
                    )
                except ValueError as e:
                    # Path validation errors (subfolder conflicts, etc.)
                    return JSONResponse(
                        {"error": str(e)},
                        status_code=400,  # Bad Request
                    )
                except RuntimeError as e:
                    return JSONResponse(
                        {"error": str(e)},
                        status_code=429,  # Too Many Requests
                    )

            # Synchronous mode: block until complete
            self.debug_log(f"Starting synchronous index for: {project_path}")

            # Check for nested path conflicts BEFORE starting indexing
            if self.project_registry:
                try:
                    self.project_registry.validate_path_not_nested(project_path)
                except ValueError as e:
                    return JSONResponse(
                        {"error": str(e)},
                        status_code=400,
                    )

            # Get file patterns from config
            include_patterns = list(self.config.indexing.include)
            exclude_patterns = list(self.config.indexing.exclude)

            # Index the directory
            result = await self.services.indexing_coordinator.process_directory(
                project_path,
                patterns=include_patterns,
                exclude_patterns=exclude_patterns,
            )

            # Generate embeddings for new chunks
            if result.get("status") != "error":
                embed_result = await self.services.indexing_coordinator.generate_missing_embeddings()
                result["embeddings_generated"] = embed_result.get("generated", 0)

            # Register project if in multi-repo mode
            if self.project_registry:
                project = self.project_registry.register_project(
                    project_path, name=name
                )

                # Apply tags if specified
                if tags:
                    self.project_registry.set_project_tags(str(project_path), tags)

                # Update file count stats
                if hasattr(self.services.provider, "update_indexed_root_stats"):
                    self.services.provider.update_indexed_root_stats(str(project_path))

                # Start watcher
                if self.watcher_manager:
                    self.watcher_manager.start_watcher(project)

            return JSONResponse(
                {
                    "status": "success",
                    "path": str(project_path),
                    "files_processed": result.get("files_processed", 0),
                }
            )

        except Exception as e:
            logger.exception("Index request failed")
            return JSONResponse(
                {"error": str(e)},
                status_code=500,
            )

    async def _handle_job_status(self, request: Request) -> JSONResponse:
        """Handle job status request."""
        origin_error = _validate_origin(request)
        if origin_error:
            return origin_error

        if not self.index_job_manager:
            return JSONResponse(
                {"error": "Job manager not available"},
                status_code=400,
            )

        job_id = request.path_params.get("job_id")
        if not job_id:
            return JSONResponse(
                {"error": "job_id is required"},
                status_code=400,
            )

        job = self.index_job_manager.get_job(job_id)
        if not job:
            return JSONResponse(
                {"error": f"Job not found: {job_id}"},
                status_code=404,
            )

        return JSONResponse(job.to_dict())

    async def _handle_jobs_list(self, request: Request) -> JSONResponse:
        """Handle jobs list request."""
        origin_error = _validate_origin(request)
        if origin_error:
            return origin_error

        if not self.index_job_manager:
            return JSONResponse(
                {"error": "Job manager not available"},
                status_code=400,
            )

        include_completed = (
            request.query_params.get("completed", "true").lower() == "true"
        )
        limit = int(request.query_params.get("limit", "20"))

        jobs = self.index_job_manager.list_jobs(
            include_completed=include_completed, limit=limit
        )

        return JSONResponse(
            {
                "jobs": [j.to_dict() for j in jobs],
                "total": len(jobs),
            }
        )

    async def _handle_job_cancel(self, request: Request) -> JSONResponse:
        """Handle job cancellation request."""
        origin_error = _validate_origin(request)
        if origin_error:
            return origin_error

        if not self.index_job_manager:
            return JSONResponse(
                {"error": "Job manager not available"},
                status_code=400,
            )

        job_id = request.path_params.get("job_id")
        if not job_id:
            return JSONResponse(
                {"error": "job_id is required"},
                status_code=400,
            )

        success = await self.index_job_manager.cancel_job(job_id)
        if success:
            return JSONResponse({"status": "cancelled", "job_id": job_id})
        else:
            return JSONResponse(
                {"error": f"Job not found or already completed: {job_id}"},
                status_code=404,
            )

    async def _handle_mcp_tools_call(self, request: Request) -> JSONResponse:
        """Handle direct MCP tool call via HTTP POST.

        This endpoint allows proxy clients to call tools without SSE.
        Accepts JSON-RPC style request and returns JSON-RPC response.
        """
        # Validate Origin header
        origin_error = _validate_origin(request)
        if origin_error:
            return origin_error

        # Validate MCP-Protocol-Version header
        version_error = _validate_protocol_version(request)
        if version_error:
            return version_error

        if not self.services:
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "error": {"code": -32603, "message": "Server not initialized"},
                    "id": None,
                },
                status_code=503,
            )

        # Check body size before parsing
        if error := _validate_body_size(request):
            return error

        # Initialize body before try block so it's available in except
        body: dict = {}
        try:
            body = await request.json()

            # Extract JSON-RPC request parts
            request_id = body.get("id", 1)
            params = body.get("params", {})
            tool_name = params.get("name")
            arguments = params.get("arguments", {})

            if not tool_name:
                return JSONResponse(
                    {
                        "jsonrpc": "2.0",
                        "error": {"code": -32602, "message": "Missing tool name"},
                        "id": request_id,
                    },
                    status_code=400,
                )

            # Check if tool exists before calling
            if tool_name not in TOOL_REGISTRY:
                return JSONResponse(
                    {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32602,
                            "message": f"Unknown tool: {tool_name}",
                        },
                        "id": request_id,
                    },
                    status_code=400,
                )

            # Extract project context from header
            project_context = request.headers.get("X-ChunkHound-Project")
            client_context = {"project": project_context} if project_context else None

            self.debug_log(f"Tool call: {tool_name}, project: {project_context}")

            # Call the unified tool handler
            result = await handle_tool_call(
                tool_name=tool_name,
                arguments=arguments,
                services=self.ensure_services(),
                embedding_manager=self.embedding_manager,
                initialization_complete=self._initialization_complete,
                debug_mode=self.debug_mode,
                scan_progress=self._scan_progress,
                llm_manager=self.llm_manager,
                client_context=client_context,
                project_registry=self.project_registry,
            )

            # Convert result to JSON-RPC response (MCP 2025-11-25 compliant)
            # Returns dict (structured) or list[TextContent] (unstructured)
            if isinstance(result, dict):
                # Structured content - include structuredContent + serialized
                import json

                return JSONResponse(
                    {
                        "jsonrpc": "2.0",
                        "result": {
                            "content": [
                                {"type": "text", "text": json.dumps(result, indent=2)}
                            ],
                            "structuredContent": result,
                            "isError": False,
                        },
                        "id": request_id,
                    }
                )
            elif result and len(result) > 0:
                # Unstructured content - text only
                response_text = result[0].text
                return JSONResponse(
                    {
                        "jsonrpc": "2.0",
                        "result": {
                            "content": [{"type": "text", "text": response_text}],
                            "isError": False,
                        },
                        "id": request_id,
                    }
                )
            else:
                return JSONResponse(
                    {
                        "jsonrpc": "2.0",
                        "result": {"content": [], "isError": False},
                        "id": request_id,
                    }
                )

        except Exception as e:
            logger.exception("MCP tool call failed")
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "error": {"code": -32603, "message": str(e)},
                    "id": body.get("id"),
                },
                status_code=500,
            )

    async def _handle_mcp_tools_list(self, request: Request) -> JSONResponse:
        """Handle MCP tools/list request via HTTP POST.

        Returns list of available tools in JSON-RPC format.
        """
        # Validate Origin header
        origin_error = _validate_origin(request)
        if origin_error:
            return origin_error

        # Validate MCP-Protocol-Version header
        version_error = _validate_protocol_version(request)
        if version_error:
            return version_error

        # Check body size before parsing
        if error := _validate_body_size(request):
            return error

        try:
            body = await request.json()
            request_id = body.get("id", 1)

            # Wait for initialization
            try:
                await asyncio.wait_for(
                    self._initialization_complete.wait(), timeout=5.0
                )
            except asyncio.TimeoutError:
                pass

            # Build tools list
            tools = []
            for tool_name, tool in TOOL_REGISTRY.items():
                if tool.requires_embeddings and (
                    not self.embedding_manager
                    or not self.embedding_manager.list_providers()
                ):
                    continue

                tool_def = {
                    "name": tool_name,
                    "description": tool.description,
                    "inputSchema": tool.parameters,
                }
                if tool.annotations:
                    tool_def["annotations"] = tool.annotations
                tools.append(tool_def)

            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "result": {"tools": tools},
                    "id": request_id,
                }
            )

        except Exception as e:
            logger.exception("MCP tools list failed")
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "error": {"code": -32603, "message": str(e)},
                    "id": None,
                },
                status_code=500,
            )

    async def _handle_project_remove(self, request: Request) -> JSONResponse:
        """Handle project removal request."""
        # Validate Origin header (security: prevents DNS rebinding attacks)
        origin_error = _validate_origin(request)
        if origin_error:
            return origin_error

        if not self.project_registry:
            return JSONResponse(
                {"error": "Multi-repo mode not enabled"},
                status_code=400,
            )

        # Check body size before parsing
        if error := _validate_body_size(request):
            return error

        try:
            body = await request.json()
            name_or_path = body.get("name") or body.get("path")
            cascade = body.get("cascade", False)

            if not name_or_path:
                return JSONResponse(
                    {"error": "name or path is required"},
                    status_code=400,
                )

            success = self.project_registry.unregister_project(
                name_or_path, cascade=cascade
            )

            if success:
                return JSONResponse(
                    {
                        "status": "success",
                        "removed": name_or_path,
                        "cascade": cascade,
                    }
                )
            else:
                return JSONResponse(
                    {"error": f"Project not found: {name_or_path}"},
                    status_code=404,
                )

        except Exception as e:
            logger.exception("Remove request failed")
            return JSONResponse(
                {"error": str(e)},
                status_code=500,
            )

    async def _handle_project_tags(self, request: Request) -> JSONResponse:
        """Handle project tags management request.

        Supports operations: add, remove, set, list
        """
        origin_error = _validate_origin(request)
        if origin_error:
            return origin_error

        if not self.project_registry:
            return JSONResponse(
                {"error": "Multi-repo mode not enabled"},
                status_code=400,
            )

        # Check body size before parsing
        if error := _validate_body_size(request):
            return error

        try:
            body = await request.json()
            operation = body.get("operation", "list")
            name_or_path = body.get("name") or body.get("path")
            tags = body.get("tags", [])

            if operation == "list":
                # List all tags or tags for a specific project
                if name_or_path:
                    project = self.project_registry.get_project(name_or_path)
                    if not project:
                        return JSONResponse(
                            {"error": f"Project not found: {name_or_path}"},
                            status_code=404,
                        )
                    return JSONResponse({"tags": project.tags})
                else:
                    all_tags = self.project_registry.get_all_tags()
                    return JSONResponse({"tags": all_tags})

            # For add/remove/set, we need a project identifier
            if not name_or_path:
                return JSONResponse(
                    {"error": "name or path is required for tag operations"},
                    status_code=400,
                )

            if operation == "add":
                success = self.project_registry.add_project_tags(name_or_path, tags)
            elif operation == "remove":
                success = self.project_registry.remove_project_tags(name_or_path, tags)
            elif operation == "set":
                success = self.project_registry.set_project_tags(name_or_path, tags)
            else:
                return JSONResponse(
                    {"error": f"Unknown operation: {operation}"},
                    status_code=400,
                )

            if success:
                # Get updated tags
                project = self.project_registry.get_project(name_or_path)
                return JSONResponse(
                    {
                        "status": "success",
                        "project": name_or_path,
                        "operation": operation,
                        "tags": project.tags if project else [],
                    }
                )
            else:
                return JSONResponse(
                    {"error": f"Project not found: {name_or_path}"},
                    status_code=404,
                )

        except Exception as e:
            logger.exception("Tags request failed")
            return JSONResponse(
                {"error": str(e)},
                status_code=500,
            )

    async def _handle_watchers_refresh(self, request: Request) -> JSONResponse:
        """Handle request to refresh file watchers.

        Restarts file watchers for all or a specific project.
        Useful when watchers become stale or stop responding.
        """
        # Validate Origin header (security: prevents DNS rebinding attacks)
        origin_error = _validate_origin(request)
        if origin_error:
            return origin_error

        if not self.watcher_manager:
            return JSONResponse(
                {"error": "Watcher manager not initialized"},
                status_code=400,
            )

        if not self.project_registry:
            return JSONResponse(
                {"error": "Multi-repo mode not enabled"},
                status_code=400,
            )

        try:
            # Check for project query parameter
            project_name = request.query_params.get("project")

            refreshed = 0
            errors: list[str] = []

            if project_name:
                # Refresh specific project
                project = self.project_registry.get_project(project_name)
                if not project:
                    return JSONResponse(
                        {"error": f"Project not found: {project_name}"},
                        status_code=404,
                    )

                try:
                    # Stop and restart watcher for this project
                    self.watcher_manager.stop_watcher(project.base_directory)
                    self.watcher_manager.start_watcher(project)
                    refreshed = 1
                    logger.info(f"Refreshed watcher for {project_name}")
                except Exception as e:
                    errors.append(f"{project_name}: {e}")
            else:
                # Refresh all project watchers
                for project in self.project_registry.list_projects():
                    try:
                        self.watcher_manager.stop_watcher(project.base_directory)
                        self.watcher_manager.start_watcher(project)
                        refreshed += 1
                        logger.info(f"Refreshed watcher for {project.project_name}")
                    except Exception as e:
                        errors.append(f"{project.project_name}: {e}")

            return JSONResponse(
                {
                    "status": "success",
                    "refreshed": refreshed,
                    "errors": errors,
                }
            )

        except Exception as e:
            logger.exception("Watcher refresh failed")
            return JSONResponse(
                {"error": str(e)},
                status_code=500,
            )

    def _create_app(self) -> Starlette:
        """Create the Starlette application with routes."""
        routes = [
            # MCP endpoints
            Route("/sse", self._handle_sse, methods=["GET"]),
            Route("/messages", self._handle_messages, methods=["POST"]),
            # MCP tool endpoints for proxy clients (JSON-RPC over HTTP)
            Route("/mcp/tools/call", self._handle_mcp_tools_call, methods=["POST"]),
            Route("/mcp/tools/list", self._handle_mcp_tools_list, methods=["POST"]),
            # Management endpoints
            Route("/health", self._handle_health, methods=["GET"]),
            Route("/stats", self._handle_stats, methods=["GET"]),
            Route("/projects", self._handle_projects_list, methods=["GET"]),
            Route("/projects/index", self._handle_project_index, methods=["POST"]),
            Route("/projects/remove", self._handle_project_remove, methods=["POST"]),
            Route("/projects/tags", self._handle_project_tags, methods=["POST"]),
            Route("/watchers/refresh", self._handle_watchers_refresh, methods=["POST"]),
            # Indexing job endpoints
            Route("/jobs", self._handle_jobs_list, methods=["GET"]),
            Route("/jobs/{job_id}", self._handle_job_status, methods=["GET"]),
            Route("/jobs/{job_id}/cancel", self._handle_job_cancel, methods=["POST"]),
        ]

        # Build CORS allowed origins list with common ports
        cors_origins = []
        for scheme in _ALLOWED_SCHEMES:
            for host in _ALLOWED_HOSTS:
                base = f"{scheme}://{host}"
                cors_origins.append(base)
                # Add common development ports
                for port in [3000, 5173, 8000, 8080]:
                    cors_origins.append(f"{base}:{port}")

        middleware = [
            Middleware(
                CORSMiddleware,
                allow_origins=cors_origins,
                allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
                allow_headers=["*"],
            ),
        ]

        @asynccontextmanager
        async def lifespan(app: Starlette) -> AsyncIterator[None]:
            """Application lifespan handler."""
            self._started_at = time.time()

            # Initialize services
            await self.initialize()

            # Initialize multi-repo support
            await self._initialize_multi_repo()

            self._initialization_complete.set()
            self.debug_log("HTTP server initialization complete")

            yield

            # Graceful shutdown: flush pending events before stopping
            self.debug_log("Starting graceful shutdown...")

            # Stop index job manager first (cancels running jobs)
            if self.index_job_manager:
                await self.index_job_manager.stop()

            if self.watcher_manager:
                pending_count = self.watcher_manager.get_pending_count()
                if pending_count > 0:
                    self.debug_log(f"Flushing {pending_count} pending file events...")
                    flushed = self.watcher_manager.flush_pending()
                    self.debug_log(f"Flushed {flushed} events")

                await self.watcher_manager.stop()

            await self.cleanup()
            self.debug_log("HTTP server shutdown complete")

        return Starlette(
            routes=routes,
            middleware=middleware,
            lifespan=lifespan,
        )

    async def _cleanup_stale_sessions(self) -> None:
        """Background task to clean up stale sessions.

        Removes sessions that have been connected longer than the max age.
        This handles edge cases where SSE connections drop without proper cleanup.
        """
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self._session_cleanup_interval)

                if self._shutdown_event.is_set():
                    break

                current_time = time.time()
                stale_sessions = []

                async with self._sessions_lock:
                    for client_id, session in self._active_sessions.items():
                        connected_at = session.get("connected_at", current_time)
                        age = current_time - connected_at
                        if age > self._session_max_age_seconds:
                            stale_sessions.append(client_id)

                    for client_id in stale_sessions:
                        self._active_sessions.pop(client_id, None)
                        self.debug_log(f"Cleaned up stale session: {client_id}")

                if stale_sessions:
                    logger.info(f"Cleaned up {len(stale_sessions)} stale sessions")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in session cleanup: {e}")
                await asyncio.sleep(60)  # Back off on error

    async def run(self) -> None:
        """Run the HTTP server."""
        if not _HTTP_AVAILABLE:
            logger.error(
                "HTTP dependencies not available. Install with: pip install starlette uvicorn"
            )
            sys.exit(1)

        if not _MCP_AVAILABLE:
            logger.error("MCP SDK not available")
            sys.exit(1)

        self.app = self._create_app()

        # Setup signal handlers
        loop = asyncio.get_event_loop()

        def signal_handler():
            self._shutdown_event.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)

        # Configure uvicorn
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
        )

        server = uvicorn.Server(config)

        logger.info(
            f"Starting ChunkHound HTTP server on http://{self.host}:{self.port}"
        )
        self.debug_log(f"HTTP server starting on {self.host}:{self.port}")

        # Start background session cleanup task
        self._session_cleanup_task = asyncio.create_task(self._cleanup_stale_sessions())

        try:
            # Run server
            await server.serve()
        finally:
            # Cancel cleanup task on shutdown
            if self._session_cleanup_task and not self._session_cleanup_task.done():
                self._session_cleanup_task.cancel()
                try:
                    await self._session_cleanup_task
                except asyncio.CancelledError:
                    pass

    def run_sync(self) -> None:
        """Synchronous wrapper for running the server."""
        asyncio.run(self.run())


async def main(
    host: str | None = None,
    port: int | None = None,
    args: Any = None,
) -> None:
    """Main entry point for the HTTP MCP server.

    Args:
        host: Host to bind to (default from config or 127.0.0.1)
        port: Port to listen on (default from config or 5173)
        args: Pre-parsed arguments
    """
    import argparse

    from chunkhound.api.cli.utils.config_factory import create_validated_config

    if args is None:
        parser = argparse.ArgumentParser(
            description="ChunkHound MCP HTTP server",
        )
        parser.add_argument(
            "--host",
            default=None,
            help="Host to bind to (default from config or 127.0.0.1)",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=None,
            help="Port to listen on (default from config or 5173)",
        )
        parser.add_argument(
            "path",
            nargs="?",
            default=".",
            help="Project path",
        )
        args = parser.parse_args()
        host = args.host
        port = args.port

    # Enable global mode for HTTP server
    os.environ["CHUNKHOUND_DATABASE__MULTI_REPO__ENABLED"] = "true"
    os.environ["CHUNKHOUND_DATABASE__MULTI_REPO__MODE"] = "global"

    # Create config
    config, errors = create_validated_config(args, "mcp")

    if errors:
        for error in errors:
            logger.error(error)
        sys.exit(1)

    # Create and run server (host/port default to config values if not specified)
    server = HTTPMCPServer(config, host=host, port=port, args=args)
    await server.run()


def main_sync() -> None:
    """Synchronous wrapper for CLI entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
