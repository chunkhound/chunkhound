"""MCP Proxy Client for stdio-to-HTTP transport.

This module provides a proxy client that forwards MCP requests from stdio
transport to a running HTTP daemon server. This enables CLI sessions to
benefit from global database mode without running their own server.

Architecture:
    stdio → MCPProxyClient → HTTP Daemon → Global Database

Usage:
    When `chunkhound mcp` starts and detects a running HTTP daemon:
    1. Create MCPProxyClient pointing to daemon URL
    2. Forward all MCP requests through the proxy
    3. Pass project context (CWD) via HTTP header

The proxy is transparent to the MCP client - it receives the same
responses as if connected directly to the HTTP server.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from chunkhound.mcp_server.common import CURRENT_PROTOCOL_VERSION
from chunkhound.mcp_server.exceptions import MCPError, ToolExecutionError

if TYPE_CHECKING:
    import httpx


class MCPProxyClient:
    """Proxy client that forwards MCP requests to HTTP daemon.

    This client implements the MCP server interface but forwards all
    requests to a running HTTP daemon. It's used when stdio mode
    should benefit from global database without running its own server.

    Features:
    - Automatic retry on transient failures (connection errors, timeouts)
    - Exponential backoff between retries
    - Reconnection handling for daemon restarts
    - Graceful fallback notification when daemon becomes unavailable

    Attributes:
        server_url: URL of the HTTP daemon server
        project_context: Current project path for scoped searches
    """

    DEFAULT_TIMEOUT = 120.0  # 2 minutes for long operations (used when no config)
    MAX_RETRIES = 3
    RETRY_BACKOFF_BASE = 0.5  # Base delay for exponential backoff (seconds)

    def __init__(
        self,
        server_url: str,
        project_context: Path | None = None,
        timeout: float | None = None,
    ):
        """Initialize proxy client.

        Args:
            server_url: Base URL of HTTP daemon (e.g., http://127.0.0.1:5173)
            project_context: Current project path to pass with requests
            timeout: Request timeout in seconds (default from config or 120.0)
        """
        self.server_url = server_url.rstrip("/")
        self.project_context = project_context or Path.cwd()
        self._timeout = timeout or self.DEFAULT_TIMEOUT

        # Lazy-initialized HTTP client
        self._client: httpx.AsyncClient | None = None
        self._initialized = False

        # Request ID counter for JSON-RPC
        self._request_id = 0

        # Connection state tracking
        self._consecutive_failures = 0
        self._max_consecutive_failures = 5  # After this many, consider daemon dead

    def _next_request_id(self) -> int:
        """Get next request ID for JSON-RPC calls.

        Returns:
            Incrementing request ID
        """
        self._request_id += 1
        return self._request_id

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client.

        Returns:
            Async HTTP client configured for daemon communication
        """
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                follow_redirects=True,
            )
        return self._client

    async def initialize(self) -> bool:
        """Initialize connection to HTTP daemon.

        Verifies the daemon is reachable and healthy.

        Returns:
            True if connection successful, False otherwise
        """
        if self._initialized:
            return True

        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.server_url}/health",
                timeout=5.0,
            )

            if response.status_code == 200:
                health = response.json()
                logger.debug(f"Connected to daemon: {health}")
                self._initialized = True
                return True
            else:
                logger.warning(f"Daemon health check failed: {response.status_code}")
                await self.close()  # Clean up client on failure
                return False

        except Exception as e:
            logger.debug(f"Failed to connect to daemon: {e}")
            await self.close()  # Clean up client on failure
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            self._initialized = False

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if an error should trigger a retry.

        Args:
            error: The exception that occurred

        Returns:
            True if the error is transient and retryable
        """
        import httpx

        # Connection errors are retryable
        if isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout)):
            return True
        # Read timeouts might be retryable for idempotent operations
        if isinstance(error, httpx.ReadTimeout):
            return True
        # Pool timeouts
        if isinstance(error, httpx.PoolTimeout):
            return True
        # Network errors
        if isinstance(error, httpx.NetworkError):
            return True
        return False

    async def _execute_with_retry(
        self,
        operation: str,
        request_fn: Any,
        max_retries: int | None = None,
    ) -> Any:
        """Execute an HTTP request with retry logic.

        Args:
            operation: Name of the operation (for logging)
            request_fn: Async function that performs the HTTP request
            max_retries: Maximum retry attempts (default: MAX_RETRIES)

        Returns:
            Response from the request function

        Raises:
            Exception: If all retries fail
        """
        import asyncio

        retries = max_retries if max_retries is not None else self.MAX_RETRIES
        last_error = None

        for attempt in range(retries + 1):
            try:
                result = await request_fn()
                # Success - reset failure counter
                self._consecutive_failures = 0
                return result

            except Exception as e:
                last_error = e

                # Track consecutive failures
                self._consecutive_failures += 1

                if not self._is_retryable_error(e):
                    # Non-retryable error - fail immediately
                    raise

                if attempt < retries:
                    # Calculate backoff with exponential increase
                    delay = self.RETRY_BACKOFF_BASE * (2**attempt)
                    logger.debug(
                        f"{operation} attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)

                    # Try to reconnect if client might be stale
                    if self._consecutive_failures >= 2:
                        await self._reconnect()
                else:
                    logger.warning(
                        f"{operation} failed after {retries + 1} attempts: {e}"
                    )

        raise last_error  # type: ignore[misc]

    async def _reconnect(self) -> bool:
        """Attempt to reconnect to the daemon.

        Returns:
            True if reconnection successful
        """
        logger.debug("Attempting to reconnect to daemon...")

        # Close existing client
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None

        # Reset initialization state
        self._initialized = False

        # Try to reinitialize
        return await self.initialize()

    @property
    def is_healthy(self) -> bool:
        """Check if the proxy connection appears healthy.

        Returns:
            False if too many consecutive failures have occurred
        """
        return self._consecutive_failures < self._max_consecutive_failures

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for requests.

        Includes project context for scoped searches.

        Returns:
            Headers dict with project context and protocol version
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "MCP-Protocol-Version": CURRENT_PROTOCOL_VERSION,
        }

        if self.project_context:
            headers["X-ChunkHound-Project"] = str(self.project_context)

        return headers

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Call an MCP tool via HTTP daemon.

        Includes automatic retry with exponential backoff for transient failures.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool execution result (includes 'content' list and 'isError' flag)

        Raises:
            ToolExecutionError: If request fails after retries
        """

        async def _do_request() -> dict[str, Any]:
            client = await self._get_client()

            # Build JSON-RPC request with incrementing ID
            request_body = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments,
                },
                "id": self._next_request_id(),
            }

            response = await client.post(
                f"{self.server_url}/mcp/tools/call",
                json=request_body,
                headers=self._get_headers(),
            )

            if response.status_code == 200:
                result = response.json()
                if "error" in result:
                    raise ToolExecutionError(
                        result["error"].get("message", str(result["error"]))
                    )
                return result.get("result", {})
            else:
                raise ToolExecutionError(
                    f"HTTP {response.status_code}: {response.text}"
                )

        try:
            return await self._execute_with_retry(
                f"call_tool({tool_name})",
                _do_request,
            )
        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            raise

    async def list_tools(self) -> list[dict[str, Any]]:
        """List available tools from HTTP daemon.

        Includes automatic retry with exponential backoff for transient failures.

        Returns:
            List of tool definitions

        Raises:
            ToolExecutionError: If request fails after retries
        """

        async def _do_request() -> list[dict[str, Any]]:
            client = await self._get_client()

            request_body = {
                "jsonrpc": "2.0",
                "method": "tools/list",
                "params": {},
                "id": self._next_request_id(),
            }

            response = await client.post(
                f"{self.server_url}/mcp/tools/list",
                json=request_body,
                headers=self._get_headers(),
            )

            if response.status_code == 200:
                result = response.json()
                if "error" in result:
                    raise ToolExecutionError(
                        result["error"].get("message", str(result["error"]))
                    )
                return result.get("result", {}).get("tools", [])
            else:
                raise ToolExecutionError(
                    f"HTTP {response.status_code}: {response.text}"
                )

        try:
            return await self._execute_with_retry("list_tools", _do_request)
        except Exception as e:
            logger.error(f"List tools failed: {e}")
            raise

    async def get_health(self) -> dict[str, Any]:
        """Get daemon health status.

        Returns:
            Health status dict

        Raises:
            Exception: If request fails
        """
        client = await self._get_client()

        try:
            response = await client.get(
                f"{self.server_url}/health",
                headers=self._get_headers(),
            )

            if response.status_code == 200:
                return response.json()
            else:
                raise MCPError(f"HTTP {response.status_code}: {response.text}")

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise


class ProxyDecision:
    """Result of should_use_proxy check.

    Attributes:
        use_proxy: True if proxy should be used
        daemon_url: URL of daemon if available
        global_mode_error: True if global mode is enabled but daemon unavailable.
                          Callers should NOT fall back to direct mode in this case.
        error_message: Human-readable error message if global_mode_error is True
    """

    __slots__ = ("use_proxy", "daemon_url", "global_mode_error", "error_message")

    def __init__(
        self,
        use_proxy: bool,
        daemon_url: str | None,
        global_mode_error: bool = False,
        error_message: str | None = None,
    ):
        self.use_proxy = use_proxy
        self.daemon_url = daemon_url
        self.global_mode_error = global_mode_error
        self.error_message = error_message

    def __iter__(self) -> Any:
        """Allow tuple unpacking for backwards compatibility."""
        return iter((self.use_proxy, self.daemon_url))


def should_use_proxy(config: Any = None) -> ProxyDecision:
    """Determine if proxy mode should be used.

    Checks if:
    1. Global mode is configured
    2. HTTP daemon is running

    Args:
        config: Optional Config object for settings

    Returns:
        ProxyDecision with proxy status and error information.

        IMPORTANT: If global_mode_error is True, callers MUST NOT fall back
        to direct database access. This prevents lock contention in global mode.
    """
    from chunkhound.services.daemon_manager import DaemonManager

    # Check configuration
    try:
        if config is None:
            from chunkhound.core.config.loader import load_config

            config = load_config()

        # Only use proxy in global mode
        if not config.database.multi_repo.enabled:
            return ProxyDecision(use_proxy=False, daemon_url=None)

        if config.database.multi_repo.mode != "global":
            return ProxyDecision(use_proxy=False, daemon_url=None)

    except Exception:
        # Config loading failed - fall back to per-repo
        return ProxyDecision(use_proxy=False, daemon_url=None)

    # Check if daemon is running using config for host/port
    daemon = DaemonManager(config=config)
    status = daemon.status()

    if status.running and status.url:
        return ProxyDecision(use_proxy=True, daemon_url=status.url)

    # Global mode configured but daemon not running
    # Try auto-start if configured
    if config.database.multi_repo.auto_start_daemon:
        logger.info("Auto-starting daemon for global mode...")
        try:
            # wait=True already polls for readiness, no extra sleep needed
            success = daemon.start(background=True, wait=True)
            if success:
                status = daemon.status()
                if status.running and status.url:
                    logger.info(f"Daemon auto-started at {status.url}")
                    return ProxyDecision(use_proxy=True, daemon_url=status.url)
            logger.warning("Failed to auto-start daemon")
        except Exception as e:
            logger.warning(f"Auto-start failed: {e}")

    # Global mode is enabled but daemon is not available
    # This is an ERROR state - do NOT allow fallback to direct mode
    error_msg = (
        "Global mode is enabled but daemon is not running. "
        "Direct database access is disabled to prevent lock contention. "
        "Start the daemon with: chunkhound daemon start --background"
    )
    logger.warning(error_msg)
    return ProxyDecision(
        use_proxy=False,
        daemon_url=None,
        global_mode_error=True,
        error_message=error_msg,
    )


async def create_proxy_client_if_available(
    config: Any = None,
    project_path: Path | None = None,
) -> MCPProxyClient | None:
    """Create proxy client if daemon is available.

    Convenience function that checks configuration and daemon status,
    then creates and initializes a proxy client.

    Args:
        config: Optional Config object
        project_path: Project context path (defaults to CWD)

    Returns:
        Initialized MCPProxyClient or None if proxy not available

    Note:
        If global mode is enabled but daemon unavailable, returns None.
        Callers should check should_use_proxy().global_mode_error if they
        need to handle this case specially.
    """
    decision = should_use_proxy(config)

    if not decision.use_proxy or not decision.daemon_url:
        return None

    # Get timeout from config if available
    timeout = None
    if config is not None:
        timeout = config.database.multi_repo.proxy_timeout_seconds

    # Create and initialize proxy
    proxy = MCPProxyClient(
        server_url=decision.daemon_url,
        project_context=project_path or Path.cwd(),
        timeout=timeout,
    )

    if await proxy.initialize():
        return proxy
    else:
        await proxy.close()
        return None
