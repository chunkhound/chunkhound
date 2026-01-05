"""
Reranking server lifecycle management for testing.

Provides fixtures and utilities for automatically starting and stopping
the mock reranking server during test execution.
"""

import asyncio
import sys
import time
from pathlib import Path

import httpx
from loguru import logger

# Add tests directory to path for imports
tests_dir = Path(__file__).parent.parent
sys.path.insert(0, str(tests_dir))


class RerankServerManager:
    """Manages the lifecycle of the mock reranking server for tests."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8001):
        self.host = host
        self.port = port
        self.process = None
        self.base_url = f"http://{host}:{port}"

    async def start(self, timeout: float = 10.0) -> None:
        """Start the mock reranking server."""
        if await self.is_running():
            logger.info("Rerank server already running")
            return

        logger.info(f"Starting mock rerank server on {self.host}:{self.port}")

        # Start server subprocess
        self.process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "tests.rerank_server",
            "--host",
            self.host,
            "--port",
            str(self.port),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=Path.cwd(),  # Run from project root
        )

        # Wait for server to be ready
        start_time = time.time()
        while time.time() - start_time < timeout:
            if await self.is_running():
                logger.info("Mock rerank server started successfully")
                return
            await asyncio.sleep(0.1)

        # Timeout - kill process and raise
        if self.process:
            self.process.terminate()
            await self.process.wait()
            self.process = None

        raise TimeoutError(
            f"Mock rerank server failed to start within {timeout} seconds"
        )

    async def stop(self) -> None:
        """Stop the mock reranking server."""
        if not self.process:
            logger.debug("No rerank server process to stop")
            return

        logger.info("Stopping mock rerank server")

        try:
            # Try graceful shutdown first
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
                logger.info("Mock rerank server stopped gracefully")
            except asyncio.TimeoutError:
                # Force kill if graceful shutdown fails
                logger.warning("Graceful shutdown failed, force killing rerank server")
                self.process.kill()
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    logger.error("Failed to kill rerank server process")
        except Exception as e:
            logger.error(f"Error stopping rerank server: {e}")
            # Ensure process is killed even if other steps fail
            try:
                self.process.kill()
                await self.process.wait()
            except Exception:
                pass
        finally:
            # Always clear the process reference
            self.process = None

    async def is_running(self) -> bool:
        """Check if the reranking server is running and responsive."""
        try:
            async with httpx.AsyncClient(timeout=1.0) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except (httpx.RequestError, httpx.TimeoutException):
            return False

    async def wait_for_ready(self, timeout: float = 10.0) -> bool:
        """Wait for the server to become ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if await self.is_running():
                return True
            await asyncio.sleep(0.1)
        return False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


def check_port_available(host: str = "127.0.0.1", port: int = 8001) -> bool:
    """Check if a port is available for binding."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


async def ensure_rerank_server_running(
    host: str = "127.0.0.1", port: int = 8001, start_if_needed: bool = True
) -> RerankServerManager | None:
    """
    Ensure a reranking server is running.

    Args:
        host: Server host
        port: Server port
        start_if_needed: Whether to start the mock server if none is running

    Returns:
        RerankServerManager if a mock server was started, None if external server detected
    """
    manager = RerankServerManager(host=host, port=port)

    # Check if a server is already running
    if await manager.is_running():
        logger.info(f"Rerank server already running on {host}:{port}")
        return None  # External server, no manager needed

    if not start_if_needed:
        return None

    # Check if port is available
    if not check_port_available(host, port):
        logger.warning(f"Port {port} is in use but server not responding")
        return None

    # Start mock server
    await manager.start()
    return manager
