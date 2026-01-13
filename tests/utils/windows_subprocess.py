"""Cross-platform subprocess utilities for tests."""

import asyncio
import os
from typing import Dict, Optional, Any

from chunkhound.utils.windows_constants import get_utf8_env


async def create_subprocess_exec_safe(
    *args: str,
    stdin: Optional[Any] = None,
    stdout: Optional[Any] = None,
    stderr: Optional[Any] = None,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
    **kwargs: Any
) -> asyncio.subprocess.Process:
    """Create subprocess with Windows-safe encoding settings.
    
    This function ensures proper UTF-8 encoding for subprocess communication
    on Windows, preventing Unicode encoding errors that break JSON-RPC protocols.
    """
    # Set up environment with UTF-8 encoding for Windows compatibility
    env = get_utf8_env(env)
    
    return await asyncio.create_subprocess_exec(
        *args,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        env=env,
        cwd=cwd,
        **kwargs
    )


def get_safe_subprocess_env(base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Get environment variables with Windows-safe encoding settings."""
    return get_utf8_env(base_env)


def terminate_process_tree(pid: int, timeout: float = 5.0) -> None:
    """Terminate a process and all its children recursively.

    This ensures complete cleanup of process trees, which is particularly important on Windows
    where subprocess.terminate() doesn't kill child processes. Works cross-platform using psutil.

    On Windows: Critical for cleaning up uv->python->chunkhound process chains
    On Unix (Linux/macOS): Provides more reliable cleanup than basic terminate()

    Args:
        pid: Process ID of the root process to terminate
        timeout: Maximum time to wait for processes to terminate
    """
    try:
        import psutil
    except ImportError:
        # Fallback to basic terminate if psutil not available
        try:
            import signal
            os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
        return

    try:
        parent = psutil.Process(pid)
        # Get all child processes recursively
        children = parent.children(recursive=True)

        # Terminate children first (in reverse order to avoid dependency issues)
        for child in reversed(children):
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass

        # Terminate parent
        try:
            parent.terminate()
        except psutil.NoSuchProcess:
            pass

        # Wait for all processes to terminate
        gone, alive = psutil.wait_procs(children + [parent], timeout=timeout)

        # Force kill any remaining processes
        for proc in alive:
            try:
                proc.kill()
            except psutil.NoSuchProcess:
                pass

    except (psutil.NoSuchProcess, psutil.AccessDenied):
        # Process already gone or access denied - nothing to do
        pass


async def terminate_async_process_tree(process: asyncio.subprocess.Process, timeout: float = 5.0) -> None:
    """Terminate an asyncio subprocess and all its children recursively.

    Args:
        process: The asyncio subprocess to terminate
        timeout: Maximum time to wait for processes to terminate
    """
    if process.returncode is not None:
        # Process already terminated
        return

    # Terminate the process tree
    terminate_process_tree(process.pid, timeout)

    # Wait for the asyncio process to acknowledge termination
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        # Force kill if still not terminated
        try:
            process.kill()
            await asyncio.wait_for(process.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass