"""Windows-compatible subprocess utilities for tests."""

import asyncio
from typing import Any

from chunkhound.utils.windows_constants import get_utf8_env


async def create_subprocess_exec_safe(
    *args: str,
    stdin: Any | None = None,
    stdout: Any | None = None,
    stderr: Any | None = None,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    **kwargs: Any,
) -> asyncio.subprocess.Process:
    """Create subprocess with Windows-safe encoding settings.

    This function ensures proper UTF-8 encoding for subprocess communication
    on Windows, preventing Unicode encoding errors that break JSON-RPC protocols.
    """
    # Set up environment with UTF-8 encoding for Windows compatibility
    env = get_utf8_env(env)

    return await asyncio.create_subprocess_exec(
        *args, stdin=stdin, stdout=stdout, stderr=stderr, env=env, cwd=cwd, **kwargs
    )


def get_safe_subprocess_env(
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Get environment variables with Windows-safe encoding settings."""
    return get_utf8_env(base_env)
