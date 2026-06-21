"""Windows-specific constants and utilities for ChunkHound.

This module centralizes all Windows-specific configuration, delays, and
environment variables to ensure consistent behavior across the codebase.
"""

import platform
import time
from pathlib import Path

# Platform detection (cached for performance)
IS_WINDOWS = platform.system() == "Windows"

# Windows-specific timing constants (in seconds)
WINDOWS_FILE_HANDLE_DELAY = 0.1  # 100ms - Standard file handle release delay
WINDOWS_DB_CLEANUP_DELAY = 0.2  # 200ms - Database cleanup delay
WINDOWS_RETRY_DELAY = 0.5  # 500ms - Retry operations delay

# Windows UTF-8 environment variables for subprocess operations
WINDOWS_UTF8_ENV: dict[str, str] = {
    "PYTHONIOENCODING": "utf-8",
    "PYTHONLEGACYWINDOWSSTDIO": "1",
    "PYTHONUTF8": "1",
}


def is_windows() -> bool:
    """Check if running on Windows platform.

    Returns:
        True if running on Windows, False otherwise.
    """
    return IS_WINDOWS


def unlink_with_retry(path: Path, max_attempts: int = 5) -> None:
    """Delete a file with exponential-backoff retry for Windows handle delays.

    On Windows, file handles from a just-closed DuckDB connection may not be
    released immediately. This function retries the deletion up to max_attempts
    times with exponential backoff (100ms base), giving Windows time to release
    the handle.

    On non-Windows platforms this behaves identically to path.unlink(missing_ok=True).
    """
    if not IS_WINDOWS:
        path.unlink(missing_ok=True)
        return
    for attempt in range(max_attempts):
        try:
            path.unlink(missing_ok=True)
            return
        except OSError:
            if attempt < max_attempts - 1:
                time.sleep(WINDOWS_FILE_HANDLE_DELAY * (2**attempt))
            else:
                raise


# Alias used by compaction code (PR #339 naming convention)
_unlink_compacted = unlink_with_retry


def get_utf8_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Get environment variables with Windows UTF-8 settings.

    Args:
        base_env: Base environment to extend, or None for os.environ

    Returns:
        Environment dictionary with Windows UTF-8 settings applied
    """
    import os

    if base_env is None:
        env = os.environ.copy()
    else:
        env = base_env.copy()

    if IS_WINDOWS:
        env.update(WINDOWS_UTF8_ENV)

    return env
