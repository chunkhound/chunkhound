"""Windows-specific constants and utilities for ChunkHound.

This module centralizes all Windows-specific configuration, delays, and
environment variables to ensure consistent behavior across the codebase.
"""

import platform
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


def _unlink_compacted(path: Path) -> None:
    """Remove a .compact_new file, ignoring Windows handle-lock errors.

    On Windows the OS handle may still be held by DuckDB's internal
    cleanup threads after the connection closes.  A PermissionError at
    this point is expected and harmless -- the stale file will be
    overwritten on the next compaction cycle.
    """
    if IS_WINDOWS:
        try:
            path.unlink(missing_ok=True)
        except PermissionError:
            pass
    else:
        path.unlink(missing_ok=True)



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
