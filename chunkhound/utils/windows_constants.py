"""Windows-specific constants and utilities for ChunkHound.

This module centralizes all Windows-specific configuration, delays, and
environment variables to ensure consistent behavior across the codebase.
"""

import platform
from pathlib import Path

# Platform detection (cached for performance)
IS_WINDOWS = platform.system() == "Windows"
IS_MACOS = platform.system() == "Darwin"

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


def fsync_path(path: Path | str) -> None:
    """Fsync a file and its parent directory for cross-platform durability.

    Ensures data is flushed to disk before subsequent reads (especially mmap).

    - Linux: fsync file + fsync parent directory
    - macOS: F_FULLFSYNC (flushes disk internal buffers)
    - Windows: FlushFileBuffers via os.fsync

    Args:
        path: Path to the file to sync
    """
    import os

    path = Path(path)

    # Fsync the file itself
    fd = os.open(str(path), os.O_RDONLY)
    try:
        if IS_MACOS:
            import fcntl

            fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
        else:
            os.fsync(fd)
    finally:
        os.close(fd)

    # Fsync parent directory (POSIX only - ensures filename is durable)
    if not IS_WINDOWS:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
