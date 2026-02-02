"""Windows compatibility utilities for tests."""

import gc
import os
import tempfile
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from loguru import logger

from chunkhound.utils.windows_constants import (
    IS_WINDOWS,
    WINDOWS_DB_CLEANUP_DELAY,
    WINDOWS_RETRY_DELAY,
)


def is_windows() -> bool:
    """Check if running on Windows."""
    return IS_WINDOWS


def normalize_path_for_comparison(path: str | Path) -> str:
    """Normalize path for cross-platform comparison.
    
    On Windows, resolves short path names (8.3 format) to full paths.
    """
    path_obj = Path(path)
    try:
        # Resolve to get the canonical path (handles symlinks and relative paths)
        resolved = path_obj.resolve()
        return str(resolved)
    except Exception:
        # Fallback to string conversion if resolve fails
        return str(path_obj)


def paths_equal(path1: str | Path, path2: str | Path) -> bool:
    """Compare two paths for equality, handling Windows short paths."""
    norm1 = normalize_path_for_comparison(path1)
    norm2 = normalize_path_for_comparison(path2)

    # On Windows, also compare case-insensitive
    if is_windows():
        return norm1.lower() == norm2.lower()
    return norm1 == norm2


def path_contains(parent: str | Path, child: str | Path) -> bool:
    """Check if parent path contains child path, handling Windows short paths."""
    parent_norm = normalize_path_for_comparison(parent)
    child_norm = normalize_path_for_comparison(child)

    if is_windows():
        return child_norm.lower().startswith(parent_norm.lower())
    return child_norm.startswith(parent_norm)


@contextmanager
def database_cleanup_context(provider: Any = None) -> Generator[None, None, None]:
    """Context manager for proper database cleanup on Windows.
    
    Args:
        provider: Database provider to cleanup (optional)
    """
    try:
        yield
    finally:
        cleanup_database_resources(provider)


def cleanup_database_resources(provider: Any = None) -> None:
    """Cleanup database resources with Windows-specific handling.
    
    Args:
        provider: Database provider to cleanup (optional)
    """
    try:
        # Close database provider if provided
        if provider is not None:
            if hasattr(provider, 'close'):
                provider.close()
            elif hasattr(provider, 'disconnect'):
                provider.disconnect()
            # Note: Some tests may use close() instead - prefer that when available

        # Force garbage collection to release resources
        gc.collect()

        # Windows-specific: Additional delay for file handle release
        if is_windows():
            time.sleep(WINDOWS_DB_CLEANUP_DELAY)

    except Exception as e:
        logger.error(f"Error during database cleanup: {e}")


@contextmanager
def windows_safe_tempdir() -> Generator[Path, None, None]:
    """Create a temporary directory with Windows-safe cleanup.
    
    Uses database cleanup utilities to ensure proper resource cleanup
    before attempting to delete the directory.
    """
    temp_dir = None
    try:
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
    finally:
        if temp_dir and temp_dir.exists():
            try:
                # Cleanup any database resources first
                cleanup_database_resources()

                # Try to remove the directory
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)

                # On Windows, retry if removal failed
                if is_windows() and temp_dir.exists():
                    time.sleep(WINDOWS_RETRY_DELAY)  # Longer delay
                    shutil.rmtree(temp_dir, ignore_errors=True)

            except Exception as e:
                logger.error(f"Error cleaning up temp directory {temp_dir}: {e}")


def wait_for_file_release(file_path: Path, max_attempts: int = 10) -> bool:
    """Wait for a file to be released on Windows.
    
    Args:
        file_path: Path to file to check
        max_attempts: Maximum number of attempts
        
    Returns:
        True if file was released, False if still locked
    """
    if not is_windows():
        return True

    for attempt in range(max_attempts):
        try:
            # Try to rename the file (this will fail if locked)
            test_path = file_path.with_suffix(f"{file_path.suffix}.test")
            file_path.rename(test_path)
            test_path.rename(file_path)
            return True
        except (OSError, PermissionError):
            if attempt < max_attempts - 1:
                time.sleep(0.1 * (attempt + 1))  # Exponential backoff
            continue

    return False


def force_close_database_files(db_path: Path) -> None:
    """Force close database files on Windows.

    Args:
        db_path: Path to database file or directory
    """
    try:
        if db_path.is_file():
            wait_for_file_release(db_path)
        elif db_path.is_dir():
            # Check all database files in directory
            for db_file in db_path.glob("*.db"):
                wait_for_file_release(db_file)
    except Exception as e:
        logger.error(f"Error force-closing database files at {db_path}: {e}")


def is_ci() -> bool:
    """Check if running in CI environment."""
    return bool(os.environ.get("CI"))


def should_use_polling() -> bool:
    """Returns True if tests should use polling mode instead of watchdog.

    Windows CI has unreliable ReadDirectoryChangesW events that can silently
    drop filesystem events, causing tests to hang or fail.
    """
    return is_windows() and is_ci()


def get_fs_event_timeout() -> float:
    """Get appropriate timeout for filesystem event detection.

    Returns longer timeouts on Windows CI where ReadDirectoryChangesW
    can be unreliable.
    """
    if is_ci():
        return 10.0 if IS_WINDOWS else 5.0
    return 3.0


async def wait_for_indexed(
    provider,
    file_path,
    timeout: float | None = None,
    poll_interval: float = 0.2
) -> bool:
    """Wait for file to appear in database index.

    Uses polling instead of fixed sleep to handle Windows CI flakiness
    where ReadDirectoryChangesW may silently drop events.

    Args:
        provider: Database provider with get_file_by_path method
        file_path: Path to file that should be indexed
        timeout: Max wait time (defaults to platform-appropriate value)
        poll_interval: Time between checks

    Returns:
        True if file was found, False on timeout
    """
    import asyncio

    if timeout is None:
        timeout = get_fs_event_timeout()

    path_str = str(Path(file_path).resolve())
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        record = provider.get_file_by_path(path_str)
        if record is not None:
            return True
        await asyncio.sleep(poll_interval)

    return False


def wait_for_indexed_sync(
    provider,
    file_path,
    timeout: float | None = None,
    poll_interval: float = 0.2
) -> bool:
    """Sync version of wait_for_indexed."""
    if timeout is None:
        timeout = get_fs_event_timeout()

    path_str = str(Path(file_path).resolve())
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        record = provider.get_file_by_path(path_str)
        if record is not None:
            return True
        time.sleep(poll_interval)

    return False


async def wait_for_removed(
    provider,
    file_path,
    timeout: float | None = None,
    poll_interval: float = 0.2
) -> bool:
    """Wait for file to be removed from database index.

    Uses polling instead of fixed sleep to handle Windows CI flakiness.

    Args:
        provider: Database provider with get_file_by_path method
        file_path: Path to file that should be removed
        timeout: Max wait time (defaults to platform-appropriate value)
        poll_interval: Time between checks

    Returns:
        True if file was removed, False on timeout
    """
    import asyncio

    if timeout is None:
        timeout = get_fs_event_timeout()

    path_str = str(Path(file_path).resolve())
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        record = provider.get_file_by_path(path_str)
        if record is None:
            return True
        await asyncio.sleep(poll_interval)

    return False


async def wait_for_regex_searchable(
    services,
    query: str,
    timeout: float | None = None,
    poll_interval: float = 0.5
) -> bool:
    """Wait for content to be regex-searchable in the index.

    Polls regex search results instead of relying on queue state,
    handling the polling monitor timing gap on Windows CI.

    Note: This helper only supports regex search. Semantic search requires
    embedding_manager threading which is test-specific.

    Args:
        services: The services object with provider access
        query: Regex pattern to poll for
        timeout: Max wait time (defaults to platform-appropriate value)
        poll_interval: Time between search polls

    Returns:
        True if content became searchable, False on timeout
    """
    import asyncio

    from chunkhound.mcp_server.tools import execute_tool

    if timeout is None:
        timeout = get_fs_event_timeout()

    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        results = await execute_tool("search", services, None, {
            "type": "regex",
            "query": query,
            "page_size": 10,
            "offset": 0
        })
        if len(results.get('results', [])) > 0:
            return True
        await asyncio.sleep(poll_interval)

    return False
