"""Path resolution service for multi-repository support.

This service resolves query paths (relative or absolute) against indexed base directories,
enabling cross-repository searches in global database mode.
"""

import os
from collections import OrderedDict
from pathlib import Path
from threading import RLock
from typing import Any

from loguru import logger

from chunkhound.interfaces.database_provider import DatabaseProvider
from chunkhound.services.base_service import BaseService


class PathResolver(BaseService):
    """Resolves query paths against indexed base directories.

    In multi-repo mode, a single database may contain files from multiple base directories.
    This service maps query paths to their containing base_directory for proper filtering.

    Thread-safe with LRU cache for performance.

    Examples:
        >>> resolver = PathResolver(provider)
        >>> # Absolute path query - finds containing base
        >>> resolver.resolve_path("/home/user/project1/src/main.py", cwd="/home/user/project2")
        "/home/user/project1/src/main.py"
        >>> # Relative path query - resolves against CWD's base
        >>> resolver.resolve_path("src/main.py", cwd="/home/user/project1")
        "/home/user/project1/src/main.py"
    """

    MAX_CACHE_SIZE = 1000  # LRU cache size limit (prevents unbounded growth)

    def __init__(self, database_provider: DatabaseProvider):
        """Initialize path resolver with database provider.

        Args:
            database_provider: Database provider for querying indexed_roots
        """
        super().__init__(database_provider)
        # Thread-safe LRU cache: path -> base_directory
        self._base_cache: OrderedDict[str, str | None] = OrderedDict()
        self._cache_lock = RLock()  # Reentrant lock for nested calls

    def resolve_path(
        self, query_path: str, current_working_dir: str | None = None
    ) -> str:
        """Resolve a query path to an absolute path for database filtering.

        This method handles both absolute and relative paths:
        - Absolute paths: Find which indexed base_directory contains the path
        - Relative paths: Resolve against the base_directory of current_working_dir

        Args:
            query_path: Path from search query (relative or absolute)
            current_working_dir: Current working directory for relative path resolution

        Returns:
            Absolute path for database lookup, or original path if unresolved

        Raises:
            ValueError: If relative path provided without current_working_dir
        """
        # Fast path: already absolute and in cache
        if os.path.isabs(query_path):
            return self._resolve_absolute_path(query_path)

        # Relative path requires CWD
        if current_working_dir is None:
            raise ValueError(
                f"Cannot resolve relative path '{query_path}' without current_working_dir"
            )

        return self._resolve_relative_path(query_path, current_working_dir)

    def _resolve_absolute_path(self, abs_path: str) -> str:
        """Resolve an absolute path by finding its containing base_directory.

        Canonicalizes paths (resolves symlinks) for security and accuracy.
        Thread-safe with LRU cache.

        Args:
            abs_path: Absolute path to resolve

        Returns:
            Canonical absolute path (validated against indexed_roots) or original if not found
        """
        # Canonicalize path (resolve symlinks, normalize)
        try:
            canonical_path = os.path.realpath(abs_path)
        except (OSError, ValueError) as e:
            logger.warning(f"Failed to canonicalize path {abs_path}: {e}")
            canonical_path = abs_path  # Fallback to original

        # Check cache first (thread-safe)
        cached_base = self._cache_get(canonical_path)
        if cached_base is not None:
            # Found in cache (even if value is empty string meaning "no base")
            if cached_base:
                logger.debug(f"Cache hit for path: {canonical_path} -> {cached_base}")
            else:
                logger.debug(f"Cache hit (no base) for path: {canonical_path}")
            return canonical_path

        # Query database for containing base_directory (outside lock to avoid holding during DB I/O)
        try:
            # Use provider's find_base_for_path method from Phase 1
            containing_base = self._find_base_for_path(canonical_path)

            if containing_base:
                logger.debug(
                    f"Resolved path {canonical_path} to base {containing_base}"
                )
                # Store in cache with canonical path as key
                self._cache_set(canonical_path, containing_base)
                return canonical_path
            else:
                logger.warning(
                    f"Path {canonical_path} not under any indexed base_directory"
                )
                # Cache negative result (empty string = "no base found")
                self._cache_set(canonical_path, "")
                return canonical_path

        except Exception as e:
            logger.error(f"Error resolving path {canonical_path}: {e}")
            return canonical_path

    def _resolve_relative_path(self, rel_path: str, cwd: str) -> str:
        """Resolve a relative path against current working directory's base.

        Args:
            rel_path: Relative path from query
            cwd: Current working directory

        Returns:
            Absolute path constructed from base_directory + relative path
        """
        # Find base_directory that contains the CWD
        cwd_base = self._find_base_for_path(cwd)

        if cwd_base is None:
            logger.warning(
                f"CWD {cwd} not under any indexed base_directory, using CWD as base"
            )
            # Fallback: join with CWD directly
            abs_path = str(Path(cwd) / rel_path)
            return abs_path

        # Join relative path with CWD's base_directory
        abs_path = str(Path(cwd_base) / rel_path)
        logger.debug(f"Resolved relative path {rel_path} (cwd={cwd}) to {abs_path}")
        return abs_path

    def _find_base_for_path(self, path: str) -> str | None:
        """Find which indexed base_directory contains the given path.

        This wraps the provider's find_base_for_path() method and adds caching.

        Args:
            path: Absolute path to search for

        Returns:
            Base directory containing the path, or None if not found
        """
        # Convert to absolute path if needed
        abs_path = path if os.path.isabs(path) else os.path.abspath(path)

        # Check cache first using thread-safe method
        cached = self._cache_get(abs_path)
        if cached is not None:
            # Empty string means "no base found" (cached negative result)
            return cached if cached else None

        # Query provider (calls _executor_find_base_for_path from Phase 1)
        try:
            base_dir = self.database.find_base_for_path(abs_path)
            # Cache result (use empty string for None to distinguish from "not cached")
            self._cache_set(abs_path, base_dir or "")

            if base_dir:
                logger.debug(f"Found base directory {base_dir} for path {abs_path}")
            else:
                logger.debug(f"No base directory found for path {abs_path}")

            return base_dir

        except AttributeError:
            # Provider doesn't support find_base_for_path (legacy mode)
            logger.warning(
                "Database provider does not support find_base_for_path, "
                "multi-repo path resolution disabled"
            )
            return None

    def get_indexed_roots(self) -> list[dict[str, Any]]:
        """Get all indexed base directories from the database.

        Returns:
            List of indexed root records with metadata (base_directory, project_name, etc.)
        """
        try:
            return self.database.get_indexed_roots()
        except AttributeError:
            # Provider doesn't support get_indexed_roots (legacy mode)
            logger.warning(
                "Database provider does not support get_indexed_roots, "
                "returning empty list"
            )
            return []

    def _cache_get(self, key: str) -> str | None:
        """Thread-safe LRU cache get.

        Args:
            key: Path to lookup

        Returns:
            Cached base_directory, empty string if path has no base, or None if not cached
        """
        with self._cache_lock:
            if key in self._base_cache:
                # Move to end (most recently used)
                self._base_cache.move_to_end(key)
                return self._base_cache[key]
            return None

    def _cache_set(self, key: str, value: str) -> None:
        """Thread-safe LRU cache set with eviction.

        Args:
            key: Path to cache
            value: Base directory (or empty string for "no base")
        """
        with self._cache_lock:
            if key in self._base_cache:
                # Update existing entry and move to end
                self._base_cache[key] = value
                self._base_cache.move_to_end(key)
            else:
                # Add new entry
                self._base_cache[key] = value
                # Evict oldest if over limit (FIFO eviction)
                if len(self._base_cache) > self.MAX_CACHE_SIZE:
                    evicted_key = self._base_cache.popitem(last=False)[0]
                    logger.debug(f"Cache eviction: removed {evicted_key}")

    def clear_cache(self) -> None:
        """Clear the base_directory lookup cache.

        Thread-safe. Useful when indexed_roots table is updated (e.g., after re-indexing).
        """
        with self._cache_lock:
            self._base_cache.clear()
        logger.debug("PathResolver cache cleared")

    def get_cache_stats(self) -> dict[str, int | float]:
        """Get cache performance statistics.

        Returns:
            Dictionary with cache size and capacity
        """
        with self._cache_lock:
            return {
                "size": len(self._base_cache),
                "capacity": self.MAX_CACHE_SIZE,
                "utilization": len(self._base_cache) / self.MAX_CACHE_SIZE,
            }
