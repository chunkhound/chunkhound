"""
File Discovery Cache for optimizing glob operations.

This module provides caching for expensive filesystem operations,
specifically glob pattern matching and file filtering.
"""

import time
from collections import OrderedDict
from fnmatch import fnmatch
from pathlib import Path

from loguru import logger


class FileDiscoveryCache:
    """Cache for file discovery operations to reduce filesystem overhead."""

    def __init__(self, max_entries: int = 100, ttl_seconds: int = 300):
        """Initialize the cache.

        Args:
            max_entries: Maximum number of cache entries to keep (LRU eviction)
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds

        # Cache structure: key -> (result, timestamp, directory_mtime)
        self._cache: OrderedDict[str, tuple[list[Path], float, float]] = OrderedDict()

        # Statistics for monitoring
        self.stats = {"hits": 0, "misses": 0, "evictions": 0, "invalidations": 0}

    def get_files(
        self,
        directory: Path,
        patterns: list[str],
        exclude_patterns: list[str] | None = None,
    ) -> list[Path]:
        """Get files matching patterns with caching.

        Args:
            directory: Directory to search
            patterns: List of glob patterns to match
            exclude_patterns: List of glob patterns to exclude

        Returns:
            List of matching file paths
        """
        # Create cache key from directory and patterns
        cache_key = self._make_cache_key(directory, patterns, exclude_patterns)

        # Check cache first
        cached_result = self._get_from_cache(cache_key, directory)
        if cached_result is not None:
            self.stats["hits"] += 1
            logger.debug(f"Cache hit for {directory} with {len(patterns)} patterns")
            return cached_result

        # Cache miss - perform file discovery
        self.stats["misses"] += 1
        logger.debug(f"Cache miss for {directory} with {len(patterns)} patterns")

        # Discover files
        files = self._discover_files(directory, patterns, exclude_patterns)

        # Store in cache
        self._store_in_cache(cache_key, files, directory)

        return files

    def invalidate_directory(self, directory: Path) -> int:
        """Invalidate all cache entries for a directory.

        Args:
            directory: Directory to invalidate

        Returns:
            Number of entries invalidated
        """
        dir_str = str(directory)
        keys_to_remove = []

        for key in self._cache:
            if key.startswith(f"{dir_str}|"):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._cache[key]
            self.stats["invalidations"] += 1

        logger.debug(f"Invalidated {len(keys_to_remove)} cache entries for {directory}")
        return len(keys_to_remove)

    def clear(self) -> None:
        """Clear all cache entries."""
        count = len(self._cache)
        self._cache.clear()
        self.stats["evictions"] += count
        logger.debug(f"Cleared {count} cache entries")

    def get_stats(self) -> dict[str, int]:
        """Get cache statistics.

        Returns:
            Dictionary with cache hit/miss statistics
        """
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (
            (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        )

        return {
            **self.stats,
            "cache_size": len(self._cache),
            "hit_rate_percent": int(round(hit_rate, 2)),
        }

    def _make_cache_key(
        self, directory: Path, patterns: list[str], exclude_patterns: list[str] | None
    ) -> str:
        """Create a cache key from directory and patterns.

        Args:
            directory: Directory path
            patterns: Include patterns
            exclude_patterns: Exclude patterns

        Returns:
            Cache key string
        """
        patterns_str = "|".join(sorted(patterns))
        exclude_str = "|".join(sorted(exclude_patterns or []))
        return f"{directory}|{patterns_str}|{exclude_str}"

    def _get_from_cache(self, cache_key: str, directory: Path) -> list[Path] | None:
        """Get entry from cache if valid.

        Args:
            cache_key: Cache key to lookup
            directory: Directory path for mtime checking

        Returns:
            Cached file list if valid, None otherwise
        """
        if cache_key not in self._cache:
            return None

        files, timestamp, cached_mtime = self._cache[cache_key]
        current_time = time.time()

        # Check TTL expiration
        if current_time - timestamp > self.ttl_seconds:
            del self._cache[cache_key]
            self.stats["evictions"] += 1
            logger.debug(f"Cache entry expired for key: {cache_key}")
            return None

        # Check directory modification time
        try:
            current_mtime = directory.stat().st_mtime
            if current_mtime > cached_mtime:
                del self._cache[cache_key]
                self.stats["invalidations"] += 1
                logger.debug(
                    f"Cache entry invalidated (directory modified): {cache_key}"
                )
                return None
        except OSError:
            # Directory might not exist anymore
            del self._cache[cache_key]
            self.stats["invalidations"] += 1
            return None

        # Move to end (LRU)
        self._cache.move_to_end(cache_key)
        return files

    def _store_in_cache(
        self, cache_key: str, files: list[Path], directory: Path
    ) -> None:
        """Store files in cache.

        Args:
            cache_key: Cache key
            files: List of file paths to cache
            directory: Directory path for mtime tracking
        """
        try:
            directory_mtime = directory.stat().st_mtime
        except OSError:
            # Can't cache if we can't get mtime
            return

        # Enforce max entries (LRU eviction)
        while len(self._cache) >= self.max_entries:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            self.stats["evictions"] += 1

        # Store entry
        self._cache[cache_key] = (files, time.time(), directory_mtime)
        logger.debug(f"Cached {len(files)} files for key: {cache_key}")

    def _discover_files(
        self, directory: Path, patterns: list[str], exclude_patterns: list[str] | None
    ) -> list[Path]:
        """Perform actual file discovery.

        Args:
            directory: Directory to search
            patterns: Include patterns
            exclude_patterns: Exclude patterns

        Returns:
            List of matching file paths
        """
        try:
            # Find all matching files from all patterns
            files: list[Path] = []
            for pattern in patterns:
                files.extend(directory.glob(pattern))

            # Remove duplicates while preserving order
            seen = set()
            unique_files = []
            for file_path in files:
                if file_path not in seen:
                    seen.add(file_path)
                    unique_files.append(file_path)
            files = unique_files

            # Filter out excluded files
            if exclude_patterns:
                filtered_files = []
                for file_path in files:
                    # Convert to relative path from directory for pattern matching
                    rel_path = file_path.relative_to(directory)
                    excluded = False
                    for exclude_pattern in exclude_patterns:
                        if fnmatch(str(rel_path), exclude_pattern) or fnmatch(
                            str(file_path), exclude_pattern
                        ):
                            excluded = True
                            break
                    if not excluded:
                        filtered_files.append(file_path)
                files = filtered_files

            return files

        except Exception as e:
            logger.error(f"Failed to discover files in {directory}: {e}")
            return []
