"""Tree cache module for ChunkHound - LRU caching of parsed syntax trees."""

import time
from collections import OrderedDict
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tree_sitter import Tree

    TreeSitterTree = Tree
else:
    TreeSitterTree = Any

from loguru import logger


class TreeCacheEntry:
    """Represents a cached syntax tree with metadata."""

    def __init__(self, tree: Any, file_path: Path, mtime: float, size: int):
        """Initialize cache entry.

        Args:
            tree: Parsed tree-sitter syntax tree
            file_path: Path to the source file
            mtime: File modification time when parsed
            size: File size in bytes when parsed
        """
        self.tree = tree
        self.file_path = file_path
        self.mtime = mtime
        self.size = size
        self.access_time = time.time()
        self.hit_count = 0

    def is_valid(self) -> bool:
        """Check if cache entry is still valid based on file modification time and size.

        Returns:
            True if file hasn't changed, False if stale
        """
        try:
            stat = self.file_path.stat()
            return stat.st_mtime == self.mtime and stat.st_size == self.size
        except (OSError, FileNotFoundError):
            # File no longer exists or inaccessible
            return False

    def touch(self) -> None:
        """Update access time and increment hit count."""
        self.access_time = time.time()
        self.hit_count += 1


class TreeCache:
    """LRU cache for parsed syntax trees with automatic invalidation."""

    def __init__(self, max_entries: int = 1000, max_memory_mb: int = 500):
        """Initialize tree cache.

        Args:
            max_entries: Maximum number of cached trees
            max_memory_mb: Approximate memory limit in MB (rough estimate)
        """
        self.max_entries = max_entries
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._cache: OrderedDict[str, TreeCacheEntry] = OrderedDict()
        self._lock = RLock()  # Thread-safe operations

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._invalidations = 0

        logger.debug(
            f"TreeCache initialized: "
            f"max_entries={max_entries}, "
            f"max_memory_mb={max_memory_mb}"
        )

    def get(self, file_path: Path) -> Any | None:
        """Get cached syntax tree for file.

        Args:
            file_path: Path to source file

        Returns:
            Cached syntax tree if valid, None if not cached or stale
        """
        cache_key = str(file_path.resolve())

        with self._lock:
            if cache_key not in self._cache:
                self._misses += 1
                logger.debug(f"Cache miss: {file_path}")
                return None

            entry = self._cache[cache_key]

            # Check if entry is still valid
            if not entry.is_valid():
                logger.debug(f"Cache invalidation (stale): {file_path}")
                del self._cache[cache_key]
                self._invalidations += 1
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(cache_key)
            entry.touch()
            self._hits += 1

            logger.debug(f"Cache hit: {file_path} (hits: {entry.hit_count})")
            return entry.tree

    def get_for_comparison(self, file_path: Path) -> Any | None:
        """Get cached syntax tree for comparison, even if stale.

        This method returns cached trees without validating freshness,
        useful for incremental parsing where we need to compare old vs new trees.

        Args:
            file_path: Path to source file

        Returns:
            Cached syntax tree if available, None if not cached
        """
        cache_key = str(file_path.resolve())

        with self._lock:
            if cache_key not in self._cache:
                self._misses += 1
                logger.debug(f"Cache miss for comparison: {file_path}")
                return None

            entry = self._cache[cache_key]

            # Return tree without validation for comparison purposes
            logger.debug(f"Cache hit for comparison (potentially stale): {file_path}")
            return entry.tree

    def put(self, file_path: Path, tree: Any) -> None:
        """Cache a parsed syntax tree.

        Args:
            file_path: Path to source file
            tree: Parsed tree-sitter syntax tree
        """
        if tree is None:
            logger.warning(f"Attempted to cache None tree for {file_path}")
            return

        cache_key = str(file_path.resolve())

        try:
            stat = file_path.stat()
            mtime = stat.st_mtime
            size = stat.st_size
        except (OSError, FileNotFoundError) as e:
            logger.warning(f"Cannot stat file for caching {file_path}: {e}")
            return

        with self._lock:
            # Create new entry
            entry = TreeCacheEntry(tree, file_path, mtime, size)

            # If key already exists, remove old entry
            if cache_key in self._cache:
                del self._cache[cache_key]

            # Add new entry
            self._cache[cache_key] = entry

            # Enforce cache limits
            self._enforce_limits()

            logger.debug(f"Cached tree: {file_path} (cache size: {len(self._cache)})")

    def invalidate(self, file_path: Path) -> bool:
        """Invalidate cached entry for a file.

        Args:
            file_path: Path to source file

        Returns:
            True if entry was found and removed, False otherwise
        """
        cache_key = str(file_path.resolve())

        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                self._invalidations += 1
                logger.debug(f"Invalidated cache entry: {file_path}")
                return True
            return False

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"Cleared tree cache ({count} entries)")

    def _enforce_limits(self) -> None:
        """Enforce cache size and memory limits using LRU eviction."""
        # Enforce entry count limit
        while len(self._cache) > self.max_entries:
            self._evict_lru()

        # Rough memory enforcement (estimated)
        # Each tree is roughly estimated based on file size
        estimated_memory = sum(entry.size for entry in self._cache.values())
        while estimated_memory > self.max_memory_bytes and self._cache:
            evicted_entry = self._evict_lru()
            if evicted_entry:
                estimated_memory -= evicted_entry.size

    def _evict_lru(self) -> TreeCacheEntry | None:
        """Evict least recently used entry.

        Returns:
            Evicted entry, or None if cache is empty
        """
        if not self._cache:
            return None

        # OrderedDict maintains insertion order, so first item is LRU
        cache_key, entry = self._cache.popitem(last=False)
        self._evictions += 1
        logger.debug(f"Evicted LRU entry: {entry.file_path}")
        return entry

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (
                (self._hits / total_requests * 100) if total_requests > 0 else 0.0
            )

            return {
                "entries": len(self._cache),
                "max_entries": self.max_entries,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_percent": round(hit_rate, 2),
                "evictions": self._evictions,
                "invalidations": self._invalidations,
                "total_requests": total_requests,
                "estimated_memory_mb": round(
                    sum(entry.size for entry in self._cache.values()) / 1024 / 1024, 2
                ),
                "max_memory_mb": round(self.max_memory_bytes / 1024 / 1024, 2),
            }

    def print_stats(self) -> None:
        """Print cache statistics to logger."""
        stats = self.get_stats()
        logger.info(
            f"TreeCache Stats: {stats['entries']}/{stats['max_entries']} entries, "
            f"{stats['hit_rate_percent']}% hit rate, "
            f"{stats['estimated_memory_mb']}/{stats['max_memory_mb']} MB"
        )

    def cleanup_stale_entries(self) -> int:
        """Remove all stale entries (files that have been modified or deleted).

        Returns:
            Number of stale entries removed
        """
        stale_keys = []

        with self._lock:
            for cache_key, entry in self._cache.items():
                if not entry.is_valid():
                    stale_keys.append(cache_key)

            for key in stale_keys:
                del self._cache[key]
                self._invalidations += 1

        if stale_keys:
            logger.info(f"Cleaned up {len(stale_keys)} stale cache entries")

        return len(stale_keys)

    def get_cache_info(self, file_path: Path) -> dict[str, Any] | None:
        """Get detailed information about a cached entry.

        Args:
            file_path: Path to source file

        Returns:
            Dictionary with cache entry info, or None if not cached
        """
        cache_key = str(file_path.resolve())

        with self._lock:
            if cache_key not in self._cache:
                return None

            entry = self._cache[cache_key]
            return {
                "file_path": str(entry.file_path),
                "cached_mtime": entry.mtime,
                "cached_size": entry.size,
                "access_time": entry.access_time,
                "hit_count": entry.hit_count,
                "is_valid": entry.is_valid(),
                "age_seconds": time.time() - entry.access_time,
            }


# Global cache instance - can be configured by applications
_default_cache: TreeCache | None = None


def get_default_cache() -> TreeCache:
    """Get or create the default global tree cache instance.

    Returns:
        Global TreeCache instance
    """
    global _default_cache
    if _default_cache is None:
        _default_cache = TreeCache()
    return _default_cache


def configure_default_cache(
    max_entries: int = 1000, max_memory_mb: int = 500
) -> TreeCache:
    """Configure the default global tree cache.

    Args:
        max_entries: Maximum number of cached trees
        max_memory_mb: Approximate memory limit in MB

    Returns:
        Configured global TreeCache instance
    """
    global _default_cache
    _default_cache = TreeCache(max_entries, max_memory_mb)
    return _default_cache
