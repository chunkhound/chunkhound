"""Bounded LRU + TTL + byte-budget cache for parsed git diff chunk batches."""

from __future__ import annotations

import hashlib
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

from chunkhound.core.models.chunk import Chunk
from chunkhound.services.vector_cache import (
    CACHE_MAX_ENTRIES_ENV,
    CACHE_TTL_ENV,
    _env_int,
)

DEFAULT_CHUNK_CACHE_MAX_BYTES = 128 * 1024 * 1024  # 128 MiB
_CHUNK_OVERHEAD_BYTES = 256
_BATCH_OVERHEAD_BYTES = 64
_ENTRY_OVERHEAD_BYTES = 128

DiffChunkCacheKey = tuple[str, str, str, str, int, int, str]


@dataclass
class _CacheEntry:
    batches: tuple[tuple[Chunk, ...], ...]
    size_bytes: int
    last_accessed: float


def _estimate_chunk_bytes(chunk: Chunk) -> int:
    """Estimate memory footprint of one parsed Chunk."""
    total = _CHUNK_OVERHEAD_BYTES + sys.getsizeof(chunk.code)
    if chunk.symbol is not None:
        total += sys.getsizeof(chunk.symbol)
    if chunk.file_path is not None:
        total += sys.getsizeof(chunk.file_path)
    if chunk.parent_header is not None:
        total += sys.getsizeof(chunk.parent_header)
    return total


def _estimate_batches_bytes(batches: tuple[tuple[Chunk, ...], ...]) -> int:
    total = _ENTRY_OVERHEAD_BYTES
    for batch in batches:
        total += _BATCH_OVERHEAD_BYTES
        for chunk in batch:
            total += _estimate_chunk_bytes(chunk)
    return total


def _digest_key(key: DiffChunkCacheKey) -> str:
    joined = "\0".join(str(part) for part in key)
    return hashlib.sha256(joined.encode("utf-8", errors="surrogateescape")).hexdigest()


class DiffChunkCache:
    """In-memory LRU + TTL + byte-budget cache for completed diff chunk streams.

    Keyed on resolved repository identity, range operator, start/end object IDs,
    chunk size limit, batch size, and parser pipeline version. Reuses the vector
    cache environment knobs (``CHUNKHOUND_VECTOR_CACHE_MAX_ENTRIES`` and
    ``CHUNKHOUND_VECTOR_CACHE_TTL_SECONDS``).
    """

    def __init__(
        self,
        *,
        max_entries: int,
        ttl_seconds: int,
        max_bytes: int = DEFAULT_CHUNK_CACHE_MAX_BYTES,
        follow_env: bool = False,
    ) -> None:
        self._max_entries = max(0, max_entries)
        self._ttl_seconds = max(0, ttl_seconds)
        self._max_bytes = max(0, max_bytes)
        self._follow_env = follow_env
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._total_bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    @classmethod
    def from_env(
        cls, *, max_bytes: int = DEFAULT_CHUNK_CACHE_MAX_BYTES
    ) -> DiffChunkCache:
        return cls(
            max_entries=_env_int(CACHE_MAX_ENTRIES_ENV, 10_000),
            ttl_seconds=_env_int(CACHE_TTL_ENV, 1_800),
            max_bytes=max_bytes,
            follow_env=True,
        )

    def _effective_limits(self) -> tuple[int, int]:
        if self._follow_env:
            return (
                max(0, _env_int(CACHE_MAX_ENTRIES_ENV, self._max_entries)),
                max(0, _env_int(CACHE_TTL_ENV, self._ttl_seconds)),
            )
        return self._max_entries, self._ttl_seconds

    def is_enabled(self) -> bool:
        max_entries, ttl_seconds = self._effective_limits()
        return max_entries > 0 and ttl_seconds > 0 and self._max_bytes > 0

    def _evict_expired(self, now: float, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            self._entries.clear()
            self._total_bytes = 0
            return
        expired = [
            key
            for key, entry in self._entries.items()
            if now - entry.last_accessed > ttl_seconds
        ]
        for key in expired:
            entry = self._entries.pop(key)
            self._total_bytes -= entry.size_bytes

    def get(self, key: DiffChunkCacheKey) -> tuple[tuple[Chunk, ...], ...] | None:
        max_entries, ttl_seconds = self._effective_limits()
        digest = _digest_key(key)
        now = time.monotonic()
        with self._lock:
            if max_entries <= 0 or ttl_seconds <= 0 or self._max_bytes <= 0:
                self.misses += 1
                return None
            self._evict_expired(now, ttl_seconds)
            entry = self._entries.get(digest)
            if entry is None:
                self.misses += 1
                return None
            entry.last_accessed = now
            self._entries.move_to_end(digest)
            self.hits += 1
            return entry.batches

    def put(
        self, key: DiffChunkCacheKey, batches: tuple[tuple[Chunk, ...], ...]
    ) -> bool:
        max_entries, ttl_seconds = self._effective_limits()
        if max_entries <= 0 or ttl_seconds <= 0 or self._max_bytes <= 0:
            return False

        entry_bytes = _estimate_batches_bytes(batches)
        if entry_bytes > self._max_bytes:
            return False

        digest = _digest_key(key)
        now = time.monotonic()
        with self._lock:
            self._evict_expired(now, ttl_seconds)
            existing = self._entries.pop(digest, None)
            if existing is not None:
                self._total_bytes -= existing.size_bytes

            while self._entries and (
                len(self._entries) >= max_entries
                or self._total_bytes + entry_bytes > self._max_bytes
            ):
                _, evicted = self._entries.popitem(last=False)
                self._total_bytes -= evicted.size_bytes
                self.evictions += 1

            self._entries[digest] = _CacheEntry(
                batches=batches,
                size_bytes=entry_bytes,
                last_accessed=now,
            )
            self._total_bytes += entry_bytes
            return True

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._total_bytes = 0

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
