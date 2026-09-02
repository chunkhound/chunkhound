"""Bounded vector cache for transient semantic search."""

import time
from collections import OrderedDict

from chunkhound.utils.hashing import compute_text_hash


class VectorCache:
    """Store vectors by text digest with LRU eviction and expiry."""

    def __init__(self, max_entries: int = 10_000, ttl_seconds: int = 300) -> None:
        self.max_entries = max(0, max_entries)
        self.ttl_seconds = max(0, ttl_seconds)
        self._entries: OrderedDict[str, tuple[list[float], float]] = OrderedDict()

    def get(self, text: str) -> list[float] | None:
        """Return a cached vector and refresh its LRU position."""
        key = compute_text_hash(text)
        entry = self._entries.get(key)
        if entry is None:
            return None

        vector, stored_at = entry
        if time.monotonic() - stored_at >= self.ttl_seconds:
            del self._entries[key]
            return None

        self._entries.move_to_end(key)
        return vector.copy()

    def put(self, text: str, vector: list[float]) -> None:
        """Cache a vector unless caching is disabled."""
        if self.max_entries == 0:
            return

        key = compute_text_hash(text)
        self._entries[key] = (vector.copy(), time.monotonic())
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)
