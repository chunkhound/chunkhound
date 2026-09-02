"""Bounded vector cache for transient semantic search.

Mutations are serialized with a threading lock so a shared process cache is
safe across threads. Async callers on one event loop do not await inside
get/put, so they never hold the lock across a yield.
"""

import threading
import time
from collections import OrderedDict

from chunkhound.utils.hashing import compute_text_hash


class VectorCache:
    """Store vectors by text digest namespaced to an embedding space."""

    def __init__(self, max_entries: int = 10_000, ttl_seconds: int = 300) -> None:
        self.max_entries = max(0, max_entries)
        self.ttl_seconds = max(0, ttl_seconds)
        self._entries: OrderedDict[str, tuple[list[float], float]] = OrderedDict()
        self._lock = threading.Lock()

    def _key(self, text: str, provider: str, model: str, dims: int) -> str:
        return f"{provider}\0{model}\0{dims}\0{compute_text_hash(text)}"

    def get(
        self,
        text: str,
        *,
        provider: str = "",
        model: str = "",
        dims: int = 0,
    ) -> list[float] | None:
        """Return a cached vector and refresh its LRU position."""
        key = self._key(text, provider, model, dims)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None

            vector, stored_at = entry
            if time.monotonic() - stored_at >= self.ttl_seconds:
                del self._entries[key]
                return None

            self._entries.move_to_end(key)
            return vector.copy()

    def put(
        self,
        text: str,
        vector: list[float],
        *,
        provider: str = "",
        model: str = "",
        dims: int = 0,
    ) -> None:
        """Cache a vector unless caching is disabled."""
        if self.max_entries == 0:
            return

        key = self._key(text, provider, model, dims)
        with self._lock:
            self._entries[key] = (vector.copy(), time.monotonic())
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
