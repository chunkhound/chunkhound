"""Bounded vector cache for transient semantic search.

Vectors are stored as ``float32`` arrays (``dim * 4`` bytes) rather than Python
float lists, which cost ~8x more per entry once object overhead is counted. A
full 10_000-entry cache at dim=1536 is ~60MB instead of ~470MB.

Mutations are serialized with a threading lock so a shared process cache is
safe across threads. Async callers on one event loop do not await inside
get/put, so they never hold the lock across a yield.
"""

import threading
import time
from collections import OrderedDict
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from chunkhound.utils.hashing import compute_text_hash


class VectorCache:
    """Store vectors by text digest namespaced to an embedding space."""

    def __init__(self, max_entries: int = 10_000, ttl_seconds: int = 300) -> None:
        self.max_entries = max(0, max_entries)
        self.ttl_seconds = max(0, ttl_seconds)
        self._entries: OrderedDict[str, tuple[npt.NDArray[np.float32], float]] = (
            OrderedDict()
        )
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
    ) -> npt.NDArray[np.float32] | None:
        """Return a cached vector, refreshing its LRU position and TTL.

        TTL measures time since last access, not since insertion, so entries
        still being queried are never evicted out from under a live session.
        """
        key = self._key(text, provider, model, dims)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None

            vector, last_used = entry
            now = time.monotonic()
            if now - last_used >= self.ttl_seconds:
                del self._entries[key]
                return None

            self._entries[key] = (vector, now)
            self._entries.move_to_end(key)
            # Copy so a caller mutating the result cannot corrupt the entry.
            copied: npt.NDArray[np.float32] = vector.copy()
            return copied

    def put(
        self,
        text: str,
        vector: Sequence[float] | np.ndarray,
        *,
        provider: str = "",
        model: str = "",
        dims: int = 0,
    ) -> None:
        """Cache a vector unless caching is disabled."""
        if self.max_entries == 0:
            return

        key = self._key(text, provider, model, dims)
        stored = np.asarray(vector, dtype=np.float32).copy()
        with self._lock:
            self._entries[key] = (stored, time.monotonic())
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
