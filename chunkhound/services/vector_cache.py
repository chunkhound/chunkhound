"""Bounded vector cache for transient semantic search.

Vectors are stored as ``float32`` arrays (``dim * 4`` bytes) rather than Python
float lists, which cost ~8x more per entry once object overhead is counted. A
full 10_000-entry cache at dim=1536 is ~60MB instead of ~470MB.

Mutations are serialized with a threading lock so a shared process cache is
safe across threads. Async callers on one event loop do not await inside
get/put, so they never hold the lock across a yield.
"""

import os
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from chunkhound.utils.hashing import compute_text_hash

CACHE_MAX_ENTRIES_ENV = "CHUNKHOUND_VECTOR_CACHE_MAX_ENTRIES"
CACHE_TTL_ENV = "CHUNKHOUND_VECTOR_CACHE_TTL_SECONDS"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


class VectorCache:
    """Store vectors by text digest namespaced to an embedding space."""

    def __init__(self, max_entries: int = 10_000, ttl_seconds: int = 300) -> None:
        self.max_entries = max(0, max_entries)
        self.ttl_seconds = max(0, ttl_seconds)
        self._entries: OrderedDict[str, tuple[npt.NDArray[np.float32], float]] = (
            OrderedDict()
        )
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> "VectorCache":
        """Build a cache sized by env.

        ``CHUNKHOUND_VECTOR_CACHE_MAX_ENTRIES=0`` disables caching entirely.
        """
        return cls(
            max_entries=_env_int(CACHE_MAX_ENTRIES_ENV, 10_000),
            ttl_seconds=_env_int(CACHE_TTL_ENV, 300),
        )

    def _key(
        self,
        text: str,
        provider: str,
        model: str,
        dims: int,
        endpoint: str,
        deployment: str,
    ) -> str:
        namespace = "\0".join(
            ("v2", provider, endpoint, deployment, model, str(dims), "float32")
        )
        return f"{namespace}\0{compute_text_hash(text)}"

    def get(
        self,
        text: str,
        *,
        provider: str = "",
        model: str = "",
        dims: int = 0,
        endpoint: str = "",
        deployment: str = "",
    ) -> npt.NDArray[np.float32] | None:
        """Return a cached vector, refreshing its LRU position and TTL.

        TTL measures time since last access, not since insertion, so entries
        still being queried are never evicted out from under a live session.
        """
        key = self._key(text, provider, model, dims, endpoint, deployment)
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
        endpoint: str = "",
        deployment: str = "",
    ) -> None:
        """Cache a vector unless caching is disabled."""
        if self.max_entries == 0:
            return

        key = self._key(text, provider, model, dims, endpoint, deployment)
        stored = np.asarray(vector, dtype=np.float32).copy()
        with self._lock:
            self._entries[key] = (stored, time.monotonic())
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
