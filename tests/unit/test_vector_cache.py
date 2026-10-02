"""Contract tests for the transient-search vector cache."""

import numpy as np

from chunkhound.services.vector_cache import VectorCache


def _vector(cache: VectorCache, text: str, **namespace) -> list[float]:
    cached = cache.get(text, **namespace)
    assert cached is not None
    return cached.tolist()


def test_vector_cache_hit_and_miss() -> None:
    cache = VectorCache(max_entries=2, ttl_seconds=60)

    assert cache.get("missing") is None
    cache.put("alpha", [1.0, 0.0])

    cached = cache.get("alpha")
    assert cached is not None
    assert cached.tolist() == [1.0, 0.0]
    # float32 keeps a cached vector at dim*4 bytes; a Python float list costs ~8x.
    assert cached.dtype == np.float32


def test_vector_cache_evicts_least_recently_used_entry() -> None:
    cache = VectorCache(max_entries=2, ttl_seconds=60)
    cache.put("alpha", [1.0])
    cache.put("beta", [2.0])
    assert _vector(cache, "alpha") == [1.0]

    cache.put("gamma", [3.0])

    assert _vector(cache, "alpha") == [1.0]
    assert cache.get("beta") is None
    assert _vector(cache, "gamma") == [3.0]


def test_vector_cache_expires_entries(monkeypatch) -> None:
    now = 100.0
    monkeypatch.setattr("chunkhound.services.vector_cache.time.monotonic", lambda: now)
    cache = VectorCache(max_entries=2, ttl_seconds=10)
    cache.put("alpha", [1.0])

    now = 111.0

    assert cache.get("alpha") is None


def test_vector_cache_max_entries_zero_disables_storage() -> None:
    cache = VectorCache(max_entries=0, ttl_seconds=60)

    cache.put("alpha", [1.0])

    assert cache.get("alpha") is None
    assert len(cache) == 0


def test_vector_cache_namespaces_provider_model_and_dims() -> None:
    cache = VectorCache(max_entries=10, ttl_seconds=60)
    cache.put("same", [1.0, 0.0], provider="a", model="m", dims=2)

    assert cache.get("same", provider="b", model="m", dims=2) is None
    assert cache.get("same", provider="a", model="other", dims=2) is None
    assert cache.get("same", provider="a", model="m", dims=3) is None
    assert _vector(cache, "same", provider="a", model="m", dims=2) == [1.0, 0.0]


def test_vector_cache_stores_hashes_and_vectors_not_text() -> None:
    cache = VectorCache(max_entries=2, ttl_seconds=60)
    secret_text = "text that must not be retained"
    cache.put(secret_text, [1.0, 2.0])

    assert secret_text not in repr(cache._entries)


def test_vector_cache_ttl_measures_time_since_last_access(monkeypatch) -> None:
    """A vector still being queried must not expire mid-session.

    TTL exists to reclaim entries nothing is asking for. Measuring it from
    insertion would force a long research session to re-embed hot chunks.
    """
    now = 100.0
    monkeypatch.setattr("chunkhound.services.vector_cache.time.monotonic", lambda: now)
    cache = VectorCache(max_entries=2, ttl_seconds=10)
    cache.put("alpha", [1.0])

    now = 105.0
    assert _vector(cache, "alpha") == [1.0]  # access refreshes the clock

    now = 112.0
    assert _vector(cache, "alpha") == [1.0]  # 12s after insert, 7s after access

    now = 130.0
    assert cache.get("alpha") is None  # 18s with no access


def test_vector_cache_returns_isolated_copies() -> None:
    cache = VectorCache(max_entries=2, ttl_seconds=60)
    source = [1.0, 2.0]
    cache.put("alpha", source)

    source[0] = 99.0
    cached = cache.get("alpha")
    assert cached is not None
    cached[1] = 99.0

    assert _vector(cache, "alpha") == [1.0, 2.0]
