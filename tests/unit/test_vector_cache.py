"""Contract tests for the transient-search vector cache."""

from chunkhound.services.vector_cache import VectorCache


def test_vector_cache_hit_and_miss() -> None:
    cache = VectorCache(max_entries=2, ttl_seconds=60)

    assert cache.get("missing") is None
    cache.put("alpha", [1.0, 0.0])

    assert cache.get("alpha") == [1.0, 0.0]


def test_vector_cache_evicts_least_recently_used_entry() -> None:
    cache = VectorCache(max_entries=2, ttl_seconds=60)
    cache.put("alpha", [1.0])
    cache.put("beta", [2.0])
    assert cache.get("alpha") == [1.0]

    cache.put("gamma", [3.0])

    assert cache.get("alpha") == [1.0]
    assert cache.get("beta") is None
    assert cache.get("gamma") == [3.0]


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
    assert cache.get("same", provider="a", model="m", dims=2) == [1.0, 0.0]


def test_vector_cache_stores_hashes_and_vectors_not_text() -> None:
    cache = VectorCache(max_entries=2, ttl_seconds=60)
    secret_text = "text that must not be retained"
    cache.put(secret_text, [1.0, 2.0])

    assert secret_text not in repr(cache._entries)
