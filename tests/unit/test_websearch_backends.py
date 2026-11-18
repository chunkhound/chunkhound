"""Multi-engine websearch orchestration contracts.

Covers engine ordering (priority / random_healthy / half-open probe),
strike-based cooldown escalation, failover on refusal / empty / drift /
transient, per-engine budget partitioning, partial-results-on-mid-search-
refusal, and the distinct blocked-vs-empty error surface.
"""

from __future__ import annotations

import asyncio
import time
import urllib.error

import pytest

from chunkhound.core.config.websearch_config import WebSearchConfig
from chunkhound.utils import websearch_backends as b
from chunkhound.utils.websearch_verdict import EngineResponse, EngineRun, Verdict


def _served(*urls: str) -> EngineResponse:
    rows = tuple((f"T{i}", url, "d") for i, url in enumerate(urls))
    return EngineResponse(rows, Verdict.SERVED, 200)


def _empty() -> EngineResponse:
    return EngineResponse((), Verdict.EMPTY, 200, "no results sentinel")


def _refused(status: int = 202) -> EngineResponse:
    return EngineResponse((), Verdict.REFUSED, status, f"refused status {status}")


class _FakeBackend:
    """Sync engine stub recording calls and delegating to a serp callable."""

    def __init__(self, name: str, serp) -> None:
        self.name = name
        self.serp = serp
        self.calls: list[tuple[str, int]] = []

    def search(self, query, limit=30, progress_callback=None, *, cancel_event=None):
        self.calls.append((query, limit))
        if progress_callback is not None:
            progress_callback("ok")
        val = self.serp(query)
        if isinstance(val, Exception):
            raise val
        return val


def _unused_backend(name: str) -> _FakeBackend:
    return _FakeBackend(name, lambda q: (_ for _ in ()).throw(AssertionError("unused")))


def _install_backends(monkeypatch, ddg=None, bing=None) -> None:
    if ddg is not None:
        monkeypatch.setitem(b.ALL_BACKENDS, "duckduckgo", ddg)
    if bing is not None:
        monkeypatch.setitem(b.ALL_BACKENDS, "bing", bing)


async def _run(
    queries, limit=10, *, config=None, progress=None, registry=None, timeout_s=None
):
    return await b.run_engine_search(
        queries,
        limit,
        progress_callback=progress.append if progress is not None else None,
        websearch_config=config or WebSearchConfig(),
        registry=registry or b.EngineRegistry(),
        timeout_s=timeout_s,
    )


def test_priority_uses_first_healthy_engine(monkeypatch) -> None:
    ddg = _FakeBackend("duckduckgo", lambda q: _served("https://a/"))
    bing = _unused_backend("bing")
    _install_backends(monkeypatch, ddg=ddg, bing=bing)

    out = asyncio.run(_run(["q"], config=WebSearchConfig()))

    assert [u for _, u, _ in out] == ["https://a/"]
    assert ddg.calls == [("q", 10)]
    assert bing.calls == []


def test_failover_when_first_engine_refused(monkeypatch) -> None:
    ddg = _FakeBackend("duckduckgo", lambda q: _refused(202))
    bing = _FakeBackend("bing", lambda q: _served("https://b/"))
    _install_backends(monkeypatch, ddg=ddg, bing=bing)
    progress: list[str] = []

    out = asyncio.run(_run(["q"], progress=progress))

    assert [u for _, u, _ in out] == ["https://b/"]
    assert bing.calls == [("q", 10)]
    assert any(
        "Blocked by duckduckgo (refused status 202)" in m and "trying bing" in m
        for m in progress
    )


def test_all_engines_refused_raises_with_names_and_evidence(monkeypatch) -> None:
    ddg = _FakeBackend("duckduckgo", lambda q: _refused(202))
    bing = _FakeBackend("bing", lambda q: _refused(429))
    _install_backends(monkeypatch, ddg=ddg, bing=bing)

    with pytest.raises(b.AllEnginesBlockedError) as exc:
        asyncio.run(_run(["q"]))
    assert exc.value.engines == ["duckduckgo", "bing"]
    message = str(exc.value)
    assert message.startswith("Blocked by duckduckgo, bing; all engines unavailable")
    assert "(duckduckgo: refused status 202; bing: refused status 429)" in message


def test_all_engines_empty_returns_no_results(monkeypatch) -> None:
    ddg = _FakeBackend("duckduckgo", lambda q: _empty())
    bing = _FakeBackend("bing", lambda q: _empty())
    _install_backends(monkeypatch, ddg=ddg, bing=bing)

    out = asyncio.run(_run(["q"]))

    assert out == []


def test_empty_from_first_engine_falls_through_to_second(monkeypatch) -> None:
    ddg = _FakeBackend("duckduckgo", lambda q: _empty())
    bing = _FakeBackend("bing", lambda q: _served("https://b/"))
    _install_backends(monkeypatch, ddg=ddg, bing=bing)
    progress: list[str] = []

    out = asyncio.run(_run(["q"], progress=progress))

    assert [u for _, u, _ in out] == ["https://b/"]
    assert any("duckduckgo returned no results; trying bing" in m for m in progress)


def test_mid_search_refusal_keeps_partial_rows_and_records_strike(
    monkeypatch,
) -> None:
    def serp(q):
        return _refused(202) if q == "blocked" else _served("https://a/")

    ddg = _FakeBackend("duckduckgo", serp)
    bing = _unused_backend("bing")
    _install_backends(monkeypatch, ddg=ddg, bing=bing)
    registry = b.EngineRegistry()

    out = asyncio.run(_run(["ok", "blocked"], registry=registry))

    assert [u for _, u, _ in out] == ["https://a/"]
    assert bing.calls == []
    assert registry.strikes("duckduckgo") == 1


def test_urlerror_all_fail_reraises(monkeypatch) -> None:
    first = urllib.error.URLError("boom")
    ddg = _FakeBackend("duckduckgo", lambda q: first)
    bing = _unused_backend("bing")
    _install_backends(monkeypatch, ddg=ddg, bing=bing)

    with pytest.raises(urllib.error.URLError) as exc:
        asyncio.run(_run(["q1", "q2"]))
    assert exc.value is first
    assert bing.calls == []


def test_search_multi_defaults_to_duckduckgo_first(monkeypatch) -> None:
    from chunkhound.utils.websearch_core import search_multi

    ddg = _FakeBackend("duckduckgo", lambda q: _served("https://a/"))
    bing = _unused_backend("bing")
    _install_backends(monkeypatch, ddg=ddg, bing=bing)

    out = asyncio.run(search_multi(["q"], limit=3))

    assert [u for _, u, _ in out] == ["https://a/"]
    assert ddg.calls == [("q", 3)]


def test_search_multi_config_pins_bing_only(monkeypatch) -> None:
    from chunkhound.utils.websearch_core import search_multi

    ddg = _unused_backend("duckduckgo")
    bing = _FakeBackend("bing", lambda q: _served("https://b/"))
    _install_backends(monkeypatch, ddg=ddg, bing=bing)
    cfg = WebSearchConfig(backends=["bing"])

    out = asyncio.run(search_multi(["q"], limit=3, websearch_config=cfg))

    assert [u for _, u, _ in out] == ["https://b/"]
    assert bing.calls == [("q", 3)]


def test_backend_name_lists_never_diverge() -> None:
    from chunkhound.core.config.websearch_config import BACKEND_NAMES

    # Config validates against BACKEND_NAMES, the runtime dispatches via
    # ALL_BACKENDS; a name added to only one list fails at runtime with a
    # bare KeyError, so pin the two sources together.
    assert set(BACKEND_NAMES) == set(b.ALL_BACKENDS)
    for key, backend in b.ALL_BACKENDS.items():
        assert backend.name == key


# ---------------------------------------------------------------------------
# Selection and health
# ---------------------------------------------------------------------------


def test_resolve_priority_skips_cooling_engine() -> None:
    registry = b.EngineRegistry(cooldown_seconds=3600)
    registry.mark_blocked("duckduckgo")
    order = b.resolve_engine_order(WebSearchConfig(), registry)
    assert order == ["bing"]


def test_all_cooling_resolves_to_single_soonest_probe() -> None:
    registry = b.EngineRegistry(cooldown_seconds=3600)
    registry.mark_blocked("duckduckgo", cooldown_seconds=100)
    registry.mark_blocked("bing", cooldown_seconds=10)
    order = b.resolve_engine_order(WebSearchConfig(), registry)
    assert order == ["bing"]
    assert registry.recover_at("bing") < registry.recover_at("duckduckgo")


def test_resolve_random_healthy_shuffles_only_healthy() -> None:
    registry = b.EngineRegistry(cooldown_seconds=3600)
    registry.mark_blocked("duckduckgo")
    order = b.resolve_engine_order(
        WebSearchConfig(selection="random_healthy"), registry
    )
    # Only bing is healthy; shuffle of a single element is deterministic.
    assert order == ["bing"]


def test_single_empty_run_does_not_cool_engine() -> None:
    registry = b.EngineRegistry(cooldown_seconds=3600)
    registry.record("bing", EngineRun([], Verdict.EMPTY, "sentinel"))

    assert registry.is_healthy("bing")
    assert registry.strikes("bing") == 0


def test_two_consecutive_refusals_cool_engine() -> None:
    registry = b.EngineRegistry(cooldown_seconds=3600)
    registry.record(
        "bing", EngineRun([], Verdict.REFUSED, "refused status 403", strike=True)
    )
    assert registry.is_healthy("bing")  # one strike is not enough

    registry.record(
        "bing", EngineRun([], Verdict.REFUSED, "refused status 403", strike=True)
    )

    assert not registry.is_healthy("bing")


def test_served_run_clears_strikes() -> None:
    registry = b.EngineRegistry(cooldown_seconds=3600)
    registry.record(
        "bing", EngineRun([], Verdict.REFUSED, "refused status 403", strike=True)
    )
    registry.record("bing", EngineRun([("T", "https://a/", "d")], Verdict.SERVED))
    registry.record(
        "bing", EngineRun([], Verdict.REFUSED, "refused status 403", strike=True)
    )

    assert registry.is_healthy("bing")
    assert registry.strikes("bing") == 1


def test_registry_cooldown_blocks_then_expires() -> None:
    registry = b.EngineRegistry(cooldown_seconds=3600)
    assert registry.is_healthy("bing")
    registry.mark_blocked("bing")
    assert not registry.is_healthy("bing")
    registry.mark_blocked("bing", cooldown_seconds=0)
    assert registry.is_healthy("bing")
    # Force expiry.
    registry._blocked_until["bing"] = time.monotonic() - 1
    assert registry.is_healthy("bing")


def test_multi_query_progress_prefixes_match_iteration(monkeypatch) -> None:
    ddg = _FakeBackend("duckduckgo", lambda q: _served("https://a/"))
    _install_backends(monkeypatch, ddg=ddg)
    progress: list[str] = []

    asyncio.run(_run(["one", "two", "three"], progress=progress))

    assert progress == ["[1/3] ok", "[2/3] ok", "[3/3] ok"]


def test_per_call_cooldown_does_not_mutate_registry(monkeypatch) -> None:
    ddg = _FakeBackend("duckduckgo", lambda q: _refused(202))
    _install_backends(monkeypatch, ddg=ddg)
    registry = b.EngineRegistry(cooldown_seconds=3600)
    cfg = WebSearchConfig(backends=["duckduckgo"], cooldown_seconds=60)

    # One refusal is a strike, not a block: ambiguous evidence must not
    # remove an engine from rotation.
    with pytest.raises(b.AllEnginesBlockedError):
        asyncio.run(_run(["q"], config=cfg, registry=registry))
    assert registry.strikes("duckduckgo") == 1
    assert "duckduckgo" not in registry._blocked_until

    # A second consecutive refusal blocks for the *config* cooldown (±20%
    # jitter), not the registry's own 3600s default.
    with pytest.raises(b.AllEnginesBlockedError):
        asyncio.run(_run(["q"], config=cfg, registry=registry))

    assert registry.cooldown_seconds == 3600
    elapsed = registry._blocked_until["duckduckgo"] - time.monotonic()
    assert 60 * 0.8 <= elapsed <= 60 * 1.2


class _HangingBackend:
    """Engine that blocks until the orchestrator's cancel_event fires."""

    name = "duckduckgo"

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def search(self, query, limit=30, progress_callback=None, *, cancel_event=None):
        self.calls.append((query, limit))
        if cancel_event is not None:
            # Released by _run_engine when the per-engine budget expires.
            cancel_event.wait(timeout=5)
        return _served("https://never/")


def test_hanging_primary_times_out_and_fallback_serves(monkeypatch) -> None:
    ddg = _HangingBackend()
    bing = _FakeBackend("bing", lambda q: _served("https://b/"))
    _install_backends(monkeypatch, ddg=ddg, bing=bing)
    monkeypatch.setattr(b, "_MIN_ENGINE_BUDGET_S", 0.05)
    progress: list[str] = []

    out = asyncio.run(_run(["q"], progress=progress, timeout_s=0.5))

    assert [u for _, u, _ in out] == ["https://b/"]
    assert ddg.calls == [("q", 10)]
    assert any("transient" in m for m in progress)
