"""Unit tests for the DuckDuckGo pagination loop (``search``) and form POST.

Covers:
- ``search`` pagination termination conditions (limit, empty page, missing
  Next form) and its forwarding of query params.
- ``_fetch`` POSTing url-encoded params to the DDG HTML endpoint with the
  required User-Agent header.
- ``search_multi`` deduping across queries, running every query without
  early-exit, and partial / total failure semantics.
"""

from __future__ import annotations

import asyncio
import urllib.error
import urllib.parse
import urllib.request
from io import BytesIO

import pytest
from loguru import logger

from chunkhound.utils import websearch_core as ws_mod

# ---------------------------------------------------------------------------
# HTML fixture builders
# ---------------------------------------------------------------------------


def _result_html(title: str, url: str, desc: str) -> str:
    return (
        f'<a class="result__a" href="{url}">{title}</a>'
        f'<a class="result__snippet">{desc}</a>'
    )


def _next_form_html(params: dict[str, str]) -> str:
    hidden = "".join(
        f'<input type="hidden" name="{n}" value="{v}">' for n, v in params.items()
    )
    return (
        "<form>"
        f"{hidden}"
        '<input type="submit" value="Next">'
        "</form>"
    )


def _page_html(
    results: list[tuple[str, str, str]],
    next_params: dict[str, str] | None = None,
) -> str:
    body = "".join(_result_html(*r) for r in results)
    if next_params is not None:
        body += _next_form_html(next_params)
    return f"<html><body>{body}</body></html>"


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def _install_fetch_sequence(monkeypatch, pages: list[str]) -> list[dict[str, str]]:
    calls: list[dict[str, str]] = []
    it = iter(pages)

    def fake_fetch(params: dict[str, str]) -> str:
        calls.append(dict(params))
        try:
            return next(it)
        except StopIteration as e:
            raise AssertionError("_fetch called more times than pages provided") from e

    monkeypatch.setattr(ws_mod, "_fetch", fake_fetch)
    return calls


def test_search_single_page_under_limit(monkeypatch) -> None:
    page = _page_html(
        [
            ("A", "https://a/", "da"),
            ("B", "https://b/", "db"),
            ("C", "https://c/", "dc"),
        ],
        next_params=None,
    )
    calls = _install_fetch_sequence(monkeypatch, [page])

    out = ws_mod.search("foo", limit=30)

    assert [url for _, url, _ in out] == ["https://a/", "https://b/", "https://c/"]
    assert len(calls) == 1


def test_search_forwards_query_on_first_fetch(monkeypatch) -> None:
    page = _page_html([("A", "https://a/", "d")])
    calls = _install_fetch_sequence(monkeypatch, [page])

    ws_mod.search("hello world", limit=30)

    assert calls[0] == {"q": "hello world", "b": ""}


def test_search_paginates_until_limit(monkeypatch) -> None:
    next_params = {"q": "q", "s": "20", "dc": "21"}
    page1 = _page_html(
        [("T" + str(i), f"https://p1-{i}/", "d") for i in range(20)],
        next_params=next_params,
    )
    page2 = _page_html([("T" + str(i), f"https://p2-{i}/", "d") for i in range(20)])
    calls = _install_fetch_sequence(monkeypatch, [page1, page2])

    out = ws_mod.search("q", limit=30)

    assert len(out) == 30
    assert len(calls) == 2
    # Second call must forward the hidden-input dict from the Next form.
    assert calls[1] == next_params


def test_search_halts_on_empty_second_page(monkeypatch) -> None:
    next_params = {"q": "q", "s": "5"}
    page1 = _page_html(
        [("A" + str(i), f"https://p1-{i}/", "d") for i in range(5)],
        next_params=next_params,
    )
    page2 = _page_html([], next_params=None)
    calls = _install_fetch_sequence(monkeypatch, [page1, page2])

    out = ws_mod.search("q", limit=30)

    assert len(out) == 5
    assert len(calls) == 2


def test_search_halts_when_no_next_form(monkeypatch) -> None:
    page1 = _page_html(
        [("A" + str(i), f"https://p1-{i}/", "d") for i in range(7)],
        next_params=None,
    )
    calls = _install_fetch_sequence(monkeypatch, [page1])

    out = ws_mod.search("q", limit=30)

    assert len(out) == 7
    assert len(calls) == 1


def test_search_progress_callback_invoked_per_page(monkeypatch) -> None:
    next_params = {"q": "q", "s": "3"}
    page1 = _page_html(
        [("A" + str(i), f"https://p1-{i}/", "d") for i in range(3)],
        next_params=next_params,
    )
    page2 = _page_html([("B0", "https://p2-0/", "d")])
    _install_fetch_sequence(monkeypatch, [page1, page2])

    seen: list[str] = []
    ws_mod.search("q", limit=30, progress_callback=seen.append)

    assert seen == ["Fetching page 1...", "Fetching page 2..."]


# ---------------------------------------------------------------------------
# _fetch
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = BytesIO(body)

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None

    def read(self) -> bytes:
        return self._body.read()


def test_fetch_posts_urlencoded_to_ddg_html_endpoint(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["data"] = req.data
        captured["headers"] = dict(req.header_items())
        captured["timeout"] = timeout
        return _FakeResponse(b"<html>ok</html>")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    body = ws_mod._fetch({"q": "cats & dogs", "b": ""})

    assert body == "<html>ok</html>"
    assert captured["url"] == "https://html.duckduckgo.com/html/"
    assert captured["data"] == urllib.parse.urlencode(
        {"q": "cats & dogs", "b": ""}
    ).encode()
    # Request.header_items() title-cases names.
    assert captured["headers"].get("User-agent") == "Mozilla/5.0"
    assert captured["timeout"] == 30


@pytest.mark.parametrize("bad_params", [{"q": ""}, {"q": "a b c"}])
def test_fetch_encodes_params_losslessly(monkeypatch, bad_params) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(req, timeout=None):
        captured["data"] = req.data
        return _FakeResponse(b"ok")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    ws_mod._fetch(bad_params)

    # Round-tripping through parse_qs must restore the input params.
    decoded = urllib.parse.parse_qs(
        captured["data"].decode(), keep_blank_values=True
    )
    for k, v in bad_params.items():
        assert decoded[k] == [v]


# ---------------------------------------------------------------------------
# search_multi
# ---------------------------------------------------------------------------


def _install_search_stub(
    monkeypatch,
    responses: dict[str, list[tuple[str, str, str]] | Exception],
) -> list[tuple[str, int]]:
    """Replace ws_mod.search with a stub returning canned per-query results.

    Captures (query, limit) per call. Exceptions in the values are raised.
    """
    calls: list[tuple[str, int]] = []

    def fake_search(query, limit=30, progress_callback=None):
        calls.append((query, limit))
        val = responses[query]
        if isinstance(val, Exception):
            raise val
        return list(val)

    monkeypatch.setattr(ws_mod, "search", fake_search)
    return calls


def test_search_multi_dedupes_across_queries_by_normalized_url(monkeypatch) -> None:
    responses = {
        "q1": [
            ("A", "https://Example.com/foo/", "d1"),
            ("B", "https://ex.com/b", "d2"),
        ],
        "q2": [
            ("A2", "https://example.com/foo", "d3"),
            ("C", "https://other.com/", "d4"),
        ],
    }
    _install_search_stub(monkeypatch, responses)

    out = asyncio.run(ws_mod.search_multi(["q1", "q2"], limit=10))

    urls = [url for _, url, _ in out]
    # Q1's "https://Example.com/foo/" wins over Q2's near-duplicate variant.
    assert urls == [
        "https://Example.com/foo/",
        "https://ex.com/b",
        "https://other.com/",
    ]


def test_search_multi_passes_full_limit_to_each_query(monkeypatch) -> None:
    # Each variant is called with the caller's full ``limit`` (not
    # ``limit / n``) so cross-variant URL overlap cannot silently drop the
    # returned count below ``limit``.
    responses: dict[str, list[tuple[str, str, str]]] = {
        "q1": [], "q2": [], "q3": [],
    }
    calls = _install_search_stub(monkeypatch, responses)

    asyncio.run(ws_mod.search_multi(["q1", "q2", "q3"], limit=10))

    assert [c[1] for c in calls] == [10, 10, 10]


def test_search_multi_runs_all_queries_even_when_first_saturates_limit(
    monkeypatch,
) -> None:
    # Even when q1 alone already returns ``limit`` distinct URLs, q2 and q3
    # must still be dispatched so their diversity contributes to the pool.
    # Under rank-major interleaving, q2's and q3's rank-0 hits land at
    # positions 1 and 2 — pushing q1's tail out of the truncated result.
    responses: dict[str, list[tuple[str, str, str]]] = {
        "q1": [(f"T{i}", f"https://a.example/{i}", "d") for i in range(10)],
        "q2": [("B", "https://b.example/", "d")],
        "q3": [("C", "https://c.example/", "d")],
    }
    calls = _install_search_stub(monkeypatch, responses)

    out = asyncio.run(ws_mod.search_multi(["q1", "q2", "q3"], limit=10))

    assert [c[0] for c in calls] == ["q1", "q2", "q3"]
    assert len(out) == 10
    assert [url for _, url, _ in out] == [
        "https://a.example/0",
        "https://b.example/",
        "https://c.example/",
        "https://a.example/1",
        "https://a.example/2",
        "https://a.example/3",
        "https://a.example/4",
        "https://a.example/5",
        "https://a.example/6",
        "https://a.example/7",
    ]


def test_search_multi_interleaves_rank_major(monkeypatch) -> None:
    # Rank-major ordering contract: every query's rank-0 hit appears
    # before any query's rank-1 hit. No single query gets to dominate the
    # head of the list even when its lower-ranked results are available.
    responses: dict[str, list[tuple[str, str, str]]] = {
        "q1": [(f"T{i}", f"https://q1.example/{i}", "d") for i in range(3)],
        "q2": [(f"U{i}", f"https://q2.example/{i}", "d") for i in range(3)],
        "q3": [(f"V{i}", f"https://q3.example/{i}", "d") for i in range(3)],
    }
    _install_search_stub(monkeypatch, responses)

    out = asyncio.run(ws_mod.search_multi(["q1", "q2", "q3"], limit=9))

    assert [url for _, url, _ in out] == [
        "https://q1.example/0",
        "https://q2.example/0",
        "https://q3.example/0",
        "https://q1.example/1",
        "https://q2.example/1",
        "https://q3.example/1",
        "https://q1.example/2",
        "https://q2.example/2",
        "https://q3.example/2",
    ]


def test_search_multi_keeps_going_until_limit_distinct_urls(monkeypatch) -> None:
    # First two queries return the SAME URLs — the third is required to
    # reach ``limit`` distinct results. The old ceil-based cap made this
    # unreachable (each query fetched ceil(3/3)=1 result and dedupe collapsed
    # them to 1); the new contract must fetch the third query.
    overlap = [("A", "https://same.example/", "d")]
    responses: dict[str, list[tuple[str, str, str]]] = {
        "q1": list(overlap),
        "q2": list(overlap),
        "q3": [
            ("B", "https://b.example/", "d"),
            ("C", "https://c.example/", "d"),
        ],
    }
    calls = _install_search_stub(monkeypatch, responses)

    out = asyncio.run(ws_mod.search_multi(["q1", "q2", "q3"], limit=3))

    assert [c[0] for c in calls] == ["q1", "q2", "q3"]
    assert {url for _, url, _ in out} == {
        "https://same.example/",
        "https://b.example/",
        "https://c.example/",
    }


def test_search_multi_truncates_to_limit(monkeypatch) -> None:
    responses = {
        "q1": [(f"T{i}", f"https://a.example/{i}", "d") for i in range(5)],
        "q2": [(f"U{i}", f"https://b.example/{i}", "d") for i in range(5)],
    }
    _install_search_stub(monkeypatch, responses)

    out = asyncio.run(ws_mod.search_multi(["q1", "q2"], limit=7))

    assert len(out) == 7


def test_search_multi_continues_on_partial_url_error(monkeypatch) -> None:
    responses: dict[str, list[tuple[str, str, str]] | Exception] = {
        "q1": urllib.error.URLError("boom"),
        "q2": [("A", "https://ok.example/", "d")],
    }
    _install_search_stub(monkeypatch, responses)

    out = asyncio.run(ws_mod.search_multi(["q1", "q2"], limit=10))

    assert [url for _, url, _ in out] == ["https://ok.example/"]


def test_search_multi_logs_when_no_failure_callback(monkeypatch) -> None:
    # No failure_callback: partial failures must still surface in the logs
    # so operators can diagnose thin result sets.
    responses: dict[str, list[tuple[str, str, str]] | Exception] = {
        "q1": urllib.error.URLError("boom"),
        "q2": [("A", "https://ok.example/", "d")],
    }
    _install_search_stub(monkeypatch, responses)

    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        asyncio.run(ws_mod.search_multi(["q1", "q2"], limit=10))
    finally:
        logger.remove(sink_id)

    assert any("q1" in m and "boom" in m for m in messages)


def test_search_multi_does_not_log_when_callback_provided(monkeypatch) -> None:
    # When a callback is provided, the logger fallback should NOT fire —
    # otherwise callers that render the callback would get the warning twice.
    responses: dict[str, list[tuple[str, str, str]] | Exception] = {
        "q1": urllib.error.URLError("boom"),
        "q2": [("A", "https://ok.example/", "d")],
    }
    _install_search_stub(monkeypatch, responses)

    log_messages: list[str] = []
    sink_id = logger.add(lambda m: log_messages.append(str(m)), level="WARNING")
    seen: list[tuple[str, urllib.error.URLError]] = []

    def _cb(q: str, e: urllib.error.URLError) -> None:
        seen.append((q, e))

    try:
        asyncio.run(
            ws_mod.search_multi(["q1", "q2"], limit=10, failure_callback=_cb)
        )
    finally:
        logger.remove(sink_id)

    assert [q for q, _ in seen] == ["q1"]
    assert not any("DDG query failed" in m for m in log_messages)


def test_search_multi_reraises_first_error_when_all_fail(monkeypatch) -> None:
    first = urllib.error.URLError("first")
    responses: dict[str, list[tuple[str, str, str]] | Exception] = {
        "q1": first,
        "q2": urllib.error.URLError("second"),
    }
    _install_search_stub(monkeypatch, responses)

    with pytest.raises(urllib.error.URLError) as exc:
        asyncio.run(ws_mod.search_multi(["q1", "q2"], limit=10))
    assert exc.value is first


def test_search_multi_empty_queries_returns_empty(monkeypatch) -> None:
    _install_search_stub(monkeypatch, {})

    out = asyncio.run(ws_mod.search_multi([], limit=10))

    assert out == []


# ---------------------------------------------------------------------------
# _normalize_url
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("https://Example.com/foo/", "https://example.com/foo"),
        ("https://X.com/", "https://x.com"),
        ("https://x.com/a#frag", "https://x.com/a"),
    ],
)
def test_normalize_url_equates_variants(a: str, b: str) -> None:
    assert ws_mod._normalize_url(a) == ws_mod._normalize_url(b)
