"""Bing SERP parsing and BingBackend transport contracts.

Fixtures mirror the live Bing markup (verified 2026-11-16): results are
``<li class="b_algo">`` blocks with an ``<h2><a href=".../ck/a?...&u=a1a<base64url>">``
title link and a ``<p class="b_lineclampN">`` snippet.
"""

from __future__ import annotations

import base64
import urllib.parse

from chunkhound.utils import websearch_backends as b
from chunkhound.utils.websearch_verdict import Verdict


def _b64url(s: str) -> str:
    return base64.urlsafe_b64encode(s.encode()).decode().rstrip("=")


def _redirect_href(destination: str) -> str:
    # Live Bing format: the u param is a short marker plus the base64url
    # destination; the marker's trailing "a" overlaps the base64's leading
    # "a", so the literal prefix reads "a1a".
    return (
        "https://www.bing.com/ck/a?!&amp;&amp;p=abcJmltdHM9..."
        f"&amp;u=a1{_b64url(destination)}"
        "&amp;ntb=1"
    )


def _result_li(
    title: str, destination: str, snippet: str, *, raw_href: str | None = None
) -> str:
    href = raw_href or _redirect_href(destination)
    return (
        f'<li class="b_algo" data-id iid="SERP.5334">'
        f'<div class="b_caption">'
        f'<h2><a target="_blank" href="{href}" h="ID=SERP,5127.2">{title}</a></h2>'
        f'<p class="b_lineclamp2" data-rslinkclamp-iid="">{snippet}</p>'
        f"</div></li>"
    )


def _serp(*lis: str) -> str:
    return f'<html><body><ol id="b_results">{"".join(lis)}</ol></body></html>'


def _urls(rows) -> list[str]:
    return [u for _, u, _ in rows]


# ---------------------------------------------------------------------------
# _decode_bing_url / parser
# ---------------------------------------------------------------------------


def test_decode_bing_url_recovers_destination() -> None:
    decoded = b._decode_bing_url(_redirect_href("https://www.python.org/"))
    assert decoded == "https://www.python.org/"


def test_decode_bing_url_non_redirect_passthrough() -> None:
    assert (
        b._decode_bing_url("https://example.com/plain") == "https://example.com/plain"
    )


def test_parse_single_result() -> None:
    html = _serp(_result_li("Python", "https://www.python.org/", "Official site"))
    rows = b._parse_bing_results(html)
    assert rows == [("Python", "https://www.python.org/", "Official site")]


def test_parse_multiple_results_preserve_order() -> None:
    html = _serp(
        _result_li("A", "https://a.example/", "da"),
        _result_li("B", "https://b.example/", "db"),
        _result_li("C", "https://c.example/", "dc"),
    )
    rows = b._parse_bing_results(html)
    assert _urls(rows) == [
        "https://a.example/",
        "https://b.example/",
        "https://c.example/",
    ]


def test_parse_unescapes_html_entities() -> None:
    html = _serp(_result_li("Rock &amp; Roll", "https://a.example/", "1 &lt; 2"))
    [(title, _, snippet)] = b._parse_bing_results(html)
    assert title == "Rock & Roll"
    assert snippet == "1 < 2"


def test_parse_falls_back_to_raw_href_when_u_missing() -> None:
    raw = "https://www.bing.com/ck/a?!&amp;p=abcJmltdHM9...&amp;ntb=1"
    html = _serp(_result_li("T", "ignored", "d", raw_href=raw))
    rows = b._parse_bing_results(html)
    assert rows[0][1] == urllib.parse.unquote(raw.replace("&amp;", "&"))


def test_parse_ignores_non_b_algo_content() -> None:
    html = _serp(
        '<li class="b_ad"><h2><a href="https://ad.example/">Ad</a></h2></li>',
        _result_li("Real", "https://real.example/", "d"),
    )
    rows = b._parse_bing_results(html)
    assert _urls(rows) == ["https://real.example/"]


def test_parse_empty_page_returns_none() -> None:
    assert b._parse_bing_results("<html><body></body></html>") == []


def test_parse_ignores_attribution_anchor_before_h2() -> None:
    # Bing emits an attribution anchor (class tilk) inside the li before the
    # real h2 result link; only the h2 anchor is the result.
    li = (
        '<li class="b_algo">'
        '<div class="b_tpcn"><a class="tilk" aria-label="python.org" href="'
        + _redirect_href("https://www.python.org/")
        + '">…</a></div>'
        '<h2><a target="_blank" href="'
        + _redirect_href("https://www.python.org/")
        + '">Welcome to Python.org</a></h2>'
        '<div class="b_caption"><p class="b_lineclamp2">snippet</p></div>'
        "</li>"
    )
    rows = b._parse_bing_results(_serp(li))
    assert rows == [("Welcome to Python.org", "https://www.python.org/", "snippet")]


# ---------------------------------------------------------------------------
# BingBackend.search
# ---------------------------------------------------------------------------


def _install_http(monkeypatch, responses):
    calls: list[dict[str, object]] = []

    def fake_http(method, url, *, params=None, data=None, headers=None, timeout=30.0):
        calls.append(
            {"method": method, "url": url, "params": params, "headers": headers}
        )
        status, body = responses.pop(0)
        return status, body

    monkeypatch.setattr(b, "_http_request", fake_http)
    return calls, fake_http


def test_bing_search_paginates_until_limit(monkeypatch) -> None:
    page1 = _serp(*[_result_li(f"T{i}", f"https://p1-{i}/", "d") for i in range(10)])
    page2 = _serp(*[_result_li(f"T{i}", f"https://p2-{i}/", "d") for i in range(10)])
    calls = _install_http(monkeypatch, [(200, page1.encode()), (200, page2.encode())])[
        0
    ]

    out = b.BingBackend().search("q", limit=15)

    assert out.verdict is Verdict.SERVED
    assert len(out.rows) == 15
    first_params = calls[0]["params"]
    assert first_params["q"] == "q"
    assert first_params["count"] == "10"
    assert first_params["first"] == "1"
    assert calls[1]["params"]["first"] == "11"


def test_bing_search_stops_when_page_short(monkeypatch) -> None:
    short = _serp(*[_result_li(f"T{i}", f"https://p1-{i}/", "d") for i in range(3)])
    calls = _install_http(monkeypatch, [(200, short.encode())])[0]

    out = b.BingBackend().search("q", limit=30)

    assert len(out.rows) == 3
    assert len(calls) == 1


def test_bing_pagination_capped_at_five_pages(monkeypatch) -> None:
    full = _serp(*[_result_li(f"T{i}", f"https://p-{i}/", "d") for i in range(10)])
    calls = _install_http(monkeypatch, [(200, full.encode()) for _ in range(6)])[0]

    out = b.BingBackend().search("q", limit=100)

    assert len(out.rows) == 50
    assert len(calls) == 5


def test_bing_empty_second_page_ends_pagination_as_served(monkeypatch) -> None:
    page1 = _serp(*[_result_li(f"T{i}", f"https://p1-{i}/", "d") for i in range(10)])
    calls = _install_http(
        monkeypatch, [(200, page1.encode()), (200, _serp().encode())]
    )[0]

    out = b.BingBackend().search("q", limit=30)

    assert out.verdict is Verdict.SERVED
    assert len(out.rows) == 10
    assert len(calls) == 2


def test_bing_refused_status_reports_refused_verdict(monkeypatch) -> None:
    _install_http(monkeypatch, [(202, b"<html>challenge</html>")])

    out = b.BingBackend().search("q", limit=10)

    assert out.verdict is Verdict.REFUSED
    assert out.status == 202
    assert "202" in out.evidence


def test_bing_empty_page_with_sentinel_is_not_refused(monkeypatch) -> None:
    # Real poison strings in the body must not flip an empty page to refused.
    page = (
        b"<html><head><script>challenges.cloudflare.com,akchal.bing.com"
        b'</script></head><body><ol id="b_results">'
        b'<li class="b_no">There are no results for q.</li>'
        b"</ol></body></html>"
    )
    _install_http(monkeypatch, [(200, page)])

    out = b.BingBackend().search("q", limit=10)

    assert out.verdict is Verdict.EMPTY


def test_bing_accept_language_header_sent(monkeypatch) -> None:
    calls = _install_http(monkeypatch, [(200, _serp().encode())])[0]
    b.BingBackend().search("q", limit=10)
    assert calls[0]["headers"]["Accept-Language"].startswith("en-US")
