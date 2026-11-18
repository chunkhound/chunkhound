"""Verdict taxonomy contracts, using markup mirrored from live pages.

The poison line (``challenges.cloudflare.com``, ``akchal.bing.com``) is real
Bing JS config that appears on served AND empty pages alike; it must never
influence a verdict. Fixtures reproduce it inside a script tag plus a real
``<li class="b_algo">`` block, so the served page and the sentinel-decided
empty page differ only in the ways the live pages did.
"""

from __future__ import annotations

import base64

import pytest

from chunkhound.utils import websearch_backends as b
from chunkhound.utils.websearch_verdict import Verdict, verdict_from_status

_POISON = (
    'var cfg={"challenges.cloudflare.com":"x","akchal.bing.com":"y",'
    '"PoWChallengeSolver":true};'
)


def _b64url(s: str) -> str:
    return base64.urlsafe_b64encode(s.encode()).decode().rstrip("=")


def _b_algo(title: str, destination: str) -> str:
    href = "https://www.bing.com/ck/a?!&amp;u=a1" + _b64url(destination) + "&amp;ntb=1"
    return (
        f'<li class="b_algo"><h2><a href="{href}">{title}</a></h2>'
        f'<p class="b_lineclamp2">snippet</p></li>'
    )


def _bing_page(*body: str) -> bytes:
    return (
        "<html><head><script>" + _POISON + "</script></head>"
        '<body><ol id="b_results">' + "".join(body) + "</ol></body></html>"
    ).encode()


def _install_http(monkeypatch, status: int, body: bytes) -> None:
    monkeypatch.setattr(b, "_http_request", lambda *a, **k: (status, body))


def test_served_bing_page_with_poison_script_is_served(monkeypatch) -> None:
    page = _bing_page(_b_algo("Python", "https://www.python.org/"))
    _install_http(monkeypatch, 200, page)

    out = b.BingBackend().search("q", limit=5)

    assert out.verdict is Verdict.SERVED
    assert len(out.rows) == 1
    assert out.rows[0][1] == "https://www.python.org/"


def test_empty_bing_page_with_sentinel_is_empty_never_refused(
    monkeypatch,
) -> None:
    empty = _bing_page('<li class="b_no"><p>There are no results for q.</p></li>')
    _install_http(monkeypatch, 200, empty)

    out = b.BingBackend().search("q", limit=5)

    assert out.verdict is Verdict.EMPTY
    assert out.rows == ()


def test_container_absent_page_is_drift(monkeypatch) -> None:
    _install_http(monkeypatch, 200, b"<html><body><p>weird</p></body></html>")

    out = b.BingBackend().search("q", limit=5)

    assert out.verdict is Verdict.DRIFT


def test_container_present_without_sentinel_is_drift(monkeypatch) -> None:
    _install_http(monkeypatch, 200, _bing_page())

    out = b.BingBackend().search("q", limit=5)

    assert out.verdict is Verdict.DRIFT


def test_non_200_status_maps_through_verdict_from_status(monkeypatch) -> None:
    _install_http(monkeypatch, 429, _bing_page())

    out = b.BingBackend().search("q", limit=5)

    assert out.verdict is Verdict.REFUSED
    assert "429" in out.evidence


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (202, Verdict.REFUSED),
        (403, Verdict.REFUSED),
        (429, Verdict.REFUSED),
        (408, Verdict.TRANSIENT),
        (301, Verdict.TRANSIENT),
        (302, Verdict.TRANSIENT),
        (500, Verdict.TRANSIENT),
        (503, Verdict.TRANSIENT),
        (400, Verdict.DRIFT),
        (401, Verdict.DRIFT),
        (404, Verdict.DRIFT),
        (410, Verdict.DRIFT),
        (200, Verdict.SERVED),
    ],
)
def test_status_mapping_table(status: int, expected: Verdict) -> None:
    assert verdict_from_status(status) is expected
