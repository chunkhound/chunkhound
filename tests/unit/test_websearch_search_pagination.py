"""Unit tests for the DuckDuckGo pagination loop (``_search``) and form POST.

Covers:
- ``_search`` pagination termination conditions (limit, empty page, missing
  Next form) and its forwarding of query params.
- ``_fetch`` POSTing url-encoded params to the DDG HTML endpoint with the
  required User-Agent header.
"""

from __future__ import annotations

import urllib.parse
import urllib.request
from io import BytesIO

import pytest

from chunkhound.api.cli.commands import websearch as ws_mod

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
# _search
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

    out = ws_mod._search("foo", limit=30)

    assert [url for _, url, _ in out] == ["https://a/", "https://b/", "https://c/"]
    assert len(calls) == 1


def test_search_forwards_query_on_first_fetch(monkeypatch) -> None:
    page = _page_html([("A", "https://a/", "d")])
    calls = _install_fetch_sequence(monkeypatch, [page])

    ws_mod._search("hello world", limit=30)

    assert calls[0] == {"q": "hello world", "b": ""}


def test_search_paginates_until_limit(monkeypatch) -> None:
    next_params = {"q": "q", "s": "20", "dc": "21"}
    page1 = _page_html(
        [("T" + str(i), f"https://p1-{i}/", "d") for i in range(20)],
        next_params=next_params,
    )
    page2 = _page_html([("T" + str(i), f"https://p2-{i}/", "d") for i in range(20)])
    calls = _install_fetch_sequence(monkeypatch, [page1, page2])

    out = ws_mod._search("q", limit=30)

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

    out = ws_mod._search("q", limit=30)

    assert len(out) == 5
    assert len(calls) == 2


def test_search_halts_when_no_next_form(monkeypatch) -> None:
    page1 = _page_html(
        [("A" + str(i), f"https://p1-{i}/", "d") for i in range(7)],
        next_params=None,
    )
    calls = _install_fetch_sequence(monkeypatch, [page1])

    out = ws_mod._search("q", limit=30)

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
    ws_mod._search("q", limit=30, progress_callback=seen.append)

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
