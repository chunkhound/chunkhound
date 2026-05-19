"""Unit tests for the DuckDuckGo HTML parsers used by websearch.

Covers ``_ResultParser`` (result-row extraction) and ``_NextFormParser``
(pagination form discovery). These are pure stateful HTML parsers; tests
feed inline HTML snippets and assert on the parsed output.
"""

from __future__ import annotations

from chunkhound.api.cli.commands.websearch import _NextFormParser, _ResultParser

# ---------------------------------------------------------------------------
# _ResultParser
# ---------------------------------------------------------------------------


def _parse_results(html_text: str) -> list[tuple[str, str, str]]:
    p = _ResultParser()
    p.feed(html_text)
    return p.results


def test_result_parser_single_result_tuple() -> None:
    html = """
    <div>
      <a class="result__a" href="https://example.com/">Example Site</a>
      <a class="result__snippet">A short description.</a>
    </div>
    """
    assert _parse_results(html) == [
        ("Example Site", "https://example.com/", "A short description.")
    ]


def test_result_parser_multiple_results_preserve_order() -> None:
    html = """
    <a class="result__a" href="https://a/">Alpha</a>
    <a class="result__snippet">desc a</a>
    <a class="result__a" href="https://b/">Beta</a>
    <a class="result__snippet">desc b</a>
    <a class="result__a" href="https://c/">Gamma</a>
    <a class="result__snippet">desc c</a>
    """
    urls = [url for _, url, _ in _parse_results(html)]
    titles = [t for t, _, _ in _parse_results(html)]
    assert urls == ["https://a/", "https://b/", "https://c/"]
    assert titles == ["Alpha", "Beta", "Gamma"]


def test_result_parser_html_entities_unescaped() -> None:
    # &amp; and &lt; should appear decoded in the final tuple.
    html = """
    <a class="result__a" href="https://example.com/">Rock &amp; Roll</a>
    <a class="result__snippet">1 &lt; 2</a>
    """
    [(title, _, desc)] = _parse_results(html)
    assert title == "Rock & Roll"
    assert desc == "1 < 2"


def test_result_parser_skips_entry_without_snippet() -> None:
    # The append happens only on the snippet closing tag — an orphan
    # result__a without a following result__snippet is silently dropped.
    html = """
    <a class="result__a" href="https://orphan/">Orphan</a>
    <a class="result__a" href="https://good/">Good</a>
    <a class="result__snippet">desc good</a>
    """
    results = _parse_results(html)
    assert results == [("Good", "https://good/", "desc good")]


def test_result_parser_empty_page_returns_empty() -> None:
    html = "<html><body><p>No search results here.</p></body></html>"
    assert _parse_results(html) == []


# ---------------------------------------------------------------------------
# _NextFormParser
# ---------------------------------------------------------------------------


def _next_params(html_text: str) -> dict[str, str] | None:
    p = _NextFormParser()
    p.feed(html_text)
    return p.next_params()


def test_next_form_parser_returns_hidden_inputs_dict() -> None:
    html = """
    <form>
      <input type="hidden" name="q" value="foo">
      <input type="hidden" name="s" value="30">
      <input type="hidden" name="dc" value="31">
      <input type="submit" value="Next">
    </form>
    """
    assert _next_params(html) == {"q": "foo", "s": "30", "dc": "31"}


def test_next_form_parser_no_next_submit_returns_none() -> None:
    html = """
    <form>
      <input type="hidden" name="q" value="foo">
      <input type="submit" value="Search">
    </form>
    """
    assert _next_params(html) is None


def test_next_form_parser_ignores_other_forms_without_next_button() -> None:
    html = """
    <form>
      <input type="hidden" name="login" value="1">
      <input type="submit" value="Login">
    </form>
    <form>
      <input type="hidden" name="q" value="foo">
      <input type="hidden" name="s" value="60">
      <input type="submit" value="Next">
    </form>
    """
    assert _next_params(html) == {"q": "foo", "s": "60"}


def test_next_form_parser_empty_page_returns_none() -> None:
    assert _next_params("<html><body></body></html>") is None
