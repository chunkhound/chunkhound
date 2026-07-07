"""Unit tests for the LLM-driven websearch query expander.

Covers:
- None-llm fallback, exception fallback, empty-result fallback
- Pad-to-3 and truncate-to-3
- Per-variant quote-preservation filtering with pad-based recovery
- ``search_query`` normalization: present / empty / missing / non-string /
  whitespace-only — every non-useful shape collapses to raw ``query``.
- ``previous_query`` steering: ``Previous Query:`` line inside
  ``<CURRENT_CONTEXT>``, empty-string coercion, quoted-previous-query
  survival through the prompt.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from chunkhound.utils import websearch_expansion as we_mod


class _FakeProvider:
    def __init__(self, payload: Any) -> None:
        self._payload = payload
        self.calls: list[dict[str, Any]] = []

    async def complete_structured(
        self,
        prompt: str,
        json_schema: dict[str, Any],
        system: str | None = None,
        max_completion_tokens: int = 4096,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        self.calls.append({"prompt": prompt, "system": system})
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _FakeLLMManager:
    def __init__(self, provider: _FakeProvider) -> None:
        self._provider = provider

    def get_utility_provider(self) -> _FakeProvider:
        return self._provider


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# queries[] contract (migration: assertions now target `.queries` field)
# ---------------------------------------------------------------------------


def test_none_llm_manager_returns_original_query() -> None:
    out = _run(we_mod.expand_web_queries("hello world", None))
    assert out.queries == ["hello world"]


def test_llm_provider_raises_falls_back_to_original() -> None:
    llm = _FakeLLMManager(_FakeProvider(RuntimeError("boom")))
    out = _run(we_mod.expand_web_queries("hello", llm))
    assert out.queries == ["hello"]


def test_empty_queries_list_falls_back_and_pads() -> None:
    # LLM returns empty — post-processing subs [query] and pads to 3.
    llm = _FakeLLMManager(_FakeProvider({"queries": [], "search_query": "hello"}))
    year = datetime.now().year
    out = _run(we_mod.expand_web_queries("hello", llm))
    assert out.queries == ["hello", f"hello {year}", "hello best practices"]


def test_pad_to_three_when_llm_returns_one() -> None:
    llm = _FakeLLMManager(
        _FakeProvider({"queries": ["one"], "search_query": "hello"})
    )
    year = datetime.now().year
    out = _run(we_mod.expand_web_queries("hello", llm))
    assert out.queries == ["one", f"hello {year}", "hello best practices"]


def test_pad_to_three_when_llm_returns_two() -> None:
    llm = _FakeLLMManager(
        _FakeProvider({"queries": ["one", "two"], "search_query": "hello"})
    )
    out = _run(we_mod.expand_web_queries("hello", llm))
    assert out.queries == ["one", "two", "hello best practices"]


def test_truncate_to_three_when_llm_returns_five() -> None:
    llm = _FakeLLMManager(
        _FakeProvider({"queries": ["a", "b", "c", "d", "e"], "search_query": "hello"})
    )
    out = _run(we_mod.expand_web_queries("hello", llm))
    assert out.queries == ["a", "b", "c"]


def test_empty_and_whitespace_queries_are_stripped_before_padding() -> None:
    llm = _FakeLLMManager(
        _FakeProvider({"queries": ["real one", "  ", ""], "search_query": "hello"})
    )
    year = datetime.now().year
    out = _run(we_mod.expand_web_queries("hello", llm))
    assert out.queries == ["real one", f"hello {year}", "hello best practices"]


def test_all_variants_drop_quotes_pads_from_raw_query() -> None:
    llm = _FakeLLMManager(
        _FakeProvider(
            {
                "queries": ["React alt 2025", "React vs Vue", "React vs Angular"],
                "search_query": 'compare "React 19" vs Vue',
            }
        )
    )
    query = 'compare "React 19" vs Vue'
    year = datetime.now().year
    out = _run(we_mod.expand_web_queries(query, llm))
    assert out.queries == [query, f"{query} {year}", f"{query} best practices"]


def test_quote_preservation_success_returns_llm_queries() -> None:
    llm = _FakeLLMManager(
        _FakeProvider(
            {
                "queries": [
                    '"React 19" concurrent features',
                    '"React 19" performance tips 2025',
                    '"React 19" upgrade guide',
                ],
                "search_query": 'compare "React 19" vs Vue',
            }
        )
    )
    query = 'compare "React 19" vs Vue'
    out = _run(we_mod.expand_web_queries(query, llm))
    assert out.queries == [
        '"React 19" concurrent features',
        '"React 19" performance tips 2025',
        '"React 19" upgrade guide',
    ]


def test_partial_quote_preservation_filters_and_pads() -> None:
    llm = _FakeLLMManager(
        _FakeProvider(
            {
                "queries": ["no quote here", '"React 19" tips', '"React 19" upgrade'],
                "search_query": 'compare "React 19" vs Vue',
            }
        )
    )
    query = 'compare "React 19" vs Vue'
    out = _run(we_mod.expand_web_queries(query, llm))
    assert out.queries == [
        '"React 19" tips',
        '"React 19" upgrade',
        f"{query} best practices",
    ]


def test_single_quote_preserving_variant_pads_to_three() -> None:
    llm = _FakeLLMManager(
        _FakeProvider(
            {
                "queries": ["no quote", '"React 19" upgrade', "another no quote"],
                "search_query": 'compare "React 19" vs Vue',
            }
        )
    )
    query = 'compare "React 19" vs Vue'
    year = datetime.now().year
    out = _run(we_mod.expand_web_queries(query, llm))
    assert out.queries == [
        '"React 19" upgrade',
        f"{query} {year}",
        f"{query} best practices",
    ]


def test_unbalanced_quote_passes_through() -> None:
    llm = _FakeLLMManager(
        _FakeProvider({"queries": ["a", "b", "c"], "search_query": 'foo " bar'})
    )
    out = _run(we_mod.expand_web_queries('foo " bar', llm))
    assert out.queries == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# search_query normalization contract
# ---------------------------------------------------------------------------


def test_search_query_returned_when_present() -> None:
    llm = _FakeLLMManager(
        _FakeProvider(
            {"queries": ["a", "b", "c"], "search_query": "React performance"}
        )
    )
    out = _run(we_mod.expand_web_queries("How to make React fast?", llm))
    assert out.search_query == "React performance"


def test_search_query_empty_falls_back_to_raw() -> None:
    llm = _FakeLLMManager(
        _FakeProvider({"queries": ["a", "b", "c"], "search_query": ""})
    )
    out = _run(we_mod.expand_web_queries("raw query", llm))
    assert out.search_query == "raw query"


def test_search_query_missing_falls_back_to_raw() -> None:
    llm = _FakeLLMManager(_FakeProvider({"queries": ["a", "b", "c"]}))
    out = _run(we_mod.expand_web_queries("raw query", llm))
    assert out.search_query == "raw query"


def test_search_query_non_string_falls_back_to_raw() -> None:
    llm = _FakeLLMManager(
        _FakeProvider({"queries": ["a", "b", "c"], "search_query": 42})
    )
    out = _run(we_mod.expand_web_queries("raw query", llm))
    assert out.search_query == "raw query"


def test_search_query_whitespace_only_falls_back_to_raw() -> None:
    llm = _FakeLLMManager(
        _FakeProvider({"queries": ["a", "b", "c"], "search_query": "   \n\t "})
    )
    out = _run(we_mod.expand_web_queries("raw query", llm))
    assert out.search_query == "raw query"


def test_none_llm_manager_normalized_is_raw() -> None:
    out = _run(we_mod.expand_web_queries("raw query", None))
    assert out.search_query == "raw query"


def test_llm_exception_normalized_is_raw() -> None:
    llm = _FakeLLMManager(_FakeProvider(RuntimeError("boom")))
    out = _run(we_mod.expand_web_queries("raw query", llm))
    assert out.search_query == "raw query"


# ---------------------------------------------------------------------------
# previous_query steering: prompt contents
# ---------------------------------------------------------------------------


def test_prompt_contains_search_query_schema() -> None:
    """Guard against silent removal of the `search_query` field instruction."""
    provider = _FakeProvider({"queries": ["a", "b", "c"], "search_query": "x"})
    llm = _FakeLLMManager(provider)
    _run(we_mod.expand_web_queries("hello", llm))
    assert provider.calls, "provider should have been called once"
    rendered = provider.calls[0]["prompt"]
    assert "search_query" in rendered


def test_previous_query_appears_in_prompt() -> None:
    provider = _FakeProvider({"queries": ["a", "b", "c"], "search_query": "hello"})
    llm = _FakeLLMManager(provider)
    _run(
        we_mod.expand_web_queries(
            "current topic", llm, previous_query="prior topic"
        )
    )
    rendered = provider.calls[0]["prompt"]
    assert 'Previous Query: "prior topic"' in rendered


def test_baseline_prompt_omits_previous_query_line() -> None:
    provider = _FakeProvider({"queries": ["a", "b", "c"], "search_query": "hello"})
    llm = _FakeLLMManager(provider)
    _run(we_mod.expand_web_queries("current only", llm))
    rendered = provider.calls[0]["prompt"]
    assert "Previous Query:" not in rendered


def test_empty_previous_query_treated_as_none() -> None:
    provider = _FakeProvider({"queries": ["a", "b", "c"], "search_query": "hello"})
    llm = _FakeLLMManager(provider)
    _run(we_mod.expand_web_queries("current only", llm, previous_query=""))
    rendered = provider.calls[0]["prompt"]
    assert "Previous Query:" not in rendered


def test_quoted_previous_query_survives_in_prompt() -> None:
    """Quoted phrases in previous_query reach the LLM verbatim.

    The `_preserves_quotes` filter only guards the current `query` — this
    test locks the guarantee that at minimum the raw quoted text of
    `previous_query` reaches the LLM.
    """
    provider = _FakeProvider({"queries": ["a", "b", "c"], "search_query": "hello"})
    llm = _FakeLLMManager(provider)
    _run(
        we_mod.expand_web_queries(
            "What alternatives exist?",
            llm,
            previous_query='"DuckDB performance tuning"',
        )
    )
    rendered = provider.calls[0]["prompt"]
    # Outer quotes come from the template's `Previous Query: "{previous_query}"`
    # slot, inner quotes come from the caller's input.
    assert 'Previous Query: ""DuckDB performance tuning""' in rendered
