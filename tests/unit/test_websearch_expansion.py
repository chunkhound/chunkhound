"""Unit tests for the LLM-driven websearch query expander.

Covers:
- None-llm fallback, exception fallback, empty-result fallback
- Pad-to-3 and truncate-to-3
- Per-variant quote-preservation filtering with pad-based recovery
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


def test_none_llm_manager_returns_original_query() -> None:
    out = _run(we_mod.expand_web_queries("hello world", None))
    assert out == ["hello world"]


def test_llm_provider_raises_falls_back_to_original() -> None:
    llm = _FakeLLMManager(_FakeProvider(RuntimeError("boom")))
    out = _run(we_mod.expand_web_queries("hello", llm))
    assert out == ["hello"]


def test_empty_queries_list_falls_back_and_pads() -> None:
    # LLM returns empty — post-processing subs [query] and pads to 3.
    llm = _FakeLLMManager(_FakeProvider({"queries": []}))
    year = datetime.now().year
    out = _run(we_mod.expand_web_queries("hello", llm))
    assert out == ["hello", f"hello {year}", "hello best practices"]


def test_pad_to_three_when_llm_returns_one() -> None:
    llm = _FakeLLMManager(_FakeProvider({"queries": ["one"]}))
    year = datetime.now().year
    out = _run(we_mod.expand_web_queries("hello", llm))
    assert out == ["one", f"hello {year}", "hello best practices"]


def test_pad_to_three_when_llm_returns_two() -> None:
    llm = _FakeLLMManager(_FakeProvider({"queries": ["one", "two"]}))
    out = _run(we_mod.expand_web_queries("hello", llm))
    assert out == ["one", "two", "hello best practices"]


def test_truncate_to_three_when_llm_returns_five() -> None:
    llm = _FakeLLMManager(
        _FakeProvider({"queries": ["a", "b", "c", "d", "e"]})
    )
    out = _run(we_mod.expand_web_queries("hello", llm))
    assert out == ["a", "b", "c"]


def test_empty_and_whitespace_queries_are_stripped_before_padding() -> None:
    llm = _FakeLLMManager(
        _FakeProvider({"queries": ["real one", "  ", ""]})
    )
    year = datetime.now().year
    out = _run(we_mod.expand_web_queries("hello", llm))
    assert out == ["real one", f"hello {year}", "hello best practices"]


def test_all_variants_drop_quotes_pads_from_raw_query() -> None:
    # All 3 variants drop the quoted phrase — no LLM variant survives, so
    # slot 0 is seeded with the raw query and the pad fills slots 1-2. Every
    # returned query then preserves the quoted phrase.
    llm = _FakeLLMManager(
        _FakeProvider(
            {"queries": ["React alt 2025", "React vs Vue", "React vs Angular"]}
        )
    )
    query = 'compare "React 19" vs Vue'
    year = datetime.now().year
    out = _run(we_mod.expand_web_queries(query, llm))
    assert out == [query, f"{query} {year}", f"{query} best practices"]


def test_quote_preservation_success_returns_llm_queries() -> None:
    llm = _FakeLLMManager(
        _FakeProvider(
            {
                "queries": [
                    '"React 19" concurrent features',
                    '"React 19" performance tips 2025',
                    '"React 19" upgrade guide',
                ]
            }
        )
    )
    query = 'compare "React 19" vs Vue'
    out = _run(we_mod.expand_web_queries(query, llm))
    assert out == [
        '"React 19" concurrent features',
        '"React 19" performance tips 2025',
        '"React 19" upgrade guide',
    ]


def test_partial_quote_preservation_filters_and_pads() -> None:
    # Two of three variants preserve the quoted phrase — keep those and
    # let the pad fallback fill the missing slot.
    llm = _FakeLLMManager(
        _FakeProvider(
            {"queries": ["no quote here", '"React 19" tips', '"React 19" upgrade']}
        )
    )
    query = 'compare "React 19" vs Vue'
    out = _run(we_mod.expand_web_queries(query, llm))
    assert out == [
        '"React 19" tips',
        '"React 19" upgrade',
        f"{query} best practices",
    ]


def test_single_quote_preserving_variant_pads_to_three() -> None:
    # Only one variant preserves the quote — both pad fallbacks kick in.
    llm = _FakeLLMManager(
        _FakeProvider(
            {"queries": ["no quote", '"React 19" upgrade', "another no quote"]}
        )
    )
    query = 'compare "React 19" vs Vue'
    year = datetime.now().year
    out = _run(we_mod.expand_web_queries(query, llm))
    assert out == [
        '"React 19" upgrade',
        f"{query} {year}",
        f"{query} best practices",
    ]


def test_multi_quote_partial_preservation_filtered_out() -> None:
    # Multi-quote query: variants that keep only one of the two quoted
    # phrases are filtered out. Only the variant preserving both survives;
    # pad fallbacks refill from the raw query (which carries both quotes).
    llm = _FakeLLMManager(
        _FakeProvider(
            {
                "queries": [
                    '"foo" migration guide',
                    '"foo" "bar" compatibility',
                    '"bar" best practices',
                ]
            }
        )
    )
    query = 'compare "foo" vs "bar"'
    year = datetime.now().year
    out = _run(we_mod.expand_web_queries(query, llm))
    assert out == [
        '"foo" "bar" compatibility',
        f"{query} {year}",
        f"{query} best practices",
    ]

    # Every variant partially preserves — each keeps only one of the two
    # quoted phrases, so none survive the filter. Slot 0 is seeded with the
    # raw query and pad fallbacks fill slots 1-2, all carrying both quotes.
    llm = _FakeLLMManager(
        _FakeProvider(
            {
                "queries": [
                    '"foo" migration guide',
                    '"bar" upgrade notes',
                    '"foo" tutorial',
                ]
            }
        )
    )
    out = _run(we_mod.expand_web_queries(query, llm))
    assert out == [query, f"{query} {year}", f"{query} best practices"]


def test_unbalanced_quote_passes_through() -> None:
    # An unbalanced quote yields no extractable phrases, so the filter
    # skips entirely and the LLM's queries pass through unchanged.
    llm = _FakeLLMManager(_FakeProvider({"queries": ["a", "b", "c"]}))
    out = _run(we_mod.expand_web_queries('foo " bar', llm))
    assert out == ["a", "b", "c"]
