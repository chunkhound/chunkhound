"""Tests for UnifiedSearch symbol extraction robustness."""

import re

import pytest

from chunkhound.services.research.shared.unified_search import UnifiedSearch


def test_build_symbol_regex_supports_nonword_prefix() -> None:
    pattern = UnifiedSearch._build_symbol_regex("$request")

    assert re.search(pattern, " $request ") is not None
    assert re.search(pattern, "$request ") is not None
    assert re.search(pattern, " $requestX ") is None


def test_build_symbol_regex_keeps_word_boundary_for_identifiers() -> None:
    pattern = UnifiedSearch._build_symbol_regex("calculate_tax")
    assert pattern == r"\bcalculate_tax\b"


@pytest.mark.asyncio
async def test_extract_symbols_handles_dict_parameters() -> None:
    search = UnifiedSearch(db_services=None, embedding_manager=None)  # type: ignore[arg-type]

    chunks = [
        {
            "symbol": "AccessQuickLinkController",
            "metadata": {
                "parameters": [
                    {"type": "Request", "name": "$request"},
                    {"type": "Resource", "name": "$resource"},
                ]
            },
        }
    ]

    symbols = await search.extract_symbols_from_chunks(chunks)
    assert "AccessQuickLinkController" in symbols
    assert "$request" in symbols
    assert "$resource" in symbols
    assert "Request" in symbols
    assert "Resource" in symbols


@pytest.mark.asyncio
async def test_extract_symbols_handles_string_parameters() -> None:
    search = UnifiedSearch(db_services=None, embedding_manager=None)  # type: ignore[arg-type]

    chunks = [
        {
            "symbol": "complete",
            "metadata": {"parameters": ["request", "resource"]},
        }
    ]

    symbols = await search.extract_symbols_from_chunks(chunks)
    assert "complete" in symbols
    assert "request" in symbols
    assert "resource" in symbols


@pytest.mark.asyncio
async def test_extract_symbols_handles_parameters_as_string() -> None:
    search = UnifiedSearch(db_services=None, embedding_manager=None)  # type: ignore[arg-type]

    chunks = [
        {
            "symbol": "thing",
            "metadata": {"parameters": "alpha"},
        }
    ]

    symbols = await search.extract_symbols_from_chunks(chunks)
    assert "thing" in symbols
    assert "alpha" in symbols
