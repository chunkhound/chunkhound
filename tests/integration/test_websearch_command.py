"""Integration contracts for the in-process websearch CLI flow."""

import argparse
import asyncio
import urllib.error
from unittest.mock import AsyncMock, MagicMock

import pytest

from chunkhound.api.cli.commands import websearch as ws_mod


def make_args(query: str = "q") -> argparse.Namespace:
    return argparse.Namespace(query=query, limit=30, verbose=False)


def search_results() -> list[tuple[str, str, str]]:
    return [("Title", "https://example.invalid/page", "snippet")]


@pytest.fixture
def providers(monkeypatch):
    embedding = MagicMock()
    llm = MagicMock()
    monkeypatch.setattr(ws_mod, "setup_embedding_manager", lambda *_: embedding)
    monkeypatch.setattr(ws_mod, "setup_llm_manager", lambda *_: llm)
    monkeypatch.setattr(ws_mod, "expand_web_queries", AsyncMock(return_value=["q"]))
    return embedding, llm


@pytest.mark.asyncio
async def test_websearch_command_runs_research_in_process(
    monkeypatch, providers
) -> None:
    monkeypatch.setattr(
        ws_mod, "search_multi", AsyncMock(return_value=search_results())
    )

    async def pages(*args, **kwargs):
        yield "https://example.invalid/page", ".md", "# In memory"

    seen: dict[str, object] = {}

    async def research(query, page_stream, *args, **kwargs):
        seen["async"] = hasattr(page_stream, "__aiter__")
        seen["pages"] = [page async for page in page_stream]
        seen["progress"] = hasattr(kwargs["progress"], "emit_event")
        return {"answer": "ANSWER"}

    monkeypatch.setattr(ws_mod, "fetch_pages", pages)
    monkeypatch.setattr(ws_mod, "research_web_pages", research)
    rendered: list[str] = []
    monkeypatch.setattr(
        ws_mod.RichOutputFormatter,
        "text_block",
        lambda self, text: rendered.append(text),
    )

    await ws_mod.websearch_command(make_args(), MagicMock())

    assert seen["async"] is True
    assert seen["pages"] == [("https://example.invalid/page", ".md", "# In memory")]
    assert seen["progress"] is True
    assert rendered == ["ANSWER"]
    assert not hasattr(ws_mod, "tempfile")
    assert not hasattr(ws_mod, "subprocess")


@pytest.mark.asyncio
async def test_websearch_timeout_covers_query_expansion(monkeypatch, providers) -> None:
    async def blocked(*args, **kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(ws_mod, "expand_web_queries", blocked)
    monkeypatch.setattr(ws_mod, "websearch_timeout", lambda: 0.01)

    with pytest.raises(SystemExit) as exc:
        await ws_mod.websearch_command(make_args(), MagicMock())

    assert exc.value.code == 124


@pytest.mark.asyncio
async def test_websearch_command_urlerror_exits_1(monkeypatch, providers) -> None:
    monkeypatch.setattr(
        ws_mod,
        "search_multi",
        AsyncMock(side_effect=urllib.error.URLError("boom")),
    )

    with pytest.raises(SystemExit) as exc:
        await ws_mod.websearch_command(make_args(), MagicMock())

    assert exc.value.code == 1


@pytest.mark.asyncio
async def test_websearch_command_empty_results_exits_10(monkeypatch, providers) -> None:
    monkeypatch.setattr(ws_mod, "search_multi", AsyncMock(return_value=[]))

    with pytest.raises(SystemExit) as exc:
        await ws_mod.websearch_command(make_args("zero"), MagicMock())

    assert exc.value.code == 10
