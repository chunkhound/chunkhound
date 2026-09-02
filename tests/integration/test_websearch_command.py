"""Integration contracts for the in-process websearch CLI flow."""

import argparse
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

    research = AsyncMock(return_value={"answer": "ANSWER"})
    monkeypatch.setattr(ws_mod, "fetch_pages", pages)
    monkeypatch.setattr(ws_mod, "research_web_pages", research)
    rendered: list[str] = []
    monkeypatch.setattr(
        ws_mod.RichOutputFormatter,
        "text_block",
        lambda self, text: rendered.append(text),
    )

    await ws_mod.websearch_command(make_args(), MagicMock())

    research.assert_awaited_once()
    assert research.await_args.args[1] == [
        ("https://example.invalid/page", ".md", "# In memory")
    ]
    assert rendered == ["ANSWER"]
    assert not hasattr(ws_mod, "tempfile")
    assert not hasattr(ws_mod, "subprocess")


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
