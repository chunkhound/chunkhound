"""Contract tests for raw Markdown content types in fetchurl."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

from chunkhound.utils.websearch_core import _fetch_page, fetch_url_to_content


@pytest.mark.asyncio
@pytest.mark.parametrize("content_type", ["text/plain", "text/markdown"])
async def test_raw_markdown_content_types_are_returned_without_html_rendering(
    content_type: str,
) -> None:
    body = b"# Raw Markdown\n\n**body**\n"

    with patch(
        "chunkhound.utils.websearch_core._fetch_url",
        return_value=(content_type, body, "utf-8"),
    ):
        kind, content, metadata = await fetch_url_to_content(
            "https://example.test/raw.md", browser=None
        )

    assert kind == ".md"
    assert content == body.decode("utf-8")
    assert metadata == {"title": None}


@pytest.mark.asyncio
async def test_fetch_page_reads_raw_text_from_pinned_cdp_response_shape() -> None:
    class _ResponseReceived:
        pass

    class Network:
        ResponseReceived = _ResponseReceived

        @staticmethod
        def enable() -> str:
            return "enable"

        @staticmethod
        def get_response_body(*, request_id: str) -> tuple[str, str]:
            return ("body", request_id)

    class Page:
        @staticmethod
        def navigate(*, url: str) -> tuple[str, str, None]:
            return ("frame", "loader", None)

    cdp = SimpleNamespace(network=Network, page=Page)
    zendriver = ModuleType("zendriver")
    zendriver.cdp = cdp

    response = SimpleNamespace(
        headers={"content-type": "text/plain; charset=utf-8"},
        url="https://example.test/raw.md",
        charset="utf-8",
    )
    event = SimpleNamespace(
        loader_id="loader", request_id="request", response=response
    )
    body_result = ("# Raw Markdown\n", False)

    class Tab:
        def __init__(self) -> None:
            self.handler = None

        def add_handler(self, event_type, handler) -> None:
            assert event_type is _ResponseReceived
            self.handler = handler

        async def send(self, command):
            if command == "enable":
                return None
            if command == ("frame", "loader", None):
                assert self.handler is not None
                await self.handler(event)
                return command
            assert command == ("body", "request")
            return body_result

        async def wait(self) -> None:
            return None

        async def close(self) -> None:
            return None

    tab = Tab()

    class Browser:
        async def get(self, url: str, *, new_tab: bool):
            assert url == "about:blank"
            assert new_tab
            return tab

    with patch.dict(sys.modules, {"zendriver": zendriver}):
        result = await _fetch_page(Browser(), "https://example.test/raw.md")

    assert result == ("text/plain", b"# Raw Markdown\n", "utf-8")
