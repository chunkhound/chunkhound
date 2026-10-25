"""Contract tests for raw Markdown content types in fetchurl."""

from __future__ import annotations

import asyncio
import base64
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

from chunkhound.utils.websearch_core import (
    _await_with_page_limit,
    _fetch_page,
    fetch_url_to_content,
)


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
@pytest.mark.parametrize(
    ("body_result", "expected_body"),
    [
        (("# Raw Markdown\n", False), b"# Raw Markdown\n"),
        (
            (base64.b64encode(b"# Raw Markdown\n").decode("ascii"), True),
            b"# Raw Markdown\n",
        ),
    ],
    ids=["plain-body", "base64-body"],
)
async def test_fetch_page_reads_raw_text_from_pinned_cdp_response_shape(
    body_result: tuple[str, bool], expected_body: bytes
) -> None:
    class _ResponseReceived:
        pass

    class _DataReceived:
        pass

    class _Command:
        def __init__(self, name: str) -> None:
            self.name = name

    enable_command = _Command("enable")
    navigate_command = _Command("navigate")
    body_command = _Command("get_response_body")

    class Network:
        ResponseReceived = _ResponseReceived
        DataReceived = _DataReceived

        @staticmethod
        def enable() -> _Command:
            return enable_command

        @staticmethod
        def get_response_body(*, request_id: str) -> _Command:
            assert request_id == "request"
            return body_command

    class Page:
        @staticmethod
        def navigate(*, url: str) -> _Command:
            assert url == "https://example.test/raw.md"
            return navigate_command

    cdp = SimpleNamespace(network=Network, page=Page)
    zendriver = ModuleType("zendriver")
    zendriver.cdp = cdp

    response = SimpleNamespace(
        headers={"content-type": "text/plain; charset=utf-8"},
        url="https://example.test/raw.md",
        charset="utf-8",
    )
    event = SimpleNamespace(loader_id="loader", request_id="request", response=response)

    class Tab:
        def __init__(self) -> None:
            self.handlers = {}

        def add_handler(self, event_type, handler) -> None:
            assert event_type in {_ResponseReceived, _DataReceived}
            self.handlers[event_type] = handler

        async def send(self, command):
            if command is enable_command:
                return None
            if command is navigate_command:
                await self.handlers[_ResponseReceived](event)
                return ("frame", "loader", None)
            if command is body_command:
                return body_result
            raise AssertionError(f"Unexpected command: {command!r}")

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

    with patch.object(
        Network, "get_response_body", return_value=body_command
    ) as get_body:
        with patch.dict(sys.modules, {"zendriver": zendriver}):
            result = await _fetch_page(Browser(), "https://example.test/raw.md")

    get_body.assert_called_once_with(request_id="request")
    assert result == ("text/plain", expected_body, "utf-8")


@pytest.mark.asyncio
async def test_page_limit_guard_cancels_cdp_operation_with_caller() -> None:
    operation_cancelled = asyncio.Event()

    async def blocked() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            operation_cancelled.set()

    task = asyncio.create_task(_await_with_page_limit(blocked(), asyncio.Event()))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert operation_cancelled.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["headers", "network-stream", "rendered-dom"])
async def test_fetch_page_rejects_oversized_cdp_page_before_body_copy(
    boundary: str, monkeypatch
) -> None:
    """CDP rejects oversized network/DOM content before materializing the body."""

    class _ResponseReceived:
        pass

    class _DataReceived:
        pass

    class _Command:
        def __init__(self, name: str) -> None:
            self.name = name

    enable_command = _Command("enable")
    navigate_command = _Command("navigate")
    body_command = _Command("get_response_body")

    class Network:
        ResponseReceived = _ResponseReceived
        DataReceived = _DataReceived

        @staticmethod
        def enable() -> _Command:
            return enable_command

        @staticmethod
        def get_response_body(*, request_id: str) -> _Command:
            return body_command

    class Page:
        @staticmethod
        def navigate(*, url: str) -> _Command:
            return navigate_command

    cdp = SimpleNamespace(network=Network, page=Page)
    zendriver = ModuleType("zendriver")
    zendriver.cdp = cdp
    content_type = (
        "text/plain" if boundary in {"headers", "network-stream"} else "text/html"
    )
    headers = {"content-type": content_type}
    if boundary == "headers":
        headers["content-length"] = "11"
    response = SimpleNamespace(headers=headers, url="https://example.test/page")
    event = SimpleNamespace(loader_id="loader", request_id="request", response=response)

    class Tab:
        def __init__(self) -> None:
            self.handlers = {}
            self.body_requested = False
            self.content_requested = False

        def add_handler(self, event_type, handler) -> None:
            self.handlers[event_type] = handler

        async def send(self, command):
            if command is enable_command:
                return None
            if command is navigate_command:
                await self.handlers[_ResponseReceived](event)
                if boundary == "network-stream":
                    await self.handlers[_DataReceived](
                        SimpleNamespace(
                            request_id="request",
                            data_length=11,
                            encoded_data_length=11,
                        )
                    )
                return "frame", "loader", None
            if command is body_command:
                self.body_requested = True
                return "body", False
            raise AssertionError(f"Unexpected command: {command!r}")

        async def wait(self) -> None:
            return None

        async def evaluate(self, expression: str) -> dict[str, object]:
            assert "TextEncoder" in expression
            if boundary == "rendered-dom":
                return {"oversized": True}
            return {"html": "x" * 10}

        async def get_content(self) -> str:
            self.content_requested = True
            raise AssertionError("CDP HTML must come from the checked snapshot")

        async def close(self) -> None:
            return None

    tab = Tab()

    class Browser:
        async def get(self, url: str, *, new_tab: bool):
            return tab

    monkeypatch.setattr("chunkhound.utils.websearch_core._MAX_PAGE_BYTES", 10)
    with patch.dict(sys.modules, {"zendriver": zendriver}):
        with pytest.raises(ValueError, match="exceeds maximum page size"):
            await _fetch_page(Browser(), "https://example.test/page")

    assert not tab.body_requested
    assert not tab.content_requested
