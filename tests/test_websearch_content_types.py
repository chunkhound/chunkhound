"""Contract tests for raw Markdown content types in fetchurl."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from chunkhound.utils.websearch_core import fetch_url_to_content


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
