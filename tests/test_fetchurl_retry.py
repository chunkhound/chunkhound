"""Contract test for `_fetch_with_retry` browser-death degradation.

If Chrome dies mid-fetch (CDP WebSocket closes) the retry loop must not keep
handing subsequent attempts a dead browser. Instead, it should rebind the
loop-local `browser` to `None` so `fetch_url_to_content` takes the urllib
branch for remaining attempts.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest

pytest.importorskip("websockets.exceptions")
from websockets.exceptions import ConnectionClosedError  # noqa: E402

from chunkhound.core.config.config import Config  # noqa: E402
from chunkhound.core.config.fetchurl_config import FetchUrlConfig  # noqa: E402
from chunkhound.utils.fetchurl import FetchUrlError, _fetch_with_retry  # noqa: E402


async def _noop_validate(url):
    return None


@pytest.mark.asyncio
async def test_browser_death_degrades_to_urllib_on_next_attempt(tmp_path):
    browser_sentinel = object()
    calls: list[object | None] = []

    async def fake_fetch(url, browser):
        calls.append(browser)
        if len(calls) == 1:
            raise ConnectionClosedError(None, None)
        return (".md", "ok", {"title": None})

    @asynccontextmanager
    async def fake_managed_browser(warning_callback=None):
        yield browser_sentinel

    warnings: list[str] = []

    with patch(
        "chunkhound.utils.fetchurl.fetch_url_to_content",
        side_effect=fake_fetch,
    ), patch(
        "chunkhound.utils.fetchurl._managed_browser",
        fake_managed_browser,
    ), patch(
        "chunkhound.utils.fetchurl._validate_url_and_resolve",
        side_effect=_noop_validate,
    ):
        result = await _fetch_with_retry(
            "https://example.com/",
            Config(target_dir=tmp_path),
            warning_callback=warnings.append,
        )

    assert result == (".md", "ok", {"title": None})
    assert calls == [browser_sentinel, None]
    assert any("Browser died mid-fetch" in w for w in warnings), warnings


@pytest.mark.asyncio
async def test_timeout_does_not_downgrade_browser(tmp_path):
    """`asyncio.TimeoutError` is retryable but NOT browser-fatal.

    A slow page load on a live browser must retry against the same browser,
    not silently downgrade the remaining attempts to urllib. Locks the
    "Deliberately narrow" invariant documented on `_is_browser_fatal`.
    """
    browser_sentinel = object()
    calls: list[object | None] = []

    async def fake_fetch(url, browser):
        calls.append(browser)
        if len(calls) == 1:
            raise asyncio.TimeoutError()
        return (".md", "ok", {"title": None})

    @asynccontextmanager
    async def fake_managed_browser(warning_callback=None):
        yield browser_sentinel

    warnings: list[str] = []

    with patch(
        "chunkhound.utils.fetchurl.fetch_url_to_content",
        side_effect=fake_fetch,
    ), patch(
        "chunkhound.utils.fetchurl._managed_browser",
        fake_managed_browser,
    ), patch(
        "chunkhound.utils.fetchurl._validate_url_and_resolve",
        side_effect=_noop_validate,
    ):
        result = await _fetch_with_retry(
            "https://example.com/",
            Config(target_dir=tmp_path),
            warning_callback=warnings.append,
        )

    assert result == (".md", "ok", {"title": None})
    assert calls == [browser_sentinel, browser_sentinel]
    assert not any("Browser died mid-fetch" in w for w in warnings), warnings


@pytest.mark.asyncio
async def test_browser_death_on_final_attempt_wraps_into_fetchurl_error(tmp_path):
    """Browser-fatal on the final attempt must surface as FetchUrlError.

    Without a wrap, the raw ``websockets`` exception would escape — the CLI
    exception whitelist in ``api/cli/commands/fetchurl.py`` does not catch
    it, and the MCP layer would surface a websockets-internal type name to
    clients. ``max_retries=1`` forces the single attempt to also be the
    final one, so there is no urllib fallback opportunity.
    """
    browser_sentinel = object()
    original = ConnectionClosedError(None, None)

    async def fake_fetch(url, browser):
        raise original

    @asynccontextmanager
    async def fake_managed_browser(warning_callback=None):
        yield browser_sentinel

    config = Config(target_dir=tmp_path, fetchurl=FetchUrlConfig(max_retries=1))
    warnings: list[str] = []

    with patch(
        "chunkhound.utils.fetchurl.fetch_url_to_content",
        side_effect=fake_fetch,
    ), patch(
        "chunkhound.utils.fetchurl._managed_browser",
        fake_managed_browser,
    ), patch(
        "chunkhound.utils.fetchurl._validate_url_and_resolve",
        side_effect=_noop_validate,
    ):
        with pytest.raises(FetchUrlError) as exc_info:
            await _fetch_with_retry(
                "https://example.com/",
                config,
                warning_callback=warnings.append,
            )

    assert exc_info.value.__cause__ is original
    assert "Browser transport died" in str(exc_info.value)
    assert "ConnectionClosedError" in str(exc_info.value)
