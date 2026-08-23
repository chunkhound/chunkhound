"""Unit tests for the remote-config HTTP fetcher.

Covers the private `_interpolate_env` helper directly and the redirect
contract of `fetch`. The pipeline e2e tests replace `fetcher.fetch` with
a fake, so this module is the sole guardian of `_interpolate_env`'s two
branches (unset-var → drop header, set-var → substitute) and of
`follow_redirects=True` behavior.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from chunkhound.core.config.remote.fetcher import _interpolate_env, fetch


def test_var_interpolation_dropped_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REMOTE_TOKEN", raising=False)
    assert _interpolate_env("Bearer ${REMOTE_TOKEN}") is None


def test_var_interpolation_substituted_when_env_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REMOTE_TOKEN", "sekret")
    assert _interpolate_env("Bearer ${REMOTE_TOKEN}") == "Bearer sekret"


def test_var_interpolation_dropped_when_env_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `export REMOTE_TOKEN=` must drop the header just like an unset var —
    # otherwise the wire sees `Authorization: Bearer ` (empty credential).
    monkeypatch.setenv("REMOTE_TOKEN", "")
    assert _interpolate_env("Bearer ${REMOTE_TOKEN}") is None


async def test_fetch_follows_redirect(monkeypatch: pytest.MonkeyPatch) -> None:
    # A 301 → 200 chain must resolve to the final JSON body. Without
    # follow_redirects, a 301 slips past the >=400 status check and its
    # non-JSON body trips envelope_parse_error instead. Guards the
    # follow_redirects=True contract so operators can front the endpoint
    # with a CDN or path rewriter.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/config":
            return httpx.Response(
                301, headers={"Location": "https://example.com/final"}
            )
        return httpx.Response(200, json={"remote": {"config": True}})

    transport = httpx.MockTransport(handler)
    # Capture the real class before monkeypatch swaps it out — otherwise
    # `make_client` would recurse into itself when fetcher.py constructs
    # the client.
    real_client_cls = httpx.AsyncClient

    def make_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real_client_cls(*args, **kwargs)

    monkeypatch.setattr(
        "chunkhound.core.config.remote.fetcher.httpx.AsyncClient", make_client
    )

    result = await fetch("https://example.com/config", auth_header=None)
    assert result == {"remote": {"config": True}}
