"""Unit tests for the remote-config HTTP fetcher.

Covers the private ``_interpolate_env`` helper directly and the scheme +
redirect contract of ``fetch``. The pipeline e2e tests replace
``fetcher.fetch`` with a fake, so this module is the sole guardian of:

- ``_interpolate_env``'s two branches (unset-var → drop header,
  set-var → substitute)
- HTTPS-or-loopback URL enforcement (initial URL and every redirect hop)
- Cross-origin ``Authorization`` stripping on redirects
- The redirect hop cap
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from chunkhound.core.config.remote.fetcher import _interpolate_env, fetch


def _install_mock_transport(
    monkeypatch: pytest.MonkeyPatch,
    handler: Any,
) -> list[httpx.Request]:
    """Route every ``httpx.AsyncClient`` request through ``handler``.

    Returns a list that the caller can inspect after the fetch to see
    every request the fetcher actually dispatched.
    """
    seen: list[httpx.Request] = []

    def recording_handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    transport = httpx.MockTransport(recording_handler)
    real_client_cls = httpx.AsyncClient

    def make_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real_client_cls(*args, **kwargs)

    monkeypatch.setattr(
        "chunkhound.core.config.remote.fetcher.httpx.AsyncClient", make_client
    )
    return seen


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
    # redirect following, the 301's non-JSON body would trip
    # envelope_parse_error. Regression for CDN/path-rewriter deployments.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/config":
            return httpx.Response(
                301, headers={"Location": "https://example.com/final"}
            )
        return httpx.Response(200, json={"remote": {"config": True}})

    _install_mock_transport(monkeypatch, handler)

    result = await fetch("https://example.com/config", auth_header=None)
    assert result == {"remote": {"config": True}}


# ---------------------------------------------------------------------------
# URL scheme / loopback enforcement
# ---------------------------------------------------------------------------


async def test_fetch_refuses_http_url(monkeypatch: pytest.MonkeyPatch) -> None:
    # HTTP to a non-loopback host must be refused before any request
    # dispatches — otherwise the Authorization header would be sent in
    # cleartext.
    seen = _install_mock_transport(
        monkeypatch, lambda r: httpx.Response(200, json={})
    )
    result = await fetch("http://example.com/config", auth_header="Bearer x")
    assert result is None
    assert seen == []


async def test_fetch_accepts_https_url(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_mock_transport(
        monkeypatch, lambda r: httpx.Response(200, json={"ok": True})
    )
    result = await fetch("https://example.com/config", auth_header=None)
    assert result == {"ok": True}


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8080/config",
        "http://LOCALHOST:8080/config",
        "http://127.0.0.1:8080/config",
        "http://127.0.0.42/config",
        "http://[::1]:8080/config",
    ],
)
async def test_fetch_accepts_http_loopback(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    _install_mock_transport(
        monkeypatch, lambda r: httpx.Response(200, json={"ok": True})
    )
    result = await fetch(url, auth_header=None)
    assert result == {"ok": True}


@pytest.mark.parametrize(
    "url",
    [
        "http://192.168.1.1/config",
        "http://10.0.0.1/config",
        "http://example.com/config",
        # `ftp` — non-http/https schemes are refused too.
        "ftp://localhost/config",
    ],
)
async def test_fetch_refuses_non_https_non_loopback(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    seen = _install_mock_transport(
        monkeypatch, lambda r: httpx.Response(200, json={})
    )
    result = await fetch(url, auth_header=None)
    assert result is None
    assert seen == []


# ---------------------------------------------------------------------------
# Redirect enforcement
# ---------------------------------------------------------------------------


async def test_fetch_rejects_https_to_http_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A same-host HTTPS → HTTP downgrade must be refused. The terminal
    # HTTP request must never dispatch, otherwise the Authorization
    # header lands on the wire in cleartext.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.scheme == "https":
            return httpx.Response(
                301, headers={"Location": "http://example.com/final"}
            )
        return httpx.Response(200, json={"leaked": True})

    seen = _install_mock_transport(monkeypatch, handler)

    result = await fetch(
        "https://example.com/config", auth_header="Bearer sekret"
    )
    assert result is None
    # Only the initial HTTPS request was dispatched; the http:// hop was
    # blocked before send.
    assert len(seen) == 1
    assert seen[0].url.scheme == "https"


async def test_fetch_accepts_https_to_https_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/config":
            return httpx.Response(
                301, headers={"Location": "https://cdn.example.com/final"}
            )
        return httpx.Response(200, json={"ok": True})

    _install_mock_transport(monkeypatch, handler)

    result = await fetch("https://example.com/config", auth_header=None)
    assert result == {"ok": True}


async def test_fetch_strips_auth_on_cross_origin_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A bearer token issued for the configured endpoint must not be
    # forwarded to a different origin's server, even when the target is
    # HTTPS.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "example.com":
            return httpx.Response(
                301, headers={"Location": "https://cdn.other.com/final"}
            )
        return httpx.Response(200, json={"ok": True})

    seen = _install_mock_transport(monkeypatch, handler)

    result = await fetch(
        "https://example.com/config", auth_header="Bearer sekret"
    )
    assert result == {"ok": True}
    assert len(seen) == 2
    assert seen[0].headers.get("Authorization") == "Bearer sekret"
    assert "authorization" not in {k.lower() for k in seen[1].headers.keys()}


async def test_fetch_carries_auth_on_same_origin_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/config":
            return httpx.Response(
                301, headers={"Location": "https://example.com/final"}
            )
        return httpx.Response(200, json={"ok": True})

    seen = _install_mock_transport(monkeypatch, handler)

    result = await fetch(
        "https://example.com/config", auth_header="Bearer sekret"
    )
    assert result == {"ok": True}
    assert len(seen) == 2
    assert seen[0].headers.get("Authorization") == "Bearer sekret"
    assert seen[1].headers.get("Authorization") == "Bearer sekret"


async def test_fetch_carries_auth_when_redirect_adds_default_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A CDN that emits ``Location: https://example.com:443/final`` is
    # same-origin — the explicit default port must not look cross-origin
    # or the operator's bearer token would be dropped for no reason.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/config":
            return httpx.Response(
                301, headers={"Location": "https://example.com:443/final"}
            )
        return httpx.Response(200, json={"ok": True})

    seen = _install_mock_transport(monkeypatch, handler)

    result = await fetch(
        "https://example.com/config", auth_header="Bearer sekret"
    )
    assert result == {"ok": True}
    assert len(seen) == 2
    assert seen[1].headers.get("Authorization") == "Bearer sekret"


async def test_fetch_client_constructed_with_trust_env_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Flag pin — asserts the kwarg is passed. httpx owns the behavioral
    # contract that ``trust_env=False`` suppresses ``HTTPS_PROXY`` /
    # ``SSL_CERT_FILE`` / netrc; re-verifying that here would test the
    # library, not our code. What we own is: the flag stays set. Any edit
    # that drops or flips ``trust_env=False`` in fetcher.py must break
    # this test.
    captured_kwargs: dict[str, Any] = {}
    real_client_cls = httpx.AsyncClient

    def capturing_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        captured_kwargs.update(kwargs)
        kwargs["transport"] = httpx.MockTransport(
            lambda r: httpx.Response(200, json={"ok": True})
        )
        return real_client_cls(*args, **kwargs)

    monkeypatch.setattr(
        "chunkhound.core.config.remote.fetcher.httpx.AsyncClient",
        capturing_client,
    )

    result = await fetch("https://example.com/config", auth_header=None)
    assert result == {"ok": True}
    assert captured_kwargs.get("trust_env") is False


async def test_fetch_aborts_on_redirect_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Every hop redirects to the next; after _MAX_REDIRECTS the fetcher
    # aborts. Guards against a hostile server exhausting the timeout
    # budget via infinite redirects.
    def handler(request: httpx.Request) -> httpx.Response:
        # Extract trailing integer from path and increment.
        n = int(request.url.path.rsplit("/", 1)[-1])
        return httpx.Response(
            301, headers={"Location": f"https://example.com/hop/{n + 1}"}
        )

    seen = _install_mock_transport(monkeypatch, handler)

    result = await fetch("https://example.com/hop/0", auth_header=None)
    assert result is None
    # _MAX_REDIRECTS = 5, so budget is 6 hops (initial + 5 follows) — the
    # 7th hop would be the one that trips the abort. The loop dispatches
    # _MAX_REDIRECTS + 1 = 6 requests in total before raising.
    assert len(seen) == 6
