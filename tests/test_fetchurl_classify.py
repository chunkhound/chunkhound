"""Retry classification for fetchurl.

`_classify_and_raise_if_terminal` decides whether a fetch attempt failure
is terminal (re-raise) or transient (return silently so the retry loop
tries again). Three of its branches match on ValueError message prefixes
emitted from `websearch_core.py`; a rename over there would silently
break retry semantics. The last test in this module reads
`websearch_core.py` and asserts the literals still exist, so drift
surfaces as a test failure instead of a production incident.
"""

from __future__ import annotations

import asyncio
import socket
import ssl
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from chunkhound.utils import websearch_core
from chunkhound.utils.fetchurl import FetchUrlError, _classify_and_raise_if_terminal


def _http(code: int) -> HTTPError:
    return HTTPError("http://example.com", code, "msg", {}, None)


# --- Terminal (re-raise) --------------------------------------------------

@pytest.mark.parametrize(
    "exc",
    [
        FetchUrlError("boom"),
        _http(400),
        _http(404),
        _http(499),
        ssl.SSLError("handshake"),
        URLError(ssl.SSLError("wrapped")),
        ValueError("Unsupported content-type: 'application/zip'"),
        ValueError("'text/html' body rendered empty (0 bytes)"),
        ValueError("some unknown ValueError shape"),
        RuntimeError("unexpected"),
    ],
)
def test_terminal_exceptions_reraise(exc: BaseException) -> None:
    with pytest.raises(type(exc)):
        _classify_and_raise_if_terminal(exc)


# --- Retryable (silent return) -------------------------------------------

@pytest.mark.parametrize(
    "exc",
    [
        _http(429),
        _http(500),
        _http(502),
        _http(503),
        ValueError("Navigation failed: net::ERR_ABORTED"),
        URLError("connection refused"),
        TimeoutError(),
        asyncio.TimeoutError(),
        socket.timeout("read"),  # noqa: UP041 — mirror the classifier's explicit tuple
    ],
)
def test_retryable_exceptions_return_silently(exc: BaseException) -> None:
    # No return value; the contract is "does not raise".
    _classify_and_raise_if_terminal(exc)


# --- Cross-module string-drift guard -------------------------------------

_WEBSEARCH_CORE = Path(websearch_core.__file__)


@pytest.mark.parametrize(
    "literal",
    [
        # These prefixes are matched verbatim by `_classify_and_raise_if_terminal`.
        # If a future edit renames the ValueError message in websearch_core.py,
        # this test fails loud instead of retry semantics failing silent.
        '"Unsupported content-type:',
        '"Navigation failed:',
        "body rendered empty",
    ],
)
def test_websearch_core_still_emits_classifier_prefixes(literal: str) -> None:
    source = _WEBSEARCH_CORE.read_text(encoding="utf-8")
    assert literal in source, (
        f"{literal!r} no longer appears in websearch_core.py — "
        f"update _classify_and_raise_if_terminal in fetchurl.py to match."
    )
