"""Contract tests for the shared URL log redaction helper."""

import pytest

from chunkhound.core.utils.url_redaction import url_for_log


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://host/path", "https://host/path"),
        ("https://user:pw@host/path?token=SECRET#frag", "https://host/path"),
        (
            "http://user:pw@127.0.0.1:8080/config?q=1#f",
            "http://127.0.0.1:8080/config",
        ),
        ("/rerank?api_key=SECRET#frag", "/rerank"),
        ("https://user:pw@host/path?q=a@b", "https://host/path"),
    ],
)
def test_redacts_userinfo_query_and_fragment(url: str, expected: str) -> None:
    redacted = url_for_log(url)
    assert expected in redacted
    assert "user" not in redacted
    assert "pw" not in redacted
    assert "SECRET" not in redacted


@pytest.mark.parametrize(
    "url",
    [
        # A delimiter inside userinfo hides the credentials from the endpoint.
        "https://user:pa?ss@host/path",
        "https://user:p#ss@host/path",
        # Malformed authority cannot distinguish credentials from the path.
        "https://user:bad/secret@host/path",
        "https://user:1234/secret\\?x@host/rerank",
        "https://user:1234/secret ?x@host/rerank",
        # Only the hidden delimiter @ makes the prefix look harmless.
        "https://user:1234/secret?x@host/rerank",
        "https://user:1234/secret#x@host/rerank",
        # urlsplit raises on this input; the log helper must not.
        "http://[::1",
    ],
)
def test_malformed_urls_do_not_raise_and_do_not_leak(url: str) -> None:
    redacted = url_for_log(url)
    assert isinstance(redacted, str)
    assert "user" not in redacted
    assert "secret" not in redacted.lower()


def test_control_characters_are_escaped() -> None:
    redacted = url_for_log("https://host/path\r\n\t\x1b\u0085")
    assert all(char.isprintable() for char in redacted)
    assert not any(char in redacted for char in "\r\n\t\x1b\u0085")
