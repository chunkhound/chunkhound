"""Unit tests for ``_url_to_filename`` in the websearch CLI module.

Guards the filename uniqueness contract: distinct URLs must produce
distinct on-disk filenames, even when the lossy ``[^\\w.-]→_``
substitution or the length cap would otherwise collapse them.
"""

from __future__ import annotations

import re

from chunkhound.api.cli.commands.websearch import _url_to_filename

_HASH_SUFFIX = re.compile(r"_[0-9a-f]{8}$")


def test_deterministic() -> None:
    url = "https://example.com/some/path?q=1"
    assert _url_to_filename(url) == _url_to_filename(url)


def test_substitution_collision_resolved() -> None:
    # Pre-fix both URLs mangled to "a.com_x_y_1".
    a = _url_to_filename("https://a.com/x?y=1")
    b = _url_to_filename("https://a.com/x_y_1")
    assert a != b


def test_truncation_collision_resolved() -> None:
    # Two URLs sharing 200 chars of mangled prefix, differing only in tail.
    long_prefix = "https://a.com/" + "x" * 200
    a = _url_to_filename(long_prefix + "?id=alpha")
    b = _url_to_filename(long_prefix + "?id=beta")
    assert a != b


def test_length_bound_default() -> None:
    long_url = "https://a.com/" + "x" * 500
    assert len(_url_to_filename(long_url)) <= 100


def test_length_bound_custom() -> None:
    long_url = "https://a.com/" + "x" * 500
    assert len(_url_to_filename(long_url, max_length=40)) <= 40


def test_hash_suffix_shape() -> None:
    name = _url_to_filename("https://example.com/")
    assert _HASH_SUFFIX.search(name), name
