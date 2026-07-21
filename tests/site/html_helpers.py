"""Shared HTML parsing helpers for site tests."""

from __future__ import annotations

import re


def attributes(tag: str) -> dict[str, str]:
    """Extract attribute key-value pairs from an HTML tag string."""
    return dict(re.findall(r'([^\s=/>]+)\s*=\s*"([^"]*)"', tag))


def canonical_href(html: str) -> str | None:
    """Return the href of the <link rel="canonical"> tag, or None."""
    for match in re.finditer(r"<link\s+[^>]*>", html):
        attrs = attributes(match.group(0))
        if attrs.get("rel") == "canonical":
            return attrs.get("href")
    return None


def visible_text(document: str) -> str:
    """Strip HTML tags and collapse whitespace to get visible text."""
    return " ".join(re.sub(r"<[^>]+>", " ", document).split())
