"""Shared output post-processing for the CLI and MCP websearch paths."""

from __future__ import annotations

import re


def replace_paths_with_urls(text: str, mapping: dict[str, str]) -> str:
    """Substitute on-disk basenames with their source URLs.

    ``(?<!\\w)`` matches at any non-word boundary on the left (path
    separator, whitespace, punctuation, line start), so leaf citations
    like ``/tmp/.../basename.md`` are rewritten. ``(?![\\w/])`` rejects a
    trailing word char (basename is a prefix of a longer name) or ``/``
    (basename is a directory component, not the leaf).
    """
    for filename, url in mapping.items():
        text = re.sub(rf"(?<!\w){re.escape(filename)}(?![\w/])", url, text)
    return text
