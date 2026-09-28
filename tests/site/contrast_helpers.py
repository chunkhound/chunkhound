"""Shared CSS token extraction and WCAG contrast math for site theme tests."""

from __future__ import annotations

import re

from tests.site.tsx_runner import ROOT

GLOBAL_CSS = ROOT / "site" / "src" / "styles" / "global.css"


def extract_block(css: str, selector: str) -> str:
    pattern = rf"{re.escape(selector)}\s*\{{(.*?)\n\}}"
    match = re.search(pattern, css, re.DOTALL)
    assert match, f"Missing CSS block for {selector}"
    return match.group(1)


def extract_tokens(block: str) -> dict[str, str]:
    return {
        name: value
        for name, value in re.findall(
            r"(--[\w-]+):\s*(#[0-9a-fA-F]{6}|var\(--[\w-]+\))\s*;", block
        )
    }


def theme_tokens() -> dict[str, dict[str, str]]:
    """Return resolved hex tokens for the dark (default) and explicit-light themes."""
    css = GLOBAL_CSS.read_text(encoding="utf-8")
    themes = {
        "dark": extract_tokens(extract_block(css, ":root")),
        "light": extract_tokens(extract_block(css, '[data-theme="light"]')),
    }
    # Resolve one level of same-theme var() aliases, e.g. --on-primary-bg.
    # Non-colour aliases (--radius-control -> --radius-md) do not resolve to a
    # colour and must be skipped: this view carries colour tokens only.
    for tokens in themes.values():
        for name, value in list(tokens.items()):
            if value.startswith("var("):
                alias = tokens.get(value[4:-1])
                if alias is None or alias.startswith("var("):
                    tokens.pop(name)
                else:
                    tokens[name] = alias
    return themes


def _srgb_to_linear(channel: float) -> float:
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def relative_luminance(hex_color: str) -> float:
    channels = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    red, green, blue = (_srgb_to_linear(channel) for channel in channels)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast_ratio(foreground: str, background: str) -> float:
    foreground_luminance = relative_luminance(foreground)
    background_luminance = relative_luminance(background)
    lighter = max(foreground_luminance, background_luminance)
    darker = min(foreground_luminance, background_luminance)
    return (lighter + 0.05) / (darker + 0.05)
