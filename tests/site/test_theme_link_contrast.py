from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "site" / "dist"
GLOBAL_CSS = ROOT / "site" / "src" / "styles" / "global.css"


def _extract_block(css: str, selector: str) -> str:
    pattern = rf"{re.escape(selector)}\s*\{{(.*?)\n\}}"
    match = re.search(pattern, css, re.DOTALL)
    assert match, f"Missing CSS block for {selector}"
    return match.group(1)


def _extract_tokens(block: str) -> dict[str, str]:
    return {
        name: value
        for name, value in re.findall(r"(--[\w-]+):\s*(#[0-9a-fA-F]{6})\s*;", block)
    }


def _build_site() -> None:
    subprocess.run(
        ["npm", "run", "build", "--prefix", "site"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def _extract_shiki_dark_token(html: str, rendered_text: str) -> str:
    match = re.search(
        rf'<span style="[^"]*--shiki-dark:(#[0-9a-fA-F]{{6}})[^"]*">'
        rf"{re.escape(rendered_text)}</span>",
        html,
    )
    assert match, f"Missing rendered Shiki token for {rendered_text!r}"
    return match.group(1)


def _srgb_to_linear(channel: float) -> float:
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def _relative_luminance(hex_color: str) -> float:
    channels = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    red, green, blue = (_srgb_to_linear(channel) for channel in channels)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast_ratio(foreground: str, background: str) -> float:
    foreground_luminance = _relative_luminance(foreground)
    background_luminance = _relative_luminance(background)
    lighter = max(foreground_luminance, background_luminance)
    darker = min(foreground_luminance, background_luminance)
    return (lighter + 0.05) / (darker + 0.05)


def test_link_tokens_meet_aa_contrast_in_light_and_dark_themes() -> None:
    css = GLOBAL_CSS.read_text()
    dark_tokens = _extract_tokens(_extract_block(css, ":root"))
    light_tokens = _extract_tokens(_extract_block(css, '[data-theme="light"]'))

    token_pairs = [
        (light_tokens["--link"], light_tokens["--bg-surface"]),
        (light_tokens["--link-hover"], light_tokens["--bg-surface"]),
        (light_tokens["--link-visited"], light_tokens["--bg-surface"]),
        (light_tokens["--link-on-primary-bg"], light_tokens["--primary-bg"]),
        (light_tokens["--link-on-primary-bg-hover"], light_tokens["--primary-bg"]),
        (light_tokens["--link-on-primary-bg-visited"], light_tokens["--primary-bg"]),
        (dark_tokens["--link"], dark_tokens["--bg-surface"]),
        (dark_tokens["--link-hover"], dark_tokens["--bg-surface"]),
        (dark_tokens["--link-visited"], dark_tokens["--bg-surface"]),
        (dark_tokens["--link-on-primary-bg"], dark_tokens["--primary-bg"]),
        (dark_tokens["--link-on-primary-bg-hover"], dark_tokens["--primary-bg"]),
        (dark_tokens["--link-on-primary-bg-visited"], dark_tokens["--primary-bg"]),
    ]

    for foreground, background in token_pairs:
        assert _contrast_ratio(foreground, background) >= 4.5


def test_astro_code_background_uses_shared_code_surface_token() -> None:
    css = GLOBAL_CSS.read_text()

    default_block = _extract_block(css, "pre.astro-code")

    assert "background-color: var(--code-bg) !important;" in default_block
    assert "color: var(--shiki-dark) !important;" in default_block


def test_astro_code_tokens_are_intentionally_pinned_to_dark_shiki_values() -> None:
    css = GLOBAL_CSS.read_text()

    span_block = _extract_block(css, "pre.astro-code span")

    assert "color: var(--shiki-dark) !important;" in span_block


def test_rendered_comment_token_meets_aa_contrast_on_shared_code_surfaces() -> None:
    _build_site()

    css = GLOBAL_CSS.read_text()
    getting_started = (DIST / "docs" / "getting-started" / "index.html").read_text()
    dark_tokens = _extract_tokens(_extract_block(css, ":root"))
    light_tokens = _extract_tokens(_extract_block(css, '[data-theme="light"]'))
    comment_token = _extract_shiki_dark_token(
        getting_started, "# Skip if you already have uv"
    )

    token_pairs = [
        (comment_token, dark_tokens["--code-bg"]),
        (comment_token, light_tokens["--code-bg"]),
    ]

    for foreground, background in token_pairs:
        assert _contrast_ratio(foreground, background) >= 4.5


def test_code_surface_text_meets_aa_contrast_in_both_site_themes() -> None:
    css = GLOBAL_CSS.read_text()
    dark_tokens = _extract_tokens(_extract_block(css, ":root"))
    light_tokens = _extract_tokens(_extract_block(css, '[data-theme="light"]'))

    token_pairs = [
        (dark_tokens["--code-text"], dark_tokens["--code-bg"]),
        (light_tokens["--code-text"], light_tokens["--code-bg"]),
    ]

    for foreground, background in token_pairs:
        assert _contrast_ratio(foreground, background) >= 4.5
