from __future__ import annotations

import re
from pathlib import Path

from tests.site.contrast_helpers import (
    GLOBAL_CSS,
    HERO_CSS,
    contrast_ratio,
    extract_block,
    extract_tokens,
    theme_tokens,
)

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "site" / "dist"

def _extract_shiki_dark_token(html: str) -> str:
    """Color of the first rendered shell-comment token (text starts with '#').

    Matched by comment grammar, not by pinned copy, so docs edits cannot break
    the probe; the token's own hex comes from the Shiki theme via --shiki-dark.
    """
    match = re.search(
        r'<span style="[^"]*--shiki-dark:(#[0-9a-fA-F]{6})[^"]*">#[^<]*</span>',
        html,
    )
    assert match, "No rendered Shiki comment token found in the docs page"
    return match.group(1)


def test_link_tokens_meet_aa_contrast_in_light_and_dark_themes() -> None:
    tokens = theme_tokens()
    dark_tokens = tokens["dark"]
    light_tokens = tokens["light"]

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
        assert contrast_ratio(foreground, background) >= 4.5


def _hero_surface_tokens() -> dict[str, dict[str, str]]:
    """Page theme tokens overlaid with the hero scope's local re-points.

    One alias level, like theme_tokens(): a surface re-points the link family to
    code tones, and those aliases must resolve to the hex the browser paints.
    """
    local = extract_tokens(extract_block(HERO_CSS.read_text(encoding="utf-8"), ".hero"))

    scoped_by_theme: dict[str, dict[str, str]] = {}
    for theme, page in theme_tokens().items():
        scoped = dict(page)
        for name, value in local.items():
            scoped[name] = page[value[4:-1]] if value.startswith("var(") else value
        scoped_by_theme[theme] = scoped
    return scoped_by_theme


def test_hero_cta_link_recipe_carries_the_code_surface_highlight() -> None:
    """The hero's main CTA must carry the code surface's highlight, not body text.

    DESIGN_SYSTEM (Code Surface Accent): on --code-bg use --code-accent, never
    the theme-adaptive --primary. The prose-link rules (`a`, `a:visited`,
    `a:hover` — element + pseudo, 0,1,1) outrank the CTA's lone class (0,1,0), so
    the surface — not the control — re-points the family; otherwise a visited
    CTA takes light theme's ink --link-visited (1.55:1 on --code-bg) and the
    page's one action disappears.
    """
    for theme, scoped in _hero_surface_tokens().items():
        highlight = scoped["--code-accent"]
        surface = scoped["--code-bg"]
        for state in ("--link", "--link-hover", "--link-visited"):
            assert scoped[state] == highlight, (
                f"{theme}: {state} must resolve to the code-surface highlight "
                f"(--code-accent), got {scoped[state]}"
            )
            ratio = contrast_ratio(scoped[state], surface)
            assert ratio >= 4.5, f"{theme}: {state} on --code-bg = {ratio:.2f}:1"
        # Non-text contrast: the focus ring must clear the surface behind it.
        ring = contrast_ratio(scoped["--link-focus"], surface)
        assert ring >= 3.0, f"{theme}: --link-focus on --code-bg = {ring:.2f}:1"


def test_astro_code_background_uses_shared_code_surface_token() -> None:
    css = GLOBAL_CSS.read_text(encoding="utf-8")

    default_block = extract_block(css, "pre.astro-code")

    assert "background-color: var(--code-bg) !important;" in default_block
    assert "color: var(--shiki-dark) !important;" in default_block


def test_astro_code_tokens_are_intentionally_pinned_to_dark_shiki_values() -> None:
    css = GLOBAL_CSS.read_text(encoding="utf-8")

    span_block = extract_block(css, "pre.astro-code span")

    assert "color: var(--shiki-dark) !important;" in span_block


def test_rendered_comment_token_meets_aa_contrast_on_shared_code_surfaces() -> None:
    getting_started = (DIST / "docs" / "getting-started" / "index.html").read_text(
        encoding="utf-8"
    )
    tokens = theme_tokens()
    dark_tokens = tokens["dark"]
    light_tokens = tokens["light"]
    comment_token = _extract_shiki_dark_token(getting_started)

    token_pairs = [
        (comment_token, dark_tokens["--code-bg"]),
        (comment_token, light_tokens["--code-bg"]),
    ]

    for foreground, background in token_pairs:
        assert contrast_ratio(foreground, background) >= 4.5


def test_code_surface_text_meets_aa_contrast_in_both_site_themes() -> None:
    tokens = theme_tokens()
    dark_tokens = tokens["dark"]
    light_tokens = tokens["light"]

    token_pairs = [
        (dark_tokens["--code-text"], dark_tokens["--code-bg"]),
        (light_tokens["--code-text"], light_tokens["--code-bg"]),
    ]

    for foreground, background in token_pairs:
        assert contrast_ratio(foreground, background) >= 4.5


def test_code_surface_accent_tokens_meet_aa_contrast_in_both_site_themes() -> None:
    tokens = theme_tokens()
    dark_tokens = tokens["dark"]
    light_tokens = tokens["light"]

    token_pairs = [
        (dark_tokens["--code-accent"], dark_tokens["--code-bg"]),
        (light_tokens["--code-accent"], light_tokens["--code-bg"]),
        (dark_tokens["--code-muted"], dark_tokens["--code-bg"]),
        (light_tokens["--code-muted"], light_tokens["--code-bg"]),
    ]

    for foreground, background in token_pairs:
        assert contrast_ratio(foreground, background) >= 4.5
