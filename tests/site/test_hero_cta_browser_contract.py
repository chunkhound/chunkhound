"""Rendered CTA contrast and keyboard focus, using the homepage's real cascade."""

import re

import pytest

from tests.site.browser_helpers import homepage_probe
from tests.site.contrast_helpers import contrast_ratio


@pytest.fixture(scope="module")
def cta_states(built_site) -> dict:
    return homepage_probe("""
const result = await homepageProbe(['light', 'dark'], ctaSnapshot);
console.log(JSON.stringify(result));
""")


def test_hero_cta_rendered_text_meets_aa(cta_states) -> None:
    for theme, states in cta_states.items():
        for state in ("normal", "hover"):
            colors = states[state]
            ratio = contrast_ratio(colors["color"], colors["background"])
            assert ratio >= 4.5, f"{theme}/{state}: CTA contrast {ratio:.2f}:1"


def test_hero_cta_keyboard_focus_is_visible(cta_states) -> None:
    for theme, states in cta_states.items():
        focus = states["focus"]
        assert focus["focusVisible"], f"{theme}: Tab must show focus on CTA"
        assert focus["outline"] and focus["outlineWidth"] >= 2
        assert focus["outlineOffset"] > 0
        assert focus["outline"] != states["normal"]["outline"]
        ratio = contrast_ratio(focus["outline"], focus["background"])
        assert ratio >= 3, f"{theme}: focus contrast {ratio:.2f}:1"


def test_hero_cta_visited_cascade_uses_contracted_surface_tokens(cta_states) -> None:
    # Browser privacy hides visited text; check matching rules and CTA-local tokens.
    for theme, states in cta_states.items():
        for state in ("normal", "hover"):
            colors = states[state]
            label = f"{theme}/{state}/visited"
            assert colors["visited"], f"{label}: no matching visited color rule"
            for visited in colors["visited"]:
                assert visited["declaration"] in {
                    "var(--link-visited)", "var(--code-accent)"
                }, f"{label}: uncontracted visited override {visited['declaration']}"
                assert re.fullmatch(r"#[0-9a-fA-F]{6}", visited["color"]), (
                    f"{label}: expected opaque sRGB token, got {visited['color']}"
                )
                ratio = contrast_ratio(visited["color"], colors["background"])
                assert ratio >= 4.5, f"{label}: CTA contrast {ratio:.2f}:1"
