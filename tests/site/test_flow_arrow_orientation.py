"""Flow-arrow orientation contract.

A flow bar states an outcome as one flow, "A → B". When a bar is too narrow to
hold both halves on one line it stacks them, so the flow becomes top-to-bottom
and the connector must point down — a right arrow in a vertical flow points
across its own meaning (the shipped bug: `.cost-result` stacked at ≤600px while
keeping `→`).

happy-dom has no layout, so the guard is structural, not pixel-based: the glyph
comes from the inherited `--flow-glyph` token (rendered by FlowArrow), and every
rule that stacks a flow bar must flip that token to the down arrow in the same
declaration. A flow bar must never hardcode the glyph as text, or it drifts out
of the token's control again.
"""

from __future__ import annotations

from tests.site.css_helpers import rules_containing
from tests.site.dom_helpers import browser_dom, dist_body
from tests.site.tsx_runner import run_tsx_json

# Built CSS may keep the escape (`\2192`) or normalize it to the glyph; accept
# either so the contract survives the minifier.
_RIGHT = ("\u2192", "\\2192")
_DOWN = ("\u2193", "\\2193")

# Flow bars that stack their halves on narrow viewports. Each keys its own
# breakpoint, so the guard is per-selector: if it stacks, it must flip down.
_STACKING_BARS = (".result", ".cost-result")


def _flat(body: str) -> str:
    """Whitespace-insensitive view of a shipped declaration block."""
    return body.replace(" ", "")


def test_flow_arrow_defaults_to_the_horizontal_glyph() -> None:
    """The one default: an unstacked flow reads left-to-right."""
    bodies = [
        rule.body
        for selector, rule in rules_containing(".flow-arrow")
        if selector.endswith(".flow-arrow") or ":before" in selector
    ]
    assert bodies, "FlowArrow must ship its glyph rule"
    assert any(glyph in body for body in bodies for glyph in _RIGHT), (
        f"the default flow glyph must point right: {bodies}"
    )


def test_stacking_flow_bars_flip_the_glyph_down() -> None:
    """If a flow bar stacks its axis, its arrow must follow it down."""
    for selector in _STACKING_BARS:
        stacked = [
            rule.body
            for _, rule in rules_containing(selector)
            if "flex-direction:column" in _flat(rule.body)
        ]
        assert stacked, f"{selector} must declare its stacked axis"
        assert all(
            any(glyph in body for glyph in _DOWN) for body in stacked
        ), f"{selector} stacks its flow but keeps a horizontal arrow: {stacked}"


def test_flow_bars_delegate_the_glyph_to_the_component() -> None:
    """Every flow bar ships exactly one decorative arrow — never a literal
    glyph in text, which would escape the orientation token."""
    script = (
        browser_dom(dist_body("index.html"))
        + """
const bars = [...document.querySelectorAll('.flow-bar')].map((bar) => ({
  arrows: bar.querySelectorAll('.flow-arrow').length,
  text: bar.textContent ?? '',
}));
const arrows = [...document.querySelectorAll('.flow-arrow')].map((el) => ({
  hidden: el.getAttribute('aria-hidden'),
}));
console.log(JSON.stringify({ bars, arrows }));
"""
    )
    result = run_tsx_json(script)

    assert result["bars"], "the homepage must render flow bars"
    assert all(bar["arrows"] >= 1 for bar in result["bars"]), result["bars"]
    glyphs = (*_RIGHT, *_DOWN)
    assert all(
        glyph not in bar["text"] for bar in result["bars"] for glyph in glyphs
    ), "flow glyphs must come from FlowArrow, not literal text"
    assert all(arrow["hidden"] == "true" for arrow in result["arrows"]), result[
        "arrows"
    ]
