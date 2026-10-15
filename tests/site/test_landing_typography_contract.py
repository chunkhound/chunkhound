"""Landing-surface typography contracts.

Two user-visible contracts the homepage drifted on:

1. A landing section intro composes the shared `.intro` shell. A section that
   re-declares its own header typography silently opts out of the page's type
   system, and then drifts from it (ProvenAtScale rendered its testimonial in
   the caption tier while its siblings' prose stayed on the lead tier).
2. Every font size comes from the type scale. A literal size means a component
   invented one nobody specified (`.trust-quote::before` shipped `2.5rem`).
3. Fine print is a labelled notice, not a box: a disclaimer is identified by its
   eyebrow label and separated by spacing (the configurator rail shipped three
   equal-weight bordered slabs beside its real cards).

All three are structural guards: happy-dom has no layout, so the narrowest
reliable check is the markup/rule shape, not a rendered pixel.
"""

from __future__ import annotations

import re

from tests.site.css_helpers import bodies, raw_font_sizes, rules_containing
from tests.site.dom_helpers import browser_dom, dist_body
from tests.site.tsx_runner import run_tsx_json

# Surfaces that carry their own scale by design (DESIGN_SYSTEM: Code Blocks,
# terminal chrome, diagrams, the brand lockup): the root font size, the docs'
# relative-em heading anchors, and the docs/code/terminal selectors.
_NON_MARKETING_SURFACES = (
    "html",
    "code",
    ".astro-code",
    ".platform-code-block",
    ".terminal-",
    ".docs-content",
    ".d-note",
    ".axis-label",
    ".cut-label",
    ".wordmark-brand",
    ".heading-link",
)

# Sections whose heading is not a prose intro: the hero owns the page's single
# h1, and the configurator's heading belongs to its config card, not a section
# intro (it is shared with the docs instance).
_HEADERLESS_SECTIONS = {"configurator"}


def test_landing_section_intros_compose_the_shared_shell() -> None:
    script = (
        browser_dom(dist_body("index.html"))
        + """
const sections = [...document.querySelectorAll('main > section')].map((section) => ({
  id: section.id,
  hasHeading: section.querySelector('h2') !== null,
  headingInIntro: section.querySelector('header.intro h2') !== null,
}));
console.log(JSON.stringify({ sections }));
"""
    )
    sections = run_tsx_json(script)["sections"]

    intro_sections = [
        s for s in sections if s["hasHeading"] and s["id"] not in _HEADERLESS_SECTIONS
    ]
    assert len(intro_sections) >= 4, (
        f"expected the landing prose sections, found {[s['id'] for s in sections]}"
    )
    assert all(s["headingInIntro"] for s in intro_sections), (
        "these sections hand-roll their header instead of composing .intro: "
        f"{[s['id'] for s in intro_sections if not s['headingInIntro']]}"
    )


def test_shipped_font_sizes_come_from_the_type_scale() -> None:
    offenders = [
        f"{selector} {{ font-size: {value} }}"
        for selector, value in raw_font_sizes()
        if not any(surface in selector for surface in _NON_MARKETING_SURFACES)
    ]
    assert not offenders, (
        "Raw font sizes on marketing surfaces — use a --text-* token or add a "
        "tier to the type scale:\n" + "\n".join(offenders)
    )


def test_labelled_notices_ship_a_label() -> None:
    """A fine-print notice is identified by its label, never by a box."""
    script = (
        browser_dom(dist_body("index.html"))
        + """
const notices = [...document.querySelectorAll('.disclaimer')].map((note) => ({
  label: note.querySelector('.eyebrow')?.textContent?.trim() ?? null,
  text: note.querySelector('.disclaimer-text')?.textContent?.trim() ?? null,
}));
console.log(JSON.stringify({ notices }));
"""
    )
    notices = run_tsx_json(script)["notices"]

    assert notices, "the landing surface must ship labelled notices"
    assert all(note["label"] for note in notices), (
        "unlabelled fine print — the label is what makes a notice legible at a "
        f"glance: {notices}"
    )
    assert all(note["text"] for note in notices), (
        f"labelled notice without its fine print: {notices}"
    )


def test_labelled_notices_take_no_border_or_surface() -> None:
    """DESIGN_SYSTEM Border Weight: notices separate with spacing, not borders."""
    boxed = [
        f"{selector} {{ {property}: ... }}"
        for selector, rule in rules_containing(".disclaimer")
        for property in ("border", "background")
        if re.search(rf"(?<![-\w]){property}\s*:", rule.body)
    ]
    assert not boxed, (
        "a labelled notice took a box; section cards must keep the only "
        "borders in view:\n" + "\n".join(boxed)
    )


def test_display_type_tiers_are_fluid_and_zoom_safe() -> None:
    """The display tiers are the scale's only fluid rungs. A fixed px page title
    is what let the mobile hero inherit a desktop size; a pure-vw clamp would
    instead break browser zoom (WCAG 1.4.4). So each tier must be a clamp() whose
    preferred value mixes rem + vw."""
    token_layer = " ".join(bodies(":root", media=""))
    for token in ("--text-4xl", "--text-3xl", "--text-2xl"):
        match = re.search(rf"{token}:\s*([^;]+)", token_layer)
        assert match, f"{token} must be declared on :root"
        value = match.group(1).strip()
        assert value.startswith("clamp("), f"{token} must be a fluid clamp(): {value}"
        assert "vw" in value and "rem" in value, (
            f"{token} preferred value must mix rem + vw for zoom: {value}"
        )


def test_components_do_not_redeclare_heading_type_on_phones() -> None:
    """The phone downshift lives in the token layer; a component-level font-size
    at the phone breakpoint is exactly the drift the fluid tokens replace."""
    for selector in (".headline", ".intro h2", ".section-heading"):
        assert not bodies(selector, media="600px"), (
            f"{selector} re-declares heading type at the phone breakpoint — "
            "the mobile size belongs to the token layer"
        )
