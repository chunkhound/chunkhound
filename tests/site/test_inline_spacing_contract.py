"""Built pages must keep the inline-flow whitespace the site depends on.

WHY: Astro 7's default `compressHTML: "jsx"` applies JSX whitespace rules and
strips spaces adjacent to newlines, gluing an inline element to the word beside
it — prose "...for custom" + <code>base_url</code> rendered as "custombase_url",
and the hero's "laptop." ran into its "Build your setup" CTA until it read like a
bulleted list item. The site is hand-written HTML that relies on HTML's own
inline-flow whitespace, so astro.config.mjs sets `compressHTML: true`. Checking
the rendered output means a future config or markup change cannot silently
reintroduce the glue (a config-only assertion would miss new markup, a markup
assertion would miss the config flip).
"""

from __future__ import annotations

import re

from tests.site.dom_helpers import DIST, dist_body

# Tags that flow inline with text and must never touch an adjacent word.
# Excludes <span> (decorative icon glyphs sit flush with their label by design)
# and <sup>/<sub> (those attach to the preceding character).
_INLINE = r"code|strong|em|b|i|a|abbr|cite|q|kbd|samp|time"
_GLUED = re.compile(rf"</(?:{_INLINE})>(?=\w)|(?<=\w)<(?:{_INLINE})\b")


def _pages() -> list[str]:
    return sorted(path.relative_to(DIST).as_posix() for path in DIST.rglob("*.html"))


def test_no_inline_element_is_glued_to_adjacent_text() -> None:
    glued = []
    for page in _pages():
        body = dist_body(page)
        for match in _GLUED.finditer(body):
            snippet = body[max(0, match.start() - 24) : match.end() + 24]
            glued.append(f"{page}: …{snippet}…")
    assert not glued, (
        "Inline whitespace was lost between elements:\n" + "\n".join(glued)
    )


def test_hero_headline_and_setup_cta_are_separated() -> None:
    # The hero's h1 is `display: inline`, so its CTA shares the headline's line
    # box; a tag-level sweep cannot know that, so assert the boundary directly.
    # The separating space must sit inside the h1 so it renders at the display
    # size (a container-sized space beside ~49px type reads as cramped) and
    # collapses when the CTA wraps to its own line.
    body = dist_body("index.html")
    # The CTA sits inside a `.hero-cta` wrapper that holds the decorative
    # hound mark beside the link, so the anchor is no longer a direct sibling
    # of the h1 — the boundary check must cross that wrapper (its opening
    # span, the hound's inline span, and its closing tag) to still pin
    # headline-to-CTA adjacency.
    match = re.search(
        r"<h1[^>]*>([\s\S]*?)</h1>\s*(?:<span[^>]*>\s*)*(?:</span>\s*)*"
        r"<a[^>]*hero-setup-link",
        body,
    )
    assert match, "hero headline and its setup CTA must share one inline flow"
    heading = match.group(1)
    assert heading != heading.rstrip(), (
        "the headline must end with the space that separates it from its CTA, so "
        f"the gap scales with the display type; found {heading[-40:]!r}."
    )
