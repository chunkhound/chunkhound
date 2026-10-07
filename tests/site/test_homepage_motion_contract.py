# ruff: noqa: E501  # Embedded JavaScript keeps browser-like expressions intact.
"""Built-output contract for the homepage's per-section entrance choreography.

The homepage must ship the reveal hooks (driven by scripts/count-reveal.ts),
while the docs configurator stays fully static — the Configurator only carries
hooks when its `animated` prop is set (homepage only).
"""

import re
from pathlib import Path

from tests.site.dom_helpers import browser_dom, dist_body, strip_scripts
from tests.site.tsx_runner import run_tsx_json

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "site" / "dist"
GLOBAL_CSS = ROOT / "site" / "src" / "styles" / "global.css"

_markup_without_scripts = strip_scripts


def _gutter_breakpoint(css: str) -> int:
    """Max viewport width where .container uses the narrow mobile gutter."""
    match = re.search(
        r"@media\s*\(max-width:\s*(\d+)px\)\s*\{[^{}]*?\.container\s*\{[^}]*--space-4",
        css,
        re.DOTALL,
    )
    assert match, "Mobile .container gutter rule missing from global.css"
    return int(match.group(1))


def _media_blocks(css: str) -> list[tuple[str, str]]:
    """(query, body) for every @media block, brace-matched."""
    blocks: list[tuple[str, str]] = []
    for match in re.finditer(r"@media\s*([^{]+)\{", css):
        depth, cursor = 1, match.end()
        while cursor < len(css) and depth:
            depth += (css[cursor] == "{") - (css[cursor] == "}")
            cursor += 1
        blocks.append((match.group(1).strip(), css[match.end() : cursor - 1]))
    return blocks


def test_horizontal_reveals_are_gated_above_the_mobile_gutter() -> None:
    """A full-bleed tile translated horizontally past the viewport edge widens
    the document, so the side-slide reveal must never run where the container
    gutter is narrower than the offset (the <=600px mobile gutter)."""
    css = GLOBAL_CSS.read_text(encoding="utf-8")
    gutter = _gutter_breakpoint(css)

    total = css.count("translateX")
    guarded = 0
    for query, body in _media_blocks(css):
        if "translateX" not in body:
            continue
        min_width = re.search(r"min-width:\s*(\d+)px", query)
        assert min_width, f"Horizontal reveal lacks a min-width guard: @media {query}"
        assert int(min_width.group(1)) > gutter, (
            f"Horizontal reveal active where the {gutter}px gutter is narrower "
            "than the 24px offset"
        )
        guarded += body.count("translateX")

    assert total, "Horizontal reveal variants missing from global.css"
    assert guarded == total, "Horizontal reveal escaped its media guard"


def test_homepage_ships_per_section_reveal_hooks() -> None:
    homepage = _markup_without_scripts(
        (DIST / "index.html").read_text(encoding="utf-8")
    )
    assert 'data-reveal="left"' in homepage  # ResearchStory repo card
    assert 'data-reveal="right"' in homepage  # ResearchStory web card
    assert 'data-reveal="pop"' in homepage  # ResearchStory merge icon
    assert 'data-reveal="draw"' not in homepage  # Configurator terminal stays static
    assert "data-reveal" in homepage  # default fade-up targets


def test_peer_tile_rows_share_one_entrance() -> None:
    """A 3-up tile row is one peer group, so every tile in a `.card-grid`
    carries the same reveal variant (the default fade-up, as in the reference
    question grid). No row may mix a sideways slide into a group that should
    enter together."""
    script = (
        browser_dom(dist_body("index.html"))
        + """
const grids = [...window.document.querySelectorAll('.card-grid')];
const rows = grids.map((grid) =>
  [...grid.querySelectorAll('[data-reveal]')].map((el) => el.getAttribute('data-reveal'))
);
console.log(JSON.stringify({ rows }));
"""
    )
    rows = run_tsx_json(script)["rows"]
    assert len(rows) >= 3, "the landing tile rows must reuse .card-grid"
    for variants in rows:
        assert len(variants) >= 3, "a .card-grid row must contain its peer tiles"
        assert len(set(variants)) == 1, f"peer row mixes reveal variants: {variants}"


def test_docs_configurator_is_static() -> None:
    getting_started = _markup_without_scripts(
        (DIST / "docs" / "getting-started" / "index.html").read_text(encoding="utf-8")
    )
    assert "data-reveal" not in getting_started
