"""One page-shell width for the nav, docs, and landing sections.

The sticky nav is the user's visual cue for "the page margin". When landing
sections cap themselves narrower than the nav (the historical 1280 vs 1440
split), every homepage section's left edge sits inside the nav's — most
visibly 80px at >=1440px. happy-dom has no layout, so the narrowest guard for
that user-visible alignment is structural: every shell must take its width
from the one shared token, never a literal.

The shell also owns two reachability/accessibility contracts: the docs hub
(/docs/) must stay linked, and a modal drawer must inert every background
region the surface declares.
"""

from __future__ import annotations

import re

from tests.site.css_helpers import bodies
from tests.site.dom_helpers import DIST

SHELLS = (".container", ".nav-inner", ".docs-shell")

_WIDTH_FROM_TOKEN = re.compile(r"max-width:\s*var\(--container-max\)")

# The page shell's two background regions differ per surface (docs: main +
# toc aside; marketing: main + footer), but both drawers must inert exactly
# the regions their page declares — never fewer.
_DOCS_PAGE = "docs/configuration/index.html"
_MARKETING_PAGE = "index.html"
_INERT_ATTR = re.compile(r"data-nav-mobile-inert")
_HEADING = re.compile(r"<h([1-6])\b")


def _max_width_bodies(selector: str) -> list[str]:
    return [body for body in bodies(selector, media="") if "max-width" in body]


def test_page_shells_share_one_width_token() -> None:
    for selector in SHELLS:
        declarations = _max_width_bodies(selector)
        assert len(declarations) == 1, (
            f"{selector} must declare its shell width exactly once; "
            f"found {len(declarations)}: {declarations}"
        )
        assert _WIDTH_FROM_TOKEN.search(declarations[0]), (
            f"{selector} must derive its width from --container-max, not a "
            f"literal: {declarations[0]}"
        )


def _inert_target_count(page: str) -> int:
    return len(_INERT_ATTR.findall((DIST / page).read_text(encoding="utf-8")))


def test_docs_hub_guide_titles_keep_a_valid_heading_outline() -> None:
    """Card titles are h2 under the page h1. An h3 skips a level and breaks
    screen-reader heading navigation."""
    document = (DIST / "docs" / "index.html").read_text(encoding="utf-8")
    levels = [int(level) for level in _HEADING.findall(document)]
    assert levels and levels[0] == 1, "docs hub must open with an h1"
    for previous, current in zip(levels, levels[1:]):
        assert current <= previous + 1, (
            f"docs hub heading level jumps from h{previous} to h{current}"
        )


def test_docs_hub_is_reachable_via_inbound_link() -> None:
    """/docs/ must be a real destination, never an orphaned page: a guide's
    breadcrumb links back to it."""
    assert (DIST / "docs" / "index.html").is_file(), "docs hub page was not built"
    assert 'href="/docs/"' in (DIST / _DOCS_PAGE).read_text(encoding="utf-8")


def test_drawer_inerts_the_same_page_shell_on_both_surfaces() -> None:
    """A missing background region lets focus escape behind the modal menu,
    so the marketing drawer must mark as many shell regions as the docs one."""
    assert _inert_target_count(_MARKETING_PAGE) == _inert_target_count(_DOCS_PAGE)
