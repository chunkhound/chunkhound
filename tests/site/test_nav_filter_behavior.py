# ruff: noqa: E501  # Embedded JavaScript keeps browser-like expressions intact.
"""Docs navigation: the grouped sidebar, its accessible names, and the filter.

The sidebar split is user-facing: each guide group is one named nav landmark,
the active guide is exposed with aria-current, and the docs hub repeats the
groups. Driving built HTML against site/src/lib/nav.ts keeps the split from
drifting into a second hand-maintained page list. `initNavFilter` then narrows
that list: matching links stay, a group left with no visible link collapses, a
query naming a group keeps the whole group, and clearing restores everything.
"""

from __future__ import annotations

import json
from functools import lru_cache

from tests.site.dom_helpers import browser_dom, dist_body
from tests.site.tsx_runner import ROOT, run_tsx_json

# Any docs page ships the shared sidebar; getting-started is the canonical entry.
_DOCS_PAGE = "docs/getting-started/index.html"

_INIT = """
const doc = window.document;
const { initNavFilter } = await import('./site/src/scripts/nav-filter.ts');
initNavFilter(doc);
"""


def test_nav_filter_hides_non_matching_links_and_restores_on_clear(built_site) -> None:
    script = (
        browser_dom(dist_body(_DOCS_PAGE), expose_document=False)
        + _INIT
        + """
const filter = doc.querySelector('[data-docs-nav-filter]');
const links = [...doc.querySelectorAll('[data-sidebar-link]')];
const sections = [...doc.querySelectorAll('[data-sidebar-section]')];
const sectionTitles = sections.map((section) =>
  section.querySelector('[data-sidebar-section-title]').textContent.trim(),
);
const visibleLinks = (section) =>
  [...section.querySelectorAll('[data-sidebar-link]')].filter(
    (link) => link.style.display !== 'none',
  ).length;
const linkByText = (needle) => links.find((link) => link.textContent.includes(needle));
const architecture = linkByText('Architecture');
const configuration = linkByText('Configuration');
const snapshot = () => ({
  architecture: architecture.style.display,
  configuration: configuration.style.display,
  sections: sections.map((section) => section.style.display),
  visiblePerSection: sections.map(visibleLinks),
  visibleCount: links.filter((link) => link.style.display !== 'none').length,
  totalCount: links.length,
});
const type = (value) => {
  filter.value = value;
  filter.dispatchEvent(new window.Event('input', { bubbles: true }));
};

const atRest = snapshot();

// A query matching exactly one guide's title (case-insensitive substring).
type('architecture');
const filtered = snapshot();

// A query that names a group keeps every link in that group.
type('pi-schematic');
const groupQuery = snapshot();
type('chunkhound');
const otherGroupQuery = snapshot();

// A query that matches no guide or group must not leave a dangling heading.
type('no-guide-matches-this-query');
const noMatches = snapshot();

// Clearing the query restores the default view.
type('');
const cleared = snapshot();

console.log(
  JSON.stringify({ sectionTitles, atRest, filtered, groupQuery, otherGroupQuery, noMatches, cleared }),
);
"""
    )
    result = run_tsx_json(script)

    titles = result["sectionTitles"]
    assert titles == ["ChunkHound", "pi-schematic"], titles
    chunkhound, sister = 0, 1

    total = result["atRest"]["totalCount"]
    per_section = result["atRest"]["visiblePerSection"]
    assert sum(per_section) == total, "Resting view must show every guide"
    assert per_section[chunkhound] > 1 and per_section[sister] > 0, per_section

    # Resting state: every link and both groups are shown.
    assert result["atRest"]["architecture"] == ""
    assert result["atRest"]["configuration"] == ""
    assert result["atRest"]["sections"] == ["", ""]

    # A matching query keeps only matching links and only the group holding them.
    assert result["filtered"]["architecture"] == ""
    assert result["filtered"]["configuration"] == "none"
    assert result["filtered"]["visibleCount"] == 1
    assert result["filtered"]["sections"] == ["", "none"]
    assert result["filtered"]["visiblePerSection"] == [1, 0]

    # A group-name query keeps that whole group and drops the other one. Neither
    # link text names a group, so only the heading match can produce this state.
    assert result["groupQuery"]["visiblePerSection"] == [0, per_section[sister]]
    assert result["groupQuery"]["sections"] == ["none", ""]
    assert result["otherGroupQuery"]["visiblePerSection"] == [
        per_section[chunkhound],
        0,
    ]
    assert result["otherGroupQuery"]["sections"] == ["", "none"]

    # A query matching nothing hides every link and collapses both groups.
    assert result["noMatches"]["visibleCount"] == 0
    assert result["noMatches"]["architecture"] == "none"
    assert result["noMatches"]["configuration"] == "none"
    assert result["noMatches"]["sections"] == ["none", "none"]

    # Clearing restores every link and both groups.
    assert result["cleared"]["architecture"] == ""
    assert result["cleared"]["configuration"] == ""
    assert result["cleared"]["visibleCount"] == total
    assert result["cleared"]["sections"] == ["", ""]


def test_docs_runtime_uses_the_injected_document_for_filter_and_shortcut(
    built_site,
) -> None:
    """Runtime initialization must not bind injected docs to the global document."""
    script = (
        browser_dom("")
        + f"""
const doc = window.document.implementation.createHTMLDocument('injected');
doc.body.innerHTML = {json.dumps(dist_body(_DOCS_PAGE))};
const {{ initDocsRuntime }} = await import('./site/src/scripts/docs-runtime.ts');
await initDocsRuntime(doc);
const filter = doc.querySelector('[data-docs-nav-filter]');
const links = [...doc.querySelectorAll('[data-sidebar-link]')];
filter.value = 'architecture';
filter.dispatchEvent(new window.Event('input', {{ bubbles: true }}));
const shortcut = new window.KeyboardEvent('keydown', {{
  key: 'k', ctrlKey: true, bubbles: true, cancelable: true,
}});
const prevented = !doc.dispatchEvent(shortcut);
console.log(JSON.stringify({{
  hiddenLinks: links.filter((link) => link.style.display === 'none').length,
  focused: doc.activeElement === filter,
  selected: filter.selectionStart === 0 && filter.selectionEnd === filter.value.length,
  prevented,
}}));
"""
    )
    result = run_tsx_json(script)

    assert result["hiddenLinks"] > 0
    assert result["focused"] is True
    assert result["selected"] is True
    assert result["prevented"] is True


_NAV = """import { DOCS_SECTIONS } from './site/src/lib/nav.ts';
console.log(JSON.stringify({ sections: DOCS_SECTIONS }));"""

_SIDEBAR_READ = """
const doc = window.document;
const readSection = (section) => {
  const nav = section.querySelector('nav');
  const labelId = nav?.getAttribute('aria-labelledby') ?? null;
  const label = labelId ? doc.getElementById(labelId) : null;
  return {
    title: section.querySelector('[data-sidebar-section-title]')?.textContent.trim(),
    labelResolves: label !== null,
    labelText: label?.textContent.trim() ?? null,
    hrefs: [...section.querySelectorAll('[data-sidebar-link]')].map(
      (link) => link.getAttribute('href'),
    ),
  };
};
console.log(JSON.stringify({
  sections: [...doc.querySelectorAll('[data-sidebar-section]')].map(readSection),
  labelledNavs: doc.querySelectorAll('nav[aria-labelledby^="docs-section-"]').length,
  current: [...doc.querySelectorAll('[data-sidebar-link][aria-current]')].map(
    (link) => ({ href: link.getAttribute('href'), value: link.getAttribute('aria-current') }),
  ),
}));
"""

_HUB_READ = """
const doc = window.document;
console.log(JSON.stringify({
  sections: [...doc.querySelectorAll('.docs-hub-section')].map((section) => ({
    title: section.querySelector('h2')?.textContent.trim(),
    cards: [...section.querySelectorAll('.docs-hub-card')].map((card) => ({
      href: card.getAttribute('href'),
      heading: card.querySelector('h3')?.textContent.trim() ?? null,
    })),
  })),
}));
"""


@lru_cache(maxsize=1)
def _nav_sections() -> list[dict]:
    return run_tsx_json(_NAV)["sections"]


def _sidebar(page: str) -> dict:
    return run_tsx_json(browser_dom(dist_body(page)) + _SIDEBAR_READ)


def test_every_docs_nav_icon_is_registered() -> None:
    """A nav icon without a registered custom element renders as empty space.

    Phosphor icons are lazy custom elements: adding `icon` to nav.ts is not
    enough. DocsNav must also import the matching component.
    """

    def component_name(icon: str) -> str:
        return "Ph" + "".join(
            part.capitalize() for part in icon.removeprefix("ph-").split("-")
        )

    icons = {page["icon"] for section in _nav_sections() for page in section["pages"]}
    source = (ROOT / "site" / "src" / "components" / "DocsNav.astro").read_text(
        encoding="utf-8"
    )
    missing = sorted(
        icon
        for icon in icons
        if f"@phosphor-icons/webcomponents/{component_name(icon)}" not in source
    )
    assert not missing, f"DocsNav does not import icons: {missing}"


def test_sidebar_renders_one_named_landmark_per_guide_section(built_site) -> None:
    """Each group is a nav landmark named by its visible title (APG: one name
    per nav landmark), and every guide appears in exactly one group."""
    groups = _nav_sections()
    assert len(groups) >= 2, "The sidebar must distinguish the two products"

    sidebar = _sidebar("docs/getting-started/index.html")

    assert [group["title"] for group in sidebar["sections"]] == [
        group["title"] for group in groups
    ]
    assert [group["hrefs"] for group in sidebar["sections"]] == [
        [page["href"] for page in group["pages"]] for group in groups
    ]

    for built, source in zip(sidebar["sections"], groups, strict=True):
        assert built["labelResolves"], f"{source['title']} nav has no accessible name"
        assert built["labelText"] == source["title"]

    assert sidebar["labelledNavs"] == len(groups), "each group owns one nav landmark"

    hrefs = [href for group in sidebar["sections"] for href in group["hrefs"]]
    assert len(hrefs) == len(set(hrefs)), "a guide must appear in exactly one group"


def test_sidebar_exposes_the_current_page_once(built_site) -> None:
    """The active class alone is invisible to assistive tech; the link must
    publish aria-current=page, and only the current guide may carry it."""
    sidebar = _sidebar("docs/architecture/index.html")

    assert sidebar["current"] == [{"href": "/docs/architecture/", "value": "page"}]


def test_docs_hub_groups_cards_by_guide_section(built_site) -> None:
    """The hub mirrors the sidebar split, and each card title is an h3 under
    the group's h2 so the heading outline matches the grouping."""
    groups = _nav_sections()
    hub = run_tsx_json(browser_dom(dist_body("docs/index.html")) + _HUB_READ)[
        "sections"
    ]

    assert [group["title"] for group in hub] == [group["title"] for group in groups]
    assert [[card["href"] for card in group["cards"]] for group in hub] == [
        [page["href"] for page in group["pages"]] for group in groups
    ]

    for group in hub:
        for card in group["cards"]:
            assert card["heading"], f"hub card {card['href']} has no h3 title"
