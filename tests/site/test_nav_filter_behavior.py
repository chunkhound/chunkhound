# ruff: noqa: E501  # Embedded JavaScript keeps browser-like expressions intact.
"""Docs nav filter: typing narrows the guide list, empty sections collapse.

`initNavFilter` is the only client-side path that hides guide links. The filter
input ships on every docs page (the shared DocsNav sidebar) and in the marketing
mobile menu, so its visible contract is user-facing: matching links stay,
non-matching links hide, a section with no visible link collapses, and clearing
restores everything. This drives the real built markup through the shared
happy-dom runtime rather than a hand-built fixture.
"""

from __future__ import annotations

import json

from tests.site.dom_helpers import browser_dom, dist_body
from tests.site.tsx_runner import run_tsx_json

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
const linkByText = (needle) => links.find((link) => link.textContent.includes(needle));
const architecture = linkByText('Architecture');
const configuration = linkByText('Configuration');
const snapshot = () => ({
  architecture: architecture.style.display,
  configuration: configuration.style.display,
  section: sections.map((section) => section.style.display),
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

// A query that matches no guide must not leave a dangling section.
type('no-guide-matches-this-query');
const noMatches = snapshot();

// Clearing the query restores the default view.
type('');
const cleared = snapshot();

console.log(JSON.stringify({ atRest, filtered, noMatches, cleared }));
"""
    )
    result = run_tsx_json(script)

    total = result["atRest"]["totalCount"]
    assert total >= 2, "Need at least two guide links to prove filtering"

    # Resting state: every link and its section are shown.
    assert result["atRest"]["architecture"] == ""
    assert result["atRest"]["configuration"] == ""
    assert result["atRest"]["visibleCount"] == total
    assert result["atRest"]["section"] == [""]

    # A matching query keeps only matching links; the section stays open.
    assert result["filtered"]["architecture"] == ""
    assert result["filtered"]["configuration"] == "none"
    assert result["filtered"]["visibleCount"] == 1
    assert result["filtered"]["section"] == [""]

    # A query matching nothing hides every link and collapses its section.
    assert result["noMatches"]["visibleCount"] == 0
    assert result["noMatches"]["architecture"] == "none"
    assert result["noMatches"]["configuration"] == "none"
    assert result["noMatches"]["section"] == ["none"]

    # Clearing restores every link and its section.
    assert result["cleared"]["architecture"] == ""
    assert result["cleared"]["configuration"] == ""
    assert result["cleared"]["visibleCount"] == total
    assert result["cleared"]["section"] == [""]


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
