"""Behavior tests for the docs mobile nav (site/src/scripts/docs-runtime.ts).

Drives the real module against real built markup (docs/configuration) in a
happy-dom window. The module's own auto-init is the instance under test — no
explicit init call, no element fakes. Two hand fakes cover what happy-dom
cannot honor: the viewport-driven matchMedia (test-driven mobile/desktop
toggle; other queries delegate to the shared browser_dom fake) and nothing
else. happy-dom honors focus, activeElement, class lists, inert, and
getClientRects, so focus trapping is exercised for real.
"""

from __future__ import annotations

from tests.site.dom_helpers import browser_dom, dist_body, viewport_fake
from tests.site.tsx_runner import run_tsx_json

_PAGE = "docs/configuration/index.html"

_VIEWPORT_FAKE = viewport_fake("(max-width: 900px)")

# The module auto-inits on import (readyState is interactive, not loading),
# so the import itself is the setup — tests only drive the resulting drawer.
_IMPORT = """
const doc = window.document;
await import('./site/src/scripts/docs-runtime.ts');
await new Promise((resolve) => setTimeout(resolve, 0));
const toggle = doc.querySelector('[data-nav-toggle]');
const sidebar = doc.getElementById('docs-sidebar');
const scrim = doc.querySelector('[data-docs-nav-scrim]');
const filter = doc.querySelector('[data-docs-nav-filter]');
"""

_HELPERS = """
const activeName = () => {
  const active = doc.activeElement;
  if (active === toggle) return 'toggle';
  if (active === filter) return 'filter';
  if (active === sidebar) return 'sidebar';
  if (active?.tagName === 'A') return active.textContent.trim();
  if (active === doc.body) return 'body';
  return active?.tagName || null;
};
const pressKey = (key, shiftKey) => {
  const event = new window.KeyboardEvent('keydown', {
    key,
    shiftKey,
    bubbles: true,
    cancelable: true,
  });
  const prevented = !doc.dispatchEvent(event);
  return { prevented, active: activeName() };
};
const drawerState = () => ({
  expanded: toggle.getAttribute('aria-expanded'),
  label: toggle.getAttribute('aria-label'),
  role: sidebar.getAttribute('role'),
  ariaModal: sidebar.getAttribute('aria-modal'),
  tabindex: sidebar.getAttribute('tabindex'),
  sidebarHidden: sidebar.getAttribute('aria-hidden'),
  sidebarOpen: sidebar.classList.contains('open'),
  bodyOverflow: doc.body.style.overflow || '',
  active: activeName(),
  inertTargets: [...doc.querySelectorAll('[data-nav-mobile-inert]')].map(
    (target) => target.inert,
  ),
});
"""


def test_mobile_nav_applies_modal_semantics_only_while_open(built_site) -> None:
    script = (
        browser_dom(dist_body(_PAGE))
        + _VIEWPORT_FAKE
        + _IMPORT
        + _HELPERS
        + """
const initial = drawerState();

toggle.click();
const afterOpen = drawerState();

const links = [...sidebar.querySelectorAll('a')];
links[links.length - 1].focus();
const forward = pressKey('Tab', false);

filter.focus();
const backward = pressKey('Tab', true);

doc.dispatchEvent(new window.KeyboardEvent('keydown', {
  key: 'Escape',
  bubbles: true,
  cancelable: true,
}));
const afterEscape = drawerState();

toggle.click();
scrim.click();
const afterScrim = drawerState();

console.log(JSON.stringify({
  initial,
  afterOpen,
  forward,
  backward,
  afterEscape,
  afterScrim,
}));
"""
    )
    rendered = run_tsx_json(script)

    assert rendered["initial"] == {
        "expanded": "false",
        "label": "Open docs menu",
        "role": None,
        "ariaModal": None,
        "tabindex": None,
        "sidebarHidden": "true",
        "sidebarOpen": False,
        "bodyOverflow": "",
        "active": "body",
        "inertTargets": [False, False, False, False, False],
    }
    assert rendered["afterOpen"] == {
        "expanded": "true",
        "label": "Close docs menu",
        "role": "dialog",
        "ariaModal": "true",
        "tabindex": "-1",
        "sidebarHidden": None,
        "sidebarOpen": True,
        "bodyOverflow": "hidden",
        "active": "filter",
        "inertTargets": [True, True, True, True, True],
    }
    assert rendered["forward"] == {"prevented": True, "active": "filter"}
    assert rendered["backward"]["prevented"] is True
    assert rendered["backward"]["active"] == "Contributing"
    assert rendered["afterEscape"] == {
        "expanded": "false",
        "label": "Open docs menu",
        "role": None,
        "ariaModal": None,
        "tabindex": None,
        "sidebarHidden": "true",
        "sidebarOpen": False,
        "bodyOverflow": "",
        "active": "toggle",
        "inertTargets": [False, False, False, False, False],
    }
    assert rendered["afterScrim"]["expanded"] == "false"
    assert rendered["afterScrim"]["active"] == "toggle"
    assert rendered["afterScrim"]["sidebarHidden"] == "true"


def test_mobile_nav_ignores_filtered_links_in_focus_wrap(built_site) -> None:
    script = (
        browser_dom(dist_body(_PAGE))
        + _VIEWPORT_FAKE
        + _IMPORT
        + _HELPERS
        + """
toggle.click();
const links = [...sidebar.querySelectorAll('a')];
const firstLabel = links[0].textContent.trim();
// Simulate the nav filter hiding every link but the first.
links.slice(1).forEach((link) => { link.style.display = 'none'; });

links[0].focus();
const forward = pressKey('Tab', false);

filter.focus();
const backward = pressKey('Tab', true);

console.log(JSON.stringify({ firstLabel, forward, backward }));
"""
    )
    rendered = run_tsx_json(script)

    assert rendered["firstLabel"] == "Getting Started"
    assert rendered["forward"] == {"prevented": True, "active": "filter"}
    assert rendered["backward"] == {
        "prevented": True,
        "active": "Getting Started",
    }


def test_mobile_nav_cleans_up_when_viewport_expands_to_desktop(built_site) -> None:
    script = (
        browser_dom(dist_body(_PAGE))
        + _VIEWPORT_FAKE
        + _IMPORT
        + _HELPERS
        + """
toggle.click();
const opened = drawerState();

globalThis.setViewportMatches(false);
const afterExpand = drawerState();

console.log(JSON.stringify({ opened, afterExpand }));
"""
    )
    rendered = run_tsx_json(script)

    assert rendered["opened"]["expanded"] == "true"
    assert rendered["afterExpand"] == {
        "expanded": "false",
        "label": "Open docs menu",
        "role": None,
        "ariaModal": None,
        "tabindex": None,
        "sidebarHidden": None,
        "sidebarOpen": False,
        "bodyOverflow": "",
        "active": "filter",
        "inertTargets": [False, False, False, False, False],
    }
