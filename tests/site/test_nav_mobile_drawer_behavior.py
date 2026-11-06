# ruff: noqa: E501  # Embedded JavaScript keeps browser-like expressions intact.
"""Behavior tests for the marketing mobile drawer (site/src/scripts/mobile-drawer.ts).

The drawer runtime is shared, but its marketing wiring lives in Nav.astro's
inline script (:96-108) — unlike docs there is no importable entry module
(docs-runtime.ts). So these tests init the real module explicitly against the
real built homepage markup, mirroring that wiring. happy-dom honors focus,
activeElement, class lists, and inert for real; the only hand fake is the
viewport-driven media query (happy-dom has no viewport width).
"""

from __future__ import annotations

from tests.site.dom_helpers import browser_dom, dist_body, viewport_fake
from tests.site.tsx_runner import run_tsx_json

_PAGE = "index.html"

# Mirrors Nav.astro:96-108. A rename of the toggle/panel/scrim hooks or the
# labels makes this init fail loudly instead of silently testing stale markup.
_INIT = """
const doc = window.document;
const { MOBILE_INERT_ATTR, initMobileDrawer } = await import('./site/src/scripts/mobile-drawer.ts');
const toggle = doc.querySelector('[data-nav-toggle]');
const panel = doc.getElementById('nav-menu-panel');
const filter = doc.querySelector('[data-docs-nav-filter]');
const inertTargets = [...doc.querySelectorAll(`[${MOBILE_INERT_ATTR}]`)];
initMobileDrawer({
  toggle,
  panel,
  scrim: doc.querySelector('[data-nav-menu-scrim]'),
  media: window.matchMedia('(max-width: 640px)'),
  inertTargets,
  labels: { open: 'Open main menu', close: 'Close main menu' },
});
"""

_HELPERS = """
const activeName = () => {
  const active = doc.activeElement;
  if (active === toggle) return 'toggle';
  if (active === filter) return 'filter';
  if (active === panel) return 'panel';
  if (active?.tagName === 'A') return active.textContent.trim();
  if (active === doc.body) return 'body';
  return active?.tagName || null;
};
const pressKey = (key, shiftKey) => {
  const event = new window.KeyboardEvent('keydown', {
    key, shiftKey, bubbles: true, cancelable: true,
  });
  return { prevented: !doc.dispatchEvent(event), active: activeName() };
};
const state = () => ({
  expanded: toggle.getAttribute('aria-expanded'),
  label: toggle.getAttribute('aria-label'),
  role: panel.getAttribute('role'),
  ariaModal: panel.getAttribute('aria-modal'),
  tabindex: panel.getAttribute('tabindex'),
  panelHidden: panel.getAttribute('aria-hidden'),
  panelInert: panel.inert,
  panelOpen: panel.classList.contains('open'),
  bodyOverflow: doc.body.style.overflow || '',
  active: activeName(),
  inert: inertTargets.map((t) => [t.inert, t.getAttribute('aria-hidden')]),
});
"""

_OPEN = {
    "expanded": "true",
    "label": "Close main menu",
    "role": "dialog",
    "ariaModal": "true",
    "tabindex": "-1",
    "panelHidden": None,
    "panelInert": False,
    "panelOpen": True,
    "bodyOverflow": "hidden",
    "active": "filter",
    "inert": [[True, "true"]] * 5,
}

_CLOSED = {
    "expanded": "false",
    "label": "Open main menu",
    "role": None,
    "ariaModal": None,
    "tabindex": None,
    "panelHidden": "true",
    "panelInert": True,
    "panelOpen": False,
    "bodyOverflow": "",
    "active": "toggle",
    "inert": [[False, None]] * 5,
}


def test_marketing_drawer_applies_modal_semantics_only_while_open(built_site) -> None:
    script = (
        browser_dom(dist_body(_PAGE))
        + viewport_fake("(max-width: 640px)")
        + _INIT
        + _HELPERS
        + """
const initial = state();

toggle.click();
const afterOpen = state();

doc.dispatchEvent(new window.KeyboardEvent('keydown', {
  key: 'Escape', bubbles: true, cancelable: true,
}));
const afterEscape = state();

toggle.click();
doc.querySelector('[data-nav-menu-scrim]').click();
const afterScrim = state();

console.log(JSON.stringify({ initial, afterOpen, afterEscape, afterScrim }));
"""
    )
    rendered = run_tsx_json(script)

    # Closed: background is live, panel is hidden + inert, no dialog semantics.
    assert rendered["initial"] == {**_CLOSED, "active": "body"}
    # Open: modal semantics on, background inert, first focusable is focused.
    assert rendered["afterOpen"] == _OPEN
    # Escape and scrim both close and hand focus back to the trigger.
    assert rendered["afterEscape"] == _CLOSED
    assert rendered["afterScrim"] == _CLOSED


def test_marketing_drawer_traps_focus_within_the_panel(built_site) -> None:
    script = (
        browser_dom(dist_body(_PAGE))
        + viewport_fake("(max-width: 640px)")
        + _INIT
        + _HELPERS
        + """
toggle.click();
const links = [...panel.querySelectorAll('a')];
// Last focusable + Tab wraps to the first; first + Shift+Tab wraps to the last.
links[links.length - 1].focus();
const forward = pressKey('Tab', false);

filter.focus();
const backward = pressKey('Tab', true);

console.log(JSON.stringify({ forward, backward, last: links[links.length - 1].textContent.trim() }));
"""
    )
    rendered = run_tsx_json(script)

    assert rendered["forward"] == {"prevented": True, "active": "filter"}
    assert rendered["backward"] == {
        "prevented": True,
        "active": rendered["last"],
    }


def test_marketing_drawer_closes_when_viewport_expands_to_desktop(built_site) -> None:
    script = (
        browser_dom(dist_body(_PAGE))
        + viewport_fake("(max-width: 640px)")
        + _INIT
        + _HELPERS
        + """
toggle.click();
const opened = state();

globalThis.setViewportMatches(false);
const afterExpand = state();

console.log(JSON.stringify({ opened, afterExpand }));
"""
    )
    rendered = run_tsx_json(script)

    assert rendered["opened"] == _OPEN
    # Resizing past the breakpoint closes and demotes modal semantics; the
    # panel is no longer inert (it is visible desktop markup) and focus is
    # deliberately not yanked back to the now-hidden trigger.
    assert rendered["afterExpand"] == {
        **_CLOSED,
        "panelHidden": None,
        "panelInert": False,
        "active": "filter",
    }
