"""Search shortcuts reveal a usable filter, not a hidden focus target."""

from tests.site.dom_helpers import browser_dom, dist_body, viewport_fake
from tests.site.tsx_runner import run_tsx_json


def test_search_shortcut_opens_mobile_drawer_and_keeps_desktop_nonmodal(built_site):
    script = (
        browser_dom(dist_body("docs/configuration/index.html"))
        + viewport_fake("(max-width: 900px)")
        + """
await import('./site/src/scripts/docs-runtime.ts');
await new Promise((resolve) => setTimeout(resolve, 0));
const doc = window.document;
const filter = doc.querySelector('[data-docs-nav-filter]');
const sidebar = doc.getElementById('docs-sidebar');
const shortcut = (modifier, key) => {
  const event = new window.KeyboardEvent('keydown', {
    [modifier]: true, key, bubbles: true, cancelable: true,
  });
  const handled = !doc.dispatchEvent(event);
  return { handled, focused: doc.activeElement === filter,
    hidden: sidebar.inert || sidebar.getAttribute('aria-hidden') === 'true',
    modal: sidebar.getAttribute('aria-modal') };
};
const mobile = shortcut('ctrlKey', 'k');
doc.dispatchEvent(new window.KeyboardEvent('keydown',
  {key: 'Escape', bubbles: true, cancelable: true}));
const closed = {
  hidden: sidebar.inert || sidebar.getAttribute('aria-hidden') === 'true',
  modal: sidebar.getAttribute('aria-modal'),
  toggleFocused: doc.activeElement === doc.querySelector('[data-nav-toggle]'),
};
setViewportMatches(false);
const desktop = shortcut('metaKey', 'K');
console.log(JSON.stringify({ mobile, closed, desktop }));
"""
    )
    rendered = run_tsx_json(script)
    assert rendered["mobile"] == {
        "handled": True,
        "focused": True,
        "hidden": False,
        "modal": "true",
    }
    # Escape must dismiss the drawer it opened and restore focus to the toggle
    # (mobile-drawer.ts closeDrawer(true)), not leave it modal behind the filter.
    assert rendered["closed"] == {"hidden": True, "modal": None, "toggleFocused": True}
    assert rendered["desktop"] == {
        "handled": True,
        "focused": True,
        "hidden": False,
        "modal": None,
    }
