"""Behavior tests for site/src/scripts/reduced-motion.ts.

Drives the real module (detectReducedMotion, onReducedMotionChange) through
the shared happy-dom bootstrap — the reduced-motion media query is the one
hand fake browser_dom owns, toggled via globalThis.setReducedMotion. The
window-absent and matchMedia-missing branches run in bare tsx with no DOM.

The CSS drawer gates are contracted against the shipped bundles (css_helpers)
rather than the source, so a rule that never reaches dist fails the test.
"""

from __future__ import annotations

from tests.site.css_helpers import bodies
from tests.site.dom_helpers import browser_dom
from tests.site.tsx_runner import run_tsx_json

_IMPORT = (
    "const { detectReducedMotion, onReducedMotionChange } = "
    "await import('./site/src/scripts/reduced-motion.ts');\n"
)


def test_detect_reduced_motion_is_true_without_window() -> None:
    """SSR/Node has no window — animations must default to off (safe)."""
    rendered = run_tsx_json(
        _IMPORT + "console.log(JSON.stringify({ v: detectReducedMotion() }));"
    )

    assert rendered == {"v": True}


def test_detect_reduced_motion_is_true_without_match_media() -> None:
    """A window without matchMedia cannot honor motion queries — default safe."""
    script = (
        "globalThis.window = {};\n"
        + _IMPORT
        + "console.log(JSON.stringify({ v: detectReducedMotion() }));"
    )
    rendered = run_tsx_json(script)

    assert rendered == {"v": True}


def test_on_change_is_noop_without_window_apis() -> None:
    """Listener registration must not throw where there is nothing to listen to."""
    script = (
        "globalThis.window = {};\n"
        + _IMPORT
        + "onReducedMotionChange(() => { throw new Error('must not fire'); });\n"
        + "console.log(JSON.stringify({ ok: true }));"
    )
    rendered = run_tsx_json(script)

    assert rendered == {"ok": True}


def test_detect_reduced_motion_follows_live_preference(built_site) -> None:
    """detectReducedMotion tracks the shared matchMedia fake through flips."""
    script = (
        browser_dom("<div></div>")
        + _IMPORT
        + """
const before = detectReducedMotion();
globalThis.setReducedMotion(true);
const reduced = detectReducedMotion();
globalThis.setReducedMotion(false);
const back = detectReducedMotion();
console.log(JSON.stringify({ before, reduced, back }));
"""
    )
    rendered = run_tsx_json(script)

    assert rendered == {"before": False, "reduced": True, "back": False}


def test_change_callback_fires_on_every_preference_flip(built_site) -> None:
    """onReducedMotionChange delivers the live matches state per flip."""
    script = (
        browser_dom("<div></div>")
        + _IMPORT
        + """
const seen = [];
onReducedMotionChange((reduced) => seen.push(reduced));
globalThis.setReducedMotion(true);
globalThis.setReducedMotion(false);
globalThis.setReducedMotion(true);
console.log(JSON.stringify({ seen }));
"""
    )
    rendered = run_tsx_json(script)

    assert rendered == {"seen": [True, False, True]}


# The drawers animate via CSS (not the TS module), so their gates are
# contracted against the shipped bundles: a rule that never reaches dist must
# fail, not just the source declaration.
def test_docs_sidebar_drawer_honors_reduced_motion() -> None:
    """The docs drawer must disable its slide under prefers-reduced-motion."""
    still = bodies(".docs-sidebar", media="prefers-reduced-motion")
    assert any("transition:none" in body for body in still), (
        ".docs-sidebar is not zeroed under prefers-reduced-motion"
    )


def test_nav_drawer_honors_reduced_motion() -> None:
    """The nav drawer must disable its slide under prefers-reduced-motion."""
    still = bodies(".nav-menu-panel", media="prefers-reduced-motion")
    assert any("transition:none" in body for body in still), (
        ".nav-menu-panel is not zeroed under prefers-reduced-motion"
    )
