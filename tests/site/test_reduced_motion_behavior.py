"""Behavior tests for site/src/scripts/reduced-motion.ts.

Drives the real module (detectReducedMotion, onReducedMotionChange) through
the shared happy-dom bootstrap — the reduced-motion media query is the one
hand fake browser_dom owns, toggled via globalThis.setReducedMotion. The
window-absent and matchMedia-missing branches run in bare tsx with no DOM.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.site.dom_helpers import browser_dom
from tests.site.tsx_runner import run_tsx_json

ROOT = Path(__file__).resolve().parents[2]
DOCS_CSS = ROOT / "site" / "src" / "styles" / "docs.css"

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


# The docs drawer animates via CSS (not the TS module), so it is contracted
# against the source stylesheet: the marketing drawer guards its transition in
# Nav.astro, and the docs drawer must behave identically for the same user
# setting. `\n\}` scopes the match to the media block's own closing brace.
_REDUCED_MOTION_DRAWER = re.compile(
    r"@media\s*\(\s*prefers-reduced-motion:\s*reduce\s*\)\s*\{(?P<body>.*?)\n\}",
    re.DOTALL,
)


def test_docs_sidebar_drawer_honors_reduced_motion() -> None:
    """The docs drawer must disable its slide under prefers-reduced-motion."""
    block = _REDUCED_MOTION_DRAWER.search(DOCS_CSS.read_text(encoding="utf-8"))
    assert block, "docs.css needs a prefers-reduced-motion block"

    body = block.group("body")
    assert ".docs-sidebar" in body
    assert "transition: none" in body
