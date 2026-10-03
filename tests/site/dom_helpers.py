"""Shared happy-dom runtime for behavior tests.

Loads real built markup from site/dist into a happy-dom Window and registers
hand fakes ONLY for the browser APIs happy-dom cannot honor: the
viewport-driven IntersectionObserver, the layout-driven rAF/clock pair, and
the reduced-motion media query. Everything else (queries, events, storage,
class lists) is the real DOM implementation.
"""

from __future__ import annotations

import json
import re

from tests.site.tsx_runner import ROOT

DIST = ROOT / "site" / "dist"
_HAPPY_DOM_URI = (
    ROOT / "site" / "node_modules" / "happy-dom" / "lib" / "index.js"
).as_uri()

_FAKES = """
// Hand fake: happy-dom has no viewport, so intersection callbacks are test-driven.
class FakeIntersectionObserver {
  constructor(callback, options) {
    this.callback = callback;
    this.options = options;
    this.targets = [];
    FakeIntersectionObserver.instances.push(this);
  }
  observe(target) { if (!this.targets.includes(target)) this.targets.push(target); }
  unobserve(target) { this.targets = this.targets.filter((t) => t !== target); }
  disconnect() { this.targets = []; }
  emit(target, isIntersecting) { this.callback([{ target, isIntersecting }], this); }
}
FakeIntersectionObserver.instances = [];
window.IntersectionObserver = FakeIntersectionObserver;
globalThis.IntersectionObserver = FakeIntersectionObserver;

// Hand fakes: deterministic rAF queue + clock so animation progress is exact.
// Handles come from a Map so cancelAnimationFrame dequeues precisely; a Map
// also preserves insertion order for exact FIFO flush semantics.
let clock = 0;
const rafQueue = new Map();
let rafHandle = 0;
const flushRaf = () => {
  const pending = [...rafQueue.values()];
  rafQueue.clear();
  for (const fn of pending) fn(clock);
};
const raf = (fn) => { rafHandle += 1; rafQueue.set(rafHandle, fn); return rafHandle; };
const cancelRaf = (handle) => { rafQueue.delete(handle); };
window.requestAnimationFrame = raf;
window.cancelAnimationFrame = cancelRaf;
globalThis.requestAnimationFrame = raf;
globalThis.cancelAnimationFrame = cancelRaf;
globalThis.performance = { now: () => clock };
globalThis.advanceClock = (ms) => { clock += ms; };
globalThis.flushRaf = flushRaf;

// Hand fake: reduced-motion media query with a test toggle. Change listeners
// registered by site scripts fire on every setReducedMotion flip, so tests
// drive live preference changes through this single shared fake.
let reducedMotion = false;
const motionListeners = new Set();
const matchMedia = (query) => ({
  get matches() { return reducedMotion && query.includes("reduce"); },
  media: query,
  addEventListener(type, cb) { if (type === "change") motionListeners.add(cb); },
  removeEventListener(type, cb) { motionListeners.delete(cb); },
});
window.matchMedia = matchMedia;
globalThis.matchMedia = matchMedia;
globalThis.setReducedMotion = (value) => {
  reducedMotion = value;
  for (const cb of [...motionListeners]) cb(value);
};
"""

_BOOTSTRAP = (
    """const { Window } = await import('%%HAPPY_DOM_URI%%');
const window = new Window({ url: 'https://chunkhound.test/' });
globalThis.window = window;
globalThis.Event = window.Event;
globalThis.CustomEvent = window.CustomEvent;
%%DOCUMENT%%
window.document.body.innerHTML = %%BODY%%;
"""
    + _FAKES
)


def hero_noscript(raw: str) -> str:
    """Return the hero transcript's <noscript> body from rendered homepage HTML.

    The marketing nav also ships a <noscript> (its no-JS mobile fallback), so
    the first <noscript> on the page is not necessarily the hero's; scope the
    search to the terminal container instead of relying on document order.
    """
    terminal = re.search(
        r'<div class="terminal-lines" id="terminal-lines"[^>]*>(.*?)'
        r'<div class="terminal-composer"',
        raw,
        flags=re.S,
    )
    if not terminal:
        raise RuntimeError("no terminal-lines container on the homepage")
    noscript = re.search(r"<noscript>(.*?)</noscript>", terminal.group(1), flags=re.S)
    if not noscript:
        raise RuntimeError("terminal container carries no <noscript> transcript")
    return noscript.group(1)


def strip_scripts(html: str) -> str:
    """Remove <script> blocks so only markup remains (tests drive real scripts)."""
    return re.sub(r"<script\b.*?</script\s*>", "", html, flags=re.S | re.I)


def dist_body(page: str) -> str:
    """Return the built page's <body> markup with scripts stripped.

    Tests drive the real site scripts themselves, so the bundled page JS must
    never load — happy-dom would otherwise fetch the astro module bundles.
    """
    html = strip_scripts((DIST / page).read_text(encoding="utf-8"))
    match = re.search(r"<body[^>]*>(.*)</body>", html, flags=re.S | re.I)
    if not match:
        raise RuntimeError(f"No <body> found in built page {DIST / page}")
    return match.group(1)


# Opt-in JS snippet: replaces setTimeout with a FIFO queue that only runs
# when globalThis.flushTimers() is called. NOT enabled by default in the
# bootstrap — other tests rely on real setTimeout semantics (e.g. awaiting
# `new Promise((resolve) => setTimeout(resolve, 0))`).
MANUAL_TIMERS = """
const pendingTimers = [];
const setManualTimer = (callback) => {
  pendingTimers.push(callback);
  return pendingTimers.length;
};
window.setTimeout = setManualTimer;
globalThis.setTimeout = setManualTimer;
globalThis.flushTimers = () => { for (const cb of pendingTimers.splice(0)) cb(); };
"""


def viewport_fake(media: str) -> str:
    """JS snippet installing a test-driven viewport media query.

    happy-dom has no viewport width, so a script that gates behavior on
    ``@media (max-width: ...)`` needs a hand fake. globalThis.setViewportMatches
    flips it and fires change listeners; non-viewport queries (e.g. reduced
    motion) fall through to the shared browser_dom fake.
    """
    return f"""
let viewportMatches = true;
const viewportListeners = new Set();
const viewportQuery = {{
  get matches() {{ return viewportMatches; }},
  media: '{media}',
  addEventListener(type, cb) {{ if (type === 'change') viewportListeners.add(cb); }},
  removeEventListener(type, cb) {{ viewportListeners.delete(cb); }},
}};
const sharedViewportMatchMedia = window.matchMedia.bind(window);
window.matchMedia = (query) => (
  query.includes('max-width') ? viewportQuery : sharedViewportMatchMedia(query)
);
globalThis.setViewportMatches = (value) => {{
  viewportMatches = value;
  for (const cb of [...viewportListeners]) cb({{ matches: value }});
}};
"""


def browser_dom(body_html: str, expose_document: bool = True) -> str:
    """Build the JS bootstrap: happy-dom window preloaded with real dist markup.

    expose_document=False keeps globalThis.document undefined so the site
    scripts' module-level auto-init blocks stay dormant — tests then init
    explicitly against window.document.
    """
    bootstrap = _BOOTSTRAP
    bootstrap = bootstrap.replace("%%HAPPY_DOM_URI%%", _HAPPY_DOM_URI)
    bootstrap = bootstrap.replace(
        "%%DOCUMENT%%",
        "globalThis.document = window.document;" if expose_document else "",
    )
    bootstrap = bootstrap.replace("%%BODY%%", json.dumps(body_html))
    return bootstrap
