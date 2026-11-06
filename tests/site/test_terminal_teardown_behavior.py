# ruff: noqa: E501  # Embedded JavaScript keeps browser-like expressions intact.
"""Behavior tests for terminal helper teardown.

A frame owner calls TerminalSession.destroy() before removing the frame. After
that call no wiring may react: control clicks, intersection deliveries, resize
deliveries, and reduced-motion flips must all be inert, and a second destroy()
must not re-run disposers. The test drives the real modules against the built
terminal markup; the ResizeObserver is a hand fake beside the shared
IntersectionObserver fake (happy-dom has neither reliably).
"""

from __future__ import annotations

from tests.site.dom_helpers import browser_dom, dist_body
from tests.site.tsx_runner import run_tsx_json

_HOMEPAGE = "index.html"

_IMPORTS = """
const { createSession } = await import('./site/src/scripts/terminal-session.ts');
const { bootTerminalFrame } = await import('./site/src/scripts/terminal-frame.ts');
"""

_RESIZE_FAKE = """
// Hand fake: layout-driven resize callbacks are test-driven, like the shared
// IntersectionObserver fake. disconnect() flips a flag; emit() still calls the
// callback so a test proves the module itself ignores post-destroy deliveries.
class FakeResizeObserver {
  constructor(callback) {
    this.callback = callback;
    this.targets = [];
    this.disconnected = false;
    FakeResizeObserver.instances.push(this);
  }
  observe(target) { this.targets.push(target); }
  unobserve(target) { this.targets = this.targets.filter((t) => t !== target); }
  disconnect() { this.disconnected = true; this.targets = []; }
  emit() { this.callback([], this); }
}
FakeResizeObserver.instances = [];
window.ResizeObserver = FakeResizeObserver;
globalThis.ResizeObserver = FakeResizeObserver;
"""

_GEOMETRY = """
const doc = window.document;
const runEl = doc.getElementById('terminal-lines');
const viewportEl = doc.getElementById('terminal-viewport');
Object.defineProperty(runEl, 'clientWidth', { value: 320 });
const realRect = window.Element.prototype.getBoundingClientRect;
window.Element.prototype.getBoundingClientRect = function () {
  if (this.classList.contains('terminal-measure')) {
    return { height: 120, top: 0, bottom: 120, left: 0, right: 0, width: 320,
             x: 0, y: 0, toJSON() {} };
  }
  return realRect.call(this);
};
"""


def test_destroyed_session_stops_every_wired_callback(built_site) -> None:
    script = (
        browser_dom(dist_body(_HOMEPAGE), expose_document=False)
        + _IMPORTS
        + _RESIZE_FAKE
        + _GEOMETRY
        + """
const session = createSession(doc);
const card = session.card;
const control = card.querySelector('.terminal-control');

let clicks = 0;
const visibility = [];
let renders = 0;
bootTerminalFrame(session, {
  render: (measure) => { renders += 1; measure.appendChild(doc.createElement('div')); },
  toggle: () => { clicks += 1; },
  onVisibility: (visible) => visibility.push(visible),
});

// Before destroy: every wiring is live.
control.click();
const clicksBefore = clicks;
const intersection = FakeIntersectionObserver.instances.at(-1);
intersection.emit(card, true);
const visibilityBefore = [...visibility];
const resize = FakeResizeObserver.instances.at(-1);
resize.emit();
const rendersBefore = renders;
const heightBefore = viewportEl.style.height;
globalThis.setReducedMotion(true);
const reducedBefore = session.reducedMotion;

// A bfcache pagehide keeps the session; a real unload destroys it.
const pagehide = (persisted) => {
  const event = new window.Event('pagehide');
  Object.defineProperty(event, 'persisted', { value: persisted });
  window.dispatchEvent(event);
};
pagehide(true);
const keptOnBfcache = !session.destroyed;

let disposals = 0;
session.onCleanup(() => { disposals += 1; });
session.destroy();
const resizeDisconnected = resize.disconnected;
const disposalsAfterFirst = disposals;
session.destroy();
const disposalsAfterSecond = disposals;

// After destroy: no wiring reacts, including deliveries that race teardown.
control.click();
intersection.emit(card, false);
resize.emit();
const heightAfterResize = viewportEl.style.height;
globalThis.setReducedMotion(false);

const unloaded = createSession(doc);
bootTerminalFrame(unloaded, { render: () => {}, toggle: () => {} });
pagehide(false);

console.log(JSON.stringify({
  keptOnBfcache,
  destroyedOnUnload: unloaded.destroyed,
  clicksBefore,
  visibilityBefore,
  rendersBefore,
  heightBefore,
  reducedBefore,
  resizeDisconnected,
  clicksAfter: clicks,
  visibilityAfter: visibility,
  rendersAfter: renders,
  heightAfterResize,
  reducedAfter: session.reducedMotion,
  disposalsAfterFirst,
  disposalsAfterSecond,
}));
"""
    )
    rendered = run_tsx_json(script)

    # Live wiring, observed at the external boundary.
    assert rendered["clicksBefore"] == 1
    assert rendered["visibilityBefore"] == [True]
    assert rendered["rendersBefore"] == 2
    assert rendered["heightBefore"] == "120px"
    assert rendered["reducedBefore"] is True
    assert rendered["resizeDisconnected"] is True

    # Teardown: every observer dropped, listeners inert, motion flip ignored.
    assert rendered["clicksAfter"] == 1
    assert rendered["visibilityAfter"] == [True]
    assert rendered["rendersAfter"] == 2
    assert rendered["heightAfterResize"] == "120px"
    assert rendered["reducedAfter"] is True

    # destroy() is idempotent: disposers run once.
    assert rendered["disposalsAfterFirst"] == 1
    assert rendered["disposalsAfterSecond"] == 1

    # The frame owns teardown: bfcache keeps the session, a real unload drops it.
    assert rendered["keptOnBfcache"] is True
    assert rendered["destroyedOnUnload"] is True


_RUN_IMPORTS = """
const { initHeroTerminal } = await import('./site/src/scripts/hero-terminal.ts');
const tick = async () => { for (let i = 0; i < 20; i += 1) await new Promise((r) => setTimeout(r, 0)); };
"""


def test_destroy_during_run_freezes_rendered_markup(built_site) -> None:
    """Teardown mid-typing (sleep #1 is the start gate) freezes the card."""
    script = (
        browser_dom(dist_body(_HOMEPAGE), expose_document=False)
        + _RUN_IMPORTS
        + """
const doc = window.document;
const card = doc.getElementById('terminal-lines').closest('.terminal-card');
const control = doc.querySelector('.terminal-control');
let sleepCalls = 0;
let frozen = null;
const session = initHeroTerminal(doc, {
  charDelay: 0, sendPause: 0, workingPause: 0, answerDelay: 0, startDelay: 0,
  sleep: async () => {
    sleepCalls += 1;
    if (sleepCalls === 2) { session.destroy(); frozen = card.innerHTML; }
  },
});
await tick();
const afterRun = card.innerHTML;
const label = control.textContent.trim();
control.click();
await tick();
console.log(JSON.stringify({ sleepCalls, frozen, afterRun, label, afterClick: card.innerHTML }));
"""
    )
    rendered = run_tsx_json(script)

    # A live partial prompt existed at teardown; it never advances afterwards.
    assert rendered["frozen"] and rendered["sleepCalls"] == 2
    assert rendered["afterRun"] == rendered["frozen"]
    assert rendered["label"] == "Stop"
    assert rendered["afterClick"] == rendered["frozen"]
