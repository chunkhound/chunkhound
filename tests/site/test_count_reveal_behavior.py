# ruff: noqa: E501  # Embedded JavaScript keeps browser-like expressions intact.
from __future__ import annotations

from tests.site.dom_helpers import browser_dom, dist_body
from tests.site.tsx_runner import run_tsx_json

# Count-reveal ships on the homepage stats strip.
_HOMEPAGE = "index.html"

# The "180M lines" stat is pinned on purpose: loud-by-design coupling to the
# marketing copy, so a stat bump SHOULD break these tests (same rationale as
# EXPECTED_DEMO in test_hero_terminal_behavior.py).

# Real markup driven via explicit init (no globalThis.document → the module's
# auto-init stays dormant, keeping each scenario's observer set deterministic).
_INIT = """
const doc = window.document;
const { initCountReveal } = await import('./site/src/scripts/count-reveal.ts');
initCountReveal(doc);
"""


def test_count_reveal_entrance_plays_once(built_site) -> None:
    script = (
        browser_dom(dist_body(_HOMEPAGE), expose_document=False)
        + _INIT
        + """
const dt = doc.querySelector('[data-count-to="180"]');
const revealDiv = dt.closest('div[data-reveal]');
const monthsDiv = doc.querySelector('dt .sr-only').closest('div[data-reveal]');
const state = () => ({
  counterText: dt.textContent,
  counterPending: revealDiv.hasAttribute('data-reveal-pending'),
  monthsPending: monthsDiv.hasAttribute('data-reveal-pending'),
  revealIndex: revealDiv.style.getPropertyValue('--i'),
});

const atInit = state();
const observer = FakeIntersectionObserver.instances.find((o) => o.targets.includes(dt));

// First entrance: pending lifts, count runs and lands exactly on the
// server-rendered value.
observer.emit(revealDiv, true);
observer.emit(dt, true);
observer.emit(monthsDiv, true);
const atEnter = state();
const afterEnterTargets = {
  counterTracked: observer.targets.includes(dt),
  monthsTracked: observer.targets.includes(monthsDiv),
};
for (let i = 0; i < 80; i += 1) { advanceClock(25); flushRaf(); }
const afterCount = state();

// Scroll out and back in: the entrance is one-shot. Nothing re-hides and
// the counter never re-runs.
observer.emit(revealDiv, false);
observer.emit(dt, false);
observer.emit(monthsDiv, false);
const afterExit = state();
observer.emit(revealDiv, true);
observer.emit(dt, true);
observer.emit(monthsDiv, true);
advanceClock(100);
flushRaf();
const duringReentry = state();
for (let i = 0; i < 80; i += 1) { advanceClock(25); flushRaf(); }
const afterReentry = state();

console.log(JSON.stringify({
  atInit, atEnter, afterEnterTargets, afterCount, afterExit,
  duringReentry, afterReentry,
  counterObserved: FakeIntersectionObserver.instances.filter((o) => o.targets.includes(dt)).length,
  monthsObserved: FakeIntersectionObserver.instances.filter((o) => o.targets.includes(monthsDiv)).length,
}));
"""
    )
    result = run_tsx_json(script)

    at_init = result["atInit"]
    at_enter = result["atEnter"]
    after_count = result["afterCount"]
    after_exit = result["afterExit"]
    during_reentry = result["duringReentry"]
    after_reentry = result["afterReentry"]

    # Init arms the stagger: both tiles hidden, reveal tile carries its index.
    assert at_init["counterPending"] and at_init["monthsPending"]
    assert at_init["revealIndex"] != ""
    assert at_init["counterText"] == "180M"

    # Entry lifts the pending state, counts up to the exact original text,
    # then stops observing both targets (one-shot).
    assert at_enter["counterPending"] is False
    assert at_enter["monthsPending"] is False
    assert after_count["counterText"] == "180M"
    assert result["afterEnterTargets"] == {
        "counterTracked": False,
        "monthsTracked": False,
    }

    # Exit does NOT re-hide (no re-arm) and re-entry does NOT replay: the
    # intro played once and the final value is stable.
    assert after_exit["counterPending"] is False
    assert after_exit["monthsPending"] is False
    assert after_exit["counterText"] == "180M"
    assert during_reentry["counterPending"] is False
    assert during_reentry["counterText"] == "180M"
    assert after_reentry["counterText"] == "180M"

    # Both targets were observed once and then dropped, so the browser can
    # never fire another intersection for them.
    assert result["counterObserved"] == 0
    assert result["monthsObserved"] == 0


def test_count_reveal_reduced_motion_never_hides_content(built_site) -> None:
    script = (
        browser_dom(dist_body(_HOMEPAGE), expose_document=False)
        + """
setReducedMotion(true);
const doc = window.document;
const { initCountReveal } = await import('./site/src/scripts/count-reveal.ts');
initCountReveal(doc);

console.log(JSON.stringify({
  text: doc.querySelector('[data-count-to="180"]').textContent,
  pendingCount: doc.querySelectorAll('[data-reveal-pending]').length,
  observerCount: FakeIntersectionObserver.instances.length,
}));
"""
    )
    result = run_tsx_json(script)

    # Reduced motion: content stays exactly as server-rendered — no hidden
    # state, no stagger variable, nothing waiting on an observer.
    assert result["text"] == "180M"
    assert result["pendingCount"] == 0
    assert result["observerCount"] == 0


def test_count_reveal_indexes_stagger_per_parent(built_site) -> None:
    script = (
        browser_dom("", expose_document=False)
        + _INIT
        + """
const reveal = () => {
  const el = doc.createElement('div');
  el.setAttribute('data-reveal', '');
  return el;
};
const firstParent = doc.createElement('div');
const a1 = reveal();
const a2 = reveal();
firstParent.append(a1, a2);
const secondParent = doc.createElement('div');
const b1 = reveal();
const b2 = reveal();
const b3 = reveal();
secondParent.append(b1, b2, b3);
doc.body.append(firstParent, secondParent);
initCountReveal(doc);

console.log(JSON.stringify({
  indices: [a1, a2, b1, b2, b3].map((el) => el.style.getPropertyValue('--i')),
}));
"""
    )
    result = run_tsx_json(script)

    # Each parent's reveal cluster restarts at --i 0 — deep elements must not
    # inherit document-global stagger delays.
    assert result["indices"] == ["0", "1", "0", "1", "2"]


def test_count_reveal_uses_adaptive_threshold_for_tall_elements(built_site) -> None:
    script = (
        browser_dom("", expose_document=False)
        + """
window.innerHeight = 1000;
const doc = window.document;
const tall = doc.createElement('div');  // > 0.6 * 1000 viewport
tall.setAttribute('data-reveal', '');
tall.dataset.testId = 'tall';
tall.getBoundingClientRect = () => ({ height: 700 });
const small = doc.createElement('div');
small.setAttribute('data-reveal', '');
small.dataset.testId = 'small';
small.getBoundingClientRect = () => ({ height: 300 });
doc.body.append(tall, small);
const { initCountReveal } = await import('./site/src/scripts/count-reveal.ts');
initCountReveal(doc);

console.log(JSON.stringify({
  groups: FakeIntersectionObserver.instances.map((observer) => ({
    threshold: observer.options.threshold,
    targets: observer.targets.map((el) => el.dataset.testId),
  })),
}));
"""
    )
    result = run_tsx_json(script)

    # User-facing contract: tall sections trigger earlier (lower threshold)
    # than short ones, each on its own observer. Exact values are tuning,
    # not contract.
    groups = result["groups"]
    assert len(groups) == 2
    threshold_for = {
        target: group["threshold"] for group in groups for target in group["targets"]
    }
    assert set(threshold_for) == {"tall", "small"}
    assert threshold_for["tall"] < threshold_for["small"]


def test_count_reveal_resize_recompute_rearms_threshold_groups(built_site) -> None:
    script = (
        browser_dom("", expose_document=False)
        + """
window.innerHeight = 1000;
const doc = window.document;
const tall = doc.createElement('div');
tall.setAttribute('data-reveal', '');
tall.dataset.testId = 'tall';
tall.getBoundingClientRect = () => ({ height: 700 });
const small = doc.createElement('div');
small.setAttribute('data-reveal', '');
small.dataset.testId = 'small';
small.getBoundingClientRect = () => ({ height: 300 });
doc.body.append(tall, small);
const { initCountReveal } = await import('./site/src/scripts/count-reveal.ts');
initCountReveal(doc);
const snapshot = () => FakeIntersectionObserver.instances.map((observer) => ({
  threshold: observer.options.threshold,
  targets: observer.targets.map((el) => el.dataset.testId),
}));
const before = snapshot();

// Collapse the viewport so the short element crosses the tall boundary, then
// run the debounced recompute the script schedules on window resize.
window.innerHeight = 400;
// Run-then-swap semantics (not MANUAL_TIMERS): the debounced recompute
// must execute synchronously to test the resize handler in isolation.
window.setTimeout = (fn) => { fn(); return 0; };
window.clearTimeout = () => {};
window.dispatchEvent(new window.Event('resize'));

console.log(JSON.stringify({ before, after: snapshot() }));
"""
    )
    result = run_tsx_json(script)

    def group_watching(groups: list[dict], target: str) -> dict:
        matches = [g for g in groups if target in g["targets"]]
        assert len(matches) == 1, f"{target} expected on exactly one observer"
        return matches[0]

    before_groups = result["before"]
    after_groups = result["after"]
    # Init: tall element on the lower-threshold observer, short on the default.
    before_tall = group_watching(before_groups, "tall")
    before_small = group_watching(before_groups, "small")
    assert before_tall["threshold"] < before_small["threshold"]
    # Recompute moves the short element onto the tall observer and empties the
    # default one — re-observing re-arms the correct observer.
    after_tall = group_watching(after_groups, "tall")
    assert after_tall["threshold"] == before_tall["threshold"]
    assert set(after_tall["targets"]) == {"tall", "small"}
    assert group_watching(after_groups, "small") is after_tall


def test_count_reveal_resize_creates_missing_threshold_observer(built_site) -> None:
    """A resize that crosses the tall/short boundary can land on a threshold
    the init never saw. The element must be re-armed on a freshly created
    observer, not left unobserved (which strands it pending/invisible)."""
    script = (
        browser_dom("", expose_document=False)
        + """
window.innerHeight = 1000;
const doc = window.document;
const el = doc.createElement('div');
el.setAttribute('data-reveal', '');
el.dataset.testId = 'el';
el.getBoundingClientRect = () => ({ height: 300 });
doc.body.append(el);
const { initCountReveal } = await import('./site/src/scripts/count-reveal.ts');
initCountReveal(doc);

// Init sees a short element, so it creates only the default observer.
const atInit = {
  observerCount: FakeIntersectionObserver.instances.length,
  pending: el.hasAttribute('data-reveal-pending'),
};

// Collapse the viewport so the element becomes tall: that threshold had no
// observer at init. Recompute must create it instead of leaving the element
// unobserved.
window.innerHeight = 400;
window.setTimeout = (fn) => { fn(); return 0; };
window.clearTimeout = () => {};
window.dispatchEvent(new window.Event('resize'));
const watchedAfterResize = FakeIntersectionObserver.instances.filter(
  (o) => o.targets.includes(el),
).length;

// Entering must lift the pending state.
const observer = FakeIntersectionObserver.instances.find((o) => o.targets.includes(el));
if (observer) observer.emit(el, true);

console.log(JSON.stringify({
  atInit,
  watchedAfterResize,
  observerCountAfter: FakeIntersectionObserver.instances.length,
  pendingAfterEnter: el.hasAttribute('data-reveal-pending'),
}));
"""
    )
    result = run_tsx_json(script)

    assert result["atInit"] == {"observerCount": 1, "pending": True}
    # The new threshold gets its own observer; the element is watched again.
    assert result["observerCountAfter"] == 2
    assert result["watchedAfterResize"] == 1
    assert result["pendingAfterEnter"] is False


def test_count_reveal_reduced_motion_flip_settles_once(built_site) -> None:
    script = (
        browser_dom(dist_body(_HOMEPAGE), expose_document=False)
        + _INIT
        + """
const dt = doc.querySelector('[data-count-to="180"]');
const revealDiv = dt.closest('div[data-reveal]');
const monthsDiv = doc.querySelector('dt .sr-only').closest('div[data-reveal]');
const state = () => ({
  counterText: dt.textContent,
  revealPending: revealDiv.hasAttribute('data-reveal-pending'),
  monthsPending: monthsDiv.hasAttribute('data-reveal-pending'),
});

const observer = FakeIntersectionObserver.instances.find((o) => o.targets.includes(dt));
observer.emit(revealDiv, true);
observer.emit(dt, true);
observer.emit(monthsDiv, true);
advanceClock(300); globalThis.flushRaf();
const midCount = state();

globalThis.setReducedMotion(true);
const afterFlip = state();
advanceClock(200); globalThis.flushRaf();
const afterSettleFlush = state();

// While reduced: scrolling out must NOT re-hide (no re-arm), and re-entry
// must stay settled — no animation, final value unchanged.
observer.emit(revealDiv, false);
observer.emit(dt, false);
observer.emit(monthsDiv, false);
const afterExitReduced = state();
observer.emit(revealDiv, true);
observer.emit(dt, true);
observer.emit(monthsDiv, true);
advanceClock(100); globalThis.flushRaf();
const afterReenterReduced = state();

globalThis.setReducedMotion(false);
const afterFlipBack = state();

observer.emit(revealDiv, false);
observer.emit(dt, false);
observer.emit(monthsDiv, false);
const afterExit = state();
observer.emit(revealDiv, true);
observer.emit(dt, true);
observer.emit(monthsDiv, true);
advanceClock(100); globalThis.flushRaf();
const duringReplay = state();
advanceClock(1400); globalThis.flushRaf();
const afterReplay = state();

console.log(JSON.stringify({
  midCount, afterFlip, afterSettleFlush, afterExitReduced, afterReenterReduced,
  afterFlipBack,
  afterExit, duringReplay, afterReplay,
}));
"""
    )
    result = run_tsx_json(script)

    mid_count = result["midCount"]
    after_flip = result["afterFlip"]
    after_settle_flush = result["afterSettleFlush"]
    after_exit_reduced = result["afterExitReduced"]
    after_reenter_reduced = result["afterReenterReduced"]
    after_flip_back = result["afterFlipBack"]
    after_exit = result["afterExit"]
    during_replay = result["duringReplay"]
    after_replay = result["afterReplay"]

    # Mid-count: counter is running (pending lifted, value in flight).
    assert mid_count["counterText"] != "180M"
    assert not mid_count["revealPending"] and not mid_count["monthsPending"]

    # Flip to reduced: pending animations skip to the server-rendered final
    # state and stay there — the cancelled count never resumes.
    assert after_flip["counterText"] == "180M"
    assert not after_flip["revealPending"] and not after_flip["monthsPending"]
    assert after_settle_flush == after_flip

    # While reduced: observers stay attached but scroll out/in must be inert —
    # content never re-hides and counts never re-animate.
    assert after_exit_reduced == after_flip
    assert after_reenter_reduced == after_flip

    # Flip back: already-settled elements are NOT re-animated.
    assert after_flip_back == after_flip

    # Once content is revealed, a live reduced-motion flip and a subsequent
    # scroll out/in must all be inert: nothing re-hides, nothing replays.
    assert after_exit == after_flip
    assert during_replay == after_flip
    assert after_replay == after_flip


def test_count_reveal_reduced_flip_settles_unseen_content_permanently(built_site) -> None:
    """Turning reduced-motion on settles content for good: a section the user
    had not scrolled to yet must not animate after the preference flips back."""
    script = (
        browser_dom(dist_body(_HOMEPAGE), expose_document=False)
        + _INIT
        + """
const dt = doc.querySelector('[data-count-to="180"]');
const revealDiv = dt.closest('div[data-reveal]');
const observer = FakeIntersectionObserver.instances.find((o) => o.targets.includes(dt));

// Never entered: still hidden, count not started.
const beforeFlip = {
  pending: revealDiv.hasAttribute('data-reveal-pending'),
  text: dt.textContent,
};

// Reduced on, then off, before this section ever enters the viewport.
globalThis.setReducedMotion(true);
globalThis.setReducedMotion(false);

// Now it enters: once-only semantics make the settled state final.
observer.emit(revealDiv, true);
observer.emit(dt, true);
advanceClock(300); globalThis.flushRaf();
const afterEnter = {
  pending: revealDiv.hasAttribute('data-reveal-pending'),
  text: dt.textContent,
};

console.log(JSON.stringify({ beforeFlip, afterEnter }));
"""
    )
    result = run_tsx_json(script)

    assert result["beforeFlip"] == {"pending": True, "text": "180M"}
    # Settled by the reduced flip: entry is inert — no hidden state, and the
    # count does not restart (a mid-flight value would read e.g. "93M").
    assert result["afterEnter"] == {"pending": False, "text": "180M"}


def test_count_reveal_dormant_when_reduced_then_observes_on_flip(built_site) -> None:
    script = (
        browser_dom(dist_body(_HOMEPAGE), expose_document=False)
        + """
globalThis.setReducedMotion(true);
"""
        + _INIT
        + """
const dt = doc.querySelector('[data-count-to="180"]');
const revealDiv = dt.closest('div[data-reveal]');

const dormant = {
  observerCount: FakeIntersectionObserver.instances.length,
  pendingCount: doc.querySelectorAll('[data-reveal-pending]').length,
  text: dt.textContent,
};

globalThis.setReducedMotion(false);
const started = {
  observerCount: FakeIntersectionObserver.instances.length,
  pendingCount: doc.querySelectorAll('[data-reveal-pending]').length,
};

const observer = FakeIntersectionObserver.instances.find((o) => o.targets.includes(dt));
observer.emit(revealDiv, true);
observer.emit(dt, true);
advanceClock(1400); globalThis.flushRaf();
const afterCount = {
  text: dt.textContent,
  revealPending: revealDiv.hasAttribute('data-reveal-pending'),
};

console.log(JSON.stringify({ dormant, started, afterCount }));
"""
    )
    result = run_tsx_json(script)

    # Reduced init: fully dormant, content stays server-rendered.
    assert result["dormant"]["observerCount"] == 0
    assert result["dormant"]["pendingCount"] == 0
    assert result["dormant"]["text"] == "180M"

    # Flip back to animated: observation starts fresh.
    assert result["started"]["observerCount"] > 0
    assert result["started"]["pendingCount"] > 0

    # Entry animates and lands on the exact server-rendered value.
    assert result["afterCount"]["text"] == "180M"
    assert result["afterCount"]["revealPending"] is False
