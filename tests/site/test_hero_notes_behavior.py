# ruff: noqa: E501  # Embedded JavaScript keeps browser-like expressions intact.
"""The hero's two narrative registers: the run's notes, and the promises.

The hero is one full-bleed terminal. Inside it, the engine's run is the proof and
each engine beat's marketing copy is a NOTE CARD anchored to the receipt it
explains — `.anchor > (.line.call + .line.evidence + aside.note)`. There is no
runbook column beside the terminal any more, so an annotation cannot drift from
the rows it annotates: they share one anchor, the note is DOM-last (so the
visual order is call → receipt → note), and CSS alone places it inline below that
band at every card width (see `test_hero_responsive_tiers.py`). Placement is
never a script's decision, so the no-JS render is the same DOM as the live one.

The promise register — the conditions and the interfaces — is a static rail in
the scale section (see test_scale_terms_behavior.py); it never stands in the
run's way.

Guarded contracts:
- NARRATIVE annotates exactly the engine's three source beats, one note each,
  and every note stays inside its two-line budget;
- each beat's receipt carries exactly one note, and the note's kind label IS the
  source class the receipt cites (they cannot disagree);
- the run raises one note at a time (`data-current` + `aria-current="step"`) and
  clears it as the next beat starts; a settled run raises none;
- the note's condition names the promise its beat proves (markup only: the
  terms rail is static chrome and never tracks the run);
- reduced motion and JS-off show all three notes, unemphasised;
- a note is never aria-hidden and never a landmark.
"""

from __future__ import annotations

import html
import re
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

from tests.site.dom_helpers import (
    DIST,
    browser_dom,
    dist_body,
    hero_noscript,
    strip_scripts,
)
from tests.site.tsx_runner import run_tsx_json

_HOMEPAGE = "index.html"

# The notes annotate the engine's run: one note per fan-out source class. The
# agent's own turns — the question and the verdict — emit beats but take no
# note, so a note can never claim engine work that did not happen.
_EXPECTED_NOTE_BEATS = ["code", "git", "web"]

# Every tagged transcript row, in order. A source beat tags the agent row AND its
# ChunkHound receipt (both highlighted together), so each source beat appears
# twice.
_EXPECTED_ROW_BEATS = [
    "prompt",
    "code",
    "code",
    "git",
    "git",
    "web",
    "web",
    "synthesis",
    "answer",
]

# A note is a two-line object: a label row (ordinal + source class + headline)
# and one line of detail. Both lines share the note's narrowest track — a phone,
# budgeted conservatively at 300px, minus the 3px rail and the 12/16px inline
# padding = 269px. The prose is Inter at --text-sm (13px), so the budget is
# MEASURED in the font's own advances (a proportional face cannot be budgeted in
# `ch`): the headline at weight 600 and the detail at 400 must each fit that
# measure. A longer line wraps, and a three-line note drags the run's rhythm with
# it.
_NOTE_TRACK_FLOOR_PX = 300
_NOTE_CONTENT_PX = _NOTE_TRACK_FLOOR_PX - 3 - 12 - 16
_NOTE_FONT_PX = 13
_INTER_FONT = (
    Path(__file__).resolve().parents[2]
    / "brand"
    / "fonts"
    / "inter-variable.ttf"
)


@lru_cache(maxsize=2)
def _inter_advance_px(weight: int):
    """Measure text in the brand's Inter face at `weight`.

    Pillow rasterizes the brand's vendored Inter variable font — the face the
    hero notes are designed against. It is a metric fixture, NOT the font the
    site serves: the live site loads Inter from Google Fonts, so these advances
    pin the note's width budget against the brand face, not the served bytes.
    """
    font = ImageFont.truetype(str(_INTER_FONT), _NOTE_FONT_PX)
    font.set_variation_by_axes(
        [
            weight if b"weight" in axis["name"].lower() else axis["default"]
            for axis in font.get_variation_axes()
        ]
    )
    return font.getlength

# The note's own vocabulary: the source classes the positioning subheadline
# promises. A note label outside this set has nothing in the run to point at
# (the demo-side guard is test_hero_terminal_behavior.py's
# test_hero_demo_source_types_are_promised_by_subheadline).
_NOTE_KINDS = {"CODE", "GIT HISTORY", "WEB PAGES"}


def _note_script(body: str, pre: str = "") -> str:
    """A real homepage DOM with the demo run at zero pacing, then `body`.

    `pre` runs before the demo starts — observers must already be watching when
    the first beat lands. 400 microtask ticks is more than the run needs once
    every sleep resolves immediately.
    """
    return (
        browser_dom(dist_body(_HOMEPAGE), expose_document=False)
        + """
const doc = window.document;
const container = doc.getElementById('terminal-lines');
"""
        + pre
        + """
const { initHeroTerminal } = await import('./site/src/scripts/hero-terminal.ts');
initHeroTerminal(doc, {
  charDelay: 0,
  sendPause: 0,
  workingPause: 0,
  answerDelay: 0,
  startDelay: 0,
  loop: false,
  sleep: async () => {},
});
for (let i = 0; i < 400; i += 1) {
  await new Promise((resolve) => setTimeout(resolve, 0));
}
"""
        + body
    )



_NOTES_OF_RUN = """
const notesOfRun = () => [...container.querySelectorAll('.anchor')].map((anchor) => ({
  parts: [...anchor.children].map((el) => el.className),
  receipt: anchor.querySelector('.evidence-text')?.textContent,
  beat: anchor.querySelector('.note')?.dataset.note,
  kind: anchor.querySelector('.note-kind')?.textContent,
  headline: anchor.querySelector('.note-headline')?.textContent,
  detail: anchor.querySelector('.note-detail')?.textContent,
  index: anchor.querySelector('.note-index')?.textContent,
  condition: anchor.querySelector('.note')?.dataset.condition,
  indexHidden: anchor.querySelector('.note-index')?.getAttribute('aria-hidden'),
  role: anchor.querySelector('.note')?.getAttribute('role'),
  hidden: anchor.querySelector('.note')?.getAttribute('aria-hidden'),
  current: anchor.querySelector('.note')?.hasAttribute('data-current'),
  ariaCurrent: anchor.querySelector('.note')?.getAttribute('aria-current'),
}));
"""


def _assert_notes_anchored_in_run_order(notes: list[dict]) -> None:
    """The pairing every placement depends on: one anchor per engine beat, the
    call and its receipt and then the note, in run order. Checked before any note
    field so a missing note reads as a lost pairing, not a KeyError."""
    assert [note["parts"] for note in notes] == [
        ["line call", "line evidence", "note"]
    ] * 3
    assert [note["beat"] for note in notes] == _EXPECTED_NOTE_BEATS
    assert [note["index"] for note in notes] == ["01", "02", "03"]
    assert [note["condition"] for note in notes] == [
        "local-first",
        "nothing-leaves",
        "your-models",
    ]


def test_hero_narrative_covers_every_engine_beat() -> None:
    """The notes annotate the engine's run: one step per fan-out source class
    (code, git history, web pages). The agent's turns are not engine steps, so no
    note may manufacture a phase for the question or the verdict. The module also
    owns the surface's two static registers: the interface entry points and the
    promise badges."""
    data = run_tsx_json(
        "import { NARRATIVE } from './site/src/scripts/hero-narrative.ts';\n"
        "import { ENTRY_POINTS, PROMISES } "
        "from './site/src/scripts/conditions.ts';\n"
        "import { DEMO } from './site/src/scripts/hero-transcript.ts';\n"
        "console.log(JSON.stringify({ NARRATIVE, ENTRY_POINTS, PROMISES, DEMO }));\n"
    )
    narrative = data["NARRATIVE"]
    entries = data["ENTRY_POINTS"]
    promises = data["PROMISES"]

    assert [step["beats"][0] for step in narrative] == _EXPECTED_NOTE_BEATS
    covered = {beat for step in narrative for beat in step["beats"]}
    assert covered == set(_EXPECTED_NOTE_BEATS), (
        f"unexpected note beats: {sorted(covered - set(_EXPECTED_NOTE_BEATS))}"
    )
    # One beat per note: the note is *placed* by any of its step's beats but
    # *tagged* with the first, so a multi-beat step would tag the wrong receipt.
    for step in narrative:
        assert len(step["beats"]) == 1, f"note spans beats: {step['beats']}"
        assert step["label"].strip()
        assert step["headline"].strip()
        assert step["detail"].strip()
        assert _inter_advance_px(600)(step["headline"]) <= _NOTE_CONTENT_PX, (
            f"note headline overruns its line: {step['headline']!r}"
        )
        assert _inter_advance_px(400)(step["detail"]) <= _NOTE_CONTENT_PX, (
            f"note detail overruns its line: {step['detail']!r}"
        )
    # One note per fan-out call: the notes can never outrun the demo.
    assert len(narrative) == len(data["DEMO"]["calls"])

    # Conditions: every note names one promise, the set is exactly the three
    # the run demonstrates (free & open-source has no beat — it needs none), so
    # 3 beats against 4 conditions never needs a fourth beat. The strip is
    # static chrome, so this mapping is markup: it is what keeps a note's
    # evidence and the value prop it proves from drifting apart.
    assert [step["condition"] for step in narrative] == [
        "local-first",
        "nothing-leaves",
        "your-models",
    ]
    promise_ids = {promise.get("id") for promise in promises}
    for step in narrative:
        assert step["condition"] in promise_ids, step["condition"]

    # Entry points: the interfaces, a tiny static peer pair. The install pill is
    # the primary action, so these stay two chips.
    assert len(entries) == 2, "the entry-point set must stay a peer pair"
    assert {entry["label"] for entry in entries} == {"CLI", "MCP"}
    for entry in entries:
        assert entry["icon"].strip()

    # Promises: the selling points, concrete and under the clutter ceiling.
    assert 1 <= len(promises) <= 4
    for promise in promises:
        assert promise["icon"].strip() and promise["label"].strip()


def test_hero_notes_are_anchored_to_their_receipts() -> None:
    """Server-rendered markup (what a no-JS visitor and a screen reader get
    first): every engine beat's group is its call and the receipt it cites, then
    exactly one note that annotates them, and the note names the same source
    class."""
    script = (
        browser_dom(dist_body(_HOMEPAGE))
        + """
const doc = window.document;
const container = doc.getElementById('terminal-lines');
const card = doc.querySelector('.terminal-card');
"""
        + _NOTES_OF_RUN
        + """
console.log(JSON.stringify({
  notes: notesOfRun(),
  anchorCount: container.querySelectorAll('.anchor').length,
  noteCount: container.querySelectorAll('aside.note').length,
  // The glyph key shares the seam row with the session dots and is decorative.
  keyItems: [...card.querySelectorAll('.terminal-key-item')].map(
    (li) => li.dataset.key,
  ),
  keyHidden: card.querySelector('.terminal-key')?.getAttribute('aria-hidden'),
  seamOrder: [...card.querySelector('.terminal-seam').children].map(
    (el) => el.className,
  ),
}));
"""
    )
    rendered = run_tsx_json(script)

    # One anchor per engine beat, and inside it: the call, the receipt, the note.
    assert rendered["anchorCount"] == rendered["noteCount"] == 3
    _assert_notes_anchored_in_run_order(rendered["notes"])

    # The note's kind label IS the source class the receipt cites, so the
    # annotation cannot be attached to the wrong kind of evidence. And the note is
    # an ancillary comment: a note role, never a landmark, never hidden — only
    # the decorative ordinal is — and nothing is emphasized before the run starts.
    for note in rendered["notes"]:
        cited = note["receipt"].split(":")[0].strip().upper()
        assert note["kind"] == cited, note
        assert note["kind"] in _NOTE_KINDS, note
        assert note["headline"] and note["detail"]
        assert note["role"] == "note"
        assert note["hidden"] is None, (
            "a note must stay in the accessibility tree"
        )
        assert note["indexHidden"] == "true"
        assert note["current"] is False and note["ariaCurrent"] is None

    # The seam row is one row of session chrome: dots, the decorative glyph key,
    # then the session control. The key carries the two roles whose visible label
    # column the narrow tier collapses — never the user turn (its glyph does).
    assert rendered["seamOrder"] == [
        "terminal-dots",
        "terminal-key",
        "terminal-control",
    ]
    assert rendered["keyItems"] == ["agent", "chunkhound"]
    assert rendered["keyHidden"] == "true"


def test_hero_notes_track_the_run(built_site) -> None:
    """The run raises the note of the beat it is narrating and clears it when the
    next beat starts: one note at a time, and a settled run raises none — the
    verdict is fused from all three, so no annotation wins."""
    script = _note_script(
        _NOTES_OF_RUN
        + """
const card = container.closest('.terminal-card');
const currentRows = [...container.querySelectorAll('.line[data-current]')];
console.log(JSON.stringify({
  notes: notesOfRun(),
  raised: [...container.querySelectorAll('.note[data-current]')].length,
  ariaCurrent: [...card.querySelectorAll('[aria-current]')].map(
    (el) => `${el.className}:${el.getAttribute('aria-current')}`,
  ),
  rowBeats: [...container.querySelectorAll('.line[data-beat]')].map(
    (line) => line.getAttribute('data-beat'),
  ),
  currentRowBeats: currentRows.map((line) => line.getAttribute('data-beat')),
}));
"""
    )
    rendered = run_tsx_json(script)

    # The settled run carries all three notes — the reader can still read the
    # whole runbook after the animation ends — with none emphasized.
    _assert_notes_anchored_in_run_order(rendered["notes"])
    assert [note["current"] for note in rendered["notes"]] == [False] * 3
    assert [note["ariaCurrent"] for note in rendered["notes"]] == [None] * 3
    assert rendered["raised"] == 0
    # Nothing in the hero ever keeps an aria-current after the run settles.
    assert rendered["ariaCurrent"] == []

    # Every run step tagged its row(s) — a source beat tags the agent row and its
    # receipt together, in order; the verdict row is the only current one.
    assert rendered["rowBeats"] == _EXPECTED_ROW_BEATS
    assert rendered["currentRowBeats"] == ["answer"]


def test_hero_source_beats_highlight_receipt_and_note_together(built_site) -> None:
    """A source beat narrates one step: the agent's row, its ChunkHound receipt
    and that receipt's note light together, and the note is the only one raised at
    that moment. Emphasis travels code → git → web and clears as the run settles."""
    script = _note_script(
        """
const stillRaised = container.querySelectorAll('.note[data-current]').length;
observer.disconnect();
console.log(JSON.stringify({ rows, raised, stillRaised }));
""",
        pre="""
const roleOf = (line) => line.querySelector('.role-label')?.textContent;
const rows = [];
const raised = [];
const observer = new window.MutationObserver((records) => {
  for (const record of records) {
    if (record.attributeName !== 'data-current') continue;
    if (record.target.classList.contains('note')) {
      if (record.target.hasAttribute('data-current')) {
        raised.push({
          beat: record.target.dataset.note,
          ariaCurrent: record.target.getAttribute('aria-current'),
          live: container.querySelectorAll('.note[data-current]').length,
        });
      }
      continue;
    }
    if (!record.target.classList.contains('line')
        || !record.target.hasAttribute('data-current')) {
      continue;
    }
    rows.push({ role: roleOf(record.target), beat: record.target.dataset.beat });
  }
});
observer.observe(container, { subtree: true, attributes: true,
  attributeFilter: ['data-current'] });
""",
    )
    rendered = run_tsx_json(script)

    # prompt → (agent + receipt) per fan-out beat → synthesis → the verdict.
    assert rendered["rows"] == [
        {"role": "You", "beat": "prompt"},
        {"role": "Agent", "beat": "code"},
        {"role": "ChunkHound", "beat": "code"},
        {"role": "Agent", "beat": "git"},
        {"role": "ChunkHound", "beat": "git"},
        {"role": "Agent", "beat": "web"},
        {"role": "ChunkHound", "beat": "web"},
        {"role": "Agent", "beat": "synthesis"},
        {"role": "Agent", "beat": "answer"},
    ]
    # Each engine beat raises exactly one note, in run order, and marks it for
    # assistive tech; never two at once; and the last beat clears them all.
    assert [item["beat"] for item in rendered["raised"]] == _EXPECTED_NOTE_BEATS
    assert all(item["ariaCurrent"] == "step" for item in rendered["raised"])
    assert all(item["live"] == 1 for item in rendered["raised"])
    assert rendered["stillRaised"] == 0


def test_hero_notes_reveal_the_whole_run_under_reduced_motion(built_site) -> None:
    """Reduced motion: the static transcript and all three notes at once — no
    emphasis, no sleeps, no spinning."""
    script = (
        browser_dom(dist_body(_HOMEPAGE), expose_document=False)
        + """
// Reduced motion must take the production defaults: the contract is that the
// run never sleeps at all, so there is no fast-paced init to share.
globalThis.setReducedMotion(true);
const doc = window.document;
const container = doc.getElementById('terminal-lines');
let sleepCalls = 0;
const { initHeroTerminal } = await import('./site/src/scripts/hero-terminal.ts');
initHeroTerminal(doc, {
  loop: false,
  sleep: async () => { sleepCalls += 1; },
});

for (let i = 0; i < 5; i += 1) {
  await new Promise((resolve) => setTimeout(resolve, 0));
}
"""
        + _NOTES_OF_RUN
        + """
console.log(JSON.stringify({
  sleepCalls,
  notes: notesOfRun(),
  rows: container.querySelectorAll('.line').length,
}));
"""
    )
    rendered = run_tsx_json(script)

    assert rendered["sleepCalls"] == 0
    assert rendered["rows"] == 9
    _assert_notes_anchored_in_run_order(rendered["notes"])
    assert [note["current"] for note in rendered["notes"]] == [False] * 3
    assert [note["ariaCurrent"] for note in rendered["notes"]] == [None] * 3
    # Same two-line note as the animated path — the copy is never trimmed here.
    assert all(
        note["kind"] and note["headline"] and note["detail"]
        for note in rendered["notes"]
    )


def test_hero_no_js_notes_resolve_by_beat() -> None:
    """The no-JS notes resolve by beat, not positionally: each call's note is
    the beat-derived one, so a DEMO.calls reorder cannot silently desync the
    static render from the live run."""
    raw = html.unescape(strip_scripts((DIST / _HOMEPAGE).read_text(encoding="utf-8")))
    static = hero_noscript(raw)
    beat_notes = run_tsx_json(
        "import { DEMO, callBeat } from './site/src/scripts/hero-transcript.ts';\n"
        "import { noteForBeat } from './site/src/scripts/hero-narrative.ts';\n"
        "console.log(JSON.stringify(DEMO.calls.map((_, i) => "
        "noteForBeat(callBeat(i))?.beats[0])));\n"
    )
    assert re.findall(r'data-note="([^"]+)"', static) == beat_notes


def test_hero_no_js_shows_the_full_narrative() -> None:
    """JS off: the three notes, the promise badges and the interface line are all
    present and readable; the question and the verdict never became notes, and the
    live run container starts empty so the transcript never competes with the
    headline."""
    raw = html.unescape(strip_scripts((DIST / _HOMEPAGE).read_text(encoding="utf-8")))
    static = hero_noscript(raw)

    for label in ("The default, located", "The reason: one caller", "The norm: lower, scoped"):
        assert label in static, f"missing from the no-JS notes: {label!r}"

    # Exactly three notes, and none for the agent's own turns.
    assert static.count('class="note"') == 3
    for beat in _EXPECTED_NOTE_BEATS:
        assert f'data-note="{beat}"' in static
    for beat in ("prompt", "synthesis", "answer"):
        assert f'data-note="{beat}"' not in static, (
            f"a non-engine phase leaked into the notes: {beat}"
        )
    # The note keeps its semantics without JS: a note role, nothing hidden.
    assert static.count('role="note"') == 3
    assert 'class="note" aria-hidden' not in static

    for promise in (
        "Free & open-source",
        "Local-first",
        "Nothing leaves your machine",
        "Your models, your bill",
    ):
        assert promise in raw, f"missing from the no-JS promises: {promise!r}"
    for interface in ("CLI", "MCP", "GitHub"):
        assert interface in raw, f"missing from the no-JS interfaces: {interface!r}"
