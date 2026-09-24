# ruff: noqa: E501  # Embedded JavaScript keeps browser-like expressions intact.
from __future__ import annotations

import html
import re

from tests.site.css_helpers import bodies
from tests.site.dom_helpers import DIST, browser_dom, dist_body, hero_noscript
from tests.site.html_helpers import NUMERIC_LANGUAGE_CLAIM
from tests.site.tsx_runner import ROOT, run_tsx_json

DIRECT_CHUNKHOUND_COMMAND = re.compile(
    r"\bchunkhound\s+(?:research|websearch|fetchurl)\b", re.IGNORECASE
)

# Source types the subheadline promises; every demo source class must be one of them.
SOURCE_TYPE_NOUNS = ("code", "git history", "web pages", "documents")

# The hero demo contract: one question, three fan-out research calls (each with a
# cited return), a synthesis beat, and one distilled verdict. This is the shipped
# homepage copy: the render tests assert every field lands in the row whose job
# it is, and the whole-object pin below makes an unannounced copy edit loud.
EXPECTED_DEMO = {
    "prompt": "Should the request timeout still be 30s?",
    "calls": [
        {
            "call": "Reading where the timeout is set today",
            "sources": [
                {"kind": "code", "cite": "api/http.ts:14 · timeout 30s"},
            ],
        },
        {
            "call": "Tracing why it was raised to 30s",
            "sources": [
                {"kind": "git history", "cite": "9b2f1c4 · 10s→30s (export job)"},
            ],
        },
        {
            "call": "Comparing 30s against what peers ship",
            "sources": [
                {
                    "kind": "web pages",
                    "cite": "Stripe, Google › 10s + per-call override",
                },
            ],
        },
    ],
    "synthesis": "One caller's need, everyone's default.",
    "answer": (
        "The 30s isn't wrong — it's global. "
        "Keep it for the export job; drop it as the default."
    ),
}


def source_text(source: dict) -> str:
    return f"{source['kind']}: {source['cite']}"


# The hero terminal ships on the homepage.
_HOMEPAGE = "index.html"

# Real markup driven via explicit init (no globalThis.document → the module's
# auto-init stays dormant, so only the test's own instance runs).
_INIT = """
const doc = window.document;
const container = doc.getElementById('terminal-lines');
const viewportEl = doc.getElementById('terminal-viewport');
const inputEl = doc.getElementById('terminal-input');
const inputText = doc.getElementById('terminal-input-text');
// Rows are addressed by class, never by depth: a beat's call and receipt share
// an .anchor with the note that explains them, so container.children misses them.
const countRows = (cls) => container.querySelectorAll(`.line.${cls}`).length;
const { initHeroTerminal } = await import('./site/src/scripts/hero-terminal.ts');
"""

_TICK = "await new Promise((resolve) => setTimeout(resolve, 0));\n"

_TRANSCRIPT_STRUCTURE = """
function textOf(element) {
  return [element.textContent, ...Array.from(element.children).map(textOf)].join('');
}

function transcriptStructure(container) {
  // A beat's call and receipt rows live inside its `.anchor` (the note is their
  // sibling), so rows are collected by class rather than as direct children:
  // the run's row sequence is a contract, the nesting depth is not.
  const lines = Array.from(container.querySelectorAll('.line'));
  const prompts = lines
    .filter((line) => line.classList.contains('user-prompt'))
    .map((line) => ({
      lineClass: line.className,
      roleClass: line.children[0].className,
      role: line.children[0].textContent,
      roleVisible: Array.from(line.children[0].childNodes)
        .filter((node) => node.nodeType === 3)
        .map((node) => node.textContent).join(''),
      glyphClass: line.children[1].className,
      glyph: line.children[1].textContent,
      inputClass: line.children[2].className,
      input: line.children[2].textContent,
    }));
  const calls = lines
    .filter((line) => line.classList.contains('call'))
    .map((line) => ({
      lineClass: line.className,
      roleClass: line.children[0].className,
      role: line.children[0].textContent,
      glyphClass: line.children[1].className,
      glyph: line.children[1].textContent,
      callClass: line.children[2].className,
      call: line.children[2].textContent,
    }));
  const evidence = lines
    .filter((line) => line.classList.contains('evidence'))
    .map((line) => ({
      lineClass: line.className,
      roleClass: line.children[0].className,
      role: line.children[0].textContent,
      glyphClass: line.children[1].className,
      glyph: line.children[1].textContent,
      listClass: line.children[2].className,
      items: Array.from(line.children[2].children).map((item) => ({
        itemClass: item.className,
        text: item.textContent,
      })),
    }));
  const answers = lines
    .filter((line) => line.classList.contains('answer'))
    .map((line) => ({
      lineClass: line.className,
      roleClass: line.children[0].className,
      role: line.children[0].textContent,
      glyphClass: line.children[1].className,
      glyph: line.children[1].textContent,
      resultClass: line.children[2].className,
      result: line.children[2].textContent,
    }));
  return { prompts, calls, evidence, answers };
}
"""


def test_hero_terminal_renders_harness_research_workflow(built_site) -> None:
    script = (
        browser_dom(dist_body(_HOMEPAGE), expose_document=False)
        + _INIT
        + """
initHeroTerminal(doc, {
  charDelay: 0,
  sendPause: 0,
  workingPause: 0,
  answerDelay: 0,
  loop: false,
  sleep: async () => {},
});

for (let i = 0; i < 40; i += 1) {
"""
        + _TICK
        + """}

console.log(JSON.stringify({
  transcript: textOf(container),
  rowClasses: Array.from(container.children).map((el) => el.className),
  anchorParts: Array.from(container.children)
    .filter((el) => el.classList.contains('anchor'))
    .map((anchor) => Array.from(anchor.children).map((el) => el.className)),
  workingCount: container.querySelectorAll('.is-working').length,
  inputText: inputText.textContent,
  ...transcriptStructure(container),
}));
"""
        + _TRANSCRIPT_STRUCTURE
    )
    rendered = run_tsx_json(script)
    transcript = rendered["transcript"]
    prompts = rendered["prompts"]
    calls = rendered["calls"]
    evidence = rendered["evidence"]
    answers = rendered["answers"]

    assert [prompt["input"] for prompt in prompts] == [EXPECTED_DEMO["prompt"]]
    assert all(
        prompt["lineClass"] == "line user-prompt"
        and prompt["roleClass"] == "role-label"
        # "You" is present for assistive tech but carries no visible text.
        and prompt["role"] == "You"
        and prompt["roleVisible"] == ""
        and prompt["glyph"] == "❯ "
        and prompt["glyphClass"] == "glyph"
        and prompt["inputClass"] == "prompt-text"
        for prompt in prompts
    )
    # The agent's visible act: three fan-out research calls, then a synthesis call.
    expected_calls = [call["call"] for call in EXPECTED_DEMO["calls"]] + [
        EXPECTED_DEMO["synthesis"]
    ]
    assert [call["call"] for call in calls] == expected_calls
    assert all(
        call["lineClass"] == "line call"
        and call["roleClass"] == "role-label"
        and call["role"] == "Agent"
        and call["glyph"] == "⏺ "
        and call["glyphClass"] == "glyph"
        and call["callClass"] == "call-text"
        for call in calls
    )
    # The main story: ChunkHound's cited return, one line per source class,
    # hanging under the ChunkHound label.
    assert [line["role"] for line in evidence] == ["ChunkHound"] * 3
    assert all(
        line["lineClass"] == "line evidence"
        and line["roleClass"] == "role-label"
        and line["glyph"] == "↳ "
        and line["glyphClass"] == "glyph"
        and line["listClass"] == "evidence-list"
        for line in evidence
    )
    assert [[item["text"] for item in line["items"]] for line in evidence] == [
        [source_text(source) for source in call["sources"]]
        for call in EXPECTED_DEMO["calls"]
    ]
    assert all(
        item["itemClass"] == "evidence-text"
        for line in evidence
        for item in line["items"]
    )
    # The agent's verdict, distilled from the evidence above it.
    assert [answer["role"] for answer in answers] == ["Agent"]
    assert [answer["result"] for answer in answers] == [EXPECTED_DEMO["answer"]]
    assert all(
        answer["lineClass"] == "line answer"
        and answer["roleClass"] == "role-label"
        and answer["glyph"] == "✓ "
        and answer["glyphClass"] == "glyph"
        and answer["resultClass"] == "result"
        for answer in answers
    )
    # One question → three fan-out research calls (each call then its cited
    # return) → a synthesis call → the verdict; the transient working state has
    # settled into the call rows. Each beat travels with its own note, so the
    # note and its call + receipt share one anchor — the note DOM-last.
    assert rendered["rowClasses"] == [
        "line user-prompt",
        "anchor",
        "anchor",
        "anchor",
        "line call",
        "line answer",
    ]
    assert rendered["anchorParts"] == [
        ["line call", "line evidence", "note"]
    ] * 3
    assert rendered["workingCount"] == 0
    # The settled input is cleared — every prompt moved into the scrollback.
    assert rendered["inputText"] == ""
    for call in EXPECTED_DEMO["calls"]:
        for source in call["sources"]:
            assert source["kind"] in transcript.lower()
    assert "Human" not in transcript
    assert "Agent" in transcript and "ChunkHound" in transcript
    assert DIRECT_CHUNKHOUND_COMMAND.search(transcript) is None
    assert "$ " not in transcript
    assert NUMERIC_LANGUAGE_CLAIM.search(transcript) is None


def test_hero_terminal_input_stays_fixed_below_transcript(built_site) -> None:
    """The harness contract: a persistent input field below the scrollback
    viewport; transcript growth never moves or re-creates it, and the
    viewport clips instead of growing the card."""
    script = (
        browser_dom(dist_body(_HOMEPAGE), expose_document=False)
        + _INIT
        + """
const inputBefore = doc.getElementById('terminal-input');
const orderBefore = viewportEl.compareDocumentPosition(inputEl);
// Count the live run's rows: the server-rendered <noscript> still carries the
// static transcript before the first render clears the container, so its rows
// must not read as "already rendered".
const rowCount = () => [...container.querySelectorAll('.line')]
  .filter((el) => !el.closest('noscript')).length;
const linesBefore = rowCount();

initHeroTerminal(doc, {
  charDelay: 0,
  sendPause: 0,
  workingPause: 0,
  answerDelay: 0,
  loop: false,
  sleep: async () => {},
});

for (let i = 0; i < 40; i += 1) {
"""
        + _TICK
        + """}

console.log(JSON.stringify({
  inputExists: inputEl !== null && inputText !== null,
  inputTag: inputEl.tagName,
  inputLabel: inputEl.querySelector('.role-label')?.textContent,
  inputVisibleLabel: Array.from(inputEl.querySelector('.role-label').childNodes)
    .filter((node) => node.nodeType === 3)
    .map((node) => node.textContent).join(''),
  orderBefore,
  orderAfter: viewportEl.compareDocumentPosition(inputEl),
  inputStable: doc.getElementById('terminal-input') === inputBefore,
  inputStillLast: inputEl === inputEl.parentElement.lastElementChild
    || inputEl.nextElementSibling.className === 'terminal-status',
  linesBefore,
  linesAfter: rowCount(),
  viewportWrapsLines: viewportEl.contains(container),
}));
"""
    )
    rendered = run_tsx_json(script)

    assert rendered["inputExists"] is True
    assert rendered["inputTag"] == "DIV"
    # The composer carries the same screen-reader-only "You" role label as the
    # submitted prompt rows, so the live input reads as the user's next turn
    # without a visible label.
    assert rendered["inputLabel"] == "You"
    assert rendered["inputVisibleLabel"] == ""
    # The input follows the viewport in document order, before and after.
    assert rendered["orderBefore"] == 4  # Node.DOCUMENT_POSITION_FOLLOWING
    assert rendered["orderAfter"] == 4
    # The same input node survives the whole run — never re-created or moved.
    assert rendered["inputStable"] is True
    assert rendered["inputStillLast"] is True
    # The transcript grew inside the viewport while the input stayed put.
    assert rendered["linesBefore"] == 0
    assert rendered["linesAfter"] == 9
    assert rendered["viewportWrapsLines"] is True

    # The shipped CSS clips the viewport and rests the transcript on the
    # composer edge (the bottom-entry contract): the first row eats the viewport's
    # free space, so the run grows upward and every append pushes the rest up.
    css = "".join(
        path.read_text(encoding="utf-8")
        for path in (DIST / "_astro").glob("*.css")
    )
    assert re.search(
        r"\.terminal-viewport[^{]*\{[^}]*overflow:\s*hidden", css
    ), "viewport must clip the transcript at the top"
    assert re.search(
        r"\.terminal-lines[^{]*>\s*[^{}]*:first-child[^{}]*"
        r"\{[^}]*margin-top:\s*auto",
        css,
    ), "the transcript must rest on the composer edge (bottom-anchored)"
    assert re.search(
        r"\.terminal-lines[^{]*\{[^}]*transition:\s*transform", css
    ), "the transcript must glide its push rather than jump"


def test_hero_terminal_bottom_entry_push_is_gated_on_motion(built_site) -> None:
    """Bottom-entry push-up: a new row enters in two phases — phase one slides
    the existing transcript up to make room, phase two then fades the row in (the
    entrance's animation is delayed by the push's own length, so the phases never
    overlap). Both phases collapse under reduced motion, where the transcript
    renders complete instead of animating in."""
    css = "".join(
        path.read_text(encoding="utf-8")
        for path in (DIST / "_astro").glob("*.css")
    )

    body = ""
    for match in re.finditer(
        r"@media\s*\(prefers-reduced-motion:reduce\)\s*\{", css
    ):
        depth, cursor = 1, match.end()
        while cursor < len(css) and depth:
            depth += (css[cursor] == "{") - (css[cursor] == "}")
            cursor += 1
        block = css[match.end() : cursor - 1]
        if ".terminal-lines" in block:
            body = block
    assert body, "the hero reduced-motion override must exist"
    assert re.search(
        r"\.terminal-lines[^{}]*\{[^}]*transition:\s*none", body
    ), "reduced motion must freeze the stack glide"

    assert "@keyframes hero-row-in" in css, (
        "a new row needs an entrance fade"
    )
    assert re.search(
        r"\.terminal-lines[^{}]*\[data-enter\][^{}]*\{[^}]*"
        r"animation:\s*hero-row-in[^}]*var\(--hero-push\)",
        css,
    ), (
        "the entrance must attach to appended rows and be delayed by the push, "
        "so the row enters only after the stack has made room"
    )
    assert re.search(
        r'\[data-enter=.?solo.?\][^{}]*\{[^}]*animation-delay:\s*0s', css
    ), "the run's first row must skip the push delay (no room to make)"
    assert re.search(
        r"\[data-enter\][^{}]*\{[^}]*animation:\s*none", body
    ), "reduced motion must freeze the entrance fade"


def test_hero_terminal_prompt_caret_runs_until_submit(built_site) -> None:
    """The caret is live (blinking) from first paint and while the prompt is being
    entered, and only goes dark once the run publishes the prompt as submitted —
    never before. Reduced motion keeps its own blanket animation:none."""
    css = "".join(
        path.read_text(encoding="utf-8")
        for path in (DIST / "_astro").glob("*.css")
    )
    assert re.search(
        r"\.terminal-card[^{]*data-composer=sent[^{]*\.terminal-input[^{]*\.cursor"
        r"[^{]*\{(?=[^}]*animation:\s*none)(?=[^}]*opacity:\s*0)",
        css,
    ), "the submitted composer caret must go dark"
    # Reduced motion still freezes the caret outright.
    assert re.search(
        r"@media[^{]*prefers-reduced-motion[^{]*reduce[^{]*\{"
        r"(?:[^{}]|\{[^{}]*\})*"
        r"\.terminal-input[^{]*\.cursor[^{]*\{[^}]*animation:\s*none",
        css,
    ), "reduced motion must freeze the prompt caret"


def test_hero_terminal_composer_recedes_after_submit(built_site) -> None:
    """The prompt field and its ❯ marker are live from first paint, then fade to
    the muted code shade once the prompt is submitted, so attention moves from the
    composer to the transcript. The run publishes that state as
    `data-composer="sent"`, keyed to the submit beat."""
    css = "".join(
        path.read_text(encoding="utf-8")
        for path in (DIST / "_astro").glob("*.css")
    )
    # Both live surfaces carry a colour transition, so submit is a fade, not a
    # jump; the marker's live shade is the bright user colour.
    assert re.search(
        r"\.terminal-input[^{]*\{[^}]*transition:[^;}]*border-color", css
    ), "the composer border must transition on submit"
    assert re.search(
        r"\.input-glyph[^{]*\{[^}]*color:\s*var\(--text-primary\)"
        r"[^}]*transition:[^;}]*color",
        css,
    ), "the live composer marker must be the bright user colour and transition"
    # Submitting recedes both: muted border and muted marker.
    assert re.search(
        r"\.terminal-card[^{]*data-composer=sent[^{]*\.terminal-input[^{]*\{"
        r"[^}]*border-color",
        css,
    ), "the submitted field must recede to its muted border"
    assert re.search(
        r"\.terminal-card[^{]*data-composer=sent[^{]*\.input-glyph[^{]*\{"
        r"[^}]*color:\s*var\(--code-muted\)",
        css,
    ), "the submitted marker must recede to the muted shade"


def test_hero_terminal_composer_starts_live_and_turns_off_on_submit(
    built_site,
) -> None:
    """The composer is live at first paint and through the entrance beat, and only
    turns off once the prompt has been submitted: the run publishes `data-composer`
    from live to sent at exactly that beat, never earlier."""
    script = (
        browser_dom(dist_body(_HOMEPAGE), expose_document=False)
        + _INIT
        + """
const card = doc.querySelector('.terminal-card');
const states = [];
initHeroTerminal(doc, {
  charDelay: 0,
  sendPause: 0,
  workingPause: 0,
  answerDelay: 0,
  startDelay: 0,
  loop: false,
  sleep: async () => {
    states.push({ composer: card.dataset.composer || '', prompts: countRows('user-prompt') });
  },
});

for (let i = 0; i < 200; i += 1) {
"""
        + _TICK
        + """}

console.log(JSON.stringify({
  start: states.length ? states[0].composer : 'none',
  sentBeforePrompt: states.some((s) => s.prompts === 0 && s.composer === 'sent'),
  sentWithPrompt: states.some((s) => s.prompts > 0 && s.composer === 'sent'),
  final: card.dataset.composer || '',
}));
"""
    )
    rendered = run_tsx_json(script)

    # First paint / entrance beat: live, never already submitted.
    assert rendered["start"] == "live"
    # The composer must never read "sent" before a prompt is in the scrollback.
    assert rendered["sentBeforePrompt"] is False
    assert rendered["sentWithPrompt"] is True
    assert rendered["final"] == "sent"


def test_hero_terminal_working_row_cycles_a_live_spinner(built_site) -> None:
    """While a tool call runs, the agent row's glyph is a *live* braille spinner —
    the terminal-agent convention for "working" — not a frozen glyph. Once the
    call settles the glyph becomes the static marker, and nothing stays working.
    (The ChunkHound row's own progress cue is a blinking dot; see
    `test_hero_terminal_chunkhound_dot_blinks_and_freezes_under_reduced_motion`.)"""
    braille = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    script = (
        browser_dom(dist_body(_HOMEPAGE), expose_document=False)
        + _INIT
        + """
const frames = new Set();
initHeroTerminal(doc, {
  charDelay: 0,
  sendPause: 0,
  // The production working beat, so the spinner gets its full ~6-7 frames.
  workingPause: 900,
  answerDelay: 0,
  startDelay: 0,
  loop: false,
  // Snapshot the working glyph on every sleep; the working class is only
  // cleared after the spinner has finished, so these are live frames.
  sleep: async () => {
    const glyph = container.querySelector('.call.is-working .glyph');
    if (glyph) frames.add(glyph.textContent);
  },
});

for (let i = 0; i < 200; i += 1) {
  await new Promise((resolve) => setTimeout(resolve, 0));
}

console.log(JSON.stringify({
  frames: Array.from(frames),
  settledGlyphs: Array.from(container.querySelectorAll('.call .glyph'))
    .map((el) => el.textContent),
  workingCount: container.querySelectorAll('.is-working').length,
}));
"""
    )
    rendered = run_tsx_json(script)

    frames = rendered["frames"]
    # More than one distinct frame was seen while working → the glyph animates.
    assert len(frames) >= 2, f"working glyph never cycled: {frames!r}"
    # Every observed frame is a braille spinner frame, never the settled marker.
    assert all(frame.strip() in braille for frame in frames), frames
    # Every call settled to the static marker; no row is left mid-spin.
    assert rendered["settledGlyphs"] == ["⏺ "] * 4
    assert rendered["workingCount"] == 0


def test_hero_terminal_working_label_sweep_freezes_under_reduced_motion(
    built_site,
) -> None:
    """The working label carries a live highlight sweep so the row reads as
    active; reduced motion freezes it to the flat muted color."""
    css = "".join(
        path.read_text(encoding="utf-8")
        for path in (DIST / "_astro").glob("*.css")
    )
    assert "@keyframes working-sweep" in css
    assert re.search(
        r"\.call\.is-working[^{]*\.call-text[^{]*\{[^}]*working-sweep", css
    ), "the working label must run the sweep while the tool call is in flight"
    assert re.search(
        r"@media[^{]*prefers-reduced-motion[^{]*reduce[^{]*\{"
        r"(?:[^{}]|\{[^{}]*\})*"
        r"\.call\.is-working[^{]*\.call-text[^{]*\{[^}]*animation:\s*none",
        css,
    ), "reduced motion must freeze the working-label sweep"


def test_hero_terminal_chunkhound_progress_runs_with_the_agent(
    built_site,
) -> None:
    """The ChunkHound call starts right after the agent's line and runs while the
    agent is still working: both rows are live and highlighted together with the
    same beat, the receipt row shows the blinking dot, then the citations land and
    the agent settles."""
    script = (
        browser_dom(dist_body(_HOMEPAGE), expose_document=False)
        + _INIT
        + """
const concurrent = [];
initHeroTerminal(doc, {
  charDelay: 0,
  sendPause: 0,
  workingPause: 0,
  answerDelay: 0,
  startDelay: 0,
  loop: false,
  sleep: async () => {
    const agent = container.querySelector('.call.is-working');
    const receipt = container.querySelector('.evidence.is-working');
    if (!agent || !receipt) return;
    concurrent.push({
      agentRole: agent.querySelector('.role-label')?.textContent,
      receiptRole: receipt.querySelector('.role-label')?.textContent,
      agentBeat: agent.getAttribute('data-beat'),
      receiptBeat: receipt.getAttribute('data-beat'),
      agentCurrent: agent.hasAttribute('data-current'),
      receiptCurrent: receipt.hasAttribute('data-current'),
      dot: receipt.querySelector('.glyph')?.textContent,
    });
  },
});

for (let i = 0; i < 400; i += 1) {
  await new Promise((resolve) => setTimeout(resolve, 0));
}

const evidence = [...container.querySelectorAll('.evidence')];
console.log(JSON.stringify({
  concurrent,
  evidenceClasses: evidence.map((line) => line.className),
  evidenceGlyphs: evidence.map((line) => line.querySelector('.glyph')?.textContent),
  evidenceItems: evidence.map((line) => line.querySelectorAll('.evidence-text').length),
  workingCount: container.querySelectorAll('.is-working').length,
}));
"""
    )
    rendered = run_tsx_json(script)

    # The agent row and its ChunkHound receipt are live at the same time, both
    # highlighted with the same beat, and the receipt carries the blinking dot.
    assert rendered["concurrent"], "the ChunkHound call never overlapped the agent"
    assert all(
        snap["agentRole"] == "Agent"
        and snap["receiptRole"] == "ChunkHound"
        and snap["agentBeat"] == snap["receiptBeat"]
        and snap["agentCurrent"] is True
        and snap["receiptCurrent"] is True
        and snap["dot"] == "● "
        for snap in rendered["concurrent"]
    )
    # Every receipt settled in place: no row left working, marker swapped to ↳,
    # one citation per receipt.
    assert rendered["workingCount"] == 0
    assert rendered["evidenceClasses"] == ["line evidence"] * 3
    assert rendered["evidenceGlyphs"] == ["↳ "] * 3
    assert rendered["evidenceItems"] == [1, 1, 1]


def test_hero_terminal_chunkhound_dot_blinks_and_freezes_under_reduced_motion(
    built_site,
) -> None:
    """The ChunkHound row's in-flight progress is a standard terminal blinking
    dot; reduced motion freezes it to a solid dot."""
    css = "".join(
        path.read_text(encoding="utf-8")
        for path in (DIST / "_astro").glob("*.css")
    )
    assert "@keyframes blink" in css
    assert re.search(
        r"\.evidence\.is-working[^{]*\.glyph[^{]*\{[^}]*animation:[^;}]*blink",
        css,
    ), "the working ChunkHound row must blink its dot"
    assert re.search(
        r"@media[^{]*prefers-reduced-motion[^{]*reduce[^{]*\{"
        r"(?:[^{}]|\{[^{}]*\})*"
        r"\.evidence\.is-working[^{]*\.glyph[^{]*\{[^}]*animation:\s*none",
        css,
    ), "reduced motion must freeze the ChunkHound progress dot"


def test_hero_terminal_submit_moves_prompt_from_input_to_transcript(
    built_site,
) -> None:
    """A typed followup visibly leaves the fixed input and lands in the
    scrollback as a submitted line — the send moment of the harness."""
    script = (
        browser_dom(dist_body(_HOMEPAGE), expose_document=False)
        + _INIT
        + """
const snaps = [];
const promptCount = () => countRows('user-prompt');
initHeroTerminal(doc, {
  charDelay: 1,
  sendPause: 11,
  workingPause: 13,
  answerDelay: 17,
  loop: false,
  sleep: async () => {
    snaps.push({ input: inputText.textContent, prompts: promptCount() });
  },
});

for (let i = 0; i < 60; i += 1) {
"""
        + _TICK
        + """}

let moved = false;
for (let i = 0; i < snaps.length; i += 1) {
  for (let j = i + 1; j < snaps.length; j += 1) {
    if (
      snaps[i].input !== "" &&
      snaps[j].prompts === snaps[i].prompts + 1 &&
      snaps[j].input === ""
    ) {
      moved = true;
      break;
    }
  }
  if (moved) break;
}

console.log(JSON.stringify({
  typed: snaps.some((snap) => snap.input !== ""),
  moved,
  finalPrompts: promptCount(),
  finalInput: inputText.textContent,
}));
"""
    )
    rendered = run_tsx_json(script)

    assert rendered["typed"] is True
    assert rendered["moved"] is True
    assert rendered["finalPrompts"] == 1
    assert rendered["finalInput"] == ""


def test_hero_terminal_types_agent_output(built_site) -> None:
    """The agent's own prose streams the way the user's prompt does: a working
    call's label grows character by character and the verdict streams to its full
    text — the model-output half of the harness feel."""
    script = (
        browser_dom(dist_body(_HOMEPAGE), expose_document=False)
        + _INIT
        + """
const callSnaps = [];
const answerSnaps = [];
initHeroTerminal(doc, {
  charDelay: 0,
  sendPause: 0,
  workingPause: 0,
  answerDelay: 0,
  loop: false,
  sleep: async () => {
    const working = container.querySelector('.call.is-working .call-text');
    if (working) callSnaps.push(working.textContent);
    const result = container.querySelector('.result');
    if (result) answerSnaps.push(result.textContent);
  },
});

for (let i = 0; i < 400; i += 1) {
"""
        + _TICK
        + """}

console.log(JSON.stringify({
  callSnaps,
  answerSnaps,
  finalCalls: Array.from(container.querySelectorAll('.call-text'))
    .map((el) => el.textContent),
  finalAnswer: container.querySelector('.result').textContent,
}));
"""
    )
    rendered = run_tsx_json(script)

    expected_calls = [call["call"] for call in EXPECTED_DEMO["calls"]] + [
        EXPECTED_DEMO["synthesis"]
    ]
    # Every agent row still ends on its complete text.
    assert rendered["finalCalls"] == expected_calls
    assert rendered["finalAnswer"] == EXPECTED_DEMO["answer"]

    def strict_prefixes(texts: list[str]) -> set[str]:
        return {text[:end] for text in texts for end in range(1, len(text))}

    # In-flight snapshots caught the text mid-stream — non-empty and a strict
    # prefix of a real row — so it grew in rather than landing whole.
    assert any(
        snap in strict_prefixes(expected_calls) for snap in rendered["callSnaps"]
    ), rendered["callSnaps"]
    assert any(
        snap in strict_prefixes([EXPECTED_DEMO["answer"]])
        for snap in rendered["answerSnaps"]
    ), rendered["answerSnaps"]


def test_hero_terminal_types_chunkhound_receipt(built_site) -> None:
    """The engine's cited return streams the way the agent's own output does:
    each citation grows character by character into the receipt. Text is
    sequential — the receipt only types after the agent's call text is complete,
    while the agent's progress indicator stays live."""
    script = (
        browser_dom(dist_body(_HOMEPAGE), expose_document=False)
        + _INIT
        + """
const snaps = [];
initHeroTerminal(doc, {
  charDelay: 0,
  sendPause: 0,
  workingPause: 0,
  answerDelay: 0,
  startDelay: 0,
  loop: false,
  sleep: async () => {
    const agent = container.querySelector('.call.is-working .call-text');
    const items = container.querySelectorAll('.evidence.is-working .evidence-text');
    for (const item of items) snaps.push({ item: item.textContent, agent: agent?.textContent });
  },
});

for (let i = 0; i < 400; i += 1) {
  await new Promise((resolve) => setTimeout(resolve, 0));
}

console.log(JSON.stringify({
  snaps,
  finalItems: Array.from(container.querySelectorAll('.evidence-text'))
    .map((el) => el.textContent),
}));
"""
    )
    rendered = run_tsx_json(script)

    expected_items = [
        source_text(source)
        for call in EXPECTED_DEMO["calls"]
        for source in call["sources"]
    ]
    full_calls = [call["call"] for call in EXPECTED_DEMO["calls"]] + [
        EXPECTED_DEMO["synthesis"]
    ]

    def strict_prefixes(texts: list[str]) -> set[str]:
        return {text[:end] for text in texts for end in range(1, len(text))}

    # Every receipt ends on its complete citation.
    assert rendered["finalItems"] == expected_items

    # In-flight snapshots caught a citation mid-stream (a strict prefix), and at
    # that moment the agent's call text was already complete — text types
    # sequentially, never concurrently.
    prefixes = strict_prefixes(expected_items)
    mid_stream = [snap for snap in rendered["snaps"] if snap["item"] in prefixes]
    assert mid_stream, rendered["snaps"]
    assert all(snap["agent"] in full_calls for snap in mid_stream), mid_stream


def test_hero_terminal_renders_instantly_under_reduced_motion(built_site) -> None:
    script = (
        browser_dom(dist_body(_HOMEPAGE), expose_document=False)
        + _INIT
        + """
let sleepCalls = 0;
initHeroTerminal(doc, {
  loop: false,
  reducedMotion: true,
  sleep: async () => { sleepCalls += 1; },
});

"""
        + _TICK
        + _TICK
        + """
console.log(JSON.stringify({
  sleepCalls,
  transcript: textOf(container),
  promptCount: countRows('user-prompt'),
  callCount: countRows('call'),
  answerCount: countRows('answer'),
  evidenceCount: countRows('evidence'),
  noteCount: container.querySelectorAll('.note').length,
  inputText: inputText.textContent,
}));
"""
        + _TRANSCRIPT_STRUCTURE
    )
    rendered = run_tsx_json(script)

    # Reduced motion: full transcript renders immediately — no typing animation,
    # no per-step delays, no looping autoplay.
    assert rendered["sleepCalls"] == 0
    assert rendered["promptCount"] == 1
    assert rendered["callCount"] == 4
    assert rendered["answerCount"] == 1
    assert rendered["evidenceCount"] == 3
    # Reduced motion is also the no-animation path for the notes: the whole run,
    # annotation included, lands at once.
    assert rendered["noteCount"] == 3
    assert rendered["inputText"] == ""
    transcript = rendered["transcript"]
    assert EXPECTED_DEMO["prompt"] in transcript
    assert EXPECTED_DEMO["synthesis"] in transcript
    assert EXPECTED_DEMO["answer"] in transcript
    for call in EXPECTED_DEMO["calls"]:
        assert call["call"] in transcript
        for source in call["sources"]:
            assert source_text(source) in transcript


def test_hero_terminal_remeasures_after_font_settling(built_site) -> None:
    """The viewport's locked height comes from a one-off geometry probe of the
    whole transcript, and it is re-run once webfonts settle (metrics change, so
    does the wrapped height).

    happy-dom has no layout engine, so the probe's geometry is stubbed at the
    true boundary. The stubs pin one user-facing invariant: the lock must be
    measured against the *live run's* inline size and let its own measure cap
    go. Measure the viewport instead (wider, by the room the notes take in the
    margin) and the probe wraps less than the run does — the box is under-
    locked and clips the verdict it exists to protect.
    """
    script = (
        browser_dom(dist_body(_HOMEPAGE), expose_document=False)
        + """
const doc = window.document;
const viewportEl = doc.getElementById('terminal-viewport');
const runEl = doc.getElementById('terminal-lines');

let measureHeight = 120;
const probe = [];
Object.defineProperty(viewportEl, 'clientWidth', { value: 640 });
Object.defineProperty(runEl, 'clientWidth', { value: 320 });
const realRect = window.Element.prototype.getBoundingClientRect;
window.Element.prototype.getBoundingClientRect = function () {
  if (this.classList.contains('terminal-measure')) {
    probe.push({ width: this.style.width, maxWidth: this.style.maxInlineSize });
    return {
      height: measureHeight, top: 0, bottom: measureHeight,
      left: 0, right: 0, width: 320, x: 0, y: 0, toJSON() {},
    };
  }
  return realRect.call(this);
};

let resolveFonts;
const fontsReady = new Promise((resolve) => { resolveFonts = resolve; });
Object.defineProperty(doc, 'fonts', { value: { ready: fontsReady } });

const { initHeroTerminal } = await import('./site/src/scripts/hero-terminal.ts');
initHeroTerminal(doc, {
  charDelay: 0,
  sendPause: 0,
  workingPause: 0,
  answerDelay: 0,
  loop: false,
  sleep: async () => {},
});

const initialHeight = viewportEl.style.height;
measureHeight = 180;
resolveFonts();
await fontsReady;
await new Promise((resolve) => setTimeout(resolve, 0));
const afterFontsHeight = viewportEl.style.height;

console.log(JSON.stringify({ initialHeight, afterFontsHeight, probe }));
"""
    )
    rendered = run_tsx_json(script)

    assert rendered["initialHeight"] == "120px"
    assert rendered["afterFontsHeight"] == "180px"
    # Every probe: the run's own width, measure cap released, transcript present.
    assert rendered["probe"] == [
        {"width": "320px", "maxWidth": "none"}
    ] * len(rendered["probe"]), rendered["probe"]
    assert len(rendered["probe"]) >= 2, "the probe must re-run after fonts settle"


def test_hero_terminal_restarts_after_pagehide_and_pageshow(built_site) -> None:
    script = (
        browser_dom(dist_body(_HOMEPAGE), expose_document=False)
        + _INIT
        + """
initHeroTerminal(doc, {
  charDelay: 0,
  sendPause: 0,
  workingPause: 0,
  answerDelay: 0,
  loop: false,
  sleep: async () => {},
});

const observer = FakeIntersectionObserver.instances.at(-1);
observer.callback([{ isIntersecting: true }]);
await new Promise((resolve) => setTimeout(resolve, 0));
const firstRenderCount = container.children.length;

window.dispatchEvent(new window.Event('pagehide'));
const afterPagehideCount = container.children.length;

window.dispatchEvent(new window.Event('pageshow'));
flushRaf();
await new Promise((resolve) => setTimeout(resolve, 0));
await new Promise((resolve) => setTimeout(resolve, 0));
const afterPageshowCount = container.children.length;

console.log(JSON.stringify({
  observedClassName: observer.targets[0].className,
  firstRenderCount,
  afterPagehideCount,
  afterPageshowCount,
  inputPersists: doc.getElementById('terminal-input') === inputEl,
}));
"""
    )
    rendered = run_tsx_json(script)

    # The observer must gate the terminal card; the one-shot run also adds
    # `is-settled` to that same element; the fixed input survives everything.
    assert rendered["observedClassName"].startswith("terminal-card")
    assert rendered["firstRenderCount"] > 0
    assert rendered["afterPagehideCount"] == 0
    assert rendered["afterPageshowCount"] > 0
    assert rendered["inputPersists"] is True


def test_hero_terminal_reduced_motion_flip_settles_in_flight_render(built_site) -> None:
    script = (
        browser_dom(dist_body(_HOMEPAGE), expose_document=False)
        + _INIT
        + """
// Gate the first sleep so the render stalls mid-typing; flips land while
// the followup is half-typed into the fixed input.
let sleepCalls = 0;
let release;
const gate = new Promise((resolve) => { release = resolve; });
initHeroTerminal(doc, {
  charDelay: 0,
  sendPause: 0,
  workingPause: 0,
  answerDelay: 0,
  loop: false,
  // Sleep #1 is the one-time start gate (let it pass); later sleeps stall the
  // render mid-typing.
  sleep: () => {
    sleepCalls += 1;
    return sleepCalls === 1 ? Promise.resolve() : gate;
  },
});

"""
        + _TICK
        + _TICK
        + _TICK
        + _TICK
        + """
const before = {
  input: inputText.textContent,
  promptCount: countRows('user-prompt'),
};

globalThis.setReducedMotion(true);
release();
"""
        + _TICK
        + _TICK
        + _TICK
        + """
const afterFlip = {
  sleepCalls,
  promptCount: countRows('user-prompt'),
  callCount: countRows('call'),
  answerCount: countRows('answer'),
  evidenceCount: countRows('evidence'),
  transcript: textOf(container),
};

globalThis.setReducedMotion(false);
"""
        + _TICK
        + _TICK
        + """
const afterFlipBack = {
  lineCount: container.querySelectorAll('.line').length,
  transcript: textOf(container),
};

console.log(JSON.stringify({ before, afterFlip, afterFlipBack }));
"""
        + _TRANSCRIPT_STRUCTURE
    )
    rendered = run_tsx_json(script)

    # The gated render stalls with the first followup half-typed in the fixed
    # input and nothing submitted yet.
    assert rendered["before"]["input"] != ""
    assert rendered["before"]["promptCount"] == 0

    # Flipping to reduced mid-render skips the pending animation to the
    # final static transcript; no further sleeps/looping happen.
    assert rendered["afterFlip"]["promptCount"] == 1
    assert rendered["afterFlip"]["callCount"] == 4
    assert rendered["afterFlip"]["answerCount"] == 1
    assert rendered["afterFlip"]["evidenceCount"] == 3
    # Sleep #1 was the one-time start gate; sleep #2 stalled the typing render.
    assert rendered["afterFlip"]["sleepCalls"] == 2
    transcript = rendered["afterFlip"]["transcript"]
    assert EXPECTED_DEMO["prompt"] in transcript
    assert EXPECTED_DEMO["synthesis"] in transcript
    assert EXPECTED_DEMO["answer"] in transcript
    for call in EXPECTED_DEMO["calls"]:
        assert call["call"] in transcript
        for source in call["sources"]:
            assert source_text(source) in transcript

    # Flipping back must not re-run the completed transcript.
    assert rendered["afterFlipBack"]["lineCount"] == 9
    assert rendered["afterFlipBack"]["transcript"] == rendered["afterFlip"]["transcript"]


def test_hero_demo_has_complete_copy() -> None:
    """DEMO is the transcript contract: one question, exactly three fan-out
    research calls each with a non-empty cited return, a synthesis beat, and a
    non-empty verdict."""
    demo = run_tsx_json(
        "import { DEMO } from './site/src/scripts/hero-transcript.ts';\n"
        "console.log(JSON.stringify(DEMO));\n"
    )

    assert set(demo) == {"prompt", "calls", "synthesis", "answer"}
    assert demo["prompt"].strip()
    assert demo["synthesis"].strip()
    assert demo["answer"].strip()
    assert len(demo["calls"]) == 3, "the demo fans out exactly three calls"
    # Fold budget (see hero-transcript.ts): the narrowest card's transcript
    # measure is ~46 chars, so a row past ~90 costs a second wrapped line and
    # pushes the composer toward the fold.
    assert len(demo["prompt"]) <= 90, f"prompt out of the fold budget: {demo['prompt']}"
    assert len(demo["answer"]) <= 90, f"verdict out of the fold budget: {demo['answer']}"
    for call in demo["calls"]:
        assert set(call) == {"call", "sources"}
        assert call["call"].strip()
        assert call["sources"], "every call must return cited sources"
        for source in call["sources"]:
            assert set(source) == {"kind", "cite"}
            assert source["kind"].strip()
            assert source["cite"].strip()
    assert demo == EXPECTED_DEMO


def test_hero_demo_source_types_are_promised_by_subheadline() -> None:
    """No copy drift: every source class the demo shows must be one the
    subheadline promises. The demo may show a focused subset — it fans out three
    calls — but must never invent a class the subheadline does not claim, and
    the cross-source story needs at least three classes."""
    import json

    positioning = json.loads(
        (ROOT / "site" / "src" / "lib" / "positioning.json").read_text(
            encoding="utf-8"
        )
    )
    subheadline = positioning["subheadline"].lower()
    demo = run_tsx_json(
        "import { DEMO } from './site/src/scripts/hero-transcript.ts';\n"
        "console.log(JSON.stringify(DEMO));\n"
    )
    kinds = {
        source["kind"] for call in demo["calls"] for source in call["sources"]
    }

    assert len(kinds) >= 3, "the demo must cross-reference at least three classes"
    for kind in kinds:
        assert kind in subheadline, (
            f"demo shows {kind!r}, which the subheadline does not promise"
        )
    for noun in SOURCE_TYPE_NOUNS:
        assert noun in subheadline, f"subheadline dropped source type {noun!r}"


def test_hero_noscript_carries_static_transcript(built_site) -> None:
    """JS-off visitors get the full settled transcript; the JS-on container
    starts empty so the run never competes with the hero copy."""
    from tests.site.dom_helpers import strip_scripts

    raw = strip_scripts((DIST / _HOMEPAGE).read_text(encoding="utf-8"))
    # Astro escapes quotes/entities in text content; compare the decoded copy.
    static = html.unescape(hero_noscript(raw))
    assert EXPECTED_DEMO["prompt"] in static
    assert EXPECTED_DEMO["synthesis"] in static
    assert EXPECTED_DEMO["answer"] in static
    for call in EXPECTED_DEMO["calls"]:
        assert call["call"] in static
        for source in call["sources"]:
            assert source["kind"] in static
            assert source["cite"] in static
    assert "ChunkHound" in static and "Agent" in static

    # The live container holds nothing but the noscript fallback pre-JS.
    match = re.search(
        r'<div class="terminal-lines" id="terminal-lines"[^>]*>', raw
    )
    assert match is not None
    seg = raw[
        match.end() : raw.index('<div class="terminal-composer"', match.end())
    ]
    without_noscript = re.sub(
        r"<noscript>.*?</noscript>", "", seg, flags=re.S
    )
    # Only the closing tags (run + viewport) remain. Whitespace between block
    # tags is minifier noise, not a contract, so compare tokens.
    assert re.sub(r"\s+", "", without_noscript) == "</div></div>"


def test_hero_terminal_stop_control_halts_the_run(built_site) -> None:
    """WCAG 2.2.2 (Pause, Stop, Hide): the auto-playing run exceeds five seconds,
    so the chrome exposes a user-operable control. It reads "Stop" while a run is
    in flight; activating it halts the run and reveals the settled transcript, and
    the control then reads "Replay"."""
    script = (
        browser_dom(dist_body(_HOMEPAGE), expose_document=False)
        + _INIT
        + """
const card = container.closest('.terminal-card');
const control = doc.querySelector('.terminal-control');
let sleepCalls = 0;
let release;
const gate = new Promise((resolve) => { release = resolve; });
initHeroTerminal(doc, {
  charDelay: 0,
  sendPause: 0,
  workingPause: 0,
  answerDelay: 0,
  startDelay: 0,
  loop: false,
  // Stall the run on its 2nd sleep (mid prompt-typing) so it is in flight.
  sleep: () => { sleepCalls += 1; return sleepCalls >= 2 ? gate : Promise.resolve(); },
});

for (let i = 0; i < 20; i += 1) {
  await new Promise((resolve) => setTimeout(resolve, 0));
}

const midRun = {
  label: control.textContent.trim(),
  aria: control.getAttribute('aria-label'),
  settled: card.classList.contains('is-settled'),
};

control.click();  // request stop
release();
for (let i = 0; i < 80; i += 1) {
  await new Promise((resolve) => setTimeout(resolve, 0));
}

const countOf = (cls) => container.querySelectorAll(`.line.${cls}`).length;
console.log(JSON.stringify({
  midRun,
  label: control.textContent.trim(),
  aria: control.getAttribute('aria-label'),
  settled: card.classList.contains('is-settled'),
  promptCount: countOf('user-prompt'),
  callCount: countOf('call'),
  evidenceCount: countOf('evidence'),
  answerCount: countOf('answer'),
  transcript: textOf(container),
}));
"""
        + _TRANSCRIPT_STRUCTURE
    )
    rendered = run_tsx_json(script)

    # While the run is in flight the control offers a stop.
    assert rendered["midRun"] == {
        "label": "Stop",
        "aria": "Stop animation",
        "settled": False,
    }
    # Stopping halts the run and reveals the full settled transcript; the control
    # reverts to its replay role.
    assert rendered["settled"] is True
    assert rendered["label"] == "Replay"
    assert rendered["aria"] == "Replay demo"
    assert rendered["promptCount"] == 1
    assert rendered["callCount"] == 4
    assert rendered["evidenceCount"] == 3
    assert rendered["answerCount"] == 1
    assert EXPECTED_DEMO["synthesis"] in rendered["transcript"]
    assert EXPECTED_DEMO["answer"] in rendered["transcript"]


def test_hero_terminal_plays_once_by_default_and_replay_reruns_once(
    built_site,
) -> None:
    """Default loop is off (one pass, then still); the context-aware control
    re-runs the transcript exactly once from a clean state."""
    script = (
        browser_dom(dist_body(_HOMEPAGE), expose_document=False)
        + _INIT
        + """
const card = container.closest('.terminal-card');
const control = doc.querySelector('.terminal-control');
const sleeps = [];
// Distinct option values let the log reveal how many passes ran: only a pass
// emits the 300ms ANSWER_DELAY sleep, and only the first run's start gate
// emits 13ms.
initHeroTerminal(doc, {
  charDelay: 7,
  sendPause: 34,
  workingPause: 31,
  answerDelay: 300,
  startDelay: 13,
  sleep: async (ms) => { sleeps.push(ms); },
});

for (let i = 0; i < 120; i += 1) {
  await new Promise((resolve) => setTimeout(resolve, 0));
}

const countOf = (ms) => sleeps.filter((value) => value === ms).length;
const afterFirst = {
  answerPasses: countOf(300),
  startGates: countOf(13),
  settled: card.classList.contains('is-settled'),
  controlLabel: control.textContent.trim(),
};

control.click();
for (let i = 0; i < 120; i += 1) {
  await new Promise((resolve) => setTimeout(resolve, 0));
}

const afterReplay = {
  answerPasses: countOf(300),
  startGates: countOf(13),
  settled: card.classList.contains('is-settled'),
  promptCount: countRows('user-prompt'),
  controlLabel: control.textContent.trim(),
};

console.log(JSON.stringify({ afterFirst, afterReplay }));
"""
    )
    rendered = run_tsx_json(script)

    assert rendered["afterFirst"] == {
        "answerPasses": 1,
        "startGates": 1,
        "settled": True,
        "controlLabel": "Replay",
    }
    assert rendered["afterReplay"] == {
        "answerPasses": 2,
        "startGates": 1,
        "settled": True,
        "promptCount": 1,
        "controlLabel": "Replay",
    }


def test_hero_terminal_scrollback_follows_tail_and_yields_to_the_reader(
    built_site,
) -> None:
    """Scrollback contract: the run follows the tail (newest output pinned by the
    composer); a reader who scrolls back takes the viewport — later output grows
    beneath it instead of yanking them down — and the jump control hands the tail
    back."""
    script = (
        browser_dom(dist_body(_HOMEPAGE), expose_document=False)
        + _INIT
        + """
// happy-dom has no layout: stub the geometry so the follow decision is exercised
// against real JS. The transcript's layout overflows to 1000 while the viewport
// reports 1200 (= layout + the transient offset a push transform would add) —
// following must pin the layout bottom, never the viewport's runtime bottom.
let scrollTop = 0;
const layoutHeight = 1000;
const runtimeScrollHeight = layoutHeight + 200;
Object.defineProperty(viewportEl, 'clientWidth', { value: 320 });
Object.defineProperty(viewportEl, 'clientHeight', { get: () => 240 });
Object.defineProperty(viewportEl, 'scrollHeight', { get: () => runtimeScrollHeight });
Object.defineProperty(container, 'scrollHeight', { get: () => layoutHeight });
Object.defineProperty(viewportEl, 'scrollTop', {
  get: () => scrollTop,
  set: (value) => { scrollTop = value; },
});

initHeroTerminal(doc, {
  charDelay: 0,
  sendPause: 0,
  workingPause: 0,
  answerDelay: 0,
  loop: false,
  sleep: async () => { await new Promise((resolve) => setTimeout(resolve, 0)); },
});

const card = container.closest('.terminal-card');
const jump = card.querySelector('.terminal-jump');
const followingAtStart = card.dataset.following;

// Advance into the run so the transcript has started filling.
for (let i = 0; i < 40; i += 1) {
"""
        + _TICK
        + """
}

// Reader scrolls back off the tail mid-run.
viewportEl.scrollTop = 0;
viewportEl.dispatchEvent(new Event('scroll'));
const linesAtScroll = container.children.length;
const followingOffTail = card.dataset.following;

// Output keeps arriving while the reader is away.
for (let i = 0; i < 160; i += 1) {
"""
        + _TICK
        + """
}
const linesWhileReading = container.children.length;
const frozenScrollTop = scrollTop;

// The jump control hands the tail back.
jump.dispatchEvent(new Event('click'));
const followingAfterJump = card.dataset.following;
const pinnedScrollTop = scrollTop;

console.log(JSON.stringify({
  followingAtStart,
  followingOffTail,
  linesAtScroll,
  linesWhileReading,
  frozenScrollTop,
  followingAfterJump,
  pinnedScrollTop,
  layoutSpan: layoutHeight - 240,
  runtimeScrollHeight,
}));
"""
    )
    rendered = run_tsx_json(script)

    # The run starts on the tail.
    assert rendered["followingAtStart"] == "true"
    # Scrolling back takes the viewport from the run...
    assert rendered["followingOffTail"] == "false"
    # ...output keeps arriving while the reader is away...
    assert rendered["linesWhileReading"] > rendered["linesAtScroll"]
    # ...and the view is never yanked back to the tail.
    assert rendered["frozenScrollTop"] == 0
    # Asking for the tail restores follow and pins the newest output.
    assert rendered["followingAfterJump"] == "true"
    assert rendered["pinnedScrollTop"] == rendered["layoutSpan"]
    assert rendered["pinnedScrollTop"] != rendered["runtimeScrollHeight"]


def test_hero_terminal_follow_ignores_transient_transform_overflow(
    built_site,
) -> None:
    """A push transform briefly inflates the viewport's runtime scrollHeight.
    Tail-follow must read the transcript's layout instead: while the transcript
    fits its box, following the tail must not scroll the view at all, or the
    push glide is cancelled and the history fade flashes mid-beat."""
    script = (
        browser_dom(dist_body(_HOMEPAGE), expose_document=False)
        + _INIT
        + """
// The transcript fits its box (layout span 0) while the viewport reports the
// transient overflow a push transform would create. The stub values are the
// boundary happy-dom cannot lay out, so geometry is pinned by hand.
let scrollTop = 0;
Object.defineProperty(viewportEl, 'clientWidth', { value: 320 });
Object.defineProperty(viewportEl, 'clientHeight', { get: () => 240 });
Object.defineProperty(viewportEl, 'scrollHeight', { get: () => 500 });
Object.defineProperty(viewportEl, 'scrollTop', {
  get: () => scrollTop,
  set: (value) => { scrollTop = value; },
});
Object.defineProperty(container, 'scrollHeight', { get: () => 240 });

initHeroTerminal(doc, {
  charDelay: 0,
  sendPause: 0,
  workingPause: 0,
  answerDelay: 0,
  loop: false,
  sleep: async () => { await new Promise((resolve) => setTimeout(resolve, 0)); },
});

const card = container.closest('.terminal-card');
for (let i = 0; i < 200; i += 1) {
"""
        + _TICK
        + """
}
console.log(JSON.stringify({
  scrollTop,
  scrollable: card.dataset.scrollable,
  lines: container.children.length,
}));
"""
    )
    rendered = run_tsx_json(script)

    # The run produced output...
    assert rendered["lines"] > 0
    # ...yet the transient runtime overflow never moved the view or the fade.
    assert rendered["scrollTop"] == 0
    assert rendered["scrollable"] == "false"


def test_hero_terminal_overflowing_scrollback_still_glides_its_push(
    built_site,
) -> None:
    """The bottom-entry glide survives the fit→overflow crossing. Once the
    transcript overflows its box (the short-viewport clamp) the first row no
    longer moves in layout, so the push must invert the pending tail scroll
    instead — otherwise every beat after the crossing jumps. Geometry is pinned
    by hand because happy-dom has no layout."""
    script = (
        browser_dom(dist_body(_HOMEPAGE), expose_document=False)
        + _INIT
        + """
// Overflow geometry: each appended beat makes the transcript 60px taller than
// its box, and the first row's viewport position is layout-fixed, so only the
// follow scroll can move it — the mobile regime the clamp creates.
let scrollTop = 0;
Object.defineProperty(viewportEl, 'clientWidth', { value: 320 });
Object.defineProperty(viewportEl, 'clientHeight', { get: () => 240 });
Object.defineProperty(container, 'scrollHeight', {
  get: () => 240 + container.children.length * 60,
});
Object.defineProperty(viewportEl, 'scrollTop', {
  get: () => scrollTop,
  set: (value) => { scrollTop = value; },
});

// Capture each glide the push sets: pushEnd flushes with a rect read right after
// pinning translateY(dy), so that read is the one moment dy is observable.
const glideOffsets = [];
const box = (top) => ({
  top, bottom: top + 64, left: 0, right: 100, width: 100, height: 64, x: 0, y: top,
});
window.Element.prototype.getBoundingClientRect = function () {
  if (this === container) {
    const match = /translateY\\(([-0-9.]+)px\\)/.exec(this.style.transform || '');
    glideOffsets.push(match ? Number(match[1]) : 0);
    return box(0);
  }
  if (this === container.firstElementChild) {
    // Layout-fixed first row: its viewport top moves only with the scroll.
    return box(100 - scrollTop);
  }
  return box(0);
};

initHeroTerminal(doc, {
  charDelay: 0,
  sendPause: 0,
  workingPause: 0,
  answerDelay: 0,
  loop: false,
  sleep: async () => { await new Promise((resolve) => setTimeout(resolve, 0)); },
});

for (let i = 0; i < 400; i += 1) {
"""
        + _TICK
        + """
}

const layoutSpan = Math.max(0, container.scrollHeight - viewportEl.clientHeight);
console.log(JSON.stringify({
  beats: countRows('call'),
  maxGlide: Math.max(0, ...glideOffsets),
  scrollTop,
  layoutSpan,
}));
"""
    )
    rendered = run_tsx_json(script)

    # The run overflowed its box and reached the tail...
    assert rendered["beats"] >= 2
    assert rendered["scrollTop"] == rendered["layoutSpan"]
    assert rendered["layoutSpan"] > 0
    # ...and the push still glided instead of only jumping the scroll.
    assert rendered["maxGlide"] > 1


def test_hero_terminal_composer_scrolls_to_keep_the_caret_visible(
    built_site,
) -> None:
    """The prompt field types on one line and scrolls right so the caret stays
    visible — the composer never wraps to a second row; sending resets it."""
    script = (
        browser_dom(dist_body(_HOMEPAGE), expose_document=False)
        + _INIT
        + """
// happy-dom has no layout: stub the single-line field's scroll geometry.
const inputLine = inputText.parentElement;
let lineScrollLeft = 0;
Object.defineProperty(inputLine, 'scrollWidth', { get: () => 420 });
Object.defineProperty(inputLine, 'scrollLeft', {
  get: () => lineScrollLeft,
  set: (value) => { lineScrollLeft = value; },
});

initHeroTerminal(doc, {
  charDelay: 0,
  sendPause: 0,
  workingPause: 0,
  answerDelay: 0,
  loop: false,
  sleep: async () => { await new Promise((resolve) => setTimeout(resolve, 0)); },
});

// Advance a few characters into the prompt.
for (let i = 0; i < 6; i += 1) {
"""
        + _TICK
        + """
}
const partial = { text: inputText.textContent, scrollLeft: lineScrollLeft };

// Sending moves the turn into the scrollback and resets the field.
for (let i = 0; i < 40; i += 1) {
"""
        + _TICK
        + """
}
const afterSend = { text: inputText.textContent, scrollLeft: lineScrollLeft };

console.log(JSON.stringify({ partial, afterSend }));
"""
    )
    rendered = run_tsx_json(script)

    # The prompt types into the single-line field, pinned to the caret.
    assert rendered["partial"]["text"] != ""
    assert rendered["partial"]["scrollLeft"] == 420
    # Sending clears the field and returns the scroll for the next turn.
    assert rendered["afterSend"] == {"text": "", "scrollLeft": 0}


def test_hero_transcript_colours_each_role_consistently(built_site) -> None:
    """One colour per narrative voice, carried by the glyph AND the copy, so the
    narrow tier (which drops the visible role column) still reads the run by
    colour alone: user = --text-primary (brightest neutral), agent = --code-text,
    ChunkHound = --code-accent (the one saturated voice in the run), verdict =
    --text-primary at 600, note = --code-muted on its own raised fill. Guards
    against a role's content drifting to the plain-text token ("no agent
    colour") and against a second saturated hue on the code surface."""
    contracts = {
        ".terminal-lines .prompt-text": "--text-primary",
        ".terminal-lines .call .glyph": "--code-text",
        ".terminal-lines .call-text": "--code-text",
        ".terminal-lines .result": "--text-primary",
        ".terminal-lines .evidence-text": "--code-accent",
        ".terminal-lines .note": "--code-muted",
        ".terminal-lines .note-headline": "--code-text",
    }
    for selector, token in contracts.items():
        assert any(
            f"var({token})" in body for body in bodies(selector)
        ), f"{selector} must carry its role colour {token}"
