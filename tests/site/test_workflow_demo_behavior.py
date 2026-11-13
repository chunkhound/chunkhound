# ruff: noqa: E501  # Embedded JavaScript keeps browser-like expressions intact.
"""User-facing contracts for the illustrative workflow run."""

import re

import pytest

from tests.site.browser_helpers import page_probe

# The demo's scope on every probe: the region and the card that frames it.
_SCOPE = """
const demo = page.getByRole('region', { name: 'Illustrative workflow run' });
const card = demo.locator('xpath=ancestor::div[contains(@class,"terminal-card")]');
"""

# Motion contracts need the animated path; the probe context requests reduced
# motion by default, so every animated test reloads under no preference.
_MOTION = "await page.emulateMedia({ reducedMotion: 'no-preference' });\n"

_ANIMATED = _MOTION + "await page.reload();\n"

# Timed choreography runs on page timers. Freeze those timers and step them, so
# a 500 ms confirmation window cannot be missed on a slow or fast CI runner.
_CLOCKED = _MOTION + """
const clockStart = new Date('2026-01-01T00:00:00Z');
await page.clock.install({ time: clockStart });
await page.clock.pauseAt(clockStart);
await page.reload();
// Clock does not control CSS smooth scrolling; this probe asserts geometry.
await page.addStyleTag({ content: 'html { scroll-behavior: auto !important; }' });
const stepUntil = async (predicate, label, maxMs = 15000) => {
  // 50 ms < the shortest asserted state window (500 ms), so no state is skipped.
  for (let elapsed = 0; elapsed < maxMs; elapsed += 50) {
    if (await predicate()) return;
    await page.clock.runFor(50);
  }
  throw new Error(`Timed out in fake time waiting for ${label}`);
};
"""


def _probe(widths: list[int], body: str) -> dict[str, dict]:
    return page_probe("/docs/workflows/", widths, _SCOPE + body)


def _one(body: str, width: int = 1000) -> dict:
    return _probe([width], body)[str(width)]


def test_static_run_shows_targets_stop_and_summary_without_javascript(
    built_site,
) -> None:
    result = _one("""
const context = await page.context().browser().newContext({ javaScriptEnabled: false });
try {
  const staticPage = await context.newPage();
  await staticPage.goto(new URL('/docs/workflows/', page.url()).href);
  const demo = staticPage.locator('[aria-label="Illustrative workflow run"]');
  const text = await demo.innerText();
  const preview = staticPage.getByText('Public preview', { exact: true });
  const tour = staticPage.getByRole('heading', { name: 'Control the harness' });
  return {
    text,
    demoCount: await demo.count(),
    previewCount: await preview.count(),
    command: (await staticPage.locator('.terminal-input .input-text').innerText()).trim(),
    beforeTour: await demo.evaluate((el, tour) =>
      !!(el.compareDocumentPosition(tour) & Node.DOCUMENT_POSITION_FOLLOWING),
      await tour.elementHandle()),
    controlVisible: await staticPage.locator('button[aria-label="Replay demo"]')
      .isVisible(),
  };
} finally { await context.close(); }
""")
    assert result["demoCount"] == 1
    assert result["previewCount"] == 1
    assert result["command"] == "/workflow implement-spec specs/checkout.md"
    assert result["beforeTour"]
    text = result["text"].lower()
    for phrase in (
        "/detect-tests",
        "baseline",
        "/skill:analyze-spec",
        "/plan-spec-fixes",
        "/apply-spec-fixes",
        "checkpoint 3",
        "checkpoints 3",
        "tests passed",
        "spend",
        "steps 4",
        "by model",
        "$0.69",
        "$0.35",
        "stopped at checkpoint",
    ):
        assert phrase in text
    assert not result["controlVisible"]


def test_reduced_motion_shows_complete_run_and_fits_every_width(built_site) -> None:
    results = _probe(
        [320, 1000, 1440],
        """
await demo.scrollIntoViewIfNeeded();
const before = await demo.innerText();
const input = card.locator('.terminal-input');
const command = (await input.locator('.input-text').innerText()).trim();
const fits = await input.evaluate(el => {
  const line = el.querySelector('.input-line').getBoundingClientRect();
  const range = document.createRange();
  range.selectNodeContents(el.querySelector('.input-text'));
  const rects = [...range.getClientRects()];
  return rects.length > 0 && rects.every(rect =>
    rect.left >= line.left - 1 && rect.right <= line.right + 1
    && rect.top >= line.top - 1 && rect.bottom <= line.bottom + 1);
});
const rect = await card.evaluate(el => {
  const r = el.getBoundingClientRect();
  const parent = el.parentElement.getBoundingClientRect();
  return {
    fits: r.left >= 0 && r.right <= innerWidth,
    centered: Math.abs((r.left + r.right) - (parent.left + parent.right)) < 2,
  };
});
await page.getByRole('button', { name: /replay demo/i }).click();
const after = await demo.innerText();
const afterCommand = (await input.locator('.input-text').innerText()).trim();
return { before, after, command, afterCommand, fits, ...rect };
""",
    )
    for width, result in results.items():
        assert "stopped at checkpoint" in result["before"].lower(), width
        assert result["before"] == result["after"], width
        assert result["command"] == "/workflow implement-spec specs/checkout.md", width
        assert result["afterCommand"] == result["command"], width
        assert result["fits"], width
        assert "checkpoints 3" in result["before"].lower(), width
        assert result["fits"] and result["centered"], width


@pytest.fixture(scope="module")
def animated_run(built_site) -> dict[str, dict]:
    """One animated run, measured once for every geometry contract.

    The contracts differ in what they assert, never in the run they observe, so
    replaying the scenario per test would only add browser launches and repeated
    waits.
    """
    return _probe(
        [320, 1000],
        _ANIMATED
        + """
await card.locator('.terminal-composer').scrollIntoViewIfNeeded();
await card.getByText('Cost target (USD)', { exact: false })
  .waitFor({ state: 'visible' });
const cost = await card.innerText();
await card.getByText(/time target — soft/i).waitFor({ state: 'visible' });
const time = await card.innerText();
const status = card.locator('.workflow-live-status');
await demo.getByText(/baseline tests passed/i).waitFor({ state: 'visible' });
const baseline = await status.innerText();
const effectStates = await status.evaluate(status => {
  const effect = status.querySelector('.workflow-effect');
  const metrics = [...status.querySelectorAll(
    '.workflow-checkpoints, .workflow-elapsed, .workflow-cost')];
  const original = effect.textContent;
  // Object method, not a named arrow: tsx keep-names rewrites named arrows into
  // __name(...) calls that cannot cross into the page.
  const measure = { bounds(el) {
    const r = el.getBoundingClientRect();
    const s = status.getBoundingClientRect();
    return { x: r.x - s.x, y: r.y - s.y, width: r.width, height: r.height };
  } };
  const states = ['', '$ test', '$ ' + 'long-running effect '.repeat(30)]
    .map(text => {
      effect.textContent = text;
      return { status: measure.bounds(status), effect: measure.bounds(effect),
        metrics: metrics.map(el => measure.bounds(el)),
        visible: getComputedStyle(effect).display !== 'none' };
    });
  effect.textContent = original;
  return states;
});
await demo.getByText('/apply-spec-fixes', { exact: true })
  .waitFor({ state: 'visible' });
const applied = await status.innerText();
await status.getByText(/stopping at next checkpoint/i).waitFor({ state: 'visible' });
const reached = await status.innerText();
const warning = card.locator('.workflow-warning');
await warning.waitFor({ state: 'visible' });
const warningGeometry = await warning.evaluate(el => {
  const bar = el.parentElement.getBoundingClientRect();
  const rect = el.getBoundingClientRect();
  return {
    text: el.innerText, fits: rect.left >= bar.left && rect.right <= bar.right,
    width: rect.width,
  };
});
const geometry = await status.evaluate(el => {
  const card = el.closest('.terminal-card').getBoundingClientRect();
  const input = el.closest('.terminal-card')
    .querySelector('.terminal-input').getBoundingClientRect();
  const rect = el.getBoundingClientRect();
  const bar = el.querySelector('.workflow-status-bar');
  const visible = [...bar.children].filter(c => getComputedStyle(c).display !== 'none');
  return {
    outside: rect.top >= input.bottom,
    inside: rect.bottom <= card.bottom && rect.left >= card.left
      && rect.right <= card.right,
    cellsFit: visible.every(c => c.getBoundingClientRect().left >= rect.left
      && c.getBoundingClientRect().right <= rect.right),
  };
});
await demo.getByText(/checkpoint 3: tested/i).waitFor({ state: 'visible' });
const tested = await status.innerText();
await demo.getByText(/stopped at checkpoint/i).waitFor({ state: 'visible' });
const summary = await demo.innerText();
return {
  cost, time, baseline, applied, reached, tested, geometry, summary,
  effectStates, warningGeometry,
  statusHidden: await status.isHidden(),
};
""",
    )


def test_animation_shows_dialogues_and_keeps_status_visible_until_checkpoint(
    animated_run,
) -> None:
    for width, result in animated_run.items():
        assert "$1" in result["cost"] and "next checkpoint" in result["cost"].lower()
        assert "1 hour" in result["time"]
        assert "next checkpoint" in result["time"].lower()
        assert "implement-spec" in result["baseline"] and "⏱" in result["baseline"], (
            f"{width}: live status must be visible before the cost target is reached"
        )
        assert "stopping at next checkpoint" not in result["applied"].lower()
        assert "stopping at next checkpoint" in result["reached"].lower()
        assert "stopping at next checkpoint" in result["tested"].lower()
        assert all(result["geometry"].values()), width
        assert result["statusHidden"], width
        for phrase in ("cost target reached", "spend", "stopped at checkpoint"):
            assert phrase in result["summary"].lower(), width


def test_replay_uses_composer_for_command_and_pickers(built_site) -> None:
    results = _probe(
        [320, 1000],
        _CLOCKED
        + """
await card.locator('.terminal-composer').scrollIntoViewIfNeeded();
const control = card.locator('.terminal-control');
const controlLabel = () => control.getAttribute('aria-label');
// Autoplay is IntersectionObserver-driven, not clock-driven. Converge to the
// settled state without assuming it fired: Replay starts, Stop finishes.
for (let attempt = 0; attempt < 50; attempt += 1) {
  const state = await controlLabel();
  await control.click();
  if (state === 'Stop animation') break;
}
await stepUntil(
  () => controlLabel().then(value => value === 'Replay demo'),
  'settled replay control',
);
const stopped = await demo.innerText();
await card.evaluate(card => {
  const input = card.querySelector('.input-text');
  const command = input.closest('.terminal-input').dataset.command;
  const pickers = [...card.querySelectorAll('.workflow-picker')];
  let lastInput = '';
  window.workflowFrames = [];
  // Object method, not a named arrow: tsx keep-names rewrites named arrows into
  // __name(...) calls that cannot cross into the page.
  const recorder = { record() {
    const text = input.textContent;
    let fits = true;
    if (text !== lastInput && text === command) {
      const line = input.parentElement.getBoundingClientRect();
      const range = document.createRange();
      range.selectNodeContents(input);
      fits = [...range.getClientRects()].every(rect =>
        rect.left >= line.left - 1 && rect.right <= line.right + 1);
    }
    lastInput = text;
    window.workflowFrames.push({
      input: text,
      inputVisible: input.getClientRects().length > 0,
      fits,
      pickers: pickers.map(picker => ({
        visible: !picker.hidden,
        keyVisible: !picker.hidden
          && !picker.querySelector('.workflow-key')?.hidden,
        options: [...picker.querySelectorAll('[data-option]')]
          .map(option => option.textContent),
      })),
    });
  } };
  new MutationObserver(recorder.record).observe(input, {
    childList: true, characterData: true,
  });
  for (const picker of pickers) {
    new MutationObserver(recorder.record).observe(picker, {
      subtree: true, childList: true, characterData: true, attributes: true,
    });
  }
});
await control.click();
const pickers = [];
for (const heading of ['Cost target (USD)', 'Time target']) {
  const picker = card.locator('.workflow-picker').filter({ hasText: heading });
  await stepUntil(() => picker.isVisible(), `${heading} picker`);
  const before = await picker.evaluate(el => {
    const rect = el.getBoundingClientRect();
    return { width: rect.width, height: rect.height };
  });
  await stepUntil(
    () => picker.locator('.workflow-key').isVisible(),
    `${heading} confirmation key`,
  );
  pickers.push(await picker.evaluate((el, before) => {
    const card = el.closest('.terminal-card');
    const viewport = card.querySelector('.terminal-viewport').getBoundingClientRect();
    const bounds = card.getBoundingClientRect();
    const rect = el.getBoundingClientRect();
    const input = card.querySelector('.terminal-input');
    return {
      belowTranscript: rect.top >= viewport.bottom - 1,
      insideCard: rect.left >= bounds.left && rect.right <= bounds.right
        && rect.bottom <= bounds.bottom,
      outsideTranscript: !card.querySelector('.terminal-viewport').contains(el),
      composerHidden: input.getClientRects().length === 0,
      confirmationKeepsGeometry: Math.abs(rect.width - before.width) < 1
        && Math.abs(rect.height - before.height) < 1,
    };
  }, before));
}
await stepUntil(
  () => demo.getByText('/detect-tests', { exact: true }).isVisible(),
  'first workflow step',
);
return {
  frames: await page.evaluate(() => window.workflowFrames), pickers, stopped,
  transcript: await demo.innerText(),
  composerRestored: await card.locator('.terminal-input').isVisible(),
  stopping: await card.getByRole('button', { name: /stop animation/i }).isVisible(),
};
""",
    )
    command = "/workflow implement-spec specs/checkout.md"
    picker_options = (
        ("Cost target (USD)", ("$1", "$5", "$10", "No target", "Custom…")),
        ("Time target", ("1 hour", "4 hours", "8 hours", "No target", "Custom…")),
    )
    for width, result in results.items():
        typed = [frame["input"] for frame in result["frames"] if frame["inputVisible"]]
        assert command in typed, width
        assert any(
            frame["input"] == command and frame["fits"] for frame in result["frames"]
        ), width
        assert (
            len({value for value in typed if value and command.startswith(value)}) > 2
        )
        assert "stopped at checkpoint" in result["stopped"].lower(), width
        assert "stopped at checkpoint" not in result["transcript"].lower(), width
        assert result["composerRestored"] and result["stopping"], width
        assert all(all(picker.values()) for picker in result["pickers"]), width

        for index, (heading, options) in enumerate(picker_options):
            frames = [
                frame for frame in result["frames"]
                if frame["pickers"][index]["visible"]
            ]
            assert any(
                frame["pickers"][index]["keyVisible"] for frame in result["frames"]
            ), f"{width}: {heading} did not show Enter confirmation"
            selections = {
                re.sub(r"^\s*❯\s*", "", option)
                for frame in frames
                for option in frame["pickers"][index]["options"]
                if option.lstrip().startswith("❯")
            }
            assert len(selections & set(options)) >= 2, (
                f"{width}: {heading} selection did not move"
            )


def test_live_effect_has_reserved_line_without_shifting_status(
    animated_run,
) -> None:
    for width, result in animated_run.items():
        empty, short, _long = result["effectStates"]
        assert all(state["visible"] for state in result["effectStates"]), width
        assert empty["effect"]["height"] > 0, width
        for state in result["effectStates"]:
            assert state["status"] == empty["status"], width
            assert state["metrics"] == empty["metrics"], width
            assert state["effect"] == empty["effect"], width
        effect = short["effect"]
        assert (
            effect["y"]
            >= max(metric["y"] + metric["height"] for metric in short["metrics"]) - 1
        ), width
        assert effect["x"] >= 0, width
        assert effect["x"] + effect["width"] <= short["status"]["width"] + 1, width


def test_mobile_footer_keeps_warning_visible(animated_run) -> None:
    result = animated_run["320"]
    for state in ("baseline", "reached"):
        assert "⏱" in result[state] and "$" in result[state]
    assert "cost target reached" in result["warningGeometry"]["text"].lower()
    assert result["warningGeometry"]["fits"]
    assert result["warningGeometry"]["width"] > 100
