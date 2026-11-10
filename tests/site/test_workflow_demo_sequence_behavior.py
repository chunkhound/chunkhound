# ruff: noqa: E501  # Embedded JavaScript keeps the choreography readable.
"""Deterministic picker choreography for the illustrative workflow run."""

from itertools import groupby

from tests.site.dom_helpers import browser_dom, dist_body
from tests.site.tsx_runner import run_tsx_json

_WORKFLOWS = "docs/workflows/index.html"


def test_picker_choreography_reveals_each_confirmation_once_and_replays(
    built_site,
) -> None:
    """The picker sequence is an observable-state contract.

    Test it with an injected sleep queue, not a browser clock: each beat
    records the real DOM state. Sleeps are implementation detail, so repeated
    snapshots collapse. A skipped state or a replay without a reset fails.
    """
    script = (
        browser_dom(dist_body(_WORKFLOWS), expose_document=False)
        + """
const doc = window.document;
const card = doc.querySelector('.workflow-demo-lines').closest('.terminal-card');
const { initWorkflowDemo } = await import('./site/src/scripts/workflow-demo.ts');

const runs = [];
const sleep = async () => {
  runs.push([...card.querySelectorAll('.workflow-picker')].map(picker => ({
    hidden: picker.hidden,
    keyHidden: picker.querySelector('.workflow-key').hidden,
    options: [...picker.querySelectorAll('[data-option]')]
      .map(option => option.textContent.trim()),
  })));
};
initWorkflowDemo(card, { charDelay: 0, sendPause: 0, sleep, reducedMotion: false });
FakeIntersectionObserver.instances.at(-1).callback([{ isIntersecting: true }]);
const tick = () => new Promise(resolve => setTimeout(resolve, 0));
for (let i = 0; i < 120; i += 1) await tick();

const afterRun = {
  settled: card.classList.contains('is-settled'),
  control: card.querySelector('.terminal-control').getAttribute('aria-label'),
  stepsVisible: [...card.querySelectorAll('.workflow-demo-step')]
    .every(step => !step.hidden),
  pickersHidden: [...card.querySelectorAll('.workflow-picker')]
    .every(picker => picker.hidden),
};
const firstPass = runs.splice(0);

card.querySelector('.terminal-control').click();
for (let i = 0; i < 120; i += 1) await tick();
const afterReplay = {
  settled: card.classList.contains('is-settled'),
  control: card.querySelector('.terminal-control').getAttribute('aria-label'),
};
console.log(JSON.stringify({ afterRun, afterReplay, firstPass, replayPass: runs }));
"""
    )
    rendered = run_tsx_json(script)

    assert rendered["afterRun"] == {
        "settled": True,
        "control": "Replay demo",
        "stepsVisible": True,
        "pickersHidden": True,
    }
    assert rendered["afterReplay"] == {"settled": True, "control": "Replay demo"}

    for states in (rendered["firstPass"], rendered["replayPass"]):
        # Collapse repeated snapshots: sleep calls are not a user-facing contract.
        # Picker beats are the remaining states with a visible picker.
        picker_states = [
            state for state, _ in groupby(states)
            if any(not picker["hidden"] for picker in state)
        ]
        assert len(picker_states) == 8
        for index in range(2):
            picker = picker_states[index * 4 : (index + 1) * 4]
            # Show, select 1, select 0, reveal. The key is hidden until the last beat.
            assert [state[index]["keyHidden"] for state in picker] == [
                True,
                True,
                True,
                False,
            ]
            assert picker[0][index]["hidden"] is False
            assert picker[3][index]["hidden"] is False
            assert picker[0][index]["options"][0].startswith("❯")
            assert picker[1][index]["options"][1].startswith("❯")
            assert picker[2][index]["options"][0].startswith("❯")

def test_late_visibility_after_manual_stop_does_not_restart_playback(built_site) -> None:
    """A manual toggle owns the run: a late visibility callback must not autoplay a stopped run."""
    script = browser_dom(dist_body(_WORKFLOWS), expose_document=False) + """
const doc = window.document;
const card = doc.querySelector('.workflow-demo-lines').closest('.terminal-card');
const control = () => card.querySelector('.terminal-control').getAttribute('aria-label');
const { initWorkflowDemo } = await import('./site/src/scripts/workflow-demo.ts');
let beats = 0;
initWorkflowDemo(card, { charDelay: 0, sendPause: 0, sleep: async () => { beats += 1; }, reducedMotion: false });
const observer = FakeIntersectionObserver.instances.at(-1);
card.querySelector('.terminal-control').click();
const during = control();
card.querySelector('.terminal-control').click();
const stop = { control: control(), beats };
observer.callback([{ isIntersecting: true }]);
const afterLate = control();
for (let i = 0; i < 120; i += 1) await new Promise((resolve) => setTimeout(resolve, 0));
console.log(JSON.stringify({ during, stop, afterLate, restartedBeats: beats - stop.beats, control: control(),
  settled: card.classList.contains('is-settled'), stepsVisible: [...card.querySelectorAll('.workflow-demo-step')].every((step) => !step.hidden) }));
"""
    rendered = run_tsx_json(script)

    assert (rendered["during"], rendered["stop"]["control"], rendered["afterLate"], rendered["restartedBeats"]) == ("Stop animation", "Replay demo", "Replay demo", 0)
    assert (rendered["control"], rendered["settled"], rendered["stepsVisible"]) == ("Replay demo", True, True)
