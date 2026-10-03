/** Shared timing constants for the hero terminal run.
 * Single source of truth: pacing, spinner cadence and the follow threshold live
 * here so the session, viewport and run modules cannot drift apart. Values are
 * byte-stable — the shipped run timing is a tested contract. */

/** Every step in the demo sequence is paced by this multiplier, so the whole run
 * can be retimed in one place. 3.375 = 50% slower than the 2.25 baseline pacing. */
export const SPEED = 3.375;

/** Prompt typing is slowed further than the agent's own stream so the human
 * question reads as it types: the prompt types at TYPE_SLOWDOWN × the agent
 * rate — people type deliberately, models stream fast. */
export const TYPE_SLOWDOWN = 2;

/** Per-character agent-stream rate (ms) — also the `charDelay` option default.
 * Scaled by SPEED; the prompt types TYPE_SLOWDOWN × slower. Baseline beats
 * (ms): send / tool-call / answer, scaled by SPEED. The run lasts ≈19 s — past
 * WCAG 2.2.2's five-second window — so the chrome's Stop control is what
 * satisfies 2.2.2: start 2025 + prompt typing ≈ 2205 + send 405 + 4 call rows ×
 * 2700 (agent typing overlaps the 2700 ms working beat) + answer 608 + answer
 * typing ≈ 3305 ≈ 19.3 s. */
export const CHAR_DELAY = 9.6 * SPEED;
export const SEND_PAUSE = 120 * SPEED;
export const WORKING_PAUSE = 800 * SPEED;
export const ANSWER_DELAY = 180 * SPEED;
export const START_DELAY = 600 * SPEED;
export const LOOP_DELAY = 3000 * SPEED;

/** The braille cadence every terminal agent cycles while a tool runs. A frozen
 * `⠋` reads as a stray dot; cycling the frames is what makes the row read as a
 * live indicator instead of a static glyph. */
export const SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/** Frame duration (ms) for the working spinner. Deliberately NOT scaled by
 * SPEED: spinner cadence is a UX constant — it must feel live whatever the
 * demo pacing is. 135ms = 50% slower than the 90ms terminal baseline, so a
 * WORKING_PAUSE beat shows ~6-7 frames. */
export const SPINNER_TICK = 135;

/** Slack (px) for the "am I at the tail?" test. A finger-flick rarely lands
 * exactly at the bottom, so the chat-standard 10px threshold is too tight on
 * touch; 24px still registers a deliberate scroll-back as "off the tail". */
export const FOLLOW_THRESHOLD = 24;

export function defaultSleep(ms: number): Promise<void> {
    return new Promise((resolve) => window.setTimeout(resolve, ms));
}
