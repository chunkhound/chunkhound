import type { TerminalSession } from "./terminal-session";
import { SPINNER_FRAMES, SPINNER_TICK } from "./terminal-timing";

/** The one typing engine for both terminal runs: timing is always `session.sleep`,
 * so the page, a fake-clock test, and reduced motion share one code path. */

/** Animate a running tool call: cycle the braille frames for `duration` ms,
 * splitting the wait into frame-sized sleeps so the injected `sleep` (real
 * timer in the page, fake in tests) still owns all timing. Bails as soon as
 * `abort()` flips so stop / reduced-motion / off-screen can fast-forward. */
export async function spinGlyph(
    session: TerminalSession,
    glyph: HTMLElement,
    duration: number,
    abort: () => boolean,
): Promise<void> {
    let frame = 0;
    for (let elapsed = 0; elapsed < duration; elapsed += SPINNER_TICK) {
        // Teardown first: destroy() sets stopRequested, so no branch below
        // may write a detached glyph.
        if (session.destroyed) {
            return;
        }
        glyph.textContent = `${SPINNER_FRAMES[frame % SPINNER_FRAMES.length]} `;
        await session.sleep(Math.min(SPINNER_TICK, duration - elapsed));
        frame += 1;
        if (abort()) {
            return;
        }
    }
}

/** The one character emitter behind every typed surface: the prompt input and
 * the agent's streamed output. Reduced motion or a stop ships the full text
 * at once; going off-screen aborts mid-way and leaves the partial text. The
 * injected `sleep` owns all timing, so reduced motion still sleeps zero
 * times. `onChar` lets a caller react per character (advance a spinner frame,
 * keep the row pinned). */
export async function typeText(
    session: TerminalSession,
    target: HTMLElement,
    text: string,
    delay: number,
    abort: () => boolean,
    onChar?: () => void,
): Promise<void> {
    for (const ch of text) {
        // Teardown first: destroy() also sets stopRequested, so the fast path
        // below would otherwise ship the full text into a dead target.
        if (session.destroyed) {
            return;
        }
        if (session.reducedMotion || session.stopRequested) {
            target.textContent = text;
            return;
        }
        if (abort()) {
            return;
        }
        target.textContent += ch;
        onChar?.();
        await session.sleep(delay);
    }
}

/** How many characters stream between two braille frames, so the working row
 * keeps ticking while its text types. A non-positive delay has no cadence to
 * divide by, so it falls back to a small fixed group. */
export function charsPerFrame(delay: number): number {
    return delay > 0 ? Math.max(1, Math.round(SPINNER_TICK / delay)) : 4;
}
