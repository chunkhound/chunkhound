import { createSession } from "./terminal-session";
import { startTerminal } from "./terminal-run";
import type { HeroTerminalOptions } from "./terminal-session";

/**
 * Hero terminal entry point (thin by design).
 *
 * The demo story lives in hero-transcript.ts so Hero.astro can server-render
 * the same transcript for no-JS visitors. Animation model = real harness
 * behaviour: one prompt types into the fixed bottom input → send → it lands
 * in the scrollback → the agent fans out three ChunkHound calls — each agent
 * row opens, its ChunkHound row follows right after it and blinks while the
 * call is in flight (both rows highlighted together, and that beat's note
 * opens with them) — then the cited receipts land and the agent settles →
 * the agent synthesizes → one verdict.
 *
 * Modules: terminal-timing (constants) · terminal-dom (builders) ·
 * terminal-session (state + beat-marking unit) · terminal-viewport (height
 * lock) · terminal-run (typing engine + orchestration).
 */

export type { HeroTerminalOptions };

export function initHeroTerminal(
    doc: Document = document,
    options: HeroTerminalOptions = {},
): void {
    const session = createSession(doc, options);
    if (!session) {
        return;
    }
    session.trackReducedMotion();
    startTerminal(session);
}

if (typeof document !== "undefined") {
    initHeroTerminal();
}
