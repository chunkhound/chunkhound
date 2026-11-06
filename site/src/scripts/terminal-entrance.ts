import type { TerminalSession } from "./terminal-session";

/** Fade in a new row after any stack push; static and reduced-motion rows stay visible. */
export function enterRow(session: TerminalSession, row: HTMLElement | null, delayed: boolean): void {
    if (!row || session.reducedMotion || !session.shouldAnimate()) return;
    row.dataset.enter = delayed ? "" : "solo";
    const finish = (event: AnimationEvent) => {
        if (event.animationName === "hero-row-in") {
            row.removeAttribute("data-enter");
            row.removeEventListener("animationend", finish);
        }
    };
    row.addEventListener("animationend", finish);
}
