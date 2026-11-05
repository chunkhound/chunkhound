import type { TerminalSession } from "./terminal-session";

export function setControl(session: TerminalSession, mode: "stop" | "replay"): void {
    const control = session.card.querySelector(".terminal-control");
    if (!control) return;
    control.setAttribute("data-state", mode);
    const label = control.querySelector(".terminal-control-label");
    if (label) label.textContent = mode === "stop" ? "Stop" : "Replay";
    control.setAttribute("aria-label", mode === "stop" ? "Stop animation" : "Replay demo");
}

/** Bind one named click handler and own its removal, so destroy() can detach
 * the exact function even after the card left the document. */
function wireClick(
    session: TerminalSession,
    selector: string,
    handler: () => void,
): void {
    const element = session.card.querySelector(selector);
    if (!element) {
        return;
    }
    element.addEventListener("click", handler);
    session.onCleanup(() => element.removeEventListener("click", handler));
}

export function wireControls(session: TerminalSession, toggle: () => void): void {
    wireClick(session, ".terminal-control", toggle);
    wireClick(session, ".terminal-jump", () => {
        session.setFollowing(true);
        session.followTail();
    });
    const onScroll = (): void => {
        session.setFollowing(session.atTail());
    };
    session.scrollViewport.addEventListener("scroll", onScroll, { passive: true });
    session.onCleanup(() => {
        session.scrollViewport.removeEventListener("scroll", onScroll);
    });
}

export function observeVisibility(
    session: TerminalSession,
    change: (visible: boolean) => void,
    target: Element = session.card,
): void {
    const observer = new IntersectionObserver(([entry]) => {
        // A queued delivery can outlive disconnect(); destroyed is the guard.
        if (session.destroyed) {
            return;
        }
        change(entry?.isIntersecting ?? false);
    }, { threshold: 0.5 });
    observer.observe(target);
    session.onCleanup(() => observer.disconnect());
}
