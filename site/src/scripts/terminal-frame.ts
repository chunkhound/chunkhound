import { observeVisibility, wireControls } from "./terminal-controls";
import type { TerminalSession } from "./terminal-session";
import { isCardVisible, keepViewportHeightLocked, lockViewportHeight, scheduleSettledHeightLock } from "./terminal-viewport";

/** Entry points keep their own render/toggle/visibility behavior. */
export interface TerminalFrameHooks {
    render: (measure: HTMLElement) => void;
    toggle: () => void;
    onMotion?: (reduced: boolean) => void;
    onVisibility?: (visible: boolean) => void;
    visibilityTarget?: Element;
}

/** Single owner of the boot order: visibility snapshot, height locks, follow,
 * controls, reduced motion, viewport observer, and pagehide teardown. A real
 * unload destroys the session; bfcache restore keeps it for pageshow. */
export function bootTerminalFrame(session: TerminalSession, hooks: TerminalFrameHooks): void {
    session.isVisible = isCardVisible(session.card);
    lockViewportHeight(session, hooks.render);
    scheduleSettledHeightLock(session, hooks.render);
    keepViewportHeightLocked(session, hooks.render);
    session.syncFollow();
    wireControls(session, hooks.toggle);
    session.trackReducedMotion(hooks.onMotion);
    observeVisibility(session, (visible) => {
        session.isVisible = visible;
        hooks.onVisibility?.(visible);
    }, hooks.visibilityTarget ?? session.card);
    session.listen(window, "pagehide", (event) => {
        if (!(event as PageTransitionEvent).persisted) session.destroy();
    });
}
