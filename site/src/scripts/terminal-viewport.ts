import type { TerminalSession } from "./terminal-session";
import { appendStaticStep, STEPS } from "./terminal-dom";

/**
 * Viewport geometry for the hero terminal: the settled-height lock (the card
 * never grows; the transcript bottom-anchors and clips at the top like a real
 * harness scrollback) plus card visibility. Operates on a TerminalSession.
 */

export function isCardVisible(element: Element): boolean {
    if (typeof window === "undefined") {
        return true;
    }
    const rect = element.getBoundingClientRect();
    const viewportHeight = window.innerHeight || 0;
    const visibleHeight = Math.min(rect.bottom, viewportHeight) - Math.max(rect.top, 0);
    return visibleHeight >= rect.height * 0.5;
}

/** Build the off-screen geometry probe: a clone of the live container holding
 * the full static transcript, measured at the run's own inline size. Returns
 * null when there is no width to measure against yet. */
function buildMeasureProbe(session: TerminalSession): HTMLElement | null {
    // Measure against the live run's own inline size, not the viewport's:
    // the run is capped at its own measure, so a clone measured at the
    // viewport's width would wrap less and under-lock the box, clipping the
    // tail of the transcript it is meant to protect.
    const width = session.terminal.clientWidth;
    if (!width) {
        return null;
    }
    // Geometry probe must be a clone of the live container, not a bare div:
    // it has to keep the class + Astro scope attribute so the scoped
    // `.terminal-lines` flex rules and monospace metrics apply. A plain div
    // under-measures (collapsed margins, wrong font) and clips the first
    // line once the transcript overflows.
    const measure = session.terminal.cloneNode(false) as HTMLElement;
    measure.removeAttribute("id");
    // Class-tagged so tests can stub its geometry at the true boundary.
    measure.classList.add("terminal-measure");
    styleMeasureProbe(measure, width);
    return measure;
}

/** Off-screen + fixed-width styling for the probe. Split from the builder
 * so neither unit grows past the composition limit. */
function styleMeasureProbe(measure: HTMLElement, width: number): void {
    measure.style.position = "absolute";
    measure.style.visibility = "hidden";
    measure.style.pointerEvents = "none";
    measure.style.inset = "0 auto auto 0";
    measure.style.width = `${width}px`;
    measure.style.maxInlineSize = "none";
}

/** Reserve the settled transcript height on the viewport so the card never
 * grows: the transcript is bottom-anchored and older lines clip at the
 * top, exactly like a real harness scrollback. */
export function lockViewportHeight(session: TerminalSession): void {
    const measure = buildMeasureProbe(session);
    if (!measure) {
        return;
    }
    measure.style.height = "auto";
    measure.style.minHeight = "0";
    measure.setAttribute("aria-hidden", "true");
    STEPS.forEach((step) => appendStaticStep(measure, session.doc, step));
    session.scrollViewport.appendChild(measure);
    const height = Math.ceil(measure.getBoundingClientRect().height);
    measure.remove();
    if (height > 0) {
        session.scrollViewport.style.height = `${height}px`;
    }
    session.syncScrollable();
}

export function runAfterLayout(callback: () => void): void {
    if (typeof window.requestAnimationFrame === "function") {
        window.requestAnimationFrame(() => {
            callback();
        });
        return;
    }
    window.setTimeout(callback, 0);
}

export function scheduleSettledHeightLock(session: TerminalSession): void {
    if (session.settledHeightLockPending) {
        return;
    }
    session.settledHeightLockPending = true;
    runAfterLayout(() => {
        session.settledHeightLockPending = false;
        lockViewportHeight(session);
    });
    const fonts = (session.doc as Document & {
        fonts?: { ready?: Promise<unknown> };
    }).fonts;
    void fonts?.ready?.then(() => {
        lockViewportHeight(session);
    });
}
