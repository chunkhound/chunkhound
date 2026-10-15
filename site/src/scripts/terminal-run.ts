import type { HeroSource } from "./hero-transcript";
import type { TerminalSession } from "./terminal-session";
import {
    isCardVisible,
    lockViewportHeight,
    runAfterLayout,
    scheduleSettledHeightLock,
} from "./terminal-viewport";
import {
    SPINNER_FRAMES,
    SPINNER_TICK,
    TYPE_SLOWDOWN,
} from "./terminal-timing";
import {
    STEPS,
    appendAnswer,
    appendCall,
    appendEvidenceItem,
    appendNote,
    appendSources,
    appendStaticStep,
    appendSubmittedPrompt,
    createAnchor,
    noteForStep,
    settleSources,
    sourceText,
    type HeroReceipt,
    type HeroStep,
} from "./terminal-dom";

/**
 * Render orchestration for the hero terminal: the typing engine, per-step
 * renderers, the run loop, and event wiring. Every function takes the session
 * explicitly and stays small; behavior (timing, abort conditions, DOM) is
 * identical to the original `initHeroTerminal` closures.
 */

// --- Entrances ---

/** Bottom-entry push-up bookkeeping, phase one: snapshot the stack's top edge
 * before a batch lands. Returns null when the effect must not run (reduced
 * motion, off-screen, or no rows yet); `pushEnd` inverts the displacement. */
export function pushBegin(session: TerminalSession): number | null {
    const first = session.terminal.firstElementChild;
    if (session.reducedMotion || !session.shouldAnimate() || !first) {
        return null;
    }
    return first.getBoundingClientRect().top;
}

export function pushEnd(session: TerminalSession, before: number | null): boolean {
    const dy = pendingPushOffset(session, before);
    if (dy === null || dy <= 1) {
        return false;
    }
    // Pin the pre-push position, flush it, then let the CSS transition ride
    // the transform home.
    session.terminal.style.transition = "none";
    session.terminal.style.transform = `translateY(${dy}px)`;
    void session.terminal.getBoundingClientRect();
    session.terminal.style.transition = "";
    session.terminal.style.transform = "";
    return true;
}

/** The visual displacement a batch added: layout shift while the transcript
 * fits, pending tail scroll once it overflows. Null when no glide applies. */
function pendingPushOffset(session: TerminalSession, before: number | null): number | null {
    const first = session.terminal.firstElementChild;
    if (before === null || !first) {
        return null;
    }
    // The append also advances the tail scroll when the transcript overflows
    // its box (the mobile clamp): the first child's own layout does not move,
    // so the rect delta alone is zero there. Include the pending follow so
    // the same inversion compensates either regime.
    const followScroll = session.following
        ? session.layoutScrollSpan() - session.scrollViewport.scrollTop
        : 0;
    return before - first.getBoundingClientRect().top + followScroll;
}

/** Phase two of a new row's entrance: after the push has made room, the row
 * fades in and lifts the last few pixels into place (CSS, held off by the
 * push's length). A data-* marker — never a class — keeps the shipped
 * transcript class contract stable; `solo` drops the delay when there was no
 * room to make (the run's first row). The marker is dropped once the fade
 * lands, and the static / reduced-motion render never gets one. */
export function enterRow(session: TerminalSession, row: HTMLElement | null, delayed: boolean): void {
    if (!row || session.reducedMotion || !session.shouldAnimate()) {
        return;
    }
    row.dataset.enter = delayed ? "" : "solo";
    row.addEventListener("animationend", (event) => {
        if ((event as AnimationEvent).animationName === "hero-row-in") {
            row.removeAttribute("data-enter");
        }
    });
}

// --- Typing engine ---

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

/** The engine's cited return, typed after the agent's call text and before
 * the agent settles: each citation streams into the list the working row
 * reserved. Text is never typed concurrently with the agent — only the two
 * progress indicators (spinner + dot) stay live while this runs. */
export async function typeSources(
    session: TerminalSession,
    receipt: HeroReceipt,
    sources: HeroSource[],
    delay: number,
    abort: () => boolean,
): Promise<void> {
    const doc = receipt.list.ownerDocument;
    for (const source of sources) {
        const item = appendEvidenceItem(receipt.list, doc);
        await typeText(session, item, sourceText(source), delay, abort, () =>
            session.followTail());
        if (session.reducedMotion || session.stopRequested) {
            return;
        }
    }
}

// --- Steps ---

/** Reduced-motion fast-forward: the full static transcript at once, input
 * cleared, zero sleeps. The run reads as complete, so the composer is
 * already submitted. */
export function renderStatic(session: TerminalSession): void {
    session.clearTerminal();
    for (const step of STEPS) {
        appendStaticStep(session.terminal, session.doc, step);
    }
    session.card.dataset.composer = "sent";
    session.followTail();
}

/** Send beat: the typed turn leaves the fixed input and lands in the
 * scrollback; the field clears and recedes so the run owns attention. */
function submitPrompt(session: TerminalSession, step: HeroStep): void {
    const push = pushBegin(session);
    const line = appendSubmittedPrompt(session.terminal, session.doc, step.text);
    session.markCurrent(line, step.beat);
    session.clearInput();
    // The submit beat: the field recedes and the caret goes dark from
    // here on, so the run owns attention.
    session.card.dataset.composer = "sent";
    enterRow(session, line, pushEnd(session, push));
    session.followTail();
}

async function renderPromptStep(session: TerminalSession, step: HeroStep): Promise<void> {
    await typeText(session, session.input, step.text,
        session.charDelay * TYPE_SLOWDOWN, () => !session.shouldAnimate(),
        () => session.pinInputCaret());
    if (session.reducedMotion || session.stopRequested || !session.shouldAnimate()) {
        return;
    }
    await session.sleep(session.sendPause);
    if (!session.shouldAnimate()) {
        return;
    }
    submitPrompt(session, step);
}

interface OpenBeat {
    anchor: HTMLElement | null;
    call: { line: HTMLDivElement; glyph: HTMLSpanElement; text: HTMLSpanElement };
    receipt: HeroReceipt | null;
    note: HTMLElement | null;
}

/** Open one call beat as a unit: the agent row working, then the engine's
 * receipt row, then the note that annotates them — so the call is already in
 * flight while the agent keeps working. Both rows highlight together and the
 * note opens with them. One push per beat so the note block never fires
 * competing shifts. */
function openCallBeat(session: TerminalSession, step: HeroStep & { type: "call" }): OpenBeat {
    const sources = step.sources;
    const noteRef = sources ? noteForStep(step) : undefined;
    const push = pushBegin(session);
    const anchor = sources ? createAnchor(session.terminal, session.doc) : null;
    const call = appendCall(anchor ?? session.terminal, session.doc, step.text, false);
    const receipt = anchor && sources ? appendSources(anchor, session.doc, sources, true) : null;
    const note = anchor && noteRef ? appendNote(anchor, session.doc, noteRef) : null;
    // The whole beat — agent row, receipt and its note — enters as one
    // unit in phase two, so the note draws in with the rows it annotates.
    session.markCurrent(receipt ? [call.line, receipt.line] : call.line, step.beat, note);
    call.text.textContent = "";
    enterRow(session, anchor ?? call.line, pushEnd(session, push));
    session.followTail();
    return { anchor, call, receipt, note };
}

/** Stream the agent's call text, ticking the braille spinner as it types.
 * Returns the frames already shown, so the receipt wait only covers the rest. */
async function streamCallText(
    session: TerminalSession,
    beat: OpenBeat,
    step: HeroStep & { type: "call" },
): Promise<number> {
    const perFrame = charsPerFrame(session.charDelay);
    let typed = 0;
    let frames = 0;
    await typeText(session, beat.call.text, step.text, session.charDelay,
        () => !session.shouldAnimate(),
        () => {
            typed += 1;
            if (typed % perFrame === 0) {
                frames += 1;
                beat.call.glyph.textContent = `${SPINNER_FRAMES[frames % SPINNER_FRAMES.length]} `;
            }
            session.followTail();
        });
    return frames;
}

/** Run the receipt typing against the remaining working beat: the agent's
 * spinner keeps cycling and the ChunkHound dot keeps blinking (CSS) while
 * the citations land, so both progress indicators stay live until then. */
async function awaitBeatReceipt(
    session: TerminalSession,
    beat: OpenBeat,
    step: HeroStep & { type: "call" },
    frames: number,
): Promise<void> {
    const abort = () => session.reducedMotion || session.stopRequested || !session.shouldAnimate();
    const remaining = Math.max(0, Math.ceil(session.workingPause / SPINNER_TICK) - frames);
    const receiptTyping = beat.receipt && step.sources
        ? typeSources(session, beat.receipt, step.sources, session.charDelay, abort)
        : Promise.resolve();
    const typedMs = (step.sources ?? []).reduce(
        (total, source) => total + sourceText(source).length, 0,
    ) * session.charDelay;
    await Promise.all([
        spinGlyph(session, beat.call.glyph,
            Math.max(remaining * SPINNER_TICK, typedMs), abort),
        receiptTyping,
    ]);
}

/** Settle one beat: the receipt dot becomes the marker, then the agent
 * settles. Both rows stay marked current until the next beat fires. */
function settleBeat(session: TerminalSession, beat: OpenBeat): void {
    if (beat.receipt) {
        settleSources(beat.receipt);
    }
    beat.call.glyph.textContent = "⏺ ";
    beat.call.line.classList.remove("is-working");
    session.followTail();
}

async function renderCallStep(session: TerminalSession, step: HeroStep & { type: "call" }): Promise<void> {
    if (session.reducedMotion || !session.shouldAnimate()) {
        return;
    }
    const beat = openCallBeat(session, step);
    const frames = await streamCallText(session, beat, step);
    if (session.reducedMotion || !session.shouldAnimate()) {
        return;
    }
    // The agent's call text is done; the engine's receipt now types in
    // turn (text is never concurrent). The agent's spinner keeps cycling
    // and the ChunkHound dot keeps blinking (CSS) while that happens, so
    // both progress indicators stay live until the receipt lands.
    await awaitBeatReceipt(session, beat, step, frames);
    if (session.reducedMotion || !session.shouldAnimate()) {
        return;
    }
    // The call returns: the receipt settles, then the agent settles. Both
    // rows stay marked current until the next beat fires.
    settleBeat(session, beat);
}

async function renderAnswerStep(session: TerminalSession, step: HeroStep): Promise<void> {
    // Answer: the agent streams its verdict, distilled from the evidence.
    if (!session.reducedMotion) {
        await session.sleep(session.answerDelay);
    }
    if (session.reducedMotion || !session.shouldAnimate()) {
        return;
    }
    const push = pushBegin(session);
    const result = appendAnswer(session.terminal, session.doc, step.text);
    session.markCurrent(result.parentElement, step.beat);
    result.textContent = "";
    enterRow(session, result.parentElement, pushEnd(session, push));
    await typeText(session, result, step.text, session.charDelay,
        () => !session.shouldAnimate(), () => session.followTail());
    session.followTail();
}

async function renderStep(session: TerminalSession, step: HeroStep): Promise<void> {
    if (step.type === "prompt") {
        await renderPromptStep(session, step);
    } else if (step.type === "call") {
        await renderCallStep(session, step);
    } else {
        await renderAnswerStep(session, step);
    }
}

async function renderOnce(session: TerminalSession): Promise<void> {
    session.clearTerminal();
    for (const step of STEPS) {
        if (!(await runStepGuarded(session, step))) {
            return;
        }
    }
}

/** One guarded step: fast-forward to static on stop/reduced-motion, wipe on
 * hide, else render. Returns false when the run must not continue. */
async function runStepGuarded(session: TerminalSession, step: HeroStep): Promise<boolean> {
    if (session.reducedMotion || session.stopRequested) {
        renderStatic(session);
        return false;
    }
    if (!session.shouldAnimate()) {
        session.clearTerminal();
        return false;
    }
    await renderStep(session, step);
    if (session.reducedMotion || session.stopRequested) {
        renderStatic(session);
        return false;
    }
    if (!session.shouldAnimate()) {
        session.clearTerminal();
        return false;
    }
    return true;
}

/** One-time beat before the first run, so the static hero copy lands first.
 * Skipped under reduced motion (which must sleep zero times) and after the
 * first run; user-initiated replay never waits. */
async function startGate(session: TerminalSession): Promise<void> {
    if (!session.startGatePending) {
        return;
    }
    session.startGatePending = false;
    if (!session.reducedMotion) {
        await session.sleep(session.startDelay);
    }
}

// --- Run control ---

/** The chrome control is context-aware: "Stop" while a run is in flight,
 * "Replay" once it settles. It is the WCAG 2.2.2 pause/stop mechanism for
 * the auto-playing run (which exceeds five seconds). The icon swaps with the
 * label via `data-state`; the visible label is the accessible text. */
function setControl(session: TerminalSession, mode: "stop" | "replay"): void {
    const control = session.card.querySelector(".terminal-control");
    if (!control) {
        return;
    }
    control.setAttribute("data-state", mode);
    const label = control.querySelector(".terminal-control-label");
    if (label) {
        label.textContent = mode === "stop" ? "Stop" : "Replay";
    }
    control.setAttribute("aria-label", mode === "stop" ? "Stop animation" : "Replay demo");
}

/** Reset per-run state and surface the stop affordance. The composer starts
 * every run live: the field is bright until its prompt is sent. */
function beginRun(session: TerminalSession): void {
    session.stopRequested = false;
    session.following = true;
    session.syncFollow();
    session.card.classList.remove("is-settled");
    session.card.dataset.composer = "live";
    setControl(session, "stop");
}

function markSettled(session: TerminalSession): void {
    if (!session.loop) {
        session.card.classList.add("is-settled");
        setControl(session, "replay");
    }
}

async function animateRun(session: TerminalSession): Promise<void> {
    await startGate(session);
    do {
        await renderOnce(session);
        if (!session.shouldAnimate()) {
            return;
        }
        if (!session.loop) {
            markSettled(session);
            return;
        }
        await session.sleep(session.loopDelay);
    } while (session.shouldAnimate());
}

/** User-initiated: re-run the transcript exactly once from a clean state,
 * regardless of `loop`, and without the entrance start gate. */
function replay(session: TerminalSession): void {
    if (session.isRendering) {
        return;
    }
    session.isRendering = true;
    beginRun(session);
    void (async () => {
        try {
            await renderOnce(session);
            markSettled(session);
        } finally {
            session.isRendering = false;
        }
    })();
}

function triggerRender(session: TerminalSession): void {
    if (!session.shouldAnimate() || session.isRendering) {
        return;
    }
    session.isRendering = true;
    beginRun(session);
    void (async () => {
        try {
            await animateRun(session);
        } finally {
            session.isRendering = false;
        }
    })();
}

// --- Wiring ---

/** One control, two intents: stop the in-flight run, or replay a settled one.
 * Plus the scrollback jump control that hands the tail back to the reader. */
function wireControls(session: TerminalSession): void {
    session.card.querySelector(".terminal-control")?.addEventListener("click", () => {
        if (session.isRendering) {
            session.stopRequested = true;
            return;
        }
        replay(session);
    });
    session.card.querySelector(".terminal-jump")?.addEventListener("click", () => {
        session.setFollowing(true);
        session.followTail();
    });
}

/** Scrollback follow: reading back off the tail hands the viewport to the
 * reader; the jump control hands it back. Passive — never blocks the scroll. */
function wireScrollFollow(session: TerminalSession): void {
    session.scrollViewport.addEventListener("scroll", () => {
        session.setFollowing(session.atTail());
    }, { passive: true });
}

function wireResize(session: TerminalSession): void {
    if (typeof ResizeObserver !== "undefined") {
        const resizeObserver = new ResizeObserver(() => {
            lockViewportHeight(session);
        });
        resizeObserver.observe(session.card);
    }
}

function wireVisibility(session: TerminalSession): void {
    const observer = new IntersectionObserver((entries) => {
        session.isVisible = entries[0]?.isIntersecting ?? false;
        if (!session.isVisible) {
            session.clearTerminal();
            return;
        }
        triggerRender(session);
    }, { threshold: 0.5 });
    observer.observe(session.card);
    wireDocumentVisibility(session);
}

/** Tab hide wipes the run; tab show re-locks the height and re-triggers. */
function wireDocumentVisibility(session: TerminalSession): void {
    session.doc.addEventListener("visibilitychange", () => {
        if (session.doc.hidden) {
            session.clearTerminal();
            return;
        }
        scheduleSettledHeightLock(session);
        runAfterLayout(() => {
            triggerRender(session);
        });
    });
}

function wirePageLifecycle(session: TerminalSession): void {
    window.addEventListener("pagehide", () => {
        session.isVisible = false;
        session.clearTerminal();
    });
    window.addEventListener("pageshow", () => {
        session.isVisible = isCardVisible(session.card);
        scheduleSettledHeightLock(session);
        runAfterLayout(() => {
            triggerRender(session);
        });
    });
}

/** Boot one terminal instance: wire everything, lock the viewport height,
 * then kick off the first gated run. */
export function startTerminal(session: TerminalSession): void {
    session.isVisible = isCardVisible(session.card);
    wireControls(session);
    wireScrollFollow(session);
    lockViewportHeight(session);
    scheduleSettledHeightLock(session);
    session.syncFollow();
    wireResize(session);
    wireVisibility(session);
    wirePageLifecycle(session);
    triggerRender(session);
}
