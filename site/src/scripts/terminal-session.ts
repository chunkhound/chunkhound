import { detectReducedMotion, onReducedMotionChange } from "./reduced-motion";
import type { HeroBeat } from "./hero-narrative";
import {
    ANSWER_DELAY,
    CHAR_DELAY,
    FOLLOW_THRESHOLD,
    LOOP_DELAY,
    SEND_PAUSE,
    START_DELAY,
    WORKING_PAUSE,
    defaultSleep,
} from "./terminal-timing";

/**
 * Mutable run state for one hero terminal instance. Owns the elements, the
 * resolved timing options, and the per-run narration/follow flags — the shared
 * state the old `initHeroTerminal` closures captured. Small methods compose;
 * the beat-marking unit (clear/set/mark) is the one invariant kept together:
 * a beat's rows and note are always raised/cleared as a unit.
 */

export interface HeroTerminalOptions {
    charDelay?: number;
    sendPause?: number;
    workingPause?: number;
    answerDelay?: number;
    loopDelay?: number;
    loop?: boolean;
    /** One-time beat before the first run so the static hero copy lands first. */
    startDelay?: number;
    /** Test hook: force the prefers-reduced-motion path regardless of media query. */
    reducedMotion?: boolean;
    sleep?: (ms: number) => Promise<void>;
}

export interface TerminalElements {
    terminal: HTMLElement;
    scrollViewport: HTMLElement;
    input: HTMLElement;
    inputLine: HTMLElement | null;
    card: HTMLElement;
    doc: Document;
}

export class TerminalSession {
    readonly terminal: HTMLElement;
    readonly scrollViewport: HTMLElement;
    readonly input: HTMLElement;
    readonly inputLine: HTMLElement | null;
    readonly card: HTMLElement;
    readonly doc: Document;
    charDelay!: number;
    sendPause!: number;
    answerDelay!: number;
    loopDelay!: number;
    startDelay!: number;
    readonly sleep: (ms: number) => Promise<void>;
    forcedReduced!: boolean | undefined;
    loopDefault!: boolean;
    workingPause!: number;
    reducedMotion!: boolean;
    loop!: boolean;
    currentRows: HTMLElement[] = [];
    currentNotes: HTMLElement[] = [];
    following = true;
    isVisible = false;
    isRendering = false;
    stopRequested = false;
    startGatePending = true;
    settledHeightLockPending = false;

    constructor(elements: TerminalElements, options: HeroTerminalOptions = {}) {
        this.terminal = elements.terminal;
        this.scrollViewport = elements.scrollViewport;
        this.input = elements.input;
        this.inputLine = elements.inputLine;
        this.card = elements.card;
        this.doc = elements.doc;
        this.sleep = options.sleep ?? defaultSleep;
        this.resolveOptions(options);
    }

    /** Resolve timing + motion options. Split from the constructor so neither
     * unit grows past the composition limit. */
    private resolveOptions(options: HeroTerminalOptions): void {
        this.charDelay = options.charDelay ?? CHAR_DELAY;
        this.sendPause = options.sendPause ?? SEND_PAUSE;
        this.workingPause = options.workingPause ?? WORKING_PAUSE;
        this.answerDelay = options.answerDelay ?? ANSWER_DELAY;
        this.loopDelay = options.loopDelay ?? LOOP_DELAY;
        this.startDelay = options.startDelay ?? START_DELAY;
        // Test hook: when reducedMotion is forced, the media query is ignored.
        this.forcedReduced = options.reducedMotion;
        this.reducedMotion = this.forcedReduced ?? detectReducedMotion();
        this.loopDefault = options.loop ?? false;
        this.loop = this.reducedMotion ? false : this.loopDefault;
        // The note's bottom bar fills for the beat's own lifetime. Publish that
        // span to CSS so the progress stroke and the run's timing stay one source.
        this.terminal.style.setProperty("--hero-note-beat", `${this.workingPause}ms`);
    }

    /** Live media-query tracking. Skipped when the test hook forces the path;
     * flipping to reduced stops looping, flipping back never re-runs. */
    trackReducedMotion(): void {
        if (this.forcedReduced !== undefined) {
            return;
        }
        onReducedMotionChange((reduced) => {
            this.reducedMotion = reduced;
            this.loop = reduced ? false : this.loopDefault;
        });
    }

    /** Drop the previous beat's markers. Runs before the next beat is raised,
     * so a beat's rows and note are always cleared as a unit. */
    clearCurrent(): void {
        for (const row of this.currentRows) {
            row.removeAttribute("data-current");
        }
        this.currentRows = [];
        for (const element of this.currentNotes) {
            element.removeAttribute("data-current");
            element.removeAttribute("aria-current");
        }
        this.currentNotes = [];
    }

    /** Raise the rows being narrated. A source beat highlights the agent row AND
     * its ChunkHound receipt together. */
    setCurrentRows(
        lines: HTMLElement | HTMLElement[] | null,
        beat: HeroBeat | undefined,
    ): void {
        this.currentRows = lines === null ? [] : (Array.isArray(lines) ? lines : [lines]);
        for (const row of this.currentRows) {
            row.setAttribute("data-current", "");
            if (beat) {
                row.dataset.beat = beat;
            }
        }
    }

    /** Raise the live beat's one note. aria-current marks it for assistive tech;
     * the run narrates one engine step, so two notes are never raised. */
    setCurrentNote(note: HTMLElement | null | undefined): void {
        this.currentNotes = note ? [note] : [];
        for (const element of this.currentNotes) {
            element.setAttribute("data-current", "");
            element.setAttribute("aria-current", "step");
        }
    }

    /** Narrate one beat: highlight its row(s) and raise its note. Attributes,
     * not classes, so the shipped transcript class contract stays byte-stable. */
    markCurrent(
        lines: HTMLElement | HTMLElement[] | null,
        beat?: HeroBeat,
        note?: HTMLElement | null,
    ): void {
        this.clearCurrent();
        this.setCurrentRows(lines, beat);
        this.setCurrentNote(note);
    }

    /** The scroll range the transcript has in the viewport by LAYOUT, not by the
     * live scroll runtime. The push transform briefly inflates
     * `scrollViewport.scrollHeight` by the push offset, so anchoring on the
     * viewport would scroll the view by that same offset and cancel the glide.
     * The transcript's own `scrollHeight` is transform-free, so it is the single
     * source of truth for "is anything hidden above, and how far is the tail". */
    layoutScrollSpan(): number {
        return Math.max(0, this.terminal.scrollHeight - this.scrollViewport.clientHeight);
    }

    /** True while the scrollback tail is in view. A finger-flick rarely lands
     * exactly at the bottom, so FOLLOW_THRESHOLD leaves slack before a reader
     * counts as having scrolled back. */
    atTail(): boolean {
        return this.scrollViewport.scrollTop >= this.layoutScrollSpan() - FOLLOW_THRESHOLD;
    }

    /** Mirror scrollability onto the card; CSS shows the history fade only when
     * rows really are hidden above. Guarded so a per-character call never churns
     * the attribute. */
    syncScrollable(): void {
        const scrollable = String(this.layoutScrollSpan() > 0);
        if (this.card.dataset.scrollable !== scrollable) {
            this.card.dataset.scrollable = scrollable;
        }
    }

    /** Mirror follow state onto the card; CSS owns the jump affordance. */
    syncFollow(): void {
        const value = String(this.following);
        if (this.card.dataset.following !== value) {
            this.card.dataset.following = value;
        }
        this.syncScrollable();
    }

    setFollowing(next: boolean): void {
        if (next === this.following) {
            return;
        }
        this.following = next;
        this.syncFollow();
    }

    /** Bottom-anchored follow: newest lines land by the fixed input and older
     * lines leave through the top. A reader who has scrolled back owns the
     * viewport — appends grow beneath it instead of yanking them to the tail. */
    followTail(): void {
        if (this.following) {
            this.scrollViewport.scrollTop = this.layoutScrollSpan();
        }
        this.syncScrollable();
    }

    /** Clear the composer and reset its horizontal scroll to the start. */
    clearInput(): void {
        this.input.textContent = "";
        if (this.inputLine) {
            this.inputLine.scrollLeft = 0;
        }
    }

    /** Keep the composer caret in view: the single-line prompt field scrolls
     * right as it types instead of wrapping, like a real harness composer. */
    pinInputCaret(): void {
        if (this.inputLine) {
            this.inputLine.scrollLeft = this.inputLine.scrollWidth;
        }
    }

    /** Wipe the transcript and the per-run narration state, so a replay starts
     * from the same clean slate as the first run. */
    clearTerminal(): void {
        this.terminal.innerHTML = "";
        this.clearInput();
        this.currentRows = [];
        this.currentNotes = [];
        this.following = true;
        this.syncFollow();
    }

    shouldAnimate(): boolean {
        return !this.doc.hidden && this.isVisible;
    }
}

/** Element lookup + guard for one terminal instance. Returns null when the
 * hero markup is absent (non-hero pages), so the caller simply does nothing. */
export function createSession(
    doc: Document,
    options: HeroTerminalOptions = {},
): TerminalSession | null {
    const container = doc.getElementById("terminal-lines");
    const viewport = doc.getElementById("terminal-viewport");
    const inputText = doc.getElementById("terminal-input-text");
    if (!container || !viewport || !inputText) {
        return null;
    }
    // Re-capture as non-null consts so the session holds narrowed HTMLElement
    // refs (the guard above only narrows at this scope).
    const terminal = container;
    const scrollViewport = viewport;
    const input = inputText;
    // The single-line field that scrolls right as the prompt types.
    const inputLine = inputText.parentElement;
    const card = container.closest<HTMLElement>(".terminal-card") || container;
    return new TerminalSession(
        { terminal, scrollViewport, input, inputLine, card, doc },
        options,
    );
}
