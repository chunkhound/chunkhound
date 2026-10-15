import { DEMO, callBeat, type HeroSource } from "./hero-transcript";
import {
    NARRATIVE,
    noteForBeat,
    type HeroBeat,
    type HeroNarrativeStep,
} from "./hero-narrative";

/**
 * Static DOM builders for the hero transcript. Pure functions: every builder
 * takes its parent + document and returns what it appended. The shipped
 * transcript class contract (class names, glyphs, structure) lives here and
 * stays byte-stable — narration marks beats with data-* only, never classes.
 */

export type HeroStep =
    | { type: "prompt"; text: string; beat: HeroBeat }
    | { type: "call"; text: string; beat: HeroBeat; sources?: HeroSource[] }
    | { type: "answer"; text: string; beat: HeroBeat };

/** The note for an engine step, if its beat carries one (the prompt and the
 * verdict are the agent's turns, so neither is annotated). Resolved by beat, so
 * a transcript edit cannot silently orphan a note. */
export function noteForStep(step: HeroStep): NoteRef | undefined {
    const note = noteForBeat(step.beat);
    if (!note) {
        return undefined;
    }
    return { step: note, index: NARRATIVE.indexOf(note) };
}

function buildSteps(): HeroStep[] {
    const steps: HeroStep[] = [
        { type: "prompt", text: DEMO.prompt, beat: "prompt" },
    ];
    DEMO.calls.forEach((call, index) => {
        steps.push({
            type: "call",
            text: call.call,
            beat: callBeat(index),
            sources: call.sources,
        });
    });
    steps.push({ type: "call", text: DEMO.synthesis, beat: "synthesis" });
    steps.push({ type: "answer", text: DEMO.answer, beat: "answer" });
    return steps;
}

export const STEPS = buildSteps();

export function createLine(parent: HTMLElement, doc: Document, className = "line"): HTMLDivElement {
    const line = doc.createElement("div");
    line.className = className;
    parent.appendChild(line);
    return line;
}

export function appendTextSpan(parent: HTMLElement, doc: Document, className: string, text: string): HTMLSpanElement {
    const span = doc.createElement("span");
    span.className = className;
    span.textContent = text;
    parent.appendChild(span);
    return span;
}

/** The role label's text is screen-reader-only: the compact tiers collapse the
 * visible label column, so visual identity travels via the glyph + role
 * colors (⏺ agent, ↳ ChunkHound, ✓ answer). */
export function appendRoleLabel(parent: HTMLElement, doc: Document, role: string): HTMLSpanElement {
    const label = appendTextSpan(parent, doc, "role-label", "");
    if (role) {
        appendTextSpan(label, doc, "sr-only", role);
    }
    return label;
}

/** The user's submitted turn in the scrollback: the `❯` glyph marks the turn
 * visually; "You" stays as a screen-reader-only role label, and the role column
 * is kept empty so the glyph aligns with the other rows. */
export function appendSubmittedPrompt(
    parent: HTMLElement,
    doc: Document,
    text: string,
): HTMLDivElement {
    const line = createLine(parent, doc, "line user-prompt");
    appendRoleLabel(line, doc, "You");
    appendTextSpan(line, doc, "glyph", "❯ ");
    appendTextSpan(line, doc, "prompt-text", text);
    return line;
}

/** The agent's visible act: it calls a ChunkHound tool. The row opens on the
 * first spinner frame, `spinGlyph()` animates it while the tool runs, then it
 * settles in place (`⠋` → `⏺`). It never vanishes, so transcript height stays
 * monotonic and nothing above it shifts. */
export function appendCall(
    parent: HTMLElement,
    doc: Document,
    text: string,
    settled: boolean,
): { line: HTMLDivElement; glyph: HTMLSpanElement; text: HTMLSpanElement } {
    const line = createLine(parent, doc, settled ? "line call" : "line call is-working");
    appendRoleLabel(line, doc, "Agent");
    const glyph = appendTextSpan(line, doc, "glyph", settled ? "⏺ " : "⠋ ");
    const callText = appendTextSpan(line, doc, "call-text", text);
    return { line, glyph, text: callText };
}

/** The engine's cited return for one call: a compact, verifiable receipt
 * hanging under the ChunkHound label. The list element always exists in the
 * DOM — the working row reserves it — so the citations only have to be filled
 * in, character by character. */
export interface HeroReceipt {
    line: HTMLDivElement;
    glyph: HTMLSpanElement;
    list: HTMLSpanElement;
}

/** One citation's rendered text — the single source of truth for the
 * `${kind}: ${cite}` layout shown in every receipt. */
export function sourceText(source: HeroSource): string {
    return `${source.kind}: ${source.cite}`;
}

/** Create one (empty) `.evidence-text` in the list. The caller fills it: the
 * static/measure path writes the whole text synchronously, the animated path
 * streams it in via `typeText`. */
export function appendEvidenceItem(list: HTMLElement, doc: Document): HTMLSpanElement {
    const item = doc.createElement("span");
    item.className = "evidence-text";
    list.appendChild(item);
    return item;
}

export function appendEvidenceItems(
    list: HTMLElement,
    doc: Document,
    sources: HeroSource[],
): void {
    for (const source of sources) {
        appendEvidenceItem(list, doc).textContent = sourceText(source);
    }
}

/** A beat group: the agent-call + ChunkHound-receipt rows, then the note that
 * annotates them. The note is DOM-last, so the reading order is call → receipt
 * → note and CSS alone places it; no script chooses layout (see Hero.astro). */
export function createAnchor(parent: HTMLElement, doc: Document): HTMLDivElement {
    const anchor = doc.createElement("div");
    anchor.className = "anchor";
    parent.appendChild(anchor);
    return anchor;
}

/** The ChunkHound row. In flight it opens with a blinking dot and an empty
 * receipt list, so the engine's progress is visible right after the agent's
 * line; the citations then type into that list and settling swaps the dot for
 * the receipt marker. */
export function appendSources(
    anchor: HTMLElement,
    doc: Document,
    sources: HeroSource[],
    working = false,
): HeroReceipt {
    const line = createLine(
        anchor,
        doc,
        working ? "line evidence is-working" : "line evidence",
    );
    appendRoleLabel(line, doc, "ChunkHound");
    const glyph = appendTextSpan(line, doc, "glyph", working ? "● " : "↳ ");
    const list = doc.createElement("span");
    list.className = "evidence-list";
    if (!working) {
        appendEvidenceItems(list, doc, sources);
    }
    line.appendChild(list);
    return { line, glyph, list };
}

/** The call returned: the working dot becomes the receipt marker. The list is
 * untouched — its citations were already typed in while the dot blinked. */
export function settleSources(receipt: HeroReceipt): void {
    receipt.glyph.textContent = "↳ ";
    receipt.line.classList.remove("is-working");
}

/** The agent's verdict: distilled from the cited returns above, never a
 * restatement of them. */
export function appendAnswer(parent: HTMLElement, doc: Document, answer: string): HTMLSpanElement {
    const line = createLine(parent, doc, "line answer");
    appendRoleLabel(line, doc, "Agent");
    appendTextSpan(line, doc, "glyph", "✓ ");
    return appendTextSpan(line, doc, "result", answer);
}

/** A note and its position in the run — the pair the renderers pass around. */
export interface NoteRef {
    step: HeroNarrativeStep;
    index: number;
}

/** The note's header: beat order + condition label + headline. Split from
 * `appendNote` so neither function grows past the composition limit. */
function appendNoteHead(note: HTMLElement, doc: Document, ref: NoteRef): void {
    const order = String(ref.index + 1).padStart(2, "0");
    const head = doc.createElement("div");
    head.className = "note-head";
    const index = appendTextSpan(head, doc, "note-index", order);
    index.setAttribute("aria-hidden", "true");
    appendTextSpan(head, doc, "note-kind", ref.step.label);
    appendTextSpan(head, doc, "note-headline", ref.step.headline);
    note.appendChild(head);
}

/** The run's marketing copy, anchored to the receipt it explains. The note
 * hangs inline below its call + receipt at every card width, railed and
 * connected to the receipt; CSS owns the placement, so the no-JS render is
 * identical. */
export function appendNote(parent: HTMLElement, doc: Document, ref: NoteRef): HTMLElement {
    const note = doc.createElement("aside");
    note.className = "note";
    // role="note", not aside's implicit complementary: an ancillary comment,
    // not a landmark — three landmarks inside one transcript would be noise.
    note.setAttribute("role", "note");
    note.dataset.note = ref.step.beats[0] ?? "";
    // The condition this beat demonstrates. Declarative: the status strip never
    // changes with the run, but the beat→value-prop mapping stays in the DOM so
    // NARRATIVE, the note and the strip cannot drift apart (asserted).
    note.dataset.condition = ref.step.condition;
    appendNoteHead(note, doc, ref);
    appendTextSpan(note, doc, "note-detail", ref.step.detail);
    parent.appendChild(note);
    return note;
}

export function appendStaticStep(parent: HTMLElement, doc: Document, step: HeroStep): void {
    if (step.type === "prompt") {
        appendSubmittedPrompt(parent, doc, step.text);
    } else if (step.type === "call") {
        appendStaticCall(parent, doc, step);
    } else if (step.type === "answer") {
        appendAnswer(parent, doc, step.text);
    }
}

/** Static render of one call beat: the call, its receipt, then the note —
 * the same DOM the animated path builds, so no-JS and reduced-motion match. */
function appendStaticCall(parent: HTMLElement, doc: Document, step: HeroStep & { type: "call" }): void {
    if (!step.sources) {
        appendCall(parent, doc, step.text, true);
        return;
    }
    const anchor = createAnchor(parent, doc);
    const noteRef = noteForStep(step);
    appendCall(anchor, doc, step.text, true);
    appendSources(anchor, doc, step.sources, false);
    if (noteRef) {
        appendNote(anchor, doc, noteRef);
    }
}
