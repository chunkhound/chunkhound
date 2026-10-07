/** The hero run's narrative register: one step per engine source class.
 *
 * NARRATIVE = the *run* (what the engine does): the three source classes
 * ChunkHound mines. Animated in step with hero-terminal.ts, which types each
 * step's note beside the receipt it explains as its beat lands. Copy is keyed
 * by beat so a transcript edit cannot silently orphan a step. The question
 * and the verdict belong to the agent, not the engine, so neither takes a
 * note.
 *
 * The static registers a run is judged against — the conditions and the
 * interfaces — live in conditions.ts. A step's `condition` names one of that
 * module's PROMISES ids, so the note and the terms rail cannot drift.
 *
 * Note labels reuse the subheadline's source nouns (code, git history, web
 * pages) so the narration and the demo's evidence rows speak the same language.
 */

import type { ConditionId } from "./conditions";

/** Beats the terminal emits to narrate the run. Each source beat
 * (`code`/`git`/`web`) carries one note; the rest drive the transcript's
 * read-along highlight without annotating a step. */
export type HeroBeat =
    | "prompt"
    | "code"
    | "git"
    | "web"
    | "synthesis"
    | "answer";

export interface HeroNarrativeStep {
    /** The source beat whose receipt this note is anchored to. */
    beats: HeroBeat[];
    /** Source-kind label, drawn from the positioning subheadline's nouns. */
    label: string;
    headline: string;
    detail: string;
    /** The condition this beat demonstrates: its note names that promise as
     * the beat runs. One of conditions.ts PROMISES' ids, so the run can never
     * claim a condition the terms rail does not name. */
    condition: ConditionId;
}

/** The note annotating one engine beat, if it carries one (the prompt and
 * the verdict are the agent's turns, so neither is annotated). Resolved by
 * beat, so a transcript reorder cannot silently orphan a note — shared by
 * hero-terminal.ts and Hero.astro's no-JS render. */
export function noteForBeat(beat: HeroBeat): HeroNarrativeStep | undefined {
    return NARRATIVE.find((entry) => entry.beats.includes(beat));
}
/** The engine's run, in order. Each step is typed as a note on the receipt it
 * explains. `headline` is the beat's contribution to the verdict — the three
 * read as one argument in sequence (located → reason → norm), so the delta is
 * what this evidence adds, not what class of source it is. `detail` is the
 * condition that makes this evidence trustworthy, and it MUST stay tethered to
 * what the receipt above it shows; a condition sold without that tether reads as
 * an ad next to someone else's proof.
 *
 * Both lines are measured against the note's narrowest track (269px of Inter at
 * 13px — roughly 22 headline / 44 detail chars), see
 * tests/site/test_hero_notes_behavior.py. */
export const NARRATIVE: HeroNarrativeStep[] = [
    {
        beats: ["code"],
        label: "CODE",
        headline: "The default, located",
        detail: "Local-first — the value, the file, the line.",
        condition: "local-first",
    },
    {
        beats: ["git"],
        label: "GIT HISTORY",
        headline: "The reason: one caller",
        detail: "Nothing leaves — your commits explain it.",
        condition: "nothing-leaves",
    },
    {
        beats: ["web"],
        label: "WEB PAGES",
        headline: "The norm: lower, scoped",
        detail: "Your models, your bill: fetched and cited.",
        condition: "your-models",
    },
];
