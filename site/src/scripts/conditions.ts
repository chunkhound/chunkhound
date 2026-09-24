/** The landing's terms register: the conditions the product runs on and the
 * interfaces you reach it through. Rendered as one icon-led rail in the
 * ProvenAtScale section (no chip chrome, always lit), split into two labelled
 * groups so the rail reads as the section's terms line rather than a claim row.
 *
 * ENTRY_POINTS = the *interfaces* (CLI, MCP). PROMISES = the *conditions*
 * (where/whose/on-what-terms it runs) — the product's core value props.
 *
 * `PROMISES[].id` is the shared vocabulary: a run beat claims the condition it
 * demonstrates by naming one of these ids (see hero-narrative.ts), so the rail
 * and the run cannot drift, and the run can never claim a condition that is not
 * named here.
 */

export interface Term {
    icon: string;
    label: string;
    /** Stable id a run note's `condition` can name. Interfaces omit it. */
    id?: string;
}

/** The interfaces you reach the engine through, as a small muted peer pair. */
export const ENTRY_POINTS: Term[] = [
    { icon: "ph-terminal-window", label: "CLI" },
    { icon: "ph-plugs-connected", label: "MCP" },
];

/** The conditions — the hero's critical selling points (open-source, deployment,
 * privacy, economics), in rail order. The literal ids are the vocabulary a run
 * note's `condition` must name. “Nothing leaves your machine” intentionally
 * means ChunkHound's runtime and index stay local, not that an operator-selected
 * remote provider receives no request content; NothingLeaves.astro discloses it. */
export const PROMISES = [
    { id: "open-source", icon: "ph-git-fork", label: "Free & open-source" },
    { id: "local-first", icon: "ph-laptop", label: "Local-first" },
    { id: "nothing-leaves", icon: "ph-shield-check", label: "Nothing leaves your machine" },
    { id: "your-models", icon: "ph-sliders-horizontal", label: "Your models, your bill" },
] as const;

/** A condition id a run beat may name to claim it demonstrates that condition. */
export type ConditionId = (typeof PROMISES)[number]["id"];
