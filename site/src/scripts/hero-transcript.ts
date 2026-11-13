/** Single source of truth for the hero demo transcript. Shared by the
 * client script (animated replay) and Hero.astro (no-JS static render).
 *
 * The demo tells one story in four beats: locate the value (code), explain why
 * it is that value (git history), benchmark it against peers (web pages), then
 * reconcile the three into one verdict.
 *
 * The cross-reference IS the demo. A later call MUST carry a fact an earlier
 * call found ("why it was raised to 30s"), so the run reads as one thread and
 * not three parallel lookups; `synthesis` names the tension that only exists
 * when all three are held together; and the verdict's clauses each resolve to a
 * DIFFERENT source class, so the answer is unreachable from any single one.

 * Row jobs MUST NOT blur:
 * - `calls[].sources` (the ChunkHound row) = the engine at work, then its
 *   receipts: the row opens blinking while the call is in flight (right after
 *   the agent's line), then settles into one compact locator per source class
 *   (path:line · fact, hash · delta, doc › section), terse and verifiable at a
 *   glance — never a clause.
 * - `answer` (the Agent row) = the distilled verdict: an inference plus a
 *   decision the evidence supports but does not state. It MUST NOT restate a
 *   citation (path:line, hash, doc section) or a source fact; if deleting the
 *   answer loses nothing, it is an echo, not a finding.
 *
 * `call` is the agent's plain-language request (no tool syntax). Source `kind`
 * nouns MUST be drawn from the subheadline (code, git history, web pages,
 * documents). The prompt and the answer MUST each stay <= ~90 chars: the hero's
 * transcript measure is ~46 chars on the narrowest card, so a longer row costs
 * an extra wrapped line inside the fold budget. */
import type { HeroBeat } from "./hero-narrative";

export interface HeroSource {
    kind: string;
    cite: string;
}

export interface HeroCall {
    call: string;
    sources: HeroSource[];
}

export interface HeroDemo {
    prompt: string;
    calls: HeroCall[];
    synthesis: string;
    answer: string;
}

export const DEMO: HeroDemo = {
    prompt: "Should the request timeout still be 30s?",
    calls: [
        {
            call: "Reading where the timeout is set today",
            sources: [{ kind: "code", cite: "api/http.ts:14 · timeout 30s" }],
        },
        {
            call: "Tracing why it was raised to 30s",
            sources: [
                { kind: "git history", cite: "9b2f1c4 · 10s→30s (export job)" },
            ],
        },
        {
            call: "Comparing 30s against what peers ship",
            sources: [
                { kind: "web pages", cite: "Stripe, Google › 10s + per-call override" },
            ],
        },
    ],
    // The collision, not a count: one caller's need became everyone's default.
    synthesis: "One caller's need, everyone's default.",
    // Verdict clauses, one source each: "isn't wrong" (git) · "it's global"
    // (code) · "keep it for the export job" (git actor) · "drop it as the
    // default" (web). The inference — the fault is scope, not value — is in
    // none of them.
    answer:
        "The 30s isn't wrong — it's global. Keep it for the export job;" +
        " drop it as the default.",
};

/** Fan-out beats, in demo order — the three source classes the subheadline
 * promises. */
export const CALL_BEATS: HeroBeat[] = ["code", "git", "web"];

/** One beat per fan-out call. Fails loud at import time if the demo grows
 * without a matching beat — a silent fallback would annotate the wrong step. */
export function callBeat(index: number): HeroBeat {
    const beat = CALL_BEATS[index];
    if (!beat) {
        throw new Error(
            `hero demo call ${index} has no beat (CALL_BEATS has ${CALL_BEATS.length})`,
        );
    }
    return beat;
}
