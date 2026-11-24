// DOM-only view layer for the workflow demo: lookups, transcript cloning, and
// status/picker mutations. The playback machine owns timers and run state.

const STATUS_CELLS = ["checkpoints", "elapsed", "cost", "effect"] as const;

/** Resolves every element the demo needs; throws the established error strings. */
export function workflowParts(card: HTMLElement) {
    const steps = [...card.querySelectorAll<HTMLElement>(".workflow-demo-step")];
    const command = card.querySelector<HTMLElement>(".terminal-input")?.dataset.command;
    if (!command) throw new Error("Workflow demo command is missing");
    const status = card.querySelector<HTMLElement>(".workflow-live-status");
    const targets = STATUS_CELLS.map((cell) => card.querySelector<HTMLElement>(`.workflow-${cell}`));
    const warning = card.querySelector<HTMLElement>(".workflow-warning");
    const composer = card.querySelector<HTMLElement>(".terminal-composer");
    if (!status || targets.some((target) => !target) || !warning) throw new Error("Workflow demo status is incomplete");
    if (!composer) throw new Error("Workflow demo composer is missing");
    // The guard proves every target resolved; the cast restores the element type.
    return { steps, command, status, targets: targets as HTMLElement[], warning, composer, pickers: pickersOf(card) };
}

/** Clones the static steps into the measure pane for the boot render. */
export function renderTranscript(steps: HTMLElement[], measure: HTMLElement): void {
    steps.forEach((step) => {
        const copy = step.cloneNode(true) as HTMLElement;
        copy.hidden = false;
        measure.appendChild(copy);
    });
}

/** Updates the live status bar for a step; summary and pre-checkpoint steps hide it. */
export function writeStatus(status: HTMLElement, targets: HTMLElement[], warning: HTMLElement, step: HTMLElement): void {
    if (step.hasAttribute("data-summary")) { status.hidden = true; return; }
    if (!step.dataset.checkpoints) return;
    status.hidden = false;
    const values = [`🚩 ${step.dataset.checkpoints}`, `⏱ ${step.dataset.elapsed}/1:00:00`, `${step.dataset.cost}/$1.00`, step.dataset.effect ? `$ ${step.dataset.effect}` : ""];
    targets.forEach((target, index) => { if (target) target.textContent = values[index]; });
    warning.hidden = !step.hasAttribute("data-warning");
}

export function pickersOf(card: HTMLElement): HTMLElement[] {
    return [...card.querySelectorAll<HTMLElement>(".workflow-picker")];
}

/** Marks one option of a picker as selected via the ❯ caret glyph. */
export function selectOption(picker: HTMLElement, selected: number): void {
    picker.querySelectorAll<HTMLElement>("[data-option]").forEach((option, index) => {
        option.textContent = `${index === selected ? "❯" : "　"} ${option.textContent?.replace(/^[❯\s]+/u, "")}`;
    });
}

/** Collapses every picker back to its first option with the key hidden. */
export function resetPickers(pickers: HTMLElement[]): void {
    pickers.forEach((picker) => {
        picker.hidden = true;
        selectOption(picker, 0);
        const key = picker.querySelector<HTMLElement>(".workflow-key");
        if (key) key.hidden = true;
    });
}

export function revealKey(picker: HTMLElement): void {
    const key = picker.querySelector<HTMLElement>(".workflow-key");
    if (key) key.hidden = false;
}
