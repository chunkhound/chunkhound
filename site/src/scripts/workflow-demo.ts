import { createSession } from "./terminal-session";
import { setControl } from "./terminal-controls";
import { bootTerminalFrame } from "./terminal-frame";
import { enterRow } from "./terminal-entrance";
import { typeText } from "./terminal-typing";
import { renderTranscript, resetPickers, revealKey, selectOption, workflowParts, writeStatus } from "./workflow-demo-view";

function startWorkflowDemo(card: HTMLElement): void {
    const session = createSession(document, {}, card);
    if (!session) throw new Error("Workflow demo terminal is incomplete");
    const { steps, command, status, targets, warning, composer, pickers } = workflowParts(card);
    resetPickers(pickers);
    let generation = 0;
    let running = false;
    let started = false;
    // Destroy aborts in-flight playback: bumping generation makes pending awaits bail.
    session.onCleanup(() => { generation += 1; running = false; });
    const finish = () => {
        generation++;
        running = false;
        steps.forEach((step) => { step.hidden = false; });
        resetPickers(pickers);
        delete card.dataset.selecting;
        status.hidden = true;
        // Keep the composer text: the reduced-motion path lands here without
        // typing, so clearing it would hide the command the run was invoked
        // with. A replay still starts clean because play() clears and types.
        session.card.dataset.composer = "sent";
        session.card.classList.add("is-settled");
        setControl(session, "replay");
        session.followTail();
    };
    const typeCommand = async (run: number) => {
        session.clearInput();
        await typeText(session, session.input, command, session.charDelay,
            () => run !== generation, () => session.pinInputCaret());
        if (run === generation) await session.sleep(session.sendPause);
    };
    const showStep = (step: HTMLElement) => {
        step.hidden = false;
        enterRow(session, step, false);
        writeStatus(status, targets, warning, step);
        session.followTail();
    };
    const choosePreset = async (picker: HTMLElement, run: number) => {
        card.dataset.selecting = "";
        picker.hidden = false;
        await session.sleep(500);
        if (run !== generation) return;
        selectOption(picker, 1);
        await session.sleep(500);
        if (run !== generation) return;
        selectOption(picker, 0);
        await session.sleep(500);
        if (run !== generation) return;
        revealKey(picker);
        await session.sleep(500);
        if (run === generation) {
            picker.hidden = true;
            delete card.dataset.selecting;
        }
    };
    const startRun = () => {
        const run = ++generation;
        running = true;
        steps.forEach((step) => { step.hidden = true; });
        resetPickers(pickers);
        status.hidden = true;
        session.following = true;
        session.syncFollow();
        session.card.classList.remove("is-settled");
        session.card.dataset.composer = "live";
        setControl(session, "stop");
        return run;
    };
    const play = async () => {
        if (session.reducedMotion) { finish(); return; }
        const run = startRun();
        await typeCommand(run);
        if (run !== generation) return;
        session.clearInput();
        session.card.dataset.composer = "sent";
        for (const picker of pickers) {
            if (run !== generation) return;
            await choosePreset(picker, run);
        }
        for (const step of steps) {
            if (run !== generation) return;
            showStep(step);
            await session.sleep(1000);
        }
        if (run === generation) finish();
    };
    session.listen(session.doc, "visibilitychange", () => {
        if (session.doc.hidden && running) finish();
    });
    if (!session.reducedMotion) steps.forEach((step) => { step.hidden = true; });
    bootTerminalFrame(session, {
        render: (measure) => renderTranscript(steps, measure),
        toggle: () => { if (running) finish(); else void play(); },
        visibilityTarget: composer,
        onVisibility: (visible) => {
            if (visible && !started) {
                started = true;
                void play();
            } else if (!visible && running) finish();
        },
        onMotion: (reduced) => { if (reduced) finish(); },
    });
}

for (const card of document.querySelectorAll<HTMLElement>(".terminal-card:has(.workflow-demo-lines)")) startWorkflowDemo(card);
