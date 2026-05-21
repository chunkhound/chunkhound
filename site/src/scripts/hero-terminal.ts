type TerminalSegment = {
    cls: string;
    text: string;
};

type TerminalStep =
    | { type: "blank" }
    | { type: "prompt"; segments: readonly TerminalSegment[] }
    | { type: "output"; text: string; dim?: boolean; cursor?: boolean };

type RenderedPrompt = {
    type: "prompt";
    spans: Array<{ span: HTMLSpanElement; text: string }>;
};

const HERO_TERMINAL_STEPS: readonly TerminalStep[] = [
    {
        type: "prompt",
        segments: [{ cls: "cmd", text: "chunkhound index ." }],
    },
    {
        type: "output",
        text: "Indexed 12,847 files · 32 languages · 2.1M LOC",
    },
    { type: "blank" },
    {
        type: "prompt",
        segments: [
            { cls: "cmd", text: "chunkhound search " },
            { cls: "str", text: '"how does authentication work"' },
        ],
    },
    {
        type: "output",
        text: "Found 42 results via 3-hop semantic traversal",
        dim: true,
    },
    { type: "blank" },
    {
        type: "prompt",
        segments: [
            { cls: "cmd", text: "chunkhound research " },
            { cls: "str", text: '"authentication architecture"' },
        ],
    },
    {
        type: "output",
        text: "Generated report · 14 files · 6 components · 23 citations",
        dim: true,
        cursor: true,
    },
];

const CHAR_DELAY = 35;
const EXEC_PAUSE = 250;
const LOOP_DELAY = 3000;

function sleep(ms: number): Promise<void> {
    return new Promise((resolve) => {
        setTimeout(resolve, ms);
    });
}

function appendStep(
    parent: HTMLElement,
    step: TerminalStep,
    fullText = false,
): RenderedPrompt | { type: "output" } | null {
    if (step.type === "blank") {
        const line = document.createElement("div");
        line.className = "line";
        line.textContent = "\u00a0";
        parent.appendChild(line);
        return null;
    }

    if (step.type === "prompt") {
        const line = document.createElement("div");
        line.className = "line";
        const prompt = document.createElement("span");
        prompt.className = "prompt";
        prompt.textContent = "$ ";
        line.appendChild(prompt);
        parent.appendChild(line);

        const spans: RenderedPrompt["spans"] = [];
        for (const segment of step.segments) {
            const span = document.createElement("span");
            span.className = segment.cls;
            if (fullText) {
                span.textContent = segment.text;
            }
            line.appendChild(span);
            spans.push({ span, text: segment.text });
        }
        return { type: "prompt", spans };
    }

    const line = document.createElement("div");
    line.className = "line";
    const span = document.createElement("span");
    span.className = step.dim ? "output dim" : "output";
    span.textContent = step.text;
    if (step.cursor) {
        const cursor = document.createElement("span");
        cursor.className = "cursor";
        span.appendChild(cursor);
    }
    line.appendChild(span);
    parent.appendChild(line);
    return { type: "output" };
}

function renderAllSteps(
    parent: HTMLElement,
    steps: readonly TerminalStep[] = HERO_TERMINAL_STEPS,
): void {
    for (const step of steps) {
        appendStep(parent, step, true);
    }
}

export function lockTerminalHeight(
    container: HTMLElement,
    steps: readonly TerminalStep[] = HERO_TERMINAL_STEPS,
): void {
    const width = container.clientWidth;
    if (!width) {
        return;
    }

    const measure = container.cloneNode(false) as HTMLElement;
    measure.style.position = "absolute";
    measure.style.visibility = "hidden";
    measure.style.pointerEvents = "none";
    measure.style.inset = "0 auto auto 0";
    measure.style.width = `${width}px`;
    measure.style.height = "auto";
    measure.style.minHeight = "0";
    measure.setAttribute("aria-hidden", "true");

    renderAllSteps(measure, steps);
    document.body.appendChild(measure);
    const height = Math.ceil(measure.getBoundingClientRect().height);
    measure.remove();

    container.style.height = `${height}px`;
}

export function observeTerminalHeight(
    container: HTMLElement,
    observeTarget: Element = container,
    steps: readonly TerminalStep[] = HERO_TERMINAL_STEPS,
): ResizeObserver | null {
    lockTerminalHeight(container, steps);
    if (typeof ResizeObserver === "undefined") {
        return null;
    }

    const resizeObserver = new ResizeObserver(() => {
        lockTerminalHeight(container, steps);
    });
    resizeObserver.observe(observeTarget);
    return resizeObserver;
}

async function typeText(
    span: HTMLSpanElement,
    text: string,
    isActive: () => boolean,
): Promise<boolean> {
    for (const character of text) {
        if (!isActive()) {
            return false;
        }
        span.textContent += character;
        await sleep(CHAR_DELAY);
    }
    return isActive();
}

export function initHeroTerminal(container: HTMLElement): () => void {
    const card = container.closest<HTMLElement>(".terminal-card") || container;
    const resizeObserver = observeTerminalHeight(container, card);

    let active = true;
    let isVisible = false;
    let resolveInView: ((value: boolean) => void) | null = null;
    let resolvePageVisible: ((value: boolean) => void) | null = null;

    const finishWaits = (value: boolean) => {
        resolveInView?.(value);
        resolveInView = null;
        resolvePageVisible?.(value);
        resolvePageVisible = null;
    };

    const handleVisibilityChange = () => {
        if (!document.hidden) {
            finishWaits(active);
        }
    };

    document.addEventListener("visibilitychange", handleVisibilityChange);

    const waitVisible = (): Promise<boolean> => {
        if (!active) {
            return Promise.resolve(false);
        }
        if (!document.hidden) {
            return Promise.resolve(true);
        }
        return new Promise((resolve) => {
            resolvePageVisible = resolve;
        });
    };

    const waitInView = (): Promise<boolean> => {
        if (!active) {
            return Promise.resolve(false);
        }
        if (isVisible) {
            return Promise.resolve(true);
        }
        return new Promise((resolve) => {
            resolveInView = resolve;
        });
    };

    const isActive = () => active && isVisible && !document.hidden;

    const runAnimation = async () => {
        while (active) {
            if (!(await waitVisible()) || !(await waitInView()) || !active) {
                break;
            }
            container.innerHTML = "";

            for (const step of HERO_TERMINAL_STEPS) {
                if (!isActive()) {
                    container.innerHTML = "";
                    break;
                }

                const rendered = appendStep(container, step);
                if (rendered?.type !== "prompt") {
                    continue;
                }

                for (const segment of rendered.spans) {
                    if (!(await typeText(segment.span, segment.text, isActive))) {
                        container.innerHTML = "";
                        break;
                    }
                }

                if (!isActive()) {
                    break;
                }
                await sleep(EXEC_PAUSE);
            }

            if (isActive()) {
                await sleep(LOOP_DELAY);
            }
        }
    };

    let observer: IntersectionObserver | null = null;
    if (typeof IntersectionObserver === "undefined") {
        isVisible = true;
    } else {
        observer = new IntersectionObserver(
            (entries) => {
                isVisible = entries[0]?.isIntersecting ?? false;
                if (isVisible) {
                    finishWaits(active);
                    return;
                }
                container.innerHTML = "";
            },
            { threshold: 0.5 },
        );
        observer.observe(card);
    }

    void runAnimation();

    return () => {
        if (!active) {
            return;
        }
        active = false;
        container.innerHTML = "";
        document.removeEventListener("visibilitychange", handleVisibilityChange);
        observer?.disconnect();
        resizeObserver?.disconnect();
        finishWaits(false);
    };
}
