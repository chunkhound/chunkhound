interface HeroStepPromptSegment {
    cls: string;
    text: string;
}

type HeroStep =
    | { type: "blank" }
    | { type: "output"; text: string; dim?: boolean; cursor?: boolean }
    | { type: "prompt"; segments: HeroStepPromptSegment[] };

interface HeroTerminalOptions {
    charDelay?: number;
    execPause?: number;
    loopDelay?: number;
    loop?: boolean;
    sleep?: (ms: number) => Promise<void>;
}

const STEPS: HeroStep[] = [
    {
        type: "prompt",
        segments: [{ cls: "cmd", text: "chunkhound index ." }],
    },
    {
        type: "output",
        text: "Indexed 12,847 files · dozens of languages/file types · 2.1M LOC",
    },
    { type: "blank" },
    {
        type: "prompt",
        segments: [
            { cls: "cmd", text: "chunkhound research " },
            { cls: "str", text: '"How does authentication work?"' },
        ],
    },
    {
        type: "output",
        text: "Cited report · 14 files · 6 components · 23 citations",
        dim: true,
    },
    { type: "blank" },
    {
        type: "prompt",
        segments: [
            { cls: "cmd", text: "chunkhound research " },
            { cls: "str", text: '"Summarize auth changes for reviewers"' },
            { cls: "cmd", text: " --commit-range main..HEAD" },
        ],
    },
    {
        type: "output",
        text: "Branch diff analyzed · reviewer brief generated",
        dim: true,
    },
    { type: "blank" },
    {
        type: "prompt",
        segments: [
            { cls: "cmd", text: "chunkhound websearch " },
            { cls: "str", text: '"OAuth refresh token rotation best practices"' },
        ],
    },
    {
        type: "output",
        text: "External docs pinpointed · cited answer generated",
        dim: true,
        cursor: true,
    },
];

function defaultSleep(ms: number): Promise<void> {
    return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function appendStaticStep(parent: HTMLElement, doc: Document, step: HeroStep): void {
    if (step.type === "blank") {
        const line = doc.createElement("div");
        line.className = "line";
        line.innerHTML = "&nbsp;";
        parent.appendChild(line);
        return;
    }

    if (step.type === "prompt") {
        const line = doc.createElement("div");
        line.className = "line";
        const prompt = doc.createElement("span");
        prompt.className = "prompt";
        prompt.textContent = "$ ";
        line.appendChild(prompt);

        step.segments.forEach((segment) => {
            const span = doc.createElement("span");
            span.className = segment.cls;
            span.textContent = segment.text;
            line.appendChild(span);
        });

        parent.appendChild(line);
        return;
    }

    const line = doc.createElement("div");
    line.className = "line";
    const span = doc.createElement("span");
    span.className = step.dim ? "output dim" : "output";
    span.textContent = step.text;
    if (step.cursor) {
        const cursor = doc.createElement("span");
        cursor.className = "cursor";
        span.appendChild(cursor);
    }
    line.appendChild(span);
    parent.appendChild(line);
}

function isCardVisible(element: Element): boolean {
    if (typeof window === "undefined") {
        return true;
    }

    const rect = element.getBoundingClientRect();
    const viewportHeight = window.innerHeight || 0;
    const visibleHeight = Math.min(rect.bottom, viewportHeight) - Math.max(rect.top, 0);
    return visibleHeight >= rect.height * 0.5;
}

export function initHeroTerminal(
    doc: Document = document,
    options: HeroTerminalOptions = {},
): void {
    const container = doc.getElementById("terminal-lines");
    if (!container) {
        return;
    }

    const card = container.closest(".terminal-card") || container;
    const charDelay = options.charDelay ?? 35;
    const execPause = options.execPause ?? 250;
    const loopDelay = options.loopDelay ?? 3000;
    const loop = options.loop ?? true;
    const sleep = options.sleep ?? defaultSleep;

    let isVisible = isCardVisible(card);
    let isRendering = false;
    let settledHeightLockPending = false;

    function lockTerminalHeight(): void {
        const body = doc.body;
        if (!body || typeof body.appendChild !== "function") {
            return;
        }

        const width = container.clientWidth;
        if (!width || typeof container.cloneNode !== "function") {
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
        STEPS.forEach((step) => appendStaticStep(measure, doc, step));
        body.appendChild(measure);
        const height = Math.ceil(measure.getBoundingClientRect().height);
        measure.remove();
        if (height > 0) {
            container.style.height = `${height}px`;
        }
    }

    function clearTerminal(): void {
        container.innerHTML = "";
    }

    function runAfterLayout(callback: () => void): void {
        if (typeof window.requestAnimationFrame === "function") {
            window.requestAnimationFrame(() => {
                callback();
            });
            return;
        }

        window.setTimeout(callback, 0);
    }

    function scheduleSettledHeightLock(): void {
        if (settledHeightLockPending) {
            return;
        }

        settledHeightLockPending = true;
        runAfterLayout(() => {
            settledHeightLockPending = false;
            lockTerminalHeight();
        });

        const fonts = (doc as Document & {
            fonts?: { ready?: Promise<unknown> };
        }).fonts;
        void fonts?.ready?.then(() => {
            lockTerminalHeight();
        });
    }

    function shouldAnimate(): boolean {
        return !doc.hidden && isVisible;
    }

    async function typeText(span: HTMLSpanElement, text: string): Promise<void> {
        for (const ch of text) {
            if (!shouldAnimate()) {
                return;
            }
            span.textContent += ch;
            await sleep(charDelay);
        }
    }

    async function renderOnce(): Promise<void> {
        clearTerminal();

        for (const step of STEPS) {
            if (!shouldAnimate()) {
                clearTerminal();
                return;
            }

            if (step.type === "blank") {
                const line = doc.createElement("div");
                line.className = "line";
                line.innerHTML = "&nbsp;";
                container.appendChild(line);
                continue;
            }

            if (step.type === "prompt") {
                const line = doc.createElement("div");
                line.className = "line";
                const prompt = doc.createElement("span");
                prompt.className = "prompt";
                prompt.textContent = "$ ";
                line.appendChild(prompt);
                container.appendChild(line);

                for (const segment of step.segments) {
                    const span = doc.createElement("span");
                    span.className = segment.cls;
                    line.appendChild(span);
                    await typeText(span, segment.text);
                }

                if (!shouldAnimate()) {
                    clearTerminal();
                    return;
                }

                await sleep(execPause);
                continue;
            }

            const line = doc.createElement("div");
            line.className = "line";
            const span = doc.createElement("span");
            span.className = step.dim ? "output dim" : "output";
            span.textContent = step.text;
            if (step.cursor) {
                const cursor = doc.createElement("span");
                cursor.className = "cursor";
                span.appendChild(cursor);
            }
            line.appendChild(span);
            container.appendChild(line);
        }
    }

    function triggerRender(): void {
        if (!shouldAnimate() || isRendering) {
            return;
        }

        isRendering = true;
        void (async () => {
            try {
                do {
                    await renderOnce();
                    if (!loop || !shouldAnimate()) {
                        return;
                    }
                    await sleep(loopDelay);
                } while (shouldAnimate());
            } finally {
                isRendering = false;
                if (loop && shouldAnimate()) {
                    triggerRender();
                }
            }
        })();
    }

    lockTerminalHeight();
    scheduleSettledHeightLock();

    if (typeof ResizeObserver !== "undefined") {
        const resizeObserver = new ResizeObserver(() => {
            lockTerminalHeight();
        });
        resizeObserver.observe(card);
    }

    const observer = new IntersectionObserver(
        (entries) => {
            isVisible = entries[0]?.isIntersecting ?? false;
            if (!isVisible) {
                clearTerminal();
                return;
            }
            triggerRender();
        },
        { threshold: 0.5 },
    );

    observer.observe(card);

    doc.addEventListener("visibilitychange", () => {
        if (doc.hidden) {
            clearTerminal();
            return;
        }

        scheduleSettledHeightLock();
        runAfterLayout(() => {
            triggerRender();
        });
    });

    window.addEventListener("pagehide", () => {
        isVisible = false;
        clearTerminal();
    });

    window.addEventListener("pageshow", () => {
        isVisible = isCardVisible(card);
        scheduleSettledHeightLock();
        runAfterLayout(() => {
            triggerRender();
        });
    });

    triggerRender();
}

if (typeof document !== "undefined") {
    initHeroTerminal();
}
