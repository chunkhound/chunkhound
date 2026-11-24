import { detectReducedMotion, onReducedMotionChange } from "./reduced-motion";

/**
 * Shared scroll-reveal + count-up primitive for marketing sections.
 * Progressive enhancement: server-rendered HTML already carries the final
 * values, so without JS (or with reduced motion) the content stays intact.
 * The entrance plays once per element: the first intersection reveals and
 * stops observing it, so scrolling back never replays the intro.
 */

const COUNT_DURATION_MS = 1400;
// Tall sections (e.g. full configurator) never reach 40% visibility, so they
// get a lower threshold; everything else reveals at 40%.
const VIEWPORT_TALL_RATIO = 0.6;
const THRESHOLD_TALL = 0.15;
const THRESHOLD_DEFAULT = 0.4;

function thresholdFor(el: HTMLElement): number {
    const height = el.getBoundingClientRect?.().height ?? 0;
    const viewportHeight = window.innerHeight ?? 0;
    return height > VIEWPORT_TALL_RATIO * viewportHeight ? THRESHOLD_TALL : THRESHOLD_DEFAULT;
}

/** Index among the PARENT's [data-reveal] children — document-global indexing
    would give deep, late-entering elements huge stagger delays. */
function siblingRevealIndex(el: HTMLElement): number {
    const siblings = el.parentElement?.querySelectorAll<HTMLElement>("[data-reveal]");
    return siblings ? Array.prototype.indexOf.call(siblings, el) : 0;
}

interface CountConfig {
    target: number;
    decimals: number;
    suffix: string;
    /** Original server-rendered text — the count must end exactly here. */
    finalText: string;
}

function easeOutCubic(progress: number): number {
    return 1 - Math.pow(1 - progress, 3);
}

function parseCountConfig(el: HTMLElement): CountConfig | null {
    const raw = el.dataset.countTo ?? "";
    const target = Number(raw);
    if (raw === "" || !Number.isFinite(target)) {
        return null;
    }
    const decimals = raw.includes(".") ? (raw.split(".")[1]?.length ?? 0) : 0;
    return {
        target,
        decimals,
        suffix: el.dataset.suffix ?? "",
        finalText: el.textContent ?? "",
    };
}

function runCountUp(
    el: HTMLElement,
    config: CountConfig,
    running: Map<HTMLElement, number>,
): void {
    const previous = running.get(el);
    if (previous !== undefined) {
        cancelAnimationFrame(previous);
    }
    const start = performance.now();
    const step = (now: number): void => {
        const progress = Math.min((now - start) / COUNT_DURATION_MS, 1);
        const value = config.target * easeOutCubic(progress);
        el.textContent = value.toFixed(config.decimals) + config.suffix;
        if (progress < 1) {
            running.set(el, requestAnimationFrame(step));
            return;
        }
        running.delete(el);
    };
    running.set(el, requestAnimationFrame(step));
}

/** Hidden pre-entrance state is applied ONLY here (never in static CSS),
    so content is always visible without JS. Removing the attribute on
    entry lets the component's CSS transition + stagger take over. */
function prepareReveal(el: HTMLElement): void {
    el.style.setProperty("--i", String(siblingRevealIndex(el)));
    el.setAttribute("data-reveal-pending", "");
}

function enterViewport(
    el: HTMLElement,
    configs: Map<HTMLElement, CountConfig>,
    running: Map<HTMLElement, number>,
    reduced: boolean,
): void {
    const config = configs.get(el);
    if (reduced) {
        // Reduced mode settles to the server-rendered value — never animate.
        if (config) {
            el.textContent = config.finalText;
        }
    } else if (config) {
        runCountUp(el, config, running);
    }
    if (el.hasAttribute("data-reveal")) {
        el.removeAttribute("data-reveal-pending");
    }
}

/** Reduced motion: everything settles to its server-rendered final state
    (running counts cancelled, pending reveals lifted). Idempotent. Marks
    elements revealed so a later flip back to animated never re-animates
    content the user already sees. */
function settleAll(
    elements: HTMLElement[],
    configs: Map<HTMLElement, CountConfig>,
    running: Map<HTMLElement, number>,
    revealed: Set<HTMLElement>,
): void {
    for (const el of elements) {
        const handle = running.get(el);
        if (handle !== undefined) {
            cancelAnimationFrame(handle);
            running.delete(el);
        }
        const config = configs.get(el);
        if (config) {
            el.textContent = config.finalText;
        }
        el.removeAttribute("data-reveal-pending");
        revealed.add(el);
    }
}

/** One observer per distinct threshold (max 2: default + tall). */
function observerFor(
    observers: Map<number, IntersectionObserver>,
    threshold: number,
    onEntries: IntersectionObserverCallback,
): IntersectionObserver {
    let observer = observers.get(threshold);
    if (!observer) {
        observer = new IntersectionObserver(onEntries, { threshold });
        observers.set(threshold, observer);
    }
    return observer;
}

/** Thresholds are derived from element height vs viewport height at init;
    a resize that moves an element across the tall/short boundary needs
    re-arming with the right observer. Re-observing with a different
    observer re-fires the initial callback for not-yet-revealed elements;
    same-observer observe() is a no-op. A boundary crossing can land on a
    threshold whose observer was never created at init, so create it here —
    otherwise the element would be unobserved everywhere and stay pending
    (invisible) forever. Revealed elements are done: re-observing them would
    only re-arm a one-shot entrance. */
function recomputeThresholds(
    elements: HTMLElement[],
    observers: Map<number, IntersectionObserver>,
    revealed: Set<HTMLElement>,
    onEntries: IntersectionObserverCallback,
): void {
    for (const el of elements) {
        if (revealed.has(el)) {
            continue;
        }
        const threshold = thresholdFor(el);
        observerFor(observers, threshold, onEntries);
        for (const [value, observer] of observers) {
            if (value === threshold) {
                observer.observe(el);
            } else {
                observer.unobserve(el);
            }
        }
    }
}

// Many sections import this module; a second init would attach duplicate
// observers. First init wins, and only once there is something to observe.
let started = false;

export function initCountReveal(root: ParentNode = document): void {
    if (started) {
        return;
    }
    if (typeof window === "undefined" || typeof IntersectionObserver === "undefined") {
        return;
    }
    const counters = Array.from(root.querySelectorAll<HTMLElement>("[data-count-to]"));
    const revealEls = Array.from(root.querySelectorAll<HTMLElement>("[data-reveal]"));
    if (counters.length === 0 && revealEls.length === 0) {
        return;
    }
    started = true;
    // Capture final texts once at init: a reduced-motion settle restores
    // them, so a count cancelled mid-flight can never poison the final value.
    const configs = new Map<HTMLElement, CountConfig>();
    counters.forEach((el) => {
        const config = parseCountConfig(el);
        if (config) {
            configs.set(el, config);
        }
    });
    const running = new Map<HTMLElement, number>();
    const elements = [...counters, ...revealEls];

    // Live preference flag: observers stay attached across flips, so the
    // enter callbacks (not observer lifecycle) enforce reduced motion.
    let reducedMotion = detectReducedMotion();
    // One-shot entrance: the first intersection reveals the element and
    // stops observing it, so re-scrolling never replays the intro.
    const revealed = new Set<HTMLElement>();
    const onEntries = (
        entries: IntersectionObserverEntry[],
        observer: IntersectionObserver,
    ): void => {
        for (const entry of entries) {
            const el = entry.target as HTMLElement;
            if (!entry.isIntersecting || revealed.has(el)) {
                continue;
            }
            revealed.add(el);
            enterViewport(el, configs, running, reducedMotion);
            observer.unobserve(el);
        }
    };

    const observers = new Map<number, IntersectionObserver>();
    let observing = false;
    const startObservation = (): void => {
        if (observing) {
            return;
        }
        observing = true;
        revealEls.forEach(prepareReveal);
        for (const el of elements) {
            observerFor(observers, thresholdFor(el), onEntries).observe(el);
        }
        // Debounced: resize only matters when an element crosses the
        // tall/short boundary — not on every pixel of a drag-resize.
        let resizeTimer: number | undefined;
        window.addEventListener("resize", () => {
            window.clearTimeout(resizeTimer);
            resizeTimer = window.setTimeout(
                () => recomputeThresholds(elements, observers, revealed, onEntries),
                200,
            );
        });
    };

    // Reduced init: content is already server-rendered final — stay dormant
    // until the preference flips back to animated.
    onReducedMotionChange((reduced) => {
        reducedMotion = reduced;
        if (reduced) settleAll(elements, configs, running, revealed);
        else startObservation();
    });
    if (reducedMotion) {
        return;
    }
    startObservation();
}

if (typeof document !== "undefined") {
    initCountReveal();
}
