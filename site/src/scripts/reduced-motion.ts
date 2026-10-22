export function detectReducedMotion(): boolean {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
        return true;
    }
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/** Fires cb with the live state whenever prefers-reduced-motion changes. */
export function onReducedMotionChange(cb: (reduced: boolean) => void): void {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
        return;
    }
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    query.addEventListener("change", () => cb(query.matches));
}
