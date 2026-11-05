export function detectReducedMotion(): boolean {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
        return true;
    }
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/** Fires cb with the live state whenever prefers-reduced-motion changes.
 * Returns a disposer; callers that ignore it keep their old behavior. */
export function onReducedMotionChange(cb: (reduced: boolean) => void): () => void {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
        return () => {};
    }
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    const handler = (): void => cb(query.matches);
    query.addEventListener("change", handler);
    return () => query.removeEventListener("change", handler);
}
