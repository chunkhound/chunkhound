/**
 * Accessible off-canvas drawer shared by the docs sidebar and the global
 * mobile menu. Owns modal semantics, focus containment/restore, inert
 * background, Escape + scrim dismissal, and body scroll-lock.
 *
 * Caller responsibilities: render a toggle (`aria-expanded`/`aria-controls`),
 * a panel, an optional scrim, and mark background regions with
 * `data-nav-mobile-inert`. The caller's CSS owns the closed state (`open`
 * class) and the mobile-only visibility of the toggle/panel.
 */

export interface MobileDrawerConfig {
    toggle: HTMLButtonElement;
    panel: HTMLElement;
    scrim?: HTMLElement | null;
    /** Drawer is active only while this matches (e.g. max-width: 640px). */
    media: MediaQueryList;
    /** Background regions made inert while the drawer is open. */
    inertTargets: HTMLElement[];
    /** Accessible toggle label, swapped between open and close. */
    labels: { open: string; close: string };
}

const FOCUSABLE_SELECTOR = [
    "a[href]",
    "button:not([disabled])",
    "input:not([disabled])",
    "select:not([disabled])",
    "textarea:not([disabled])",
    "[tabindex]:not([tabindex='-1'])",
].join(", ");

/** Marks `data-nav-mobile-inert` regions so callers stay in sync with this module. */
export const MOBILE_INERT_ATTR = "data-nav-mobile-inert";

export function initMobileDrawer(config: MobileDrawerConfig): void {
    const { toggle, panel, scrim, media, inertTargets, labels } = config;
    const doc = toggle.ownerDocument;
    let open = false;

    const isVisibleForFocus = (element: HTMLElement): boolean => {
        if (element.hasAttribute("hidden") || element.getAttribute("aria-hidden") === "true") {
            return false;
        }

        if (element.style.display === "none") {
            return false;
        }

        if (typeof element.getClientRects === "function") {
            return element.getClientRects().length > 0;
        }

        return true;
    };

    const getFocusable = (): HTMLElement[] =>
        Array.from(panel.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)).filter(
            isVisibleForFocus,
        );

    const setToggleState = (expanded: boolean) => {
        toggle.setAttribute("aria-expanded", String(expanded));
        toggle.setAttribute("aria-label", expanded ? labels.close : labels.open);
    };

    const setBackgroundInert = (value: boolean) => {
        inertTargets.forEach((target) => {
            target.inert = value;
            if (value) {
                target.setAttribute("aria-hidden", "true");
                return;
            }
            target.removeAttribute("aria-hidden");
        });
    };

    const setModalSemantics = (value: boolean) => {
        if (value) {
            panel.setAttribute("role", "dialog");
            panel.setAttribute("aria-modal", "true");
            panel.setAttribute("tabindex", "-1");
            return;
        }
        panel.removeAttribute("role");
        panel.removeAttribute("aria-modal");
        panel.removeAttribute("tabindex");
    };

    const syncClosedState = () => {
        panel.classList.remove("open");
        scrim?.classList.remove("open");
        setToggleState(false);
        setModalSemantics(false);

        if (media.matches) {
            panel.setAttribute("aria-hidden", "true");
            panel.inert = true;
            return;
        }

        panel.removeAttribute("aria-hidden");
        panel.inert = false;
    };

    const closeDrawer = (restoreFocus = false) => {
        open = false;
        setBackgroundInert(false);
        doc.body.style.overflow = "";
        syncClosedState();
        if (restoreFocus) {
            toggle.focus({ preventScroll: true });
        }
    };

    const openDrawer = () => {
        open = true;
        panel.classList.add("open");
        scrim?.classList.add("open");
        setToggleState(true);
        setModalSemantics(true);
        panel.removeAttribute("aria-hidden");
        panel.inert = false;
        setBackgroundInert(true);
        doc.body.style.overflow = "hidden";

        const firstFocusable = getFocusable()[0];
        if (firstFocusable) {
            firstFocusable.focus({ preventScroll: true });
            return;
        }

        panel.focus({ preventScroll: true });
    };

    const handleKeydown = (event: KeyboardEvent) => {
        if (!open || !media.matches) {
            return;
        }

        if (event.key === "Escape") {
            closeDrawer(true);
            return;
        }

        if (event.key !== "Tab") {
            return;
        }

        const focusable = getFocusable();
        if (!focusable.length) {
            event.preventDefault();
            panel.focus({ preventScroll: true });
            return;
        }

        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        const active = doc.activeElement;
        if (event.shiftKey && (active === first || active === panel)) {
            event.preventDefault();
            last.focus({ preventScroll: true });
            return;
        }

        if (!event.shiftKey && active === last) {
            event.preventDefault();
            first.focus({ preventScroll: true });
        }
    };

    const handleViewportChange = () => {
        if (!media.matches) {
            closeDrawer(false);
            return;
        }

        if (!open) {
            syncClosedState();
        }
    };

    syncClosedState();

    toggle.addEventListener("click", () => {
        if (!media.matches) {
            return;
        }

        if (open) {
            closeDrawer(true);
            return;
        }

        openDrawer();
    });
    scrim?.addEventListener("click", () => closeDrawer(true));
    panel.querySelectorAll<HTMLAnchorElement>("a").forEach((link) => {
        link.addEventListener("click", () => closeDrawer());
    });
    doc.addEventListener("keydown", handleKeydown);
    media.addEventListener("change", handleViewportChange);
}
