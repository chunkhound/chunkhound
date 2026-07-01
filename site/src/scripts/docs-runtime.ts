import "./copy-handler.js";

function enhanceTOC(): void {
    const toc = document.querySelector<HTMLElement>("[data-toc]");
    const content = document.querySelector<HTMLElement>(".docs-content");
    if (!toc || !content) {
        return;
    }

    const headings = content.querySelectorAll<HTMLHeadingElement>("h2, h3");

    headings.forEach((heading) => {
        if (!heading.id) {
            heading.id = heading.textContent
                ?.trim()
                .toLowerCase()
                .replace(/[^a-z0-9]+/g, "-")
                .replace(/^-|-$/g, "") || "";
        }

        if (!heading.querySelector(".heading-link")) {
            const label = heading.textContent?.trim() || "section";
            const button = document.createElement("button");
            button.className = "heading-link";
            button.type = "button";
            button.setAttribute("aria-label", `Copy link to ${label}`);
            button.textContent = "#";
            button.addEventListener("click", () => {
                const url = `${location.origin}${location.pathname}#${heading.id}`;
                navigator.clipboard.writeText(url).then(() => {
                    button.textContent = "\u2713";
                    window.setTimeout(() => {
                        button.textContent = "#";
                    }, 1500);
                });
            });
            heading.appendChild(button);
        }
    });

    const tocLinks = toc.querySelectorAll<HTMLAnchorElement>(".toc-link");
    if (!tocLinks.length) {
        return;
    }

    const observer = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (!entry.isIntersecting) {
                    return;
                }

                tocLinks.forEach((link) => link.classList.remove("active"));
                const active = toc.querySelector<HTMLAnchorElement>(
                    `a[href="#${entry.target.id}"]`,
                );
                active?.classList.add("active");
            });
        },
        { rootMargin: "-80px 0px -70% 0px", threshold: 0 },
    );

    headings.forEach((heading) => observer.observe(heading));
}

function initNavFilter(): void {
    const filter = document.querySelector<HTMLInputElement>("[data-docs-nav-filter]");
    if (!filter) {
        return;
    }

    filter.addEventListener("input", () => {
        const query = filter.value.trim().toLowerCase();
        const links = document.querySelectorAll<HTMLElement>("[data-sidebar-link]");
        const sections = document.querySelectorAll<HTMLElement>("[data-sidebar-section]");

        links.forEach((link) => {
            const text = link.textContent?.toLowerCase() || "";
            link.style.display = !query || text.includes(query) ? "" : "none";
        });

        sections.forEach((section) => {
            const visibleLinks = section.querySelectorAll<HTMLElement>("[data-sidebar-link]");
            const anyVisible = Array.from(visibleLinks).some(
                (link) => link.style.display !== "none",
            );
            section.style.display = anyVisible ? "" : "none";
        });
    });
}

export function initMobileNav(doc: Document = document): void {
    const toggle = doc.querySelector<HTMLButtonElement>("[data-docs-nav-toggle]");
    const sidebar = doc.getElementById("docs-sidebar");
    const scrim = doc.querySelector<HTMLElement>("[data-docs-nav-scrim]");
    if (!toggle || !sidebar || typeof window === "undefined") {
        return;
    }

    const mobileMedia = window.matchMedia("(max-width: 900px)");
    const inertTargets = Array.from(
        doc.querySelectorAll<HTMLElement>("[data-docs-mobile-inert]"),
    );
    const focusableSelector = [
        "a[href]",
        "button:not([disabled])",
        "input:not([disabled])",
        "select:not([disabled])",
        "textarea:not([disabled])",
        "[tabindex]:not([tabindex='-1'])",
    ].join(", ");
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
        Array.from(sidebar.querySelectorAll<HTMLElement>(focusableSelector)).filter(
            isVisibleForFocus,
        );

    const setToggleState = (expanded: boolean) => {
        toggle.setAttribute("aria-expanded", String(expanded));
        toggle.setAttribute(
            "aria-label",
            expanded ? "Close docs menu" : "Open docs menu",
        );
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
            sidebar.setAttribute("role", "dialog");
            sidebar.setAttribute("aria-modal", "true");
            sidebar.setAttribute("tabindex", "-1");
            return;
        }
        sidebar.removeAttribute("role");
        sidebar.removeAttribute("aria-modal");
        sidebar.removeAttribute("tabindex");
    };

    const syncClosedState = () => {
        sidebar.classList.remove("open");
        scrim?.classList.remove("open");
        setToggleState(false);
        setModalSemantics(false);

        if (mobileMedia.matches) {
            sidebar.setAttribute("aria-hidden", "true");
            sidebar.inert = true;
            return;
        }

        sidebar.removeAttribute("aria-hidden");
        sidebar.inert = false;
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
        sidebar.classList.add("open");
        scrim?.classList.add("open");
        setToggleState(true);
        setModalSemantics(true);
        sidebar.removeAttribute("aria-hidden");
        sidebar.inert = false;
        setBackgroundInert(true);
        doc.body.style.overflow = "hidden";

        const firstFocusable = getFocusable()[0];
        if (firstFocusable) {
            firstFocusable.focus({ preventScroll: true });
            return;
        }

        sidebar.focus({ preventScroll: true });
    };

    const handleKeydown = (event: KeyboardEvent) => {
        if (!open || !mobileMedia.matches) {
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
            sidebar.focus({ preventScroll: true });
            return;
        }

        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        const active = doc.activeElement;
        if (event.shiftKey && (active === first || active === sidebar)) {
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
        if (!mobileMedia.matches) {
            closeDrawer(false);
            return;
        }

        if (!open) {
            syncClosedState();
        }
    };

    syncClosedState();

    toggle.addEventListener("click", () => {
        if (!mobileMedia.matches) {
            return;
        }

        if (open) {
            closeDrawer(true);
            return;
        }

        openDrawer();
    });
    scrim?.addEventListener("click", () => closeDrawer(true));
    sidebar.querySelectorAll<HTMLAnchorElement>("a").forEach((link) => {
        link.addEventListener("click", () => closeDrawer());
    });
    doc.addEventListener("keydown", handleKeydown);
    mobileMedia.addEventListener("change", handleViewportChange);
}

async function initMermaid(): Promise<void> {
    const codes = document.querySelectorAll<HTMLElement>("pre code.language-mermaid");
    if (!codes.length) {
        return;
    }

    const figures: HTMLElement[] = [];
    codes.forEach((code) => {
        const pre = code.parentElement;
        if (!pre) {
            return;
        }

        const figure = document.createElement("div");
        figure.className = "mermaid-figure mermaid";
        figure.textContent = code.textContent || "";
        pre.replaceWith(figure);
        figures.push(figure);
    });

    if (!figures.length) {
        return;
    }

    const mermaidModule = await import("mermaid");
    const mermaid = mermaidModule.default;
    mermaid.initialize({
        startOnLoad: false,
        theme: "base",
        themeVariables: {
            primaryColor: "#164e63",
            primaryTextColor: "#f4f4f4",
            primaryBorderColor: "#0e7490",
            lineColor: "#22d3ee",
            secondaryColor: "#3c3c3c",
            tertiaryColor: "#4f4f4f",
        },
    });
    await mermaid.run({ nodes: figures });
}

function initSearchShortcut(): void {
    document.addEventListener("keydown", (event) => {
        if ((event.metaKey || event.ctrlKey) && event.key === "k") {
            event.preventDefault();
        }
    });
}

export async function initDocsRuntime(doc: Document = document): Promise<void> {
    enhanceTOC();
    initNavFilter();
    initMobileNav(doc);
    initSearchShortcut();
    await initMermaid();
}

if (typeof document !== "undefined") {
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", () => {
            void initDocsRuntime(document);
        });
    } else {
        void initDocsRuntime(document);
    }
}
