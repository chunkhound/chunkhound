import "./copy-handler.js";
import { MOBILE_INERT_ATTR, initMobileDrawer } from "./mobile-drawer";
import { initNavFilter } from "./nav-filter";

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
                // Clipboard is undefined on insecure origins: guard it like
                // copy-handler.js does, or the sync throw skips .catch and
                // the heading anchor never shows the failure glyph.
                if (!navigator.clipboard) {
                    button.textContent = "\u00d7";
                    window.setTimeout(() => {
                        button.textContent = "#";
                    }, 1500);
                    return;
                }
                navigator.clipboard.writeText(url).then(() => {
                    button.textContent = "\u2713";
                    window.setTimeout(() => {
                        button.textContent = "#";
                    }, 1500);
                }).catch(() => {
                    // Clipboard access can be denied (insecure origin or
                    // permissions policy). Show it rather than a silent no-op;
                    // the heading anchor still works for manual copying.
                    button.textContent = "\u00d7";
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

    const linkedIds = new Set(Array.from(tocLinks, (link) => link.getAttribute("href")));
    headings.forEach((heading) => {
        if (linkedIds.has(`#${heading.id}`)) {
            observer.observe(heading);
        }
    });
}

export function initMobileNav(doc: Document = document): void {
    const toggle = doc.querySelector<HTMLButtonElement>("[data-nav-toggle]");
    const panel = doc.getElementById("docs-sidebar");
    if (!toggle || !panel || typeof window === "undefined") {
        return;
    }

    initMobileDrawer({
        toggle,
        panel,
        scrim: doc.querySelector<HTMLElement>("[data-docs-nav-scrim]"),
        media: window.matchMedia("(max-width: 900px)"),
        inertTargets: Array.from(
            doc.querySelectorAll<HTMLElement>(`[${MOBILE_INERT_ATTR}]`),
        ),
        labels: { open: "Open docs menu", close: "Close docs menu" },
    });
}

/** ⌘K / Ctrl+K focuses the sidebar filter input. */
function initSearchShortcut(doc: Document = document): void {
    const filter = doc.querySelector<HTMLInputElement>("[data-docs-nav-filter]");
    if (!filter) return;
    doc.addEventListener("keydown", (event) => {
        if ((event.metaKey || event.ctrlKey) && event.key === "k") {
            event.preventDefault();
            filter.focus();
            filter.select();
        }
    });
}

export async function initDocsRuntime(doc: Document = document): Promise<void> {
    enhanceTOC();
    initNavFilter(doc);
    initMobileNav(doc);
    initSearchShortcut(doc);
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
