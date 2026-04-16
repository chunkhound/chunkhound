import "./copy-handler.js";

function buildTOC(): void {
    const toc = document.querySelector<HTMLElement>("[data-toc]");
    const content = document.querySelector<HTMLElement>(".docs-content");
    if (!toc || !content) {
        return;
    }

    const headings = content.querySelectorAll<HTMLHeadingElement>("h2, h3");
    const frag = document.createDocumentFragment();

    headings.forEach((heading) => {
        if (!heading.id) {
            heading.id = heading.textContent
                ?.trim()
                .toLowerCase()
                .replace(/[^a-z0-9]+/g, "-")
                .replace(/^-|-$/g, "") || "";
        }

        if (!heading.querySelector(".heading-link")) {
            const button = document.createElement("button");
            button.className = "heading-link";
            button.type = "button";
            button.setAttribute(
                "aria-label",
                `Copy link to ${heading.textContent?.trim() || "section"}`,
            );
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

        const link = document.createElement("a");
        link.className = "toc-link";
        link.href = `#${heading.id}`;
        link.textContent = heading.textContent?.replace(/#$/, "").trim() || "";
        link.setAttribute("data-depth", heading.tagName === "H3" ? "3" : "2");
        frag.appendChild(link);
    });

    toc.appendChild(frag);

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

function initMobileNav(): void {
    const toggle = document.querySelector<HTMLElement>("[data-docs-nav-toggle]");
    const sidebar = document.getElementById("docs-sidebar");
    const scrim = document.querySelector<HTMLElement>("[data-docs-nav-scrim]");
    if (!toggle || !sidebar) {
        return;
    }

    const open = () => {
        sidebar.classList.add("open");
        scrim?.classList.add("open");
    };

    const close = () => {
        sidebar.classList.remove("open");
        scrim?.classList.remove("open");
    };

    toggle.addEventListener("click", () => {
        if (sidebar.classList.contains("open")) {
            close();
        } else {
            open();
        }
    });
    scrim?.addEventListener("click", close);
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
            primaryColor: "#263100",
            primaryTextColor: "#f4f4f4",
            primaryBorderColor: "#5d7000",
            lineColor: "#a9c650",
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

async function initDocsRuntime(): Promise<void> {
    buildTOC();
    initNavFilter();
    initMobileNav();
    initSearchShortcut();
    await initMermaid();
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => {
        void initDocsRuntime();
    });
} else {
    void initDocsRuntime();
}
