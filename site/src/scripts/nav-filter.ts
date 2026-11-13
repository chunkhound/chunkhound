/**
 * Client-side filter for the shared documentation navigation (docs sidebar
 * and the mobile menu). Matching is per guide group: a query that names a
 * section ("pi-schematic") keeps that whole group, otherwise links match on
 * their title. A group left with no visible link collapses, so no bare
 * heading survives a query that matches nothing.
 */
function applySectionFilter(section: HTMLElement, query: string): void {
    const title =
        section
            .querySelector("[data-sidebar-section-title]")
            ?.textContent?.toLowerCase() ?? "";
    const groupMatches = query !== "" && title.includes(query);
    const links = Array.from(
        section.querySelectorAll<HTMLElement>("[data-sidebar-link]"),
    );

    links.forEach((link) => {
        const text = link.textContent?.toLowerCase() ?? "";
        const matches = groupMatches || !query || text.includes(query);
        link.style.display = matches ? "" : "none";
    });

    const anyVisible = links.some((link) => link.style.display !== "none");
    section.style.display = anyVisible ? "" : "none";
}

export function initNavFilter(doc: Document = document): void {
    const filter = doc.querySelector<HTMLInputElement>("[data-docs-nav-filter]");
    if (!filter) {
        return;
    }

    filter.addEventListener("input", () => {
        const query = filter.value.trim().toLowerCase();
        doc.querySelectorAll<HTMLElement>("[data-sidebar-section]").forEach(
            (section) => {
                applySectionFilter(section, query);
            },
        );
    });
}
