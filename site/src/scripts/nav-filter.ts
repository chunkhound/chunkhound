/**
 * Client-side filter for the shared documentation navigation (docs sidebar
 * and the mobile menu). Hides guide links that do not match the query and
 * collapses sections that end up empty.
 */
export function initNavFilter(doc: Document = document): void {
    const filter = doc.querySelector<HTMLInputElement>("[data-docs-nav-filter]");
    if (!filter) {
        return;
    }

    filter.addEventListener("input", () => {
        const query = filter.value.trim().toLowerCase();
        const links = doc.querySelectorAll<HTMLElement>("[data-sidebar-link]");
        const sections = doc.querySelectorAll<HTMLElement>("[data-sidebar-section]");

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
