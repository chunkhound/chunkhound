import type { ConfiguratorEditor, ConfiguratorRequirement } from "./types.ts";

/** Escape tag delimiters only; quotes stay raw for downstream token wrapping. */
export function escapeHtmlTags(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

export function escapeHtml(s: string): string {
  return escapeHtmlTags(s).replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

export function requirementLabels(
  requirements: ConfiguratorRequirement[],
): string {
  return requirements.map(({ label }) => label).join(" · ");
}

export function optionSearchText(
  option: Pick<ConfiguratorEditor, "name" | "requirements"> & {
    description?: string;
  },
): string {
  return [
    option.name,
    option.description,
    requirementLabels(option.requirements),
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
}

export function renderRequirements(
  requirements: ConfiguratorRequirement[],
): string {
  return requirements
    .map(
      ({ label, url, svg, optional }) =>
        `<li class="prerequisite-item${optional ? " prerequisite-item-optional" : ""}"><a class="prerequisite-link" href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer"><span class="prerequisite-logo" aria-hidden="true">${svg}</span>${escapeHtml(label)}</a></li>`,
    )
    .join("");
}
