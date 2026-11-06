import {
  DEFAULT_PLATFORM,
  PLATFORM_STORAGE_KEY,
  type ConfiguratorPlatform,
} from "../../components/configurator/index.ts";
import { bindRovingKeys } from "./roving-focus.ts";

export function isConfiguratorPlatform(
  value: string | null,
): value is ConfiguratorPlatform {
  return value === "posix" || value === "powershell";
}

export function loadPlatformPreference(): ConfiguratorPlatform {
  if (typeof window === "undefined") return DEFAULT_PLATFORM;
  try {
    const stored = window.localStorage.getItem(PLATFORM_STORAGE_KEY);
    return isConfiguratorPlatform(stored) ? stored : DEFAULT_PLATFORM;
  } catch {
    return DEFAULT_PLATFORM;
  }
}

export function persistPlatformPreference(
  platform: ConfiguratorPlatform,
): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(PLATFORM_STORAGE_KEY, platform);
  } catch {
    // Storage is an optional convenience, never a setup dependency.
  }
}

export function applyPlatformToSelector(
  root: ParentNode,
  platform: ConfiguratorPlatform,
): void {
  root
    .querySelectorAll<HTMLElement>("[data-platform-option]")
    .forEach((pill) => {
      const selected = pill.dataset.platformOption === platform;
      pill.classList.toggle("selected", selected);
      // PlatformCodeBlock declares the semantics per usage via
      // data-platform-mode: "tabs" owns real tabpanels, "switcher" toggles
      // a single panel's content and uses aria-pressed.
      const attr =
        pill.closest<HTMLElement>("[data-platform-mode]")?.dataset
          .platformMode === "tabs"
          ? "aria-selected"
          : "aria-pressed";
      pill.setAttribute(attr, String(selected));
      pill.setAttribute("tabindex", selected ? "0" : "-1");
    });
}

export function applyPlatformToCodeBlocks(
  platform: ConfiguratorPlatform,
): void {
  if (typeof document === "undefined") return;
  document
    .querySelectorAll<HTMLElement>("[data-platform-code]")
    .forEach((block) => {
      const visible = block.dataset.platformCode === platform;
      block.hidden = !visible;
      if (visible && block.dataset.platformCopy !== undefined) {
        block
          .closest<HTMLElement>(".platform-code-block")
          ?.querySelector<HTMLElement>(".copy-btn")
          ?.setAttribute("data-copy", block.dataset.platformCopy);
      }
    });
}

export function applyPlatform(
  platform: ConfiguratorPlatform,
  persist = false,
): void {
  if (persist) persistPlatformPreference(platform);
  if (typeof document === "undefined") return;
  applyPlatformToSelector(document, platform);
  applyPlatformToCodeBlocks(platform);
  document.dispatchEvent(
    new CustomEvent("chunkhound:platform-change", {
      detail: { platform },
    }),
  );
}

// Sibling pills share a DOM group; the roving handler needs that sibling list
// detached from the `forEach` closure so it reads live from each button.
function platformOptionButtons(button: HTMLElement): HTMLElement[] {
  return Array.from(
    button.parentElement?.querySelectorAll<HTMLElement>(
      "[data-platform-option]",
    ) ?? [],
  );
}

// Read a platform off a pill and apply it, returning whether the pill carried
// a valid platform. Lets click and roving handlers share one code path.
function applyPlatformFromButton(button: HTMLElement): boolean {
  const platform = button.dataset.platformOption ?? null;
  if (!isConfiguratorPlatform(platform)) return false;
  applyPlatform(platform, true);
  return true;
}

export function initPlatformSelectors(root: ParentNode): void {
  root
    .querySelectorAll<HTMLElement>("[data-platform-option]")
    .forEach((button) => {
      button.addEventListener("click", () => applyPlatformFromButton(button));
      bindRovingKeys(
        button,
        () => platformOptionButtons(button),
        (nextButton) => {
          if (applyPlatformFromButton(nextButton)) nextButton.focus();
        },
      );
    });
}
