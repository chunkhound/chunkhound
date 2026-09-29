import {
  buildCompactConfiguratorOutput,
  buildFullConfiguratorOutput,
  DEFAULT_AGENT,
  DEFAULT_RESEARCH,
  DEFAULT_RETRIEVAL,
  findEmbeddingProvider,
  findLlmProvider,
  highlightInlineShellLine,
  RERANK_URL_ERROR_ID,
  rerankerState,
  rerankerStatus,
  rerankerSummary,
  rerankerValidationError,
  type RerankerErrorField,
  type ConfiguratorEmbeddingProviderOption,
  type ConfiguratorPlatform,
  type ConfiguratorProviderOption,
  type ConfiguratorReranker,
  isRerankFormat,
} from "../components/configurator/index.ts";
import { bindRovingKeys } from "./configurator/roving-focus.ts";
import {
  applyPlatformToCodeBlocks,
  applyPlatformToSelector,
  initPlatformSelectors,
  isConfiguratorPlatform,
  loadPlatformPreference,
} from "./configurator/platform-state.ts";

function updateRerankerValidity(
  inputs: Array<HTMLInputElement | HTMLSelectElement | null>,
  error: string | undefined,
  field: RerankerErrorField | undefined,
): void {
  // The shared error <p> describes all three inputs while an error is shown.
  inputs.forEach((input) => {
    if (error) input?.setAttribute("aria-describedby", RERANK_URL_ERROR_ID);
    else input?.removeAttribute("aria-describedby");
  });
  // Only the offending input is announced invalid: a model-required error
  // must not mark the URL field invalid.
  const [urlInput, formatInput, modelInput] = inputs;
  urlInput?.setAttribute("aria-invalid", String(field === "url"));
  formatInput?.setAttribute("aria-invalid", String(field === "format"));
  modelInput?.setAttribute("aria-invalid", String(field === "model"));
}

function rerankerHint(retrieval: ConfiguratorProviderOption): string {
  return typeof retrieval.config.base_url === "string"
    ? "Absolute URLs are used as-is. Relative paths use this preset’s base URL."
    : "Use an absolute HTTP(S) URL for this preset.";
}

function updateRerankerError(section: Element, error: string | undefined): void {
  const element = section.querySelector<HTMLElement>("[data-rerank-url-error]");
  if (error) element?.removeAttribute("hidden");
  else element?.setAttribute("hidden", "");
  if (element) element.textContent = error ?? "";
}

/**
 * A required reranker is not a choice to reveal: it is the pick's own field, so
 * its form opens itself and its summary is hidden (configurator CSS), leaving
 * nothing to collapse. Leaving "required" collapses it again — the forced-open
 * state belongs to that state, not to "included", whose disclosure is the
 * user's. The caller writes `data-rerank-state` before calling this, so the
 * toggle guard below already sees the new state on either event timing.
 */
function syncRerankerDisclosure(
  endpoint: Element | null,
  previous: string | null,
  state: string,
): void {
  const details = endpoint?.querySelector<HTMLDetailsElement>("[data-rerank-details]");
  if (!details) return;
  if (state === "required") details.open = true;
  else if (previous === "required") details.open = false;
}

// Label copy keyed by selector so the disclosure body and the error path
// share one source of truth for the reranker's visible text.
function rerankerLabelTexts(
  retrieval: ConfiguratorEmbeddingProviderOption,
  state: string,
): Array<[string, string]> {
  return [
    ["[data-rerank-state-label]", state === "required" ? "Required dependency" : ""],
    ["[data-rerank-status]", rerankerStatus(retrieval)],
    ["[data-rerank-details-summary]", rerankerSummary(retrieval)],
    ["[data-rerank-url-hint]", rerankerHint(retrieval)],
  ];
}

function updateRerankerText(
  section: Element,
  retrieval: ConfiguratorEmbeddingProviderOption,
  error: string | undefined,
): void {
  const state = rerankerState(retrieval);
  const endpoint = section.querySelector<HTMLElement>("[data-rerank-endpoint]");
  const previous = endpoint?.getAttribute("data-rerank-state") ?? null;
  endpoint?.setAttribute("data-rerank-state", state);
  syncRerankerDisclosure(endpoint, previous, state);
  rerankerLabelTexts(retrieval, state).forEach(([selector, text]) => {
    const element = section.querySelector<HTMLElement>(selector);
    if (element) element.textContent = text;
  });
  updateRerankerError(section, error);
}

// Mirrors the closure the tab list previously built inline: it sets the
// selected tab/panel state, then focuses only when activated via keyboard.
function activateStageTab(
  section: Element,
  tabs: readonly HTMLElement[],
  tab: HTMLElement,
  focus: boolean,
): void {
  tabs.forEach((other) => {
    const active = other === tab;
    other.setAttribute("aria-selected", String(active));
    other.tabIndex = active ? 0 : -1;
    const panel = section.querySelector<HTMLElement>(
      `[data-stage-panel="${other.dataset.stageTab}"]`,
    );
    if (panel) panel.hidden = !active;
  });
  if (focus) tab.focus();
}

function initStageTabs(section: Element): void {
  const tabs = Array.from(
    section.querySelectorAll<HTMLElement>("[data-stage-tab]"),
  );
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => activateStageTab(section, tabs, tab, false));
    bindRovingKeys(
      tab,
      () => tabs,
      (next) => activateStageTab(section, tabs, next, true),
    );
  });
}

interface RerankerInputs {
  url: HTMLInputElement | null;
  format: HTMLSelectElement | null;
  model: HTMLInputElement | null;
}

function rerankerInputs(section: Element): RerankerInputs {
  return {
    url: section.querySelector<HTMLInputElement>("[data-rerank-url]"),
    format: section.querySelector<HTMLSelectElement>("[data-rerank-format]"),
    model: section.querySelector<HTMLInputElement>("[data-rerank-model]"),
  };
}

function rerankerValue(
  url: string,
  format: string,
  model: string,
  error: string | undefined,
): ConfiguratorReranker | undefined {
  if (error) return undefined;
  // The format <select> only offers valid options, so a non-empty value is
  // already a known format — the guard narrows without a cast.
  if (!url && !format && !model) return undefined;
  const value: ConfiguratorReranker = { url, ...(model && { model }) };
  // The format <select> only offers valid options, so a non-empty value is
  // already a known format — the guard narrows without a cast.
  if (format) {
    if (!isRerankFormat(format)) return undefined;
    value.format = format;
  }
  return value;
}

type RerankerUpdate = (
  retrieval: ConfiguratorEmbeddingProviderOption,
) => ConfiguratorReranker | undefined;

function updateReranker(
  section: Element,
  inputs: RerankerInputs,
  retrieval: ConfiguratorEmbeddingProviderOption,
): ConfiguratorReranker | undefined {
  const url = inputs.url?.value.trim() ?? "";
  const format = inputs.format?.value.trim() ?? "";
  const model = inputs.model?.value.trim() ?? "";
  const validation = rerankerValidationError(retrieval, url, format, model);
  const error = validation?.message;
  const fields = [inputs.url, inputs.format, inputs.model];
  updateRerankerValidity(fields, error, validation?.field);
  updateRerankerText(section, retrieval, error);
  return rerankerValue(url, format, model, error);
}

function createRerankerController(
  section: Element,
  inputs: RerankerInputs,
): RerankerUpdate {
  return (retrieval) => updateReranker(section, inputs, retrieval);
}

// HTML `data-foo-bar` maps to `dataset.fooBar`; slicing "data-" alone returns
// undefined for hyphenated names, so convert the attribute properly.
function datasetKey(attribute: string): string {
  return attribute
    .slice(5)
    .replace(/-([a-z])/g, (_, char: string) => char.toUpperCase());
}

function selectedId(
  section: Element,
  attribute: string,
  fallback: string,
): string {
  return (
    section.querySelector<HTMLElement>(`[${attribute}]:checked`)?.dataset[
      datasetKey(attribute)
    ] ?? fallback
  );
}

function selectOption(
  section: Element,
  attribute: string,
  id: string,
): void {
  section
    .querySelectorAll<HTMLInputElement>(`[${attribute}]`)
    .forEach((input) => {
      // Native radio inputs: `checked` alone carries the state — no ARIA
      // mirror needed (the old aria-checked write conflicted with it).
      input.checked = input.dataset[datasetKey(attribute)] === id;
    });
}

function bindSelection(
  section: Element,
  attribute: string,
  select: (id: string) => void,
  rerender: () => void,
): void {
  section
    .querySelectorAll<HTMLInputElement>(`[${attribute}]`)
    .forEach((input) => {
      input.addEventListener("change", () => {
        const id = input.dataset[datasetKey(attribute)];
        if (!id) return;
        select(id);
        selectOption(section, attribute, id);
        rerender();
      });
    });
}

function updateSetupCommands(
  section: Element,
  ...hints: Array<string | undefined>
): void {
  const callout = section.querySelector<HTMLElement>("#prerequisite-callout");
  if (!callout) return;
  const commands = hints.filter(Boolean).join("\n");
  const copy = callout.querySelector<HTMLElement>("#prerequisite-copy-btn");
  const code = callout.querySelector<HTMLElement>("code");
  if (!commands) {
    callout.setAttribute("hidden", "");
    copy?.removeAttribute("data-copy");
    return;
  }
  callout.removeAttribute("hidden");
  if (code)
    code.innerHTML = commands.split("\n").map(highlightInlineShellLine).join("\n");
  copy?.setAttribute("data-copy", commands);
}

function renderOutput(
  context: ConfiguratorCore,
  reranker: ConfiguratorReranker | undefined,
): void {
  const { mode, state, output, copyButton } = context;
  const retrieval = findEmbeddingProvider(state.retrievalId);
  const research = findLlmProvider(state.researchId);
  const build =
    mode === "full" ? buildFullConfiguratorOutput : buildCompactConfiguratorOutput;
  const rendered = build(retrieval, research, state.agentId, state.platform, reranker);
  output.innerHTML = rendered.html;
  copyButton.setAttribute("data-copy", rendered.copy);
}

function placeReranker(section: Element): void {
  const reranker = section.querySelector<HTMLElement>("[data-rerank-endpoint]");
  if (!reranker) return;
  const target = section
    .querySelector<HTMLInputElement>("[data-retrieval]:checked")
    ?.closest<HTMLElement>(".provider-option")
    ?.querySelector<HTMLElement>(".option-reveal-content");
  if (target && reranker.parentElement !== target) target.append(reranker);
}

function createRenderSelection(context: ConfiguratorCore): () => void {
  return () => {
    const retrieval = findEmbeddingProvider(context.state.retrievalId);
    const research = findLlmProvider(context.state.researchId);
    // Relocate first: the reranker belongs to the checked row, and moving it
    // synchronously keeps it inside the reveal that opens this same tick.
    placeReranker(context.section);
    const reranker = context.reranker(retrieval);
    renderOutput(context, reranker);
    updateSetupCommands(context.section, retrieval.setupHint, research.setupHint);
  };
}

interface ConfiguratorCore {
  section: Element;
  output: HTMLElement;
  copyButton: HTMLElement;
  mode: string;
  state: {
    agentId: string;
    retrievalId: string;
    researchId: string;
    platform: ConfiguratorPlatform;
  };
  inputs: RerankerInputs;
  reranker: RerankerUpdate;
}

interface ConfiguratorContext extends ConfiguratorCore {
  renderSelection: () => void;
}

function initialState(section: Element): ConfiguratorContext["state"] {
  return {
    agentId: selectedId(section, "data-agent", DEFAULT_AGENT),
    retrievalId: selectedId(section, "data-retrieval", DEFAULT_RETRIEVAL),
    researchId: selectedId(section, "data-research", DEFAULT_RESEARCH),
    platform: loadPlatformPreference(),
  };
}

function createContext(
  section: Element,
  output: HTMLElement,
  copyButton: HTMLElement,
): ConfiguratorContext {
  const inputs = rerankerInputs(section);
  const base = {
    section,
    output,
    copyButton,
    mode: section.getAttribute("data-mode") || "compact",
    state: initialState(section),
    inputs,
    reranker: createRerankerController(section, inputs),
  };
  return { ...base, renderSelection: createRenderSelection(base) };
}

function bindRerankerInputs(context: ConfiguratorContext): void {
  const { url, format, model } = context.inputs;
  url?.addEventListener("input", context.renderSelection);
  format?.addEventListener("change", context.renderSelection);
  model?.addEventListener("input", context.renderSelection);
}

/**
 * A required reranker's form has no summary left to reopen it with, so a close
 * arriving from anywhere but this module — an assistive technology, future code
 * — would strand the user outside a field the pick exists to configure.
 * Reopen on the spot; the second `toggle` is not fired because nothing changed.
 */
function guardRequiredRerankerOpen(section: Element): void {
  const details = section.querySelector<HTMLDetailsElement>("[data-rerank-details]");
  const endpoint = details?.closest<HTMLElement>("[data-rerank-endpoint]");
  if (!details || !endpoint) return;
  details.addEventListener("toggle", () => {
    if (!details.open && endpoint.getAttribute("data-rerank-state") === "required") {
      details.open = true;
    }
  });
}

function clearRerankerInputs(inputs: RerankerInputs): void {
  if (inputs.url) inputs.url.value = "";
  if (inputs.format) inputs.format.value = "";
  if (inputs.model) inputs.model.value = "";
}

function bindStageOption(
  section: Element,
  context: ConfiguratorContext,
  role: "agentId" | "retrievalId" | "researchId",
  attribute: string,
): void {
  // A pick repaints the terminal/copy first — the reveal itself is CSS
  // (`:has(input:checked)`), so no layout work belongs here.
  bindSelection(
    section,
    attribute,
    (id) => {
      // The one shared editor must not carry a previous retrieval's override.
      if (role === "retrievalId" && context.state.retrievalId !== id) {
        clearRerankerInputs(context.inputs);
      }
      context.state[role] = id;
    },
    () => context.renderSelection(),
  );
}

function bindConfiguratorEvents(
  section: Element,
  context: ConfiguratorContext,
): void {
  initStageTabs(section);
  bindStageOption(section, context, "agentId", "data-agent");
  bindStageOption(section, context, "retrievalId", "data-retrieval");
  bindStageOption(section, context, "researchId", "data-research");
  bindRerankerInputs(context);
  guardRequiredRerankerOpen(section);
  initProviderSearch(section);
}

function filterProviderOptions(
  section: Element,
  role: string,
  query: string,
): number {
  let visible = 0;
  section.querySelectorAll<HTMLElement>(`[data-provider-option="${role}"]`).forEach((option) => {
    const matches = !query || (option.dataset.searchText || "").includes(query);
    option.hidden = !matches;
    visible += Number(matches);
  });
  section.querySelectorAll<HTMLElement>(`[data-provider-group="${role}"]`).forEach((group) => {
    group.hidden = !group.querySelector("[data-provider-option]:not([hidden])");
  });
  return visible;
}

function initProviderSearch(section: Element): void {
  section
    .querySelectorAll<HTMLInputElement>("[data-provider-search]")
    .forEach((input) => {
      input.addEventListener("input", () => {
        const role = input.dataset.providerSearch;
        if (!role) return;
        const query = input.value.toLowerCase().trim();
        const visible = filterProviderOptions(section, role, query);
        const status = section.querySelector<HTMLElement>(
          `[data-search-status="${role}"]`,
        );
        if (status) {
          const noun = visible === 1 ? "result" : "results";
          status.textContent = visible
            ? `${visible} ${noun}`
            : "No providers found";
        }
      });
    });
}

function watchPlatformChanges(context: ConfiguratorContext): void {
  if (typeof document === "undefined") return;
  document.addEventListener("chunkhound:platform-change", (event: Event) => {
    const next = (event as CustomEvent<{ platform: ConfiguratorPlatform }>)
      .detail?.platform;
    if (!isConfiguratorPlatform(next)) return;
    context.state.platform = next;
    applyPlatformToSelector(context.section, next);
    context.renderSelection();
  });
}

export function initConfigurator(section: Element): void {
  const output = section.querySelector<HTMLElement>("#config-output");
  const copyButton = section.querySelector<HTMLElement>("#config-copy-btn");
  if (!output || !copyButton) {
    // Only reachable with broken markup — say so instead of failing silently.
    console.warn(
      "configurator: section missing #config-output or #config-copy-btn, init skipped",
    );
    return;
  }
  const context = createContext(section, output, copyButton);
  bindConfiguratorEvents(section, context);
  applyPlatformToSelector(section, context.state.platform);
  watchPlatformChanges(context);
  context.renderSelection();
}


if (typeof document !== "undefined") {
  initPlatformSelectors(document);
  const platform = loadPlatformPreference();
  applyPlatformToSelector(document, platform);
  applyPlatformToCodeBlocks(platform);
  document.querySelectorAll(".configurator").forEach(initConfigurator);
}
