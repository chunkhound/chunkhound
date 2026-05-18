import {
    buildCompactConfiguratorOutput,
    buildFullConfiguratorOutput,
    DEFAULT_PLATFORM,
    embeddingProviders,
    highlightInlineShellLine,
    llmProviders,
    PLATFORM_STORAGE_KEY,
    type ConfiguratorPlatform,
} from "../components/configurator-data";

type ProviderOption = (typeof embeddingProviders)[number];

const editorLlmDefaults: Record<string, string> = {
    "claude-code": "claude-code-cli",
    codex: "codex-cli",
    opencode: "opencode-cli",
};

function isConfiguratorPlatform(value: string | null): value is ConfiguratorPlatform {
    return value === "posix" || value === "powershell";
}

function loadPlatformPreference(): ConfiguratorPlatform {
    if (typeof window === "undefined") {
        return DEFAULT_PLATFORM;
    }

    try {
        const stored = window.localStorage.getItem(PLATFORM_STORAGE_KEY);
        if (isConfiguratorPlatform(stored)) {
            return stored;
        }
    } catch {
        // Ignore storage access failures and fall back to the default shell.
    }

    return DEFAULT_PLATFORM;
}

function persistPlatformPreference(platform: ConfiguratorPlatform): void {
    if (typeof window === "undefined") {
        return;
    }

    try {
        window.localStorage.setItem(PLATFORM_STORAGE_KEY, platform);
    } catch {
        // Ignore storage access failures.
    }
}

function applyPlatformToSelector(root: ParentNode, platform: ConfiguratorPlatform): void {
    root.querySelectorAll<HTMLElement>("[data-platform-option]").forEach((pill) => {
        const selected = pill.dataset.platformOption === platform;
        pill.classList.toggle("selected", selected);
        pill.setAttribute("aria-selected", selected ? "true" : "false");
        pill.setAttribute("tabindex", selected ? "0" : "-1");
    });
}

function applyPlatformToCodeBlocks(platform: ConfiguratorPlatform): void {
    if (typeof document === "undefined") {
        return;
    }

    document.querySelectorAll<HTMLElement>("[data-platform-code]").forEach((block) => {
        const visible = block.dataset.platformCode === platform;
        block.hidden = !visible;
        if (visible && block.dataset.platformCopy !== undefined) {
            const codeBlock = block.closest<HTMLElement>(".platform-code-block");
            const copyBtn = codeBlock?.querySelector<HTMLElement>(".copy-btn");
            if (copyBtn) copyBtn.dataset.copy = block.dataset.platformCopy;
        }
    });
}

function applyPlatform(platform: ConfiguratorPlatform, persist = false): void {
    if (persist) {
        persistPlatformPreference(platform);
    }

    if (typeof document === "undefined") {
        return;
    }

    applyPlatformToSelector(document, platform);
    applyPlatformToCodeBlocks(platform);
    document.dispatchEvent(
        new CustomEvent("chunkhound:platform-change", {
            detail: { platform },
        }),
    );
}

function initPlatformSelectors(root: ParentNode): void {
    root.querySelectorAll<HTMLElement>("[data-platform-option]").forEach((btn) => {
        btn.addEventListener("click", () => {
            const selectedPlatform = btn.dataset.platformOption;
            if (!isConfiguratorPlatform(selectedPlatform)) {
                return;
            }
            applyPlatform(selectedPlatform, true);
        });
        btn.addEventListener("keydown", (event: KeyboardEvent) => {
            if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
                return;
            }

            const tablist = btn.parentElement;
            if (!tablist) {
                return;
            }

            const buttons = Array.from(
                tablist.querySelectorAll<HTMLElement>("[data-platform-option]"),
            );
            const currentIndex = buttons.indexOf(btn);
            if (currentIndex < 0) {
                return;
            }

            event.preventDefault();

            let nextIndex = currentIndex;
            if (event.key === "ArrowLeft") {
                nextIndex =
                    currentIndex === 0 ? buttons.length - 1 : currentIndex - 1;
            } else if (event.key === "ArrowRight") {
                nextIndex =
                    currentIndex === buttons.length - 1 ? 0 : currentIndex + 1;
            } else if (event.key === "Home") {
                nextIndex = 0;
            } else if (event.key === "End") {
                nextIndex = buttons.length - 1;
            }

            const nextButton = buttons[nextIndex];
            const selectedPlatform = nextButton?.dataset.platformOption;
            if (!nextButton || !isConfiguratorPlatform(selectedPlatform)) {
                return;
            }

            applyPlatform(selectedPlatform, true);
            nextButton.focus();
        });
    });
}

export function initConfigurator(section: Element): void {
    const doc = typeof document !== "undefined" ? document : null;
    const output = section.querySelector<HTMLElement>("#config-output");
    const copyBtn = section.querySelector<HTMLElement>("#config-copy-btn");
    const rerankCallout = section.querySelector<HTMLElement>("#rerank-callout");
    const mode = section.getAttribute("data-mode") || "compact";

    if (!output || !copyBtn) return;

    let embeddingId =
        section.querySelector<HTMLElement>("[data-embedding].selected")?.dataset
            .embedding || "";
    let llmId =
        section.querySelector<HTMLElement>("[data-llm].selected")?.dataset.llm ||
        "";
    let editorId =
        section.querySelector<HTMLElement>("[data-editor].selected")?.dataset
            .editor || "";
    let platform = loadPlatformPreference();

    function findEmbedding(id: string): ProviderOption {
        const option = embeddingProviders.find((provider) => provider.id === id);
        if (!option) throw new Error(`Unknown embedding provider: ${id}`);
        return option;
    }

    function findLlm(id: string): ProviderOption {
        const option = llmProviders.find((provider) => provider.id === id);
        if (!option) throw new Error(`Unknown llm provider: ${id}`);
        return option;
    }

    function updateOutput(): void {
        const embedding = findEmbedding(embeddingId);
        const llm = findLlm(llmId);
        const rendered =
            mode === "full"
                ? buildFullConfiguratorOutput(embedding, llm, editorId, platform)
                : buildCompactConfiguratorOutput(embedding, llm, editorId, platform);
        output.innerHTML = rendered.html;
        copyBtn.setAttribute("data-copy", rendered.copy);
    }

    function updateSetupCallout(): void {
        if (!rerankCallout) return;
        const embedHint = findEmbedding(embeddingId).setupHint || "";
        const llmHint = findLlm(llmId).setupHint || "";
        const hints = [embedHint, llmHint].filter(Boolean).join("\n");
        const rerankCopyBtn =
            rerankCallout.querySelector<HTMLElement>("#rerank-copy-btn");
        const code = rerankCallout.querySelector<HTMLElement>("code");
        if (hints) {
            rerankCallout.removeAttribute("hidden");
            if (code) {
                code.innerHTML = hints
                    .split("\n")
                    .map(highlightInlineShellLine)
                    .join("\n");
            }
            rerankCopyBtn?.setAttribute("data-copy", hints);
            return;
        }

        rerankCallout.setAttribute("hidden", "");
        rerankCopyBtn?.removeAttribute("data-copy");
    }

    function selectPill(attr: string, btn: HTMLElement): void {
        section.querySelectorAll<HTMLElement>(`[${attr}]`).forEach((pill) => {
            pill.classList.remove("selected");
            pill.setAttribute("aria-checked", "false");
        });
        btn.classList.add("selected");
        btn.setAttribute("aria-checked", "true");
    }

    function selectLlmById(id: string): void {
        const llmBtn = section.querySelector<HTMLElement>(`[data-llm='${id}']`);
        if (!llmBtn) return;
        selectPill("data-llm", llmBtn);
        llmId = id;
    }

    section.querySelectorAll<HTMLElement>("[data-editor]").forEach((btn) => {
        btn.addEventListener("click", () => {
            const selectedEditorId = btn.dataset.editor;
            if (!selectedEditorId) return;
            selectPill("data-editor", btn);
            editorId = selectedEditorId;
            const defaultLlm = editorLlmDefaults[selectedEditorId];
            if (defaultLlm) selectLlmById(defaultLlm);
            updateOutput();
            updateSetupCallout();
        });
    });

    section.querySelectorAll<HTMLElement>("[data-embedding]").forEach((btn) => {
        btn.addEventListener("click", () => {
            const selectedEmbeddingId = btn.dataset.embedding;
            if (!selectedEmbeddingId) return;
            selectPill("data-embedding", btn);
            embeddingId = selectedEmbeddingId;
            updateOutput();
            updateSetupCallout();
        });
    });

    section.querySelectorAll<HTMLElement>("[data-llm]").forEach((btn) => {
        btn.addEventListener("click", () => {
            const selectedLlmId = btn.dataset.llm;
            if (!selectedLlmId) return;
            selectPill("data-llm", btn);
            llmId = selectedLlmId;
            updateOutput();
            updateSetupCallout();
        });
    });

    applyPlatformToSelector(section, platform);
    doc?.addEventListener("chunkhound:platform-change", (event: Event) => {
        const nextPlatform = (
            event as CustomEvent<{ platform: ConfiguratorPlatform }>
        ).detail?.platform;
        if (!isConfiguratorPlatform(nextPlatform)) {
            return;
        }
        platform = nextPlatform;
        applyPlatformToSelector(section, platform);
        updateOutput();
    });

    updateOutput();
    updateSetupCallout();
}

if (typeof document !== "undefined") {
    initPlatformSelectors(document);
    const initialPlatform = loadPlatformPreference();
    applyPlatformToSelector(document, initialPlatform);
    applyPlatformToCodeBlocks(initialPlatform);
    document.querySelectorAll(".configurator").forEach(initConfigurator);
}
