// Derived backend provider facts. DO NOT hand-edit provider-catalog.gen.json:
//   uv run python scripts/generate_site_provider_catalog.py
// Source of truth: chunkhound/core/config/{provider_registry,llm_config,embedding_config}.py
import type {
  ConfiguratorEmbeddingProviderOption,
  ConfiguratorProviderOption,
} from "./types.ts";
import payload from "./provider-catalog.gen.json";

const authUrls = new Map(
  payload.llm.map(({ id, authUrl }) => [id, authUrl] as const),
);

/** Canonical key portal from the registry, or undefined when it holds none. */
export function backendAuthUrl(providerId: string): string | undefined {
  return authUrls.get(providerId) || undefined;
}

/**
 * Every way the hand-authored presets disagree with the backend catalog.
 * Returned as readable lines so a failing guard names the exact fix.
 */
export function catalogDrift(
  llm: readonly ConfiguratorProviderOption[],
  embedding: readonly ConfiguratorEmbeddingProviderOption[],
): string[] {
  return [
    ...driftFor(payload.llm, llm, "LLM"),
    ...driftFor(
      payload.embedding.map((id) => ({ id, requiresModel: false })),
      embedding,
      "embedding",
    ),
  ];
}

interface ProviderFact {
  id: string;
  requiresModel: boolean;
  /** Non-official `base_url` also forces a model; absent for embeddings. */
  customEndpointRequiresModel?: boolean;
}

/** Mirrors chunkhound/core/utils/openai_utils.is_official_openai_endpoint. */
const isOfficialOpenAIBaseUrl = (baseUrl: unknown): boolean =>
  !baseUrl ||
  baseUrl === "https://api.openai.com" ||
  String(baseUrl).startsWith("https://api.openai.com/");

function driftFor(
  facts: readonly ProviderFact[],
  presets: readonly ConfiguratorProviderOption[],
  label: string,
): string[] {
  const known = new Map(facts.map((fact) => [fact.id, fact] as const));
  const target = (preset: ConfiguratorProviderOption) =>
    String(preset.config.provider);
  const targeted = new Set(presets.map(target));

  return [
    ...facts
      .filter(({ id }) => !targeted.has(id))
      .map(
        ({ id }) =>
          `backend ${label} provider '${id}' has no configurator preset ` +
          "(add one in providers.ts)",
      ),
    ...presets
      .filter((preset) => !known.has(target(preset)))
      .map(
        (preset) =>
          `preset '${preset.id}' targets unknown ${label} provider ` +
          `'${target(preset)}' (register it in the backend, or drop the preset)`,
      ),
    // One line per preset: a preset has one model-pin problem, not several.
    ...presets.flatMap((preset) => {
      const fact = known.get(target(preset));
      const baseUrl = preset.config.base_url;
      if (!fact || preset.config.model) return [];
      const reason = fact.requiresModel
        ? `'${fact.id}' has no backend default model`
        : fact.customEndpointRequiresModel &&
            baseUrl &&
            !isOfficialOpenAIBaseUrl(baseUrl)
          ? `custom OpenAI-compatible endpoint '${String(baseUrl)}' has no ` +
            "backend default (pin config.model in providers.ts)"
          : undefined;
      return reason ? [`preset '${preset.id}' must pin a model: ${reason}`] : [];
    }),
  ];
}
