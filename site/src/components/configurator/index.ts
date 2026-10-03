// Curated public API — internal helpers stay private.
export type {
  ConfiguratorEditor,
  ConfiguratorProviderOption,
  ConfiguratorEmbeddingProviderOption,
  ConfiguratorPlatform,
  ConfiguratorReranker,
  ConfiguratorRerankFormat,
  ConfiguratorRequirement,
  PlatformOption,
  StageOption,
} from "./types.ts";
export { requirementLabels, renderRequirements, optionSearchText } from "./utils.ts";
export { requirements } from "./requirements.ts";
export {
  PROVIDER_GROUP_ORDER,
  DEFAULT_AGENT,
  DEFAULT_RETRIEVAL,
  DEFAULT_RESEARCH,
  DEFAULT_PLATFORM,
  PLATFORM_STORAGE_KEY,
  PLATFORM_OPTIONS,
} from "./constants.ts";
export {
  editors,
  embeddingProviders,
  llmProviders,
  findEditor,
  findEmbeddingProvider,
  findLlmProvider,
  groupProviders,
} from "./providers.ts";
export { highlightInlineShellLine } from "./highlight.ts";
export {
  RERANK_URL_ERROR_ID,
  rerankUrlValidationError,
  rerankerState,
  rerankerStateLabel,
  rerankerStatus,
  rerankerSummary,
  rerankerValidationError,
  isRerankFormat,
} from "./rerank-validate.ts";
export type { RerankerErrorField } from "./rerank-validate.ts";
export {
  buildCompactConfiguratorOutput,
  buildFullConfiguratorOutput,
  buildEditorCommands,
  buildChunkhoundConfig,
  configWithApiKey,
} from "./builders.ts";
