import type {
  ConfiguratorPlatform,
  ConfiguratorProviderGroup,
  PlatformOption,
} from "./types.ts";

// Public group identifiers for provider picker sections. Order defines the
// rendering order in popovers.
export const PROVIDER_GROUP_ORDER: readonly ConfiguratorProviderGroup[] = [
  "cloud",
  "local",
  "agent",
];

export const DEFAULT_AGENT = "pi";
// Default to VoyageAI for the strongest out-of-box retrieval results, ROI, and
// setup ease; Ollama and vLLM remain explicit choices for a local data boundary.
export const DEFAULT_RETRIEVAL = "voyageai";
export const DEFAULT_RESEARCH = "openrouter";
export const DEFAULT_PLATFORM: ConfiguratorPlatform = "posix";
export const PLATFORM_STORAGE_KEY = "chunkhound:platform";
export const PLATFORM_OPTIONS: PlatformOption[] = [
  { id: "posix", label: "macOS/Linux" },
  { id: "powershell", label: "PowerShell" },
];

export const INDEX_CMD = "chunkhound index .";
export const CONFIG_FILENAME = ".chunkhound.json";

export const CONFIGURATION_DOCS_URL = "https://chunkhound.ai/docs/configuration/";
