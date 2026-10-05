export type ConfigRecord = Record<string, unknown>;

export interface ConfiguratorEditor {
  id: string;
  name: string;
  svg: string;
  mcpFile?: string;
  mcpFilePowerShell?: string;
  mcp?: ConfigRecord;
  rawCmd?: string;
  gitignoreEntries?: string[];
  requirements: ConfiguratorRequirement[];
}

/** Provider picker section; `PROVIDER_GROUP_ORDER` defines render order. */
export type ConfiguratorProviderGroup = "cloud" | "local" | "agent";

export interface ConfiguratorProviderOption {
  id: string;
  name: string;
  svg: string;
  description: string;
  config: ConfigRecord;
  apiKeyPlaceholder?: string;
  setupHint?: string;
  group: ConfiguratorProviderGroup;
  requirements: ConfiguratorRequirement[];
  /** Marks this option as the recommended default, rendered as a pill. */
  recommended?: boolean;
  /** Cheapest/fastest model for this provider, shown when it is selected. */
  recommendation?: string;
}

export type StageOption = ConfiguratorEditor | ConfiguratorProviderOption;

export interface ConfiguratorRerankerCapability {
  included: boolean;
  status?: string;
}

export interface ConfiguratorEmbeddingProviderOption
  extends ConfiguratorProviderOption {
  reranker: ConfiguratorRerankerCapability;
}

export interface ConfiguratorRequirement {
  id: string;
  label: string;
  url: string;
  svg: string;
  optional?: boolean;
}

export type ConfiguratorPlatform = "posix" | "powershell";
export type ConfiguratorRerankFormat = "cohere" | "tei" | "voyage";

export interface ConfiguratorReranker {
  url?: string;
  format?: ConfiguratorRerankFormat;
  model?: string;
}

export interface PlatformOption {
  id: ConfiguratorPlatform;
  label: string;
}
