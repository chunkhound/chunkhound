import type {
  ConfiguratorEditor,
  ConfiguratorEmbeddingProviderOption,
  ConfiguratorProviderGroup,
  ConfiguratorProviderOption,
} from "./types.ts";
import {
  ANTHROPIC_SVG,
  DEEPSEEK_SVG,
  GEMINI_SVG,
  GROK_SVG,
  OLLAMA_SVG,
  OPENAI_SVG,
  OPENCODE_SVG,
  OPENROUTER_SVG,
  ORCAROUTER_SVG,
  VERCEL_SVG,
  VLLM_SVG,
  VOYAGEAI_SVG,
} from "./icons.ts";
import { PROVIDER_GROUP_ORDER } from "./constants.ts";
import { requirements } from "./requirements.ts";

export const editors: ConfiguratorEditor[] = [
  {
    id: "pi",
    name: "Pi",
    svg: requirements.pi.svg,
    requirements: [requirements.pi],
    mcpFile: ".mcp.json",
    mcp: {
      mcpServers: {
        // requestTimeoutMs is ms (20 min for long research calls); no idleTimeout:
        // lifecycle "eager" keeps the server connected per pi-mcp-adapter docs.
        ChunkHound: {
          command: "chunkhound",
          args: ["mcp"],
          directTools: true,
          lifecycle: "eager",
          requestTimeoutMs: 1200000,
        },
      },
      settings: {
        outputGuard: false,
      },
    },
    installCommand: "pi install npm:pi-mcp-adapter",
    gitignoreEntries: [".mcp.json"],
  },
  {
    id: "cursor",
    name: "Cursor",
    svg: requirements.cursor.svg,
    requirements: [requirements.cursor],
    mcpFile: ".cursor/mcp.json",
    mcp: {
      mcpServers: {
        ChunkHound: {
          command: "chunkhound",
          args: ["mcp"],
        },
      },
    },
  },
  {
    id: "claude-code",
    name: "Claude Code",
    svg: requirements.claudeCode.svg,
    requirements: [requirements.claudeCode],
    rawCmd: "claude mcp add ChunkHound -- chunkhound mcp",
  },
  {
    id: "vscode",
    name: "VS Code",
    svg: requirements.vscode.svg,
    requirements: [requirements.vscode],
    mcpFile: ".vscode/mcp.json",
    mcp: {
      servers: {
        ChunkHound: {
          type: "stdio",
          command: "chunkhound",
          args: ["mcp"],
        },
      },
    },
  },
  {
    id: "opencode",
    name: "OpenCode",
    svg: requirements.opencode.svg,
    requirements: [requirements.opencode],
    mcpFile: "opencode.json",
    mcp: {
      mcp: {
        ChunkHound: {
          type: "local",
          command: ["chunkhound", "mcp"],
        },
      },
    },
  },
  {
    id: "codex",
    name: "Codex",
    svg: requirements.codex.svg,
    requirements: [requirements.codex],
    rawCmd: "codex mcp add ChunkHound -- chunkhound mcp",
  },
  {
    id: "windsurf",
    name: "Windsurf",
    svg: requirements.windsurf.svg,
    requirements: [requirements.windsurf],
    mcpFile: "~/.codeium/windsurf/mcp_config.json",
    mcpFilePowerShell: "$HOME/.codeium/windsurf/mcp_config.json",
    mcp: {
      mcpServers: {
        ChunkHound: {
          command: "chunkhound",
          args: ["mcp"],
        },
      },
    },
  },
  {
    id: "roo-code",
    name: "Roo Code",
    svg: requirements.rooCode.svg,
    requirements: [requirements.rooCode],
    mcpFile: ".roo/mcp.json",
    mcp: {
      mcpServers: {
        ChunkHound: {
          command: "chunkhound",
          args: ["mcp"],
        },
      },
    },
  },
  {
    id: "zed",
    name: "Zed",
    svg: requirements.zed.svg,
    requirements: [requirements.zed],
    mcpFile: ".zed/settings.json",
    mcp: {
      context_servers: {
        chunkhound: {
          command: "chunkhound",
          args: ["mcp"],
        },
      },
    },
  },
];

export const embeddingProviders: ConfiguratorEmbeddingProviderOption[] = [
  {
    id: "voyageai",
    name: "VoyageAI",
    svg: VOYAGEAI_SVG,
    description: "VoyageAI neural embeddings with built-in reranking",
    config: { provider: "voyageai", model: "voyage-4-lite" },
    apiKeyPlaceholder: "<YOUR_VOYAGE_API_KEY>",
    group: "cloud",
    requirements: [requirements.voyageApiKey, requirements.voyageTrainingOptOut],
    reranker: { included: true, status: "Included by VoyageAI." },
    // VoyageAI is recommended for the best out-of-box results, ROI, and setup
    // ease. Local providers remain an explicit choice when their data boundary
    // matters more; embeddings only set recall and its bundled reranker drives relevance.
    recommended: true,
    recommendation:
      "<strong>Voyage-4-Lite</strong> — a fast, low-cost embedder paired with VoyageAI's included reranker, which drives retrieval quality.",
  },
  {
    id: "openai-embed",
    name: "OpenAI",
    svg: OPENAI_SVG,
    description: "OpenAI text embedding model",
    config: {
      provider: "openai",
      model: "text-embedding-3-small",
    },
    apiKeyPlaceholder: "<YOUR_OPENAI_API_KEY>",
    group: "cloud",
    requirements: [requirements.openaiApiKey],
    reranker: { included: false },
  },
  {
    id: "ollama-embed",
    name: "Ollama",
    svg: OLLAMA_SVG,
    description: "Local embedding model via Ollama",
    config: {
      provider: "openai",
      model: "qwen3-embedding",
      base_url: "http://localhost:11434/v1",
    },
    setupHint: "ollama pull qwen3-embedding",
    group: "local",
    requirements: [requirements.ollama, requirements.ollamaEmbedModel],
    reranker: { included: false },
  },
  {
    id: "vllm-embed",
    name: "vLLM",
    svg: VLLM_SVG,
    description: "Local embedding model via vLLM",
    config: {
      provider: "openai",
      model: "Qwen/Qwen3-Embedding-0.6B",
      base_url: "http://localhost:8000/v1",
      rerank_model: "Qwen/Qwen3-Reranker-0.6B",
      // vLLM serves one model per process, so the reranker is a second service.
      // The Qwen reranker needs `--task score` + HF overrides to expose the
      // Cohere-compatible /rerank API (see docs/configuration.md "vLLM").
      rerank_url: "http://localhost:8001/v1/rerank",
      rerank_format: "cohere",
    },
    setupHint: `# Embeddings
vllm serve Qwen/Qwen3-Embedding-0.6B --port 8000
# Reranker
vllm serve Qwen/Qwen3-Reranker-0.6B --task score --hf-overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}' --port 8001`,
    group: "local",
    requirements: [requirements.vllm, requirements.vllmEmbedEndpoints],
    reranker: {
      included: true,
      status: "Included by this preset — runs as a separate local vLLM service on port 8001.",
    },
  },
];

// Model IDs verified against live catalogs (2026-10): OpenRouter's API
// (openrouter.ai/api/v1/models), Vercel AI Gateway (ai-gateway.vercel.sh/v1/models),
// models.dev (orcarouter, opencode), and provider docs (ai.google.dev,
// api-docs.deepseek.com, docs.x.ai). The backend does not validate LLM model IDs —
// a typo surfaces only as a provider API error.
export const llmProviders: ConfiguratorProviderOption[] = [
  {
    id: "vercel",
    name: "Vercel AI Gateway",
    svg: VERCEL_SVG,
    description: "Recommended model: Poolside Laguna S 2.1",
    config: {
      provider: "vercel",
      model: "poolside/laguna-s-2.1",
    },
    apiKeyPlaceholder: "<YOUR_VERCEL_API_KEY>",
    group: "cloud",
    requirements: [requirements.vercelApiKey],
    recommendation:
      "<strong>Laguna S 2.1</strong> — Poolside's open-weight model for agentic coding and long-horizon work, served through Vercel AI Gateway. It is the default because it serves the same model through a first-class provider — the best balance, not a compromise.",
  },
  {
    id: "openrouter",
    name: "OpenRouter",
    svg: OPENROUTER_SVG,
    description: "Recommended model: Poolside Laguna S 2.1",
    config: {
      provider: "openrouter",
      model: "poolside/laguna-s-2.1",
    },
    apiKeyPlaceholder: "<YOUR_OPENROUTER_API_KEY>",
    group: "cloud",
    requirements: [requirements.openrouterApiKey],
    recommendation:
      "<strong>Laguna S 2.1</strong> — Poolside's open-weight model for agentic coding and long-horizon work, routed through OpenRouter.",
  },
  {
    id: "orcarouter",
    name: "OrcaRouter",
    svg: ORCAROUTER_SVG,
    description: "Qwen3.7 Flash via OrcaRouter",
    config: { provider: "orcarouter", model: "qwen/qwen3.7-flash" },
    apiKeyPlaceholder: "<YOUR_ORCAROUTER_API_KEY>",
    group: "cloud",
    requirements: [requirements.orcarouterApiKey],
    recommendation:
      "<strong>Qwen3.7 Flash</strong> — a fast, cost-effective research model routed through OrcaRouter.",
  },
  {
    id: "anthropic",
    name: "Anthropic",
    svg: ANTHROPIC_SVG,
    description: "Anthropic Claude",
    config: { provider: "anthropic" },
    apiKeyPlaceholder: "<YOUR_ANTHROPIC_API_KEY>",
    group: "cloud",
    requirements: [requirements.anthropicApiKey],
    recommendation:
      "<strong>Claude Haiku 4.5</strong> — Anthropic's fastest and cheapest model.",
  },
  {
    id: "openai-llm",
    name: "OpenAI",
    svg: OPENAI_SVG,
    description: "OpenAI GPT",
    config: { provider: "openai" },
    apiKeyPlaceholder: "<YOUR_OPENAI_API_KEY>",
    group: "cloud",
    requirements: [requirements.openaiApiKey],
    recommendation:
      "<strong>GPT-5</strong> — OpenAI's default model for deep research.",
  },
  {
    id: "codex-cli",
    name: "Codex CLI",
    svg: OPENAI_SVG,
    description: "OpenAI Codex CLI",
    config: { provider: "codex-cli" },
    group: "agent",
    requirements: [requirements.codex],
  },
  {
    id: "claude-code-cli",
    name: "Claude Code CLI",
    svg: ANTHROPIC_SVG,
    description: "Claude Code CLI",
    config: { provider: "claude-code-cli" },
    group: "agent",
    requirements: [requirements.claudeCode],
  },
  {
    id: "gemini",
    name: "Gemini",
    svg: GEMINI_SVG,
    description: "Google Gemini",
    config: { provider: "gemini", model: "gemini-3.5-flash" },
    apiKeyPlaceholder: "<YOUR_GEMINI_API_KEY>",
    group: "cloud",
    requirements: [requirements.geminiApiKey],
    recommendation:
      "<strong>Gemini 3.5 Flash</strong> — Google's fast, low-cost research model.",
  },
  {
    id: "deepseek",
    name: "DeepSeek",
    svg: DEEPSEEK_SVG,
    description: "DeepSeek",
    config: { provider: "deepseek", model: "deepseek-v4-flash" },
    apiKeyPlaceholder: "<YOUR_DEEPSEEK_API_KEY>",
    group: "cloud",
    requirements: [requirements.deepseekApiKey],
    recommendation:
      "<strong>DeepSeek V4 Flash</strong> — DeepSeek's fastest and cheapest model.",
  },
  {
    id: "grok",
    name: "Grok",
    svg: GROK_SVG,
    description: "xAI Grok",
    config: { provider: "grok", model: "grok-4.3" },
    apiKeyPlaceholder: "<YOUR_XAI_API_KEY>",
    group: "cloud",
    requirements: [requirements.grokApiKey],
    recommendation:
      "<strong>Grok 4.3</strong> — xAI's fastest model for research.",
  },
  {
    id: "ollama-llm",
    name: "Ollama",
    svg: OLLAMA_SVG,
    description: "Local Qwen3 Coder via Ollama",
    config: {
      provider: "openai",
      model: "qwen3-coder:30b",
      base_url: "http://localhost:11434/v1",
    },
    setupHint: "ollama pull qwen3-coder:30b",
    group: "local",
    requirements: [requirements.ollama, requirements.ollamaLlmModel],
    recommendation:
      "<strong>Qwen3 Coder 30B</strong> — the recommended local model for research.",
  },
  {
    id: "vllm-llm",
    name: "vLLM",
    svg: VLLM_SVG,
    description: "Local Qwen3 Coder via vLLM",
    config: {
      provider: "openai",
      model: "Qwen/Qwen3-Coder-30B-A3B-Instruct",
      // Separate port so the LLM cannot collide with the embedding/rerank services.
      base_url: "http://localhost:8002/v1",
    },
    setupHint:
      "# Research\nvllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct --port 8002",
    group: "local",
    requirements: [requirements.vllm, requirements.vllmLlmModel],
    recommendation:
      "<strong>Qwen3 Coder 30B</strong> — the recommended local model for research.",
  },
  {
    id: "opencode-cli",
    name: "OpenCode CLI",
    svg: OPENCODE_SVG,
    description: "OpenCode CLI",
    config: { provider: "opencode-cli", model: "opencode/deepseek-v4.1-flash" },
    group: "agent",
    requirements: [requirements.opencode],
    recommendation:
      "<strong>DeepSeek V4.1 Flash</strong> — OpenCode's fastest and cheapest model.",
  },
];

function findById<T extends { id: string }>(
  options: readonly T[],
  id: string,
  kind: string,
): T {
  const option = options.find((item) => item.id === id);
  if (!option) throw new Error(`Unknown ${kind}: ${id}`);
  return option;
}

export function findEditor(id: string): ConfiguratorEditor {
  return findById(editors, id, "editor");
}

export function findEmbeddingProvider(
  id: string,
): ConfiguratorEmbeddingProviderOption {
  return findById(embeddingProviders, id, "embedding provider");
}

export function findLlmProvider(id: string): ConfiguratorProviderOption {
  return findById(llmProviders, id, "LLM provider");
}

export function groupProviders(
  providers: ConfiguratorProviderOption[],
): Array<{ label: string; options: ConfiguratorProviderOption[] }> {
  const groupMap = new Map<
    ConfiguratorProviderGroup,
    ConfiguratorProviderOption[]
  >();
  for (const provider of providers) {
    // Fail fast: a group missing from PROVIDER_GROUP_ORDER would otherwise be
    // silently dropped from the rendered picker.
    if (!PROVIDER_GROUP_ORDER.includes(provider.group)) {
      throw new Error(
        `Provider '${provider.id}' has group '${provider.group}' not present in PROVIDER_GROUP_ORDER`,
      );
    }
    const options = groupMap.get(provider.group);
    if (options) options.push(provider);
    else groupMap.set(provider.group, [provider]);
  }
  return PROVIDER_GROUP_ORDER.flatMap((group) => {
    const options = groupMap.get(group);
    return options ? [{ label: group, options }] : [];
  });
}
