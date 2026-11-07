import type { ConfiguratorRequirement } from "./types.ts";
import { backendAuthUrl } from "./catalog.ts";
import {
  ANTHROPIC_SVG,
  API_ROUTE_SVG,
  ATLASCLOUD_SVG,
  CHEAPAIMODEL_SVG,
  CURSOR_SVG,
  DEEPSEEK_SVG,
  GEMINI_SVG,
  GROK_SVG,
  OBSCURIFY_SVG,
  OLLAMA_SVG,
  OPENAI_SVG,
  OPENCODE_SVG,
  OPENROUTER_SVG,
  OPPER_SVG,
  ORCAROUTER_SVG,
  PI_SVG,
  REQUESTY_SVG,
  ROO_CODE_SVG,
  RUNCRATE_SVG,
  VERCEL_SVG,
  VLLM_SVG,
  VOYAGEAI_SVG,
  VSCODE_SVG,
  WINDSURF_SVG,
  ZED_SVG,
} from "./icons.ts";

/**
 * API-key prerequisite whose URL derives from the registry's `auth_url`, so a
 * new provider never means retyping its key portal. `url` is only for providers
 * the registry does not name (native SDK providers); overriding a registry URL
 * is rejected so the backend stays the single owner.
 */
function apiKeyRequirement(input: {
  id: string;
  label: string;
  svg: string;
  provider: string;
  url?: string;
}): ConfiguratorRequirement {
  const registryUrl = backendAuthUrl(input.provider);
  if (input.url && registryUrl) {
    throw new Error(
      `Requirement '${input.id}' overrides '${input.provider}' auth_url; ` +
        "update the backend registry instead.",
    );
  }
  const url = input.url ?? registryUrl;
  if (!url) {
    throw new Error(
      `Requirement '${input.id}' has no url and '${input.provider}' has no registry auth_url.`,
    );
  }
  return { id: input.id, label: input.label, url, svg: input.svg };
}

export const requirements = {
  pi: {
    id: "pi",
    label: "Pi 1.0+",
    url: "https://pi.dev",
    svg: PI_SVG,
  },
  cursor: {
    id: "cursor",
    label: "Cursor with MCP support",
    url: "https://cursor.com/docs/mcp",
    svg: CURSOR_SVG,
  },
  claudeCode: {
    id: "claude-code",
    label: "Claude Code CLI (signed in)",
    url: "https://code.claude.com/docs/en/mcp",
    svg: ANTHROPIC_SVG,
  },
  vscode: {
    id: "vscode",
    label: "VS Code with MCP support",
    url: "https://code.visualstudio.com/docs/agent-customization/mcp-servers",
    svg: VSCODE_SVG,
  },
  opencode: {
    id: "opencode",
    label: "OpenCode CLI (signed in)",
    url: "https://opencode.ai/docs/cli/",
    svg: OPENCODE_SVG,
  },
  codex: {
    id: "codex",
    label: "Codex CLI (signed in)",
    url: "https://learn.chatgpt.com/docs/codex/cli",
    svg: OPENAI_SVG,
  },
  windsurf: {
    id: "windsurf",
    label: "Windsurf with MCP support",
    url: "https://docs.devin.ai/desktop/cascade/mcp",
    svg: WINDSURF_SVG,
  },
  rooCode: {
    id: "roo-code",
    label: "Roo Code extension",
    url: "https://roocodeinc.github.io/Roo-Code/features/mcp/using-mcp-in-roo/",
    svg: ROO_CODE_SVG,
  },
  zed: {
    id: "zed",
    label: "Zed with MCP support",
    url: "https://zed.dev/docs/ai/mcp",
    svg: ZED_SVG,
  },
  voyageApiKey: {
    id: "voyage-api-key",
    label: "VoyageAI API key",
    url: "https://dashboard.voyageai.com/api-keys",
    svg: VOYAGEAI_SVG,
  },
  voyageTrainingOptOut: {
    id: "voyage-training-opt-out",
    label: "Opt out of VoyageAI training for true privacy",
    url: "https://dashboard.voyageai.com/organization/tos",
    svg: VOYAGEAI_SVG,
    optional: true,
  },
  openaiApiKey: apiKeyRequirement({
    id: "openai-api-key",
    label: "OpenAI API key",
    provider: "openai",
    url: "https://platform.openai.com/api-keys",
    svg: OPENAI_SVG,
  }),
  ollama: {
    id: "ollama",
    label: "Ollama",
    url: "https://ollama.com/download",
    svg: OLLAMA_SVG,
  },
  ollamaEmbedModel: {
    id: "ollama-embed-model",
    label: "qwen3-embedding model",
    url: "https://ollama.com/library/qwen3-embedding",
    svg: OLLAMA_SVG,
  },
  vllm: {
    id: "vllm",
    label: "vLLM",
    url: "https://docs.vllm.ai/en/latest/",
    svg: VLLM_SVG,
  },
  vllmEmbedEndpoints: {
    id: "vllm-embed-endpoints",
    label: "Qwen3 embedding + reranker endpoints",
    url: "https://docs.vllm.ai/en/latest/",
    svg: VLLM_SVG,
  },
  anthropicApiKey: apiKeyRequirement({
    id: "anthropic-api-key",
    label: "Anthropic API key",
    provider: "anthropic",
    url: "https://console.anthropic.com/settings/keys",
    svg: ANTHROPIC_SVG,
  }),
  geminiApiKey: apiKeyRequirement({
    id: "gemini-api-key",
    label: "Gemini API key",
    provider: "gemini",
    url: "https://aistudio.google.com/api-keys",
    svg: GEMINI_SVG,
  }),
  deepseekApiKey: apiKeyRequirement({
    id: "deepseek-api-key",
    label: "DeepSeek API key",
    provider: "deepseek",
    svg: DEEPSEEK_SVG,
  }),
  grokApiKey: apiKeyRequirement({
    id: "grok-api-key",
    label: "xAI API key",
    provider: "grok",
    svg: GROK_SVG,
  }),
  ollamaLlmModel: {
    id: "ollama-llm-model",
    label: "qwen3-coder:30b model",
    url: "https://ollama.com/library/qwen3-coder",
    svg: OLLAMA_SVG,
  },
  vllmLlmModel: {
    id: "vllm-llm-model",
    label: "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    url: "https://docs.vllm.ai/en/latest/",
    svg: VLLM_SVG,
  },
  openrouterApiKey: apiKeyRequirement({
    id: "openrouter-api-key",
    label: "OpenRouter API key",
    provider: "openrouter",
    svg: OPENROUTER_SVG,
  }),
  vercelApiKey: apiKeyRequirement({
    id: "vercel-api-key",
    label: "Vercel AI Gateway API key",
    provider: "vercel",
    svg: VERCEL_SVG,
  }),
  orcarouterApiKey: apiKeyRequirement({
    id: "orcarouter-api-key",
    label: "OrcaRouter API key",
    provider: "orcarouter",
    svg: ORCAROUTER_SVG,
  }),
  requestyApiKey: apiKeyRequirement({
    id: "requesty-api-key",
    label: "Requesty API key",
    provider: "requesty",
    svg: REQUESTY_SVG,
  }),
  apiRouteApiKey: apiKeyRequirement({
    id: "api-route-api-key",
    label: "API Route API key",
    provider: "api-route",
    svg: API_ROUTE_SVG,
  }),
  opperApiKey: apiKeyRequirement({
    id: "opper-api-key",
    label: "Opper API key",
    provider: "opper",
    svg: OPPER_SVG,
  }),
  obscurifyApiKey: apiKeyRequirement({
    id: "obscurify-api-key",
    label: "Obscurify.ai API key",
    provider: "obscurify-ai",
    svg: OBSCURIFY_SVG,
  }),
  atlascloudApiKey: apiKeyRequirement({
    id: "atlascloud-api-key",
    label: "Atlas Cloud API key",
    provider: "atlascloud",
    svg: ATLASCLOUD_SVG,
  }),
  runcrateApiKey: apiKeyRequirement({
    id: "runcrate-api-key",
    label: "Runcrate API key",
    provider: "runcrate",
    svg: RUNCRATE_SVG,
  }),
  cheapaimodelApiKey: apiKeyRequirement({
    id: "cheapaimodel-api-key",
    label: "cheapaimodel API key",
    provider: "cheapaimodel",
    svg: CHEAPAIMODEL_SVG,
  }),
} satisfies Record<string, ConfiguratorRequirement>;
