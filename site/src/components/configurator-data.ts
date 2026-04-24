export type ConfigRecord = Record<string, unknown>;

export interface ConfiguratorEditor {
    id: string;
    name: string;
    svg: string;
    mcpFile?: string;
    mcpFilePowerShell?: string;
    mcp?: ConfigRecord;
    rawCmd?: string;
}

export interface ConfiguratorProviderOption {
    id: string;
    name: string;
    svg: string;
    config: ConfigRecord;
    apiKeyPlaceholder?: string;
    setupHint?: string;
}

export type ConfiguratorPlatform = "posix" | "powershell";

export interface PlatformOption {
    id: ConfiguratorPlatform;
    label: string;
}

export function configWithApiKey(
    option: ConfiguratorProviderOption,
): ConfigRecord {
    return option.apiKeyPlaceholder
        ? { ...option.config, api_key: option.apiKeyPlaceholder }
        : option.config;
}

export const OPENCODE_SVG = `<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" fill="currentColor"><path d="M15 10.5v6H9v-6h6Z" opacity=".5"/><path fill-rule="evenodd" d="M18 19.5H6V4.5h12v15ZM15 7.5H9v9h6v-9Z"/></svg>`;
export const OPENAI_SVG = `<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" fill="currentColor"><path d="M22.2819 9.8211a5.9847 5.9847 0 0 0-.5157-4.9108 6.0462 6.0462 0 0 0-6.5098-2.9A6.0651 6.0651 0 0 0 4.9807 4.1818a5.9847 5.9847 0 0 0-3.9977 2.9 6.0462 6.0462 0 0 0 .7427 7.0966 5.98 5.98 0 0 0 .511 4.9107 6.051 6.051 0 0 0 6.5146 2.9001A5.9847 5.9847 0 0 0 13.2599 24a6.0557 6.0557 0 0 0 5.7718-4.2058 5.9894 5.9894 0 0 0 3.9977-2.9001 6.0557 6.0557 0 0 0-.7475-7.0729zm-9.022 12.6081a4.4755 4.4755 0 0 1-2.8764-1.0408l.1419-.0804 4.7783-2.7582a.7948.7948 0 0 0 .3927-.6813v-6.7369l2.02 1.1686a.071.071 0 0 1 .038.052v5.5826a4.504 4.504 0 0 1-4.4945 4.4944zm-9.6607-4.1254a4.4708 4.4708 0 0 1-.5346-3.0137l.142.0852 4.783 2.7582a.7712.7712 0 0 0 .7806 0l5.8428-3.3685v2.3324a.0804.0804 0 0 1-.0332.0615L9.74 19.9502a4.4992 4.4992 0 0 1-6.1408-1.6464zM2.3408 7.8956a4.485 4.485 0 0 1 2.3655-1.9728V11.6a.7664.7664 0 0 0 .3879.6765l5.8144 3.3543-2.0201 1.1685a.0757.0757 0 0 1-.071 0l-4.8303-2.7865A4.504 4.504 0 0 1 2.3408 7.872zm16.5963 3.8558L13.1038 8.364 15.1192 7.2a.0757.0757 0 0 1 .071 0l4.8303 2.7913a4.4944 4.4944 0 0 1-.6765 8.1042v-5.6772a.79.79 0 0 0-.407-.667zm2.0107-3.0231l-.142-.0852-4.7735-2.7818a.7759.7759 0 0 0-.7854 0L9.409 9.2297V6.8974a.0662.0662 0 0 1 .0284-.0615l4.8303-2.7866a4.4992 4.4992 0 0 1 6.6802 4.66zM8.3065 12.863l-2.02-1.1638a.0804.0804 0 0 1-.038-.0567V6.0742a4.4992 4.4992 0 0 1 7.3757-3.4537l-.142.0805L8.704 5.459a.7948.7948 0 0 0-.3927.6813zm1.0976-2.3654l2.602-1.4998 2.6069 1.4998v2.9994l-2.5974 1.4997-2.6067-1.4997Z"/></svg>`;
export const OLLAMA_SVG = `<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" fill="currentColor"><path d="M16.361 10.26a.894.894 0 0 0-.558.47l-.072.148.001.207c0 .193.004.217.059.353.076.193.152.312.291.448.24.238.51.3.872.205a.86.86 0 0 0 .517-.436.752.752 0 0 0 .08-.498c-.064-.453-.33-.782-.724-.897a1.06 1.06 0 0 0-.466 0zm-9.203.005c-.305.096-.533.32-.65.639a1.187 1.187 0 0 0-.06.52c.057.309.31.59.598.667.362.095.632.033.872-.205.14-.136.215-.255.291-.448.055-.136.059-.16.059-.353l.001-.207-.072-.148a.894.894 0 0 0-.565-.472 1.02 1.02 0 0 0-.474.007Zm4.184 2c-.131.071-.223.25-.195.383.031.143.157.288.353.407.105.063.112.072.117.136.004.038-.01.146-.029.243-.02.094-.036.194-.036.222.002.074.07.195.143.253.064.052.076.054.255.059.164.005.198.001.264-.03.169-.082.212-.234.15-.525-.052-.243-.042-.28.087-.355.137-.08.281-.219.324-.314a.365.365 0 0 0-.175-.48.394.394 0 0 0-.181-.033c-.126 0-.207.03-.355.124l-.085.053-.053-.032c-.219-.13-.259-.145-.391-.143a.396.396 0 0 0-.193.032zm.39-2.195c-.373.036-.475.05-.654.086-.291.06-.68.195-.951.328-.94.46-1.589 1.226-1.787 2.114-.04.176-.045.234-.045.53 0 .294.005.357.043.524.264 1.16 1.332 2.017 2.714 2.173.3.033 1.596.033 1.896 0 1.11-.125 2.064-.727 2.493-1.571.114-.226.169-.372.22-.602.039-.167.044-.23.044-.523 0-.297-.005-.355-.045-.531-.288-1.29-1.539-2.304-3.072-2.497a6.873 6.873 0 0 0-.855-.031zm.645.937a3.283 3.283 0 0 1 1.44.514c.223.148.537.458.671.662.166.251.26.508.303.82.02.143.01.251-.043.482-.08.345-.332.705-.672.957a3.115 3.115 0 0 1-.689.348c-.382.122-.632.144-1.525.138-.582-.006-.686-.01-.853-.042-.57-.107-1.022-.334-1.35-.68-.264-.28-.385-.535-.45-.946-.03-.192.025-.509.137-.776.136-.326.488-.73.836-.963.403-.269.934-.46 1.422-.512.187-.02.586-.02.773-.002zm-5.503-11a1.653 1.653 0 0 0-.683.298C5.617.74 5.173 1.666 4.985 2.819c-.07.436-.119 1.04-.119 1.503 0 .544.064 1.24.155 1.721.02.107.031.202.023.208a8.12 8.12 0 0 1-.187.152 5.324 5.324 0 0 0-.949 1.02 5.49 5.49 0 0 0-.94 2.339 6.625 6.625 0 0 0-.023 1.357c.091.78.325 1.438.727 2.04l.13.195-.037.064c-.269.452-.498 1.105-.605 1.732-.084.496-.095.629-.095 1.294 0 .67.009.803.088 1.266.095.555.288 1.143.503 1.534.071.128.243.393.264.407.007.003-.014.067-.046.141a7.405 7.405 0 0 0-.548 1.873c-.062.417-.071.552-.071.991 0 .56.031.832.148 1.279L3.42 24h1.478l-.05-.091c-.297-.552-.325-1.575-.068-2.597.117-.472.25-.819.498-1.296l.148-.29v-.177c0-.165-.003-.184-.057-.293a.915.915 0 0 0-.194-.25 1.74 1.74 0 0 1-.385-.543c-.424-.92-.506-2.286-.208-3.451.124-.486.329-.918.544-1.154a.787.787 0 0 0 .223-.531c0-.195-.07-.355-.224-.522a3.136 3.136 0 0 1-.817-1.729c-.14-.96.114-2.005.69-2.834.563-.814 1.353-1.336 2.237-1.475.199-.033.57-.028.776.01.226.04.367.028.512-.041.179-.085.268-.19.374-.431.093-.215.165-.333.36-.576.234-.29.46-.489.822-.729.413-.27.884-.467 1.352-.561.17-.035.25-.04.569-.04.319 0 .398.005.569.04a4.07 4.07 0 0 1 1.914.997c.117.109.398.457.488.602.034.057.095.177.132.267.105.241.195.346.374.43.14.068.286.082.503.045.343-.058.607-.053.943.016 1.144.23 2.14 1.173 2.581 2.437.385 1.108.276 2.267-.296 3.153-.097.15-.193.27-.333.419-.301.322-.301.722-.001 1.053.493.539.801 1.866.708 3.036-.062.772-.26 1.463-.533 1.854a2.096 2.096 0 0 1-.224.258.916.916 0 0 0-.194.25c-.054.109-.057.128-.057.293v.178l.148.29c.248.476.38.823.498 1.295.253 1.008.231 2.01-.059 2.581a.845.845 0 0 0-.044.098c0 .006.329.009.732.009h.73l.02-.074.036-.134c.019-.076.057-.3.088-.516.029-.217.029-1.016 0-1.258-.11-.875-.295-1.57-.597-2.226-.032-.074-.053-.138-.046-.141.008-.005.057-.074.108-.152.376-.569.607-1.284.724-2.228.031-.26.031-1.378 0-1.628-.083-.645-.182-1.082-.348-1.525a6.083 6.083 0 0 0-.329-.7l-.038-.064.131-.194c.402-.604.636-1.262.727-2.04a6.625 6.625 0 0 0-.024-1.358 5.512 5.512 0 0 0-.939-2.339 5.325 5.325 0 0 0-.95-1.02 8.097 8.097 0 0 1-.186-.152.692.692 0 0 1 .023-.208c.208-1.087.201-2.443-.017-3.503-.19-.924-.535-1.658-.98-2.082-.354-.338-.716-.482-1.15-.455-.996.059-1.8 1.205-2.116 3.01a6.805 6.805 0 0 0-.097.726c0 .036-.007.066-.015.066a.96.96 0 0 1-.149-.078A4.857 4.857 0 0 0 12 3.03c-.832 0-1.687.243-2.456.698a.958.958 0 0 1-.148.078c-.008 0-.015-.03-.015-.066a6.71 6.71 0 0 0-.097-.725C8.997 1.392 8.337.319 7.46.048a2.096 2.096 0 0 0-.585-.041Zm.293 1.402c.248.197.523.759.682 1.388.03.113.06.244.069.292.007.047.026.152.041.233.067.365.098.76.102 1.24l.002.475-.12.175-.118.178h-.278c-.324 0-.646.041-.954.124l-.238.06c-.033.007-.038-.003-.057-.144a8.438 8.438 0 0 1 .016-2.323c.124-.788.413-1.501.696-1.711.067-.05.079-.049.157.013zm9.825-.012c.17.126.358.46.498.888.28.854.36 2.028.212 3.145-.019.14-.024.151-.057.144l-.238-.06a3.693 3.693 0 0 0-.954-.124h-.278l-.119-.178-.119-.175.002-.474c.004-.669.066-1.19.214-1.772.157-.623.434-1.185.68-1.382.078-.062.09-.063.159-.012z"/></svg>`;
export const ANTHROPIC_SVG = `<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" fill="currentColor"><path d="M17.3041 3.541h-3.6718l6.696 16.918H24Zm-10.6082 0L0 20.459h3.7442l1.3693-3.5527h7.0052l1.3693 3.5528h3.7442L10.5363 3.5409Zm-.3712 10.2232 2.2914-5.9456 2.2914 5.9456Z"/></svg>`;
export const VOYAGEAI_SVG = `<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" fill="currentColor"><path fill-rule="evenodd" d="M13.93 3.058C12.981 1.931 12.165.786 11.996.548c-.018-.018-.044-.018-.062 0-.166.238-.985 1.383-1.934 2.51C1.851 13.461 11.285 20.484 11.285 20.484l.079.054c.072 1.083.245 2.643.245 2.643h.704s.177-1.55.246-2.643l.079-.062s9.441-7.012 1.292-17.415v-.003zM11.963 20.325s-.423-.361-.538-.545v-.018l.509-11.328c0-.036.054-.036.054 0l.509 11.328v.018c-.115.184-.537.545-.537.545h.003z"/></svg>`;
export const VLLM_SVG = `<svg viewBox="28 66 74 74" xmlns="http://www.w3.org/2000/svg" fill="currentColor"><path d="M58.465 81.843v55.307L30.811 81.843z"/><path d="M58.464 137.15l21.73 0 18.653-70.386-25.575 13.462z"/></svg>`;

export const editors: ConfiguratorEditor[] = [
    {
        id: "cursor",
        name: "Cursor",
        svg: `<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" fill="currentColor"><path d="M11.503.131 1.891 5.678a.84.84 0 0 0-.42.726v11.188c0 .3.162.575.42.724l9.609 5.55a1 1 0 0 0 .998 0l9.61-5.55a.84.84 0 0 0 .42-.724V6.404a.84.84 0 0 0-.42-.726L12.497.131a1.01 1.01 0 0 0-.996 0M2.657 6.338h18.55c.263 0 .43.287.297.515L12.23 22.918c-.062.107-.229.064-.229-.06V12.335a.59.59 0 0 0-.295-.51l-9.11-5.257c-.109-.063-.064-.23.061-.23"/></svg>`,
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
        svg: ANTHROPIC_SVG,
        rawCmd: "claude mcp add ChunkHound -- chunkhound mcp",
    },
    {
        id: "vscode",
        name: "VS Code",
        svg: `<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" fill="currentColor"><path d="M23.15 2.587L18.21.21a1.494 1.494 0 0 0-1.705.29l-9.46 8.63-4.12-3.128a.999.999 0 0 0-1.276.057L.327 7.261A1 1 0 0 0 .326 8.74L3.899 12 .326 15.26a1 1 0 0 0 .001 1.479L1.65 17.94a.999.999 0 0 0 1.276.057l4.12-3.128 9.46 8.63a1.492 1.492 0 0 0 1.704.29l4.942-2.377A1.5 1.5 0 0 0 24 20.06V3.939a1.5 1.5 0 0 0-.85-1.352zm-5.146 14.861L10.826 12l7.178-5.448v10.896z"/></svg>`,
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
        svg: OPENCODE_SVG,
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
        svg: OPENAI_SVG,
        rawCmd: "codex mcp add ChunkHound -- chunkhound mcp",
    },
    {
        id: "windsurf",
        name: "Windsurf",
        svg: `<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" fill="currentColor"><path d="M23.55 5.067c-1.2038-.002-2.1806.973-2.1806 2.1765v4.8676c0 .972-.8035 1.7594-1.7597 1.7594-.568 0-1.1352-.286-1.4718-.7659l-4.9713-7.1003c-.4125-.5896-1.0837-.941-1.8103-.941-1.1334 0-2.1533.9635-2.1533 2.153v4.8957c0 .972-.7969 1.7594-1.7596 1.7594-.57 0-1.1363-.286-1.4728-.7658L.4076 5.1598C.2822 4.9798 0 5.0688 0 5.2882v4.2452c0 .2147.0656.4228.1884.599l5.4748 7.8183c.3234.462.8006.8052 1.3509.9298 1.3771.313 2.6446-.747 2.6446-2.0977v-4.893c0-.972.7875-1.7593 1.7596-1.7593h.003a1.798 1.798 0 0 1 1.4718.7658l4.9723 7.0994c.4135.5905 1.05.941 1.8093.941 1.1587 0 2.1515-.9645 2.1515-2.153v-4.8948c0-.972.7875-1.7594 1.7596-1.7594h.194a.22.22 0 0 0 .2204-.2202v-4.622a.22.22 0 0 0-.2203-.2203Z"/></svg>`,
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
        svg: `<svg viewBox="0 0 34 21" xmlns="http://www.w3.org/2000/svg" fill="currentColor"><path d="M27.6939 0.698856L27.0853 2.88004C27.0531 2.99552 26.9315 3.06224 26.8157 3.02799L16.6072 0.00893058C16.5393 -0.0111487 16.4658 0.00308892 16.4106 0.047029L6.30049 8.08333C6.27097 8.10679 6.23575 8.12215 6.19836 8.12786L0.18292 9.04674C0.0715357 9.06375 -0.0078759 9.16286 0.000622655 9.27425L0.0267609 9.61687C0.035109 9.72631 0.125665 9.81172 0.236347 9.81456L7.22353 9.99359L7.30322 9.9958L12.4661 7.2622C12.5383 7.22393 12.6263 7.22955 12.693 7.27669L16.3507 9.86111C16.408 9.90164 16.4418 9.96737 16.4412 10.0372L16.4103 13.4664C16.4099 13.5108 16.4235 13.5542 16.4491 13.5906L21.5943 20.9084C21.6346 20.9658 21.7007 21 21.7713 21H23.4001C23.5626 21 23.6668 20.8287 23.5909 20.6863L19.9237 13.8002C19.8878 13.7328 19.8908 13.6516 19.9315 13.5871L21.8435 10.551C21.8643 10.5179 21.8939 10.4912 21.9289 10.4735L28.7648 7.03514C28.8342 7.00021 28.9173 7.00508 28.9821 7.0479L30.9355 8.33851C30.971 8.36193 31.0126 8.37442 31.0552 8.37442H32.83C33.0019 8.37442 33.1047 8.18497 33.0101 8.04276L28.082 0.63809C27.98 0.484786 27.7433 0.521844 27.6939 0.698856Z"/></svg>`,
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
        svg: `<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" fill="currentColor"><path d="M2.25 1.5a.75.75 0 0 0-.75.75v16.5H0V2.25A2.25 2.25 0 0 1 2.25 0h20.095c1.002 0 1.504 1.212.795 1.92L10.764 14.298h3.486V12.75h1.5v1.922a1.125 1.125 0 0 1-1.125 1.125H9.264l-2.578 2.578h11.689V9h1.5v9.375a1.5 1.5 0 0 1-1.5 1.5H5.185L2.562 22.5H21.75a.75.75 0 0 0 .75-.75V5.25H24v16.5A2.25 2.25 0 0 1 21.75 24H1.655C.653 24 .151 22.788.86 22.08L13.19 9.75H9.75v1.5h-1.5V9.375A1.125 1.125 0 0 1 9.375 8.25h5.314l2.625-2.625H5.625V15h-1.5V5.625a1.5 1.5 0 0 1 1.5-1.5h13.19L21.438 1.5z"/></svg>`,
        mcpFile: "settings.json",
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

export const embeddingProviders: ConfiguratorProviderOption[] = [
    {
        id: "voyageai",
        name: "VoyageAI",
        svg: VOYAGEAI_SVG,
        config: { provider: "voyageai", model: "voyage-3.5" },
        apiKeyPlaceholder: "<YOUR_VOYAGE_API_KEY>",
    },
    {
        id: "openai-embed",
        name: "OpenAI",
        svg: OPENAI_SVG,
        config: {
            provider: "openai",
            model: "text-embedding-3-small",
        },
        apiKeyPlaceholder: "<YOUR_OPENAI_API_KEY>",
    },
    {
        id: "ollama-embed",
        name: "Ollama",
        svg: OLLAMA_SVG,
        config: {
            provider: "openai",
            model: "qwen3-embedding",
            base_url: "http://localhost:11434/v1",
            rerank_model: "qwen3-reranker",
            rerank_format: "cohere",
        },
        setupHint: "ollama pull qwen3-embedding && ollama pull qwen3-reranker",
    },
    {
        id: "vllm-embed",
        name: "vLLM",
        svg: VLLM_SVG,
        config: {
            provider: "openai",
            model: "Qwen/Qwen3-Embedding-0.6B",
            base_url: "http://localhost:8000/v1",
            rerank_model: "Qwen/Qwen3-Reranker-0.6B",
            rerank_format: "cohere",
        },
        setupHint: "# Serve embeddings and reranking on the same vLLM OpenAI-compatible endpoint\n# Then point embedding.base_url at that endpoint.",
    },
];

export const llmProviders: ConfiguratorProviderOption[] = [
    {
        id: "anthropic",
        name: "Anthropic",
        svg: ANTHROPIC_SVG,
        config: { provider: "anthropic" },
        apiKeyPlaceholder: "<YOUR_ANTHROPIC_API_KEY>",
    },
    {
        id: "openai-llm",
        name: "OpenAI",
        svg: OPENAI_SVG,
        config: { provider: "openai" },
        apiKeyPlaceholder: "<YOUR_OPENAI_API_KEY>",
    },
    {
        id: "codex-cli",
        name: "Codex CLI",
        svg: OPENAI_SVG,
        config: { provider: "codex-cli" },
    },
    {
        id: "claude-code-cli",
        name: "Claude Code CLI",
        svg: ANTHROPIC_SVG,
        config: { provider: "claude-code-cli" },
    },
    {
        id: "gemini",
        name: "Gemini",
        svg: `<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" fill="currentColor"><path d="M11.04 19.32Q12 21.51 12 24q0-2.49.93-4.68.96-2.19 2.58-3.81t3.81-2.55Q21.51 12 24 12q-2.49 0-4.68-.93a12.3 12.3 0 0 1-3.81-2.58 12.3 12.3 0 0 1-2.58-3.81Q12 2.49 12 0q0 2.49-.96 4.68-.93 2.19-2.55 3.81a12.3 12.3 0 0 1-3.81 2.58Q2.49 12 0 12q2.49 0 4.68.96 2.19.93 3.81 2.55t2.55 3.81"/></svg>`,
        config: { provider: "gemini" },
        apiKeyPlaceholder: "<YOUR_GEMINI_API_KEY>",
    },
    {
        id: "grok",
        name: "Grok",
        svg: `<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" fill="currentColor"><path d="M2.3 0L10.684 10.984 2.048 21.336h1.944l7.478-8.962 6.06 8.962H22.3L13.636 9.96 21.74 0H19.8l-6.95 8.332L7.07 0zm3.108 1.464h2.7l12.492 18.408h-2.7z"/></svg>`,
        config: { provider: "grok" },
        apiKeyPlaceholder: "<YOUR_XAI_API_KEY>",
    },
    {
        id: "ollama-llm",
        name: "Ollama",
        svg: OLLAMA_SVG,
        config: {
            provider: "openai",
            model: "qwen3-coder:30b",
            base_url: "http://localhost:11434/v1",
        },
        setupHint: "ollama pull qwen3-coder:30b",
    },
    {
        id: "vllm-llm",
        name: "vLLM",
        svg: VLLM_SVG,
        config: {
            provider: "openai",
            model: "Qwen/Qwen3-Coder-30B-A3B-Instruct",
            base_url: "http://localhost:8000/v1",
        },
        setupHint: "vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct --port 8000",
    },
    {
        id: "opencode-cli",
        name: "OpenCode CLI",
        svg: OPENCODE_SVG,
        config: { provider: "opencode-cli" },
    },
];

export const DEFAULT_EDITOR = "cursor";
export const DEFAULT_EMBEDDING = "voyageai";
export const DEFAULT_LLM = "anthropic";
export const DEFAULT_PLATFORM: ConfiguratorPlatform = "posix";
export const PLATFORM_STORAGE_KEY = "chunkhound:platform";
export const platformOptions: PlatformOption[] = [
    { id: "posix", label: "macOS/Linux" },
    { id: "powershell", label: "PowerShell" },
];
export const INDEX_CMD = "chunkhound index .";

function getParentDir(filePath: string): string | null {
    const lastSlash = filePath.lastIndexOf("/");
    if (lastSlash <= 0) {
        return null;
    }
    return filePath.slice(0, lastSlash);
}

function withParentDirSetup(
    filePath: string,
    command: string,
    platform: ConfiguratorPlatform,
): string {
    const parentDir = getParentDir(filePath);
    if (!parentDir) {
        return command;
    }
    if (platform === "powershell") {
        return (
            `New-Item -ItemType Directory -Force -Path ${quotePowerShell(parentDir)} | Out-Null\n` +
            command
        );
    }
    return `mkdir -p ${parentDir}\n${command}`;
}

function quotePowerShell(value: string): string {
    if (value.startsWith("$HOME/")) {
        const escaped = value.replace(/`/g, "``").replace(/"/g, '`"');
        return `"${escaped}"`;
    }
    return `'${value.replace(/'/g, "''")}'`;
}

function getEditorFilePath(
    editor: ConfiguratorEditor,
    platform: ConfiguratorPlatform,
): string {
    if (platform === "powershell" && editor.mcpFilePowerShell) {
        return editor.mcpFilePowerShell;
    }
    if (!editor.mcpFile) throw new Error(`Editor '${editor.id}' is missing mcpFile`);
    return editor.mcpFile;
}

function buildGitignoreCommand(platform: ConfiguratorPlatform): string {
    if (platform === "powershell") {
        return [
            "if (-not (Test-Path .gitignore)) { New-Item -ItemType File -Path .gitignore | Out-Null }",
            "Add-Content -Path .gitignore -Value '.chunkhound.json'",
        ].join("\n");
    }
    return "echo .chunkhound.json >> .gitignore";
}

function buildJsonWriteCommand(
    filename: string,
    content: string,
    platform: ConfiguratorPlatform,
): string {
    if (platform === "powershell") {
        const writeCommand = [
            "@'",
            content,
            "'@ | Set-Content -Path " + quotePowerShell(filename) + " -Encoding utf8",
        ].join("\n");
        return withParentDirSetup(filename, writeCommand, platform);
    }

    const writeCommand = `cat > ${filename} <<'CHUNKHOUND_EOF'\n${content}\nCHUNKHOUND_EOF`;
    return withParentDirSetup(filename, writeCommand, platform);
}

export function echoCommand(
    filename: string,
    content: ConfigRecord,
    platform: ConfiguratorPlatform = DEFAULT_PLATFORM,
): string {
    return buildJsonWriteCommand(filename, JSON.stringify(content, null, 2), platform);
}

export function buildChunkhoundConfig(
    embedding: ConfiguratorProviderOption,
    llm: ConfiguratorProviderOption,
): ConfigRecord {
    return {
        embedding: configWithApiKey(embedding),
        llm: configWithApiKey(llm),
    };
}

export function buildChunkhoundCommand(
    embedding: ConfiguratorProviderOption,
    llm: ConfiguratorProviderOption,
    platform: ConfiguratorPlatform = DEFAULT_PLATFORM,
): string {
    return echoCommand(
        ".chunkhound.json",
        buildChunkhoundConfig(embedding, llm),
        platform,
    );
}

export function buildEditorCommands(
    platform: ConfiguratorPlatform = DEFAULT_PLATFORM,
): Record<string, string> {
    return Object.fromEntries(
        editors.map((editor) => [
            editor.id,
            editor.rawCmd ??
                echoCommand(getEditorFilePath(editor, platform), editor.mcp ?? {}, platform),
        ]),
    );
}

export function buildPrettyEditorCommand(
    editor: ConfiguratorEditor,
    platform: ConfiguratorPlatform = DEFAULT_PLATFORM,
): { shellCmd: string; htmlHighlighted: string; plainCopy: string } {
    if (editor.rawCmd) {
        const html = highlightInlineShellBlock(editor.rawCmd);
        return {
            shellCmd: editor.rawCmd,
            htmlHighlighted: html,
            plainCopy: editor.rawCmd,
        };
    }
    const { plain, html } = prettifyJsonBlock(editor.mcp ?? {});
    const editorFilePath = getEditorFilePath(editor, platform);
    const shell = buildJsonWriteCommand(editorFilePath, plain, platform);
    const htmlOut = renderMixedJsonWriteBlock(
        editorFilePath,
        html.split("\n"),
        platform,
    );
    return { shellCmd: shell, htmlHighlighted: htmlOut, plainCopy: shell };
}

export function buildPrettyEditorCommands(): Record<
    string,
    { shellCmd: string; htmlHighlighted: string; plainCopy: string }
> {
    return Object.fromEntries(
        editors.map((editor) => [
            editor.id,
            buildPrettyEditorCommand(editor, DEFAULT_PLATFORM),
        ]),
    );
}

// Stable annotations for the docs-page (full-mode) Configurator. Keyed by
// dotted JSON path. Values describe what the key *is*, never which value is
// currently picked, so the table is independent of pill selection.
export const CONFIG_KEY_ANNOTATIONS: Record<string, string> = {
    "embedding.provider": "embedding service identifier",
    "embedding.model": "model name",
    "embedding.base_url": "OpenAI-compatible endpoint URL",
    "embedding.rerank_url": "separate reranker endpoint",
    "embedding.rerank_model": "model name for reranking endpoint",
    "embedding.rerank_format": "reranker response format",
    "embedding.api_key": "replace with your API key",
    "llm.provider": "which provider runs `chunkhound research`",
    "llm.base_url": "OpenAI-compatible endpoint URL",
    "llm.api_key": "replace with your API key",
};

function escapeHtml(s: string): string {
    return s
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}

export function highlightJsonLine(line: string): string {
    const m = line.match(/^(\s*)(.*)$/);
    if (!m) return escapeHtml(line);
    const [, indent, content] = m;
    if (!content) return indent;

    // "key": "value",  or  "key": "value"
    let match = content.match(/^"([^"]+)":\s*"([^"]*)"(,?)$/);
    if (match) {
        const [, key, value, comma] = match;
        return (
            indent +
            `<span class="json-key">"${escapeHtml(key)}"</span>` +
            `<span class="json-punct">:</span> ` +
            `<span class="json-string">"${escapeHtml(value)}"</span>` +
            (comma ? `<span class="json-punct">${comma}</span>` : "")
        );
    }
    // "key": <number/bool/null>,?
    match = content.match(/^"([^"]+)":\s*([^"{}\[\],]+)(,?)$/);
    if (match) {
        const [, key, value, comma] = match;
        return (
            indent +
            `<span class="json-key">"${escapeHtml(key)}"</span>` +
            `<span class="json-punct">:</span> ` +
            `<span class="json-string">${escapeHtml(value)}</span>` +
            (comma ? `<span class="json-punct">${comma}</span>` : "")
        );
    }
    // "key": {  or  "key": [
    match = content.match(/^"([^"]+)":\s*([{[])$/);
    if (match) {
        const [, key, bracket] = match;
        return (
            indent +
            `<span class="json-key">"${escapeHtml(key)}"</span>` +
            `<span class="json-punct">: ${escapeHtml(bracket)}</span>`
        );
    }
    // bare string array element: "value",?
    match = content.match(/^"([^"]*)"(,?)$/);
    if (match) {
        const [, value, comma] = match;
        return (
            indent +
            `<span class="json-string">"${escapeHtml(value)}"</span>` +
            (comma ? `<span class="json-punct">${comma}</span>` : "")
        );
    }
    // structural punctuation: { } [ ] }, ],
    if (/^[{}\[\]](,?)$/.test(content)) {
        return (
            indent +
            `<span class="json-punct">${escapeHtml(content)}</span>`
        );
    }
    return indent + escapeHtml(content);
}

function prettifyJsonBlock(obj: unknown): { plain: string; html: string } {
    const plain = JSON.stringify(obj, null, 2);
    const html = plain.split("\n").map(highlightJsonLine).join("\n");
    return { plain, html };
}

export function highlightShellLine(line: string): string {
    // Mirror Configurator.astro's bash highlighter for the heredoc lines so
    // tokens render with the same colours as the rest of the code panel.
    // Escape HTML first because heredoc lines contain `<<` which the browser
    // would otherwise mis-parse as the start of a tag.
    const escaped = line
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
    return escaped
        .replace(/^(\w+)/, '<span class="sh-cmd">$1</span>')
        .replace(/'([^']*)'/, "'<span class=\"sh-str\">$1</span>'")
        .replace(/ &gt;&gt; /, ' <span class="sh-op">&gt;&gt;</span> ')
        .replace(/ &gt; /, ' <span class="sh-op">&gt;</span> ');
}

function wrapShellToken(value: string, cls: string): string {
    return `<span class="${cls}">${escapeHtml(value)}</span>`;
}

function highlightPowerShellLine(line: string): string {
    if (!line) return "";

    const setContentMatch = line.match(
        /^'@ \| Set-Content -Path (.+) -Encoding (.+)$/,
    );
    if (setContentMatch) {
        const [, path, encoding] = setContentMatch;
        return [
            wrapShellToken("'@", "sh-op"),
            " ",
            wrapShellToken("|", "sh-op"),
            " ",
            wrapShellToken("Set-Content", "sh-cmd"),
            " ",
            wrapShellToken("-Path", "sh-op"),
            " ",
            wrapShellToken(path, "sh-file"),
            " ",
            wrapShellToken("-Encoding", "sh-op"),
            " ",
            escapeHtml(encoding),
        ].join("");
    }

    const tokens = line.match(/"[^"]*"|'(?:''|[^'])*'|[|]|[^\s|]+|\s+/g) ?? [line];
    const rendered: string[] = [];
    let expectCommand = true;
    let expectPath = false;

    for (const token of tokens) {
        if (/^\s+$/.test(token)) {
            rendered.push(token);
            continue;
        }

        if (token === "|") {
            rendered.push(wrapShellToken(token, "sh-op"));
            expectCommand = true;
            expectPath = false;
            continue;
        }

        if (token === "@'" || token === "'@") {
            rendered.push(wrapShellToken(token, "sh-op"));
            expectCommand = false;
            expectPath = false;
            continue;
        }

        if (token.startsWith("-")) {
            rendered.push(wrapShellToken(token, "sh-op"));
            expectCommand = false;
            expectPath = token === "-Path";
            continue;
        }

        if (expectPath) {
            rendered.push(wrapShellToken(token, "sh-file"));
            expectPath = false;
            expectCommand = false;
            continue;
        }

        if (expectCommand) {
            rendered.push(wrapShellToken(token, "sh-cmd"));
            expectCommand = false;
            continue;
        }

        if (
            (/^".*"$/.test(token) || /^'.*'$/.test(token)) &&
            (token.includes("/") || token.includes("\\") || token.includes(".json"))
        ) {
            rendered.push(wrapShellToken(token, "sh-file"));
            continue;
        }

        if (/^".*"$/.test(token) || /^'.*'$/.test(token)) {
            rendered.push(wrapShellToken(token, "sh-str"));
            continue;
        }

        rendered.push(escapeHtml(token));
    }

    return rendered.join("");
}

export function highlightInlineShellLine(line: string): string {
    if (!line) return "";
    if (
        line.includes("Set-Content") ||
        line.includes("New-Item") ||
        line.includes("Add-Content") ||
        line.includes("Test-Path") ||
        line === "@'" ||
        line.startsWith("'@")
    ) {
        return highlightPowerShellLine(line);
    }

    return highlightShellLine(line)
        .replace(/(^| )([\w.~/-]+)$/g, (_match, prefix: string, path: string) =>
            `${prefix}<span class="sh-file">${escapeHtml(path)}</span>`,
        );
}

export function highlightInlineShellBlock(text: string): string {
    return text.split("\n").map(highlightInlineShellLine).join("\n");
}

type ConfigLine = { text: string; path?: string };

function renderAnnotatedJsonLines(lines: ConfigLine[]): string[] {
    const annotatedLines = lines.filter(
        (line) => line.path && CONFIG_KEY_ANNOTATIONS[line.path],
    );
    const maxLen = annotatedLines.reduce(
        (longest, line) => Math.max(longest, line.text.length),
        0,
    );

    return lines.map((line) => {
        const tokenized = highlightJsonLine(line.text);
        if (line.path && CONFIG_KEY_ANNOTATIONS[line.path]) {
            const pad = " ".repeat(maxLen - line.text.length + 2);
            const comment = `${pad}// ${CONFIG_KEY_ANNOTATIONS[line.path]}`;
            return (
                tokenized +
                `<span class="json-comment">${escapeHtml(comment)}</span>`
            );
        }
        return tokenized;
    });
}

function renderMixedJsonWriteBlock(
    filename: string,
    jsonHtmlLines: string[],
    platform: ConfiguratorPlatform,
): string {
    const htmlLines: string[] = [];
    const parentDir = getParentDir(filename);

    if (parentDir) {
        if (platform === "powershell") {
            htmlLines.push(
                highlightInlineShellLine(
                    `New-Item -ItemType Directory -Force -Path ${quotePowerShell(parentDir)} | Out-Null`,
                ),
            );
        } else {
            htmlLines.push(highlightShellLine(`mkdir -p ${parentDir}`));
        }
    }

    if (platform === "powershell") {
        htmlLines.push(
            highlightInlineShellLine("@'"),
            ...jsonHtmlLines,
            highlightInlineShellLine(
                `'@ | Set-Content -Path ${quotePowerShell(filename)} -Encoding utf8`,
            ),
        );
        return htmlLines.join("\n");
    }

    htmlLines.push(
        highlightShellLine(`cat > ${filename} <<'CHUNKHOUND_EOF'`),
        ...jsonHtmlLines,
        highlightShellLine("CHUNKHOUND_EOF"),
    );
    return htmlLines.join("\n");
}

export function buildPrettyConfigJson(
    embedding: ConfiguratorProviderOption,
    llm: ConfiguratorProviderOption,
    platform: ConfiguratorPlatform = DEFAULT_PLATFORM,
): { shellCmd: string; htmlAnnotated: string; plainCopy: string } {
    const config = buildChunkhoundConfig(embedding, llm);
    const sections: Array<[string, ConfigRecord]> = [
        ["embedding", config.embedding as ConfigRecord],
        ["llm", config.llm as ConfigRecord],
    ];

    const lines: ConfigLine[] = [];
    lines.push({ text: "{" });
    sections.forEach(([sectionKey, sectionVal], sIdx) => {
        const sectionTrailing = sIdx === sections.length - 1 ? "" : ",";
        lines.push({ text: `  "${sectionKey}": {` });
        const entries = Object.entries(sectionVal);
        entries.forEach(([k, v], eIdx) => {
            const trailing = eIdx === entries.length - 1 ? "" : ",";
            const valueStr = JSON.stringify(v);
            lines.push({
                text: `    "${k}": ${valueStr}${trailing}`,
                path: `${sectionKey}.${k}`,
            });
        });
        lines.push({ text: `  }${sectionTrailing}` });
    });
    lines.push({ text: "}" });

    const cleanJson = lines.map((l) => l.text).join("\n");
    const plainCopy = buildJsonWriteCommand(".chunkhound.json", cleanJson, platform);
    const htmlAnnotated = renderMixedJsonWriteBlock(
        ".chunkhound.json",
        renderAnnotatedJsonLines(lines),
        platform,
    );

    return { shellCmd: plainCopy, htmlAnnotated, plainCopy };
}

export function buildCompactConfiguratorOutput(
    embedding: ConfiguratorProviderOption,
    llm: ConfiguratorProviderOption,
    editorId: string,
    platform: ConfiguratorPlatform = DEFAULT_PLATFORM,
): { copy: string; html: string } {
    const editorCommands = buildEditorCommands(platform);
    const providerCmd = buildChunkhoundCommand(embedding, llm, platform);
    const copy = [
        buildGitignoreCommand(platform),
        providerCmd,
        editorCommands[editorId],
        INDEX_CMD,
    ].join("\n");
    return { copy, html: highlightInlineShellBlock(copy) };
}

export function buildFullConfiguratorOutput(
    embedding: ConfiguratorProviderOption,
    llm: ConfiguratorProviderOption,
    editorId: string,
    platform: ConfiguratorPlatform = DEFAULT_PLATFORM,
): { copy: string; html: string } {
    const pretty = buildPrettyConfigJson(embedding, llm, platform);
    const prettyEditorCommands = Object.fromEntries(
        editors.map((editor) => [
            editor.id,
            buildPrettyEditorCommand(editor, platform),
        ]),
    );
    const editor = prettyEditorCommands[editorId];
    const gitignoreCommand = buildGitignoreCommand(platform);
    const copy = `${gitignoreCommand}\n${pretty.plainCopy}\n\n${editor.plainCopy}`;
    const html =
        `${highlightInlineShellBlock(gitignoreCommand)}\n${pretty.htmlAnnotated}\n\n${editor.htmlHighlighted}`;
    return { copy, html };
}
