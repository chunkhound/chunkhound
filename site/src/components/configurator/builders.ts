import type {
  ConfigRecord,
  ConfiguratorEditor,
  ConfiguratorPlatform,
  ConfiguratorProviderOption,
  ConfiguratorReranker,
  ConfiguratorRequirement,
} from "./types.ts";
import {
  highlightInlineShellBlock,
  prettifyJsonBlock,
  renderMixedGuardedJsonWriteBlock,
  renderMixedJsonWriteBlock,
} from "./highlight.ts";
import {
  CONFIG_FILENAME,
  CONFIGURATION_DOCS_URL,
  DEFAULT_PLATFORM,
  INDEX_CMD,
} from "./constants.ts";
import { editors, findEditor } from "./providers.ts";
import { requirementLabels } from "./utils.ts";
import {
  assembleGuardedJsonWrite,
  assembleJsonWrite,
  guardedJsonWriteScaffold,
  jsonWriteScaffold,
} from "./shell-write.ts";

export function configWithApiKey(
  option: ConfiguratorProviderOption,
): ConfigRecord {
  return option.apiKeyPlaceholder
    ? { ...option.config, api_key: option.apiKeyPlaceholder }
    : option.config;
}

function getEditorFilePath(
  editor: ConfiguratorEditor,
  platform: ConfiguratorPlatform,
): string {
  if (platform === "powershell" && editor.mcpFilePowerShell) {
    return editor.mcpFilePowerShell;
  }
  if (!editor.mcpFile)
    throw new Error(`Editor '${editor.id}' is missing mcpFile`);
  return editor.mcpFile;
}

// Single source of truth for the gitignore annotation used in both output modes.
const GITIGNORE_COMMENT =
  "Keep generated configuration files out of version control.";

function buildGitignoreCommand(
  editor: ConfiguratorEditor,
  platform: ConfiguratorPlatform,
): string {
  const entries = [CONFIG_FILENAME, ...(editor.gitignoreEntries ?? [])];
  if (platform === "powershell") {
    return [
      "if (-not (Test-Path .gitignore)) { New-Item -ItemType File -Path .gitignore | Out-Null }",
      ...entries.map(
        (entry) => `Add-Content -Path .gitignore -Value '${entry}'`,
      ),
    ].join("\n");
  }
  return entries.map((entry) => `echo ${entry} >> .gitignore`).join("\n");
}

// guard: editor MCP files can hold servers/settings the configurator does not
// own, so their writes are merge-guarded against clobbering an existing file.
// .chunkhound.json is fully generated and stays unguarded (re-runs rewrite it).
function buildJsonWriteCommand(
  filename: string,
  content: string,
  platform: ConfiguratorPlatform,
  guard: boolean,
): string {
  if (guard) {
    return assembleGuardedJsonWrite(
      guardedJsonWriteScaffold(filename, platform),
      content,
    );
  }
  return assembleJsonWrite(jsonWriteScaffold(filename, platform), content);
}

// Writes a heredoc/here-string block for `filename`, not an echo — named for
// what it does, not the platform-specific shell builtin it may delegate to.
function writeBlockCommand(
  filename: string,
  content: ConfigRecord,
  platform: ConfiguratorPlatform,
  guard: boolean,
): string {
  return buildJsonWriteCommand(
    filename,
    JSON.stringify(content, null, 2),
    platform,
    guard,
  );
}

function configWithReranker(
  embedding: ConfiguratorProviderOption,
  reranker?: ConfiguratorReranker,
): ConfigRecord {
  const url = reranker?.url?.trim();
  const format = reranker?.format?.trim();
  const model = reranker?.model?.trim();
  return {
    ...configWithApiKey(embedding),
    ...(url && { rerank_url: url }),
    ...(format && { rerank_format: format }),
    ...(model && { rerank_model: model }),
  };
}

export function buildChunkhoundConfig(
  embedding: ConfiguratorProviderOption,
  llm: ConfiguratorProviderOption,
  reranker?: ConfiguratorReranker,
): ConfigRecord {
  return {
    embedding: configWithReranker(embedding, reranker),
    llm: configWithApiKey(llm),
  };
}

export function buildChunkhoundCommand(
  embedding: ConfiguratorProviderOption,
  llm: ConfiguratorProviderOption,
  platform: ConfiguratorPlatform = DEFAULT_PLATFORM,
  reranker?: ConfiguratorReranker,
): string {
  return writeBlockCommand(
    CONFIG_FILENAME,
    buildChunkhoundConfig(embedding, llm, reranker),
    platform,
    false,
  );
}

function joinSetupCommands(commands: Array<string | undefined>): string {
  return commands
    .filter((command): command is string => Boolean(command))
    .join("\n");
}

function shellComment(text: string): string {
  return `# ${text}`;
}

function annotateCommand(
  comment: string | Array<string | undefined>,
  command: string,
): string {
  const comments = (Array.isArray(comment) ? comment : [comment]).filter(
    (text): text is string => Boolean(text),
  );
  return joinSetupCommands([...comments.map(shellComment), command]);
}

function configComment(
  embedding: ConfiguratorProviderOption,
  llm: ConfiguratorProviderOption,
): string {
  return `Configure ChunkHound with ${embedding.name} embeddings and ${llm.name} for research. Embedding and LLM providers are independent — either works without the other. Field details: ${CONFIGURATION_DOCS_URL}`;
}

// API-key prerequisites are the secrets a user must obtain before running;
// model/tool requirements stay in the stage panel ("You'll need").
// Dedupe by id: the embedding and LLM presets can require the same key
// (e.g. OpenAI for both) and must render once.
function requirementsComment(
  embedding: ConfiguratorProviderOption,
  llm: ConfiguratorProviderOption,
): string | undefined {
  const keys = new Map<string, ConfiguratorRequirement>();
  for (const requirement of [...embedding.requirements, ...llm.requirements]) {
    if (requirement.id.endsWith("-api-key")) {
      keys.set(requirement.id, requirement);
    }
  }
  const labels = requirementLabels([...keys.values()]);
  return labels ? `You'll need: ${labels}` : undefined;
}

function editorComment(editor: ConfiguratorEditor): string {
  return `Configure ${editor.name} to connect to ChunkHound via MCP.`;
}

function buildEditorCommand(
  editor: ConfiguratorEditor,
  platform: ConfiguratorPlatform,
): string {
  return editor.rawCmd ??
    writeBlockCommand(
      getEditorFilePath(editor, platform),
      editor.mcp ?? {},
      platform,
      true,
    );
}

export function buildEditorCommands(
  platform: ConfiguratorPlatform = DEFAULT_PLATFORM,
): Record<string, string> {
  return Object.fromEntries(
    editors.map((editor) => [editor.id, buildEditorCommand(editor, platform)]),
  );
}

export function buildPrettyEditorCommand(
  editor: ConfiguratorEditor,
  platform: ConfiguratorPlatform = DEFAULT_PLATFORM,
): { htmlHighlighted: string; plainCopy: string } {
  if (editor.rawCmd) {
    return {
      htmlHighlighted: highlightInlineShellBlock(editor.rawCmd),
      plainCopy: editor.rawCmd,
    };
  }
  const { plain, html } = prettifyJsonBlock(editor.mcp ?? {});
  const editorFilePath = getEditorFilePath(editor, platform);
  const configShell = buildJsonWriteCommand(editorFilePath, plain, platform, true);
  return {
    htmlHighlighted: renderMixedGuardedJsonWriteBlock(
      editorFilePath,
      html.split("\n"),
      platform,
    ),
    plainCopy: configShell,
  };
}

export function buildPrettyConfigJson(
  embedding: ConfiguratorProviderOption,
  llm: ConfiguratorProviderOption,
  platform: ConfiguratorPlatform = DEFAULT_PLATFORM,
  reranker?: ConfiguratorReranker,
): { htmlAnnotated: string; plainCopy: string } {
  const config = buildChunkhoundConfig(embedding, llm, reranker);
  const { plain, html } = prettifyJsonBlock(config);
  const plainCopy = buildJsonWriteCommand(
    CONFIG_FILENAME,
    plain,
    platform,
    false,
  );
  const htmlAnnotated = renderMixedJsonWriteBlock(
    CONFIG_FILENAME,
    html.split("\n"),
    platform,
  );

  return { htmlAnnotated, plainCopy };
}

// Build the compact plain-text copy as one command-per-line list so the
// ordering and join separator are the single source of truth for both the
// plain and highlighted output.
function buildCompactCopy(
  editor: ConfiguratorEditor,
  embedding: ConfiguratorProviderOption,
  llm: ConfiguratorProviderOption,
  platform: ConfiguratorPlatform,
  reranker: ConfiguratorReranker | undefined,
): string {
  const providerCmd = buildChunkhoundCommand(embedding, llm, platform, reranker);
  return [
    annotateCommand(GITIGNORE_COMMENT, buildGitignoreCommand(editor, platform)),
    annotateCommand(
      [configComment(embedding, llm), requirementsComment(embedding, llm)],
      providerCmd,
    ),
    annotateCommand(editorComment(editor), buildEditorCommand(editor, platform)),
    annotateCommand(
      "Build ChunkHound's search index for this project.",
      INDEX_CMD,
    ),
  ].join("\n");
}

export function buildCompactConfiguratorOutput(
  embedding: ConfiguratorProviderOption,
  llm: ConfiguratorProviderOption,
  editorId: string,
  platform: ConfiguratorPlatform = DEFAULT_PLATFORM,
  reranker?: ConfiguratorReranker,
): { copy: string; html: string } {
  const editor = findEditor(editorId);
  const copy = buildCompactCopy(editor, embedding, llm, platform, reranker);
  return { copy, html: highlightInlineShellBlock(copy) };
}

interface FullHtmlParams {
  gitignore: string;
  configCommentText: string;
  configNotesText?: string;
  configHtml: string;
  editorCommentText: string;
  editorHtml: string;
}

function buildFullHtml(params: FullHtmlParams): string {
  const configComments = [params.configCommentText, params.configNotesText]
    .filter((text): text is string => Boolean(text))
    .map((text) => highlightInlineShellBlock(shellComment(text)))
    .join("\n");
  const config = `${configComments}\n${params.configHtml}`;
  const editor = `${highlightInlineShellBlock(shellComment(params.editorCommentText))}\n${params.editorHtml}`;
  return `${highlightInlineShellBlock(params.gitignore)}\n${config}\n\n${editor}`;
}

// Comment lines are computed once here so the plain copy and the highlighted
// HTML can never disagree about which comments wrap a given command.
interface FullComments {
  gitignore: string;
  configCommentText: string;
  configNotesText: string | undefined;
  editorCommentText: string;
}

function buildFullComments(
  editor: ConfiguratorEditor,
  embedding: ConfiguratorProviderOption,
  llm: ConfiguratorProviderOption,
  platform: ConfiguratorPlatform,
): FullComments {
  return {
    gitignore: annotateCommand(
      GITIGNORE_COMMENT,
      buildGitignoreCommand(editor, platform),
    ),
    configCommentText: configComment(embedding, llm),
    configNotesText: requirementsComment(embedding, llm),
    editorCommentText: editorComment(editor),
  };
}

function buildFullCopy(
  comments: FullComments,
  configPlainCopy: string,
  editorPlainCopy: string,
): string {
  const config = annotateCommand(
    [comments.configCommentText, comments.configNotesText],
    configPlainCopy,
  );
  const editorConfig = annotateCommand(
    comments.editorCommentText,
    editorPlainCopy,
  );
  return `${comments.gitignore}\n${config}\n\n${editorConfig}`;
}

export function buildFullConfiguratorOutput(
  embedding: ConfiguratorProviderOption,
  llm: ConfiguratorProviderOption,
  editorId: string,
  platform: ConfiguratorPlatform = DEFAULT_PLATFORM,
  reranker?: ConfiguratorReranker,
): { copy: string; html: string } {
  const pretty = buildPrettyConfigJson(embedding, llm, platform, reranker);
  const selectedEditor = findEditor(editorId);
  const editor = buildPrettyEditorCommand(selectedEditor, platform);
  const comments = buildFullComments(
    selectedEditor,
    embedding,
    llm,
    platform,
  );
  return {
    copy: buildFullCopy(comments, pretty.plainCopy, editor.plainCopy),
    html: buildFullHtml({
      gitignore: comments.gitignore,
      configCommentText: comments.configCommentText,
      configNotesText: comments.configNotesText,
      configHtml: pretty.htmlAnnotated,
      editorCommentText: comments.editorCommentText,
      editorHtml: editor.htmlHighlighted,
    } satisfies FullHtmlParams),
  };
}
