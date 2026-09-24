from __future__ import annotations

import re

from tests.site.tsx_runner import run_tsx_json


def _render_full_output(embedding_id: str, llm_id: str, editor_id: str) -> dict:
    script = f"""
import {{
  buildFullConfiguratorOutput,
  embeddingProviders,
  llmProviders,
}} from './site/src/components/configurator/index.ts';

const embedding = embeddingProviders.find(
  (provider) => provider.id === '{embedding_id}'
);
const llm = llmProviders.find(
  (provider) => provider.id === '{llm_id}'
);
if (!embedding || !llm) {{
  throw new Error('missing provider');
}}

console.log(JSON.stringify(buildFullConfiguratorOutput(embedding, llm, '{editor_id}')));
"""
    return run_tsx_json(script)


def _render_compact_output(embedding_id: str, llm_id: str, editor_id: str) -> dict:
    script = f"""
import {{
  buildCompactConfiguratorOutput,
  embeddingProviders,
  llmProviders,
}} from './site/src/components/configurator/index.ts';

const embedding = embeddingProviders.find(
  (provider) => provider.id === '{embedding_id}'
);
const llm = llmProviders.find(
  (provider) => provider.id === '{llm_id}'
);
if (!embedding || !llm) {{
  throw new Error('missing provider');
}}

console.log(JSON.stringify(buildCompactConfiguratorOutput(embedding, llm, '{editor_id}')));
"""
    return run_tsx_json(script)


def test_full_mode_heredoc_opener_keeps_initial_tokenization_across_selections() -> (
    None
):
    default_output = _render_full_output("voyageai", "anthropic", "cursor")
    alternate_output = _render_full_output("vllm-embed", "grok", "cursor")

    assert "echo .chunkhound.json >> .gitignore" in default_output["copy"]
    assert "cat > .chunkhound.json <<'CHUNKHOUND_EOF'" in default_output["copy"]
    assert "echo .chunkhound.json >> .gitignore" in alternate_output["copy"]
    assert "cat > .chunkhound.json <<'CHUNKHOUND_EOF'" in alternate_output["copy"]
    assert ".chunkhound.json" in default_output["html"]
    assert ".chunkhound.json" in alternate_output["html"]
    assert "<<'CHUNKHOUND_EOF'" in default_output["copy"]
    assert "<<'CHUNKHOUND_EOF'" in alternate_output["copy"]
    assert "\nCHUNKHOUND_EOF" in default_output["copy"]
    assert "CHUNKHOUND_EOF" in default_output["html"]
    assert "CHUNKHOUND_EOF" in alternate_output["html"]


def test_default_route_surfaces_required_api_keys_in_copy() -> None:
    """The default route's copyable output must name the secrets a user has to
    obtain before running — previously the requirement lived only in the UI."""
    compact = _render_compact_output("voyageai", "openrouter", "cursor")
    full = _render_full_output("voyageai", "openrouter", "cursor")

    expected = "# You'll need: VoyageAI API key · OpenRouter API key"
    for rendered in (compact, full):
        assert expected in rendered["copy"]
        assert "You'll need:" in rendered["copy"]
        assert "OpenRouter API key" in rendered["copy"]
        assert "VoyageAI API key" in rendered["copy"]


def test_output_omits_requirements_comment_without_api_key_requirements() -> None:
    """Ollama/vLLM routes need no API key, so no spurious prerequisite line."""
    rendered = _render_compact_output("ollama-embed", "ollama-llm", "cursor")

    assert "You'll need:" not in rendered["copy"]


def test_output_deduplicates_shared_api_key_requirements() -> None:
    """OpenAI embedding + OpenAI LLM share one key: list it exactly once."""
    rendered = _render_compact_output("openai-embed", "openai-llm", "cursor")

    assert rendered["copy"].count("OpenAI API key") == 1
    assert "# You'll need: OpenAI API key" in rendered["copy"]


def test_full_mode_output_is_copy_pasteable_without_json_comments() -> None:
    """The displayed config must parse as-is: ChunkHound's config parser
    rejects `//` comments, so the rendered HTML carries none."""
    rendered = _render_full_output("voyageai", "openrouter", "cursor")

    assert '"model": "voyage-4-lite"' in rendered["copy"]
    assert '"model": "google/gemini-3.5-flash"' in rendered["copy"]

    assert "json-comment" not in rendered["html"]
    stripped = re.sub(r"<[^>]+>", "", rendered["html"])
    for line in stripped.split("\n"):
        # (?<!:) allows URL values like "https://..." while catching
        # trailing `// note` annotations.
        assert not re.search(r"(?<!:)//", line), line


def test_every_configurator_output_has_copyable_shell_comments() -> None:
    script = """
import {
  buildCompactConfiguratorOutput,
  buildFullConfiguratorOutput,
  editors,
  embeddingProviders,
  llmProviders,
  PLATFORM_OPTIONS,
} from './site/src/components/configurator/index.ts';

const errors = [];
let visited = 0;
const builders = [
  ['compact', buildCompactConfiguratorOutput],
  ['full', buildFullConfiguratorOutput],
];

function jsonBodies(copy, platform) {
  // Only the write branch parses as JSON: the merge-guard print branch carries
  // a human instruction line above the payload. Posix anchors on `cat >`
  // (write) vs `cat <<` (print); PowerShell anchors on a bare `@'` opener line
  // (the print branch opens with `Write-Host @'`) plus the `'@ | Set-Content`
  // closer — both only exist on the write branch.
  const pattern = platform === 'posix'
    ? /cat > [^\\n]*<<'CHUNKHOUND_EOF'\\n([\\s\\S]*?)\\nCHUNKHOUND_EOF/g
    : /^@'\\n([\\s\\S]*?)\\n'@ \\| Set-Content/gm;
  return [...copy.matchAll(pattern)].map((match) => match[1]);
}

function textFromHtml(html) {
  return html.replace(/<[^>]+>/g, '')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>');
}

function expect(condition, label) {
  if (!condition) errors.push(label);
}

for (const embedding of embeddingProviders) {
  for (const llm of llmProviders) {
    for (const editor of editors) {
      for (const platform of PLATFORM_OPTIONS) {
        for (const [mode, build] of builders) {
          visited += 1;
          const rendered = build(embedding, llm, editor.id, platform.id);
          const configComment = [
            `# Configure ChunkHound with ${embedding.name} embeddings`,
            `and ${llm.name} for research. Embedding and LLM providers are independent`,
            '— either works without the other. Field details:',
            'https://chunkhound.ai/docs/configuration/',
          ].join(' ');
          const editorComment = [
            `# Configure ${editor.name} to connect to ChunkHound`,
            'via MCP.',
          ].join(' ');
          const bodies = jsonBodies(rendered.copy, platform.id);
          const expectedBodyCount = 1 + Number(!editor.rawCmd);
          const label = `${mode}/${platform.id}/${editor.id}`;

          expect(rendered.copy.includes(configComment), `${label}: config comment`);
          expect(rendered.copy.includes(editorComment), `${label}: editor comment`);
          expect(
            rendered.copy.includes(
              '# Keep generated configuration files out of version control.',
            ),
            `${label}: gitignore comment`,
          );
          expect(
            mode !== 'compact' || rendered.copy.includes(
              "# Build ChunkHound's search index for this project.",
            ),
            `${label}: index comment`,
          );
          expect(rendered.html.includes('sh-comment'), `${label}: highlighted comment`);
          expect(
            textFromHtml(rendered.html) === rendered.copy,
            `${label}: rendered text differs from copy`,
          );
          expect(bodies.length === expectedBodyCount, `${label}: JSON body count`);

          for (const body of bodies) {
            try {
              JSON.parse(body);
            } catch {
              errors.push(`${mode}/${platform.id}/${editor.id}: invalid JSON`);
            }
          }
        }
      }
    }
  }
}

console.log(JSON.stringify({
  count:
    embeddingProviders.length * llmProviders.length * editors.length *
    PLATFORM_OPTIONS.length * builders.length,
  visited,
  errors,
}));
"""
    rendered = run_tsx_json(script)

    assert rendered["visited"] == rendered["count"]
    assert rendered["errors"] == []


def test_full_mode_renderer_outputs_stable_html_and_copy_for_non_default_selection(
) -> None:
    rendered = _render_full_output("ollama-embed", "codex-cli", "vscode")

    assert "echo .chunkhound.json >> .gitignore" in rendered["copy"]
    assert "cat > .chunkhound.json <<'CHUNKHOUND_EOF'" in rendered["copy"]
    assert "mkdir -p .vscode" in rendered["copy"]
    assert "cat > .vscode/mcp.json <<'CHUNKHOUND_EOF'" in rendered["copy"]
    assert "\nCHUNKHOUND_EOF" in rendered["copy"]
    assert "qwen3-embedding" in rendered["html"]
    assert "codex-cli" in rendered["html"]
    assert '<span class="json-comment">' not in rendered["html"]


def test_pi_output_installs_adapter_writes_project_config_and_ignores_it() -> None:
    rendered = _render_full_output("voyageai", "openrouter", "pi")

    assert rendered["copy"].startswith(
        "# Keep generated configuration files out of version control.\n"
        "echo .chunkhound.json >> .gitignore\necho .mcp.json >> .gitignore"
    )
    assert "pi install npm:pi-mcp-adapter" in rendered["copy"]
    assert "cat > .mcp.json <<'CHUNKHOUND_EOF'" in rendered["copy"]
    html = re.sub(r"<[^>]+>", "", rendered["html"])
    for value in (
        '"directTools": true',
        '"lifecycle": "eager"',
        '"requestTimeoutMs": 1200000',
        '"outputGuard": false',
    ):
        assert value in rendered["copy"]
        assert value in html


def test_compact_mode_prepends_parent_directory_creation_for_nested_editor_files() -> (
    None
):
    script = """
import {
  buildCompactConfiguratorOutput,
  embeddingProviders,
  llmProviders,
} from './site/src/components/configurator/index.ts';

const embedding = embeddingProviders.find((provider) => provider.id === 'voyageai');
const llm = llmProviders.find((provider) => provider.id === 'anthropic');
if (!embedding || !llm) {
  throw new Error('missing provider');
}

console.log(JSON.stringify(buildCompactConfiguratorOutput(embedding, llm, 'cursor')));
"""
    rendered = run_tsx_json(script)
    assert "\nmkdir -p .cursor\ncat > " in rendered["copy"]


def test_compact_mode_skips_parent_directory_creation_for_root_editor_files() -> None:
    script = """
import {
  buildCompactConfiguratorOutput,
  embeddingProviders,
  llmProviders,
} from './site/src/components/configurator/index.ts';

const embedding = embeddingProviders.find((provider) => provider.id === 'voyageai');
const llm = llmProviders.find((provider) => provider.id === 'anthropic');
if (!embedding || !llm) {
  throw new Error('missing provider');
}

console.log(JSON.stringify(buildCompactConfiguratorOutput(embedding, llm, 'opencode')));
"""
    rendered = run_tsx_json(script)
    assert "\nmkdir -p " not in rendered["copy"]
    assert "\ncat > opencode.json <<'CHUNKHOUND_EOF'\n" in rendered["copy"]
    assert "\nCHUNKHOUND_EOF" in rendered["copy"]


def test_full_mode_renders_powershell_commands_for_windows_selection() -> None:
    script = """
import {
  buildFullConfiguratorOutput,
  embeddingProviders,
  llmProviders,
} from './site/src/components/configurator/index.ts';

const embedding = embeddingProviders.find((provider) => provider.id === 'voyageai');
const llm = llmProviders.find((provider) => provider.id === 'anthropic');
if (!embedding || !llm) {
  throw new Error('missing provider');
}

console.log(JSON.stringify(
  buildFullConfiguratorOutput(embedding, llm, 'vscode', 'powershell')
));
"""
    rendered = run_tsx_json(script)

    assert "Add-Content -Path .gitignore -Value '.chunkhound.json'" in rendered["copy"]
    assert "@'\n{" in rendered["copy"]
    assert (
        "'@ | Set-Content -Path '.chunkhound.json' -Encoding utf8" in rendered["copy"]
    )
    assert (
        "New-Item -ItemType Directory -Force -Path '.vscode' | Out-Null"
        in rendered["copy"]
    )
    assert (
        "'@ | Set-Content -Path '.vscode/mcp.json' -Encoding utf8" in rendered["copy"]
    )
    assert ".chunkhound.json" in rendered["html"]
    assert ".vscode/mcp.json" in rendered["html"]
    assert "Set-Content" in rendered["html"]
    assert "New-Item" in rendered["html"]
    assert "json-comment" not in rendered["html"]


def test_full_mode_renders_windsurf_powershell_path_with_home_expansion() -> None:
    script = """
import {
  buildFullConfiguratorOutput,
  embeddingProviders,
  llmProviders,
} from './site/src/components/configurator/index.ts';

const embedding = embeddingProviders.find((provider) => provider.id === 'voyageai');
const llm = llmProviders.find((provider) => provider.id === 'anthropic');
if (!embedding || !llm) {
  throw new Error('missing provider');
}

console.log(JSON.stringify(
  buildFullConfiguratorOutput(embedding, llm, 'windsurf', 'powershell')
));
"""
    rendered = run_tsx_json(script)

    assert "~/.codeium/windsurf/mcp_config.json" not in rendered["copy"]
    assert (
        'New-Item -ItemType Directory -Force -Path "$HOME/.codeium/windsurf" '
        "| Out-Null" in rendered["copy"]
    )
    assert (
        '\'@ | Set-Content -Path "$HOME/.codeium/windsurf/mcp_config.json" '
        "-Encoding utf8" in rendered["copy"]
    )
    assert "$HOME/.codeium/windsurf/mcp_config.json" in rendered["html"]
