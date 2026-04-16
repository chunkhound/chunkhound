from __future__ import annotations

from tests.site.tsx_runner import run_tsx_json


def _render_full_output(embedding_id: str, llm_id: str, editor_id: str) -> dict:
    script = f"""
import {{
  buildFullConfiguratorOutput,
  embeddingProviders,
  llmProviders,
}} from './site/src/components/configurator-data.ts';

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


def test_full_mode_heredoc_opener_keeps_initial_tokenization_across_selections(
) -> None:
    default_output = _render_full_output("voyageai", "anthropic", "cursor")
    alternate_output = _render_full_output("vllm-embed", "grok", "cursor")

    assert "echo .chunkhound.json >> .gitignore" in default_output["copy"]
    assert "cat > .chunkhound.json <<'EOF'" in default_output["copy"]
    assert "echo .chunkhound.json >> .gitignore" in alternate_output["copy"]
    assert "cat > .chunkhound.json <<'EOF'" in alternate_output["copy"]
    assert ".chunkhound.json" in default_output["html"]
    assert ".chunkhound.json" in alternate_output["html"]
    assert "<<'EOF'" in default_output["copy"]
    assert "<<'EOF'" in alternate_output["copy"]


def test_full_mode_renderer_outputs_stable_html_and_copy_for_non_default_selection(
) -> None:
    rendered = _render_full_output("ollama-embed", "codex-cli", "vscode")

    assert "echo .chunkhound.json >> .gitignore" in rendered["copy"]
    assert "cat > .chunkhound.json <<'EOF'" in rendered["copy"]
    assert "mkdir -p .vscode" in rendered["copy"]
    assert "cat > .vscode/mcp.json <<'EOF'" in rendered["copy"]
    assert "qwen3-embedding" in rendered["html"]
    assert "codex-cli" in rendered["html"]
    assert '<span class="json-comment">' in rendered["html"]


def test_compact_mode_prepends_parent_directory_creation_for_nested_editor_files(
) -> None:
    script = """
import {
  buildCompactConfiguratorOutput,
  embeddingProviders,
  llmProviders,
} from './site/src/components/configurator-data.ts';

const embedding = embeddingProviders.find((provider) => provider.id === 'voyageai');
const llm = llmProviders.find((provider) => provider.id === 'anthropic');
if (!embedding || !llm) {
  throw new Error('missing provider');
}

console.log(JSON.stringify(buildCompactConfiguratorOutput(embedding, llm, 'cursor')));
"""
    rendered = run_tsx_json(script)
    assert "\nmkdir -p .cursor\necho " in rendered["copy"]


def test_compact_mode_skips_parent_directory_creation_for_root_editor_files() -> None:
    script = """
import {
  buildCompactConfiguratorOutput,
  embeddingProviders,
  llmProviders,
} from './site/src/components/configurator-data.ts';

const embedding = embeddingProviders.find((provider) => provider.id === 'voyageai');
const llm = llmProviders.find((provider) => provider.id === 'anthropic');
if (!embedding || !llm) {
  throw new Error('missing provider');
}

console.log(JSON.stringify(buildCompactConfiguratorOutput(embedding, llm, 'opencode')));
"""
    rendered = run_tsx_json(script)
    assert "\nmkdir -p " not in rendered["copy"]
    assert "\necho '{\"mcp\"" in rendered["copy"]


def test_full_mode_renders_powershell_commands_for_windows_selection() -> None:
    script = """
import {
  buildFullConfiguratorOutput,
  embeddingProviders,
  llmProviders,
} from './site/src/components/configurator-data.ts';

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
        "'@ | Set-Content -Path '.chunkhound.json' -Encoding utf8"
        in rendered["copy"]
    )
    assert (
        "New-Item -ItemType Directory -Force -Path '.vscode' | Out-Null"
        in rendered["copy"]
    )
    assert (
        "'@ | Set-Content -Path '.vscode/mcp.json' -Encoding utf8"
        in rendered["copy"]
    )
    assert ".chunkhound.json" in rendered["html"]
    assert ".vscode/mcp.json" in rendered["html"]
    assert "Set-Content" in rendered["html"]
    assert "New-Item" in rendered["html"]
    assert "json-comment" in rendered["html"]


def test_full_mode_renders_windsurf_powershell_path_with_home_expansion() -> None:
    script = """
import {
  buildFullConfiguratorOutput,
  embeddingProviders,
  llmProviders,
} from './site/src/components/configurator-data.ts';

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
        '| Out-Null' in rendered["copy"]
    )
    assert (
        '\'@ | Set-Content -Path "$HOME/.codeium/windsurf/mcp_config.json" '
        '-Encoding utf8' in rendered["copy"]
    )
    assert "$HOME/.codeium/windsurf/mcp_config.json" in rendered["html"]
