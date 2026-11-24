"""Merge-guard contract for editor MCP config writes.

WHY: editor config files (.vscode/mcp.json, .zed/settings.json, …) can hold
servers and settings the configurator does not own, and re-running the
generated setup is common. Every editor file write must therefore guard on
file existence and print a merge payload instead of clobbering. The
project-owned .chunkhound.json is fully generated, so it stays unguarded
(re-runs rewrite it) — that asymmetry is deliberate, not an oversight.
"""

from __future__ import annotations

import functools
import re

from tests.site.tsx_runner import run_tsx_json

_SCRIPT = """
import {
  buildCompactConfiguratorOutput,
  buildEditorCommands,
  editors,
  embeddingProviders,
  llmProviders,
  PLATFORM_OPTIONS,
} from './site/src/components/configurator/index.ts';

const out = { editorWrites: {}, compactCopies: {} };
for (const platform of PLATFORM_OPTIONS) {
  const commands = buildEditorCommands(platform.id);
  for (const editor of editors) {
    // rawCmd editors register via their CLI and never write a file.
    if (editor.rawCmd) continue;
    const path =
      platform.id === 'powershell' && editor.mcpFilePowerShell
        ? editor.mcpFilePowerShell
        : editor.mcpFile;
    out.editorWrites[`${platform.id}/${editor.id}`] = {
      path,
      payload: JSON.stringify(editor.mcp ?? {}, null, 2),
      command: commands[editor.id],
    };
  }
  out.compactCopies[platform.id] = buildCompactConfiguratorOutput(
    embeddingProviders[0],
    llmProviders[0],
    'vscode',
    platform.id,
  ).copy;
}
console.log(JSON.stringify(out));
"""


@functools.lru_cache(maxsize=1)
def _rendered() -> dict:
    return run_tsx_json(_SCRIPT)


def _writes(platform: str) -> dict[str, dict]:
    return {
        key.split("/")[1]: value
        for key, value in _rendered()["editorWrites"].items()
        if key.startswith(f"{platform}/")
    }


def test_every_editor_write_guards_on_file_existence_posix() -> None:
    writes = _writes("posix")
    assert writes, "no editor file writes rendered"
    for editor_id, write in writes.items():
        command = write["command"]
        assert f"if [ -f {write['path']} ]; then" in command, editor_id
        assert (
            f"{write['path']} already exists — merge this block into it manually:"
            in command
        ), editor_id
        assert "\nelse\n" in command, editor_id
        assert command.endswith("\nfi"), editor_id


def test_every_editor_write_guards_on_file_existence_powershell() -> None:
    writes = _writes("powershell")
    assert writes, "no editor file writes rendered"
    for editor_id, write in writes.items():
        command = write["command"]
        assert "if (Test-Path " in command, editor_id
        assert f"{write['path']}" in command, editor_id
        assert "Write-Host @'" in command, editor_id
        assert (
            f"{write['path']} already exists — merge this block into it manually:"
            in command
        ), editor_id
        assert "\n} else {\n" in command, editor_id
        assert command.endswith("\n}"), editor_id


def test_guard_prints_byte_identical_payload_to_write_branch() -> None:
    """The payload shown for manual merging must equal the payload a fresh
    write would produce — any drift and the merge instructions lie."""
    for platform in ("posix", "powershell"):
        for editor_id, write in _writes(platform).items():
            occurrences = write["command"].count(write["payload"])
            assert occurrences == 2, (
                f"{platform}/{editor_id}: payload must appear exactly twice "
                "(print branch + write branch)"
            )


def test_chunkhound_config_write_stays_unguarded() -> None:
    """.chunkhound.json is fully owned by the generator: re-runs rewrite it."""
    for platform, copy in _rendered()["compactCopies"].items():
        assert not re.search(r"if \[ -f [^\n]*\.chunkhound\.json", copy), platform
        assert not re.search(r"Test-Path [^\n]*\.chunkhound\.json", copy), platform
