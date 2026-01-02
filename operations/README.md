Operations experiments and diagnostics
======================================

The ``operations`` directory contains ad-hoc experiments and operational
documentation that are **not** part of the main ChunkHound library API.

- ``database_concurrency.md`` – notes and conclusions from DuckDB/LanceDB
  concurrency probes, including the single-owner / SerialDatabaseProvider
  policy enforced by ChunkHound.
- ``experiments/duckdb_concurrency_probe.py`` – standalone script that
  demonstrates DuckDB lock failures under multi-process access.
- ``experiments/lancedb_concurrency_probe.py`` – standalone script that
  explores LanceDB behaviour under one-writer / many-reader scenarios.

These scripts are intended for local investigation and validation of
operational assumptions; they should not be imported by production code.


## Code Mapper (agent-facing docs)

Generate HyDE-planned, deep-research-based documentation for a scoped folder.

Requirements:
- An existing ChunkHound index for your workspace (run `chunkhound index` first).
- Embedding + LLM configured in `.chunkhound.json` (or via `CHUNKHOUND_*` env vars).

Basic usage (single project, auto config detection):

```bash
# From project root that already has .chunkhound.json and a ChunkHound DB
chunkhound map . --out .code_mapper
```

Explicit config override:

```bash
# Agent doc for current folder, using an explicit config file
chunkhound map . \
  --config .chunkhound.json \
  --out .code_mapper

# Plan-only: print HyDE points of interest, write HyDE prompt/plan to .code_mapper/
chunkhound map . \
  --config .chunkhound.json \
  --out .code_mapper \
  --plan
```

Workspace example (shared index across multiple projects):

```bash
# Workspace-level config and DB under /workspaces
chunkhound map arguseek \
  --config /workspaces/.chunkhound.json \
  --out /workspaces/arguseek/.code_mapper
```

Path resolution semantics:
- If you omit `--config` and `CHUNKHOUND_CONFIG_FILE`, ChunkHound detects the
  project root (where it finds `.chunkhound.json`) based on your current
  directory; the `path` positional argument is resolved relative to that root.
- If you pass `--config /path/to/.chunkhound.json`, the directory containing
  that file acts as the logical root for Code Mapper; `path` is interpreted
  relative to that root (e.g. `arguseek`, `arguseek/backend`).
- `chunkhound map .` always uses the current working directory when it is under
  the configured root; use an explicit relative path (e.g. `chunkhound`) to map
  a different folder.
- If you set `CHUNKHOUND_CONFIG_FILE` to a workspace-level config, Code Mapper
  treats that workspace directory as the root and resolves `path` relative to
  it, regardless of your current working directory.

Comprehensiveness:
- `--comprehensiveness {minimal,low,medium,high,ultra}` (default: `medium`)
  - Controls how many HyDE points of interest are planned (≈1/5/10/15/20).
  - Adjusts how much code is sampled for planning; file coverage uses the full index.
  - HyDE scope file list cap scales with comprehensiveness (≈200/500/2000/3000/5000).

Audience:
- `--audience {technical,balanced,end-user}` (default: `balanced`)
  - Controls how technical the generated map topics are.

Custom planning context:
- `--context /path/to/context.md`
  - Uses the file contents as the authoritative input to HyDE planning.
  - Fully replaces repo-derived HyDE context (file lists + sampled snippets) for both
    architectural and operational maps.

HyDE planning LLM configuration (optional):
- Code Mapper's HyDE planning (overview/PoI generation) can use a dedicated provider/model:
  - In `.chunkhound.json` under `llm`:
    - `map_hyde_provider` (e.g. `"codex-cli"` or `"openai"`)
    - `map_hyde_model`
    - `map_hyde_reasoning_effort` (`minimal|low|medium|high|xhigh`)
  - Or via environment:
    - `CHUNKHOUND_LLM_MAP_HYDE_PROVIDER`
    - `CHUNKHOUND_LLM_MAP_HYDE_MODEL`
    - `CHUNKHOUND_LLM_MAP_HYDE_REASONING_EFFORT`
- If none of these are set, Code Mapper falls back to the synthesis provider/model/effort.
- The effective HyDE planning provider/model/effort are recorded in
  `agent_doc_metadata.llm_config` as `map_hyde_provider`, `map_hyde_model`,
  and `map_hyde_reasoning_effort`.

Outputs:
- Combined document written under `--out` when `--combined` is set.
  If `--combined/--no-combined` is omitted, ChunkHound falls back to
  `CH_CODE_MAPPER_WRITE_COMBINED=1` for backward compatibility (disabled by default).
  Includes `agent_doc_metadata` header and coverage summary; stdout prints paths only.
- In `--overview-only` mode:
  - HyDE scope prompt + PoI plan written under `--out` (always).
- In full mode (default):
  - `<scope>_code_mapper_index.md` listing all topics.
  - One `<scope>_arch_topic_NN_<slug>.md` / `<scope>_ops_topic_NN_<slug>.md` per non-empty topic.


## AutoDoc (docsite generation)

Generate an Astro documentation site from existing AutoDoc / Code Mapper outputs.

Basic usage:

```bash
# Generate an Astro site under <map>/autodoc/
chunkhound autodoc /path/to/map_output_dir --out-dir /path/to/map_output_dir/autodoc
```

Generate maps automatically (interactive):

```bash
# If you omit map-in, AutoDoc can prompt to run Code Mapper first.
chunkhound autodoc --out-dir /path/to/output/autodoc
```

Optional overrides:

```bash
# Custom output directory
chunkhound autodoc /path/to/map_output_dir \
  --out-dir /path/to/output_dir/autodoc_site \
  --cleanup-mode llm

# Audience steering for cleanup/IA
chunkhound autodoc /path/to/map_output_dir \
  --out-dir /path/to/output_dir/autodoc_site \
  --audience end-user

# Override index file glob(s) (repeatable)
chunkhound autodoc /path/to/map_output_dir \
  --index-pattern "*_autodoc_index.md" \
  --index-pattern "*_code_mapper_index.md"

# If AutoDoc needs to auto-run Code Mapper, optionally provide steering context
chunkhound autodoc --out-dir /path/to/output/autodoc \
  --map-context /path/to/context.md
```

Notes:
- `--out-dir` is required; a common convention is `<map-in>/autodoc/`.
- AutoDoc accepts both `*_autodoc_index.md` and `*_code_mapper_index.md` by default.
- AutoDoc cleanup requires an LLM provider; use `--assets-only` to update UI assets
  without regenerating topic pages.
- If `--out-dir/src/pages/topics` already exists, AutoDoc will prompt before deleting
  it; non-interactive runs must pass `--force`.
- References are normalized from the original `## Sources` tree:
  flattened into a deterministic `## References` list with `[N]` and chunk ranges.
- If the provided `map-in` directory does not contain an index file, AutoDoc can
  prompt to run Code Mapper and retry.
  - It will prompt for the map output directory (default: `map_<out-dir-name>/`)
    unless you pass `--map-out-dir`.
  - It will prompt for Code Mapper comprehensiveness unless you pass
    `--map-comprehensiveness {minimal,low,medium,high,ultra}`.
