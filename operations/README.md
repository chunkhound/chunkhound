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
chunkhound code_mapper . --out-dir .code_mapper
```

Explicit config override:

```bash
# Agent doc for current folder, using an explicit config file
chunkhound code_mapper . \
  --config .chunkhound.json \
  --out-dir .code_mapper

# Plan-only: print HyDE points of interest, write HyDE prompt/plan to .code_mapper/
chunkhound code_mapper . \
  --config .chunkhound.json \
  --out-dir .code_mapper \
  --overview-only
```

Workspace example (shared index across multiple projects):

```bash
# Workspace-level config and DB under /workspaces
chunkhound code_mapper arguseek \
  --config /workspaces/.chunkhound.json \
  --out-dir /workspaces/arguseek/.code_mapper
```

Path resolution semantics:
- If you omit `--config` and `CHUNKHOUND_CONFIG_FILE`, ChunkHound detects the
  project root (where it finds `.chunkhound.json`) based on your current
  directory; the `path` positional argument is resolved relative to that root.
- If you pass `--config /path/to/.chunkhound.json`, the directory containing
  that file acts as the logical root for Code Mapper; `path` is interpreted
  relative to that root (e.g. `arguseek`, `arguseek/backend`).
- If you set `CHUNKHOUND_CONFIG_FILE` to a workspace-level config, Code Mapper
  treats that workspace directory as the root and resolves `path` relative to
  it, regardless of your current working directory.

Comprehensiveness:
- `--comprehensiveness {minimal,low,medium,high,ultra}` (default: `medium`)
  - Controls how many HyDE points of interest are planned (≈1/5/10/15/20).
  - Adjusts how much code is sampled for planning; file coverage uses the full index.
  - HyDE scope file list cap scales with comprehensiveness (≈200/500/2000/3000/5000).

Assembly LLM configuration (optional):
- Code Mapper's HyDE planning (overview/PoI generation) can use a dedicated “assembly” model:
  - In `.chunkhound.json` under `llm`:
    - `assembly_provider` (e.g. `"codex-cli"` or `"openai"`)
    - `assembly_model` (or legacy `assembly_synthesis_model`)
    - `assembly_reasoning_effort` (`minimal|low|medium|high|xhigh`)
  - Or via environment:
    - `CH_AGENT_DOC_ASSEMBLY_PROVIDER`
    - `CH_AGENT_DOC_ASSEMBLY_MODEL`
    - `CH_AGENT_DOC_ASSEMBLY_REASONING_EFFORT`
- If none of these are set, Code Mapper falls back to the synthesis provider/model,
  and those values are recorded as the effective assembly
  configuration.
- In all cases, the effective assembly provider/model/effort are recorded in
  `agent_doc_metadata.llm_config` as `assembly_synthesis_provider`,
  `assembly_synthesis_model`, and `assembly_reasoning_effort`.

Outputs:
- Combined document on stdout with an `agent_doc_metadata` header and coverage summary.
- In `--overview-only` mode:
  - HyDE scope prompt + PoI plan written under `--out-dir`.
- In full mode (default):
  - `<scope>_code_mapper_index.md` listing all topics.
  - One `<scope>_topic_NN_<slug>.md` file per non-empty topic.
