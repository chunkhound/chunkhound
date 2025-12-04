You are generating a hypothetical, best-effort research plan for a software project.

Workspace commit context (if available):
- created_from_sha: {created}

In-scope project (relative to the workspace root):
- {scope_display}

Within this scope, the following files and directories currently exist (sampled and capped):
{files_block}

Sample code snippets from files (might be a subset, not full content, try to fill in gaps):
{code_context_block}

HyDE objective:
- Hallucinate a plausible picture of how this in-scope project is likely structured.
- Your primary deliverable is a list of focused research hooks (questions / topics)
  that a separate deep code research system will investigate using real code.
- Hooks should aim to cover the full breadth of the subsystem (core, providers,
  services, interfaces, CLI/API, MCP servers, operations, tests, etc.), even if your
  guesses are imperfect.

Output format:
- Produce a single markdown document with this structure:

  # HyDE Research Plan for {scope_display}

  ## Scope Summary
  Write 1–3 short sentences (not bullets) summarizing what this folder is *likely*
  responsible for in the wider project and which subdirectories or file clusters
  appear most important (e.g., core/, providers/, services/, interfaces/, api/,
  mcp_server.py, operations/, tests/).

  ## Global Hooks
  - Each bullet here should be a broad, high-level research question about the overall
    architecture, lifecycle, or invariants of this scope.
  - Examples (adapt and generalize them based on filenames you see):
    - Describe the overall architecture and lifecycle of this scope: main entrypoints,
      core services, and how data probably flows between them.
    - Explain how configuration is likely loaded and propagated through the system,
      including any critical constraints or environment-dependent behavior.
    - Map out how single-threaded constraints, batching, and index management might be
      enforced across storage and services.

  ## Subsystem Hooks
  - Group hooks by inferred subsystems (for example: "Core / Config / Models",
    "Providers / Storage", "Services", "Interfaces / API / CLI", "MCP Servers",
    "Operations", "Tests").
  - Under each group, add multiple bullet points.
    Each bullet must be a *single research question or investigation target*, not a full answer.
  - Phrase hooks as things to investigate, such as:
    - Identify the core configuration and data model modules and describe how they
      likely anchor the rest of the system.
    - Investigate how database providers and storage adapters are organized, and how
      they might expose higher-level services.
    - Explore how indexing, search, or deep-research services might fit together and
      where orchestration logic probably lives.
    - Examine how CLI and API entrypoints are structured and how they might wire into
      core services and providers.
    - Identify tests and operations docs that appear to encode important invariants,
      failure modes, or performance guardrails.

Guidelines:
- Make generous assumptions based on naming (for example, `core`, `config`,
  `services`, `providers`, `parsers`, `tests`) and on the inlined project content
- Prefer bullet lists, but you may use vivid, imaginative wording inside each bullet as long as the research question remains clear.
- Err on the side of being over-generative rather than conservative: for a non-trivial
  scope, produce a long, exploratory plan with many hooks and
  several distinct subsystem groups rather than a short outline.
- Every `- ` bullet under "Global Hooks" and "Subsystem Hooks" should be a standalone
  research question that would make sense as a deep code_research query.
