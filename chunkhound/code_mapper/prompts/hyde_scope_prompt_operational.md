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
- Hallucinate a plausible operator/runbook style guide for this in-scope project.

Output format:
- Produce a single markdown document with plausible operational documentation

Guidelines:
- Prefer “how to run this end-to-end” content over API reference:
  - Quickstart / local run path
  - Configuration (env vars, config files) only when strongly suggested by names/content
  - Common workflows / recipes (3–6)
  - Troubleshooting / common failure modes
- Be explicit about uncertainty; don’t invent exact command flags or env var names unless the repo strongly suggests them

