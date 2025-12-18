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
- Hallucinate a plausible deep-wiki style documentation of this in-scope project.

Output format:
- Produce a single markdown document with the plausible documentation


Guidelines:
- Make generous assumptions based on naming (for example, `core`, `config`,
  `services`, `providers`, `parsers`, `tests`) and on the inlined project content
- You may use vivid, imaginative wording
- Err on the side of being over-generative rather than conservative: for a non-trivial
  scope, produce a long, exploratory documentation with many branches

