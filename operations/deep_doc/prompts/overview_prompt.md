Use the code_research tool to generate provenance-based, highly grounded documentation
for the in-scope folder: {scope_display}

HyDE-style synthesis backbone for this run:
- First, imagine an ideal, well-structured documentation set for this scope
  using the HyDE planning block below.
- Then, re-derive all explanations from the actual code, tests, configuration,
  and docs instead of copying text from the HyDE output.

Commit context:
- Created-from SHA: {created}
- Previous agent-doc SHA: {previous}
- Target SHA: {target}

Change summary since created-from:
{diff_created_block}

Change summary since previous agent doc:
{diff_previous_block}

=== HYDE BLOCK ===
{plan_block}
=== END HYDE BLOCK ===

Goals:
1. Rewrite the possibly highly halucinated HyDe documentation into a semantic
   and research-driven document grounded in the repository state.
2. Explain major subsystems and how they fit together: how configuration, data
   models, services, and entrypoints (CLI/API/tests) interact.
3. Highlight any critical constraints and invariants you discover in configs,
   architecture notes, or error-handling code (for example, single-threaded
   access policies, batching requirements, or index management rules).
4. Include practical debugging patterns and "gotchas" that show up in tests,
   operations docs, and failure paths.
5. Use a chain-of-thought style inside the document itself: for each major
   subsystem and design decision, spell out the reasoning, tradeoffs, and
   failure modes in small, explicit steps instead of only stating conclusions.
6. Use tables and flow charts at will whenever they fit the picture.

Output format: Markdown

Guidelines:
- Aim for a long, detailed document rather than a brief overview. It is
  acceptable (and preferred) to spend many paragraphs on each subsystem as long
  as the structure stays clear.
- Do NOT include any commit metadata comments; the caller will prepend them.
  Only output the markdown content starting from "# Agent Doc".
