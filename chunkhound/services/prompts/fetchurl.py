"""Fetch-URL prompts for the fetchurl feature.

Extracts a faithful Markdown answer (focused) or summary (generic) from a
single fetched web source. Used by the truncate (token-truncate) and
chunk-rerank (chunk + rerank + elbow) dispatch paths.
"""

# Terse system prompt per ChunkHound convention (see query_expansion.py,
# followup_generation.py; question_synthesis.py is a variant).
# Detailed rules, content framing, and output shape live in the user template.
SYSTEM_MESSAGE = "Extract a faithful Markdown answer from a single web source."


FOCUSED_USER_TEMPLATE = """TASK: Answer the user's question using only the content of one fetched webpage.

<source>
<url>{url}</url>
<title>{title}</title>
<query>{query}</query>
<content>
{content}
</content>
</source>

REQUIREMENTS:
1. Ground every claim in <content>. Do not add information from outside knowledge.
2. Preserve technical facts, numbers, code, identifiers, and quoted terms exactly as they appear.
3. Prefer short direct quotes when precision matters (e.g., configuration values, API names, error strings).
4. If <content> does not contain enough information to answer the query, respond with exactly:
   "The source at {url} (title: {title}) does not contain information about: {query}"
   Then briefly note what related information IS available, if any.
5. Use ATX headings (##, ###) only when the source has clear sections worth mirroring.
6. When quoting or paraphrasing specific spans, you may reference the `[L<start>-<end>]` (source-line span, Markdown) or `[P<page>]` (page number, PDF) markers that precede each chunk to help a reader locate the passage. Do not fabricate markers that were not shown, and do not invent a marker form that was not present on the chunk you are quoting (e.g. do not synthesize `[L...]` for a PDF chunk that carried `[P...]`).

Answer directly — no preamble, no restatement of the query."""


GENERIC_USER_TEMPLATE = """TASK: Summarize the fetched webpage faithfully.

<source>
<url>{url}</url>
<title>{title}</title>
<content>
{content}
</content>
</source>

REQUIREMENTS:
1. Ground every claim in <content>. Do not add information from outside knowledge.
2. Preserve technical facts, numbers, code, identifiers, and quoted terms exactly as they appear.
3. Structure with ATX headings (##, ###) only when the source has clear sections worth mirroring.
4. Prioritize what a reader would need to act on this page — key claims, values, and named entities first.

Produce the summary directly — no preamble."""
