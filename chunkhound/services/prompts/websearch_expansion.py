# ruff: noqa: E501
"""Websearch query expansion prompt (DuckDuckGo).

Emits exactly 3 web-optimized query variants. Two-branch template: baseline
(single-shot) or follow-up (previous-query aware). All prompt text lives here
as plain-string constants; the builder in
``chunkhound/utils/websearch_expansion.py`` selects between the variants and
pre-resolves ``{current_year}`` inside the follow-up examples so that value
arrives at the outer ``USER_TEMPLATE.format(...)`` fully substituted — the
outer format is single-pass and won't re-scan slot values. Mirrors the
``build_follow_up_section`` pattern in
``chunkhound/services/prompts/synthesis.py``.
"""

SYSTEM_MESSAGE = "Rewrite the user query into effective DuckDuckGo search queries."

USER_TEMPLATE = """You are a search query optimizer that generates strategic search queries for DuckDuckGo.

TASK: When given a current query and optional previous query, generate search queries that explore new aspects while avoiding previously explored concepts.

<CURRENT_CONTEXT>
Current Year: {current_year}
{context_lines}
</CURRENT_CONTEXT>

<QUERY_NORMALIZATION>
Transform the input into effective DuckDuckGo search queries by:
- Converting questions to search terms (e.g., "What is X?" → "X explanation guide")
- Organizing keyword dumps into coherent searches
- Removing unnecessary words (how, what, when, etc.) unless essential
- Preserving technical terms, specific models, brands or products, and quoted phrases exactly
</QUERY_NORMALIZATION>

{instructions_block}

<EXAMPLES>
{follow_up_examples}Example 1 - Docker container orchestration:
<query>Docker container orchestration</query>
Output: {{
  "queries": [
    "Docker container orchestration",
    "Docker container orchestration {current_year}",
    "container orchestration best practices"
  ]
}}

Example 2 - Natural language question:
<query>What are the best ways to optimize React performance?</query>
Output: {{
  "queries": [
    "React performance optimization techniques {current_year}",
    "React rendering optimization best practices",
    "React performance profiling tools guide"
  ]
}}

Example 3 - Keyword dump:
<query>python async await concurrency parallelism</query>
Output: {{
  "queries": [
    "python async await concurrency patterns",
    "python parallelism vs concurrency guide {current_year}",
    "asyncio concurrent programming best practices"
  ]
}}
</EXAMPLES>

<TEMPORAL_RULES>
- Technical queries: Add {current_year} for current year
- Historical queries: Preserve specific years
- Quoted phrases: Keep exactly as-is
</TEMPORAL_RULES>

Return JSON of the form:
{{"queries": ["<query1>", "<query2>", "<query3>"]}}"""


INSTRUCTIONS_BASELINE = """<INSTRUCTIONS>
1. Generate comprehensive search variations following <QUERY_NORMALIZATION>
2. Add temporal markers where appropriate

IMPORTANT: Always generate EXACTLY 3 queries - no more, no less.
</INSTRUCTIONS>"""


INSTRUCTIONS_FOLLOWUP = """<INSTRUCTIONS>
1. Identify the domain/technology from the previous query (e.g., "React hooks" → React domain)
2. Extract the NEW aspect from the current query (e.g., "performance optimization")
3. Generate queries that combine: [domain context] + [new aspect], following <QUERY_NORMALIZATION>
4. AVOID repeating the exact previous query terms
5. Keep domain awareness but explore the new dimension

IMPORTANT: Always generate EXACTLY 3 queries - no more, no less.
</INSTRUCTIONS>"""


# Prepended to `<EXAMPLES>` in follow-up mode; rendered once via `.format()`
# in the utils builder, then substituted verbatim into the outer template.
FOLLOWUP_EXAMPLES = """Example (with previous context) — React:
<previous_query>React hooks best practices 2024</previous_query>
<query>React hooks performance optimization</query>
Analysis: "React hooks" overlaps, "performance optimization" is new
Output: {{
  "queries": [
    "React hooks performance optimization",
    "useMemo useCallback performance patterns {current_year}",
    "React rendering optimization strategies {current_year}"
  ]
}}

Example (with previous context) — Node:
<previous_query>node js memory leak production</previous_query>
<query>chrome devtools heap snapshot nodejs</query>
Analysis: "nodejs" and "heap memory" overlap, "chrome devtools heap snapshot" is new
Output: {{
  "queries": [
    "chrome devtools heap snapshot nodejs",
    "chrome devtools heap profiling techniques {current_year}",
    "memory snapshot interpretation guide nodejs"
  ]
}}

"""
