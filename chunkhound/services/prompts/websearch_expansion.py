# ruff: noqa: E501
"""Websearch query expansion prompt (DuckDuckGo).

Emits exactly 3 web-optimized query variants plus a normalized ``search_query``
form. Two-branch template: baseline (single-shot) or follow-up (previous-query
aware). The follow-up-mode ``context_lines``/``instructions_block``/
``follow_up_examples`` slots are populated in Python by
``chunkhound/utils/websearch_expansion.py`` so this file stays free of
runtime conditionals.
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
Query: "Docker container orchestration"
Output: {{
  "queries": [
    "Docker container orchestration",
    "Docker container orchestration {current_year}",
    "container orchestration best practices"
  ],
  "search_query": "Docker container orchestration"
}}

Example 2 - Natural language question:
Query: "What are the best ways to optimize React performance?"
Output: {{
  "queries": [
    "React performance optimization techniques {current_year}",
    "React rendering optimization best practices",
    "React performance profiling tools guide"
  ],
  "search_query": "React performance optimization"
}}

Example 3 - Keyword dump:
Query: "python async await concurrency parallelism"
Output: {{
  "queries": [
    "python async await concurrency patterns",
    "python parallelism vs concurrency guide {current_year}",
    "asyncio concurrent programming best practices"
  ],
  "search_query": "python async await concurrency parallelism"
}}
</EXAMPLES>

<TEMPORAL_RULES>
- Technical queries: Add {current_year} for current year
- Historical queries: Preserve specific years
- Quoted phrases: Keep exactly as-is
</TEMPORAL_RULES>

Return JSON of the form:
{{"queries": ["<query1>", "<query2>", "<query3>"], "search_query": "<normalized form of Query, or empty string to defer>"}}"""


INSTRUCTIONS_BASELINE = """<INSTRUCTIONS>
1. First, produce the normalized query for DuckDuckGo (return in `search_query`):
   - If it's a natural language question, extract key search terms
   - If it's a keyword dump, organize into coherent phrase
   - Keep quoted phrases, technical terms and specific brands intact
2. Generate comprehensive search variations
3. Add temporal markers where appropriate

IMPORTANT: Always generate EXACTLY 3 queries - no more, no less.
</INSTRUCTIONS>"""


INSTRUCTIONS_FOLLOWUP = """<INSTRUCTIONS>
For all queries:
1. First, produce the normalized query for DuckDuckGo (return in `search_query`):
   - If it's a natural language question, extract key search terms
   - If it's a keyword dump, organize into coherent phrase
   - Keep quoted phrases, technical terms and specific brands intact

When previous_query exists:
2. Identify the domain/technology from previous query (e.g., "React hooks" → React domain)
3. Extract the NEW aspect from current query (e.g., "performance optimization")
4. Generate queries that combine: [domain context] + [new aspect]
5. AVOID repeating the exact previous query terms
6. Keep domain awareness but explore the new dimension

IMPORTANT: Always generate EXACTLY 3 queries - no more, no less.
</INSTRUCTIONS>"""
