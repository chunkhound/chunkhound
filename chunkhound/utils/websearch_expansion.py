"""LLM-driven websearch query expansion.

Kept out of ``websearch_core.py`` so that module stays LLM-agnostic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from loguru import logger

from chunkhound.llm_manager import LLMManager
from chunkhound.services import prompts

_WEB_EXPANSION_TOKENS = 800
_NUM_WEB_QUERIES = 3

_QUOTE_RE = re.compile(r'"([^"]*)"')


@dataclass(frozen=True, slots=True)
class WebExpansionResult:
    """Return contract of :func:`expand_web_queries`.

    ``queries`` — exactly 3 DDG-optimized variants (pad/truncate enforced).
    ``search_query`` — LLM-emitted normalized form of the input; falls back to
    the raw ``query`` on any failure (empty / missing / non-string /
    whitespace-only, or when the LLM call itself fails).
    """

    queries: list[str]
    search_query: str


def _extract_quoted_phrases(q: str) -> list[str]:
    """Inner contents of double-quoted substrings (excluding the quotes)."""
    return _QUOTE_RE.findall(q)


def _preserves_quotes(proc: str, originals: list[str]) -> bool:
    """Whether ``proc`` keeps at least one of ``originals`` inside its own quotes.

    Per-variant predicate — the caller filters bad variants individually
    instead of discarding the whole batch.
    """
    proc_quotes = _extract_quoted_phrases(proc)
    if not proc_quotes:
        return False
    # Substring, not equality: a variant that quotes a tighter phrase (e.g.
    # "React 19.1" for the user's "React 19") still preserves the intent.
    return any(o in pq for o in originals for pq in proc_quotes)


def _build_context_lines(query: str, previous_query: str | None) -> str:
    if previous_query:
        return f'Query: "{query}"\nPrevious Query: "{previous_query}"'
    return f'Query: "{query}"'


def _build_instructions_block(previous_query: str | None) -> str:
    if previous_query:
        return prompts.WEBSEARCH_EXPANSION_INSTRUCTIONS_FOLLOWUP
    return prompts.WEBSEARCH_EXPANSION_INSTRUCTIONS_BASELINE


def _build_follow_up_examples(current_year: int, previous_query: str | None) -> str:
    if not previous_query:
        return ""
    # f-string interpolation runs now; the returned literal `{` / `}` chars
    # then pass through the outer `.format()` verbatim (substituted values are
    # not re-parsed as format slots).
    return f"""Example (with previous context) — React:
Previous: "React hooks best practices 2024"
Current: "React hooks performance optimization"
Analysis: "React hooks" overlaps, "performance optimization" is new
Output: {{
  "queries": [
    "React hooks performance optimization",
    "useMemo useCallback performance patterns {current_year}",
    "React rendering optimization strategies {current_year}"
  ],
  "search_query": "React hooks performance optimization"
}}

Example (with previous context) — Node:
Previous: "node js memory leak production"
Current: "chrome devtools heap snapshot nodejs"
Analysis: "nodejs" and "heap memory" overlap, "chrome devtools heap snapshot" is new
Output: {{
  "queries": [
    "chrome devtools heap snapshot nodejs",
    "chrome devtools heap profiling techniques {current_year}",
    "memory snapshot interpretation guide nodejs"
  ],
  "search_query": "chrome devtools heap snapshot nodejs"
}}

"""


async def expand_web_queries(
    query: str,
    llm_manager: LLMManager | None,
    previous_query: str | None = None,
) -> WebExpansionResult:
    """Generate 3 DuckDuckGo-optimized queries + a normalized ``search_query``.

    Contract:
    - Success: exactly 3 queries — the shape the LLM prompt asks for
      (primary + temporal + best-practices). LLM-produced when possible;
      padded from the raw ``query`` (year-tagged, then best-practices) when
      the LLM returns fewer than 3.
    - Individual variants that drop the query's quoted phrases are filtered
      out and the remainder is padded back up to 3 with the same
      quote-preserving fallbacks. When every variant drops quotes, the raw
      ``query`` seeds slot 0 and the pad fills slots 1-2 the same way.
    - ``search_query`` collapses to the raw ``query`` when the LLM emits
      anything non-useful (empty / missing / non-string / whitespace-only) or
      when the call itself fails.
    - When ``previous_query`` is set, the expansion prompt steers the LLM
      toward *new* dimensions relative to the prior topic. Empty string is
      coerced to ``None`` — no chaining, baseline prompt path.
    - Fallback to ``WebExpansionResult(queries=[query], search_query=query)``
      when ``llm_manager`` is ``None`` or the LLM call raises.
    """
    # Defensive re-coercion: surface entry points also coerce, but direct
    # in-process callers (tests, notebooks) bypass the boundary.
    previous_query = previous_query or None

    if llm_manager is None:
        return WebExpansionResult(queries=[query], search_query=query)

    schema = {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    f"Array of exactly {_NUM_WEB_QUERIES} "
                    "DuckDuckGo search queries"
                ),
            },
            "search_query": {
                "type": "string",
                "description": (
                    "The user query rewritten for DuckDuckGo per "
                    "<QUERY_NORMALIZATION>. Empty string permitted only if "
                    "no normalization applies."
                ),
            },
        },
        "required": ["queries", "search_query"],
        "additionalProperties": False,
    }

    current_year = datetime.now().year
    prompt = prompts.WEBSEARCH_EXPANSION_USER.format(
        current_year=current_year,
        context_lines=_build_context_lines(query, previous_query),
        instructions_block=_build_instructions_block(previous_query),
        follow_up_examples=_build_follow_up_examples(current_year, previous_query),
    )

    try:
        llm = llm_manager.get_utility_provider()
        # No explicit timeout — provider uses its default.
        result = await llm.complete_structured(
            prompt=prompt,
            json_schema=schema,
            system=prompts.WEBSEARCH_EXPANSION_SYSTEM,
            max_completion_tokens=_WEB_EXPANSION_TOKENS,
        )
        raw_queries = result.get("queries", []) or []
        raw_norm = result.get("search_query", "") or ""
    except Exception as e:
        logger.warning(
            f"Websearch query expansion failed: {e}, using original query only"
        )
        return WebExpansionResult(queries=[query], search_query=query)

    if not isinstance(raw_norm, str):
        raw_norm = ""
    raw_norm = raw_norm.strip()
    normalized = raw_norm or query

    queries = [q.strip() for q in raw_queries if isinstance(q, str) and q.strip()]
    if not queries:
        queries = [query]
    queries = queries[:_NUM_WEB_QUERIES]

    # Filter variants that dropped quoted phrases.
    # Runs before padding so the pad fallbacks — which interpolate the raw
    # ``query`` and thus preserve its quotes — naturally fill any gaps.
    originals = _extract_quoted_phrases(query)
    if originals:
        kept = [q for q in queries if _preserves_quotes(q, originals)]
        if not kept:
            logger.warning(
                "Websearch query expansion dropped quoted phrases in all "
                "variants; padding from raw query"
            )
            queries = [query]
        else:
            if len(kept) < len(queries):
                logger.debug(
                    f"Filtered {len(queries) - len(kept)} expansion variant(s) "
                    "that lost quoted phrases"
                )
            queries = kept

    # Pad to exactly 3. Pads interpolate the raw ``query``, which is what
    # lets the quote-preservation filter above rely on padding to fill
    # dropped variants without losing quoted phrases.
    while len(queries) < _NUM_WEB_QUERIES:
        if len(queries) == 1:
            queries.append(f"{query} {current_year}")
        else:
            queries.append(f"{query} best practices")

    logger.debug(f"Websearch expanded into {len(queries)} queries: {queries}")
    return WebExpansionResult(queries=queries, search_query=normalized)
