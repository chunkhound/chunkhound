"""LLM-driven websearch query expansion.

Kept out of ``websearch_core.py`` so that module stays LLM-agnostic.
"""

from __future__ import annotations

import re
from datetime import datetime

from loguru import logger

from chunkhound.llm_manager import LLMManager
from chunkhound.services import prompts

_WEB_EXPANSION_TOKENS = 800
_NUM_WEB_QUERIES = 3

_QUOTE_RE = re.compile(r'"([^"]*)"')


def _extract_quoted_phrases(q: str) -> list[str]:
    """Inner contents of double-quoted substrings (excluding the quotes)."""
    return _QUOTE_RE.findall(q)


def _preserves_quotes(variant: str, user_quotes: list[str]) -> bool:
    """Whether ``variant`` keeps every phrase in ``user_quotes`` inside its own quotes.

    Per-variant predicate — the caller filters bad variants individually
    instead of discarding the whole batch. Partial preservation fails: a
    variant that drops any quoted phrase from a multi-quote query is
    rejected so padding can supply a quote-preserving replacement.
    """
    variant_quotes = _extract_quoted_phrases(variant)
    if not variant_quotes:
        return False
    # Substring, not equality: a variant that quotes a tighter phrase (e.g.
    # "React 19.1" for the user's "React 19") still preserves the intent.
    return all(any(uq in vq for vq in variant_quotes) for uq in user_quotes)


async def expand_web_queries(
    query: str,
    llm_manager: LLMManager | None,
) -> list[str]:
    """Generate 3 DuckDuckGo-optimized queries via LLM.

    Contract:
    - Success: exactly 3 queries — the shape the LLM prompt asks for
      (primary + temporal + best-practices). LLM-produced when possible;
      padded from the raw ``query`` (year-tagged, then best-practices) when
      the LLM returns fewer than 3.
    - Individual variants that drop the query's quoted phrases are filtered
      out and the remainder is padded back up to 3 with the same
      quote-preserving fallbacks. When every variant drops quotes, the raw
      ``query`` seeds slot 0 and the pad fills slots 1-2 the same way.
    - Fallback to ``[query]`` (length 1) only when ``llm_manager`` is None
      or the LLM call raises.
    """
    if llm_manager is None:
        return [query]

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
            }
        },
        "required": ["queries"],
        "additionalProperties": False,
    }

    current_year = datetime.now().year
    prompt = prompts.WEBSEARCH_EXPANSION_USER.format(
        query=query, current_year=current_year
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
    except Exception as e:
        logger.warning(
            f"Websearch query expansion failed: {e}, using original query only"
        )
        return [query]

    queries = [q.strip() for q in raw_queries if isinstance(q, str) and q.strip()]
    if not queries:
        queries = [query]
    queries = queries[:_NUM_WEB_QUERIES]

    # Filter variants that dropped quoted phrases.
    # Runs before padding so the pad fallbacks — which interpolate the raw
    # ``query`` and thus preserve its quotes — naturally fill any gaps.
    user_quotes = _extract_quoted_phrases(query)
    if user_quotes:
        kept = [q for q in queries if _preserves_quotes(q, user_quotes)]
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
    return queries
