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


# JSON schema example values in the "Return JSON of the form:" line, not
# structural delimiters — excluded from the strip list.
_PLACEHOLDER_TAGS = frozenset({"<query1>", "<query2>", "<query3>"})

# Every `<name>` literal in the templates, stripped from user input so a
# crafted query can't terminate a delimiter and impersonate a new section.
# Regex requires the full `<name>` shape so `std::vector<int>` survives.
_PROMPT_TAG_LITERALS: tuple[str, ...] = tuple(
    sorted(
        set(
            re.findall(
                r"</?[A-Za-z_][A-Za-z0-9_]*>",
                prompts.WEBSEARCH_EXPANSION_USER
                + prompts.WEBSEARCH_EXPANSION_INSTRUCTIONS_BASELINE
                + prompts.WEBSEARCH_EXPANSION_INSTRUCTIONS_FOLLOWUP
                + prompts.WEBSEARCH_EXPANSION_FOLLOWUP_EXAMPLES,
            )
        )
        - _PLACEHOLDER_TAGS,
        key=len,
        reverse=True,
    )
)


def _strip_prompt_tags(s: str) -> str:
    # Fixpoint loop: `str.replace` is single-pass and does not rescan its
    # output, so a composed input like `<qu<query>ery>` would otherwise
    # collapse into a fresh `<query>` that survives the sanitizer.
    stripped: set[str] = set()
    while True:
        new = s
        for literal in _PROMPT_TAG_LITERALS:
            after = new.replace(literal, "")
            if after != new:
                stripped.add(literal)
                new = after
        if new == s:
            if stripped:
                logger.debug(
                    "Websearch expansion stripped prompt tag literals from user "
                    f"input: {sorted(stripped)}"
                )
            return s
        s = new


def _build_context_lines(query: str, previous_query: str | None) -> str:
    # Tag delimiters (not quotes) so inputs containing double quotes don't
    # produce malformed `""..."" ` sequences the LLM has to disambiguate.
    # Strip tag-name literals from the body so user text can't terminate
    # the delimiter and inject fake structure.
    query = _strip_prompt_tags(query)
    if previous_query:
        previous_query = _strip_prompt_tags(previous_query)
        return (
            f"<query>{query}</query>\n"
            f"<previous_query>{previous_query}</previous_query>"
        )
    return f"<query>{query}</query>"


def _build_instructions_block(previous_query: str | None) -> str:
    if previous_query:
        return prompts.WEBSEARCH_EXPANSION_INSTRUCTIONS_FOLLOWUP
    return prompts.WEBSEARCH_EXPANSION_INSTRUCTIONS_BASELINE


def _build_follow_up_examples(current_year: int, previous_query: str | None) -> str:
    if not previous_query:
        return ""
    return prompts.WEBSEARCH_EXPANSION_FOLLOWUP_EXAMPLES.format(
        current_year=current_year
    )


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
    - Structural hardening only, not full injection defense: a fixed list of
      the prompt template's own tag literals (e.g. ``</query>``,
      ``</CURRENT_CONTEXT>``) is stripped from ``query`` and ``previous_query``
      before rendering (logged at DEBUG when any literal is removed), so a
      crafted input can't terminate a delimiter and pose as a new section. Any
      other adversarial content in the body still reaches the LLM as-is —
      callers must not treat the returned queries as trusted.
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
    return WebExpansionResult(queries=queries, search_query=normalized)
