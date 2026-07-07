"""Fact Extractor: LLM-based extraction of atomic facts from source material.

INTERNAL: Extracted facts feed into EvidenceLedger which an LLM synthesizer
reads during map-reduce research. Skipped facts (returning ``None`` from
``_parse_fact_item``) simply mean that evidence is absent from the LLM's
synthesis context, not user-facing data loss.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING

from loguru import logger

from .ledger import EvidenceLedger
from .models import ConfidenceLevel, FactEntry
from .prompts import FACT_EXTRACTION_SYSTEM, FACT_EXTRACTION_USER

if TYPE_CHECKING:
    from chunkhound.interfaces.llm_provider import LLMProvider

# Token budget for fact extraction responses
FACT_EXTRACTION_TOKENS = 8000

# Maximum characters for fact statements (enforced at extraction time)
MAX_STATEMENT_CHARS = 100

# Pattern to extract JSON from LLM response (may be wrapped in ```json ... ```)
_JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)

# Parse location strings like "lines 45-52", "lines 45", "L45-L52", or "L45"
_LOCATION_LINE_PATTERN = re.compile(
    r"(?:lines?\s*|(?<!\w)L\s*)(\d+)(?:\s*(?:-|to)\s*(?:L)?(\d+))?",
    re.IGNORECASE,
)

# Parse bare line ranges like "45-52" or "45" (no prefix)
_BARE_LINE_RANGE_PATTERN = re.compile(
    r"^(\d+)(?:\s*(?:-|to)\s*(\d+))?$",
    re.IGNORECASE,
)


def _extract_json_array(response: str) -> list[dict]:
    """Extract JSON array from LLM response.

    Handles responses wrapped in markdown code blocks.

    Args:
        response: Raw LLM response text

    Returns:
        Parsed JSON array, or empty list on failure
    """
    text = response.strip()

    # Try to extract from code block first
    match = _JSON_BLOCK_PATTERN.search(text)
    if match:
        text = match.group(1).strip()

    # If response starts with [ directly, use as-is
    if not text.startswith("["):
        # Try to find array in text
        bracket_start = text.find("[")
        bracket_end = text.rfind("]")
        if bracket_start != -1 and bracket_end > bracket_start:
            text = text[bracket_start : bracket_end + 1]

    try:
        result = json.loads(text)
        if isinstance(result, list):
            # LLM responses occasionally include bare strings/numbers instead of dicts
            dicts = [item for item in result if isinstance(item, dict)]
            if len(dicts) < len(result):
                logger.debug(
                    "Filtering "
                    f"{len(result) - len(dicts)} non-dict items from JSON array"
                )
            return dicts
        return []
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON from LLM response: {e}")
        return []


def _parse_confidence(value: str) -> ConfidenceLevel:
    """Parse confidence level from string.

    Args:
        value: String confidence value

    Returns:
        ConfidenceLevel enum, defaults to UNCERTAIN on parse failure
    """
    try:
        return ConfidenceLevel(value.lower().strip())
    except (ValueError, AttributeError):
        logger.debug(f"Unknown confidence level '{value}', defaulting to UNCERTAIN")
        return ConfidenceLevel.UNCERTAIN


class FactExtractor:
    """LLM-based extraction of atomic facts from cluster chunks."""

    def __init__(self, llm_provider: LLMProvider):
        """Initialize fact extractor.

        Args:
            llm_provider: LLM provider for fact extraction calls
        """
        self._llm = llm_provider

    def _format_source_context(self, cluster_content: dict[str, str]) -> str:
        """Format cluster sources for LLM prompt.

        Args:
            cluster_content: Mapping of file_path -> content

        Returns:
            Formatted string with all sources
        """
        parts = []
        for file_path, content in sorted(cluster_content.items()):
            parts.append(f"### {file_path}\n```\n{content}\n```")
        return "\n\n".join(parts)

    @staticmethod
    def _parse_fact_item(
        item: dict,
        cluster_content: dict[str, str],
    ) -> (
        tuple[
            str,  # statement
            str,  # file_path
            int,  # start_line
            int,  # end_line
            str | None,  # url
            str | None,  # source_section
            str,  # category
            ConfidenceLevel,  # confidence
            tuple[str, ...],  # entities
            str,  # fact_id
        ]
        | None
    ):
        """Parse a single fact item from the LLM response.

        Handles both new-style (source + location) and old-style
        (file_path + start_line + end_line) formats. Returns None
        for items that should be skipped (empty statement, no source).

        Args:
            item: Raw fact dict from LLM response
            cluster_content: Mapping of file_path -> content for scope resolution

        Returns:
            Tuple of parsed fields, or None if item should be skipped
        """
        # Validate required fields
        statement = item.get("statement", "").strip()
        if not statement:
            logger.debug(f"Skipping fact with empty statement: {item}")
            return None

        # Truncate statement if needed (defense in depth)
        if len(statement) > MAX_STATEMENT_CHARS:
            statement = statement[: MAX_STATEMENT_CHARS - 3] + "..."

        # Parse source: accept both new-style (source + location)
        # and old-style (file_path + start_line + end_line)
        source = (item.get("source") or "").strip() or (
            item.get("file_path") or ""
        ).strip()
        location_raw = (item.get("location") or "").strip()
        url = (item.get("url") or "").strip() or None
        has_new_style_fields = any(key in item for key in ("source", "location", "url"))
        has_old_style_fields = any(
            key in item for key in ("file_path", "start_line", "end_line")
        )

        # Keep prompt and parser aligned: canonical web provenance lives in
        # `url`, while `file_path` should remain a stable cluster lookup key.
        normalized_source_url = False
        if (
            source.lower().startswith(("http://", "https://"))
            and source not in cluster_content
        ):
            if not url:
                logger.debug(f"Normalizing source URL into url field: {source[:80]}")
                url = source
            normalized_source_url = True

        if normalized_source_url:
            if len(cluster_content) != 1:
                logger.warning(
                    "Skipping URL-backed fact with ambiguous multi-source cluster "
                    f"provenance: {source[:80]}"
                )
                return None
            source = next(iter(cluster_content))
        elif not source and url:
            if len(cluster_content) != 1:
                logger.warning(
                    "Skipping URL-only fact with ambiguous multi-source "
                    f"cluster provenance \u2014 url: {url[:80]}"
                )
                return None
            source = next(iter(cluster_content))

        if not source and not url:
            logger.debug(f"Skipping fact with no source or URL: {item}")
            return None

        # Parse location into line numbers or section
        start_line = 0
        end_line = 0
        source_section: str | None = None

        if location_raw:
            line_match = _LOCATION_LINE_PATTERN.search(location_raw)
            if line_match:
                start_line = int(line_match.group(1))
                end_line = int(line_match.group(2) or line_match.group(1))
            elif bare_match := _BARE_LINE_RANGE_PATTERN.match(location_raw):
                # Fallback for bare line ranges like "45-52" or "45"
                start_line = int(bare_match.group(1))
                end_line = int(bare_match.group(2) or bare_match.group(1))
            else:
                source_section = location_raw
        elif has_old_style_fields:
            logger.debug(
                f"No location parsed — using old-style line numbers for "
                f"fact: {item.get('statement', '')[:50]}"
            )
            start_line = int(item.get("start_line", 1))
            end_line = int(item.get("end_line", start_line))
        elif has_new_style_fields:
            statement_preview = item.get("statement", "")[:50]
            logger.warning(
                f"Skipping fact with missing location in new-style payload: "
                f"{statement_preview}"
            )
            return None

        file_path = source

        category = (item.get("category") or "general").strip()
        confidence = _parse_confidence(item.get("confidence") or "uncertain")
        # Safely handle string-valued entities (iterating a string yields chars)
        entities_raw = item.get("entities", [])
        if isinstance(entities_raw, str):
            entities_raw = [entities_raw]
        entities = tuple(e.strip() for e in entities_raw if e.strip())

        fact_id = FactEntry.generate_id(
            statement,
            primary_source=(url or source),
            start_line=start_line,
            end_line=end_line,
            section=source_section,
        )

        return (
            statement,
            file_path,
            start_line,
            end_line,
            url,
            source_section,
            category,
            confidence,
            entities,
            fact_id,
        )

    async def extract_from_cluster(
        self,
        cluster_id: int,
        cluster_content: dict[str, str],
        root_query: str,
        max_facts: int = 30,
    ) -> EvidenceLedger:
        """Extract facts from a single cluster via LLM call.

        Args:
            cluster_id: Identifier for this cluster
            cluster_content: Mapping of file_path -> content
            root_query: Research query for context
            max_facts: Maximum facts to extract

        Returns:
            EvidenceLedger containing extracted facts
        """
        ledger = EvidenceLedger()

        if not cluster_content:
            return ledger

        source_context = self._format_source_context(cluster_content)
        system = FACT_EXTRACTION_SYSTEM.format(max_facts=max_facts)
        prompt = FACT_EXTRACTION_USER.format(
            root_query=root_query, source_context=source_context
        )

        try:
            response = await self._llm.complete(
                prompt,
                system=system,
                max_completion_tokens=FACT_EXTRACTION_TOKENS,
            )
        except Exception as e:
            logger.warning(f"LLM call failed for cluster {cluster_id}: {e}")
            return ledger

        facts_data = _extract_json_array(response.content)

        for item in facts_data:
            try:
                parsed = self._parse_fact_item(item, cluster_content)
                if parsed is None:
                    continue

                (
                    statement,
                    file_path,
                    start_line,
                    end_line,
                    url,
                    source_section,
                    category,
                    confidence,
                    entities,
                    fact_id,
                ) = parsed

                fact = FactEntry(
                    fact_id=fact_id,
                    statement=statement,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    url=url,
                    source_section=source_section,
                    category=category,
                    confidence=confidence,
                    entities=entities,
                    cluster_id=cluster_id,
                )

                ledger.add_fact(fact)

            except (KeyError, TypeError, ValueError, AttributeError) as e:
                # Belt-and-suspenders: _extract_json_array already filters
                # non-dicts, but callers may supply raw parsed JSON directly.
                # AttributeError guards against null/number values from LLM JSON
                # where .strip() on None/int would be called.
                logger.warning(f"Skipping malformed fact entry: {e}")
                continue

        logger.info(
            f"Extracted {ledger.facts_count} facts from cluster {cluster_id} "
            f"({len(cluster_content)} sources)"
        )

        return ledger

    async def extract_from_clusters(
        self,
        clusters: list[tuple[int, dict[str, str], int]],
        root_query: str,
        max_concurrency: int = 4,
    ) -> EvidenceLedger:
        """Extract from all clusters in parallel, merge into unified ledger.

        Args:
            clusters: List of (cluster_id, {file_path: content}, max_facts) tuples
            root_query: Research query for context
            max_concurrency: Maximum parallel LLM calls

        Returns:
            Merged EvidenceLedger with all extracted facts
        """
        if not clusters:
            return EvidenceLedger()

        semaphore = asyncio.Semaphore(max_concurrency)

        async def extract_with_limit(
            cluster_id: int, content: dict[str, str], max_facts: int
        ) -> EvidenceLedger:
            async with semaphore:
                return await self.extract_from_cluster(
                    cluster_id, content, root_query, max_facts
                )

        tasks = [
            extract_with_limit(cid, content, max_facts)
            for cid, content, max_facts in clusters
        ]
        ledgers = await asyncio.gather(*tasks)

        # Merge all ledgers
        merged = EvidenceLedger()
        for ledger in ledgers:
            merged = merged.merge(ledger)

        # Detect conflicts in merged result
        conflicts = merged.detect_conflicts()
        merged.conflicts.extend(conflicts)

        logger.info(
            f"Extracted {merged.facts_count} total facts from "
            f"{len(clusters)} clusters, {len(merged.conflicts)} conflicts detected"
        )

        return merged
