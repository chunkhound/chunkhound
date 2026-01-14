"""Fact Extractor: LLM-based extraction of atomic facts from code.

Extracts structured facts from cluster chunks using LLM calls,
then aggregates them into a unified EvidenceLedger.
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
            return result
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
    except ValueError:
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

    def _format_code_context(self, cluster_content: dict[str, str]) -> str:
        """Format cluster files for LLM prompt.

        Args:
            cluster_content: Mapping of file_path -> content

        Returns:
            Formatted string with all files
        """
        parts = []
        for file_path, content in sorted(cluster_content.items()):
            parts.append(f"### {file_path}\n```\n{content}\n```")
        return "\n\n".join(parts)

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

        code_context = self._format_code_context(cluster_content)
        system = FACT_EXTRACTION_SYSTEM.format(max_facts=max_facts)
        prompt = FACT_EXTRACTION_USER.format(
            root_query=root_query, code_context=code_context
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
                # Validate required fields
                statement = item.get("statement", "").strip()
                file_path = item.get("file_path", "").strip()

                if not statement or not file_path:
                    logger.debug(f"Skipping fact with missing required fields: {item}")
                    continue

                # Truncate statement if needed (defense in depth)
                if len(statement) > MAX_STATEMENT_CHARS:
                    statement = statement[: MAX_STATEMENT_CHARS - 3] + "..."

                start_line = int(item.get("start_line", 1))
                end_line = int(item.get("end_line", start_line))
                category = item.get("category", "general").strip()
                confidence = _parse_confidence(item.get("confidence", "uncertain"))
                entities = tuple(
                    e.strip() for e in item.get("entities", []) if e.strip()
                )

                fact_id = FactEntry.generate_id(
                    statement, file_path, start_line, end_line
                )

                fact = FactEntry(
                    fact_id=fact_id,
                    statement=statement,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    category=category,
                    confidence=confidence,
                    entities=entities,
                    cluster_id=cluster_id,
                )

                ledger.add_fact(fact)

            except (KeyError, TypeError, ValueError) as e:
                logger.debug(f"Skipping malformed fact entry: {e}")
                continue

        logger.info(
            f"Extracted {ledger.facts_count} facts from cluster {cluster_id} "
            f"({len(cluster_content)} files)"
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
            f"Extracted {merged.facts_count} total facts from {len(clusters)} clusters, "
            f"{len(merged.conflicts)} conflicts detected"
        )

        return merged
