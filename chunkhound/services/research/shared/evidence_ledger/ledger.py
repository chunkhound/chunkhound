"""Evidence Ledger: Unified aggregation of constants and facts for research context.

Combines ConstantsLedger and FactsLedger into a single class for:
1. Providing verified evidence to LLM during synthesis
2. Constants extraction from chunk metadata
3. Entity-based cross-referencing between facts
4. Conflict detection for reconciliation
5. Unified report appendix generation
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .models import ConfidenceLevel, ConstantEntry, EntityLink, FactConflict, FactEntry
from .prompts import (
    CONSTANTS_INSTRUCTION_FULL,
    CONSTANTS_INSTRUCTION_SHORT,
    FACTS_MAP_INSTRUCTION,
    FACTS_REDUCE_INSTRUCTION,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


# Negation patterns that suggest conflicting facts
_NEGATION_PATTERNS = re.compile(
    r"\b(never|always|cannot|must not|does not|is not|are not|"
    r"no longer|impossible|forbidden|prohibited)\b",
    re.IGNORECASE,
)

# Numeric value pattern for conflict detection
_NUMERIC_PATTERN = re.compile(r"\b(\d+(?:\.\d+)?)\b")

# Maximum facts before truncation (~30 tokens per fact = 15k token cap)
MAX_FACTS_LIMIT = 500


def _normalize_entity_name(name: str) -> str:
    """Normalize entity names for consistent linking.

    Args:
        name: Raw entity name from fact extraction

    Returns:
        Normalized entity name (lowercase, stripped)
    """
    return name.strip().lower()


@dataclass
class EvidenceLedger:
    """Unified evidence ledger combining constants and facts."""

    constants: dict[str, ConstantEntry] = field(default_factory=dict)
    # Key: "file_path:name" for uniqueness

    facts: dict[str, FactEntry] = field(default_factory=dict)
    # Key: fact_id -> FactEntry

    entity_index: dict[str, EntityLink] = field(default_factory=dict)
    # Key: normalized entity_name -> EntityLink (for facts only)

    conflicts: list[FactConflict] = field(default_factory=list)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def constants_count(self) -> int:
        """Number of constants in the ledger."""
        return len(self.constants)

    @property
    def facts_count(self) -> int:
        """Number of facts in the ledger."""
        return len(self.facts)

    def __len__(self) -> int:
        """Total evidence count (constants + facts)."""
        return self.constants_count + self.facts_count

    # =========================================================================
    # Construction
    # =========================================================================

    @classmethod
    def from_chunks(cls, chunks: Iterable[dict]) -> EvidenceLedger:
        """Build ledger from chunk metadata.

        Extracts constants from chunk metadata.constants field.

        Args:
            chunks: Iterable of chunk dicts with optional 'metadata.constants' field

        Returns:
            New EvidenceLedger with extracted constants
        """
        ledger = cls()
        for chunk in chunks:
            file_path = chunk.get("file_path", "")
            metadata = chunk.get("metadata") or {}
            chunk_constants = metadata.get("constants") or []

            for const in chunk_constants:
                name = const.get("name")
                if not name:
                    continue
                entry = ConstantEntry(
                    name=name,
                    file_path=file_path,
                    value=const.get("value"),
                    type=const.get("type"),
                )
                ledger.add_constant(entry)
        return ledger

    # =========================================================================
    # Mutation
    # =========================================================================

    def add_constant(self, entry: ConstantEntry) -> None:
        """Add constant with dedup by "file_path:name".

        Args:
            entry: ConstantEntry to add
        """
        key = f"{entry.file_path}:{entry.name}"
        if key not in self.constants:
            self.constants[key] = entry

    def add_fact(self, fact: FactEntry) -> None:
        """Add fact and update entity index.

        Args:
            fact: FactEntry to add to the ledger
        """
        self.facts[fact.fact_id] = fact

        # Update entity index
        for entity in fact.entities:
            normalized = _normalize_entity_name(entity)
            if normalized in self.entity_index:
                existing = self.entity_index[normalized]
                if fact.fact_id not in existing.fact_ids:
                    self.entity_index[normalized] = EntityLink(
                        entity_name=existing.entity_name,
                        fact_ids=(*existing.fact_ids, fact.fact_id),
                    )
            else:
                self.entity_index[normalized] = EntityLink(
                    entity_name=entity,  # Keep original case
                    fact_ids=(fact.fact_id,),
                )

    def merge(self, other: EvidenceLedger) -> EvidenceLedger:
        """Merge another ledger into this one (immutable).

        Args:
            other: Another EvidenceLedger to merge

        Returns:
            New EvidenceLedger with entries from both
        """
        merged = EvidenceLedger()

        # Merge constants
        merged.constants = dict(self.constants)
        merged.constants.update(other.constants)

        # Merge facts (add_fact handles entity index)
        for fact in self.facts.values():
            merged.add_fact(fact)
        for fact in other.facts.values():
            merged.add_fact(fact)

        # Merge conflicts
        merged.conflicts = list(self.conflicts) + list(other.conflicts)

        return merged

    def replace_constants_from_chunks(self, chunks: Iterable[dict]) -> EvidenceLedger:
        """Replace constants with those from new chunk set, preserving facts.

        Use when chunk set may have had items removed. Constants are rebuilt
        from current chunks while accumulated facts are preserved.

        Args:
            chunks: New chunk set to extract constants from

        Returns:
            New EvidenceLedger with fresh constants and preserved facts
        """
        # Build fresh constants from current chunks
        fresh = EvidenceLedger.from_chunks(chunks)

        # Preserve facts (expensive LLM extractions)
        for fact in self.facts.values():
            fresh.add_fact(fact)

        # Preserve conflicts
        fresh.conflicts = list(self.conflicts)

        return fresh

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_facts_for_files(self, file_paths: set[str]) -> list[FactEntry]:
        """Get facts from specific files.

        Args:
            file_paths: Set of file paths to filter by

        Returns:
            List of FactEntry objects from those files
        """
        return [f for f in self.facts.values() if f.file_path in file_paths]

    def get_facts_for_entity(self, entity: str) -> list[FactEntry]:
        """Get facts referencing an entity.

        Args:
            entity: Entity name to look up (case-insensitive)

        Returns:
            List of FactEntry objects referencing that entity
        """
        normalized = _normalize_entity_name(entity)
        link = self.entity_index.get(normalized)
        if not link:
            return []
        return [self.facts[fid] for fid in link.fact_ids if fid in self.facts]

    def get_related_facts(self, file_paths: set[str]) -> list[FactEntry]:
        """Get facts related via entity links.

        Finds facts from files, then expands to include facts
        about entities mentioned in those facts.

        Args:
            file_paths: Set of file paths to start from

        Returns:
            List of related FactEntry objects
        """
        # Get direct facts from files
        direct_facts = self.get_facts_for_files(file_paths)

        # Collect all entities mentioned
        entities: set[str] = set()
        for fact in direct_facts:
            entities.update(fact.entities)

        # Get facts for all related entities
        related_ids: set[str] = set()
        for entity in entities:
            for fact in self.get_facts_for_entity(entity):
                related_ids.add(fact.fact_id)

        # Add direct fact IDs
        for fact in direct_facts:
            related_ids.add(fact.fact_id)

        return [self.facts[fid] for fid in related_ids if fid in self.facts]

    # =========================================================================
    # Conflict Detection
    # =========================================================================

    def detect_conflicts(self) -> list[FactConflict]:
        """Heuristic conflict detection for facts about same entities.

        Uses simple pattern matching:
        - Negation words in facts about same entity
        - Different numeric values for same entity

        Returns:
            List of detected FactConflict objects
        """
        conflicts: list[FactConflict] = []

        for entity_name, link in self.entity_index.items():
            if len(link.fact_ids) < 2:
                continue

            facts_list = [
                self.facts[fid] for fid in link.fact_ids if fid in self.facts
            ]
            if len(facts_list) < 2:
                continue

            # Check all pairs
            for i, fact_a in enumerate(facts_list):
                for fact_b in facts_list[i + 1 :]:
                    conflict = self._check_conflict(fact_a, fact_b, entity_name)
                    if conflict:
                        conflicts.append(conflict)

        return conflicts

    def _check_conflict(
        self, fact_a: FactEntry, fact_b: FactEntry, entity_name: str
    ) -> FactConflict | None:
        """Check if two facts conflict.

        Args:
            fact_a: First fact
            fact_b: Second fact
            entity_name: Shared entity name

        Returns:
            FactConflict if conflict detected, None otherwise
        """
        # Check negation patterns
        a_has_negation = bool(_NEGATION_PATTERNS.search(fact_a.statement))
        b_has_negation = bool(_NEGATION_PATTERNS.search(fact_b.statement))

        if a_has_negation != b_has_negation:
            return FactConflict(
                fact_id_a=fact_a.fact_id,
                fact_id_b=fact_b.fact_id,
                reason=f"Potential negation conflict for entity '{entity_name}'",
            )

        # Check numeric value differences
        a_nums = set(_NUMERIC_PATTERN.findall(fact_a.statement))
        b_nums = set(_NUMERIC_PATTERN.findall(fact_b.statement))

        if a_nums and b_nums and a_nums != b_nums:
            return FactConflict(
                fact_id_a=fact_a.fact_id,
                fact_id_b=fact_b.fact_id,
                reason=(
                    f"Different numeric values for entity '{entity_name}': "
                    f"{a_nums} vs {b_nums}"
                ),
            )

        return None

    # =========================================================================
    # Constants Prompt Generation
    # =========================================================================

    def _format_constants_by_file(
        self, max_entries: int | None = None
    ) -> tuple[list[str], int]:
        """Format constants grouped by file.

        Args:
            max_entries: Maximum entries to include, or None for unlimited

        Returns:
            Tuple of (formatted_lines, entries_included_count)
        """
        by_file: dict[str, list[ConstantEntry]] = {}
        for entry in self.constants.values():
            by_file.setdefault(entry.file_path, []).append(entry)

        lines: list[str] = []
        count = 0

        for file_path in sorted(by_file.keys()):
            if max_entries is not None and count >= max_entries:
                break
            entries = sorted(by_file[file_path], key=lambda e: e.name)
            file_lines: list[str] = []
            for entry in entries:
                if max_entries is not None and count >= max_entries:
                    break
                # Format: NAME = value (type) or NAME = value or NAME (type)
                parts = [f"  - {entry.name}"]
                if entry.value is not None:
                    parts.append(f" = {entry.value}")
                if entry.type:
                    parts.append(f" ({entry.type})")
                file_lines.append("".join(parts))
                count += 1
            if file_lines:
                lines.append(f"\n**{file_path}**:")
                lines.extend(file_lines)

        return lines, count

    def get_constants_prompt_context(self, max_entries: int = 50) -> str:
        """Generate LLM prompt context section for constants.

        Args:
            max_entries: Maximum constants to include (default 50)

        Returns:
            Markdown-formatted context string, or empty string if no constants
        """
        if not self.constants:
            return ""

        entry_lines, count = self._format_constants_by_file(max_entries)
        lines = ["## Global Constants"] + entry_lines

        if count < len(self.constants):
            remaining = len(self.constants) - count
            lines.append(f"\n... and {remaining} more constants")

        return "\n".join(lines)

    def get_constants_prompt_instruction(
        self, max_entries: int = 50, use_short_form: bool = False
    ) -> str:
        """Generate constants context with instruction text for LLM prompts.

        Args:
            max_entries: Maximum constants to include (default 50)
            use_short_form: Use shorter instruction text (default False)

        Returns:
            Formatted constants section with instruction, or empty string
        """
        context = self.get_constants_prompt_context(max_entries)
        if not context:
            return ""

        instruction = (
            CONSTANTS_INSTRUCTION_SHORT if use_short_form else CONSTANTS_INSTRUCTION_FULL
        )
        return f"\n\n{context}\n\n{instruction}"

    # =========================================================================
    # Facts Prompt Generation
    # =========================================================================

    def _format_facts_simple(self, facts_list: list[FactEntry]) -> str:
        """Format facts as simple markdown list with confidence marker.

        Output format:
        - [DEF] Uses exponential backoff (search.py:45-52)
        - [LIK] Config loads from env (settings.py:10-15)

        Includes 15k token safety cap (500 facts max).

        Args:
            facts_list: List of facts to format

        Returns:
            Markdown-formatted facts list
        """
        if not facts_list:
            return ""

        # Sort by confidence (definite first) - most valuable survive truncation
        sorted_facts = self._sort_facts_by_priority(facts_list)

        # Apply 500 fact limit (~15k tokens)
        if len(sorted_facts) > MAX_FACTS_LIMIT:
            sorted_facts = sorted_facts[:MAX_FACTS_LIMIT]
            truncated = True
        else:
            truncated = False

        lines = []
        for fact in sorted_facts:
            conf = fact.confidence.value[:3].upper()  # DEF, LIK, INF, UNC
            file_name = Path(fact.file_path).name
            lines.append(
                f"- [{conf}] {fact.statement} "
                f"({file_name}:{fact.start_line}-{fact.end_line})"
            )

        result = "\n".join(lines)

        if truncated:
            result += f"\n\n... truncated to {MAX_FACTS_LIMIT} facts (15k token limit)"

        return result

    def _sort_facts_by_priority(self, facts_list: list[FactEntry]) -> list[FactEntry]:
        """Sort facts by confidence (definite first) then category.

        Args:
            facts_list: Facts to sort

        Returns:
            Sorted list of facts
        """
        confidence_order = {
            ConfidenceLevel.DEFINITE: 0,
            ConfidenceLevel.LIKELY: 1,
            ConfidenceLevel.INFERRED: 2,
            ConfidenceLevel.UNCERTAIN: 3,
        }
        return sorted(
            facts_list,
            key=lambda f: (confidence_order.get(f.confidence, 4), f.category),
        )

    def get_facts_map_prompt_context(self, cluster_files: set[str]) -> str:
        """Generate context for map phase synthesis.

        Shows facts from cluster files plus related facts via entity links.
        Uses simple markdown format with 15k token cap.

        Args:
            cluster_files: Files in the current cluster

        Returns:
            Formatted prompt context, or empty string if no facts
        """
        related_facts = self.get_related_facts(cluster_files)
        if not related_facts:
            return ""

        facts_context = self._format_facts_simple(related_facts)
        if not facts_context:
            return ""

        return FACTS_MAP_INSTRUCTION.format(facts_context=facts_context)

    def get_facts_reduce_prompt_context(self) -> str:
        """Generate context for reduce phase synthesis.

        Uses simple markdown format with 15k token cap (500 facts).

        Returns:
            Formatted prompt context, or empty string if no facts
        """
        if not self.facts:
            return ""

        facts_list = list(self.facts.values())
        facts_context = self._format_facts_simple(facts_list)

        if not facts_context:
            return ""

        # Build conflicts section (limited to 5 for brevity)
        conflicts_section = ""
        if self.conflicts:
            conflict_lines = ["\n### Potential Conflicts (require verification)"]
            for conflict in self.conflicts[:5]:
                reason = conflict.reason
                if len(reason) > 50:
                    reason = reason[:50] + "..."
                conflict_lines.append(
                    f"- [{conflict.fact_id_a[:6]}] vs [{conflict.fact_id_b[:6]}]: "
                    f"{reason}"
                )
            if len(self.conflicts) > 5:
                remaining = len(self.conflicts) - 5
                conflict_lines.append(f"... and {remaining} more conflicts")
            conflicts_section = "\n".join(conflict_lines)

        return FACTS_REDUCE_INSTRUCTION.format(
            facts_context=facts_context,
            conflicts_section=conflicts_section,
        )

    # =========================================================================
    # Report Generation
    # =========================================================================

    def _get_constants_report_section(self) -> str:
        """Generate markdown section for constants.

        Returns:
            Markdown-formatted constants list
        """
        if not self.constants:
            return ""

        entry_lines, _ = self._format_constants_by_file()  # No limit for report
        lines = ["### Constants Referenced"] + entry_lines

        return "\n".join(lines)

    def _get_facts_report_section(self) -> str:
        """Generate markdown section for facts.

        Returns:
            Markdown-formatted facts list
        """
        if not self.facts:
            return ""

        lines = ["### Verified Facts"]

        # Group by confidence for report
        by_confidence: dict[ConfidenceLevel, list[FactEntry]] = {}
        for fact in self.facts.values():
            by_confidence.setdefault(fact.confidence, []).append(fact)

        confidence_order = [
            ConfidenceLevel.DEFINITE,
            ConfidenceLevel.LIKELY,
            ConfidenceLevel.INFERRED,
            ConfidenceLevel.UNCERTAIN,
        ]

        for confidence in confidence_order:
            facts_list = by_confidence.get(confidence, [])
            if not facts_list:
                continue
            lines.append(f"\n#### {confidence.value.title()} Facts")
            for fact in sorted(facts_list, key=lambda f: (f.category, f.file_path)):
                lines.append(
                    f"- [F-{fact.fact_id}] {fact.statement} "
                    f"({fact.file_path}:{fact.start_line})"
                )

        return "\n".join(lines)

    def get_report_suffix(self) -> str:
        """Generate unified markdown suffix for final report.

        Placed before Sources section in research output.
        Includes both constants and facts sections.

        Returns:
            Markdown-formatted evidence sections, or empty string if no evidence
        """
        sections: list[str] = []

        if self.constants or self.facts:
            sections.append("\n## Evidence")

        constants_section = self._get_constants_report_section()
        if constants_section:
            sections.append(constants_section)

        facts_section = self._get_facts_report_section()
        if facts_section:
            sections.append(facts_section)

        if not sections:
            return ""

        return "\n\n".join(sections)

    def insert_into_report(self, answer: str) -> str:
        """Insert evidence suffix into report before Sources section.

        Args:
            answer: The research report text

        Returns:
            Report with evidence section inserted, or unchanged if no evidence
        """
        suffix = self.get_report_suffix()
        if not suffix:
            return answer
        if "## Sources" in answer:
            return answer.replace("## Sources", f"{suffix}\n\n## Sources")
        return f"{answer}\n{suffix}"

    # =========================================================================
    # Progress Display
    # =========================================================================

    def format_progress_table(self) -> str:
        """Format evidence in LLM prompt format for progress display.

        Shows the same content that gets sent to the LLM,
        so users can see exactly what context the model receives.

        Returns:
            Markdown-formatted evidence with summary footer
        """
        sections: list[str] = []

        # Constants section
        if self.constants:
            entry_lines, _ = self._format_constants_by_file(max_entries=None)
            if entry_lines:
                sections.append("### Constants")
                sections.extend(entry_lines)

        # Facts section (simple format)
        if self.facts:
            facts_section = self.get_facts_reduce_prompt_context()
            if facts_section:
                sections.append(facts_section)

        # Build output
        if sections:
            output = ["## Evidence Context", ""] + sections
        else:
            output = []

        # Add summary
        summary = f"Constants: {self.constants_count} | Facts: {self.facts_count}"
        if output:
            output.append(f"\n---\n{summary}")
        else:
            output.append(summary)

        return "\n".join(output)

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> dict:
        """Serialize ledger for JSON transport.

        Returns:
            Dictionary representation
        """
        return {
            "constants": {
                key: {
                    "name": entry.name,
                    "file_path": entry.file_path,
                    "value": entry.value,
                    "type": entry.type,
                }
                for key, entry in self.constants.items()
            },
            "facts": {
                fact_id: {
                    "fact_id": fact.fact_id,
                    "statement": fact.statement,
                    "file_path": fact.file_path,
                    "start_line": fact.start_line,
                    "end_line": fact.end_line,
                    "category": fact.category,
                    "confidence": fact.confidence.value,
                    "entities": list(fact.entities),
                    "cluster_id": fact.cluster_id,
                }
                for fact_id, fact in self.facts.items()
            },
            "conflicts": [
                {
                    "fact_id_a": c.fact_id_a,
                    "fact_id_b": c.fact_id_b,
                    "reason": c.reason,
                }
                for c in self.conflicts
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> EvidenceLedger:
        """Deserialize ledger from JSON transport.

        Args:
            data: Dictionary representation

        Returns:
            New EvidenceLedger instance
        """
        ledger = cls()

        # Load constants
        for key, const_data in data.get("constants", {}).items():
            entry = ConstantEntry(
                name=const_data["name"],
                file_path=const_data["file_path"],
                value=const_data.get("value"),
                type=const_data.get("type"),
            )
            ledger.constants[key] = entry

        # Load facts (add_fact handles entity index)
        for fact_data in data.get("facts", {}).values():
            fact = FactEntry(
                fact_id=fact_data["fact_id"],
                statement=fact_data["statement"],
                file_path=fact_data["file_path"],
                start_line=fact_data["start_line"],
                end_line=fact_data["end_line"],
                category=fact_data["category"],
                confidence=ConfidenceLevel(fact_data["confidence"]),
                entities=tuple(fact_data.get("entities", [])),
                cluster_id=fact_data.get("cluster_id", 0),
            )
            ledger.add_fact(fact)

        # Load conflicts
        for conflict_data in data.get("conflicts", []):
            ledger.conflicts.append(
                FactConflict(
                    fact_id_a=conflict_data["fact_id_a"],
                    fact_id_b=conflict_data["fact_id_b"],
                    reason=conflict_data["reason"],
                )
            )

        return ledger
