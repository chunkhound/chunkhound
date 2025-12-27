"""Constants Ledger: Aggregates constant metadata for research context.

Collects constants from source chunks to:
1. Provide global scope awareness to LLM with actual values
2. Help LLM understand system behavior through configuration values
3. Generate report appendix with constant reference table
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


# Standard instruction text for LLM prompts when constants are present
CONSTANTS_INSTRUCTION_FULL = (
    "IMPORTANT: When your answer references configuration values, "
    "limits, or magic numbers from the code, refer to them by their "
    "constant names (e.g., 'the system retries up to MAX_RETRIES times') "
    "rather than embedding raw values."
)

CONSTANTS_INSTRUCTION_SHORT = (
    "When referencing configuration values or limits, use constant names "
    "rather than raw values."
)


@dataclass(frozen=True, slots=True)
class ConstantEntry:
    """Single constant with metadata."""

    name: str
    file_path: str
    value: str | None = None
    type: str | None = None


@dataclass
class ConstantsLedger:
    """Aggregated constants from research source chunks."""

    entries: dict[str, ConstantEntry] = field(default_factory=dict)
    # Key: "file_path:name" for uniqueness

    @classmethod
    def from_chunks(cls, chunks: Iterable[dict]) -> ConstantsLedger:
        """Build ledger from chunk metadata.

        Args:
            chunks: Iterable of chunk dicts with optional 'metadata.constants' field

        Returns:
            New ConstantsLedger with extracted constants
        """
        ledger = cls()
        for chunk in chunks:
            file_path = chunk.get("file_path", "")
            metadata = chunk.get("metadata") or {}
            constants = metadata.get("constants") or []

            for const in constants:
                name = const.get("name")
                if not name:
                    continue
                key = f"{file_path}:{name}"
                if key not in ledger.entries:
                    ledger.entries[key] = ConstantEntry(
                        name=name,
                        file_path=file_path,
                        value=const.get("value"),
                        type=const.get("type"),
                    )
        return ledger

    def merge(self, other: ConstantsLedger) -> ConstantsLedger:
        """Merge another ledger into this one (immutable).

        Args:
            other: Another ConstantsLedger to merge

        Returns:
            New ConstantsLedger with entries from both
        """
        merged = ConstantsLedger(entries=dict(self.entries))
        merged.entries.update(other.entries)
        return merged

    def _format_entries_by_file(
        self, max_entries: int | None = None
    ) -> tuple[list[str], int]:
        """Format entries grouped by file (shared helper).

        Args:
            max_entries: Maximum entries to include, or None for unlimited

        Returns:
            Tuple of (formatted_lines, entries_included_count)
        """
        by_file: dict[str, list[ConstantEntry]] = {}
        for entry in self.entries.values():
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

    def get_prompt_context(self, max_entries: int = 50) -> str:
        """Generate LLM prompt context section.

        Presents constants with their values to help LLM understand system behavior.

        Args:
            max_entries: Maximum constants to include (default 50)

        Returns:
            Markdown-formatted context string, or empty string if no constants
        """
        if not self.entries:
            return ""

        entry_lines, count = self._format_entries_by_file(max_entries)
        lines = ["## Global Constants"] + entry_lines

        if count < len(self.entries):
            remaining = len(self.entries) - count
            lines.append(f"\n... and {remaining} more constants")

        return "\n".join(lines)

    def get_prompt_instruction(
        self, max_entries: int = 50, use_short_form: bool = False
    ) -> str:
        """Generate constants context with instruction text for LLM prompts.

        Combines the prompt context from get_prompt_context() with standard
        instruction text about how to reference constants in answers.

        Args:
            max_entries: Maximum constants to include (default 50)
            use_short_form: Use shorter instruction text (default False)

        Returns:
            Formatted constants section with instruction, or empty string if no constants
        """
        context = self.get_prompt_context(max_entries)
        if not context:
            return ""

        instruction = (
            CONSTANTS_INSTRUCTION_SHORT if use_short_form else CONSTANTS_INSTRUCTION_FULL
        )
        return f"\n\n{context}\n\n{instruction}"

    def get_report_suffix(self) -> str:
        """Generate markdown suffix for final report.

        Placed before Sources section in research output.
        Uses same format as prompt context for consistency.

        Returns:
            Markdown-formatted constants list, or empty string if no constants
        """
        if not self.entries:
            return ""

        entry_lines, _ = self._format_entries_by_file()  # No limit for report
        lines = ["", "## Constants Referenced"] + entry_lines

        return "\n".join(lines)

    def insert_into_report(self, answer: str) -> str:
        """Insert constants suffix into report before Sources section.

        Args:
            answer: The research report text

        Returns:
            Report with constants section inserted, or unchanged if no constants
        """
        suffix = self.get_report_suffix()
        if not suffix:
            return answer
        if "## Sources" in answer:
            return answer.replace("## Sources", f"{suffix}\n\n## Sources")
        return f"{answer}\n{suffix}"

    def __len__(self) -> int:
        return len(self.entries)
