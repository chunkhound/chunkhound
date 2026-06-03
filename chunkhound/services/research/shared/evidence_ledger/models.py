"""Evidence Ledger Data Models: Unified facts and constants from source analysis.

Defines structures for:
1. Evidence types (constants vs facts)
2. Confidence levels for facts (categorical, not numeric - LLMs are bad at estimation)
3. Constant entries from chunk metadata
4. Fact entries with source provenance and entity linking
5. Entity-to-fact relationships for cross-referencing
6. Conflict tracking between contradictory facts
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256


class EvidenceType(Enum):
    """Type of evidence entry."""

    CONSTANT = "constant"  # Metadata-extracted config value
    FACT = "fact"  # LLM-extracted behavioral claim


class ConfidenceLevel(Enum):
    """Confidence labels for extracted facts.

    Uses categorical labels rather than numeric scores because
    LLMs are unreliable at numeric confidence estimation.
    """

    DEFINITE = "definite"  # Explicitly stated, directly verifiable
    LIKELY = "likely"  # Strongly implied by patterns/structure
    INFERRED = "inferred"  # Reasonable inference from context
    UNCERTAIN = "uncertain"  # Possible interpretation, needs verification


@dataclass(frozen=True, slots=True)
class ConstantEntry:
    """Single constant with metadata.

    Extracted from chunk metadata.constants field.
    Constants have implicit DEFINITE confidence (metadata-extracted).
    """

    name: str
    file_path: str
    value: str | None = None
    type: str | None = None


@dataclass(frozen=True, slots=True)
class FactEntry:
    """Single atomic fact with source provenance.

    Each fact represents one verifiable claim about the source material,
    linked to its source location and the entities it references.
    Supports both file-based (file_path + line) and URL-based (url + section) sources.
    """

    fact_id: str  # sha256 hash of statement, primary source, and location [:12]
    statement: str  # The atomic fact (one verifiable claim)
    file_path: str  # Stable cluster/file lookup key for file-scoped retrieval
    start_line: int  # Line range start
    end_line: int  # Line range end
    category: str  # LLM-determined (architecture, behavior, etc.)
    confidence: ConfidenceLevel
    entities: tuple[str, ...]  # Named entities referenced (for linking)
    cluster_id: int  # Which cluster extracted this
    url: str | None = None  # Source URL (for web sources)
    source_section: str | None = (
        None  # Section within source for non-line-based locations
    )

    @staticmethod
    def generate_id(
        statement: str,
        primary_source: str,
        start_line: int = 0,
        end_line: int = 0,
        section: str | None = None,
    ) -> str:
        """Generate deterministic fact ID from content and location.

        NOTE: The ID formula changed from the old (statement:file_path:start-end)
        to the new (statement:primary_source:section-or-start-end). All fact IDs
        from previous versions are invalidated. Old and new ledgers cannot be
        merged — same logical fact has a different ID.

        Args:
            statement: The fact statement
            primary_source: Source file path or URL
            start_line: Start line number (0 if not applicable)
            end_line: End line number (0 if not applicable)
            section: Section within source (for non-line-based locations)

        Returns:
            12-character hex hash
        """
        location = section if section else f"{start_line}-{end_line}"
        content = f"{statement}:{primary_source}:{location}"
        return sha256(content.encode()).hexdigest()[:12]


@dataclass(frozen=True, slots=True)
class EntityLink:
    """Maps an entity name to facts that reference it.

    Enables cross-referencing: find all facts about a given
    class, function, or module.
    """

    entity_name: str  # Normalized name (e.g., "SearchService")
    fact_ids: tuple[str, ...]  # Facts referencing this entity


@dataclass(frozen=True, slots=True)
class FactConflict:
    """Records a conflict between two facts.

    Used during synthesis to flag contradictions that need
    resolution or clarification in the final output.
    """

    fact_id_a: str
    fact_id_b: str
    reason: str  # Why they conflict
