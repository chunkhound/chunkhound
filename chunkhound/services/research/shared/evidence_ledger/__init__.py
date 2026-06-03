"""Evidence Ledger: Unified constants and facts for research synthesis.

INTERNAL: Produces structured evidence consumed by LLM synthesizers during
the map-reduce research pipeline, not rendered to end-users directly.
The ``_prompt_context`` / ``_prompt_instruction`` methods format evidence
into markdown that an LLM reads during synthesis.

Provides:
1. EvidenceLedger - Unified collection of constants and facts
2. FactExtractor - LLM-based fact extraction from source clusters
3. extract_facts_with_clustering - Clustered extraction with reusable clusters
4. Data models and prompt templates
"""

from chunkhound.services.research.shared.evidence_ledger.clustered_extractor import (
    ClusteredExtractionResult,
    extract_facts_with_clustering,
)
from chunkhound.services.research.shared.evidence_ledger.extractor import FactExtractor
from chunkhound.services.research.shared.evidence_ledger.ledger import EvidenceLedger
from chunkhound.services.research.shared.evidence_ledger.models import (
    ConfidenceLevel,
    ConstantEntry,
    EntityLink,
    EvidenceType,
    FactConflict,
    FactEntry,
)
from chunkhound.services.research.shared.evidence_ledger.prompts import (
    CONSTANTS_INSTRUCTION_FULL,
    CONSTANTS_INSTRUCTION_SHORT,
    FACT_EXTRACTION_SYSTEM,
    FACT_EXTRACTION_USER,
    FACTS_MAP_INSTRUCTION,
    FACTS_REDUCE_INSTRUCTION,
)

__all__ = [
    # Main classes
    "EvidenceLedger",
    "FactExtractor",
    # Clustered extraction
    "ClusteredExtractionResult",
    "extract_facts_with_clustering",
    # Models
    "ConfidenceLevel",
    "ConstantEntry",
    "EntityLink",
    "EvidenceType",
    "FactConflict",
    "FactEntry",
    # Prompts
    "CONSTANTS_INSTRUCTION_FULL",
    "CONSTANTS_INSTRUCTION_SHORT",
    "FACT_EXTRACTION_SYSTEM",
    "FACT_EXTRACTION_USER",
    "FACTS_MAP_INSTRUCTION",
    "FACTS_REDUCE_INSTRUCTION",
]
