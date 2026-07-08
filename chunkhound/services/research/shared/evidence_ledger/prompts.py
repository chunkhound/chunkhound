"""Prompt templates for evidence ledger extraction and synthesis integration.

Templates for:
1. Constants instruction text for LLM prompts
2. Extracting atomic facts from source material during map phase
3. Integrating facts into synthesis prompts (map and reduce)
"""

from __future__ import annotations

# =============================================================================
# Constants Prompts
# =============================================================================

# Standard instruction text for LLM prompts when constants are present
CONSTANTS_INSTRUCTION_FULL = (
    "IMPORTANT: When your answer references configuration values, "
    "limits, or key figures from the source material, refer to them by their "
    "names (e.g., 'the system processes up to MAX_BATCH_SIZE items') "
    "rather than embedding raw values."
)

CONSTANTS_INSTRUCTION_SHORT = (
    "When referencing configuration values or key figures, use their names "
    "rather than raw values."
)


# =============================================================================
# Facts Extraction Prompts
# =============================================================================

# System prompt for fact extraction LLM calls
FACT_EXTRACTION_SYSTEM = """You extract atomic facts from
source material for research synthesis.

An ATOMIC FACT is ONE verifiable claim about the source material:
- Specific enough to cite with source:location
- Grounded in literal content, not inference alone
- MUST be 3-5 words only

Confidence levels (pick most appropriate):
- definite: Explicitly stated, directly verifiable
- likely: Strongly implied by structure/patterns
- inferred: Reasonable inference, may need verification
- uncertain: Possible interpretation, depends on context

REASONING APPROACH (Chain of Draft):
Before extracting facts, analyze using minimal draft notes (5 words max per step):
1. Scan source → identify key behaviors
2. Note patterns → structure, constraints
3. Spot key figures → values, purposes
4. Filter → query-relevant facts only

Draft example:
- "rate limiting with thresholds"
- "BATCH_SIZE controls chunking"
- "concurrent processing used"

Then extract atomic facts from your draft insights.

For each fact, extract:
1. statement: The atomic claim (3-5 WORDS ONLY - ultra terse)
2. source: Source file path or source identifier
3. location: Location within source (line range like "lines 45-52" / "L45" or a section)
4. url: Canonical source URL when the source material comes from the web (optional)
5. category: Your classification (architecture, behavior, constraint, etc.)
6. confidence: One of the levels above
7. entities: Named entities referenced (components/sources/concepts)

Statement examples:
GOOD: "Limits requests per second" (4 words)
GOOD: "Batches up to BATCH_SIZE" (4 words)
BAD: "Service throttles requests with rate limiting" (6 words - too long)

IMPORTANT:
- Extract facts RELEVANT to the query
- Prioritize DEFINITE facts over inferred
- Maximum {max_facts} facts
- Each fact must be independently verifiable
- Keep statements to 3-5 words"""


# User prompt template for extraction
FACT_EXTRACTION_USER = """Query: {root_query}

Extract atomic facts from this source material:

{source_context}

Respond with JSON array:
```json
[
  {{
    "statement": "Limits requests per second",
    "source": "services/search.py",
    "location": "lines 45-52",
    "category": "behavior",
    "confidence": "definite",
    "entities": ["SearchService", "MAX_RETRIES"]
  }},
  {{
    "statement": "Supports cursor pagination",
    "source": "docs/api.md",
    "location": "Pagination",
    "url": "https://docs.example.com/api#pagination",
    "category": "behavior",
    "confidence": "definite",
    "entities": ["API"]
  }}
]
```"""


# =============================================================================
# Facts Synthesis Integration Prompts
# =============================================================================

# Instruction for map phase synthesis (per-cluster)
FACTS_MAP_INSTRUCTION = """## Verified Facts (This Cluster)
Use these verified facts to ground your analysis alongside source refs [N].
If you find contradictory evidence, note the discrepancy.

{facts_context}"""


# Instruction for reduce phase synthesis (global)
FACTS_REDUCE_INSTRUCTION = """## Verified Facts Ledger
These facts were extracted from source material. Ground synthesis in verified evidence.

Confidence: DEFINITE (cite directly) > LIKELY (confident)
> INFERRED (qualify) > UNCERTAIN (verify)

{facts_context}

{conflicts_section}"""
