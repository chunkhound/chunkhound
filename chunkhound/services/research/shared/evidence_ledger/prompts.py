"""Prompt templates for evidence ledger extraction and synthesis integration.

Templates for:
1. Constants instruction text for LLM prompts
2. Extracting atomic facts from code during map phase
3. Integrating facts into synthesis prompts (map and reduce)
"""

from __future__ import annotations


# =============================================================================
# Constants Prompts
# =============================================================================

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


# =============================================================================
# Facts Extraction Prompts
# =============================================================================

# System prompt for fact extraction LLM calls
FACT_EXTRACTION_SYSTEM = """You extract atomic facts from code for research synthesis.

An ATOMIC FACT is ONE verifiable claim about the code:
- Specific enough to cite with file:line
- Grounded in literal code, not naming inference alone
- MUST be 3-5 words only

Confidence levels (pick most appropriate):
- definite: Explicitly stated, directly verifiable
- likely: Strongly implied by structure/patterns
- inferred: Reasonable inference, may need verification
- uncertain: Possible interpretation, depends on context

REASONING APPROACH (Chain of Draft):
Before extracting facts, analyze using minimal draft notes (5 words max per step):
1. Scan code → identify key behaviors
2. Note patterns → architecture, constraints
3. Spot constants → values, purposes
4. Filter → query-relevant facts only

Draft example:
- "retry logic with backoff"
- "MAX_RETRIES limits attempts"
- "async pattern throughout"

Then extract atomic facts from your draft insights.

For each fact, extract:
1. statement: The atomic claim (3-5 WORDS ONLY - ultra terse)
2. file_path: Source file
3. start_line, end_line: Line range
4. category: Your classification (architecture, behavior, constraint, etc.)
5. confidence: One of the levels above
6. entities: Code entities referenced (class/function/module names)

Statement examples:
GOOD: "Uses exponential backoff" (3 words)
GOOD: "Retries up to MAX_RETRIES" (4 words)
BAD: "SearchService uses exponential backoff for retries" (6 words - too long)

IMPORTANT:
- Extract facts RELEVANT to the query
- Prioritize DEFINITE facts over inferred
- Maximum {max_facts} facts
- Each fact must be independently verifiable
- Keep statements to 3-5 words"""


# User prompt template for extraction
FACT_EXTRACTION_USER = """Query: {root_query}

Extract atomic facts from this code cluster:

{code_context}

Respond with JSON array:
```json
[
  {{
    "statement": "Retries up to MAX_RETRIES",
    "file_path": "services/search.py",
    "start_line": 45,
    "end_line": 52,
    "category": "behavior",
    "confidence": "definite",
    "entities": ["SearchService", "MAX_RETRIES"]
  }}
]
```"""


# =============================================================================
# Facts Synthesis Integration Prompts
# =============================================================================

# Instruction for map phase synthesis (per-cluster)
FACTS_MAP_INSTRUCTION = """## Verified Facts (This Cluster)
Use these verified facts to ground your analysis. Cite fact IDs [F-xxx] alongside file refs [N].
If you find contradictory evidence, note the discrepancy.

{facts_context}"""


# Instruction for reduce phase synthesis (global)
FACTS_REDUCE_INSTRUCTION = """## Verified Facts Ledger
These facts were extracted from analyzed code. Ground synthesis in verified evidence.

Confidence: DEFINITE (cite directly) > LIKELY (confident) > INFERRED (qualify) > UNCERTAIN (verify)

{facts_context}

{conflicts_section}"""
