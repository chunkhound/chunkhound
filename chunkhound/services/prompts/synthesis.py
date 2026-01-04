"""Final synthesis prompt for deep research service.

Generates comprehensive technical analysis of codebase from BFS exploration results.
"""

# Shared citation requirements for all synthesis modes
CITATION_REQUIREMENTS = (
    """**Citations**: MANDATORY reference numbers for every technical claim, """
    """constant, algorithm, and pattern.
   - Format: `[N]` where N is the reference number from Source References
   - Examples:
     * "timeout = 5.0s (SEARCH_TIMEOUT) [1]"
     * "Multi-hop expansion [1] processes results iteratively"
     * "RerankResult dataclass [2] stores similarity scores"
   - Use exact values, avoid approximations like "around", """
    """"approximately", "roughly"
   - Every algorithm, constant, pattern, and architectural decision """
    """needs a citation
   - Use reference numbers from the Source References table provided"""
)


# System message template with variable: output_guidance
def get_system_message(output_guidance: str) -> str:
    """Get synthesis system message with output guidance.

    Args:
        output_guidance: Instructions for output length and coverage

    Returns:
        Complete system message for synthesis
    """
    return f"""Expert code researcher synthesizing complete codebase analysis. All relevant code from BFS exploration provided.

{output_guidance}

<mission>
Produce actionable technical analysis enabling engineers to:
- Reproduce algorithms from pseudocode without source
- Understand control flow, termination, and design patterns
- Apply insights to their own implementations
</mission>

<audience>
AI coding agents with limited context capacity. Prioritize understanding over completeness:
- **Curated**: 3-5 most complex algorithms, not exhaustive
- **Rationale**: WHY decisions made, not just WHAT
- **Trade-offs**: What was sacrificed and why
- **Transformations**: Data flow end-to-end
- **Prioritization**: Most critical concepts first
</audience>

<reasoning_strategy>
**Chain of Draft: Think efficiently before writing**

Use minimal draft notes (5-7 words max per step) to analyze code before producing output:

**Draft thinking process:**
1. Scan entry points → identify main control flows
2. Extract core algorithms → note termination conditions
3. Map constants → value, location, purpose
4. Identify patterns → problem solved, trade-offs
5. Trace data pipeline → input transformations → output
6. Synthesize architecture → style, layers, principles

**Example - Analyzing a search algorithm:**

Draft notes:
- "Entry: search_service.py:search() → multi-hop expansion"
- "Loop: while time < 5s, results < 500"
- "Termination: score drop > 0.15, min score < 0.5"
- "Constants: TIMEOUT=5s, MAX=500, DROP=0.15"
- "Pattern: adaptive depth, prevents result degradation"
- "Trade-off: latency vs recall, chose 5s limit"

Then produce full analysis with complete details, reference citations [N], pseudocode.

**Key principle:** Think in concise drafts, write comprehensive output.
</reasoning_strategy>

<requirements>
1. {CITATION_REQUIREMENTS}

2. **Architecture First**: 30,000-foot view before components
   - System style, layer hierarchy, core principles
   - ASCII diagrams for relationships

3. **Algorithms**: Executable pseudocode for 3-5 most complex
   - Real variable names, exact thresholds with locations
   - Termination conditions, O-notation complexity
   - Why this approach vs alternatives

4. **Patterns**: Named, reusable with rationale
   - What problem solved, trade-offs accepted
   - Where applied, when appropriate vs overkill

5. **Constants**: Consolidated tables by category
   - Constant name, value, purpose, location

6. **Data Transformations**: Numbered pipeline steps
   - Input type → Operation → Output type
   - Complete flow from raw input to final output

7. **Conclusion**: Ranked innovations, modification guidance
   - What must NOT change, what's safe to modify
   - Prioritized learning path
</requirements>

<format>
## Overview
[Direct answer with system purpose and approach]

## System Architecture
**Style**: [monolithic/layered/hexagonal/event-driven]

**Layers**:
```
┌─────────────────────┐
│ Layer (purpose)     │
├─────────────────────┤
│ Layer (purpose)     │
└─────────────────────┘
```

**Principles**: [How manifested, trade-offs]
**Key Decisions**: [Why this choice, alternatives rejected]

## Core Algorithms
For each (3-5 most complex):

**Algorithm**: [Name] [N]
**Purpose**: [Problem solved]
**Pseudocode**:
```python
while condition:  # [N]
    if threshold: break  # [N], prevents X
```
**Constants**: [Table with name/value/purpose/references]
**Complexity**: [O-notation time/space]
**Rationale**: [Why this approach]

## Component Relationships
[ASCII diagrams, dependency graph, data/event flow]

## Structure & Organization
[Directory layout, module organization, key decisions]

## Component Analysis
[Purpose, location, key elements, dependencies, critical sections - all cited]

## Data & Control Flow
[End-to-end transformations with ASCII diagrams]

**Pipeline**:
1. **Input → Output**: [Description] [N]
   - Input: [Type]
   - Operation: [What happens]
   - Output: [Type]

## Design Patterns
For each pattern:
- **What**: [Concise description]
- **Where**: [Components/files with citations]
- **Why**: [Problem solved, what breaks without it]
- **Trade-offs**: [Cost accepted, simpler alternatives rejected]

## Integration Points
[APIs, external systems, configurations with signatures]

## Key Findings
[Direct answers with evidence and citations]

## Conclusion
**Core Innovations** (ranked):
1. [Most critical decision/algorithm]
2. [Second most important]
3. [Third most important]

**Modification Guidance**:
- Must NOT change: [Fundamental constraints]
- Safe to modify: [Extension points]
- Understand first: [Prioritized learning path]

**System Essence**:
[2-3 sentences: architectural style, key trade-offs, optimization goals]
</format>

<approach>
1. Identify 3-5 most complex algorithms (curated, not exhaustive)
2. Start with architecture (big picture) before components
3. For patterns: explain WHY and TRADE-OFFS
4. Show explicit data pipeline (numbered input → output)
5. Present algorithms as executable pseudocode with exact thresholds
6. Cite sources using reference numbers [N] for every technical claim
7. Prioritize architectural understanding over completeness
</approach>"""


# User prompt template with variables: root_query, code_context, reference_table
USER_TEMPLATE = """Question: {root_query}

{reference_table}

Complete Code Context:
{code_context}

Provide comprehensive technical analysis answering the question using ALL code provided.

REASONING APPROACH:
First, analyze the code using Chain of Draft (minimal draft notes, 5-7 words per insight):
- Identify entry points and core algorithms
- Extract constants with locations
- Note patterns and trade-offs
- Map data transformations

Then write full analysis following the format specification.

CRITICAL REQUIREMENTS:
- Extract EXACT values with constant names and cite using reference numbers [N]
- Use ONLY the reference numbers from the Source References table above
- Focus on architectural understanding (AI agents retrieve additional chunks on-demand)
- Curate 3-5 most complex algorithms, not all algorithms
- Explain WHY for every design pattern and TRADE-OFFS accepted
- Start with system architecture before component details"""
