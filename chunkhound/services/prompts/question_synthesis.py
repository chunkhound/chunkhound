"""Question synthesis prompt for deep research service.

Synthesizes multiple research questions into distinct unexplored aspects.
"""

# Direct, unambiguous prompt optimized for GPT-5-Nano instruction adherence
SYSTEM_MESSAGE = (
    """Synthesize research questions to explore unexplored aspects."""
)

# User prompt template with variables: root_query, questions_str, target_count
USER_TEMPLATE = """TASK: Synthesize research questions to explore distinct unexplored aspects.

ROOT QUERY: {root_query}

INPUT QUESTIONS TO SYNTHESIZE:
{questions_str}

REQUIREMENTS:
- You MUST return at least 1 synthesized question (returning zero is not acceptable)
- You MAY return up to {target_count} questions if there are that many distinct aspects to explore
- Each question must explore a DISTINCT aspect not fully covered by the inputs
- Questions must be specific and reference concrete elements from the source material (specific names where relevant)
- Focus on structural angles: component interactions, how things work, edge cases, performance, verification

EXAMPLE:
Input: "How is data validated?", "Where is validation defined?", "What validation rules exist?"
Output: {{"reasoning": "Input questions cover validation details but miss architecture and error flow. Synthesizing broader questions.", "questions": ["What is the complete validation architecture from input to storage?", "How do validation errors propagate and get handled throughout the system?"]}}

Generate your synthesized questions now (minimum 1, maximum {target_count})."""
