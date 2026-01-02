"""Follow-up question generation prompt for deep research service.

Generates follow-up questions to deepen code understanding during BFS exploration.
"""

# Simplified system prompt per GPT-5-Nano best practices
SYSTEM_MESSAGE = """Generate follow-up questions to deepen code understanding."""

# User prompt template with variables: gist_section, context, code_section, chunks_preview, constants_section, max_questions, target_instruction
USER_TEMPLATE = """{gist_section}Root: {root_query}
Current: {query}
Context: {ancestors}

Code:
{code_section}

Chunks:
{chunks_preview}
{constants_section}
Generate 0-{max_questions} follow-up questions about {target_instruction}. Focus on:
1. Component interactions
2. Data/control flow
3. Dependencies

Use exact function/class/file names. If code fully answers the question, return fewer questions or empty array."""
