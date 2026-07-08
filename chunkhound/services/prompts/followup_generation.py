"""Follow-up question generation prompt for deep research service.

Generates follow-up questions to deepen understanding during BFS exploration.
"""

# Simplified system prompt per GPT-5-Nano best practices
SYSTEM_MESSAGE = """Generate follow-up questions to deepen understanding."""

# User prompt template with variables: gist_section,
# root_query, query, ancestors, source_content,
# chunks_preview, constants_section, max_questions,
# target_instruction
USER_TEMPLATE = """{gist_section}Root: {root_query}
Current: {query}
Context: {ancestors}

Source:
{source_content}

Fragments:
{chunks_preview}
{constants_section}
Generate 0-{max_questions} follow-up questions about {target_instruction}. Focus on:
1. Component interactions
2. Information flow
3. Dependencies

Use exact names. If the source fully answers the
question, return fewer questions or empty array."""
