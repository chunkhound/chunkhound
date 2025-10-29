"""Question filtering prompt for deep research service.

Filters research questions for architectural relevance.
"""

SYSTEM_MESSAGE = """You are filtering research questions for architectural relevance."""

# User prompt template with variables: root_query, current_query, questions_str, max_questions
USER_TEMPLATE = """Root Query: {root_query}
Current Question: {current_query}

Candidate Follow-ups:
{questions_str}

Select the questions that:
1. Help understand system architecture (component interactions, data flow)
2. Are directly related to code elements already found
3. Deepen understanding of the ROOT query (not tangents)

Return ONLY the question numbers (comma-separated, e.g., "1,3") for the most relevant questions.
Maximum {max_questions} questions."""
