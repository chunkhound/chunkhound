"""Query expansion prompt for deep research service.

Generates semantically diverse search queries for embedding-based retrieval.
"""

# Simplified system prompt per GPT-5-Nano best practices
SYSTEM_MESSAGE = (
    """Generate diverse code search queries for semantic embedding systems."""
)

# User prompt template with variables: query, context_str, NUM_LLM_EXPANDED_QUERIES
USER_TEMPLATE = """Query: {query}
Context: {context_root_query}{context_str}

Generate {num_queries} semantically diverse search queries for code retrieval:

1. Rephrase using synonyms and alternative perspectives
   - Use different words with same meaning (e.g., "implement" → "build/create", "authentication" → "login/verification")
   - Ask from a different angle while preserving intent

2. Implementation-focused natural language
   - Target code elements: "Find classes that...", "What functions handle..."
   - Mention specific code concepts (services, handlers, middleware, etc.)

Requirements:
- COMPLETE SENTENCES only - no keyword lists, no pseudo-code
- Each query must be self-contained and semantically rich
- Different semantic angles of the same question
- Concise (1-2 sentences max per query)"""
