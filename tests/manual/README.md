# Manual Test Scripts

These scripts test ChunkHound features with real API calls. They require valid API keys and make actual requests to external services.

## Gemini Thinking Tests

Test the Gemini provider with thinking support (Gemini 3 uses `thinking_level`, Gemini 2.5 uses `thinking_budget`).

### Prerequisites

```bash
# Install dependencies
uv sync

# Set API key
export GOOGLE_API_KEY=AIza...
```

### Run Tests

```bash
uv run python tests/manual/test_gemini_thinking.py
```

### What It Tests

1. **Basic Completion**: Standard completion without explicit thinking control
2. **Gemini 3 Thinking**: Completion with thinking_level (low/high)
3. **Gemini 2.5 Thinking**: Completion with thinking_budget (converted from level)
4. **Structured Output**: JSON schema-based structured output
5. **Health Check**: Provider connectivity and configuration
6. **Usage Stats**: Token usage tracking

### Expected Output

```
================================================================================
Gemini Provider Thinking Tests
================================================================================
API Key: XXXXXXX...

================================================================================
TEST 1: Basic Completion (Gemini 3)
================================================================================
Prompt: What is 2+2? Answer in one sentence.
Response: 2 + 2 equals 4.
Tokens used: 23
Finish reason: STOP

================================================================================
TEST 2: Gemini 3 with High Thinking
================================================================================
...

================================================================================
TEST SUMMARY
================================================================================
✅ PASS: Basic Completion
✅ PASS: Gemini 3 High Thinking
✅ PASS: Gemini 2.5 Low Thinking
✅ PASS: Structured Output
✅ PASS: Health Check
✅ PASS: Usage Stats

🎉 All tests passed!
```

## Notes

- Gemini 3 uses `thinking_level` ("low", "high")
- Gemini 2.5 uses `thinking_budget` (converted from thinking_level: low=0, high=auto)
- Default temperature is 1.0 (optimized for Gemini 3)
- Token usage includes prompt, completion, and total counts
