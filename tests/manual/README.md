# Manual Test Scripts

These scripts test ChunkHound features with real API calls. They require valid API keys and make actual requests to external services.

## Anthropic Extended Thinking Tests

Test the Anthropic provider with extended thinking support.

### Prerequisites

```bash
# Install dependencies
uv sync

# Set API key
export ANTHROPIC_API_KEY=sk-ant-api03-...
```

### Run Tests

```bash
uv run python tests/manual/test_anthropic_thinking.py
```

### What It Tests

1. **Basic Completion**: Standard completion without thinking
2. **Thinking Completion**: Completion with extended thinking enabled
3. **Structured Output**: JSON schema-based structured output
4. **Health Check**: Provider connectivity and configuration
5. **Usage Stats**: Token usage tracking

### Expected Output

```
================================================================================
Anthropic Provider Extended Thinking Tests
================================================================================
API Key: sk-ant-api03-XXXXXXX...

================================================================================
TEST 1: Basic Completion (no thinking)
================================================================================
Prompt: What is 2+2? Answer in one sentence.
Response: 2 + 2 equals 4.
Tokens used: 23
Finish reason: end_turn

================================================================================
TEST 2: Completion with Extended Thinking
================================================================================
...

================================================================================
TEST SUMMARY
================================================================================
✅ PASS: Basic Completion
✅ PASS: Thinking Completion
✅ PASS: Structured Output
✅ PASS: Health Check
✅ PASS: Usage Stats

🎉 All tests passed!
```

## Notes

- Extended thinking is only supported on Claude Opus 4.1, Opus 4, Sonnet 4, and Sonnet 3.7
- Thinking budget must be at least 1024 tokens
- Thinking blocks are processed but not included in text output by default
- Token usage includes full thinking tokens (not just summary) for billing
