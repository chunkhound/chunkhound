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

- ChunkHound intentionally defaults both Anthropic utility and synthesis roles to the `claude-haiku` sentinel. Haiku is capable enough for synthesis, is Anthropic's cheapest available Claude model, and Anthropic does not currently offer a true low-cost utility tier.
- Extended thinking has two modes:
  - Adaptive (Claude Opus 4.7, Opus 4.6, Sonnet 4.6, Mythos). No budget_tokens needed. Auto-enables interleaved thinking.
  - Manual (Opus 4.5 and older Claude 4 models). Requires thinking.budget_tokens of at least 1024.
- Opus 4.7 accepts only adaptive thinking; manual is rejected with a 400 error.
- The effort parameter (low/medium/high/xhigh/max) is supported on Opus 4.5/4.6/4.7, Sonnet 4.6, and Mythos. xhigh is Opus 4.7 only; max is 4.6 and later.
- Thinking blocks are processed but not included in text output by default.
- Token usage includes full thinking tokens (not just summary) for billing.
