#!/usr/bin/env python3
"""Manual test script for Anthropic provider with extended thinking and Opus 4.5 features.

This script tests the Anthropic provider with real API calls to verify:
1. Basic completion without thinking
2. Completion with thinking enabled
3. Structured JSON output
4. Content block handling
5. Opus 4.5 effort parameter
6. Context management

Usage:
    export ANTHROPIC_API_KEY=sk-ant-api03-...
    uv run python tests/manual/test_anthropic_thinking.py
"""

import asyncio
import os
import sys

import pytest

from chunkhound.providers.llm.anthropic_llm_provider import (
    AnthropicLLMProvider,
    EFFORT_SUPPORTED_MODELS,
)

# Skip all tests in this file if no API key is available
pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set - manual integration tests require real API key"
)


async def test_basic_completion():
    """Test basic completion without thinking."""
    print("\n" + "=" * 80)
    print("TEST 1: Basic Completion (no thinking)")
    print("=" * 80)

    provider = AnthropicLLMProvider(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-sonnet-4-5-20250929",
        thinking_enabled=False,
    )

    prompt = "What is 2+2? Answer in one sentence."
    print(f"\nPrompt: {prompt}")

    response = await provider.complete(prompt, max_completion_tokens=100)

    print(f"\nResponse: {response.content}")
    print(f"Tokens used: {response.tokens_used}")
    print(f"Finish reason: {response.finish_reason}")

    return response.tokens_used > 0


async def test_thinking_completion():
    """Test completion with thinking enabled."""
    print("\n" + "=" * 80)
    print("TEST 2: Completion with Extended Thinking")
    print("=" * 80)

    provider = AnthropicLLMProvider(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-sonnet-4-5-20250929",
        thinking_enabled=True,
        thinking_budget_tokens=5000,
    )

    prompt = """Solve this step-by-step:

If a train travels at 60 mph for 2.5 hours, then increases speed to 80 mph for 1.5 hours,
how far did it travel in total?"""

    print(f"\nPrompt: {prompt}")

    response = await provider.complete(prompt, max_completion_tokens=2000)

    print(f"\nResponse: {response.content}")
    print(f"Tokens used: {response.tokens_used}")
    print(f"Finish reason: {response.finish_reason}")

    # Note: With thinking enabled, the actual thinking process is in separate blocks
    # that we don't include in the final text output by default
    print("\nNote: Extended thinking blocks are processed but not included in output by default.")

    return response.tokens_used > 0


async def test_structured_output():
    """Test structured JSON output."""
    print("\n" + "=" * 80)
    print("TEST 3: Structured JSON Output")
    print("=" * 80)

    provider = AnthropicLLMProvider(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-sonnet-4-5-20250929",
        thinking_enabled=False,
    )

    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "number"},
            "explanation": {"type": "string"},
        },
        "required": ["answer", "explanation"],
    }

    prompt = "What is 15 * 23? Provide the answer and a brief explanation."
    print(f"\nPrompt: {prompt}")
    print(f"\nRequired schema: {schema}")

    response = await provider.complete_structured(
        prompt, schema, max_completion_tokens=200
    )

    print(f"\nStructured Response:")
    import json

    print(json.dumps(response, indent=2))

    # Verify it matches schema
    assert "answer" in response
    assert "explanation" in response
    assert isinstance(response["answer"], (int, float))

    return True


async def test_health_check():
    """Test provider health check."""
    print("\n" + "=" * 80)
    print("TEST 4: Provider Health Check")
    print("=" * 80)

    provider = AnthropicLLMProvider(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        thinking_enabled=True,
    )

    health = await provider.health_check()

    print(f"\nHealth status: {health}")

    assert health["status"] == "healthy"
    assert health["provider"] == "anthropic"
    assert health["thinking_enabled"] is True

    return True


async def test_usage_stats():
    """Test usage statistics tracking."""
    print("\n" + "=" * 80)
    print("TEST 5: Usage Statistics Tracking")
    print("=" * 80)

    provider = AnthropicLLMProvider(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        thinking_enabled=False,
    )

    # Make a few completions
    await provider.complete("Say hello", max_completion_tokens=50)
    await provider.complete("Say goodbye", max_completion_tokens=50)

    stats = provider.get_usage_stats()

    print(f"\nUsage Statistics:")
    print(f"  Requests made: {stats['requests_made']}")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Prompt tokens: {stats['prompt_tokens']}")
    print(f"  Completion tokens: {stats['completion_tokens']}")
    print(f"  Thinking tokens: {stats['thinking_tokens']}")

    assert stats["requests_made"] == 2
    assert stats["total_tokens"] > 0

    return True


async def test_opus_45_effort():
    """Test Opus 4.5 with effort parameter."""
    print("\n" + "=" * 80)
    print("TEST 6: Opus 4.5 Effort Parameter")
    print("=" * 80)

    provider = AnthropicLLMProvider(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-opus-4-5-20251101",
        thinking_enabled=True,
        thinking_budget_tokens=8000,
        effort="medium",
    )

    print(f"\nModel: {provider.model}")
    print(f"Effort: {provider._effort}")
    print(f"Model in supported list: {provider.model in EFFORT_SUPPORTED_MODELS}")

    prompt = "What are three key benefits of functional programming? Be concise."
    print(f"\nPrompt: {prompt}")

    response = await provider.complete(prompt, max_completion_tokens=2000)

    print(f"\nResponse: {response.content[:500]}...")
    print(f"Tokens used: {response.tokens_used}")
    print(f"Finish reason: {response.finish_reason}")

    return response.tokens_used > 0


async def test_opus_45_full_features():
    """Test Opus 4.5 with all features enabled."""
    print("\n" + "=" * 80)
    print("TEST 7: Opus 4.5 Full Features (thinking + effort + context mgmt)")
    print("=" * 80)

    provider = AnthropicLLMProvider(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-opus-4-5-20251101",
        thinking_enabled=True,
        thinking_budget_tokens=10000,
        interleaved_thinking=True,
        effort="high",
        context_management_enabled=True,
        clear_thinking_keep_turns=2,
    )

    print(f"\nConfiguration:")
    print(f"  Model: {provider.model}")
    print(f"  Thinking: {provider._thinking_enabled}")
    print(f"  Interleaved: {provider._interleaved_thinking}")
    print(f"  Effort: {provider._effort}")
    print(f"  Context Mgmt: {provider._context_management_enabled}")

    # Check beta headers
    headers = provider._get_beta_headers()
    print(f"  Beta headers: {headers}")

    prompt = "Explain the difference between async/await and callbacks in 2-3 sentences."
    print(f"\nPrompt: {prompt}")

    response = await provider.complete(prompt, max_completion_tokens=3000)

    print(f"\nResponse: {response.content}")
    print(f"Tokens used: {response.tokens_used}")
    print(f"Finish reason: {response.finish_reason}")

    # Verify health check includes new fields
    health = await provider.health_check()
    print(f"\nHealth check: {health}")

    assert health["interleaved_thinking"] is True
    assert health["context_management_enabled"] is True
    assert health.get("effort") == "high"

    return response.tokens_used > 0


async def main():
    """Run all tests."""
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        print("\nUsage:")
        print("  export ANTHROPIC_API_KEY=sk-ant-api03-...")
        print("  uv run python tests/manual/test_anthropic_thinking.py")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("Anthropic Provider Extended Thinking Tests")
    print("=" * 80)
    print(f"\nAPI Key: {os.getenv('ANTHROPIC_API_KEY')[:20]}...")

    results = []

    try:
        results.append(("Basic Completion", await test_basic_completion()))
        results.append(("Thinking Completion", await test_thinking_completion()))
        results.append(("Structured Output", await test_structured_output()))
        results.append(("Health Check", await test_health_check()))
        results.append(("Usage Stats", await test_usage_stats()))
        results.append(("Opus 4.5 Effort", await test_opus_45_effort()))
        results.append(("Opus 4.5 Full Features", await test_opus_45_full_features()))
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    if all(passed for _, passed in results):
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
