#!/usr/bin/env python3
"""Manual test script for Gemini provider with thinking support.

This script tests the Gemini provider with real API calls to verify:
1. Basic completion (Gemini 3)
2. Gemini 3 with high thinking level
3. Gemini 2.5 Flash with low thinking (thinking_budget=0)
4. Structured JSON output

Usage:
    export GOOGLE_API_KEY=AIza...
    uv run python tests/manual/test_gemini_thinking.py
"""

import asyncio
import os
import sys

import pytest

from chunkhound.providers.llm.gemini_llm_provider import GeminiLLMProvider

# Skip all tests in this file if no API key is available
pytestmark = pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set - manual integration tests require real API key",
)


async def test_basic_completion_gemini_3():
    """Test basic Gemini 3 completion with default thinking."""
    print("\n" + "=" * 80)
    print("TEST 1: Basic Completion (Gemini 3)")
    print("=" * 80)

    provider = GeminiLLMProvider(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-3-pro-preview",
        thinking_level="high",  # Default for Gemini 3
    )

    prompt = "What is 2+2? Answer in one sentence."
    print(f"\nPrompt: {prompt}")
    print(f"Model: {provider.model}")
    print(f"Thinking level: {provider._thinking_level}")

    response = await provider.complete(prompt, max_completion_tokens=100)

    print(f"\nResponse: {response.content}")
    print(f"Tokens used: {response.tokens_used}")
    print(f"Finish reason: {response.finish_reason}")

    return response.tokens_used > 0


async def test_gemini_3_high_thinking():
    """Test Gemini 3 with explicit high thinking for complex reasoning."""
    print("\n" + "=" * 80)
    print("TEST 2: Gemini 3 with High Thinking")
    print("=" * 80)

    provider = GeminiLLMProvider(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-3-pro-preview",
        thinking_level="high",
    )

    prompt = """Solve this step-by-step:

If a train travels at 60 mph for 2.5 hours, then increases speed to 80 mph for 1.5 hours,
how far did it travel in total?"""

    print(f"\nPrompt: {prompt}")
    print(f"Model: {provider.model}")
    print(f"Thinking level: {provider._thinking_level}")

    response = await provider.complete(prompt, max_completion_tokens=2000)

    print(f"\nResponse: {response.content[:500]}...")  # Truncate if long
    print(f"Tokens used: {response.tokens_used}")
    print(f"Finish reason: {response.finish_reason}")

    print(
        "\nNote: Gemini 3 uses thinking_level parameter for controlling reasoning depth."
    )

    return response.tokens_used > 0


async def test_gemini_2_5_low_thinking():
    """Test Gemini 2.5 Flash with low thinking (thinking_budget=0)."""
    print("\n" + "=" * 80)
    print("TEST 3: Gemini 2.5 Flash with Low Thinking")
    print("=" * 80)

    provider = GeminiLLMProvider(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-2.5-flash",
        thinking_level="low",  # Converted to thinking_budget=0 for 2.5
    )

    prompt = "List three benefits of exercise."
    print(f"\nPrompt: {prompt}")
    print(f"Model: {provider.model}")
    print(
        f"Thinking level: {provider._thinking_level} (converted to thinking_budget=0)"
    )

    response = await provider.complete(prompt, max_completion_tokens=200)

    print(f"\nResponse: {response.content}")
    print(f"Tokens used: {response.tokens_used}")
    print(f"Finish reason: {response.finish_reason}")

    print("\nNote: Gemini 2.5 uses thinking_budget parameter (low=0, high=auto).")

    return response.tokens_used > 0


async def test_structured_output():
    """Test structured JSON output with schema validation."""
    print("\n" + "=" * 80)
    print("TEST 4: Structured JSON Output")
    print("=" * 80)

    provider = GeminiLLMProvider(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-3-pro-preview",
        thinking_level="high",
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

    print("\nStructured Response:")
    import json

    print(json.dumps(response, indent=2))

    # Verify it matches schema
    assert "answer" in response
    assert "explanation" in response
    assert isinstance(response["answer"], (int, float))

    return True


async def test_health_check():
    """Test provider health check and connectivity."""
    print("\n" + "=" * 80)
    print("TEST 5: Provider Health Check")
    print("=" * 80)

    provider = GeminiLLMProvider(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-3-pro-preview",
        thinking_level="high",
    )

    health = await provider.health_check()

    print(f"\nHealth status: {health}")

    assert health["status"] == "healthy"
    assert health["provider"] == "gemini"
    assert health["model"] == "gemini-3-pro-preview"
    assert health["thinking_level"] == "high"

    return True


async def test_usage_stats():
    """Test usage statistics tracking."""
    print("\n" + "=" * 80)
    print("TEST 6: Usage Statistics Tracking")
    print("=" * 80)

    provider = GeminiLLMProvider(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-2.5-flash",  # Use Flash for speed
        thinking_level="low",
    )

    # Make a few completions
    await provider.complete("Say hello", max_completion_tokens=50)
    await provider.complete("Say goodbye", max_completion_tokens=50)

    stats = provider.get_usage_stats()

    print("\nUsage Statistics:")
    print(f"  Requests made: {stats['requests_made']}")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Prompt tokens: {stats['prompt_tokens']}")
    print(f"  Completion tokens: {stats['completion_tokens']}")

    assert stats["requests_made"] == 2
    assert stats["total_tokens"] > 0

    return True


async def main():
    """Run all tests."""
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY environment variable not set")
        print("\nUsage:")
        print("  export GOOGLE_API_KEY=AIza...")
        print("  uv run python tests/manual/test_gemini_thinking.py")
        print("\nGet your API key at: https://aistudio.google.com/apikey")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("Gemini Provider Thinking Tests")
    print("=" * 80)
    api_key = os.getenv("GOOGLE_API_KEY", "")
    # Mask API key for security - only show it exists
    masked_key = "X" * min(len(api_key), 20) + "..." if api_key else "NOT_SET"
    print(f"\nAPI Key: {masked_key}")

    results = []

    try:
        results.append(
            ("Basic Completion (Gemini 3)", await test_basic_completion_gemini_3())
        )
        results.append(("Gemini 3 High Thinking", await test_gemini_3_high_thinking()))
        results.append(
            ("Gemini 2.5 Low Thinking", await test_gemini_2_5_low_thinking())
        )
        results.append(("Structured Output", await test_structured_output()))
        results.append(("Health Check", await test_health_check()))
        results.append(("Usage Stats", await test_usage_stats()))
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
