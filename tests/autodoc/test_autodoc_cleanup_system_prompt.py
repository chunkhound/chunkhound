from __future__ import annotations

from hashlib import sha256

from chunkhound.autodoc.cleanup import _build_cleanup_system_prompt
from chunkhound.autodoc.models import CleanupConfig


def _prompt_digest(prompt: str) -> str:
    return sha256(prompt.encode("utf-8")).hexdigest()


def test_build_cleanup_system_prompt_balanced_is_byte_stable() -> None:
    prompt = _build_cleanup_system_prompt(
        CleanupConfig(
            mode="llm",
            batch_size=1,
            max_completion_tokens=1,
            audience="balanced",
        )
    )

    assert prompt.startswith("You are editing existing engineering documentation")
    assert "Audience:" not in prompt
    assert _prompt_digest(prompt) == (
        "482ffed772133cb914e2640bf1fb5bbfe17a5e787994b55ca74bd4c044e46e12"
    )


def test_build_cleanup_system_prompt_technical_includes_guidance() -> None:
    prompt = _build_cleanup_system_prompt(
        CleanupConfig(
            mode="llm",
            batch_size=1,
            max_completion_tokens=1,
            audience="technical",
        )
    )

    assert "Audience: technical" in prompt
    assert _prompt_digest(prompt) == (
        "26e2ec0c357f9bac4c5b4d437cbd6e8120895a99a7579fd8b20e1a69a64691cf"
    )


def test_build_cleanup_system_prompt_end_user_includes_guidance() -> None:
    prompt = _build_cleanup_system_prompt(
        CleanupConfig(
            mode="llm",
            batch_size=1,
            max_completion_tokens=1,
            audience="end-user",
        )
    )

    assert "Audience: end-user" in prompt
    assert _prompt_digest(prompt) == (
        "ededaf8a3f9d36d6d76662dd29d2f689f6dcec17cfc07ab0db283fd0cd645c20"
    )
