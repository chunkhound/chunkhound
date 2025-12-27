from chunkhound.autodoc import docsite


def test_build_cleanup_prompt_v2_includes_schema_and_injects_inputs() -> None:
    prompt = docsite._build_cleanup_prompt(
        title="My Title",
        body="## Overview\nBody line.",
    )

    assert "do NOT force a fixed schema" in prompt
    assert "input markdown is agent-facing documentation derived from code" in prompt
    assert "You are NOT a coding agent" in prompt
    assert "Do not ask questions" in prompt
    assert "Do not request file paths" in prompt
    assert "Input page title: My Title" in prompt
    assert "Input markdown:" in prompt
    assert "Body line." in prompt


def test_build_cleanup_prompt_end_user_uses_end_user_template() -> None:
    prompt = docsite._build_cleanup_prompt(
        title="My Title",
        body="## Overview\nBody line.",
        taint="end-user",
    )

    assert "Audience goal (end-user)" in prompt
    assert "set up, configured, and used" in prompt
    assert "Keep code identifiers" in prompt
    assert "Do NOT invent recommendations" in prompt
