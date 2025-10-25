from chunkhound.llm_manager import LLMManager


def test_llm_manager_registry_includes_codex_cli():
    # Red test: registry should include codex-cli provider key
    assert "codex-cli" in LLMManager._providers

