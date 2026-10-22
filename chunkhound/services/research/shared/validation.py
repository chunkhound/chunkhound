"""Shared preflight checks for CLI and MCP code research."""

from chunkhound.embeddings import EmbeddingManager
from chunkhound.llm_manager import LLMManager


def validate_research_prerequisites(
    embedding_manager: EmbeddingManager | None,
    llm_manager: LLMManager | None,
) -> None:
    """Reject missing research capabilities before starting any research work."""
    if not llm_manager or not llm_manager.is_configured():
        raise ValueError(
            "LLM not configured. Configure an LLM provider via:\n"
            "1. Create .chunkhound.json with llm configuration, OR\n"
            "2. Set CHUNKHOUND_LLM_API_KEY environment variable"
        )

    if not embedding_manager or not embedding_manager.list_providers():
        raise ValueError(
            "No embedding providers available. Code research requires reranking "
            "support."
        )

    provider = embedding_manager.get_provider()
    if not (hasattr(provider, "supports_reranking") and provider.supports_reranking()):
        raise ValueError(
            "Code research requires a provider with reranking support. "
            "Configure a rerank_model in your embedding configuration."
        )
