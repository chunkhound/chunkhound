"""Core constants for ChunkHound."""

# OpenAI embedding model defaults
OPENAI_DEFAULT_MODEL = "text-embedding-3-small"

# Voyage AI embedding model defaults
VOYAGE_DEFAULT_MODEL = "voyage-3.5"
VOYAGE_DEFAULT_RERANK_MODEL = "rerank-2.5"

# Superseded embedding models mapped to their current replacement. Surfaced as
# a suggestion only: adopting one re-embeds the whole index, so the choice
# stays with the operator. Code models point at the code-specialised
# successor; general models keep to their own family.
EMBEDDING_MODEL_UPGRADES: dict[str, str] = {
    "voyage-code-3": "voyage-code-4",
    "voyage-3.5": "voyage-4",
    "voyage-3.5-lite": "voyage-4-lite",
    "voyage-3-large": "voyage-4-large",
}
