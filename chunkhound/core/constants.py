"""Core constants for ChunkHound."""

# OpenAI embedding model defaults
OPENAI_DEFAULT_MODEL = "text-embedding-3-small"

# Voyage AI embedding model defaults (runtime fallback when config is unset).
# Frozen for backward compatibility: existing configs that omit `model` keep
# resolving here, so these values must not track the site's recommendation.
VOYAGE_DEFAULT_MODEL = "voyage-3.5"
VOYAGE_DEFAULT_RERANK_MODEL = "rerank-2.5"

# Superseded embedding models mapped to their replacement, per provider. This is
# a print-only recommendation, independent of the frozen runtime fallback above,
# and does NOT change the model the operator is using. Adopting a successor
# re-embeds the whole index, so the choice stays with the operator. Keyed by
# provider so a colliding model name on another provider never picks up a
# suggestion. The code and large tiers keep their own successors; the general
# tiers both point at the cheap lite model.
EMBEDDING_MODEL_UPGRADES: dict[str, dict[str, str]] = {
    "voyageai": {
        "voyage-code-3": "voyage-code-4",
        "voyage-3.5": "voyage-4-lite",
        "voyage-3.5-lite": "voyage-4-lite",
        "voyage-3-large": "voyage-4-large",
    },
}
