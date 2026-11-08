"""Shared offline policy for resolving configured embedding dimensions."""

from chunkhound.interfaces.embedding_provider import EmbeddingProvider


def configured_embedding_dims(provider: EmbeddingProvider) -> int | None:
    """Dimensions the provider will embed at, or None if it cannot know yet.

    An explicit ``output_dims`` wins. Otherwise a provider that knows its
    model's dimensions reports them; an unknown model on a custom endpoint only
    learns them from its first response. Never probe the endpoint or guess from
    the provider's fallback dimensions.
    """
    if provider.output_dims is not None:
        return provider.output_dims
    if provider.supported_dimensions:
        return provider.dims
    return None
