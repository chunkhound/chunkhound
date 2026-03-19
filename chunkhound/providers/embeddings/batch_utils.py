"""Token-aware batching utilities for embedding providers."""

import logging
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Protocol, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def handle_token_limit_error(
    texts: list[str],
    total_tokens: int,
    token_limit: int,
    embed_function: Callable[[list[str]], Awaitable[list[list[float]]]],
    chunk_text_function: Callable[[str, int], list[str]],
    single_text_fallback: bool = True,
) -> list[list[float]]:
    """Generic handler for token limit exceeded errors.

    Args:
        texts: List of texts that exceeded token limit
        total_tokens: Total estimated tokens in the batch
        token_limit: Maximum tokens allowed (should include safety margin)
        embed_function: Function to embed a batch of texts
        chunk_text_function: Function to chunk a single text by tokens
        single_text_fallback: Whether to use first chunk as fallback for oversized text

    Returns:
        List of embeddings for all texts

    Raises:
        ValueError: If single text can't be chunked further
    """
    if len(texts) > 1:
        # Calculate optimal number of splits based on token estimates
        num_splits = max(
            2, (total_tokens + token_limit - 1) // token_limit
        )  # Ceiling division

        logger.debug(
            f"Token limit exceeded for batch of {len(texts)} texts "
            f"({total_tokens} tokens), splitting into {num_splits} batches"
        )

        # Split texts into optimal-sized chunks
        chunk_size = len(texts) // num_splits
        remainder = len(texts) % num_splits

        all_embeddings = []
        start_idx = 0

        for i in range(num_splits):
            # Distribute remainder across first chunks
            current_chunk_size = chunk_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_chunk_size
            chunk_texts = texts[start_idx:end_idx]

            if chunk_texts:
                chunk_embeddings = await embed_function(chunk_texts)
                all_embeddings.extend(chunk_embeddings)

            start_idx = end_idx

        return all_embeddings
    else:
        # Single text is too large, chunk it by tokens
        text = texts[0]

        logger.debug("Single text too large, chunking by token limit")
        chunks = chunk_text_function(text, token_limit)

        if len(chunks) == 1:
            # Text can't be split further, raise error
            raise ValueError(
                f"Text too large to embed even after chunking: {len(text)} chars"
            )

        if single_text_fallback:
            # Return embedding of first chunk as representative
            chunk_embeddings = await embed_function([chunks[0]])
            return chunk_embeddings
        else:
            # Return embeddings for all chunks
            chunk_embeddings = await embed_function(chunks)
            return chunk_embeddings


def calculate_optimal_batch_splits(total_tokens: int, token_limit: int) -> int:
    """Calculate optimal number of batch splits for token limit.

    Args:
        total_tokens: Total estimated tokens
        token_limit: Maximum tokens per batch

    Returns:
        Number of splits needed (minimum 2)
    """
    return max(2, (total_tokens + token_limit - 1) // token_limit)


def split_texts_evenly(texts: list[str], num_splits: int) -> list[list[str]]:
    """Split texts into evenly-sized chunks.

    Args:
        texts: List of texts to split
        num_splits: Number of chunks to create

    Returns:
        List of text chunks
    """
    if num_splits <= 1:
        return [texts]

    chunk_size = len(texts) // num_splits
    remainder = len(texts) % num_splits

    chunks = []
    start_idx = 0

    for i in range(num_splits):
        # Distribute remainder across first chunks
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size
        chunk_texts = texts[start_idx:end_idx]

        if chunk_texts:
            chunks.append(chunk_texts)

        start_idx = end_idx

    return chunks


class TokenLimitHandler(Protocol):
    """Protocol for providers that can handle token limits."""

    def estimate_batch_tokens(self, texts: list[str]) -> int:
        """Estimate total tokens for a batch of texts."""
        ...

    def get_model_token_limit(self) -> int:
        """Get the model's token limit."""
        ...

    def chunk_text_by_tokens(self, text: str, token_limit: int) -> list[str]:
        """Chunk a single text by token limit."""
        ...


def with_token_limit_handling(
    error_check_func: Callable[[Exception], bool],
    safety_margin: int = 100,
    single_text_fallback: bool = True,
):
    """Decorator to add token limit error handling to embedding methods.

    Args:
        error_check_func: Function to check if exception indicates token limit exceeded
        safety_margin: Safety margin to subtract from token limit
        single_text_fallback: Whether to use first chunk as fallback for oversized single text

    Returns:
        Decorator function
    """

    def decorator(embed_func: Callable[..., Awaitable[list[list[float]]]]):
        @wraps(embed_func)
        async def wrapper(
            self: TokenLimitHandler, texts: list[str], *args, **kwargs
        ) -> list[list[float]]:
            try:
                return await embed_func(self, texts, *args, **kwargs)
            except Exception as e:
                if error_check_func(e):
                    total_tokens = self.estimate_batch_tokens(texts)
                    token_limit = self.get_model_token_limit() - safety_margin

                    return await handle_token_limit_error(
                        texts=texts,
                        total_tokens=total_tokens,
                        token_limit=token_limit,
                        embed_function=lambda batch: embed_func(
                            self, batch, *args, **kwargs
                        ),
                        chunk_text_function=self.chunk_text_by_tokens,
                        single_text_fallback=single_text_fallback,
                    )
                else:
                    raise

        return wrapper

    return decorator


def openai_token_limit_check(error: Exception) -> bool:
    """Check if error indicates OpenAI token limit exceeded."""
    try:
        import openai

        if hasattr(openai, "BadRequestError") and isinstance(
            error, openai.BadRequestError
        ):
            error_message = str(error)
            return (
                "maximum context length" in error_message and "tokens" in error_message
            )
    except ImportError:
        pass
    return False


def anthropic_token_limit_check(error: Exception) -> bool:
    """Check if error indicates Anthropic token limit exceeded."""
    # Add Anthropic-specific error checking when provider is implemented
    error_message = str(error).lower()
    return "token limit" in error_message or "context length" in error_message


def generic_token_limit_check(error: Exception) -> bool:
    """Generic token limit error checker for unknown providers."""
    error_message = str(error).lower()
    return any(
        phrase in error_message
        for phrase in [
            "token limit",
            "context length",
            "maximum tokens",
            "too many tokens",
        ]
    )


# Convenience decorators for specific providers
with_openai_token_handling = lambda: with_token_limit_handling(openai_token_limit_check)
with_anthropic_token_handling = lambda: with_token_limit_handling(
    anthropic_token_limit_check
)
with_generic_token_handling = lambda: with_token_limit_handling(
    generic_token_limit_check
)
