from __future__ import annotations


def safe_scope_label(scope_label: str) -> str:
    """Normalize a scope label for use in filenames."""
    safe_scope = scope_label.replace("/", "_")
    return safe_scope or "root"


def slugify_kebab(
    text: str,
    *,
    fallback: str = "topic",
    max_length: int | None = None,
    ascii_only: bool = False,
) -> str:
    """Convert text into a lowercase dash-separated slug.

    Normalization:
    - Lowercases input.
    - Replaces any sequence of non-alphanumerics with a single dash.
    - Trims leading/trailing dashes.
    - Uses `fallback` when the slug would be empty.

    Args:
        text: Input text to normalize.
        fallback: Slug to use when the normalized result is empty.
        max_length: Optional maximum slug length.
        ascii_only: When True, only ASCII letters/digits are preserved.

    Returns:
        A filesystem-friendly slug string.
    """
    normalized = text.strip().lower()
    slug_chars: list[str] = []
    prev_dash = False
    for ch in normalized:
        if ch.isalnum() and (not ascii_only or ch.isascii()):
            slug_chars.append(ch)
            prev_dash = False
            continue
        if not prev_dash:
            slug_chars.append("-")
            prev_dash = True

    slug = "".join(slug_chars).strip("-")
    if not slug:
        slug = fallback

    if max_length is not None and len(slug) > max_length:
        slug = slug[:max_length].rstrip("-")
        if not slug:
            slug = fallback

    return slug
