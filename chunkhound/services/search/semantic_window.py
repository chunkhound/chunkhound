"""Shared semantic result-window behavior."""


def normalize_semantic_window_cap(value: object) -> int | None:
    """Return a valid cap; malformed capabilities preserve uncapped behavior."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def validate_semantic_window(offset: int, page_size: int, cap: object) -> None:
    """Reject an invalid page or one outside an optional semantic window."""
    if page_size <= 0:
        raise ValueError("Semantic search page_size must be greater than 0")
    if offset < 0:
        raise ValueError("Semantic search offset must be greater than or equal to 0")
    window_cap = normalize_semantic_window_cap(cap)
    if window_cap is None or offset + page_size <= window_cap:
        return
    raise ValueError(
        f"Semantic search result window [{offset}, {offset + page_size}) "
        f"crosses the exclusive [0, {window_cap}) cap; reduce page size/offset"
    )


def semantic_hybrid_fetch_size(page_size: int, offset: int, cap: object) -> int:
    """Keep hybrid's surplus fetch inside an optional semantic window."""
    window_cap = normalize_semantic_window_cap(cap)
    requested = page_size * 2
    if window_cap is not None:
        return min(requested, max(0, window_cap - offset))
    return requested


def semantic_total(exact_total: int, cap: object) -> int | None:
    """Hide totals derived from a capped, approximate result prefix."""
    return None if normalize_semantic_window_cap(cap) is not None else exact_total


def semantic_next_offset(
    offset: int, page_size: int, has_more_results: bool, cap: object
) -> int | None:
    """Return an addressable next semantic page, if one exists."""
    next_offset = offset + page_size
    window_cap = normalize_semantic_window_cap(cap)
    if has_more_results and (window_cap is None or next_offset < window_cap):
        return next_offset
    return None


def semantic_hybrid_next_offset(
    offset: int,
    page_size: int,
    merged_count: int,
    child_paginations: list[dict[str, object]],
    cap: object,
) -> int | None:
    """Continue hybrid search only when merged or child results prove surplus."""
    has_surplus = merged_count > page_size or any(
        bool(pagination.get("has_more")) for pagination in child_paginations
    )
    return semantic_next_offset(offset, page_size, has_surplus, cap)
