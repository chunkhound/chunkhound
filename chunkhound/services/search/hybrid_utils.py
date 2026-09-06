"""Shared helpers for hybrid search scaffold."""

from typing import Any

from .semantic_window import semantic_hybrid_next_offset


def finalize_hybrid_pagination(
    offset: int,
    page_size: int,
    combined: list[dict[str, Any]],
    pagination_data: dict[str, dict[str, Any]],
    cap: int | None,
) -> dict[str, Any]:
    """Build hybrid pagination dict from merged results and child paginations."""
    # Hybrid merges heterogeneous sources — total is always unknown.
    next_offset = semantic_hybrid_next_offset(
        offset, page_size, len(combined), list(pagination_data.values()), cap
    )
    return {
        "offset": offset,
        "page_size": page_size,
        "has_more": next_offset is not None,
        "next_offset": next_offset,
        "total": None,
        # Only semantic child can exhaust candidate budget.
        "candidate_budget_exhausted": bool(
            pagination_data.get("semantic", {}).get("candidate_budget_exhausted", False)
        ),
    }
