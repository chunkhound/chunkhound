from __future__ import annotations

from chunkhound.mcp_server.tools import limit_response_size


def test_limit_response_size_preserves_total_is_estimate_and_sets_next_offset() -> None:
    response = {
        "results": [
            {"file_path": "a.py", "content": "x" * 1000},
            {"file_path": "b.py", "content": "x" * 1000},
            {"file_path": "c.py", "content": "x" * 1000},
        ],
        "pagination": {
            "offset": 5,
            "page_size": 3,
            "has_more": False,
            "next_offset": None,
            "total": 9,
            "total_is_estimate": True,
        },
    }

    # Ensure the full response is too large, but at least one result fits.
    limited = limit_response_size(response, max_tokens=450)

    assert limited["results"], "Expected at least one result to survive truncation"
    assert limited["pagination"]["total_is_estimate"] is True
    assert limited["pagination"]["has_more"] is True
    assert limited["pagination"]["next_offset"] == 5 + len(limited["results"])


def test_limit_response_size_minimal_response_preserves_total_is_estimate() -> None:
    response = {
        "results": [{"file_path": "a.py", "content": "x" * 50000}],
        "pagination": {
            "offset": 0,
            "page_size": 1,
            "has_more": False,
            "next_offset": None,
            "total": 1,
            "total_is_estimate": True,
        },
    }

    limited = limit_response_size(response, max_tokens=1)

    assert limited["results"] == []
    assert limited["pagination"]["page_size"] == 0
    assert limited["pagination"]["next_offset"] is None
    assert limited["pagination"]["total_is_estimate"] is True
