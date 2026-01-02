from __future__ import annotations

from typing import Any

from chunkhound.api.cli.commands.search import _format_search_results


class _StubFormatter:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def section_header(self, message: str) -> None:
        self.messages.append(message)

    def info(self, message: str) -> None:
        self.messages.append(message)


def test_search_cli_renders_estimated_total_with_marker() -> None:
    formatter = _StubFormatter()
    result: dict[str, Any] = {
        "results": [
            {
                "file_path": "src/a.py",
                "content": "",
            }
        ],
        "pagination": {
            "offset": 0,
            "page_size": 1,
            "has_more": True,
            "total": 10,
            "total_is_estimate": True,
        },
    }

    _format_search_results(formatter, result, query="needle", is_regex=False)

    assert any("Results:" in msg for msg in formatter.messages)
    assert any("of ≈10" in msg for msg in formatter.messages)


def test_search_cli_renders_exact_total_without_marker() -> None:
    formatter = _StubFormatter()
    result: dict[str, Any] = {
        "results": [
            {
                "file_path": "src/a.py",
                "content": "",
            }
        ],
        "pagination": {
            "offset": 0,
            "page_size": 1,
            "has_more": True,
            "total": 10,
        },
    }

    _format_search_results(formatter, result, query="needle", is_regex=False)

    assert any("of 10" in msg for msg in formatter.messages)
    assert not any("of ≈10" in msg for msg in formatter.messages)

