"""Tests for MCP search markdown formatter and execute_tool str return."""
import pytest
from unittest.mock import AsyncMock, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result(
    file_path: str = "src/auth/login.py",
    start_line: int = 10,
    end_line: int = 20,
    content: str = "def my_method(self):\n    pass",
    symbol: str | None = "MyClass.my_method",
    name: str | None = None,
    similarity: float | None = 0.92,
    language: str = "python",
    **extra: object,
) -> dict:
    return dict(
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        content=content,
        symbol=symbol,
        name=name,
        similarity=similarity,
        # fields that must NOT appear in markdown output:
        chunk_id=42,
        chunk_type="function",
        language=language,
        metadata={"raw_content": content},
        file_extension=".py",
        line_count=end_line - start_line + 1,
        code_preview=content[:200],
        is_truncated=False,
        similarity_percentage=round((similarity or 0) * 100, 2),
        **extra,
    )


def _pagination(
    offset: int = 0,
    page_size: int = 10,
    has_more: bool = True,
    total: int = 47,
    next_offset: int = 10,
) -> dict:
    return dict(
        offset=offset, page_size=page_size, has_more=has_more,
        total=total, next_offset=next_offset,
    )


# ---------------------------------------------------------------------------
# Tests for format_search_results_markdown
# ---------------------------------------------------------------------------

class TestFormatSearchResultsMarkdown:

    def _fmt(self, results: list, pagination: dict | None = None, search_type: str = "regex") -> str:
        from chunkhound.mcp_server.tools import format_search_results_markdown
        return format_search_results_markdown(
            results, pagination or _pagination(), search_type
        )

    def test_heading_contains_file_path(self) -> None:
        md = self._fmt([_result()])
        assert "src/auth/login.py" in md

    def test_heading_contains_line_range(self) -> None:
        md = self._fmt([_result(start_line=10, end_line=20)])
        assert "L10" in md
        assert "L20" in md

    def test_heading_contains_symbol(self) -> None:
        md = self._fmt([_result(symbol="MyClass.my_method")])
        assert "MyClass.my_method" in md

    def test_heading_uses_name_when_symbol_absent(self) -> None:
        md = self._fmt([_result(symbol=None, name="my_func")])
        assert "my_func" in md

    def test_similarity_shown_for_semantic(self) -> None:
        md = self._fmt([_result(similarity=0.92)], search_type="semantic")
        assert "92%" in md

    def test_similarity_hidden_for_regex(self) -> None:
        import re
        md = self._fmt([_result()], search_type="regex")
        assert not re.search(r'\(\d+%\)', md), "Similarity percentage should not appear in regex results"

    def test_code_block_contains_content(self) -> None:
        md = self._fmt([_result(content="def my_method(self):\n    pass")])
        assert "def my_method" in md
        assert "```" in md

    def test_python_fence_hint(self) -> None:
        md = self._fmt([_result(file_path="src/auth.py")])
        assert "```python" in md

    def test_go_fence_hint(self) -> None:
        md = self._fmt([_result(file_path="main.go", language="go", symbol=None, name=None)])
        assert "```go" in md

    def test_unknown_ext_fence_hint_is_empty(self) -> None:
        md = self._fmt([_result(file_path="Makefile", language="unknown", symbol=None, name=None)])
        assert "```\n" in md

    def test_pagination_footer_shows_totals(self) -> None:
        md = self._fmt([_result()], pagination=_pagination(total=47, next_offset=10))
        assert "47" in md
        assert "next_offset=10" in md

    def test_pagination_no_next_when_no_more(self) -> None:
        md = self._fmt(
            [_result()],
            pagination=_pagination(has_more=False, next_offset=None),
        )
        assert "next_offset" not in md

    def test_empty_results_returns_no_results_message(self) -> None:
        md = self._fmt([])
        assert "No results" in md

    def test_dropped_fields_absent(self) -> None:
        md = self._fmt([_result()])
        for field in (
            "chunk_id", "chunk_type", "file_extension", "metadata",
            "is_truncated", "code_preview", "line_count", "similarity_percentage",
        ):
            assert field not in md, f"Dropped field '{field}' leaked into markdown"

    def test_multiple_results_have_separators(self) -> None:
        md = self._fmt([_result(), _result(start_line=30, end_line=40)])
        assert md.count("---") >= 2

    def test_shorter_than_equivalent_json(self) -> None:
        import json
        results = [_result() for _ in range(5)]
        md = self._fmt(results)
        json_str = json.dumps({"results": results, "pagination": _pagination()})
        assert len(md) < len(json_str), (
            f"Markdown ({len(md)} chars) not shorter than JSON ({len(json_str)} chars)"
        )

    def test_symbol_none_no_dangling_separator(self) -> None:
        md = self._fmt([_result(symbol=None, name=None)])
        assert " — None" not in md

    def test_content_with_triple_backticks_produces_valid_fence(self) -> None:
        """Content containing unindented ``` must not prematurely close the fence."""
        # Unindented triple-backticks at column 0 close a CommonMark fence.
        content_with_backticks = "before\n```\nafter"
        md = self._fmt([_result(content=content_with_backticks)])
        # The content must appear intact inside one balanced fence pair.
        assert "before" in md
        assert "after" in md
        # Confirm the outer fence is longer than 3 backticks so the inner ```
        # cannot close it.
        import re
        opening = re.search(r"^(`{3,})\w*$", md, re.MULTILINE)
        assert opening is not None, "No opening fence found"
        fence = opening.group(1)
        assert len(fence) > 3, (
            f"Fence is only {len(fence)} backticks — inner ``` will close it prematurely"
        )


# ---------------------------------------------------------------------------
# Tests for execute_tool returning str for search
# ---------------------------------------------------------------------------

class TestExecuteToolSearchReturnsMarkdown:

    def _make_services(self, results: list, pagination: dict) -> MagicMock:
        services = MagicMock()
        services.search_service = MagicMock()
        services.search_service.search_regex_async = AsyncMock(
            return_value=(results, pagination)
        )
        services.search_service.search_semantic = AsyncMock(
            return_value=(results, pagination)
        )
        return services

    async def test_regex_search_returns_str(self) -> None:
        from chunkhound.mcp_server.tools import execute_tool

        svc = self._make_services(
            [_result(file_path="auth.py", symbol="login", similarity=None)],
            _pagination(has_more=False, next_offset=None, total=1),
        )
        result = await execute_tool(
            tool_name="search",
            services=svc,
            embedding_manager=None,
            arguments={"type": "regex", "query": "def login"},
        )
        assert isinstance(result, str), f"Expected str, got {type(result)}"
        assert "auth.py" in result
        assert "```" in result

    async def test_no_dropped_fields_in_mcp_output(self) -> None:
        from chunkhound.mcp_server.tools import execute_tool

        svc = self._make_services(
            [_result()],
            _pagination(has_more=False, next_offset=None, total=1),
        )
        result = await execute_tool(
            tool_name="search",
            services=svc,
            embedding_manager=None,
            arguments={"type": "regex", "query": "def my_method"},
        )
        for field in ("chunk_id", "chunk_type", "\"language\"",
                      "file_extension", "is_truncated", "code_preview",
                      "line_count", "similarity_percentage"):
            assert field not in result, f"Dropped field '{field}' leaked into MCP output"

    async def test_trim_loop_reduces_oversized_results(self) -> None:
        """Token-limiting trim loop in execute_tool removes results until under MAX_RESPONSE_TOKENS.

        10 results × 7 000-char content ≈ 70 000 chars → ~23 300 tokens > MAX_RESPONSE_TOKENS
        (20 000).  The loop trims by 1/4 each pass; after one pass 8 results remain (~18 700
        tokens < limit).  The response must reflect the trim: fewer blocks and next_offset set.
        """
        from chunkhound.mcp_server.tools import execute_tool

        large_content = "x" * 7000
        results = [
            _result(file_path=f"file_{i}.py", content=large_content, symbol=f"func_{i}")
            for i in range(10)
        ]
        svc = self._make_services(results, _pagination(has_more=False, next_offset=None, total=10))

        result = await execute_tool(
            tool_name="search",
            services=svc,
            embedding_manager=None,
            arguments={"type": "regex", "query": "x"},
        )

        assert isinstance(result, str)
        result_block_count = result.count("## `")
        assert result_block_count < 10, (
            f"Trim loop should have removed results; found {result_block_count} blocks"
        )
        assert "next_offset=" in result, "Trimmed response must set next_offset for the caller to page"
