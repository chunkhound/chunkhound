"""Token measurement: JSON vs markdown search responses.

Regression guard: asserts markdown is at least 40% smaller than equivalent JSON.
Pass -s to see the printed breakdown tables:
  uv run pytest tests/test_token_measurement.py -v -s
"""
import json
import pytest


# ---------------------------------------------------------------------------
# Realistic mock result — mirrors what result_enhancer produces
# ---------------------------------------------------------------------------

_REALISTIC_CONTENT = '''def authenticate_user(username: str, password: str) -> Optional[User]:
    """Authenticate a user with username and password."""
    if not username or not password:
        raise ValueError("Username and password are required")

    user = db.query(User).filter(User.username == username).first()
    if user is None:
        return None

    if not check_password_hash(user.password_hash, password):
        return None

    user.last_login = datetime.utcnow()
    db.commit()
    return user'''


def _make_result(i: int = 0) -> dict:
    """Realistic result as returned by result_enhancer.enhance_search_result()."""
    return {
        "chunk_id": 1000 + i,
        "symbol": f"auth.login.authenticate_user",
        "name": f"authenticate_user",
        "content": _REALISTIC_CONTENT,
        "code_preview": _REALISTIC_CONTENT[:500],     # partial duplicate
        "chunk_type": "function",
        "start_line": 10 + i * 30,
        "end_line": 28 + i * 30,
        "line_count": 19,
        "file_path": f"src/auth/login.py",
        "file_extension": ".py",
        "language": "python",
        "similarity": 0.87,
        "similarity_percentage": 87.0,
        "metadata": {"raw_content": _REALISTIC_CONTENT},  # full duplicate
        "is_truncated": False,
    }


def _make_pagination(n: int) -> dict:
    return {"offset": 0, "page_size": n, "has_more": True, "total": 47, "next_offset": n}


def _estimate_tokens(text: str) -> int:
    """Same heuristic used by tools.py: len // 3."""
    return len(text) // 3


# ---------------------------------------------------------------------------
# Measurement tests (require -s to see output)
# ---------------------------------------------------------------------------

class TestTokenReduction:

    @pytest.mark.parametrize("n_results", [1, 3, 5, 10])
    def test_measure_reduction_by_result_count(self, n_results: int, capsys: pytest.CaptureFixture[str]) -> None:
        """Compare JSON vs markdown tokens for N results. Fails if reduction < 40%."""
        # NOTE: before the formatter is implemented this test will ImportError — expected.
        # Run it after the formatter is done to get real post-change numbers.
        from chunkhound.mcp_server.tools import format_search_results_markdown

        results = [_make_result(i) for i in range(n_results)]
        pagination = _make_pagination(n_results)

        json_str = json.dumps({"results": results, "pagination": pagination}, default=str)
        json_tokens = _estimate_tokens(json_str)

        md_str = format_search_results_markdown(results, pagination, "semantic")
        md_tokens = _estimate_tokens(md_str)

        reduction_pct = (1 - md_tokens / json_tokens) * 100

        with capsys.disabled():
            print(f"\n{'-'*52}")
            print(f"  Results:   {n_results}")
            print(f"  JSON:      {json_tokens:>6} tokens  ({len(json_str):>7} chars)")
            print(f"  Markdown:  {md_tokens:>6} tokens  ({len(md_str):>7} chars)")
            print(f"  Reduction: {reduction_pct:>5.1f}%")
            print(f"{'-'*52}")

        # 40 % threshold is calibrated to a realistic multi-field result that carries
        # metadata.raw_content (full duplicate) and code_preview (partial duplicate).
        # For minimal single-line content these duplicates dominate overhead, so the
        # threshold remains conservative even for small functions.
        assert reduction_pct >= 40.0, (
            f"Reduction {reduction_pct:.1f}% is below the 40% minimum. "
            f"JSON: {json_tokens} tok, Markdown: {md_tokens} tok"
        )

    def test_field_overhead_breakdown(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Print token cost of every field that will be dropped."""
        result = _make_result()

        dropped = [
            "chunk_id", "chunk_type", "language", "file_extension",
            "line_count", "is_truncated", "similarity_percentage",
            "code_preview", "metadata",
        ]

        total_dropped = 0
        with capsys.disabled():
            print(f"\n{'-'*52}")
            print(f"  {'Field':<30} {'Tokens':>6}")
            print(f"{'-'*52}")
            for field in dropped:
                if field in result:
                    tokens = _estimate_tokens(json.dumps({field: result[field]}))
                    total_dropped += tokens
                    print(f"  {field:<30} {tokens:>6}")
            total_result = _estimate_tokens(json.dumps(result))
            print(f"{'-'*52}")
            print(f"  {'TOTAL dropped':<30} {total_dropped:>6}")
            print(f"  {'TOTAL result (JSON)':<30} {total_result:>6}")
            print(f"  {'Dropped %':<30} {total_dropped/total_result*100:>5.1f}%")
            print(f"{'-'*52}")

        total_result = _estimate_tokens(json.dumps(result))
        assert total_dropped > 0, "Dropped fields must have non-zero token cost"
        assert total_dropped / total_result > 0.5, (
            f"Dropped fields should account for >50% of tokens, got {total_dropped/total_result:.1%}"
        )

    def test_metadata_duplication_overhead(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Show how much metadata.raw_content duplicates the content field."""
        result = _make_result()
        content_tokens = _estimate_tokens(result["content"])
        metadata_tokens = _estimate_tokens(json.dumps(result["metadata"]))
        code_preview_tokens = _estimate_tokens(json.dumps(result["code_preview"]))

        with capsys.disabled():
            print(f"\n{'-'*52}")
            print(f"  content tokens:            {content_tokens:>6}")
            print(f"  metadata.raw_content:      {metadata_tokens:>6}  (full duplicate)")
            print(f"  code_preview:              {code_preview_tokens:>6}  (partial duplicate)")
            print(f"  Total duplication:         {metadata_tokens + code_preview_tokens:>6} tokens")
            print(f"{'-'*52}")

        assert metadata_tokens >= content_tokens, (
            "metadata.raw_content should be at least as large as content (it's a full copy)"
        )
        assert code_preview_tokens > 0, "code_preview must have non-zero token cost"
