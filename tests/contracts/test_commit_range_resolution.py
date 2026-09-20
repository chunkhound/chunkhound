"""Contract: _resolve_commit_range maps mutually exclusive inputs correctly.

The MCP search and code_research tools accept three mutually exclusive
commit parameters: commit_range, commit_hash, last_n_commits.
_resolve_commit_range converts them to a single git revision range string.
This contract protects the user-facing semantics of git diff search.
"""

import pytest

from chunkhound.mcp_server.tools import _resolve_commit_range


class TestResolveCommitRange:
    """_resolve_commit_range must enforce mutual exclusivity and correct format."""

    def test_commit_hash_to_parent_range(self):
        result = _resolve_commit_range(
            commit_range=None, commit_hash="abc123", last_n_commits=None
        )
        assert result == "abc123^..abc123"

    def test_last_n_commits_to_head_range(self):
        result = _resolve_commit_range(
            commit_range=None, commit_hash=None, last_n_commits=5
        )
        assert result == "HEAD~5..HEAD"

    def test_commit_range_passed_through(self):
        result = _resolve_commit_range(
            commit_range="main..feature", commit_hash=None, last_n_commits=None
        )
        assert result == "main..feature"

    def test_all_none_returns_none(self):
        result = _resolve_commit_range(
            commit_range=None, commit_hash=None, last_n_commits=None
        )
        assert result is None

    def test_multiple_inputs_raises_valueerror(self):
        with pytest.raises(ValueError, match="at most one"):
            _resolve_commit_range(
                commit_range="main..dev",
                commit_hash="abc123",
                last_n_commits=None,
            )

    def test_all_three_inputs_raises_valueerror(self):
        with pytest.raises(ValueError, match="at most one"):
            _resolve_commit_range(
                commit_range="main..dev",
                commit_hash="abc123",
                last_n_commits=3,
            )
