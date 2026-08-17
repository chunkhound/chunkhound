"""Public-contract unit tests for the remote-config rules module.

Scope kept narrow per AGENTS.md `TESTING_PHILOSOPHY`: only the two public
contracts the pipeline (and reviewers) depend on directly are locked here.

- ENCLOSING_SUB_MODEL: the path→BaseModel map. Its correctness is invisible
  from higher-level tests, but the `X | None` unwrapping bug it protects
  against would silently break per-rule sub-model validation for
  ``embedding.*``, ``llm.*``, and ``remote_config.*``.
- ``parse_path``: user-facing rule authoring accepts multiple syntaxes;
  a regression here silently changes rule matching.
"""

from __future__ import annotations

import pytest

from chunkhound.core.config.config import Config
from chunkhound.core.config.embedding_config import EmbeddingConfig
from chunkhound.core.config.llm_config import LLMConfig
from chunkhound.core.config.mcp_config import MCPConfig
from chunkhound.core.config.remote.rules import (
    ENCLOSING_SUB_MODEL,
    _predicate_matches,
    parse_path,
)
from chunkhound.core.config.remote_config import RemoteConfig


class TestEnclosingSubModelMap:
    def test_optional_fields_unwrap_to_basemodel(self) -> None:
        # The bug-prone cases: `X | None` must resolve to `X`, not `UnionType`.
        assert ENCLOSING_SUB_MODEL["embedding"] is EmbeddingConfig
        assert ENCLOSING_SUB_MODEL["llm"] is LLMConfig
        assert ENCLOSING_SUB_MODEL["remote_config"] is RemoteConfig

    def test_required_field_binds_directly(self) -> None:
        assert ENCLOSING_SUB_MODEL["mcp"] is MCPConfig

    def test_scalar_fields_omitted(self) -> None:
        # `debug: bool` etc. are not BaseModels and must not appear here.
        assert "debug" not in ENCLOSING_SUB_MODEL
        assert "target_dir" not in ENCLOSING_SUB_MODEL


class TestParsePath:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("a.b.c", ["a", "b", "c"]),
            ("a/b/c", ["a", "b", "c"]),
            (".a.b", ["a", "b"]),
            ("/a/b", ["a", "b"]),
            ("mcp.host", ["mcp", "host"]),
            ("embedding.api_key", ["embedding", "api_key"]),
        ],
    )
    def test_accepted_forms(self, text: str, expected: list[str]) -> None:
        assert parse_path(text) == expected

    @pytest.mark.parametrize("bad", ["", "..a", "a..b", "/", "."])
    def test_rejects_empty_or_double_separators(self, bad: str) -> None:
        with pytest.raises(ValueError):
            parse_path(bad)


class TestPredicateMatches:
    def test_none_predicates_match(self) -> None:
        assert _predicate_matches(None, Config()) == (True, None)

    def test_unknown_key_returns_schema_error(self) -> None:
        # Typo like "wehn" or "existng" must produce an operator-visible
        # signal, not a silent skip — see module docstring rationale.
        matches, err = _predicate_matches({"wehn": "linux"}, Config())
        assert matches is False
        assert err is not None
        assert "wehn" in err

    def test_os_mismatch_is_silent_miss(self) -> None:
        matches, err = _predicate_matches(
            {"os": "definitely-not-a-real-platform"}, Config()
        )
        assert (matches, err) == (False, None)
