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

import typing

import pytest
from pydantic import BaseModel

from chunkhound.core.config.config import Config
from chunkhound.core.config.embedding_config import EmbeddingConfig
from chunkhound.core.config.llm_config import LLMConfig
from chunkhound.core.config.mcp_config import MCPConfig
from chunkhound.core.config.remote.rules import (
    ENCLOSING_SUB_MODEL,
    _predicate_matches,
    _unwrap_optional_basemodel,
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


def test_config_tree_is_depth_2() -> None:
    """Tripwire for the depth-2 assumption in ``apply_rule``'s payload check.

    ``apply_rule`` validates ``segments[1]`` against the sub-model's fields
    and ``_check_payload_keys`` walks payload dicts one level via
    ``_unwrap_optional_basemodel``. Every sub-config in the real tree
    stops at scalar leaves today, so the walk only ever reaches depth 2
    through real config.

    If a sub-config gains a nested ``BaseModel`` field this test fails.
    Before relaxing the assertion:
    1. Extend the walker in ``apply_rule`` / ``_check_payload_keys`` to
       validate segments at the new depth — an unknown key beyond depth 2
       would otherwise be silently absorbed by ``extra="ignore"``.
    2. Add a pipeline_e2e test exercising a rule like
       ``id=<sub>.<nested>.<leaf>`` end-to-end.
    """
    violations: list[str] = []
    for top_name, top_info in Config.model_fields.items():
        top_model = _unwrap_optional_basemodel(top_info.annotation)
        if top_model is None:
            continue  # scalar top-level field — no sub-tree
        for nested_name, nested_info in top_model.model_fields.items():
            nested_model = _unwrap_optional_basemodel(nested_info.annotation)
            if nested_model is not None:
                violations.append(
                    f"{top_name}.{nested_name} → nested {nested_model.__name__}"
                )

    assert not violations, (
        "Config tree gained BaseModel-typed fields under a sub-config — "
        f"the depth-2 walker in remote/rules.py must be extended: "
        f"{violations}. See test docstring for the checklist."
    )


def test_config_has_no_ambiguous_basemodel_unions() -> None:
    """Tripwire for ambiguous ``BaseModel`` unions in the Config tree.

    ``_unwrap_optional_basemodel`` and ``_build_enclosing_map`` both pick
    the *first* ``BaseModel`` candidate in a union. That's safe only while
    no field is typed as ``A | B`` where both are ``BaseModel`` subclasses
    — if one appears, per-rule sub-model validation silently binds to one
    schema and ignores the other. If this test fails, extend the walker
    in ``remote/rules.py`` to refuse ambiguous unions rather than
    silently picking a candidate.
    """
    violations: list[str] = []

    def walk(model: type[BaseModel], prefix: str) -> None:
        for name, info in model.model_fields.items():
            candidates = typing.get_args(info.annotation) or (info.annotation,)
            models = [
                c
                for c in candidates
                if isinstance(c, type) and issubclass(c, BaseModel)
            ]
            if len(models) > 1:
                violations.append(
                    f"{prefix}{name}: {[m.__name__ for m in models]}"
                )
            elif len(models) == 1:
                walk(models[0], f"{prefix}{name}.")

    walk(Config, "")

    assert not violations, (
        "Config tree gained an ambiguous BaseModel union — the walker in "
        "remote/rules.py silently picks the first candidate and would "
        "validate rules against the wrong schema. Extend the walker to "
        f"reject rather than pick: {violations}"
    )
