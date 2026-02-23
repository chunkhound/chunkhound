#!/usr/bin/env python3
"""Unit tests for token-bounded deep LLM labeling helpers."""

from __future__ import annotations

from pathlib import Path

import duckdb

import chunkhound.api.cli.commands.snapshot as snapshot_mod

from chunkhound.api.cli.commands.snapshot import (
    _ChunkNameHints,
    _SnapshotItem,
    _cluster_is_test_heavy,
    _cluster_is_test_only,
    _cluster_is_cryptic,
    _infer_python_enclosing_class,
    _is_cryptic_symbol,
    _label_is_generic,
    _normalize_tests_prefix,
    _pack_chunk_code_excerpts,
    _read_text_lines_best_effort,
)


def test_is_cryptic_symbol_dunder_and_unknowns() -> None:
    assert _is_cryptic_symbol("__repr__") is True
    assert _is_cryptic_symbol("__str__") is True
    assert _is_cryptic_symbol("__init__") is True
    assert _is_cryptic_symbol("") is True
    assert _is_cryptic_symbol("<unknown>") is True
    assert _is_cryptic_symbol("a") is True

    assert _is_cryptic_symbol("api_key_display") is False
    assert _is_cryptic_symbol("render_chunk_systems_viz_html") is False


def test_cluster_is_cryptic_from_top_symbols_ratio() -> None:
    cryptic = {
        "cluster_id": 20,
        "size": 9,
        "top_symbols": [
            {"symbol": "__repr__", "count": 6},
            {"symbol": "api_key_display", "count": 2},
            {"symbol": "__str__", "count": 1},
        ],
    }
    assert _cluster_is_cryptic(cluster=cryptic) is True

    non_cryptic = {
        "cluster_id": 1,
        "size": 5,
        "top_symbols": [
            {"symbol": "render_chunk_systems_viz_html", "count": 3},
            {"symbol": "build_system_adjacency_json", "count": 2},
        ],
    }
    assert _cluster_is_cryptic(cluster=non_cryptic) is False


def test_label_is_generic_dunder_and_pathy() -> None:
    assert _label_is_generic("Config __repr__/__str__ methods") is True
    assert _label_is_generic("misc") is True
    assert _label_is_generic("foo/bar.py config helpers") is True

    assert _label_is_generic("Config redaction / summary strings") is False


def test_pack_chunk_code_excerpts_respects_budget_and_is_deterministic() -> None:
    item_by_chunk_id = {
        1: _SnapshotItem(
            chunk_id=1,
            path="pkg/a.py",
            symbol="api_key_display",
            chunk_type="function",
            start_line=1,
            end_line=20,
        ),
        2: _SnapshotItem(
            chunk_id=2,
            path="pkg/a.py",
            symbol="__repr__",
            chunk_type="method",
            start_line=30,
            end_line=40,
        ),
        3: _SnapshotItem(
            chunk_id=3,
            path="pkg/b.py",
            symbol="make_client",
            chunk_type="function",
            start_line=1,
            end_line=120,
        ),
    }
    chunk_hints_by_id = {
        1: _ChunkNameHints(symbol="api_key_display", signature_line=None, identifier_tokens=["api", "key"]),
        2: _ChunkNameHints(symbol="__repr__", signature_line=None, identifier_tokens=[]),
        3: _ChunkNameHints(symbol="make_client", signature_line=None, identifier_tokens=["make", "client"]),
    }

    big_code = " ".join(["alpha"] * 600)
    code_by_chunk_id = {1: big_code, 2: big_code, 3: big_code}

    cluster = {
        "cluster_id": 1,
        "size": 3,
        "chunk_ids": [1, 2, 3],
        "top_files": [{"path": "pkg/a.py", "count": 2}, {"path": "pkg/b.py", "count": 1}],
    }

    estimate_tokens = lambda s: len(str(s).split())

    lines1 = _pack_chunk_code_excerpts(
        cluster=cluster,
        item_by_chunk_id=item_by_chunk_id,
        chunk_hints_by_id=chunk_hints_by_id,
        code_by_chunk_id=code_by_chunk_id,
        max_content_tokens=200,
        estimate_tokens=estimate_tokens,
    )
    text1 = "\n".join(lines1)
    assert estimate_tokens(text1) <= 200
    assert "```" in text1

    lines2 = _pack_chunk_code_excerpts(
        cluster=cluster,
        item_by_chunk_id=item_by_chunk_id,
        chunk_hints_by_id=chunk_hints_by_id,
        code_by_chunk_id=code_by_chunk_id,
        max_content_tokens=200,
        estimate_tokens=estimate_tokens,
    )
    assert lines1 == lines2


def test_disk_assist_best_effort_missing_file_does_not_fail(tmp_path: Path) -> None:
    missing = tmp_path / "nope.py"
    assert _read_text_lines_best_effort(abs_path=missing) is None


def test_infer_python_enclosing_class(tmp_path: Path) -> None:
    p = tmp_path / "x.py"
    p.write_text(
        "class Foo:\n"
        "    def __repr__(self) -> str:\n"
        "        return 'x'\n",
        encoding="utf-8",
    )
    lines = _read_text_lines_best_effort(abs_path=p)
    assert isinstance(lines, list)
    cls = _infer_python_enclosing_class(file_lines=lines, start_line=2)
    assert cls == "Foo"


def test_cluster_is_test_only_from_top_files() -> None:
    cluster = {
        "cluster_id": 1,
        "size": 100,
        "top_files": [
            {"path": "chunkhound/tests/test_a.py", "count": 95},
            {"path": "chunkhound/tests/test_b.py", "count": 5},
        ],
    }
    assert _cluster_is_test_only(cluster) is True

    mixed = {
        "cluster_id": 2,
        "size": 100,
        "top_files": [
            {"path": "chunkhound/tests/test_a.py", "count": 80},
            {"path": "chunkhound/chunkhound/core/x.py", "count": 20},
        ],
    }
    assert _cluster_is_test_only(mixed) is False


def test_cluster_is_test_heavy_from_top_files() -> None:
    cluster = {
        "cluster_id": 3,
        "size": 100,
        "top_files": [
            {"path": "chunkhound/tests/test_a.py", "count": 90},
            {"path": "chunkhound/chunkhound/core/x.py", "count": 10},
        ],
    }
    assert _cluster_is_test_heavy(cluster) is True
    assert _cluster_is_test_only(cluster) is False


def test_normalize_tests_prefix() -> None:
    assert _normalize_tests_prefix("Tests: foo") == "Tests: foo"
    assert _normalize_tests_prefix("Testing class parsing") == "Tests: class parsing"
    assert _normalize_tests_prefix("Test suite: embedding storage") == "Tests: embedding storage"


async def test_label_chunk_systems_deep_falls_back_to_disk_on_duckdb_lock(
    tmp_path: Path, monkeypatch
) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "a.py").write_text(
        "def foo() -> int:\n"
        "    return 1\n"
        "\n"
        "def bar() -> int:\n"
        "    return 2\n",
        encoding="utf-8",
    )

    item_by_chunk_id = {
        1: _SnapshotItem(
            chunk_id=1,
            path="pkg/a.py",
            symbol="foo",
            chunk_type="function",
            start_line=1,
            end_line=2,
        ),
        2: _SnapshotItem(
            chunk_id=2,
            path="pkg/a.py",
            symbol="bar",
            chunk_type="function",
            start_line=4,
            end_line=5,
        ),
    }
    chunk_hints_by_id = {
        1: _ChunkNameHints(symbol="foo", signature_line=None, identifier_tokens=["foo"]),
        2: _ChunkNameHints(symbol="bar", signature_line=None, identifier_tokens=["bar"]),
    }

    cluster = {
        "cluster_id": 1,
        "size": 2,
        "chunk_ids": [1, 2],
        "top_files": [{"path": "pkg/a.py", "count": 2}],
        "top_symbols": [{"symbol": "foo", "count": 1}, {"symbol": "bar", "count": 1}],
    }

    def _boom(*_args, **_kwargs):
        raise duckdb.IOException("locked")

    monkeypatch.setattr(snapshot_mod, "_read_chunk_code_map", _boom)

    labels, llm_calls = await snapshot_mod._label_chunk_systems_llm_one_pass(
        clusters=[cluster],
        out_dir=tmp_path,
        llm_manager=None,
        llm_dry_run=True,
        item_by_chunk_id=item_by_chunk_id,
        chunk_hints_by_id=chunk_hints_by_id,
        llm_label_batching=False,
        llm_label_batch_max_items=1,
        llm_label_batch_max_tokens=10_000,
        db_file=tmp_path / "locked.duckdb",
        base_directory=tmp_path,
        scope_roots=["."],
        deep_cluster_ids={1},
        prompt_suffix="",
        batch_prefix="llm_chunk_system_batch",
        estimate_tokens=lambda s: len(str(s).split()),
        progress=None,
        progress_task_id=None,
        reset_progress=True,
    )

    assert llm_calls == 0
    assert [label.label for label in labels] == ["CHUNK_SYSTEM_1"]

    prompt_text = (tmp_path / "llm_chunk_system_1.md").read_text(encoding="utf-8")
    assert "Code excerpts" in prompt_text
    assert "```" in prompt_text
    assert "def foo" in prompt_text
