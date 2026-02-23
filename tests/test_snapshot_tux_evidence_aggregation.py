#!/usr/bin/env python3
"""Evidence aggregation tests for the snapshot chunk-systems TUI (items.jsonl + fallback)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from chunkhound.snapshot.chunk_systems_tui import (
    build_system_evidence_fallback,
    build_system_evidence_from_items,
)


def test_snapshot_tux_evidence_from_items_jsonl_system_group_all(tmp_path: Path) -> None:
    systems_ordered = [101, 202]

    lines = [
        # Out-of-order chunk_ids to ensure we sort deterministically.
        {
            "chunk_id": 2,
            "cluster_id": 101,
            "path": "pkg/a.py",
            "symbol": "bar",
            "chunk_type": "function",
            "start_line": 10,
            "end_line": 20,
        },
        {
            "chunk_id": 1,
            "cluster_id": 101,
            "path": "pkg/a.py",
            "symbol": "foo",
            "chunk_type": "function",
            "start_line": 1,
            "end_line": 9,
        },
        {
            "chunk_id": 3,
            "cluster_id": 202,
            "path": "pkg/a.py",
            "symbol": None,
            "chunk_type": "class",
            "start_line": 1,
            "end_line": 50,
        },
        # Should be ignored (cluster not in systems_ordered)
        {
            "chunk_id": 4,
            "cluster_id": 999,
            "path": "pkg/ignored.py",
            "symbol": "nope",
            "chunk_type": "function",
            "start_line": 1,
            "end_line": 2,
        },
    ]

    items_path = tmp_path / "snapshot.chunk_systems.items.jsonl"
    items_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in lines) + "\n",
        encoding="utf-8",
    )
    evidence_by_id = build_system_evidence_from_items(
        items_jsonl_path=items_path, systems_ordered=systems_ordered
    )

    ev_101 = evidence_by_id[101]
    assert ev_101.file_counts["pkg/a.py"] == 2
    assert ev_101.symbol_counts["foo"] == 1
    assert ev_101.symbol_counts["bar"] == 1
    assert ev_101.chunks is not None
    assert [c.chunk_id for c in ev_101.chunks] == [1, 2]

    ev_202 = evidence_by_id[202]
    assert ev_202.file_counts["pkg/a.py"] == 1
    assert ev_202.symbol_counts[None] == 1
    assert ev_202.chunks is not None
    assert [c.chunk_id for c in ev_202.chunks] == [3]

    # Group/all scope aggregation (union over systems).
    group_file_counts: Counter[str] = Counter()
    group_symbol_counts: Counter[str | None] = Counter()
    for cid in systems_ordered:
        group_file_counts.update(evidence_by_id[cid].file_counts)
        group_symbol_counts.update(evidence_by_id[cid].symbol_counts)

    assert group_file_counts["pkg/a.py"] == 3
    assert group_symbol_counts["foo"] == 1
    assert group_symbol_counts["bar"] == 1
    assert group_symbol_counts[None] == 1


def test_snapshot_tux_evidence_fallback_uses_top_files_and_symbols() -> None:
    clusters_by_id = {
        101: {
            "top_files": [
                {"path": "pkg\\a.py", "count": 3},
                {"path": "tests/test_x.py", "count": 1},
            ],
            "top_symbols": [{"symbol": "foo", "count": 2}, {"symbol": None, "count": 1}],
        }
    }
    systems_ordered = [101, 202]

    evidence_by_id = build_system_evidence_fallback(
        clusters_by_id=clusters_by_id,
        systems_ordered=systems_ordered,
    )

    ev_101 = evidence_by_id[101]
    assert ev_101.file_counts["pkg/a.py"] == 3
    assert ev_101.file_counts["tests/test_x.py"] == 1
    assert ev_101.symbol_counts["foo"] == 2
    assert ev_101.symbol_counts[None] == 1
    assert ev_101.chunks is None

    ev_202 = evidence_by_id[202]
    assert ev_202.file_counts == Counter()
    assert ev_202.symbol_counts == Counter()
    assert ev_202.chunks is None
