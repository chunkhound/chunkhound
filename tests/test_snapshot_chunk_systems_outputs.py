#!/usr/bin/env python3
"""Unit tests for optional snapshot chunk-systems outputs (graph + adjacency)."""

from __future__ import annotations

from dataclasses import dataclass

from chunkhound.snapshot.chunk_systems_outputs import (
    build_system_adjacency_json,
    build_system_adjacency_json_directed,
    build_system_groups_json_from_chunk_edges,
    build_system_groups_json_from_directed_arcs,
    iter_graph_edges_jsonl,
    iter_graph_nodes_jsonl,
)


@dataclass(frozen=True)
class _It:
    chunk_id: int
    path: str
    symbol: str | None
    start_line: int
    end_line: int


def test_iter_graph_nodes_and_edges_filters_by_kept_clusters_and_kinds() -> None:
    items = [
        _It(1, "a.py", "A", 1, 2),
        _It(2, "a.py", "B", 3, 4),
        _It(3, "b.py", "C", 1, 2),
    ]
    chunk_id_to_system = {1: 10, 2: 10, 3: 30}
    kept = {10}

    nodes = list(
        iter_graph_nodes_jsonl(
            items=items, chunk_id_to_system=chunk_id_to_system, kept_cluster_ids=kept
        )
    )
    assert [n["chunk_id"] for n in nodes] == [1, 2]
    assert all(int(n["cluster_id"]) == 10 for n in nodes)

    # Edge 1-2 is strict mutual, 2-3 is fallback. Only 1-2 should survive kept filter.
    edges = [(1, 2, 0.9), (2, 3, 0.8)]
    strict = {(1, 2)}
    rows = list(
        iter_graph_edges_jsonl(
            edges=edges,
            strict_edge_keys=strict,
            chunk_id_to_system=chunk_id_to_system,
            kept_cluster_ids=kept,
        )
    )
    assert len(rows) == 1
    assert rows[0]["a_chunk_id"] == 1
    assert rows[0]["b_chunk_id"] == 2
    assert rows[0]["kind"] == "mutual"


def test_build_system_adjacency_evidence_and_neighbor_cap_union() -> None:
    # Systems: 1,2,3 (each with two chunks)
    items = [
        _It(1, "s1.py", "S1a", 1, 2),
        _It(2, "s1.py", "S1b", 3, 4),
        _It(3, "s2.py", "S2a", 1, 2),
        _It(4, "s2.py", "S2b", 3, 4),
        _It(5, "s3.py", "S3a", 1, 2),
        _It(6, "s3.py", "S3b", 3, 4),
    ]
    item_by_chunk_id = {int(it.chunk_id): it for it in items}
    chunk_id_to_system = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3}
    kept = {1, 2, 3}

    # Cross-system chunk edges (undirected).
    # (1,2) total weight_sum 10.0
    # (1,3) total weight_sum 9.0
    # (2,3) total weight_sum 1.0
    edges = [
        (1, 3, 6.0),
        (2, 4, 4.0),
        (1, 5, 5.0),
        (2, 6, 4.0),
        (3, 5, 1.0),
    ]
    strict = {(1, 3), (2, 4), (3, 5)}

    report_payload = {
        "clusters": [
            {
                "cluster_id": 1,
                "size": 2,
                "label": "s1",
                "top_files": [],
                "top_symbols": [],
            },
            {
                "cluster_id": 2,
                "size": 2,
                "label": "s2",
                "top_files": [],
                "top_symbols": [],
            },
            {
                "cluster_id": 3,
                "size": 2,
                "label": "s3",
                "top_files": [],
                "top_symbols": [],
            },
        ]
    }

    # With max_neighbors_per_system=1:
    # - system1 keeps neighbor2 (10.0 > 9.0)
    # - system3 keeps neighbor1 (9.0 > 1.0)
    # Symmetric union should keep links (1,2) and (1,3).
    adj = build_system_adjacency_json(
        report_payload=report_payload,
        edges=edges,
        strict_edge_keys=strict,
        chunk_id_to_system=chunk_id_to_system,
        item_by_chunk_id=item_by_chunk_id,
        kept_cluster_ids=kept,
        evidence_k=2,
        max_neighbors_per_system=1,
        min_cluster_size=2,
    )

    links = adj.get("links") or []
    assert isinstance(links, list)
    pairs = {
        (int(link["a"]), int(link["b"])) for link in links if isinstance(link, dict)
    }
    assert pairs == {(1, 2), (1, 3)}

    # Evidence must be top-K by weight for the pair.
    link_12 = next(
        link for link in links if int(link["a"]) == 1 and int(link["b"]) == 2
    )
    ev12 = link_12.get("evidence") or []
    assert len(ev12) == 2
    assert float(ev12[0]["w"]) >= float(ev12[1]["w"])


def test_build_system_adjacency_filters_dropped_clusters() -> None:
    items = [
        _It(1, "s1.py", "S1", 1, 2),
        _It(2, "s2.py", "S2", 1, 2),
        _It(3, "s3.py", "S3", 1, 2),
    ]
    item_by_chunk_id = {int(it.chunk_id): it for it in items}
    chunk_id_to_system = {1: 1, 2: 2, 3: 3}
    kept = {1, 2}  # drop system 3

    edges = [(1, 2, 1.0), (1, 3, 9.0)]
    strict = {(1, 2), (1, 3)}
    report_payload = {
        "clusters": [
            {
                "cluster_id": 1,
                "size": 1,
                "label": "s1",
                "top_files": [],
                "top_symbols": [],
            },
            {
                "cluster_id": 2,
                "size": 1,
                "label": "s2",
                "top_files": [],
                "top_symbols": [],
            },
        ]
    }

    adj = build_system_adjacency_json(
        report_payload=report_payload,
        edges=edges,
        strict_edge_keys=strict,
        chunk_id_to_system=chunk_id_to_system,
        item_by_chunk_id=item_by_chunk_id,
        kept_cluster_ids=kept,
        evidence_k=5,
        max_neighbors_per_system=10,
        min_cluster_size=2,
    )
    links = adj.get("links") or []
    pairs = {
        (int(link["a"]), int(link["b"])) for link in links if isinstance(link, dict)
    }
    assert pairs == {(1, 2)}


def test_build_system_adjacency_directed_outgoing_neighbor_cap() -> None:
    items = [
        _It(1, "s1.py", "S1a", 1, 2),
        _It(2, "s1.py", "S1b", 3, 4),
        _It(3, "s2.py", "S2a", 1, 2),
        _It(4, "s2.py", "S2b", 3, 4),
        _It(5, "s3.py", "S3a", 1, 2),
        _It(6, "s3.py", "S3b", 3, 4),
    ]
    item_by_chunk_id = {int(it.chunk_id): it for it in items}
    chunk_id_to_system = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3}
    kept = {1, 2, 3}

    # Directed chunk arcs (A -> B). Aggregate into directed system links.
    directed_arcs = [
        # system1 -> system2 (sum 1.7)
        (1, 3, 0.9),
        (2, 4, 0.8),
        # system1 -> system3 (sum 0.85)
        (1, 5, 0.85),
        # system2 -> system1 (sum 0.5)
        (3, 1, 0.5),
    ]

    report_payload = {
        "clusters": [
            {
                "cluster_id": 1,
                "size": 2,
                "label": "s1",
                "top_files": [],
                "top_symbols": [],
            },
            {
                "cluster_id": 2,
                "size": 2,
                "label": "s2",
                "top_files": [],
                "top_symbols": [],
            },
            {
                "cluster_id": 3,
                "size": 2,
                "label": "s3",
                "top_files": [],
                "top_symbols": [],
            },
        ]
    }

    # With max_neighbors_per_system=1, system1 keeps only its strongest outgoing
    # neighbor (system2), so system1->system3 is dropped.
    # system2->system1 survives independently.
    adj = build_system_adjacency_json_directed(
        report_payload=report_payload,
        directed_arcs=directed_arcs,
        chunk_id_to_system=chunk_id_to_system,
        item_by_chunk_id=item_by_chunk_id,
        kept_cluster_ids=kept,
        evidence_k=2,
        max_neighbors_per_system=1,
        min_cluster_size=2,
        k=30,
        tau=0.25,
    )

    assert adj.get("directed") is True
    links = adj.get("links") or []
    assert isinstance(links, list)

    pairs = {
        (int(link["source"]), int(link["target"]))
        for link in links
        if isinstance(link, dict)
    }
    assert pairs == {(1, 2), (2, 1)}

    # Evidence should use directed fields and be top-K by weight.
    link_12 = next(
        link for link in links if int(link["source"]) == 1 and int(link["target"]) == 2
    )
    ev12 = link_12.get("evidence") or []
    assert len(ev12) == 2
    assert float(ev12[0]["w"]) >= float(ev12[1]["w"])
    assert str(ev12[0]["kind"]) == "directed_knn"


def test_build_system_groups_from_chunk_edges_is_deterministic() -> None:
    # Three systems, with strongest edge between 1-2.
    # Only first three are system clusters in report; chunk ids map into those systems.
    chunk_id_to_system = {10: 1, 20: 2, 30: 3}
    kept = {1, 2, 3}

    report_payload = {
        "clusters": [
            {
                "cluster_id": 1,
                "size": 1,
                "label": "s1",
                "top_files": [],
                "top_symbols": [],
            },
            {
                "cluster_id": 2,
                "size": 1,
                "label": "s2",
                "top_files": [],
                "top_symbols": [],
            },
            {
                "cluster_id": 3,
                "size": 1,
                "label": "s3",
                "top_files": [],
                "top_symbols": [],
            },
        ]
    }

    # Chunk edges are undirected; use ids 10/20/30.
    edges = [
        (10, 20, 2.0),
        (10, 30, 0.2),
    ]

    payload = build_system_groups_json_from_chunk_edges(
        report_payload=report_payload,
        edges=edges,
        chunk_id_to_system=chunk_id_to_system,
        kept_cluster_ids=kept,
        resolutions=[0.5, 1.0],
        seed=0,
        edge_weight="weight_sum",
    )
    assert (
        str(payload.get("schema_version"))
        == "snapshot.chunk_systems.system_groups.v1"
    )
    assert isinstance(payload.get("systems"), list)
    parts = payload.get("partitions") or []
    assert isinstance(parts, list)
    assert len(parts) == 2
    for part in parts:
        assert "membership" in part
        mem = part["membership"]
        assert isinstance(mem, list)
        assert len(mem) == 3
        # remapped ids start at 1 and are contiguous
        uniq = sorted(set(int(x) for x in mem))
        assert uniq and uniq[0] == 1
        assert uniq == list(range(1, 1 + len(uniq)))


def test_build_system_groups_from_directed_arcs_symmetrizes() -> None:
    report_payload = {
        "clusters": [
            {
                "cluster_id": 1,
                "size": 1,
                "label": "s1",
                "top_files": [],
                "top_symbols": [],
            },
            {
                "cluster_id": 2,
                "size": 1,
                "label": "s2",
                "top_files": [],
                "top_symbols": [],
            },
        ]
    }
    # Chunk ids map 10->1, 20->2
    chunk_id_to_system = {10: 1, 20: 2}
    kept = {1, 2}

    # Provide directed arcs both directions; symmetrization creates one undirected edge.
    arcs = [
        (10, 20, 1.0),  # 1->2
        (20, 10, 0.5),  # 2->1
    ]
    payload = build_system_groups_json_from_directed_arcs(
        report_payload=report_payload,
        directed_arcs=arcs,
        chunk_id_to_system=chunk_id_to_system,
        kept_cluster_ids=kept,
        resolutions=[1.0],
        seed=0,
        edge_weight="weight_sum",
    )
    params = payload.get("params") or {}
    assert isinstance(params, dict)
    assert params.get("directed_input") is True
    assert str(params.get("symmetrization")) == "sum"
