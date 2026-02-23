#!/usr/bin/env python3
"""Unit tests for experimental snapshot chunk systems."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import chunkhound.snapshot.chunk_systems as chunk_systems_mod
from chunkhound.snapshot.partitioning import (
    SystemComponent,
    SystemEdge,
    SystemNode,
    partition_leiden,
)
from chunkhound.snapshot.chunk_systems import (
    build_chunk_systems_views,
    _mutual_edges_from_neighbors,
    compute_chunk_systems,
)


def test_mutual_edges_from_neighbors_filters_and_weights() -> None:
    neighbors = {
        1: {2: 0.9, 3: 0.2},
        2: {1: 0.8, 3: 0.5},
        3: {1: 0.3},
        4: {5: 0.3},
        5: {4: 0.2},
    }
    edges = _mutual_edges_from_neighbors(neighbors=neighbors, tau=0.25)
    assert len(edges) == 1
    e = edges[0]
    assert (e.a, e.b) == (1, 2)
    assert e.w == pytest.approx(0.8)


def test_compute_chunk_systems_min_degree_fallback_adds_non_mutual_edge() -> None:
    @dataclass(frozen=True)
    class _It:
        chunk_id: int
        path: str
        symbol: str | None
        start_line: int
        end_line: int

    # 1 and 2 are close; 2 and 3 are also close. With k=1:
    # - 1's nearest is 2
    # - 2's nearest is 3 (so 1-2 is not mutual)
    # - 3's nearest is 2 (so 2-3 is mutual)
    items = [
        _It(1, "a.py", "A", 1, 2),
        _It(2, "a.py", "B", 3, 4),
        _It(3, "b.py", "C", 1, 2),
    ]
    embeddings = [
        [1.0, 0.0],
        [0.9, 0.1],
        [0.85, 0.15],
    ]

    payload = compute_chunk_systems(
        items=items,
        embeddings=embeddings,
        k=1,
        tau=0.8,
        max_nodes=100,
        partitioner="cc",
        leiden_resolution=1.0,
        leiden_seed=0,
        leiden_resolutions_auto=None,
        edge_min_degree=1,
        edge_fallback_tau=None,
    )

    counts = payload.get("counts") or {}
    assert isinstance(counts, dict)
    assert counts.get("mutual_edges") == 1
    assert counts.get("fallback_edges") == 1
    assert counts.get("edges") == 2
    assert counts.get("degree_zero_nodes") == 0

    # With the fallback edge 1-2 present, all three nodes should connect via CC.
    assert counts.get("clusters") == 1
    assert counts.get("singletons") == 0


def test_partition_leiden_two_communities_stable() -> None:
    nodes = [
        SystemNode(node_key=str(i), change_indexes=[i], vector=[]) for i in range(1, 7)
    ]
    edges: list[SystemEdge] = []

    clique_a = [1, 2, 3]
    clique_b = [4, 5, 6]
    for clique in (clique_a, clique_b):
        for i, a in enumerate(clique):
            for b in clique[i + 1 :]:
                edges.append(
                    SystemEdge(
                        node_a=str(a),
                        node_b=str(b),
                        w_sem=1.0,
                        w_struct=None,
                        w_total=1.0,
                    )
                )

    # Weak bridge so CC would be 1 component, but Leiden should find 2 communities.
    edges.append(
        SystemEdge(
            node_a="3",
            node_b="4",
            w_sem=1e-6,
            w_struct=None,
            w_total=1e-6,
        )
    )

    components, _node_to_system = partition_leiden(
        nodes=nodes, edges=edges, resolution=1.0, seed=0
    )
    by_node: dict[str, int] = {}
    for comp in components:
        for k in comp.node_keys:
            by_node[k] = int(comp.system_id)

    assert by_node["1"] == by_node["2"] == by_node["3"]
    assert by_node["4"] == by_node["5"] == by_node["6"]
    assert by_node["1"] != by_node["4"]


def test_compute_chunk_systems_payload_invariants() -> None:
    @dataclass(frozen=True)
    class _It:
        chunk_id: int
        path: str
        symbol: str | None
        start_line: int
        end_line: int

    items = [
        _It(1, "a.py", "A", 1, 2),
        _It(2, "a.py", "B", 3, 4),
        _It(3, "a.py", "C", 5, 6),
        _It(4, "b.py", "D", 1, 2),
        _It(5, "b.py", "E", 3, 4),
        _It(6, "b.py", "F", 5, 6),
    ]

    embeddings = [
        [1.0, 0.0],
        [0.9, 0.1],
        [0.95, 0.05],
        [0.0, 1.0],
        [0.1, 0.9],
        [0.05, 0.95],
    ]

    payload = compute_chunk_systems(
        items=items,
        embeddings=embeddings,
        k=2,
        tau=0.5,
        max_nodes=100,
        partitioner="cc",
        leiden_resolution=1.0,
        leiden_seed=0,
        leiden_resolutions_auto=None,
    )
    assert payload["schema_version"] == "snapshot.chunk_systems.v1"
    assert payload["schema_revision"] == "2026-02-17"

    counts = payload.get("counts") or {}
    assert isinstance(counts, dict)
    assert counts.get("nodes") == 6

    clusters = payload.get("clusters") or []
    assert isinstance(clusters, list)

    all_ids: list[int] = []
    for c in clusters:
        assert isinstance(c, dict)
        chunk_ids = c.get("chunk_ids")
        assert isinstance(chunk_ids, list)
        assert all(isinstance(x, int) for x in chunk_ids)
        all_ids.extend(int(x) for x in chunk_ids)

    assert sorted(all_ids) == [1, 2, 3, 4, 5, 6]
    assert len(set(all_ids)) == 6


def test_compute_chunk_systems_directed_knn_arc_count_matches_k() -> None:
    @dataclass(frozen=True)
    class _It:
        chunk_id: int
        path: str
        symbol: str | None
        start_line: int
        end_line: int

    items = [
        _It(1, "a.py", "A", 1, 2),
        _It(2, "a.py", "B", 3, 4),
        _It(3, "a.py", "C", 5, 6),
        _It(4, "b.py", "D", 1, 2),
        _It(5, "b.py", "E", 3, 4),
        _It(6, "b.py", "F", 5, 6),
    ]

    embeddings = [
        [1.0, 0.0],
        [0.9, 0.1],
        [0.95, 0.05],
        [0.0, 1.0],
        [0.1, 0.9],
        [0.05, 0.95],
    ]

    n = len(items)
    for k in (2, 5):
        payload = compute_chunk_systems(
            items=items,
            embeddings=embeddings,
            k=k,
            tau=0.5,
            max_nodes=100,
            partitioner="cc",
            leiden_resolution=1.0,
            leiden_seed=0,
            leiden_resolutions_auto=None,
        )

        counts = payload.get("counts") or {}
        assert isinstance(counts, dict)
        assert counts.get("directed_knn_arcs") == n * min(k, n - 1)


def test_build_chunk_systems_views_prunes_singletons() -> None:
    raw = {
        "schema_version": "snapshot.chunk_systems.v1",
        "schema_revision": "2026-02-17",
        "run": {},
        "params": {},
        "counts": {"nodes": 6, "clusters": 3, "singletons": 1, "largest_cluster": 3},
        "clusters": [
            {"cluster_id": 1, "size": 1, "chunk_ids": [1]},
            {"cluster_id": 2, "size": 2, "chunk_ids": [2, 3]},
            {"cluster_id": 3, "size": 3, "chunk_ids": [4, 5, 6]},
        ],
    }

    pruned, dropped = build_chunk_systems_views(payload=raw, min_cluster_size=2)

    assert isinstance(pruned.get("view"), dict)
    assert pruned["view"]["min_cluster_size"] == 2
    assert pruned["view"]["kind"] == "pruned"
    assert [c["cluster_id"] for c in pruned["clusters"]] == [2, 3]
    assert pruned["counts_view"]["clusters"] == 2
    assert pruned["counts_view"]["nodes"] == 5
    assert pruned["counts_view"]["singletons"] == 0
    assert pruned["counts_view"]["largest_cluster"] == 3

    assert isinstance(dropped.get("view"), dict)
    assert dropped["view"]["kind"] == "dropped"
    assert [c["cluster_id"] for c in dropped["clusters"]] == [1]
    assert dropped["counts_view"]["clusters"] == 1
    assert dropped["counts_view"]["nodes"] == 1
    assert dropped["counts_view"]["singletons"] == 1


def test_compute_chunk_systems_min_degree_fallback_respects_same_dir_path_mode() -> None:
    @dataclass(frozen=True)
    class _It:
        chunk_id: int
        path: str
        symbol: str | None
        start_line: int
        end_line: int

    # With k=1:
    # - 1's nearest is 2 (cross-dir)
    # - 2's nearest is 3 (same-dir)
    # - 3's nearest is 2 (same-dir)
    #
    # Strict mutual-kNN yields only edge 2-3. Node 1 would need a fallback edge
    # to reach min-degree, but same_dir blocks it.
    items = [
        _It(1, "a/x.py", "A", 1, 2),
        _It(2, "b/y.py", "B", 3, 4),
        _It(3, "b/z.py", "C", 1, 2),
    ]
    embeddings = [
        [1.0, 0.0],
        [0.9, 0.1],
        [0.85, 0.15],
    ]

    payload = compute_chunk_systems(
        items=items,
        embeddings=embeddings,
        k=1,
        tau=0.8,
        max_nodes=100,
        partitioner="cc",
        leiden_resolution=1.0,
        leiden_seed=0,
        leiden_resolutions_auto=None,
        edge_min_degree=1,
        edge_fallback_tau=None,
        edge_fallback_path_mode="same_dir",
    )

    counts = payload.get("counts") or {}
    assert isinstance(counts, dict)
    assert counts.get("mutual_edges") == 1
    assert counts.get("fallback_edges") == 0
    assert counts.get("degree_zero_nodes") == 1
    assert counts.get("clusters") == 2
    assert counts.get("singletons") == 1


def test_adjusted_rand_index_basic_properties() -> None:
    ari_same = chunk_systems_mod._adjusted_rand_index(
        labels_a=[0, 0, 1, 1], labels_b=[0, 0, 1, 1]
    )
    assert ari_same == pytest.approx(1.0)

    ari_diff = chunk_systems_mod._adjusted_rand_index(
        labels_a=[0, 0, 1, 1], labels_b=[0, 1, 0, 1]
    )
    assert -1.0 <= ari_diff <= 1.0
    assert ari_diff < 1.0


def test_objective_stable_selector_prefers_stable_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    nodes = [SystemNode(node_key=str(i), change_indexes=[i], vector=[]) for i in range(1, 13)]
    edges: list[SystemEdge] = []

    def _components_from_membership(
        *, ordered_keys: list[str], membership: list[int]
    ) -> tuple[list[SystemComponent], dict[str, int]]:
        grouped: dict[int, list[str]] = {}
        for key, grp in zip(ordered_keys, membership, strict=True):
            grouped.setdefault(int(grp), []).append(str(key))
        groups = [sorted(keys) for keys in grouped.values()]
        groups.sort(key=lambda keys: (-len(keys), keys[0] if keys else ""))
        components: list[SystemComponent] = []
        node_to_system: dict[str, int] = {}
        for idx, keys in enumerate(groups, start=1):
            for k in keys:
                node_to_system[str(k)] = int(idx)
            change_indexes = sorted({int(k) for k in keys if str(k).isdigit()})
            components.append(
                SystemComponent(system_id=int(idx), node_keys=list(keys), change_indexes=change_indexes)
            )
        return components, node_to_system

    def _stub_partition_leiden_with_membership(
        *, nodes: list[SystemNode], edges: list[SystemEdge], resolution: float, seed: int
    ) -> tuple[list[SystemComponent], dict[str, int], list[str], list[int], float]:
        ordered = sorted(nodes, key=lambda n: str(n.node_key))
        ordered_keys = [str(n.node_key) for n in ordered]
        n = len(ordered_keys)
        res = float(resolution)
        s = int(seed)

        if res == 1.0:
            membership = [0] * 4 + [1] * 4 + [2] * 4
            obj_q = 10.0
        elif res == 2.0:
            # Same cluster count, but membership drifts with seed (low ARI).
            if s == 0:
                membership = [0] * 4 + [1] * 4 + [2] * 4
            elif s == 1:
                membership = [0] * 3 + [1] + [1] * 3 + [2] + [2] * 4
            else:
                membership = [0] * 2 + [1] * 2 + [1] * 2 + [2] * 2 + [2] * 4
            obj_q = 20.0
        else:
            membership = list(range(n))
            obj_q = 0.0

        comps, node_to_system = _components_from_membership(
            ordered_keys=ordered_keys, membership=membership
        )
        return comps, node_to_system, ordered_keys, membership, obj_q

    monkeypatch.setattr(
        chunk_systems_mod,
        "partition_leiden_with_membership",
        _stub_partition_leiden_with_membership,
    )

    _components, _node_to_system, chosen = chunk_systems_mod._choose_partition(
        nodes=nodes,
        edges=edges,
        partitioner="auto",
        leiden_resolution=1.0,
        leiden_seed=0,
        leiden_resolutions_auto=[1.0, 2.0],
        leiden_auto_selector="objective_stable",
        leiden_auto_stability_seeds=[0, 1, 2],
        leiden_auto_stability_min=0.90,
        leiden_auto_min_avg_size=3,
        leiden_auto_max_avg_size=6,
        leiden_auto_max_largest_frac_nodes=1.0,
        leiden_auto_max_singleton_frac_nodes=1.0,
        record_partition_sweep=True,
    )

    assert chosen.get("method") == "leiden"
    assert chosen.get("resolution") == pytest.approx(1.0)
    assert isinstance(chosen.get("_partition_sweep"), dict)


def test_objective_stable_selector_rejects_out_of_bounds_fragmentation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nodes = [SystemNode(node_key=str(i), change_indexes=[i], vector=[]) for i in range(1, 13)]
    edges: list[SystemEdge] = []

    def _components_from_membership(
        *, ordered_keys: list[str], membership: list[int]
    ) -> tuple[list[SystemComponent], dict[str, int]]:
        grouped: dict[int, list[str]] = {}
        for key, grp in zip(ordered_keys, membership, strict=True):
            grouped.setdefault(int(grp), []).append(str(key))
        groups = [sorted(keys) for keys in grouped.values()]
        groups.sort(key=lambda keys: (-len(keys), keys[0] if keys else ""))
        components: list[SystemComponent] = []
        node_to_system: dict[str, int] = {}
        for idx, keys in enumerate(groups, start=1):
            for k in keys:
                node_to_system[str(k)] = int(idx)
            change_indexes = sorted({int(k) for k in keys if str(k).isdigit()})
            components.append(
                SystemComponent(system_id=int(idx), node_keys=list(keys), change_indexes=change_indexes)
            )
        return components, node_to_system

    def _stub_partition_leiden_with_membership(
        *, nodes: list[SystemNode], edges: list[SystemEdge], resolution: float, seed: int
    ) -> tuple[list[SystemComponent], dict[str, int], list[str], list[int], float]:
        ordered = sorted(nodes, key=lambda n: str(n.node_key))
        ordered_keys = [str(n.node_key) for n in ordered]
        n = len(ordered_keys)
        res = float(resolution)

        if res == 1.0:
            membership = [0] * 4 + [1] * 4 + [2] * 4
            obj_q = 10.0
        else:
            membership = list(range(n))  # extreme fragmentation
            obj_q = 100.0

        comps, node_to_system = _components_from_membership(
            ordered_keys=ordered_keys, membership=membership
        )
        return comps, node_to_system, ordered_keys, membership, obj_q

    monkeypatch.setattr(
        chunk_systems_mod,
        "partition_leiden_with_membership",
        _stub_partition_leiden_with_membership,
    )

    _components, _node_to_system, chosen = chunk_systems_mod._choose_partition(
        nodes=nodes,
        edges=edges,
        partitioner="auto",
        leiden_resolution=1.0,
        leiden_seed=0,
        leiden_resolutions_auto=[1.0, 3.0],
        leiden_auto_selector="objective_stable",
        leiden_auto_stability_seeds=[0, 1, 2],
        leiden_auto_stability_min=0.0,
        leiden_auto_min_avg_size=3,
        leiden_auto_max_avg_size=6,
        leiden_auto_max_largest_frac_nodes=1.0,
        leiden_auto_max_singleton_frac_nodes=1.0,
    )

    assert chosen.get("method") == "leiden"
    assert chosen.get("resolution") == pytest.approx(1.0)
