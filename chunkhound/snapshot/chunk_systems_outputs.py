"""Optional output artifacts for snapshot chunk-systems (graph + adjacency).

These helpers are intentionally pure/deterministic. Emission is gated behind
CLI flags so the baseline snapshot output remains unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from chunkhound.snapshot.partitioning import (
    SystemEdge,
    SystemNode,
    partition_leiden_with_membership,
)


class SnapshotItemLike(Protocol):
    chunk_id: int
    path: str
    symbol: str | None
    start_line: int
    end_line: int


@dataclass(frozen=True)
class _AdjEvidence:
    a_chunk_id: int
    b_chunk_id: int
    w: float
    kind: str
    a_path: str
    b_path: str
    a_symbol: str | None
    b_symbol: str | None
    a_start_line: int
    a_end_line: int
    b_start_line: int
    b_end_line: int


@dataclass
class _AdjAgg:
    edge_count: int
    weight_sum: float
    weight_max: float
    evidence: list[_AdjEvidence]


@dataclass(frozen=True)
class _AdjEvidenceDirected:
    source_chunk_id: int
    target_chunk_id: int
    w: float
    kind: str
    source_path: str
    target_path: str
    source_symbol: str | None
    target_symbol: str | None
    source_start_line: int
    source_end_line: int
    target_start_line: int
    target_end_line: int


@dataclass
class _AdjAggDirected:
    edge_count: int
    weight_sum: float
    weight_max: float
    evidence: list[_AdjEvidenceDirected]


def _systems_from_report_payload(
    *, report_payload: dict[str, object], kept_cluster_ids: set[int]
) -> list[dict[str, object]]:
    clusters = report_payload.get("clusters") or []
    systems: list[dict[str, object]] = []
    for c in clusters:
        if not isinstance(c, dict):
            continue
        cid = c.get("cluster_id")
        if cid is None:
            continue
        try:
            cid_i = int(cid)
        except Exception:
            continue
        if cid_i not in kept_cluster_ids:
            continue
        systems.append(
            {
                "cluster_id": int(cid_i),
                "size": int(c.get("size") or 0),
                "label": str(c.get("label") or ""),
                "top_files": c.get("top_files") or [],
                "top_symbols": c.get("top_symbols") or [],
            }
        )
    systems.sort(key=lambda s: int(s.get("cluster_id") or 0))
    return systems


def _remap_membership_deterministic(
    *, ordered_cluster_ids: list[int], membership: list[int]
) -> tuple[list[int], int]:
    # Map original group -> list of member cluster_ids.
    groups: dict[int, list[int]] = {}
    for cid, g in zip(ordered_cluster_ids, membership):
        try:
            gid = int(g)
        except Exception:
            gid = 0
        groups.setdefault(gid, []).append(int(cid))

    ranked: list[tuple[int, int, int]] = []
    for gid, members in groups.items():
        if not members:
            continue
        ranked.append((int(gid), int(len(members)), int(min(members))))
    ranked.sort(key=lambda t: (-int(t[1]), int(t[2]), int(t[0])))

    remap: dict[int, int] = {}
    for idx, (gid, _, _) in enumerate(ranked, start=1):
        remap[int(gid)] = int(idx)

    out = [int(remap.get(int(g), 0)) for g in membership]
    return out, int(len(ranked))


def build_system_groups_json(
    *,
    report_payload: dict[str, object],
    systems: list[dict[str, object]],
    system_edges: list[tuple[int, int, float]],
    resolutions: list[float],
    seed: int,
    edge_weight: str,
    directed_input: bool,
    symmetrization: str,
) -> dict[str, object]:
    ordered_cluster_ids = [int(s.get("cluster_id") or 0) for s in systems]
    nodes = [
        SystemNode(node_key=str(int(cid)), change_indexes=[int(cid)], vector=[])
        for cid in ordered_cluster_ids
        if int(cid) > 0
    ]
    edges = [
        SystemEdge(
            node_a=str(int(a)),
            node_b=str(int(b)),
            w_sem=float(w),
            w_struct=None,
            w_total=float(w),
        )
        for a, b, w in system_edges
        if int(a) != int(b) and float(w) > 0.0
    ]

    partitions: list[dict[str, object]] = []
    for res in resolutions:
        try:
            _, _, ordered_keys, membership, _ = partition_leiden_with_membership(
                nodes=nodes, edges=edges, resolution=float(res), seed=int(seed)
            )
            ordered_ids = [int(k) for k in ordered_keys]
            membership_remap, group_count = _remap_membership_deterministic(
                ordered_cluster_ids=ordered_ids,
                membership=[int(x) for x in membership],
            )
        except ImportError:
            ordered_ids = [int(cid) for cid in ordered_cluster_ids]
            membership_remap = list(range(1, 1 + len(ordered_ids)))
            group_count = int(len(ordered_ids))
        partitions.append(
            {
                "resolution": float(res),
                "group_count": int(group_count),
                "membership": membership_remap,
            }
        )

    return {
        "schema_version": "snapshot.chunk_systems.system_groups.v1",
        "schema_revision": "2026-02-20",
        "source": "snapshot.chunk_systems.system_adjacency.json",
        "params": {
            "mode": "leiden",
            "seed": int(seed),
            "resolutions": [float(r) for r in resolutions],
            "edge_weight": str(edge_weight),
            "directed_input": bool(directed_input),
            "symmetrization": str(symmetrization),
        },
        "systems": systems,
        "partitions": partitions,
    }


def build_system_groups_json_from_chunk_edges(
    *,
    report_payload: dict[str, object],
    edges: list[tuple[int, int, float]],
    chunk_id_to_system: dict[int, int],
    kept_cluster_ids: set[int],
    resolutions: list[float],
    seed: int,
    edge_weight: str,
) -> dict[str, object]:
    systems = _systems_from_report_payload(
        report_payload=report_payload, kept_cluster_ids=kept_cluster_ids
    )
    pairs: dict[tuple[int, int], tuple[int, float, float]] = {}
    for a, b, w in edges:
        a_i = int(a)
        b_i = int(b)
        if a_i == b_i:
            continue
        ca = chunk_id_to_system.get(a_i)
        cb = chunk_id_to_system.get(b_i)
        if ca is None or cb is None:
            continue
        ca_i = int(ca)
        cb_i = int(cb)
        if ca_i == cb_i:
            continue
        if ca_i not in kept_cluster_ids or cb_i not in kept_cluster_ids:
            continue
        u, v = (ca_i, cb_i) if ca_i < cb_i else (cb_i, ca_i)
        key = (int(u), int(v))
        cnt, wsum, wmax = pairs.get(key, (0, 0.0, 0.0))
        wf = float(w)
        pairs[key] = (int(cnt) + 1, float(wsum) + wf, max(float(wmax), wf))

    system_edges: list[tuple[int, int, float]] = []
    for (u, v), (_cnt, wsum, wmax) in pairs.items():
        w = float(wsum) if str(edge_weight) == "weight_sum" else float(wmax)
        if w <= 0.0:
            continue
        system_edges.append((int(u), int(v), float(w)))
    system_edges.sort(key=lambda t: (-float(t[2]), int(t[0]), int(t[1])))

    return build_system_groups_json(
        report_payload=report_payload,
        systems=systems,
        system_edges=system_edges,
        resolutions=resolutions,
        seed=int(seed),
        edge_weight=str(edge_weight),
        directed_input=False,
        symmetrization="n/a",
    )


def build_system_groups_json_from_directed_arcs(
    *,
    report_payload: dict[str, object],
    directed_arcs: list[tuple[int, int, float]],
    chunk_id_to_system: dict[int, int],
    kept_cluster_ids: set[int],
    resolutions: list[float],
    seed: int,
    edge_weight: str,
) -> dict[str, object]:
    systems = _systems_from_report_payload(
        report_payload=report_payload, kept_cluster_ids=kept_cluster_ids
    )

    # Aggregate directed system links.
    directed: dict[tuple[int, int], tuple[int, float, float]] = {}
    for a, b, w in directed_arcs:
        a_i = int(a)
        b_i = int(b)
        if a_i == b_i:
            continue
        ca = chunk_id_to_system.get(a_i)
        cb = chunk_id_to_system.get(b_i)
        if ca is None or cb is None:
            continue
        src = int(ca)
        dst = int(cb)
        if src == dst:
            continue
        if src not in kept_cluster_ids or dst not in kept_cluster_ids:
            continue
        key = (int(src), int(dst))
        cnt, wsum, wmax = directed.get(key, (0, 0.0, 0.0))
        wf = float(w)
        directed[key] = (int(cnt) + 1, float(wsum) + wf, max(float(wmax), wf))

    # Symmetrize for meta-Leiden.
    pairs: dict[tuple[int, int], tuple[int, float, float]] = {}
    for (src, dst), (cnt, wsum, wmax) in directed.items():
        u, v = (src, dst) if src < dst else (dst, src)
        key = (int(u), int(v))
        cnt0, wsum0, wmax0 = pairs.get(key, (0, 0.0, 0.0))
        pairs[key] = (
            int(cnt0) + int(cnt),
            float(wsum0) + float(wsum),
            max(float(wmax0), float(wmax)),
        )

    system_edges: list[tuple[int, int, float]] = []
    for (u, v), (_cnt, wsum, wmax) in pairs.items():
        w = float(wsum) if str(edge_weight) == "weight_sum" else float(wmax)
        if w <= 0.0:
            continue
        system_edges.append((int(u), int(v), float(w)))
    system_edges.sort(key=lambda t: (-float(t[2]), int(t[0]), int(t[1])))

    return build_system_groups_json(
        report_payload=report_payload,
        systems=systems,
        system_edges=system_edges,
        resolutions=resolutions,
        seed=int(seed),
        edge_weight=str(edge_weight),
        directed_input=True,
        symmetrization="sum",
    )


def iter_graph_nodes_jsonl(
    *,
    items: list[SnapshotItemLike],
    chunk_id_to_system: dict[int, int],
    kept_cluster_ids: set[int],
) -> Iterator[dict[str, object]]:
    ordered = sorted(items, key=lambda it: int(it.chunk_id))
    for it in ordered:
        cid = int(it.chunk_id)
        cluster_id = chunk_id_to_system.get(int(cid))
        if cluster_id is None:
            continue
        if int(cluster_id) not in kept_cluster_ids:
            continue
        yield {
            "chunk_id": int(cid),
            "cluster_id": int(cluster_id),
            "path": str(it.path),
            "symbol": it.symbol if it.symbol is not None else None,
            "start_line": int(it.start_line),
            "end_line": int(it.end_line),
        }


def iter_graph_edges_jsonl(
    *,
    edges: list[tuple[int, int, float]],
    strict_edge_keys: set[tuple[int, int]],
    chunk_id_to_system: dict[int, int],
    kept_cluster_ids: set[int],
) -> Iterator[dict[str, object]]:
    for a, b, w in edges:
        a_i = int(a)
        b_i = int(b)
        if a_i == b_i:
            continue
        ca = chunk_id_to_system.get(a_i)
        cb = chunk_id_to_system.get(b_i)
        if ca is None or cb is None:
            continue
        if int(ca) not in kept_cluster_ids or int(cb) not in kept_cluster_ids:
            continue
        aa, bb = (a_i, b_i) if a_i < b_i else (b_i, a_i)
        key = (int(aa), int(bb))
        kind = "mutual" if key in strict_edge_keys else "fallback"
        yield {
            "a_chunk_id": int(aa),
            "b_chunk_id": int(bb),
            "a_cluster_id": int(chunk_id_to_system[int(aa)]),
            "b_cluster_id": int(chunk_id_to_system[int(bb)]),
            "w": float(w),
            "kind": str(kind),
        }


def build_system_adjacency_json(
    *,
    report_payload: dict[str, object],
    edges: list[tuple[int, int, float]],
    strict_edge_keys: set[tuple[int, int]],
    chunk_id_to_system: dict[int, int],
    item_by_chunk_id: dict[int, SnapshotItemLike],
    kept_cluster_ids: set[int],
    evidence_k: int,
    max_neighbors_per_system: int,
    min_cluster_size: int,
) -> dict[str, object]:
    evidence_k_i = int(evidence_k)
    max_neighbors_i = int(max_neighbors_per_system)
    if evidence_k_i <= 0:
        raise ValueError("evidence_k must be positive")
    if max_neighbors_i <= 0:
        raise ValueError("max_neighbors_per_system must be positive")

    clusters = report_payload.get("clusters") or []
    systems: list[dict[str, object]] = []
    for c in clusters:
        if not isinstance(c, dict):
            continue
        cid = c.get("cluster_id")
        if cid is None:
            continue
        try:
            cid_i = int(cid)
        except Exception:
            continue
        if cid_i not in kept_cluster_ids:
            continue
        systems.append(
            {
                "cluster_id": int(cid_i),
                "size": int(c.get("size") or 0),
                "label": str(c.get("label") or ""),
                "top_files": c.get("top_files") or [],
                "top_symbols": c.get("top_symbols") or [],
            }
        )
    systems.sort(key=lambda s: int(s.get("cluster_id") or 0))

    # Aggregate cross-system edges (only among kept systems).
    pairs: dict[tuple[int, int], _AdjAgg] = {}
    for a, b, w in edges:
        a_i = int(a)
        b_i = int(b)
        if a_i == b_i:
            continue
        ca = chunk_id_to_system.get(a_i)
        cb = chunk_id_to_system.get(b_i)
        if ca is None or cb is None:
            continue
        ca_i = int(ca)
        cb_i = int(cb)
        if ca_i == cb_i:
            continue
        if ca_i not in kept_cluster_ids or cb_i not in kept_cluster_ids:
            continue
        u, v = (ca_i, cb_i) if ca_i < cb_i else (cb_i, ca_i)
        key = (int(u), int(v))
        agg = pairs.get(key)
        if agg is None:
            agg = _AdjAgg(edge_count=0, weight_sum=0.0, weight_max=0.0, evidence=[])
            pairs[key] = agg

        wf = float(w)
        agg.edge_count = int(agg.edge_count) + 1
        agg.weight_sum = float(agg.weight_sum) + float(wf)
        if float(wf) > float(agg.weight_max):
            agg.weight_max = float(wf)

        it_a = item_by_chunk_id.get(a_i)
        it_b = item_by_chunk_id.get(b_i)
        if it_a is None or it_b is None:
            continue
        aa, bb = (a_i, b_i) if a_i < b_i else (b_i, a_i)
        edge_key = (int(aa), int(bb))
        kind = "mutual" if edge_key in strict_edge_keys else "fallback"
        if a_i < b_i:
            left, right = it_a, it_b
        else:
            left, right = it_b, it_a
        ev = _AdjEvidence(
            a_chunk_id=int(aa),
            b_chunk_id=int(bb),
            w=float(wf),
            kind=str(kind),
            a_path=str(left.path),
            b_path=str(right.path),
            a_symbol=left.symbol if left.symbol is not None else None,
            b_symbol=right.symbol if right.symbol is not None else None,
            a_start_line=int(left.start_line),
            a_end_line=int(left.end_line),
            b_start_line=int(right.start_line),
            b_end_line=int(right.end_line),
        )
        agg.evidence.append(ev)
        agg.evidence.sort(
            key=lambda e: (-float(e.w), int(e.a_chunk_id), int(e.b_chunk_id))
        )
        if len(agg.evidence) > evidence_k_i:
            agg.evidence = agg.evidence[:evidence_k_i]

    links_before = int(len(pairs))

    # Neighbor cap: keep top-M neighbors per system by weight_sum.
    neighbors: dict[int, list[tuple[int, float, float, int]]] = {}
    for (u, v), agg in pairs.items():
        neighbors.setdefault(int(u), []).append(
            (int(v), float(agg.weight_sum), float(agg.weight_max), int(agg.edge_count))
        )
        neighbors.setdefault(int(v), []).append(
            (int(u), float(agg.weight_sum), float(agg.weight_max), int(agg.edge_count))
        )

    top_neighbors: dict[int, set[int]] = {}
    for sys_id, nbrs in neighbors.items():
        ordered = sorted(
            nbrs,
            key=lambda x: (-float(x[1]), -float(x[2]), -int(x[3]), int(x[0])),
        )
        keep = ordered[: max_neighbors_i]
        top_neighbors[int(sys_id)] = {int(n) for n, _, _, _ in keep}

    def _keep_pair(u: int, v: int) -> bool:
        return (int(v) in top_neighbors.get(int(u), set())) or (
            int(u) in top_neighbors.get(int(v), set())
        )

    kept_pairs: list[tuple[int, int]] = []
    for (u, v) in pairs.keys():
        if _keep_pair(int(u), int(v)):
            kept_pairs.append((int(u), int(v)))

    kept_pairs.sort(
        key=lambda uv: (
            -float(pairs[uv].weight_sum),
            -float(pairs[uv].weight_max),
            -int(pairs[uv].edge_count),
            int(uv[0]),
            int(uv[1]),
        )
    )

    links: list[dict[str, object]] = []
    for u, v in kept_pairs:
        agg = pairs[(u, v)]
        links.append(
            {
                "a": int(u),
                "b": int(v),
                "edge_count": int(agg.edge_count),
                "weight_sum": float(agg.weight_sum),
                "weight_max": float(agg.weight_max),
                "evidence": [
                    {
                        "a_chunk_id": int(ev.a_chunk_id),
                        "b_chunk_id": int(ev.b_chunk_id),
                        "w": float(ev.w),
                        "kind": str(ev.kind),
                        "a_path": str(ev.a_path),
                        "b_path": str(ev.b_path),
                        "a_symbol": ev.a_symbol if ev.a_symbol is not None else None,
                        "b_symbol": ev.b_symbol if ev.b_symbol is not None else None,
                        "a_start_line": int(ev.a_start_line),
                        "a_end_line": int(ev.a_end_line),
                        "b_start_line": int(ev.b_start_line),
                        "b_end_line": int(ev.b_end_line),
                    }
                    for ev in agg.evidence
                ],
            }
        )

    payload: dict[str, object] = {
        "schema_version": "snapshot.chunk_systems.system_adjacency.v1",
        "schema_revision": "2026-02-17",
        "source": "snapshot.chunk_systems.json",
        "directed": False,
        "params": {
            "evidence_k": int(evidence_k_i),
            "max_neighbors_per_system": int(max_neighbors_i),
            "min_cluster_size": int(min_cluster_size),
            "link_score": "weight_sum",
        },
        "systems": systems,
        "links": links,
        "truncation": {
            "method": "top_neighbors_per_system",
            "max_neighbors_per_system": int(max_neighbors_i),
            "links_before": int(links_before),
            "links_after": int(len(links)),
        },
    }

    return payload


def build_system_adjacency_json_directed(
    *,
    report_payload: dict[str, object],
    directed_arcs: list[tuple[int, int, float]],
    chunk_id_to_system: dict[int, int],
    item_by_chunk_id: dict[int, SnapshotItemLike],
    kept_cluster_ids: set[int],
    evidence_k: int,
    max_neighbors_per_system: int,
    min_cluster_size: int,
    k: int,
    tau: float,
) -> dict[str, object]:
    evidence_k_i = int(evidence_k)
    max_neighbors_i = int(max_neighbors_per_system)
    if evidence_k_i <= 0:
        raise ValueError("evidence_k must be positive")
    if max_neighbors_i <= 0:
        raise ValueError("max_neighbors_per_system must be positive")

    clusters = report_payload.get("clusters") or []
    systems: list[dict[str, object]] = []
    for c in clusters:
        if not isinstance(c, dict):
            continue
        cid = c.get("cluster_id")
        if cid is None:
            continue
        try:
            cid_i = int(cid)
        except Exception:
            continue
        if cid_i not in kept_cluster_ids:
            continue
        systems.append(
            {
                "cluster_id": int(cid_i),
                "size": int(c.get("size") or 0),
                "label": str(c.get("label") or ""),
                "top_files": c.get("top_files") or [],
                "top_symbols": c.get("top_symbols") or [],
            }
        )
    systems.sort(key=lambda s: int(s.get("cluster_id") or 0))

    pairs: dict[tuple[int, int], _AdjAggDirected] = {}
    for a, b, w in directed_arcs:
        a_i = int(a)
        b_i = int(b)
        if a_i == b_i:
            continue
        ca = chunk_id_to_system.get(a_i)
        cb = chunk_id_to_system.get(b_i)
        if ca is None or cb is None:
            continue
        ca_i = int(ca)
        cb_i = int(cb)
        if ca_i == cb_i:
            continue
        if ca_i not in kept_cluster_ids or cb_i not in kept_cluster_ids:
            continue

        key = (int(ca_i), int(cb_i))
        agg = pairs.get(key)
        if agg is None:
            agg = _AdjAggDirected(
                edge_count=0,
                weight_sum=0.0,
                weight_max=0.0,
                evidence=[],
            )
            pairs[key] = agg

        wf = float(w)
        agg.edge_count = int(agg.edge_count) + 1
        agg.weight_sum = float(agg.weight_sum) + float(wf)
        if float(wf) > float(agg.weight_max):
            agg.weight_max = float(wf)

        it_a = item_by_chunk_id.get(a_i)
        it_b = item_by_chunk_id.get(b_i)
        if it_a is None or it_b is None:
            continue
        ev = _AdjEvidenceDirected(
            source_chunk_id=int(a_i),
            target_chunk_id=int(b_i),
            w=float(wf),
            kind="directed_knn",
            source_path=str(it_a.path),
            target_path=str(it_b.path),
            source_symbol=it_a.symbol if it_a.symbol is not None else None,
            target_symbol=it_b.symbol if it_b.symbol is not None else None,
            source_start_line=int(it_a.start_line),
            source_end_line=int(it_a.end_line),
            target_start_line=int(it_b.start_line),
            target_end_line=int(it_b.end_line),
        )
        agg.evidence.append(ev)
        agg.evidence.sort(
            key=lambda e: (
                -float(e.w),
                int(e.source_chunk_id),
                int(e.target_chunk_id),
            )
        )
        if len(agg.evidence) > evidence_k_i:
            agg.evidence = agg.evidence[:evidence_k_i]

    links_before = int(len(pairs))

    # Outgoing neighbor cap: keep top-M outgoing neighbors per source system.
    neighbors: dict[int, list[tuple[int, float, float, int]]] = {}
    for (src, dst), agg in pairs.items():
        neighbors.setdefault(int(src), []).append(
            (
                int(dst),
                float(agg.weight_sum),
                float(agg.weight_max),
                int(agg.edge_count),
            )
        )

    top_out: dict[int, set[int]] = {}
    for sys_id, nbrs in neighbors.items():
        ordered = sorted(
            nbrs,
            key=lambda x: (-float(x[1]), -float(x[2]), -int(x[3]), int(x[0])),
        )
        keep = ordered[: max_neighbors_i]
        top_out[int(sys_id)] = {int(n) for n, _, _, _ in keep}

    kept_pairs: list[tuple[int, int]] = []
    for (src, dst) in pairs.keys():
        if int(dst) in top_out.get(int(src), set()):
            kept_pairs.append((int(src), int(dst)))

    kept_pairs.sort(
        key=lambda uv: (
            -float(pairs[uv].weight_sum),
            -float(pairs[uv].weight_max),
            -int(pairs[uv].edge_count),
            int(uv[0]),
            int(uv[1]),
        )
    )

    links: list[dict[str, object]] = []
    for src, dst in kept_pairs:
        agg = pairs[(src, dst)]
        links.append(
            {
                "source": int(src),
                "target": int(dst),
                "edge_count": int(agg.edge_count),
                "weight_sum": float(agg.weight_sum),
                "weight_max": float(agg.weight_max),
                "evidence": [
                    {
                        "source_chunk_id": int(ev.source_chunk_id),
                        "target_chunk_id": int(ev.target_chunk_id),
                        "w": float(ev.w),
                        "kind": str(ev.kind),
                        "source_path": str(ev.source_path),
                        "target_path": str(ev.target_path),
                        "source_symbol": (
                            ev.source_symbol if ev.source_symbol is not None else None
                        ),
                        "target_symbol": (
                            ev.target_symbol if ev.target_symbol is not None else None
                        ),
                        "source_start_line": int(ev.source_start_line),
                        "source_end_line": int(ev.source_end_line),
                        "target_start_line": int(ev.target_start_line),
                        "target_end_line": int(ev.target_end_line),
                    }
                    for ev in agg.evidence
                ],
            }
        )

    payload: dict[str, object] = {
        "schema_version": "snapshot.chunk_systems.system_adjacency_directed.v1",
        "schema_revision": "2026-02-19",
        "source": "snapshot.chunk_systems.json",
        "directed": True,
        "params": {
            "edge_rule": "directed_knn",
            "k": int(k),
            "tau": float(tau),
            "evidence_k": int(evidence_k_i),
            "max_neighbors_per_system": int(max_neighbors_i),
            "min_cluster_size": int(min_cluster_size),
            "link_score": "weight_sum",
        },
        "systems": systems,
        "links": links,
        "truncation": {
            "method": "top_out_neighbors_per_system",
            "max_neighbors_per_system": int(max_neighbors_i),
            "links_before": int(links_before),
            "links_after": int(len(links)),
        },
    }
    return payload


__all__ = [
    "SnapshotItemLike",
    "build_system_adjacency_json",
    "build_system_adjacency_json_directed",
    "build_system_groups_json",
    "build_system_groups_json_from_chunk_edges",
    "build_system_groups_json_from_directed_arcs",
    "iter_graph_edges_jsonl",
    "iter_graph_nodes_jsonl",
]
