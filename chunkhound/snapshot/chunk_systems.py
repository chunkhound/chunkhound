"""Experimental snapshot chunk-level community detection (mutual kNN + Leiden).

This module is intentionally independent of the project's persistent database
providers: it operates on in-memory embeddings already fetched for a snapshot
scope and uses an in-memory DuckDB connection (VSS + HNSW) only as an ANN engine.
"""

from __future__ import annotations

from collections.abc import Callable
from collections import Counter
from dataclasses import dataclass
import math
import posixpath
from typing import Protocol

import duckdb
from loguru import logger

from chunkhound.snapshot.partitioning import (
    SystemComponent,
    SystemEdge,
    SystemNode,
    partition_connected_components,
    partition_leiden,
    partition_leiden_with_membership,
)


class SnapshotItemLike(Protocol):
    chunk_id: int
    path: str
    symbol: str | None
    start_line: int
    end_line: int


@dataclass(frozen=True)
class _Edge:
    a: int
    b: int
    w: float


@dataclass(frozen=True)
class ChunkSystemsGraphContext:
    edges: list[tuple[int, int, float]]
    strict_edge_keys: set[tuple[int, int]]
    chunk_id_to_system: dict[int, int]
    directed_arcs: list[tuple[int, int, float]] | None = None


def _validate_embeddings(
    *, items: list[SnapshotItemLike], embeddings: list[list[float]]
) -> tuple[int, list[int]]:
    if len(items) != len(embeddings):
        raise ValueError("items and embeddings must have identical length")

    chunk_ids: list[int] = []
    seen: set[int] = set()
    for it in items:
        cid = int(it.chunk_id)
        if cid in seen:
            raise ValueError(f"Duplicate chunk_id: {cid}")
        seen.add(cid)
        chunk_ids.append(cid)

    if not embeddings:
        return 0, chunk_ids

    dims = len(embeddings[0])
    if dims <= 0:
        raise ValueError("Embeddings must be non-empty vectors")
    for i, vec in enumerate(embeddings):
        if len(vec) != dims:
            raise ValueError(
                "Embedding vectors must have identical dimensions "
                f"(index={i}, expected={dims}, got={len(vec)})"
            )
    return dims, chunk_ids


def _load_duckdb_vss(con: duckdb.DuckDBPyConnection) -> None:
    # Best-effort install (may be offline or already installed); LOAD must work.
    try:
        con.execute("INSTALL vss")
    except Exception:
        pass
    try:
        con.execute("LOAD vss")
    except Exception as exc:
        raise RuntimeError(
            "DuckDB VSS extension failed to load. "
            "Ensure DuckDB can access the 'vss' extension (INSTALL/LOAD vss)."
        ) from exc


def _build_ann_table(
    *,
    con: duckdb.DuckDBPyConnection,
    chunk_ids: list[int],
    embeddings: list[list[float]],
    dims: int,
) -> None:
    import pandas as pd

    con.execute(
        f"""
        CREATE TABLE scope_embeddings(
            chunk_id INTEGER PRIMARY KEY,
            embedding FLOAT[{int(dims)}]
        )
        """
    )

    df = pd.DataFrame(
        {
            "chunk_id": [int(x) for x in chunk_ids],
            "embedding": [list(v) for v in embeddings],
        }
    )
    con.register("df_scope", df)
    con.execute(
        f"""
        INSERT INTO scope_embeddings
        SELECT chunk_id, embedding::FLOAT[{int(dims)}]
        FROM df_scope
        """
    )

    con.execute(
        """
        CREATE INDEX idx_scope_hnsw
        ON scope_embeddings
        USING HNSW (embedding)
        WITH (metric='cosine')
        """
    )


def _directed_knn_neighbors(
    *,
    con: duckdb.DuckDBPyConnection,
    chunk_ids: list[int],
    embeddings: list[list[float]],
    dims: int,
    k: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[int, dict[int, float]]:
    n = len(chunk_ids)
    eff_k = min(int(k), max(0, n - 1))
    if eff_k <= 0:
        return {int(cid): {} for cid in chunk_ids}

    # Some ANN planners may still consider the "self" row when applying LIMIT even
    # if a WHERE filter is present, resulting in k-1 returned rows. Oversample
    # slightly and truncate after filtering to reliably return eff_k neighbors.
    query_k = min(int(n), int(eff_k) + 2)

    query = f"""
        SELECT chunk_id,
               array_cosine_distance(embedding, ?::FLOAT[{int(dims)}]) AS distance
        FROM scope_embeddings
        WHERE chunk_id != ?
        ORDER BY distance ASC, chunk_id ASC
        LIMIT ?
    """

    neighbors: dict[int, dict[int, float]] = {}
    shortfall_nodes = 0
    min_returned = eff_k
    for idx, (cid, vec) in enumerate(zip(chunk_ids, embeddings)):
        rows = con.execute(query, [list(vec), int(cid), int(query_k)]).fetchall()
        out: dict[int, float] = {}
        for nbr_id, dist in rows:
            if int(nbr_id) == int(cid):
                continue
            sim = 1.0 - float(dist)
            out[int(nbr_id)] = float(sim)
            if len(out) >= eff_k:
                break

        if (n - 1) >= eff_k and len(out) < eff_k:
            shortfall_nodes += 1
            min_returned = min(min_returned, len(out))

        neighbors[int(cid)] = out

        if progress_callback is not None and ((idx + 1) % 100 == 0 or (idx + 1) == n):
            try:
                progress_callback(int(idx + 1), int(n))
            except Exception:
                pass

        if (idx + 1) % 1000 == 0:
            logger.info(
                f"Chunk systems: kNN queried {idx + 1}/{n} nodes "
                f"({(idx + 1) / float(n):.0%})"
            )

    if shortfall_nodes > 0:
        logger.warning(
            "Chunk systems: kNN neighbor shortfall for "
            f"{shortfall_nodes}/{n} nodes (expected {eff_k}, min_returned={min_returned})"
        )
    return neighbors


def _mutual_edges_from_neighbors(
    *, neighbors: dict[int, dict[int, float]], tau: float
) -> list[_Edge]:
    edges: dict[tuple[int, int], float] = {}
    tau_f = float(tau)
    for u, nbrs in neighbors.items():
        for v, sim_uv in nbrs.items():
            if float(sim_uv) < tau_f:
                continue
            sim_vu = neighbors.get(int(v), {}).get(int(u))
            if sim_vu is None or float(sim_vu) < tau_f:
                continue
            a, b = (int(u), int(v)) if int(u) < int(v) else (int(v), int(u))
            edges[(a, b)] = float(min(float(sim_uv), float(sim_vu)))

    out = [_Edge(a=a, b=b, w=w) for (a, b), w in edges.items()]
    out.sort(key=lambda e: (e.a, e.b, -float(e.w)))
    return out


def _augment_edges_min_degree(
    *,
    chunk_ids: list[int],
    neighbors: dict[int, dict[int, float]],
    strict_edges: list[_Edge],
    min_degree: int,
    fallback_tau: float,
    fallback_key_by_id: dict[int, str] | None = None,
) -> tuple[list[_Edge], int, int]:
    """Augment strict mutual-kNN edges to reduce false negatives.

    This keeps strict mutual edges as-is, then (optionally) adds additional
    undirected edges from the directed kNN lists for nodes that have too few
    strict edges.

    Returns:
        (all_edges, fallback_edges_added, degree_zero_nodes_after)
    """
    min_degree_i = max(0, int(min_degree))
    tau_f = float(fallback_tau)

    # Map undirected edge -> weight (keep the strongest weight if duplicated).
    edge_map: dict[tuple[int, int], float] = {}
    degree: dict[int, int] = {int(cid): 0 for cid in chunk_ids}

    def _add_edge(*, u: int, v: int, w: float, enforce_key: bool) -> bool:
        if u == v:
            return False
        if enforce_key and fallback_key_by_id is not None:
            ku = fallback_key_by_id.get(int(u))
            kv = fallback_key_by_id.get(int(v))
            if ku is None or kv is None or str(ku) != str(kv):
                return False
        a, b = (int(u), int(v)) if int(u) < int(v) else (int(v), int(u))
        key = (a, b)
        if key in edge_map:
            return False
        edge_map[key] = float(w)
        degree[a] = int(degree.get(a, 0)) + 1
        degree[b] = int(degree.get(b, 0)) + 1
        return True

    for e in strict_edges:
        # Strict mutual edges must not be filtered by fallback path constraints.
        _add_edge(u=int(e.a), v=int(e.b), w=float(e.w), enforce_key=False)

    fallback_added = 0
    if min_degree_i > 0:
        for u in sorted(int(cid) for cid in chunk_ids):
            if int(degree.get(int(u), 0)) >= min_degree_i:
                continue
            cand = neighbors.get(int(u), {})
            # Deterministic ordering: strongest similarity first, then stable tie-break.
            ordered = sorted(
                ((int(v), float(sim)) for v, sim in cand.items()),
                key=lambda x: (-float(x[1]), int(x[0])),
            )
            for v, sim_uv in ordered:
                if float(sim_uv) < tau_f:
                    continue
                if int(degree.get(int(u), 0)) >= min_degree_i:
                    break
                if _add_edge(u=int(u), v=int(v), w=float(sim_uv), enforce_key=True):
                    fallback_added += 1

    all_edges = [_Edge(a=a, b=b, w=w) for (a, b), w in edge_map.items()]
    all_edges.sort(key=lambda e: (e.a, e.b, -float(e.w)))
    degree_zero = int(sum(1 for cid in chunk_ids if int(degree.get(int(cid), 0)) <= 0))
    return all_edges, int(fallback_added), int(degree_zero)


def _partition_quality(
    *, nodes: list[SystemNode], components: list[SystemComponent]
) -> tuple[int, int, int]:
    if not nodes or not components:
        return (0, 0, 0)
    node_count = len(nodes)
    system_count = int(len(components))
    largest_nodes = int(max((len(c.node_keys) for c in components), default=0))
    singleton_nodes = int(sum(1 for c in components if len(c.node_keys) == 1))
    non_singleton_nodes = int(node_count - singleton_nodes)
    return (non_singleton_nodes, -largest_nodes, -system_count)


def _comb2(n: int) -> int:
    n_i = int(n)
    if n_i <= 1:
        return 0
    return int(n_i * (n_i - 1) // 2)


def _adjusted_rand_index(*, labels_a: list[int], labels_b: list[int]) -> float:
    """Adjusted Rand Index (ARI) between two clusterings.

    Lightweight implementation to avoid pulling in sklearn as a dependency.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("ARI inputs must have identical length")

    n = int(len(labels_a))
    if n <= 1:
        return 1.0

    contingency: dict[tuple[int, int], int] = {}
    row_sum: dict[int, int] = {}
    col_sum: dict[int, int] = {}

    for a, b in zip(labels_a, labels_b, strict=True):
        ai = int(a)
        bi = int(b)
        row_sum[ai] = int(row_sum.get(ai, 0)) + 1
        col_sum[bi] = int(col_sum.get(bi, 0)) + 1
        key = (ai, bi)
        contingency[key] = int(contingency.get(key, 0)) + 1

    sum_comb_c = float(sum(_comb2(v) for v in contingency.values()))
    sum_comb_a = float(sum(_comb2(v) for v in row_sum.values()))
    sum_comb_b = float(sum(_comb2(v) for v in col_sum.values()))
    comb_n = float(_comb2(n))

    if comb_n <= 0.0:
        return 1.0

    expected = (sum_comb_a * sum_comb_b) / comb_n
    max_index = 0.5 * (sum_comb_a + sum_comb_b)
    denom = float(max_index - expected)
    if denom == 0.0:
        return 1.0
    return float((sum_comb_c - expected) / denom)


def _avg_pairwise_ari(*, memberships: list[list[int]]) -> tuple[float, float]:
    if len(memberships) <= 1:
        return (1.0, 1.0)
    total = 0.0
    pairs = 0
    min_ari = 1.0
    for i in range(len(memberships)):
        for j in range(i + 1, len(memberships)):
            ari = _adjusted_rand_index(labels_a=memberships[i], labels_b=memberships[j])
            total += float(ari)
            pairs += 1
            if float(ari) < float(min_ari):
                min_ari = float(ari)
    if pairs <= 0:
        return (1.0, 1.0)
    return (float(total / float(pairs)), float(min_ari))


def _derive_cluster_count_bounds(
    *, node_count: int, min_avg_size: int, max_avg_size: int
) -> tuple[int, int]:
    n = int(node_count)
    if n <= 0:
        return (0, 0)
    min_avg = int(min_avg_size)
    max_avg = int(max_avg_size)
    if min_avg <= 0 or max_avg <= 0:
        raise ValueError("min_avg_size and max_avg_size must be positive")
    if min_avg > max_avg:
        raise ValueError("min_avg_size must be <= max_avg_size")

    # Average cluster size is ~ n / k.
    # Enforce: min_avg <= n/k <= max_avg.
    min_clusters = int(max(1, math.ceil(float(n) / float(max_avg))))
    max_clusters = int(max(1, math.floor(float(n) / float(min_avg))))
    max_clusters = int(min(n, max_clusters))
    min_clusters = int(min(min_clusters, max_clusters))
    return (min_clusters, max_clusters)


def _partition_stats(
    *, node_count: int, components: list[SystemComponent]
) -> dict[str, float | int]:
    n = int(node_count)
    k = int(len(components))
    if n <= 0 or k <= 0:
        return {
            "clusters": int(k),
            "singletons": 0,
            "largest_cluster": 0,
            "singleton_frac_nodes": 0.0,
            "largest_frac_nodes": 0.0,
        }
    largest = int(max((len(c.node_keys) for c in components), default=0))
    singletons = int(sum(1 for c in components if len(c.node_keys) == 1))
    singleton_nodes = int(singletons)  # singleton cluster -> 1 node
    return {
        "clusters": int(k),
        "singletons": int(singletons),
        "largest_cluster": int(largest),
        "singleton_frac_nodes": float(singleton_nodes) / float(n),
        "largest_frac_nodes": float(largest) / float(n),
    }


def _passes_bounds(
    *,
    stats: dict[str, float | int],
    min_clusters: int,
    max_clusters: int,
    max_singleton_frac_nodes: float,
    max_largest_frac_nodes: float,
) -> bool:
    clusters = int(stats.get("clusters") or 0)
    singleton_frac = float(stats.get("singleton_frac_nodes") or 0.0)
    largest_frac = float(stats.get("largest_frac_nodes") or 0.0)
    if clusters < int(min_clusters) or clusters > int(max_clusters):
        return False
    if float(singleton_frac) > float(max_singleton_frac_nodes):
        return False
    if float(largest_frac) > float(max_largest_frac_nodes):
        return False
    return True


def _choose_partition(
    *,
    nodes: list[SystemNode],
    edges: list[SystemEdge],
    partitioner: str,
    leiden_resolution: float,
    leiden_seed: int,
    leiden_resolutions_auto: list[float] | None,
    leiden_auto_selector: str = "legacy",
    leiden_auto_stability_seeds: list[int] | None = None,
    leiden_auto_stability_min: float = 0.90,
    leiden_auto_min_avg_size: int = 10,
    leiden_auto_max_avg_size: int = 80,
    leiden_auto_max_largest_frac_nodes: float = 0.20,
    leiden_auto_max_singleton_frac_nodes: float = 0.02,
    record_partition_sweep: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[list[SystemComponent], dict[str, int], dict[str, object]]:
    requested = str(partitioner).strip().lower()
    if requested not in {"auto", "cc", "leiden"}:
        raise ValueError("partitioner must be one of: auto, cc, leiden")

    if requested == "cc":
        if progress_callback is not None:
            try:
                progress_callback(0, 1, "cc")
            except Exception:
                pass
        components, node_to_system = partition_connected_components(
            nodes=nodes, edges=edges
        )
        if progress_callback is not None:
            try:
                progress_callback(1, 1, "cc done")
            except Exception:
                pass
        return (
            components,
            node_to_system,
            {"method": "cc", "resolution": None, "quality": None},
        )

    cc_components, cc_node_to_system = partition_connected_components(
        nodes=nodes, edges=edges
    )
    cc_quality = _partition_quality(nodes=nodes, components=cc_components)

    resolutions: list[float]
    if requested == "auto":
        resolutions = (
            [float(x) for x in (leiden_resolutions_auto or [])]
            if leiden_resolutions_auto is not None
            else [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
        )
        if not resolutions:
            resolutions = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    else:
        resolutions = [float(leiden_resolution)]

    pending_partition_sweep: dict[str, object] | None = None

    auto_selector = str(leiden_auto_selector or "legacy").strip().lower()
    if requested == "auto" and auto_selector == "objective_stable":
        seeds = (
            [int(s) for s in (leiden_auto_stability_seeds or [])]
            if leiden_auto_stability_seeds is not None
            else [0, 1, 2]
        )
        if not seeds:
            seeds = [int(leiden_seed)]

        min_clusters, max_clusters = _derive_cluster_count_bounds(
            node_count=len(nodes),
            min_avg_size=int(leiden_auto_min_avg_size),
            max_avg_size=int(leiden_auto_max_avg_size),
        )

        total_steps_obj = int(1 + len(resolutions) * len(seeds))
        step_done_obj = 0
        if progress_callback is not None:
            try:
                progress_callback(step_done_obj, total_steps_obj, "cc baseline")
            except Exception:
                pass
        step_done_obj += 1
        if progress_callback is not None:
            try:
                progress_callback(step_done_obj, total_steps_obj, "cc baseline done")
            except Exception:
                pass

        sweep: dict[str, object] | None = None
        if record_partition_sweep:
            sweep = {
                "mode": "objective_stable",
                "node_count": int(len(nodes)),
                "edge_count": int(len(edges)),
                "bounds": {
                    "min_avg_size": int(leiden_auto_min_avg_size),
                    "max_avg_size": int(leiden_auto_max_avg_size),
                    "min_clusters": int(min_clusters),
                    "max_clusters": int(max_clusters),
                    "max_singleton_frac_nodes": float(leiden_auto_max_singleton_frac_nodes),
                    "max_largest_frac_nodes": float(leiden_auto_max_largest_frac_nodes),
                },
                "stability": {
                    "seeds": list(seeds),
                    "min_avg_ari": float(leiden_auto_stability_min),
                },
                "resolutions": [],
                "selected": None,
            }

        bounded: list[dict[str, object]] = []
        stable: list[dict[str, object]] = []

        for ridx, res in enumerate(resolutions, start=1):
            runs: list[dict[str, object]] = []
            memberships: list[list[int]] = []
            ordered_keys_ref: list[str] | None = None

            for sidx, seed in enumerate(seeds, start=1):
                if progress_callback is not None:
                    try:
                        progress_callback(
                            step_done_obj,
                            total_steps_obj,
                            f"leiden r={float(res):g} seed={int(seed)} ({ridx}/{len(resolutions)}:{sidx}/{len(seeds)})",
                        )
                    except Exception:
                        pass
                try:
                    lc, lmap, ordered_keys, membership, obj_q = partition_leiden_with_membership(
                        nodes=nodes, edges=edges, resolution=float(res), seed=int(seed)
                    )
                except ImportError:
                    if progress_callback is not None:
                        try:
                            progress_callback(
                                total_steps_obj, total_steps_obj, "leiden unavailable; using cc"
                            )
                        except Exception:
                            pass
                    return (
                        cc_components,
                        cc_node_to_system,
                        {"method": "cc", "resolution": None, "quality": None},
                    )

                if ordered_keys_ref is None:
                    ordered_keys_ref = list(ordered_keys)
                elif ordered_keys_ref != list(ordered_keys):
                    raise RuntimeError("Leiden membership ordering mismatch across runs")

                memberships.append([int(x) for x in membership])
                stats = _partition_stats(node_count=len(nodes), components=lc)
                passes = _passes_bounds(
                    stats=stats,
                    min_clusters=int(min_clusters),
                    max_clusters=int(max_clusters),
                    max_singleton_frac_nodes=float(leiden_auto_max_singleton_frac_nodes),
                    max_largest_frac_nodes=float(leiden_auto_max_largest_frac_nodes),
                )
                runs.append(
                    {
                        "resolution": float(res),
                        "seed": int(seed),
                        "objective_quality": float(obj_q),
                        "passes_bounds": bool(passes),
                        "stats": dict(stats),
                        "components": lc,
                        "node_to_system": lmap,
                    }
                )

                step_done_obj += 1
                if progress_callback is not None:
                    try:
                        progress_callback(
                            step_done_obj,
                            total_steps_obj,
                            f"leiden r={float(res):g} seed={int(seed)} done",
                        )
                    except Exception:
                        pass

            stability_avg, stability_min = _avg_pairwise_ari(memberships=memberships)

            runs_sorted = sorted(
                runs,
                key=lambda r: (
                    -float(r.get("objective_quality") or 0.0),
                    int(r.get("seed") or 0),
                ),
            )
            rep = runs_sorted[0]

            if sweep is not None:
                sweep_resolutions = sweep.get("resolutions")
                if isinstance(sweep_resolutions, list):
                    sweep_resolutions.append(
                        {
                            "resolution": float(res),
                            "stability_avg_ari": float(stability_avg),
                            "stability_min_ari": float(stability_min),
                            "rep_seed": int(rep.get("seed") or 0),
                            "rep_objective_quality": float(rep.get("objective_quality") or 0.0),
                            "rep_passes_bounds": bool(rep.get("passes_bounds") or False),
                            "rep_stats": rep.get("stats"),
                            "runs": [
                                {
                                    "seed": int(r.get("seed") or 0),
                                    "objective_quality": float(r.get("objective_quality") or 0.0),
                                    "passes_bounds": bool(r.get("passes_bounds") or False),
                                    "stats": r.get("stats"),
                                }
                                for r in runs_sorted
                            ],
                        }
                    )

            if bool(rep.get("passes_bounds") or False):
                candidate = {
                    "resolution": float(res),
                    "stability_avg_ari": float(stability_avg),
                    "stability_min_ari": float(stability_min),
                    "rep_objective_quality": float(rep.get("objective_quality") or 0.0),
                    "components": rep["components"],
                    "node_to_system": rep["node_to_system"],
                }
                bounded.append(candidate)
                if float(stability_avg) >= float(leiden_auto_stability_min):
                    stable.append(candidate)

        selected: dict[str, object] | None = None
        if stable:
            stable_sorted = sorted(
                stable,
                key=lambda c: (
                    -float(c.get("rep_objective_quality") or 0.0),
                    -float(c.get("stability_avg_ari") or 0.0),
                    float(c.get("resolution") or 0.0),
                ),
            )
            selected = stable_sorted[0]
        elif bounded:
            bounded_sorted = sorted(
                bounded,
                key=lambda c: (
                    -float(c.get("stability_avg_ari") or 0.0),
                    -float(c.get("rep_objective_quality") or 0.0),
                    float(c.get("resolution") or 0.0),
                ),
            )
            selected = bounded_sorted[0]

        if selected is not None:
            res_used = float(selected.get("resolution") or 0.0)
            components = selected["components"]
            node_to_system = selected["node_to_system"]
            quality = _partition_quality(nodes=nodes, components=components)
            chosen: dict[str, object] = {
                "method": "leiden",
                "resolution": float(res_used),
                "quality": {
                    "non_singleton_nodes": int(quality[0]),
                    "largest_nodes_neg": int(quality[1]),
                    "system_count_neg": int(quality[2]),
                },
            }
            if sweep is not None:
                sweep["selected"] = {
                    "resolution": float(res_used),
                    "stability_avg_ari": float(selected.get("stability_avg_ari") or 0.0),
                    "stability_min_ari": float(selected.get("stability_min_ari") or 0.0),
                    "rep_objective_quality": float(selected.get("rep_objective_quality") or 0.0),
                }
                chosen["_partition_sweep"] = sweep
            return (components, node_to_system, chosen)

        # Nothing passed bounds: fall back to legacy selection logic below.
        if sweep is not None:
            sweep["fallback"] = "legacy"
            pending_partition_sweep = sweep
    total_steps = int(1 + len(resolutions))
    step_done = 0
    if progress_callback is not None:
        try:
            progress_callback(step_done, total_steps, "cc baseline")
        except Exception:
            pass

    best: tuple[str, float, tuple[int, int, int], list[SystemComponent], dict[str, int]]
    best = ("cc", 0.0, cc_quality, cc_components, cc_node_to_system)
    step_done += 1
    if progress_callback is not None:
        try:
            progress_callback(step_done, total_steps, "cc baseline done")
        except Exception:
            pass

    for idx, res in enumerate(resolutions, start=1):
        if progress_callback is not None:
            try:
                progress_callback(
                    step_done, total_steps, f"leiden r={float(res):g} ({idx}/{len(resolutions)})"
                )
            except Exception:
                pass
        try:
            lc, lmap = partition_leiden(
                nodes=nodes, edges=edges, resolution=float(res), seed=int(leiden_seed)
            )
        except ImportError:
            if requested == "leiden":
                if progress_callback is not None:
                    try:
                        progress_callback(total_steps, total_steps, "leiden unavailable")
                    except Exception:
                        pass
                raise
            step_done = total_steps
            if progress_callback is not None:
                try:
                    progress_callback(step_done, total_steps, "leiden unavailable; using cc")
                except Exception:
                    pass
            break
        q = _partition_quality(nodes=nodes, components=lc)
        candidate = ("leiden", float(res), q, lc, lmap)
        if candidate[2] > best[2]:
            best = candidate
        elif candidate[2] == best[2]:
            if best[0] == "cc" and candidate[0] == "leiden":
                best = candidate
            elif best[0] == "leiden" and candidate[0] == "leiden":
                if candidate[1] < best[1]:
                    best = candidate
        step_done += 1
        if progress_callback is not None:
            try:
                progress_callback(step_done, total_steps, f"leiden r={float(res):g} done")
            except Exception:
                pass

    method, res_used, quality, components, node_to_system = best
    if progress_callback is not None and step_done < total_steps:
        try:
            progress_callback(total_steps, total_steps, f"selected {method}")
        except Exception:
            pass
    chosen = {
        "method": method,
        "resolution": float(res_used) if method == "leiden" else None,
        "quality": {
            "non_singleton_nodes": int(quality[0]),
            "largest_nodes_neg": int(quality[1]),
            "system_count_neg": int(quality[2]),
        },
    }
    if pending_partition_sweep is not None:
        chosen["_partition_sweep"] = pending_partition_sweep
    return (components, node_to_system, chosen)


def compute_chunk_systems(
    *,
    items: list[SnapshotItemLike],
    embeddings: list[list[float]],
    k: int,
    tau: float,
    max_nodes: int,
    partitioner: str,
    leiden_resolution: float,
    leiden_seed: int,
    leiden_resolutions_auto: list[float] | None,
    capture_directed_arcs: bool = False,
    leiden_auto_selector: str = "legacy",
    leiden_auto_stability_seeds: list[int] | None = None,
    leiden_auto_stability_min: float = 0.90,
    leiden_auto_min_avg_size: int = 10,
    leiden_auto_max_avg_size: int = 80,
    leiden_auto_max_largest_frac_nodes: float = 0.20,
    leiden_auto_max_singleton_frac_nodes: float = 0.02,
    record_partition_sweep: bool = False,
    edge_min_degree: int = 0,
    edge_fallback_tau: float | None = None,
    edge_fallback_path_mode: str = "any",
    knn_progress_callback: Callable[[int, int], None] | None = None,
    partition_progress_callback: Callable[[int, int, str], None] | None = None,
    graph_context_callback: Callable[[ChunkSystemsGraphContext], None] | None = None,
) -> dict[str, object]:
    if int(max_nodes) <= 0:
        raise ValueError("max_nodes must be positive")

    if len(items) > int(max_nodes):
        raise ValueError(
            f"Chunk systems max_nodes exceeded: nodes={len(items)} "
            f"> max_nodes={int(max_nodes)}"
        )

    if int(k) <= 0:
        raise ValueError("k must be positive")
    if not (0.0 <= float(tau) <= 1.0):
        raise ValueError("tau must be between 0.0 and 1.0 (inclusive)")
    if int(edge_min_degree) < 0:
        raise ValueError("edge_min_degree must be >= 0")
    if edge_fallback_tau is not None and not (0.0 <= float(edge_fallback_tau) <= 1.0):
        raise ValueError("edge_fallback_tau must be between 0.0 and 1.0 (inclusive)")
    mode = str(edge_fallback_path_mode or "any").strip().lower()
    if mode not in {"any", "same_file", "same_dir"}:
        raise ValueError("edge_fallback_path_mode must be one of: any, same_file, same_dir")
    if float(leiden_resolution) <= 0.0:
        raise ValueError("leiden_resolution must be positive")

    dims, chunk_ids = _validate_embeddings(items=items, embeddings=embeddings)
    nodes_count = int(len(chunk_ids))

    payload: dict[str, object] = {
        "schema_version": "snapshot.chunk_systems.v1",
        "schema_revision": "2026-02-17",
        "run": {},
        "params": {
            "edge_rule": "mutual_knn",
            "k": int(k),
            "tau": float(tau),
            "ann_backend": "duckdb_hnsw_inmemory",
            "partitioner": {
                "requested": str(partitioner),
                "by_view": {},
                "leiden": {
                    "seed": int(leiden_seed),
                    "resolutions_auto": (
                        list(leiden_resolutions_auto)
                        if leiden_resolutions_auto is not None
                        else [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
                    ),
                },
            },
            "max_nodes": int(max_nodes),
        },
        "counts": {
            "nodes": int(nodes_count),
            "directed_knn_arcs": 0,
            "mutual_edges": 0,
            "edges": 0,
            "fallback_edges": 0,
            "degree_zero_nodes": 0,
            "clusters": 0,
            "singletons": 0,
            "largest_cluster": 0,
        },
        "clusters": [],
    }

    if nodes_count <= 0:
        return payload
    if dims <= 0:
        return payload

    con = duckdb.connect(":memory:")
    try:
        _load_duckdb_vss(con)
        _build_ann_table(con=con, chunk_ids=chunk_ids, embeddings=embeddings, dims=dims)
        neighbors = _directed_knn_neighbors(
            con=con,
            chunk_ids=chunk_ids,
            embeddings=embeddings,
            dims=dims,
            k=int(k),
            progress_callback=knn_progress_callback,
        )
    finally:
        try:
            con.close()
        except Exception:
            pass

    directed_knn_arcs = int(sum(len(v) for v in neighbors.values()))
    strict_edges = _mutual_edges_from_neighbors(neighbors=neighbors, tau=float(tau))
    strict_edges_count = int(len(strict_edges))
    strict_edge_keys: set[tuple[int, int]] = {
        (int(e.a), int(e.b)) if int(e.a) < int(e.b) else (int(e.b), int(e.a))
        for e in strict_edges
    }

    directed_arcs: list[tuple[int, int, float]] | None = None
    if graph_context_callback is not None and bool(capture_directed_arcs):
        tau_f = float(tau)
        directed_arcs = []
        for u, nbrs in neighbors.items():
            u_i = int(u)
            for v, sim_uv in nbrs.items():
                if float(sim_uv) < tau_f:
                    continue
                directed_arcs.append((u_i, int(v), float(sim_uv)))

    fallback_tau_used = float(edge_fallback_tau) if edge_fallback_tau is not None else float(tau)
    if int(edge_min_degree) > 0:
        fallback_key_by_id: dict[int, str] | None = None
        if mode != "any":
            item_by_id = {int(it.chunk_id): it for it in items}
            path_by_id = {
                int(cid): str(item_by_id[int(cid)].path).replace("\\", "/")
                for cid in chunk_ids
                if int(cid) in item_by_id
            }
            if mode == "same_file":
                fallback_key_by_id = dict(path_by_id)
            else:
                # same_dir: parent directory of the file path
                fallback_key_by_id = {
                    int(cid): posixpath.dirname(str(path)) for cid, path in path_by_id.items()
                }

        edges, fallback_edges_added, degree_zero_nodes = _augment_edges_min_degree(
            chunk_ids=chunk_ids,
            neighbors=neighbors,
            strict_edges=strict_edges,
            min_degree=int(edge_min_degree),
            fallback_tau=float(fallback_tau_used),
            fallback_key_by_id=fallback_key_by_id,
        )
        params = payload.get("params")
        if isinstance(params, dict):
            params["edge_fallback"] = {
                "mode": "min_degree",
                "min_degree": int(edge_min_degree),
                "tau": float(fallback_tau_used),
                "path_mode": str(mode),
            }
    else:
        edges = strict_edges
        fallback_edges_added = 0
        # Degree-zero nodes from strict edges only.
        deg: dict[int, int] = {int(cid): 0 for cid in chunk_ids}
        for e in edges:
            deg[int(e.a)] = int(deg.get(int(e.a), 0)) + 1
            deg[int(e.b)] = int(deg.get(int(e.b), 0)) + 1
        degree_zero_nodes = int(sum(1 for cid in chunk_ids if int(deg.get(int(cid), 0)) <= 0))

    nodes = [
        SystemNode(node_key=str(int(cid)), change_indexes=[int(cid)], vector=[])
        for cid in chunk_ids
    ]
    sys_edges = [
        SystemEdge(
            node_a=str(int(e.a)),
            node_b=str(int(e.b)),
            w_sem=float(e.w),
            w_struct=None,
            w_total=float(e.w),
        )
        for e in edges
    ]

    components, _node_to_system, chosen = _choose_partition(
        nodes=nodes,
        edges=sys_edges,
        partitioner=str(partitioner),
        leiden_resolution=float(leiden_resolution),
        leiden_seed=int(leiden_seed),
        leiden_resolutions_auto=leiden_resolutions_auto,
        leiden_auto_selector=str(leiden_auto_selector),
        leiden_auto_stability_seeds=leiden_auto_stability_seeds,
        leiden_auto_stability_min=float(leiden_auto_stability_min),
        leiden_auto_min_avg_size=int(leiden_auto_min_avg_size),
        leiden_auto_max_avg_size=int(leiden_auto_max_avg_size),
        leiden_auto_max_largest_frac_nodes=float(leiden_auto_max_largest_frac_nodes),
        leiden_auto_max_singleton_frac_nodes=float(leiden_auto_max_singleton_frac_nodes),
        record_partition_sweep=bool(record_partition_sweep),
        progress_callback=partition_progress_callback,
    )

    if graph_context_callback is not None:
        chunk_id_to_system: dict[int, int] = {}
        if isinstance(_node_to_system, dict):
            for k, v in _node_to_system.items():
                try:
                    chunk_id_to_system[int(k)] = int(v)
                except Exception:
                    continue
        graph_context_callback(
            ChunkSystemsGraphContext(
                edges=[(int(e.a), int(e.b), float(e.w)) for e in edges],
                strict_edge_keys=set(strict_edge_keys),
                chunk_id_to_system=chunk_id_to_system,
                directed_arcs=directed_arcs,
            )
        )

    if bool(record_partition_sweep) and isinstance(chosen, dict):
        sweep = chosen.get("_partition_sweep")
        if sweep is not None:
            payload["_partition_sweep"] = sweep

    params = payload.get("params")
    if isinstance(params, dict):
        part = params.get("partitioner")
        if isinstance(part, dict):
            by_view = part.get("by_view")
            if isinstance(by_view, dict):
                by_view["chunk"] = {
                    "method": chosen.get("method"),
                    "resolution": chosen.get("resolution"),
                    "repair_applied": False,
                    "quality": chosen.get("quality"),
                }

    item_by_chunk_id = {int(it.chunk_id): it for it in items}

    clusters: list[dict[str, object]] = []
    singleton_count = 0
    largest_cluster = 0
    for comp in components:
        chunk_ids_cluster = [int(x) for x in (comp.change_indexes or [])]
        chunk_ids_cluster.sort()
        size = int(len(chunk_ids_cluster))
        if size == 1:
            singleton_count += 1
        if size > largest_cluster:
            largest_cluster = size

        file_counts = Counter()
        symbol_counts = Counter()
        for cid in chunk_ids_cluster:
            it = item_by_chunk_id.get(int(cid))
            if it is None:
                continue
            file_counts[str(it.path)] += 1
            symbol_counts[it.symbol] += 1

        top_files_pairs = [(p, int(c)) for p, c in file_counts.items()]
        top_files_pairs.sort(key=lambda x: (-int(x[1]), str(x[0])))
        top_symbols_pairs = [(s, int(c)) for s, c in symbol_counts.items()]
        top_symbols_pairs.sort(key=lambda x: (-int(x[1]), str(x[0] or "")))

        top_files = [{"path": str(p), "count": int(c)} for p, c in top_files_pairs[:10]]
        top_symbols = [
            {"symbol": s if s is not None else None, "count": int(c)}
            for s, c in top_symbols_pairs[:10]
        ]

        example_chunks: list[dict[str, object]] = []
        for cid in chunk_ids_cluster[:3]:
            it = item_by_chunk_id.get(int(cid))
            if it is None:
                continue
            example_chunks.append(
                {
                    "chunk_id": int(cid),
                    "path": str(it.path),
                    "symbol": it.symbol if it.symbol is not None else None,
                    "start_line": int(it.start_line),
                    "end_line": int(it.end_line),
                }
            )

        if top_files and top_symbols:
            label = (
                f"{top_files[0]['path']} · "
                f"{top_symbols[0]['symbol'] or 'no-symbol'}"
            )
        elif top_files:
            label = f"{top_files[0]['path']} · no-symbol"
        else:
            label = "empty"

        clusters.append(
            {
                "cluster_id": int(comp.system_id),
                "size": int(size),
                "label": str(label),
                "top_files": top_files,
                "top_symbols": top_symbols,
                "example_chunks": example_chunks,
                "chunk_ids": chunk_ids_cluster,
            }
        )

    clusters.sort(
        key=lambda c: (-int(c.get("size") or 0), int(c.get("cluster_id") or 0))
    )

    counts = payload.get("counts")
    if isinstance(counts, dict):
        counts["directed_knn_arcs"] = int(directed_knn_arcs)
        counts["mutual_edges"] = int(strict_edges_count)
        counts["edges"] = int(len(edges))
        counts["fallback_edges"] = int(fallback_edges_added)
        counts["degree_zero_nodes"] = int(degree_zero_nodes)
        counts["clusters"] = int(len(components))
        counts["singletons"] = int(singleton_count)
        counts["largest_cluster"] = int(largest_cluster)

    payload["clusters"] = clusters
    return payload


def build_chunk_systems_views(
    *, payload: dict[str, object], min_cluster_size: int
) -> tuple[dict[str, object], dict[str, object]]:
    """Build reporting views from a raw chunk-systems payload.

    This is intentionally a pure transformation: it does not recompute the graph
    or partitioning, it only filters the cluster list into:
      - a pruned view (clusters with size >= min_cluster_size)
      - a dropped view (clusters with size < min_cluster_size)

    The raw compute counts remain in `payload["counts"]`. View-local counts are
    recorded under `counts_view`, with additional metadata under `view` and
    `view_summary`.
    """
    min_size = int(min_cluster_size)
    if min_size < 1:
        raise ValueError("min_cluster_size must be >= 1")

    raw_clusters = payload.get("clusters") or []
    clusters_list: list[object]
    if isinstance(raw_clusters, list):
        clusters_list = raw_clusters
    else:
        clusters_list = []

    def _cluster_size(cluster: object) -> int:
        if not isinstance(cluster, dict):
            return 0
        size = cluster.get("size")
        if size is not None:
            try:
                return int(size)
            except Exception:
                return 0
        ids = cluster.get("chunk_ids")
        if isinstance(ids, list):
            return int(len(ids))
        return 0

    kept: list[object] = []
    dropped: list[object] = []
    kept_nodes = 0
    dropped_nodes = 0
    for c in clusters_list:
        sz = _cluster_size(c)
        if sz >= min_size:
            kept.append(c)
            kept_nodes += int(sz)
        else:
            dropped.append(c)
            dropped_nodes += int(sz)

    def _counts_view(clusters: list[object]) -> dict[str, int]:
        sizes = [_cluster_size(c) for c in clusters]
        return {
            "clusters": int(len(clusters)),
            "nodes": int(sum(int(s) for s in sizes)),
            "singletons": int(sum(1 for s in sizes if int(s) == 1)),
            "largest_cluster": int(max((int(s) for s in sizes), default=0)),
        }

    view_summary = {
        "kept_clusters": int(len(kept)),
        "dropped_clusters": int(len(dropped)),
        "kept_nodes": int(kept_nodes),
        "dropped_nodes": int(dropped_nodes),
    }

    pruned = dict(payload)
    pruned["clusters"] = list(kept)
    pruned["view"] = {
        "min_cluster_size": int(min_size),
        "kind": "pruned",
        "source": "snapshot.chunk_systems.json",
    }
    pruned["view_summary"] = dict(view_summary)
    pruned["counts_view"] = _counts_view(kept)

    dropped_payload = dict(payload)
    dropped_payload["clusters"] = list(dropped)
    dropped_payload["view"] = {
        "min_cluster_size": int(min_size),
        "kind": "dropped",
        "source": "snapshot.chunk_systems.json",
    }
    dropped_payload["view_summary"] = dict(view_summary)
    dropped_payload["counts_view"] = _counts_view(dropped)

    return pruned, dropped_payload


def render_chunk_systems_markdown(payload: dict[str, object]) -> str:
    run = payload.get("run") or {}
    params = payload.get("params") or {}
    counts = payload.get("counts") or {}
    clusters = payload.get("clusters") or []
    view = payload.get("view") or {}
    view_summary = payload.get("view_summary") or {}
    counts_view = payload.get("counts_view") or {}

    def _get(d: object, key: str, default: object = None) -> object:
        if not isinstance(d, dict):
            return default
        return d.get(key, default)

    schema_version = str(payload.get("schema_version") or "")
    schema_revision = str(payload.get("schema_revision") or "")
    scope_hash = str(_get(run, "scope_hash", "") or "")

    emb = _get(run, "embedding", {}) or {}
    emb_provider = _get(emb, "provider", None)
    emb_model = _get(emb, "model", None)
    emb_dims = _get(emb, "dims", None)
    emb_matryoshka = _get(emb, "matryoshka_dims", None)

    lines: list[str] = []
    lines.append("# Snapshot Chunk Systems")
    lines.append("")
    lines.append(f"- schema: `{schema_version}` (rev `{schema_revision}`)")
    if scope_hash:
        lines.append(f"- scope_hash: `{scope_hash}`")
    if emb_provider or emb_model or emb_dims:
        lines.append(
            "- embedding: "
            f"provider={emb_provider!r} model={emb_model!r} dims={emb_dims!r} "
            f"matryoshka_dims={emb_matryoshka!r}"
        )
    lines.append("")

    lines.append("## Params")
    lines.append("")
    lines.append(f"- edge_rule: `{_get(params, 'edge_rule', '')}`")
    lines.append(f"- k: `{_get(params, 'k', '')}`")
    lines.append(f"- tau: `{_get(params, 'tau', '')}`")
    lines.append(f"- ann_backend: `{_get(params, 'ann_backend', '')}`")
    lines.append(f"- max_nodes: `{_get(params, 'max_nodes', '')}`")
    part = _get(params, "partitioner", {}) or {}
    lines.append(f"- partitioner.requested: `{_get(part, 'requested', '')}`")
    lines.append("")

    if isinstance(view, dict) and view:
        lines.append("## View")
        lines.append("")
        lines.append(f"- kind: `{_get(view, 'kind', '')}`")
        lines.append(f"- min_cluster_size: `{_get(view, 'min_cluster_size', '')}`")
        if isinstance(view_summary, dict) and view_summary:
            lines.append(f"- kept_clusters: `{_get(view_summary, 'kept_clusters', '')}`")
            lines.append(f"- dropped_clusters: `{_get(view_summary, 'dropped_clusters', '')}`")
            lines.append(f"- kept_nodes: `{_get(view_summary, 'kept_nodes', '')}`")
            lines.append(f"- dropped_nodes: `{_get(view_summary, 'dropped_nodes', '')}`")
        if isinstance(counts_view, dict) and counts_view:
            lines.append(f"- counts_view.clusters: `{_get(counts_view, 'clusters', 0)}`")
            lines.append(f"- counts_view.nodes: `{_get(counts_view, 'nodes', 0)}`")
            lines.append(
                f"- counts_view.largest_cluster: `{_get(counts_view, 'largest_cluster', 0)}`"
            )
        lines.append(
            "- note: Counts below are raw compute for full scope; Top Clusters reflect this view."
        )
        lines.append("")

    lines.append("## Counts")
    lines.append("")
    lines.append(f"- nodes: `{_get(counts, 'nodes', 0)}`")
    lines.append(f"- directed_knn_arcs: `{_get(counts, 'directed_knn_arcs', 0)}`")
    lines.append(f"- mutual_edges: `{_get(counts, 'mutual_edges', 0)}`")
    if "edges" in counts:
        lines.append(f"- edges: `{_get(counts, 'edges', 0)}`")
    if "fallback_edges" in counts:
        lines.append(f"- fallback_edges: `{_get(counts, 'fallback_edges', 0)}`")
    if "degree_zero_nodes" in counts:
        lines.append(f"- degree_zero_nodes: `{_get(counts, 'degree_zero_nodes', 0)}`")
    lines.append(f"- clusters: `{_get(counts, 'clusters', 0)}`")
    lines.append(f"- singletons: `{_get(counts, 'singletons', 0)}`")
    lines.append(f"- largest_cluster: `{_get(counts, 'largest_cluster', 0)}`")
    lines.append("")

    lines.append("## Top Clusters")
    lines.append("")

    cluster_rows = [c for c in clusters if isinstance(c, dict)]
    cluster_rows.sort(
        key=lambda c: (-int(c.get("size") or 0), int(c.get("cluster_id") or 0))
    )

    for c in cluster_rows:
        cid = int(c.get("cluster_id") or 0)
        size = int(c.get("size") or 0)
        label = str(c.get("label") or "")
        lines.append(f"### Cluster {cid} (size: {size}) — {label}")
        lines.append("")

        top_files = c.get("top_files") or []
        if isinstance(top_files, list) and top_files:
            tf_parts = []
            for tf in top_files[:10]:
                if not isinstance(tf, dict):
                    continue
                path = tf.get("path")
                cnt = tf.get("count")
                tf_parts.append(f"`{path}` ({cnt})")
            if tf_parts:
                lines.append("- top_files: " + ", ".join(tf_parts))

        top_symbols = c.get("top_symbols") or []
        if isinstance(top_symbols, list) and top_symbols:
            ts_parts = []
            for ts in top_symbols[:10]:
                if not isinstance(ts, dict):
                    continue
                sym = ts.get("symbol")
                cnt = ts.get("count")
                ts_parts.append(f"`{sym or 'no-symbol'}` ({cnt})")
            if ts_parts:
                lines.append("- top_symbols: " + ", ".join(ts_parts))

        examples = c.get("example_chunks") or []
        if isinstance(examples, list) and examples:
            lines.append("- examples:")
            for ex in examples[:3]:
                if not isinstance(ex, dict):
                    continue
                p = ex.get("path")
                sym = ex.get("symbol")
                sl = ex.get("start_line")
                el = ex.get("end_line")
                ex_id = ex.get("chunk_id")
                lines.append(
                    f"  - `{p}:{sl}-{el}` "
                    f"(chunk_id={ex_id}, symbol={sym or 'no-symbol'})"
                )

        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


__all__: list[str] = [
    "SnapshotItemLike",
    "compute_chunk_systems",
    "build_chunk_systems_views",
    "render_chunk_systems_markdown",
]
