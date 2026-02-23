"""Partition helpers for snapshot chunk-system clustering."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SystemNode:
    node_key: str
    change_indexes: list[int]
    vector: list[float]


@dataclass(frozen=True)
class SystemEdge:
    node_a: str
    node_b: str
    w_sem: float | None
    w_struct: float | None
    w_total: float


@dataclass(frozen=True)
class SystemComponent:
    system_id: int
    node_keys: list[str]
    change_indexes: list[int]


def partition_connected_components(
    *, nodes: list[SystemNode], edges: list[SystemEdge]
) -> tuple[list[SystemComponent], dict[str, int]]:
    adjacency: dict[str, set[str]] = {str(node.node_key): set() for node in nodes}
    for edge in edges:
        a = str(edge.node_a)
        b = str(edge.node_b)
        if a == b:
            continue
        if a not in adjacency or b not in adjacency:
            continue
        adjacency[a].add(b)
        adjacency[b].add(a)

    node_by_key = {str(node.node_key): node for node in nodes}
    visited: set[str] = set()
    groups: list[list[str]] = []

    for node in sorted(node_by_key.keys()):
        if node in visited:
            continue
        stack = [node]
        visited.add(node)
        group: list[str] = []
        while stack:
            cur = stack.pop()
            group.append(cur)
            for neighbor in sorted(adjacency.get(cur, ())):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                stack.append(neighbor)
        group.sort()
        groups.append(group)

    groups.sort(key=lambda keys: (-len(keys), keys[0] if keys else ""))
    components: list[SystemComponent] = []
    node_to_system: dict[str, int] = {}
    for idx, keys in enumerate(groups, start=1):
        change_indexes: list[int] = []
        for key in keys:
            node_to_system[key] = idx
            change_indexes.extend(node_by_key[key].change_indexes)
        change_indexes = sorted({int(x) for x in change_indexes})
        components.append(
            SystemComponent(system_id=idx, node_keys=list(keys), change_indexes=change_indexes)
        )

    return components, node_to_system


def partition_leiden(
    *, nodes: list[SystemNode], edges: list[SystemEdge], resolution: float, seed: int
) -> tuple[list[SystemComponent], dict[str, int]]:
    if not nodes:
        return [], {}

    try:
        import igraph as ig  # type: ignore[import-not-found]
        import leidenalg  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "Leiden partitioner requires python-igraph and leidenalg"
        ) from exc

    ordered_nodes = sorted(nodes, key=lambda n: str(n.node_key))
    index_by_key = {str(node.node_key): idx for idx, node in enumerate(ordered_nodes)}

    graph = ig.Graph(n=len(ordered_nodes), directed=False)
    graph.vs["name"] = [str(node.node_key) for node in ordered_nodes]

    graph_edges: list[tuple[int, int]] = []
    edge_weights: list[float] = []
    for edge in edges:
        a = str(edge.node_a)
        b = str(edge.node_b)
        if a == b:
            continue
        if a not in index_by_key or b not in index_by_key:
            continue
        graph_edges.append((index_by_key[a], index_by_key[b]))
        edge_weights.append(max(0.0, float(edge.w_total)))

    if graph_edges:
        graph.add_edges(graph_edges)
    else:
        singletons = [
            SystemComponent(
                system_id=i,
                node_keys=[str(node.node_key)],
                change_indexes=sorted({int(x) for x in node.change_indexes}),
            )
            for i, node in enumerate(ordered_nodes, start=1)
        ]
        node_to_system = {comp.node_keys[0]: comp.system_id for comp in singletons}
        return singletons, node_to_system

    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=edge_weights,
        resolution_parameter=float(resolution),
        seed=int(seed),
    )

    grouped: dict[int, list[str]] = {}
    for vertex_idx, group_idx in enumerate(partition.membership):
        node_key = str(ordered_nodes[vertex_idx].node_key)
        grouped.setdefault(int(group_idx), []).append(node_key)

    groups = [sorted(keys) for keys in grouped.values()]
    groups.sort(key=lambda keys: (-len(keys), keys[0] if keys else ""))

    node_lookup = {str(node.node_key): node for node in ordered_nodes}
    components: list[SystemComponent] = []
    node_to_system: dict[str, int] = {}
    for idx, keys in enumerate(groups, start=1):
        change_indexes: list[int] = []
        for key in keys:
            node_to_system[key] = idx
            change_indexes.extend(node_lookup[key].change_indexes)
        components.append(
            SystemComponent(
                system_id=idx,
                node_keys=keys,
                change_indexes=sorted({int(x) for x in change_indexes}),
            )
        )

    return components, node_to_system


def partition_leiden_with_membership(
    *, nodes: list[SystemNode], edges: list[SystemEdge], resolution: float, seed: int
) -> tuple[list[SystemComponent], dict[str, int], list[str], list[int], float]:
    """Leiden partitioning plus membership and objective quality.

    This is intended for experimentation/benchmarking (e.g., stability across
    seeds) without changing the default snapshot chunk-systems pipeline.

    Returns:
        (components, node_to_system, ordered_node_keys, membership, objective_quality)
    """
    if not nodes:
        return [], {}, [], [], 0.0

    try:
        import igraph as ig  # type: ignore[import-not-found]
        import leidenalg  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "Leiden partitioner requires python-igraph and leidenalg"
        ) from exc

    ordered_nodes = sorted(nodes, key=lambda n: str(n.node_key))
    ordered_keys = [str(node.node_key) for node in ordered_nodes]
    index_by_key = {str(node.node_key): idx for idx, node in enumerate(ordered_nodes)}

    graph = ig.Graph(n=len(ordered_nodes), directed=False)
    graph.vs["name"] = list(ordered_keys)

    graph_edges: list[tuple[int, int]] = []
    edge_weights: list[float] = []
    for edge in edges:
        a = str(edge.node_a)
        b = str(edge.node_b)
        if a == b:
            continue
        if a not in index_by_key or b not in index_by_key:
            continue
        graph_edges.append((index_by_key[a], index_by_key[b]))
        edge_weights.append(max(0.0, float(edge.w_total)))

    if graph_edges:
        graph.add_edges(graph_edges)
        partition = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            weights=edge_weights,
            resolution_parameter=float(resolution),
            seed=int(seed),
        )
        membership = [int(x) for x in partition.membership]
        objective_quality = float(partition.quality())
    else:
        membership = list(range(len(ordered_nodes)))
        objective_quality = 0.0

    grouped: dict[int, list[str]] = {}
    for vertex_idx, group_idx in enumerate(membership):
        node_key = str(ordered_nodes[vertex_idx].node_key)
        grouped.setdefault(int(group_idx), []).append(node_key)

    groups = [sorted(keys) for keys in grouped.values()]
    groups.sort(key=lambda keys: (-len(keys), keys[0] if keys else ""))

    node_lookup = {str(node.node_key): node for node in ordered_nodes}
    components: list[SystemComponent] = []
    node_to_system: dict[str, int] = {}
    for idx, keys in enumerate(groups, start=1):
        change_indexes: list[int] = []
        for key in keys:
            node_to_system[key] = idx
            change_indexes.extend(node_lookup[key].change_indexes)
        components.append(
            SystemComponent(
                system_id=idx,
                node_keys=keys,
                change_indexes=sorted({int(x) for x in change_indexes}),
            )
        )

    return components, node_to_system, ordered_keys, membership, objective_quality
