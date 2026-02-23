"""Terminal UI (TUX/TUI) for exploring snapshot chunk-systems artifacts.

This module binds to an existing snapshot `--out-dir` and never accesses the DB.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ChunkSystemsTuiAssets:
    chunk_systems: dict[str, object]
    system_groups: dict[str, object]
    system_adjacency: dict[str, object]


@dataclass(frozen=True)
class ChunkRef:
    chunk_id: int
    path: str
    symbol: str | None
    chunk_type: str
    start_line: int
    end_line: int

    def format_ref(self) -> str:
        sym = self.symbol or "no-symbol"
        return f"{self.chunk_id}  {self.path}:{self.start_line}-{self.end_line}  {sym}"


@dataclass(frozen=True)
class SystemRow:
    cluster_id: int
    label: str
    label_source: str  # "llm" | "heuristic"
    label_confidence: float | None
    size: int
    in_degree: int
    out_degree: int
    deg: int
    test_ratio: float
    area: str
    in_weight_sum: float
    out_weight_sum: float
    top_files: list[tuple[str, int]]
    top_symbols: list[tuple[str | None, int]]
    example_chunks: list[ChunkRef]


@dataclass(frozen=True)
class GroupRow:
    gid: int
    label: str
    system_ids: list[int]
    systems: int
    total_size: int
    avg_deg: float
    test_heavy_systems: int


@dataclass(frozen=True)
class _AdjMetrics:
    in_degree: int
    out_degree: int
    in_weight_sum: float
    out_weight_sum: float

    @property
    def deg(self) -> int:
        return int(self.in_degree) + int(self.out_degree)


@dataclass
class _SystemEvidence:
    file_counts: Counter[str]
    symbol_counts: Counter[str | None]
    chunks: list[ChunkRef] | None


@dataclass(frozen=True)
class _Partition:
    resolution: float
    group_count: int
    membership: list[int]  # index-aligned with `systems_ordered`


_AREA_BUCKETS: list[str] = [
    "chunkhound/api/cli",
    "chunkhound/snapshot",
    "chunkhound/parsers",
    "chunkhound/providers",
    "chunkhound/services/research",
    "chunkhound/services",
    "chunkhound/core",
    "tests",
    "scripts",
    "operations",
    "other",
]


def _read_json_file(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object in {path.name}")
    return raw


def validate_chunk_systems_tui_assets(out_dir: Path) -> list[str]:
    """Validate required artifacts for binding the chunk-systems TUI.

    Returns a list of human-readable error strings (empty means valid).
    """
    errors: list[str] = []

    required = [
        ("snapshot.chunk_systems.json", {"snapshot.chunk_systems.v1"}),
        (
            "snapshot.chunk_systems.system_groups.json",
            {"snapshot.chunk_systems.system_groups.v1"},
        ),
        (
            "snapshot.chunk_systems.system_adjacency.json",
            {
                "snapshot.chunk_systems.system_adjacency.v1",
                "snapshot.chunk_systems.system_adjacency_directed.v1",
            },
        ),
    ]

    if not out_dir.exists():
        return [f"Out-dir does not exist: {out_dir}"]
    if not out_dir.is_dir():
        return [f"Out-dir is not a directory: {out_dir}"]

    for filename, versions in required:
        path = out_dir / filename
        if not path.exists():
            errors.append(f"Missing: {filename}")
            continue
        try:
            payload = _read_json_file(path)
        except Exception as exc:
            errors.append(f"Invalid JSON: {filename} ({exc})")
            continue

        schema = str(payload.get("schema_version") or "")
        if schema not in versions:
            errors.append(
                f"Invalid schema_version in {filename}: {schema!r} (expected one of: {sorted(versions)!r})"
            )

        if filename.endswith("system_groups.json"):
            if not isinstance(payload.get("systems"), list) or not isinstance(
                payload.get("partitions"), list
            ):
                errors.append(
                    f"Invalid structure: {filename} must contain 'systems' and 'partitions' lists"
                )
        if filename.endswith("system_adjacency.json"):
            if not isinstance(payload.get("systems"), list) or not isinstance(
                payload.get("links"), list
            ):
                errors.append(
                    f"Invalid structure: {filename} must contain 'systems' and 'links' lists"
                )

    return errors


def load_chunk_systems_tui_assets(out_dir: Path) -> ChunkSystemsTuiAssets:
    errors = validate_chunk_systems_tui_assets(out_dir)
    if errors:
        joined = "\n".join(f"- {e}" for e in errors)
        raise RuntimeError(f"Snapshot out-dir is missing required TUI assets:\n{joined}")

    chunk_systems = _read_json_file(out_dir / "snapshot.chunk_systems.json")
    system_groups = _read_json_file(out_dir / "snapshot.chunk_systems.system_groups.json")
    system_adjacency = _read_json_file(out_dir / "snapshot.chunk_systems.system_adjacency.json")
    return ChunkSystemsTuiAssets(
        chunk_systems=chunk_systems,
        system_groups=system_groups,
        system_adjacency=system_adjacency,
    )


def _normalize_path(path: str) -> str:
    return str(path or "").replace("\\", "/").lstrip("./")


def _is_test_path(path: str) -> bool:
    p = _normalize_path(path)
    if not p:
        return False
    if "/tests/" in p or p.startswith("tests/") or p.startswith("chunkhound/tests/"):
        return True
    if "/test_" in p or p.endswith("_test.py"):
        return True
    return False


def _area_bucket(path: str) -> str:
    p = _normalize_path(path)
    if not p:
        return "other"

    prefixes: list[tuple[str, str]] = [
        ("chunkhound/api/cli/", "chunkhound/api/cli"),
        ("chunkhound/snapshot/", "chunkhound/snapshot"),
        ("chunkhound/parsers/", "chunkhound/parsers"),
        ("chunkhound/providers/", "chunkhound/providers"),
        ("chunkhound/services/research/", "chunkhound/services/research"),
        ("chunkhound/services/", "chunkhound/services"),
        ("chunkhound/core/", "chunkhound/core"),
        ("tests/", "tests"),
        ("scripts/", "scripts"),
        ("operations/", "operations"),
    ]
    for prefix, bucket in prefixes:
        if p == bucket or p.startswith(prefix):
            return bucket
    return "other"


def _dominant_area(file_counts: Counter[str], *, stable_order: list[str]) -> str:
    if not file_counts:
        return "other"
    by_bucket: dict[str, int] = {}
    for path, cnt in file_counts.items():
        bucket = _area_bucket(path)
        by_bucket[bucket] = int(by_bucket.get(bucket, 0)) + int(cnt)
    ranked = sorted(
        by_bucket.items(),
        key=lambda kv: (-int(kv[1]), stable_order.index(kv[0]) if kv[0] in stable_order else 999, kv[0]),
    )
    return ranked[0][0] if ranked else "other"


def _test_ratio(file_counts: Counter[str]) -> float:
    total = int(sum(int(v) for v in file_counts.values()))
    if total <= 0:
        return 0.0
    test_total = int(sum(int(v) for p, v in file_counts.items() if _is_test_path(p)))
    return float(test_total) / float(total)


def _parse_clusters_by_id(chunk_systems_payload: dict[str, object]) -> dict[int, dict[str, object]]:
    clusters_obj = chunk_systems_payload.get("clusters") or []
    clusters: dict[int, dict[str, object]] = {}
    if not isinstance(clusters_obj, list):
        return clusters
    for c in clusters_obj:
        if not isinstance(c, dict):
            continue
        cid_obj = c.get("cluster_id")
        try:
            cid = int(cid_obj)
        except Exception:
            continue
        clusters[int(cid)] = c
    return clusters


def _parse_top_files(cluster: dict[str, object]) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    tf = cluster.get("top_files") or []
    if not isinstance(tf, list):
        return out
    for row in tf:
        if not isinstance(row, dict):
            continue
        path = row.get("path")
        count = row.get("count")
        if not isinstance(path, str) or not path:
            continue
        try:
            cnt_i = int(count or 0)
        except Exception:
            cnt_i = 0
        if cnt_i <= 0:
            continue
        out.append((_normalize_path(path), int(cnt_i)))
    return out


def _parse_top_symbols(cluster: dict[str, object]) -> list[tuple[str | None, int]]:
    out: list[tuple[str | None, int]] = []
    ts = cluster.get("top_symbols") or []
    if not isinstance(ts, list):
        return out
    for row in ts:
        if not isinstance(row, dict):
            continue
        sym = row.get("symbol")
        count = row.get("count")
        sym_s: str | None
        if sym is None:
            sym_s = None
        elif isinstance(sym, str):
            sym_s = sym
        else:
            sym_s = str(sym)
        try:
            cnt_i = int(count or 0)
        except Exception:
            cnt_i = 0
        if cnt_i <= 0:
            continue
        out.append((sym_s, int(cnt_i)))
    return out


def _parse_example_chunks(cluster: dict[str, object]) -> list[ChunkRef]:
    out: list[ChunkRef] = []
    examples = cluster.get("example_chunks") or []
    if not isinstance(examples, list):
        return out
    for row in examples:
        if not isinstance(row, dict):
            continue
        try:
            chunk_id = int(row.get("chunk_id") or 0)
            start_line = int(row.get("start_line") or 0)
            end_line = int(row.get("end_line") or 0)
        except Exception:
            continue
        path = row.get("path")
        if not isinstance(path, str) or not path:
            continue
        sym = row.get("symbol")
        symbol_s: str | None
        if sym is None:
            symbol_s = None
        elif isinstance(sym, str):
            symbol_s = sym
        else:
            symbol_s = str(sym)
        out.append(
            ChunkRef(
                chunk_id=int(chunk_id),
                path=_normalize_path(path),
                symbol=symbol_s,
                chunk_type="",
                start_line=int(start_line),
                end_line=int(end_line),
            )
        )
    out.sort(key=lambda c: (int(c.chunk_id), str(c.path), int(c.start_line)))
    return out


def build_system_evidence_from_items(
    *,
    items_jsonl_path: Path,
    systems_ordered: list[int],
) -> dict[int, _SystemEvidence]:
    """Build per-system evidence indexes from snapshot.chunk_systems.items.jsonl.

    The JSONL format is one record per chunk (kept systems only), sorted by chunk_id.
    """
    by_system: dict[int, _SystemEvidence] = {
        int(cid): _SystemEvidence(file_counts=Counter(), symbol_counts=Counter(), chunks=[])
        for cid in systems_ordered
    }

    for line in items_jsonl_path.read_text("utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        obj: object
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        try:
            chunk_id = int(obj.get("chunk_id") or 0)
            cluster_id = int(obj.get("cluster_id") or 0)
            start_line = int(obj.get("start_line") or 0)
            end_line = int(obj.get("end_line") or 0)
        except Exception:
            continue
        if int(cluster_id) not in by_system:
            continue
        path = obj.get("path")
        if not isinstance(path, str) or not path:
            continue
        sym = obj.get("symbol")
        symbol_s: str | None
        if sym is None:
            symbol_s = None
        elif isinstance(sym, str):
            symbol_s = sym
        else:
            symbol_s = str(sym)
        chunk_type = str(obj.get("chunk_type") or "")

        ref = ChunkRef(
            chunk_id=int(chunk_id),
            path=_normalize_path(path),
            symbol=symbol_s,
            chunk_type=str(chunk_type),
            start_line=int(start_line),
            end_line=int(end_line),
        )

        ev = by_system[int(cluster_id)]
        ev.file_counts[_normalize_path(path)] += 1
        ev.symbol_counts[symbol_s] += 1
        if ev.chunks is not None:
            ev.chunks.append(ref)

    # Keep chunks sorted by chunk_id for deterministic browsing.
    for ev in by_system.values():
        if ev.chunks is not None:
            ev.chunks.sort(key=lambda c: int(c.chunk_id))

    return by_system


def build_system_evidence_fallback(
    *, clusters_by_id: dict[int, dict[str, object]], systems_ordered: list[int]
) -> dict[int, _SystemEvidence]:
    by_system: dict[int, _SystemEvidence] = {}
    for cid in systems_ordered:
        cluster = clusters_by_id.get(int(cid), {})
        top_files = _parse_top_files(cluster)
        top_symbols = _parse_top_symbols(cluster)

        file_counts = Counter({p: int(c) for p, c in top_files})
        symbol_counts = Counter({s: int(c) for s, c in top_symbols})
        by_system[int(cid)] = _SystemEvidence(
            file_counts=file_counts,
            symbol_counts=symbol_counts,
            chunks=None,
        )
    return by_system


def _parse_system_groups_payload(
    payload: dict[str, object],
) -> tuple[list[int], list[_Partition]]:
    systems_obj = payload.get("systems")
    partitions_obj = payload.get("partitions")
    if not isinstance(systems_obj, list) or not isinstance(partitions_obj, list):
        raise RuntimeError("system groups payload missing systems/partitions")

    systems_ordered: list[int] = []
    for row in systems_obj:
        if not isinstance(row, dict):
            continue
        cid = row.get("cluster_id")
        try:
            cid_i = int(cid)
        except Exception:
            continue
        if cid_i <= 0:
            continue
        systems_ordered.append(int(cid_i))

    partitions: list[_Partition] = []
    for part in partitions_obj:
        if not isinstance(part, dict):
            continue
        try:
            resolution = float(part.get("resolution") or 0.0)
            group_count = int(part.get("group_count") or 0)
        except Exception:
            continue
        membership_obj = part.get("membership")
        if not isinstance(membership_obj, list):
            continue
        membership = [int(x or 0) for x in membership_obj]
        if len(membership) != len(systems_ordered):
            continue
        partitions.append(
            _Partition(
                resolution=float(resolution),
                group_count=int(group_count),
                membership=membership,
            )
        )

    if not partitions:
        raise RuntimeError("system groups payload has no valid partitions")

    return systems_ordered, partitions


def _load_group_labels(out_dir: Path) -> dict[tuple[int, int], str]:
    path = out_dir / "snapshot.chunk_systems.system_group_labels.json"
    if not path.exists():
        return {}
    try:
        payload = _read_json_file(path)
    except Exception:
        return {}
    if str(payload.get("schema_version") or "") != "snapshot.chunk_systems.system_group_labels.v1":
        return {}

    out: dict[tuple[int, int], str] = {}
    parts_obj = payload.get("partitions")
    if not isinstance(parts_obj, list):
        return out

    def res_key(resolution: float) -> int:
        return int(round(float(resolution) * 1000.0))

    for part in parts_obj:
        if not isinstance(part, dict):
            continue
        try:
            res = float(part.get("resolution") or 0.0)
        except Exception:
            continue
        labels_obj = part.get("labels")
        if not isinstance(labels_obj, list):
            continue
        for idx, row in enumerate(labels_obj, start=1):
            if not isinstance(row, dict):
                continue
            label = row.get("label")
            if not isinstance(label, str) or not label.strip():
                continue
            out[(res_key(res), int(idx))] = label.strip()
    return out


def _load_chunk_system_labels(out_dir: Path) -> dict[int, tuple[str, float | None]]:
    path = out_dir / "snapshot.labels.json"
    if not path.exists():
        return {}
    try:
        payload = _read_json_file(path)
    except Exception:
        return {}
    if str(payload.get("schema_version") or "") != "snapshot.labels.v1":
        return {}

    chunk_systems_obj = payload.get("chunk_systems")
    if not isinstance(chunk_systems_obj, dict):
        return {}

    out: dict[int, tuple[str, float | None]] = {}
    for cid_key, row in chunk_systems_obj.items():
        if not isinstance(row, dict):
            continue
        label = row.get("label")
        if not isinstance(label, str) or not label.strip():
            continue
        try:
            cid = int(cid_key)
        except Exception:
            continue
        conf_obj = row.get("confidence")
        conf: float | None
        if conf_obj is None:
            conf = None
        else:
            try:
                conf = float(conf_obj)  # type: ignore[arg-type]
            except Exception:
                conf = None
        out[int(cid)] = (label.strip(), conf)
    return out


def _load_snapshot_run(out_dir: Path) -> dict[str, object] | None:
    path = out_dir / "snapshot.run.json"
    if not path.exists():
        return None
    try:
        payload = _read_json_file(path)
    except Exception:
        return None
    if str(payload.get("schema_version") or "") != "snapshot.run.v1":
        return None
    return payload


def _build_adjacency_metrics(payload: dict[str, object]) -> tuple[dict[int, _AdjMetrics], bool]:
    schema = str(payload.get("schema_version") or "")
    directed = schema == "snapshot.chunk_systems.system_adjacency_directed.v1"

    systems_obj = payload.get("systems") or []
    links_obj = payload.get("links") or []
    systems: list[int] = []
    if isinstance(systems_obj, list):
        for row in systems_obj:
            if not isinstance(row, dict):
                continue
            cid = row.get("cluster_id")
            try:
                systems.append(int(cid))
            except Exception:
                continue

    metrics: dict[int, _AdjMetrics] = {
        int(cid): _AdjMetrics(in_degree=0, out_degree=0, in_weight_sum=0.0, out_weight_sum=0.0)
        for cid in systems
    }

    if not isinstance(links_obj, list):
        return metrics, directed

    if directed:
        for link in links_obj:
            if not isinstance(link, dict):
                continue
            try:
                src = int(link.get("source") or 0)
                dst = int(link.get("target") or 0)
            except Exception:
                continue
            if src <= 0 or dst <= 0:
                continue
            wsum_obj = link.get("weight_sum")
            try:
                wsum = float(wsum_obj or 0.0)
            except Exception:
                wsum = 0.0

            if src not in metrics:
                metrics[src] = _AdjMetrics(in_degree=0, out_degree=0, in_weight_sum=0.0, out_weight_sum=0.0)
            if dst not in metrics:
                metrics[dst] = _AdjMetrics(in_degree=0, out_degree=0, in_weight_sum=0.0, out_weight_sum=0.0)

            m_src = metrics[src]
            metrics[src] = _AdjMetrics(
                in_degree=int(m_src.in_degree),
                out_degree=int(m_src.out_degree) + 1,
                in_weight_sum=float(m_src.in_weight_sum),
                out_weight_sum=float(m_src.out_weight_sum) + float(wsum),
            )
            m_dst = metrics[dst]
            metrics[dst] = _AdjMetrics(
                in_degree=int(m_dst.in_degree) + 1,
                out_degree=int(m_dst.out_degree),
                in_weight_sum=float(m_dst.in_weight_sum) + float(wsum),
                out_weight_sum=float(m_dst.out_weight_sum),
            )
        return metrics, True

    # Undirected: treat each link as a neighbor for both a and b.
    for link in links_obj:
        if not isinstance(link, dict):
            continue
        try:
            a = int(link.get("a") or 0)
            b = int(link.get("b") or 0)
        except Exception:
            continue
        if a <= 0 or b <= 0:
            continue
        wsum_obj = link.get("weight_sum")
        try:
            wsum = float(wsum_obj or 0.0)
        except Exception:
            wsum = 0.0

        for u, v in [(a, b), (b, a)]:
            if u not in metrics:
                metrics[u] = _AdjMetrics(in_degree=0, out_degree=0, in_weight_sum=0.0, out_weight_sum=0.0)
            m = metrics[u]
            metrics[u] = _AdjMetrics(
                in_degree=int(m.in_degree) + 1,
                out_degree=int(m.out_degree) + 1,
                in_weight_sum=float(m.in_weight_sum) + float(wsum),
                out_weight_sum=float(m.out_weight_sum) + float(wsum),
            )

    return metrics, False


def _slice_window(
    *, items_count: int, selected: int, scroll: int, page_size: int
) -> tuple[int, int, int]:
    if items_count <= 0:
        return 0, 0, 0
    sel = max(0, min(int(selected), items_count - 1))
    pg = max(1, int(page_size))
    scr = max(0, min(int(scroll), max(0, items_count - 1)))
    if sel < scr:
        scr = sel
    elif sel >= scr + pg:
        scr = sel - pg + 1
    scr = max(0, min(scr, max(0, items_count - pg)))
    end = min(items_count, scr + pg)
    return sel, scr, end


def run_chunk_systems_tui(out_dir: Path) -> None:
    """Run the interactive chunk-systems explorer (TUX)."""

    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    from chunkhound.api.cli.keyboard import KeyboardInput
    from chunkhound.api.cli.utils.text_input import TextInputState, create_text_input_display

    assets = load_chunk_systems_tui_assets(out_dir)
    clusters_by_id = _parse_clusters_by_id(assets.chunk_systems)

    systems_ordered, partitions = _parse_system_groups_payload(assets.system_groups)
    group_label_by_res_gid = _load_group_labels(out_dir)

    adjacency_metrics_by_id, adjacency_is_directed = _build_adjacency_metrics(
        assets.system_adjacency
    )

    items_path = out_dir / "snapshot.chunk_systems.items.jsonl"
    if items_path.exists():
        evidence_by_id = build_system_evidence_from_items(
            items_jsonl_path=items_path,
            systems_ordered=systems_ordered,
        )
        items_available = True
    else:
        evidence_by_id = build_system_evidence_fallback(
            clusters_by_id=clusters_by_id,
            systems_ordered=systems_ordered,
        )
        items_available = False

    llm_labels_by_system_id = _load_chunk_system_labels(out_dir)
    snapshot_run = _load_snapshot_run(out_dir)

    system_rows: list[SystemRow] = []
    for cid in systems_ordered:
        cluster = clusters_by_id.get(int(cid), {})
        llm_label = llm_labels_by_system_id.get(int(cid))
        if llm_label is not None:
            label, label_confidence = llm_label
            label_source = "llm"
        else:
            label = str(cluster.get("label") or "")
            label_confidence = None
            label_source = "heuristic"
        try:
            size = int(cluster.get("size") or 0)
        except Exception:
            size = 0
        top_files = _parse_top_files(cluster)
        top_symbols = _parse_top_symbols(cluster)
        example_chunks = _parse_example_chunks(cluster)

        ev = evidence_by_id.get(int(cid))
        file_counts = ev.file_counts if ev is not None else Counter()

        test_ratio = _test_ratio(file_counts)
        area = _dominant_area(file_counts, stable_order=_AREA_BUCKETS)

        metrics = adjacency_metrics_by_id.get(int(cid), _AdjMetrics(0, 0, 0.0, 0.0))
        row = SystemRow(
            cluster_id=int(cid),
            label=label.strip(),
            label_source=str(label_source),
            label_confidence=(
                float(label_confidence) if label_confidence is not None else None
            ),
            size=int(size),
            in_degree=int(metrics.in_degree),
            out_degree=int(metrics.out_degree),
            deg=int(metrics.deg),
            test_ratio=float(test_ratio),
            area=str(area),
            in_weight_sum=float(metrics.in_weight_sum),
            out_weight_sum=float(metrics.out_weight_sum),
            top_files=top_files,
            top_symbols=top_symbols,
            example_chunks=example_chunks,
        )
        system_rows.append(row)

    system_by_id: dict[int, SystemRow] = {int(r.cluster_id): r for r in system_rows}

    def resolution_key(resolution: float) -> int:
        return int(round(float(resolution) * 1000.0))

    def group_rows_for_resolution(res: float) -> list[GroupRow]:
        part = next((p for p in partitions if float(p.resolution) == float(res)), None)
        if part is None:
            return []

        members_by_gid: dict[int, list[int]] = {}
        for idx, gid in enumerate(part.membership):
            if gid <= 0:
                continue
            cid = systems_ordered[idx]
            members_by_gid.setdefault(int(gid), []).append(int(cid))

        out: list[GroupRow] = []
        for gid, member_ids in members_by_gid.items():
            total_size = int(sum(int(system_by_id[c].size) for c in member_ids if c in system_by_id))
            degs = [int(system_by_id[c].deg) for c in member_ids if c in system_by_id]
            avg_deg = (sum(degs) / float(len(degs))) if degs else 0.0

            test_heavy = 0
            for c in member_ids:
                sr = system_by_id.get(int(c))
                if sr is None:
                    continue
                if sr.size >= 10 and float(sr.test_ratio) >= 0.90:
                    test_heavy += 1

            label = group_label_by_res_gid.get((resolution_key(res), int(gid)), "")
            out.append(
                GroupRow(
                    gid=int(gid),
                    label=str(label),
                    system_ids=list(member_ids),
                    systems=int(len(member_ids)),
                    total_size=int(total_size),
                    avg_deg=float(avg_deg),
                    test_heavy_systems=int(test_heavy),
                )
            )

        out.sort(key=lambda g: int(g.gid))
        return out

    group_rows_by_resolution: dict[float, list[GroupRow]] = {
        float(p.resolution): group_rows_for_resolution(float(p.resolution)) for p in partitions
    }

    # --- Interactive state ---
    console = Console()
    keyboard = KeyboardInput()

    view: str = "groups"  # groups | systems | evidence
    help_visible = False
    status_panel_visible = False

    res_index = 0
    resolutions = [float(p.resolution) for p in partitions]

    group_selected = 0
    group_scroll = 0

    systems_context_gid: int | None = None  # None => all
    systems_selected = 0
    systems_scroll = 0

    sort_key = "area"
    sort_asc = True

    search_query = ""
    min_size_buckets = [0, 2, 5, 10, 20, 50, 100, 200]
    min_deg_buckets = [0, 1, 2, 3, 5, 10, 20, 50]
    min_size_idx = 0
    min_deg_idx = 0
    test_filter = "all"  # all | test_only | test_heavy | non_test_heavy
    top_files_contains = ""

    evidence_scope_ids: list[int] = []
    evidence_scope_title = ""
    evidence_scope_key: tuple[str, float | None, int | None] = ("all", None, None)
    evidence_tab = "files"  # files | symbols | chunks
    evidence_selected = 0
    evidence_scroll = 0
    evidence_search = ""
    evidence_sort_key = "count_desc"  # per-tab meaning

    prompt_kind: str | None = None  # text: search|path|evidence_search ; select: resolution
    prompt_text_state: TextInputState | None = None
    prompt_select_choices: list[tuple[str, str]] | None = None
    prompt_select_index = 0

    chunk_scope_cache: dict[tuple[str, float | None, int | None], list[ChunkRef]] = {}
    evidence_return_view = "groups"

    HEADER_HEIGHT = 3
    FOOTER_HEIGHT = 3

    def _body_height() -> int:
        screen_h = max(1, int(console.size.height))
        return max(3, int(screen_h) - int(HEADER_HEIGHT) - int(FOOTER_HEIGHT))

    def _page_size_for_table(body_height: int) -> int:
        # Panel border (2) + Table header (1) => remaining rows.
        return max(1, int(body_height) - 3)

    def _screen_layout(*, header: object, body: object, footer: object) -> Layout:
        root = Layout()
        root.split_column(
            Layout(header, size=int(HEADER_HEIGHT)),
            Layout(body, ratio=1),
            Layout(footer, size=int(FOOTER_HEIGHT)),
        )
        return root

    def _fullscreen_layout(renderable: object) -> Layout:
        root = Layout()
        root.update(renderable)
        return root

    def _render_run_status_panel() -> Panel:
        out_dir_name = out_dir.name or str(out_dir)

        scope_root = ""
        scope_git_head_sha = ""
        scope_git_head_shas: dict[str, str] = {}
        scope_hash = ""
        files = 0
        chunks = 0
        provider = ""
        model = ""
        dims = 0
        matryoshka_dims: int | None = None
        llm_provider = ""
        llm_model = ""
        llm_effort = ""
        llm_label = "llm"
        if snapshot_run is not None:
            try:
                scope_hash = str(snapshot_run.get("scope_hash") or "")
            except Exception:
                scope_hash = ""
            try:
                scope_git_head_sha = str(snapshot_run.get("scope_git_head_sha") or "")
            except Exception:
                scope_git_head_sha = ""
            try:
                shas_obj = snapshot_run.get("scope_git_head_shas") or {}
                if isinstance(shas_obj, dict):
                    scope_git_head_shas = {
                        str(k): str(v)
                        for k, v in shas_obj.items()
                        if str(k).strip() and str(v).strip()
                    }
            except Exception:
                scope_git_head_shas = {}
            scope_root_obj = snapshot_run.get("scope_root")
            if isinstance(scope_root_obj, list):
                scope_root = ", ".join(str(x) for x in scope_root_obj if str(x))
            else:
                scope_root = str(scope_root_obj or "")

            counts = snapshot_run.get("counts") or {}
            if isinstance(counts, dict):
                files = int(counts.get("files") or 0)
                chunks = int(counts.get("chunks") or 0)

            emb = snapshot_run.get("embedding") or {}
            if isinstance(emb, dict):
                provider = str(emb.get("provider") or "")
                model = str(emb.get("model") or "")
                dims = int(emb.get("dims") or 0)
                md_obj = emb.get("matryoshka_dims")
                if md_obj is None:
                    matryoshka_dims = None
                else:
                    try:
                        matryoshka_dims = int(md_obj)  # type: ignore[arg-type]
                    except Exception:
                        matryoshka_dims = None

            config_snapshot = snapshot_run.get("config_snapshot") or {}
            if isinstance(config_snapshot, dict):
                llm_used = config_snapshot.get("llm_synthesis_used") or config_snapshot.get(
                    "llm_synthesis"
                )
                llm_cfg = config_snapshot.get("llm_synthesis_configured")
                llm_row = llm_used if isinstance(llm_used, dict) else llm_cfg
                if isinstance(llm_row, dict):
                    llm_provider = str(llm_row.get("provider") or "")
                    llm_model = str(llm_row.get("model") or "")
                    llm_effort = str(llm_row.get("reasoning_effort") or "")
                    if not isinstance(llm_used, dict) and isinstance(llm_cfg, dict):
                        llm_label = "llm(cfg)"

        links_obj = assets.system_adjacency.get("links") or []
        links = len(links_obj) if isinstance(links_obj, list) else 0

        system_labels = len(llm_labels_by_system_id)
        group_labels = bool(group_label_by_res_gid)

        text = Text()
        text.append(out_dir_name, style="bold")
        text.append("\n")
        if scope_root:
            text.append(f"scope: {scope_root}\n")
        if scope_git_head_sha:
            text.append(f"sha:   {scope_git_head_sha}\n")
        elif scope_git_head_shas:
            unique = sorted({str(v) for v in scope_git_head_shas.values() if str(v)})
            if len(unique) == 1:
                text.append(f"sha:   {unique[0]}\n")
            else:
                pairs = sorted(scope_git_head_shas.items(), key=lambda kv: str(kv[0]))
                summary = ", ".join(f"{k}={v[:8]}" for k, v in pairs)
                summary = summary[:120]
                text.append("sha:   (multiple)\n")
                text.append(f"sha_roots: {summary}\n")
        if scope_hash:
            text.append(f"hash:  {scope_hash}\n")
        text.append(f"systems: {len(systems_ordered)}  res: {len(resolutions)} + final\n")
        text.append(f"adj: {'directed' if adjacency_is_directed else 'undirected'}  links: {links}\n")
        text.append(f"evidence: {'full' if items_available else 'fallback'}\n")
        if files or chunks:
            text.append(f"files: {files}  chunks: {chunks}\n")
        if provider or model or dims:
            matryoshka_s = f" matryoshka={matryoshka_dims}" if matryoshka_dims else ""
            text.append(f"emb: {provider}/{model} dims={dims}{matryoshka_s}\n")
        if llm_provider or llm_model:
            effort_s = f" effort={llm_effort}" if llm_effort.strip() else ""
            text.append(f"{llm_label}: {llm_provider}/{llm_model}{effort_s}\n")
        if system_labels:
            text.append(f"labels: systems={system_labels}/{len(systems_ordered)} groups={'yes' if group_labels else 'no'}\n")

        return Panel(text, title="Status", border_style="green", padding=(0, 1))

    def _render_empty_state(*, title: str, lines: list[str]) -> Panel:
        text = Text()
        text.append(title, style="bold yellow")
        for line in lines:
            text.append("\n")
            text.append(line)
        return Panel(text, title="Empty", border_style="yellow", padding=(0, 1))

    def _range_text(*, total: int, scr: int, end: int, sel: int) -> str:
        if total <= 0:
            return "0/0"
        showing = f"{scr + 1}-{end}/{total}"
        pos = f"{sel + 1}/{total}"
        more = []
        if scr > 0:
            more.append("↑")
        if end < total:
            more.append("↓")
        suffix = f" ({''.join(more)} more)" if more else ""
        return f"{pos} · {showing}{suffix}"

    def _scope_total_chunks(system_ids: list[int]) -> int:
        return int(sum(int(system_by_id[c].size) for c in system_ids if c in system_by_id))

    def _scope_file_counts(system_ids: list[int]) -> Counter[str]:
        out_counts: Counter[str] = Counter()
        for cid in system_ids:
            ev = evidence_by_id.get(int(cid))
            if ev is None:
                continue
            out_counts.update(ev.file_counts)
        return out_counts

    def _scope_symbol_counts(system_ids: list[int]) -> Counter[str | None]:
        out_counts: Counter[str | None] = Counter()
        for cid in system_ids:
            ev = evidence_by_id.get(int(cid))
            if ev is None:
                continue
            out_counts.update(ev.symbol_counts)
        return out_counts

    def _scope_example_chunks(system_ids: list[int], *, limit: int) -> list[ChunkRef]:
        seen: set[int] = set()
        chunks: list[ChunkRef] = []
        for cid in system_ids:
            sr = system_by_id.get(int(cid))
            if sr is None:
                continue
            for ref in sr.example_chunks:
                if int(ref.chunk_id) in seen:
                    continue
                seen.add(int(ref.chunk_id))
                chunks.append(ref)
                if len(chunks) >= int(limit):
                    break
            if len(chunks) >= int(limit):
                break
        chunks.sort(key=lambda c: (int(c.chunk_id), str(c.path), int(c.start_line)))
        return chunks[: max(0, int(limit))]

    def _scope_chunks_full(
        *, scope_key: tuple[str, float | None, int | None], system_ids: list[int]
    ) -> list[ChunkRef]:
        cached = chunk_scope_cache.get(scope_key)
        if cached is not None:
            return cached
        if not items_available:
            out = _scope_example_chunks(system_ids, limit=5000)
            chunk_scope_cache[scope_key] = out
            return out

        all_chunks: list[ChunkRef] = []
        for cid in system_ids:
            ev = evidence_by_id.get(int(cid))
            if ev is None or ev.chunks is None:
                continue
            all_chunks.extend(ev.chunks)
        all_chunks.sort(key=lambda c: int(c.chunk_id))
        chunk_scope_cache[scope_key] = all_chunks
        return all_chunks

    def _filtered_sorted_systems(system_ids: list[int]) -> list[SystemRow]:
        rows = [system_by_id[c] for c in system_ids if c in system_by_id]

        q = search_query.strip().lower()
        if q:
            rows = [r for r in rows if q in (r.label or "").lower()]

        min_size = int(min_size_buckets[min_size_idx])
        min_deg = int(min_deg_buckets[min_deg_idx])
        if min_size > 0:
            rows = [r for r in rows if int(r.size) >= min_size]
        if min_deg > 0:
            rows = [r for r in rows if int(r.deg) >= min_deg]

        if top_files_contains.strip():
            needle = top_files_contains.strip().lower()

            def has_path(r: SystemRow) -> bool:
                for path, _cnt in r.top_files:
                    if needle in str(path).lower():
                        return True
                return False

            rows = [r for r in rows if has_path(r)]

        if test_filter != "all":
            def classify(r: SystemRow) -> str:
                if int(r.size) < 10:
                    return "small"
                ratio = float(r.test_ratio)
                if ratio >= 0.95:
                    return "test_only"
                if ratio >= 0.90:
                    return "test_heavy"
                if ratio <= 0.10:
                    return "non_test_heavy"
                return "mixed"

            rows = [r for r in rows if classify(r) == test_filter]

        def key_for(r: SystemRow) -> tuple:
            if sort_key == "label":
                return (str(r.label).lower(), int(r.cluster_id))
            if sort_key == "size":
                return (int(r.size), int(r.cluster_id))
            if sort_key == "in":
                return (int(r.in_degree), int(r.cluster_id))
            if sort_key == "out":
                return (int(r.out_degree), int(r.cluster_id))
            if sort_key == "deg":
                return (int(r.deg), int(r.cluster_id))
            if sort_key == "test%":
                return (float(r.test_ratio), int(r.cluster_id))
            if sort_key == "area":
                return (str(r.area), int(r.cluster_id))
            if sort_key == "in_weight":
                return (float(r.in_weight_sum), int(r.cluster_id))
            if sort_key == "out_weight":
                return (float(r.out_weight_sum), int(r.cluster_id))
            return (int(r.cluster_id),)

        rows = sorted(rows, key=key_for, reverse=not sort_asc)
        return rows

    def _resolution_mode() -> str:
        if not resolutions:
            return "groups"
        return "systems" if int(res_index) >= int(len(resolutions)) else "groups"

    def _current_resolution() -> float | None:
        if not resolutions:
            return None
        if _resolution_mode() == "systems":
            return None
        return float(resolutions[int(res_index)])

    def _apply_resolution_view_transition() -> None:
        nonlocal view, systems_context_gid, systems_selected, systems_scroll
        # After the last resolution, show final systems (no grouping).
        if _resolution_mode() == "systems":
            view = "systems"
            systems_context_gid = None
            systems_selected = 0
            systems_scroll = 0
            return

        # If we were inside a specific group, changing resolution invalidates the group id.
        if view == "systems" and systems_context_gid is not None:
            view = "groups"
            systems_context_gid = None
            systems_selected = 0
            systems_scroll = 0

        # Leaving the final systems view returns to groups.
        if view == "systems" and systems_context_gid is None:
            view = "groups"
            systems_selected = 0
            systems_scroll = 0

    def _cycle_resolution() -> None:
        nonlocal res_index
        if not resolutions:
            return
        if int(res_index) < int(len(resolutions)):
            res_index = int(res_index) + 1
        else:
            res_index = 0
        _apply_resolution_view_transition()

    def _current_group_rows() -> list[GroupRow]:
        res = _current_resolution()
        if res is None:
            return []
        groups = list(group_rows_by_resolution.get(float(res), []))
        q = search_query.strip().lower()
        if q:
            groups = [g for g in groups if q in str(g.label or "").lower()]
        return groups

    def _open_systems_view(*, gid: int | None) -> None:
        nonlocal view, systems_context_gid, systems_selected, systems_scroll
        view = "systems"
        systems_context_gid = gid
        systems_selected = 0
        systems_scroll = 0

    def _open_evidence_view(
        *,
        scope_ids: list[int],
        title: str,
        scope_key: tuple[str, float | None, int | None],
        return_view: str,
    ) -> None:
        nonlocal view, evidence_scope_ids, evidence_scope_title, evidence_scope_key, evidence_return_view
        nonlocal evidence_tab, evidence_selected, evidence_scroll, evidence_search, evidence_sort_key
        evidence_return_view = str(return_view)
        view = "evidence"
        evidence_scope_ids = list(scope_ids)
        evidence_scope_title = str(title)
        evidence_scope_key = tuple(scope_key)
        evidence_tab = "files"
        evidence_selected = 0
        evidence_scroll = 0
        evidence_search = ""
        evidence_sort_key = "count_desc"

    # Prompt question storage for select prompts (kind -> question).
    prompt_kind_question: dict[str, str] = {}

    def _render_select_prompt() -> Panel:
        assert prompt_kind is not None
        choices = prompt_select_choices or []
        question = prompt_kind_question.get(prompt_kind, "")
        text = Text()
        text.append(f"\n{question}\n\n", style="bold")
        for idx, (display, _value) in enumerate(choices):
            if idx == prompt_select_index:
                text.append("▶ ", style="bold cyan")
                text.append(display, style="bold cyan")
            else:
                text.append("  ")
                text.append(display)
            if idx < len(choices) - 1:
                text.append("\n")
        text.append("\n\n(Up/Down to navigate, Enter to confirm, ESC to cancel)", style="dim")
        return Panel(text, title="Select", border_style="cyan")

    def _render_prompt() -> Panel:
        if prompt_text_state is not None:
            question = getattr(prompt_text_state, "question", "Enter value:")  # type: ignore[attr-defined]
            return Panel(
                create_text_input_display(question, prompt_text_state),
                title="Input",
                border_style="cyan",
            )
        return _render_select_prompt()

    def _render_help() -> Layout:
        lines = [
            "Global: q/ESC quit · h/? help · b back",
            "Navigation: ↑/↓ select · PgUp/PgDn page",
            "Groups: Enter open systems · Space evidence",
            "Systems: Enter/Space evidence",
            "Resolution: r cycle (after last → final systems) · R pick",
            "Search/sort: / search · c clear · s sort · a asc/desc",
            "Filters: t testness · >/< min_size · +/- min_deg · p file path contains",
            "Evidence: Space open/close · (inside) Space/b back · 1/2/3 tabs · / search · s sort",
            "Status: v toggle panel",
        ]
        if not adjacency_is_directed:
            lines.append("")
            lines.append("Note: adjacency is undirected; in/out counts mirror degree.")
        if not items_available:
            lines.append("")
            lines.append(
                "Note: full evidence index missing; rerun with --out-dir-mode force to generate items.jsonl."
            )
        panel = Panel("\n".join(lines), title="Help", border_style="cyan", padding=(0, 1))
        return _fullscreen_layout(panel)

    def _render_groups() -> Layout:
        nonlocal group_selected, group_scroll
        groups = _current_group_rows()
        page_size = _page_size_for_table(_body_height())
        sel, scr, end = _slice_window(
            items_count=len(groups), selected=group_selected, scroll=group_scroll, page_size=page_size
        )
        group_selected = int(sel)
        group_scroll = int(scr)

        header = Text()
        header.append("Snapshot TUX — Groups", style="bold")
        res = _current_resolution()
        assert res is not None
        header.append(f"  res={res:g}", style="cyan")
        header.append(" [r/R]", style="dim")
        header.append(f"  groups={len(groups)}", style="dim")
        header.append(f"  {_range_text(total=len(groups), scr=scr, end=end, sel=sel)}", style="dim")
        if search_query.strip():
            header.append(f"  /{search_query.strip()}", style="green")
            header.append(" [/]", style="dim")
        header.append(f"  status={'on' if status_panel_visible else 'off'}", style="dim")
        header.append(" [v]", style="dim")

        if not adjacency_is_directed:
            header.append("  (undirected adjacency)", style="yellow")
        if not items_available:
            header.append("  (no items.jsonl)", style="yellow")

        if not groups:
            body_main: Any = _render_empty_state(
                title="No groups match the current filters.",
                lines=[
                    "Try: press 'c' to clear, '/' to search, or 'r/R' to change resolution.",
                ],
            )
        else:
            table = Table(show_header=True, header_style="bold", box=None, expand=True)
            table.add_column("", width=2, no_wrap=True)
            table.add_column("gid", justify="right", width=4, no_wrap=True)
            table.add_column("label", overflow="ellipsis", no_wrap=True)
            table.add_column("systems", justify="right", width=8, no_wrap=True)
            table.add_column("total_size", justify="right", width=10, no_wrap=True)
            table.add_column("avg_deg", justify="right", width=8, no_wrap=True)
            table.add_column("test_heavy", justify="right", width=10, no_wrap=True)

            for idx, g in enumerate(groups[scr:end], start=scr):
                is_sel = idx == sel
                style = "bold cyan" if is_sel else ""
                table.add_row(
                    "▶" if is_sel else " ",
                    str(g.gid),
                    str(g.label or ""),
                    str(g.systems),
                    str(g.total_size),
                    f"{g.avg_deg:.1f}",
                    str(g.test_heavy_systems),
                    style=style,
                )
            body_main = table

        footer = Text()
        footer.append(
            "↑/↓ select · PgUp/PgDn page · Enter open · Space evidence · r/R resolution · / search · v status · c clear · h help · q quit",
            style="dim",
        )

        header_panel = Panel(header, border_style="cyan", padding=(0, 1))
        footer_panel = Panel(footer, border_style="cyan", padding=(0, 1))
        if isinstance(body_main, Panel):
            main_panel = body_main
        else:
            main_panel = Panel(body_main, border_style="blue", padding=(0, 1))

        if status_panel_visible:
            status_panel = _render_run_status_panel()
            body = Layout()
            body.split_row(
                Layout(main_panel, name="main"),
                Layout(status_panel, name="status", size=48),
            )
        else:
            body = main_panel

        return _screen_layout(header=header_panel, body=body, footer=footer_panel)

    def _render_systems() -> Layout:
        nonlocal systems_selected, systems_scroll
        res = _current_resolution()
        groups = group_rows_by_resolution.get(float(res), []) if res is not None else []
        context_ids: list[int]
        title: str
        if systems_context_gid is None:
            context_ids = list(systems_ordered)
            title = "All systems"
        else:
            if res is None:
                g = None
            else:
                g = next(
                    (gr for gr in groups if int(gr.gid) == int(systems_context_gid)),
                    None,
                )
            context_ids = list(g.system_ids) if g is not None else []
            title = f"Group {systems_context_gid}"

        rows = _filtered_sorted_systems(context_ids)
        page_size = _page_size_for_table(_body_height())
        sel, scr, end = _slice_window(
            items_count=len(rows),
            selected=systems_selected,
            scroll=systems_scroll,
            page_size=page_size,
        )
        systems_selected = int(sel)
        systems_scroll = int(scr)

        header = Text()
        header.append("Snapshot TUX — Systems", style="bold")
        header.append(f"  [{title}]", style="cyan")
        header.append(f"  res={(f'{res:g}' if res is not None else 'final')}", style="dim")
        header.append(" [r/R]", style="dim")
        header.append(f"  sort={sort_key}{'↑' if sort_asc else '↓'}", style="dim")
        header.append(" [s/a]", style="dim")
        header.append(f"  systems={len(rows)}", style="dim")
        header.append(f"  {_range_text(total=len(rows), scr=scr, end=end, sel=sel)}", style="dim")
        if search_query.strip():
            header.append(f"  /{search_query.strip()}", style="green")
            header.append(" [/]", style="dim")
        header.append(f"  status={'on' if status_panel_visible else 'off'}", style="dim")
        header.append(" [v]", style="dim")
        header.append(
            f"  min_size>={min_size_buckets[min_size_idx]} [>/<]"
            f" min_deg>={min_deg_buckets[min_deg_idx]} [+/-]"
            f" t={test_filter} [t]",
            style="dim",
        )
        if top_files_contains.strip():
            header.append(f" path~={top_files_contains.strip()}", style="yellow")
            header.append(" [p]", style="dim")

        if not rows:
            body_main2: Any = _render_empty_state(
                title="No systems match the current filters.",
                lines=[
                    "Try: press 'c' to clear, adjust min_size/min_deg, or toggle testness/path filters.",
                ],
            )
        else:
            table = Table(show_header=True, header_style="bold", box=None, expand=True)
            table.add_column("", width=2, no_wrap=True)
            table.add_column("id", justify="right", width=6, no_wrap=True)
            table.add_column("label", overflow="ellipsis", no_wrap=True)
            table.add_column("size", justify="right", width=6, no_wrap=True)
            table.add_column("in", justify="right", width=5, no_wrap=True)
            table.add_column("out", justify="right", width=5, no_wrap=True)
            table.add_column("deg", justify="right", width=5, no_wrap=True)
            table.add_column("test%", justify="right", width=6, no_wrap=True)
            table.add_column("area", width=18, no_wrap=True)

            for idx, r in enumerate(rows[scr:end], start=scr):
                is_sel = idx == sel
                style = "bold cyan" if is_sel else ""
                table.add_row(
                    "▶" if is_sel else " ",
                    str(r.cluster_id),
                    str(r.label or ""),
                    str(r.size),
                    str(r.in_degree),
                    str(r.out_degree),
                    str(r.deg),
                    f"{(r.test_ratio * 100.0):.0f}",
                    str(r.area),
                    style=style,
                )
            body_main2 = table

        footer = Text()
        footer.append(
            "b back · ↑/↓ select · PgUp/PgDn page · Enter/Space evidence · r/R resolution · / search · v status · c clear · s sort · a order · t testness · p path · q quit",
            style="dim",
        )

        header_panel = Panel(header, border_style="cyan", padding=(0, 1))
        footer_panel = Panel(footer, border_style="cyan", padding=(0, 1))
        if isinstance(body_main2, Panel):
            main_panel = body_main2
        else:
            main_panel = Panel(body_main2, border_style="blue", padding=(0, 1))
        if status_panel_visible:
            status_panel = _render_run_status_panel()
            body = Layout()
            body.split_row(
                Layout(main_panel, name="main"),
                Layout(status_panel, name="status", size=48),
            )
        else:
            body = main_panel
        return _screen_layout(header=header_panel, body=body, footer=footer_panel)

    def _evidence_items() -> tuple[list[tuple[str, int]] | list[ChunkRef], str]:
        ids = list(evidence_scope_ids)
        if evidence_tab == "files":
            counts = _scope_file_counts(ids)
            rows = [(p, int(c)) for p, c in counts.items()]
            q = evidence_search.strip().lower()
            if q:
                rows = [r for r in rows if q in str(r[0]).lower()]
            if evidence_sort_key == "path_asc":
                rows.sort(key=lambda r: (str(r[0]), -int(r[1])))
            else:
                rows.sort(key=lambda r: (-int(r[1]), str(r[0])))
            return rows, "Files"

        if evidence_tab == "symbols":
            counts = _scope_symbol_counts(ids)
            rows2 = [(s or "no-symbol", int(c)) for s, c in counts.items()]
            q = evidence_search.strip().lower()
            if q:
                rows2 = [r for r in rows2 if q in str(r[0]).lower()]
            if evidence_sort_key == "symbol_asc":
                rows2.sort(key=lambda r: (str(r[0]), -int(r[1])))
            else:
                rows2.sort(key=lambda r: (-int(r[1]), str(r[0])))
            return rows2, "Symbols"

        # chunks
        chunks = _scope_chunks_full(scope_key=evidence_scope_key, system_ids=ids)
        q = evidence_search.strip().lower()
        if q:
            chunks = [c for c in chunks if q in c.path.lower() or q in (c.symbol or "").lower()]
        if evidence_sort_key == "path_asc":
            chunks = sorted(chunks, key=lambda c: (str(c.path), int(c.start_line), int(c.chunk_id)))
        else:
            chunks = sorted(chunks, key=lambda c: int(c.chunk_id))
        return chunks, "Chunks"

    def _render_evidence() -> Layout:
        nonlocal evidence_selected, evidence_scroll
        items, tab_title = _evidence_items()
        page_size = _page_size_for_table(_body_height())
        sel, scr, end = _slice_window(
            items_count=len(items),
            selected=evidence_selected,
            scroll=evidence_scroll,
            page_size=page_size,
        )
        evidence_selected = int(sel)
        evidence_scroll = int(scr)

        header = Text()
        header.append("Snapshot TUX — Evidence", style="bold")
        header.append(f"  [{evidence_scope_title}]", style="cyan")
        header.append(f"  tab={tab_title}", style="dim")
        header.append(" [1/2/3]", style="dim")
        header.append(f"  sort={evidence_sort_key}", style="dim")
        header.append(" [s]", style="dim")
        header.append(f"  items={len(items)}", style="dim")
        header.append(f"  {_range_text(total=len(items), scr=scr, end=end, sel=sel)}", style="dim")
        if evidence_search.strip():
            header.append(f"  /{evidence_search.strip()}", style="green")
            header.append(" [/]", style="dim")
        if not items_available:
            header.append("  (no items.jsonl)", style="yellow")

        if not items:
            body_ev: Any = _render_empty_state(
                title="No evidence items for this scope.",
                lines=["Try: press 'b' to go back, or clear searches/filters."],
            )
        else:
            table = Table(show_header=True, header_style="bold", box=None, expand=True)
            table.add_column("", width=2, no_wrap=True)
            if evidence_tab == "chunks":
                table.add_column("chunk_id", justify="right", width=8, no_wrap=True)
                table.add_column("path", overflow="ellipsis", no_wrap=True)
                table.add_column("range", width=12, no_wrap=True)
                table.add_column("symbol", overflow="ellipsis", no_wrap=True)
                for idx, ref in enumerate(items[scr:end], start=scr):  # type: ignore[index]
                    assert isinstance(ref, ChunkRef)
                    is_sel = idx == sel
                    style = "bold cyan" if is_sel else ""
                    table.add_row(
                        "▶" if is_sel else " ",
                        str(ref.chunk_id),
                        str(ref.path),
                        f"{ref.start_line}-{ref.end_line}",
                        str(ref.symbol or "no-symbol"),
                        style=style,
                    )
            else:
                table.add_column("item", overflow="ellipsis", no_wrap=True)
                table.add_column("count", justify="right", width=8, no_wrap=True)
                for idx, row in enumerate(items[scr:end], start=scr):  # type: ignore[index]
                    is_sel = idx == sel
                    style = "bold cyan" if is_sel else ""
                    item_s, cnt = row  # type: ignore[misc]
                    table.add_row(
                        "▶" if is_sel else " ",
                        str(item_s),
                        str(int(cnt)),
                        style=style,
                    )
            body_ev = table

        footer = Text()
        footer.append(
            "Space/b back · ↑/↓ select · PgUp/PgDn page · 1/2/3 tabs · / search · s sort · q quit",
            style="dim",
        )

        header_panel = Panel(header, border_style="cyan", padding=(0, 1))
        footer_panel = Panel(footer, border_style="cyan", padding=(0, 1))
        if isinstance(body_ev, Panel):
            main_panel = body_ev
        else:
            main_panel = Panel(body_ev, border_style="blue", padding=(0, 1))
        return _screen_layout(header=header_panel, body=main_panel, footer=footer_panel)

    def _render() -> Any:
        if prompt_kind is not None:
            return _fullscreen_layout(_render_prompt())
        if help_visible:
            return _render_help()
        if view == "systems":
            return _render_systems()
        if view == "evidence":
            return _render_evidence()
        return _render_groups()

    def _cycle_sort_key() -> None:
        nonlocal sort_key
        keys = ["label", "size", "in", "out", "deg", "test%", "area", "in_weight", "out_weight", "id"]
        if sort_key not in keys:
            sort_key = "deg"
            return
        idx = keys.index(sort_key)
        sort_key = keys[(idx + 1) % len(keys)]

    def _cycle_test_filter() -> None:
        nonlocal test_filter
        cycle = ["all", "test_only", "test_heavy", "non_test_heavy"]
        test_filter = cycle[(cycle.index(test_filter) + 1) % len(cycle)] if test_filter in cycle else "all"

    def _handle_prompt_key(key: str) -> bool:
        nonlocal prompt_kind, prompt_text_state, prompt_select_choices, prompt_select_index
        nonlocal search_query, top_files_contains, evidence_search, res_index
        if key in {"ESC", "CTRL_C"}:
            prompt_kind = None
            prompt_text_state = None
            prompt_select_choices = None
            return True

        if prompt_text_state is not None:
            state = prompt_text_state
            if key in {"LEFT", "RIGHT", "HOME", "END"}:
                state.move_cursor(key)
                return True
            if key in {"BACKSPACE", "DELETE"}:
                state.delete_char(key)
                return True
            if key == "ENTER":
                value = state.text
                if prompt_kind == "search":
                    search_query = value
                elif prompt_kind == "path":
                    top_files_contains = value
                elif prompt_kind == "evidence_search":
                    evidence_search = value
                prompt_kind = None
                prompt_text_state = None
                return True
            if len(key) == 1 and key.isprintable():
                state.insert_char(key)
                return True
            return False

        choices = prompt_select_choices or []
        if key == "UP":
            prompt_select_index = max(0, prompt_select_index - 1)
            return True
        if key == "DOWN":
            prompt_select_index = min(len(choices) - 1, prompt_select_index + 1)
            return True
        if key.isdigit():
            digit = int(key)
            if 1 <= digit <= len(choices):
                prompt_select_index = digit - 1
                return True
        if key == "ENTER":
            if choices:
                _display, value = choices[prompt_select_index]
                if prompt_kind == "resolution":
                    if str(value) == "__systems__":
                        res_index = int(len(resolutions))
                    else:
                        try:
                            picked = float(value)
                            if picked in resolutions:
                                res_index = resolutions.index(picked)
                        except Exception:
                            pass
                    _apply_resolution_view_transition()
                prompt_kind = None
                prompt_select_choices = None
                return True
        return False

    try:
        with Live(_render(), auto_refresh=False, console=console, screen=True) as live:
            while True:
                key = keyboard.getkey()

                if prompt_kind is not None:
                    changed = _handle_prompt_key(key)
                    if changed:
                        live.update(_render(), refresh=True)
                    continue

                if key in {"q", "Q", "ESC"}:
                    break

                if key in {"h", "H", "?"}:
                    help_visible = not help_visible
                    live.update(_render(), refresh=True)
                    continue

                if help_visible:
                    continue

                if view == "evidence":
                    if key in {" ", "b"}:
                        view = evidence_return_view
                    elif key == "1":
                        evidence_tab = "files"
                        evidence_selected = 0
                        evidence_scroll = 0
                    elif key == "2":
                        evidence_tab = "symbols"
                        evidence_selected = 0
                        evidence_scroll = 0
                    elif key == "3":
                        evidence_tab = "chunks"
                        evidence_selected = 0
                        evidence_scroll = 0
                    elif key == "/":
                        prompt_kind = "evidence_search"
                        prompt_text_state = TextInputState(evidence_search)
                        prompt_text_state.question = "Search evidence:"  # type: ignore[attr-defined]
                    elif key == "s":
                        if evidence_tab == "files":
                            evidence_sort_key = "path_asc" if evidence_sort_key != "path_asc" else "count_desc"
                        elif evidence_tab == "symbols":
                            evidence_sort_key = "symbol_asc" if evidence_sort_key != "symbol_asc" else "count_desc"
                        else:
                            evidence_sort_key = "path_asc" if evidence_sort_key != "path_asc" else "chunk_id_asc"
                    elif key == "UP":
                        evidence_selected = max(0, evidence_selected - 1)
                    elif key == "DOWN":
                        evidence_selected = evidence_selected + 1
                    elif key == "PAGE_UP":
                        evidence_selected = max(0, evidence_selected - 10)
                    elif key == "PAGE_DOWN":
                        evidence_selected = evidence_selected + 10
                    live.update(_render(), refresh=True)
                    continue

                if view == "groups":
                    groups = _current_group_rows()
                    if key == "UP":
                        group_selected = max(0, group_selected - 1)
                    elif key == "DOWN":
                        group_selected = min(max(0, len(groups) - 1), group_selected + 1)
                    elif key == "PAGE_UP":
                        group_selected = max(0, group_selected - 10)
                    elif key == "PAGE_DOWN":
                        group_selected = group_selected + 10
                    elif key == "ENTER":
                        if groups:
                            g = groups[group_selected]
                            _open_systems_view(gid=int(g.gid))
                    elif key in {" ", "e"}:
                        if groups:
                            g = groups[group_selected]
                            scope_title = g.label or f"Group {g.gid}"
                            res = _current_resolution()
                            assert res is not None
                            scope_key = ("group", float(res), int(g.gid))
                            _open_evidence_view(
                                scope_ids=g.system_ids,
                                title=scope_title,
                                scope_key=scope_key,
                                return_view="groups",
                            )
                    elif key in {"o", "O"}:
                        if groups:
                            g = groups[group_selected]
                            _open_systems_view(gid=int(g.gid))
                    elif key in {"v", "V"}:
                        status_panel_visible = not status_panel_visible
                    elif key == "r":
                        _cycle_resolution()
                    elif key == "R":
                        prompt_kind = "resolution"
                        prompt_select_choices = [(f"{r:g}", f"{r:g}") for r in resolutions] + [
                            ("Final systems (no grouping)", "__systems__"),
                        ]
                        prompt_select_index = (
                            int(res_index)
                            if int(res_index) < int(len(resolutions))
                            else int(len(resolutions))
                        )
                        prompt_kind_question["resolution"] = "Pick resolution:"
                    elif key == "/":
                        prompt_kind = "search"
                        prompt_text_state = TextInputState(search_query)
                        prompt_text_state.question = "Search (label substring):"  # type: ignore[attr-defined]
                    elif key == "c":
                        search_query = ""
                        min_size_idx = 0
                        min_deg_idx = 0
                        test_filter = "all"
                        top_files_contains = ""
                    live.update(_render(), refresh=True)
                    continue

                # systems view
                if key == "b":
                    view = "groups"
                    systems_context_gid = None
                    if _resolution_mode() == "systems" and resolutions:
                        res_index = int(len(resolutions)) - 1
                elif key == "UP":
                    systems_selected = max(0, systems_selected - 1)
                elif key == "DOWN":
                    systems_selected = systems_selected + 1
                elif key == "PAGE_UP":
                    systems_selected = max(0, systems_selected - 10)
                elif key == "PAGE_DOWN":
                    systems_selected = systems_selected + 10
                elif key == "r":
                    _cycle_resolution()
                elif key == "R":
                    prompt_kind = "resolution"
                    prompt_select_choices = [(f"{r:g}", f"{r:g}") for r in resolutions] + [
                        ("Final systems (no grouping)", "__systems__"),
                    ]
                    prompt_select_index = (
                        int(res_index)
                        if int(res_index) < int(len(resolutions))
                        else int(len(resolutions))
                    )
                    prompt_kind_question["resolution"] = "Pick resolution:"
                elif key in {" ", "ENTER", "e"}:
                    # Evidence scope: current system selection.
                    if systems_context_gid is None:
                        context_ids = list(systems_ordered)
                    else:
                        res = _current_resolution()
                        groups = (
                            group_rows_by_resolution.get(float(res), [])
                            if res is not None
                            else []
                        )
                        g = next((gr for gr in groups if int(gr.gid) == int(systems_context_gid)), None)
                        context_ids = list(g.system_ids) if g is not None else []
                    rows = _filtered_sorted_systems(context_ids)
                    if rows:
                        sr = rows[min(max(0, systems_selected), len(rows) - 1)]
                        _open_evidence_view(
                            scope_ids=[sr.cluster_id],
                            title=f"System {sr.cluster_id}",
                            scope_key=("system", None, int(sr.cluster_id)),
                            return_view="systems",
                        )
                elif key == "/":
                    prompt_kind = "search"
                    prompt_text_state = TextInputState(search_query)
                    prompt_text_state.question = "Search (label substring):"  # type: ignore[attr-defined]
                elif key == "c":
                    search_query = ""
                    min_size_idx = 0
                    min_deg_idx = 0
                    test_filter = "all"
                    top_files_contains = ""
                    sort_key = "area"
                    sort_asc = True
                elif key == "s":
                    _cycle_sort_key()
                elif key == "a":
                    sort_asc = not sort_asc
                elif key == "t":
                    _cycle_test_filter()
                elif key == ">":
                    min_size_idx = min(len(min_size_buckets) - 1, min_size_idx + 1)
                elif key == "<":
                    min_size_idx = max(0, min_size_idx - 1)
                elif key == "+":
                    min_deg_idx = min(len(min_deg_buckets) - 1, min_deg_idx + 1)
                elif key == "-":
                    min_deg_idx = max(0, min_deg_idx - 1)
                elif key == "p":
                    prompt_kind = "path"
                    prompt_text_state = TextInputState(top_files_contains)
                    prompt_text_state.question = "Filter: top file path contains (empty clears):"  # type: ignore[attr-defined]
                elif key in {"v", "V"}:
                    status_panel_visible = not status_panel_visible

                live.update(_render(), refresh=True)

    finally:
        try:
            keyboard.cleanup()
        except Exception:
            pass


__all__ = [
    "ChunkRef",
    "ChunkSystemsTuiAssets",
    "GroupRow",
    "SystemRow",
    "build_system_evidence_fallback",
    "build_system_evidence_from_items",
    "load_chunk_systems_tui_assets",
    "run_chunk_systems_tui",
    "validate_chunk_systems_tui_assets",
]
