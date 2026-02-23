"""Snapshot command focused on chunk-systems artifacts."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import xxhash
from loguru import logger

from chunkhound.core.config.config import Config
from chunkhound.core.types.common import ChunkType
from chunkhound.llm_manager import LLMManager
from chunkhound.providers.database.like_utils import escape_like_pattern
from chunkhound.snapshot.chunk_systems import (
    ChunkSystemsGraphContext,
    build_chunk_systems_views,
    compute_chunk_systems,
    render_chunk_systems_markdown,
)
from chunkhound.snapshot.chunk_systems_outputs import (
    build_system_adjacency_json,
    build_system_adjacency_json_directed,
    build_system_groups_json_from_chunk_edges,
    build_system_groups_json_from_directed_arcs,
    iter_graph_edges_jsonl,
    iter_graph_nodes_jsonl,
)
from chunkhound.snapshot.chunk_systems_tui import (
    run_chunk_systems_tui,
    validate_chunk_systems_tui_assets,
)
from chunkhound.snapshot.chunk_systems_viz import render_chunk_systems_viz_html
from chunkhound.utils.git_safe import run_git

from ..utils.rich_output import RichOutputFormatter

_RUN_SCHEMA_VERSION = "snapshot.run.v1"
_LABELS_SCHEMA_VERSION = "snapshot.labels.v1"
_SYSTEM_GROUP_LABELS_SCHEMA_VERSION = "snapshot.chunk_systems.system_group_labels.v1"
_SCHEMA_REVISION = "2026-02-17"

_GIT_HEAD_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")

MAX_CHUNK_CONTENT_TOKENS_PER_CLUSTER = 5000
LOW_CONF_THRESHOLD = 0.75
TEST_HEAVY_EVIDENCE_RATIO = 0.90
LABEL_MAX_CHARS = 80
_LABEL_NO_NEWLINES_PATTERN = r"^[^\r\n]*$"

_CHUNK_SYSTEM_LABEL_SYSTEM_PROMPT = (
    "You label clusters for an operator-facing snapshot report. "
    "A label describes behavior/responsibility (what it does), not an "
    "artifact/stage (what exists). "
    "Prefer a short, natural noun phrase naming the responsibility (2–7 words). "
    "Prefer a concrete component noun when clear "
    "(system, engine, parser, registry, resolver, writer, harness). "
    "'X system' / 'X subsystem' is allowed but not required "
    "(use only if X is specific). "
    "Avoid file-path-ish labels, pure method-name labels, and artifact/stage "
    "labels like '... content' or 'seed data'. "
    "Avoid vague/filler '-ing' labels "
    "(handling/processing/dispatching/writing/formatting/limiting/searching) "
    "when a noun form is clearer. "
    "If the cluster is overwhelmingly tests/fixtures, the label must start with "
    "'Tests:'."
)


@dataclass(frozen=True)
class _SnapshotSelector:
    provider: str
    model: str
    dims: int


@dataclass(frozen=True)
class _SnapshotItem:
    chunk_id: int
    path: str
    symbol: str | None
    chunk_type: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class _ChunkNameHints:
    symbol: str
    signature_line: str | None
    identifier_tokens: list[str]


@dataclass(frozen=True)
class _AvailableSelector:
    provider: str
    model: str
    dims: int
    count: int


@dataclass(frozen=True)
class _LLMChunkSystemLabel:
    cluster_id: int
    label: str
    confidence: float
    prompt_path: str | None


@dataclass(frozen=True)
class _LLMChunkSystemGroupLabel:
    resolution: float
    group_id: int
    label: str
    confidence: float | None
    prompt_path: str | None
    kind: str  # "llm" | "inherited"


@dataclass(frozen=True)
class _BatchLabelRequest:
    item_id: str
    content: str


def _xxh3_64_hexdigest(value: str) -> str:
    return str(xxhash.xxh3_64_hexdigest(value.encode("utf-8")))


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", "utf-8")


def _write_markdown(path: Path, text: str) -> None:
    path.write_text(str(text), "utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _chunk_systems_items_jsonl_rows(
    *,
    report_payload: dict[str, object],
    item_by_chunk_id: dict[int, _SnapshotItem],
) -> list[dict[str, object]]:
    clusters_obj = report_payload.get("clusters") or []
    if not isinstance(clusters_obj, list):
        return []

    by_chunk_id: dict[int, dict[str, object]] = {}
    for cluster in clusters_obj:
        if not isinstance(cluster, dict):
            continue
        cluster_id_obj = cluster.get("cluster_id")
        try:
            cluster_id = int(cluster_id_obj)
        except Exception:
            continue

        chunk_ids_obj = cluster.get("chunk_ids") or []
        if not isinstance(chunk_ids_obj, list):
            continue

        for raw_chunk_id in chunk_ids_obj:
            try:
                chunk_id = int(raw_chunk_id)
            except Exception:
                continue
            item = item_by_chunk_id.get(int(chunk_id))
            if item is None:
                continue
            by_chunk_id[int(chunk_id)] = {
                "chunk_id": int(chunk_id),
                "cluster_id": int(cluster_id),
                "path": str(item.path),
                "symbol": item.symbol if item.symbol is not None else None,
                "chunk_type": str(item.chunk_type),
                "start_line": int(item.start_line),
                "end_line": int(item.end_line),
            }

    return [by_chunk_id[cid] for cid in sorted(by_chunk_id)]


def _env_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def _out_dir_is_non_empty(out_dir: Path) -> bool:
    try:
        return any(out_dir.iterdir())
    except Exception:
        return False


def _tui_is_effective(*, args: argparse.Namespace, formatter: RichOutputFormatter) -> bool:
    if not bool(getattr(args, "tui", True)):
        return False
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False
    if _env_truthy(os.environ.get("CHUNKHOUND_NO_TUI")):
        return False
    return bool(getattr(formatter, "_terminal_compatible", False)) and formatter.console is not None


def _rich_prompt_supported(*, formatter: RichOutputFormatter) -> bool:
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False
    return bool(getattr(formatter, "_terminal_compatible", False)) and formatter.console is not None


def _rich_select_out_dir_mode(
    *,
    question: str,
    choices: list[tuple[str, str]],
    default_index: int,
    console: Any,
) -> str:
    from rich.live import Live
    from rich.text import Text

    from ..keyboard import KeyboardInput

    selected = max(0, min(int(default_index), len(choices) - 1))
    keyboard_handler = KeyboardInput()

    console.print(f"\n[bold]{question}[/bold]\n")

    def create_display() -> Text:
        text = Text()
        for idx, (display, _) in enumerate(choices):
            if idx == selected:
                text.append("▶ ", style="bold cyan")
                text.append(display, style="bold cyan")
            else:
                text.append("  ")
                if display.lower().startswith("abort"):
                    text.append(display, style="dim")
                else:
                    text.append(display)
            if idx < len(choices) - 1:
                text.append("\n")
        return text

    with Live(create_display(), auto_refresh=False, console=console) as live:
        while True:
            try:
                key = keyboard_handler.getkey()
                if key == "UP":
                    selected = max(0, selected - 1)
                    live.update(create_display(), refresh=True)
                elif key == "DOWN":
                    selected = min(len(choices) - 1, selected + 1)
                    live.update(create_display(), refresh=True)
                elif key == "ENTER":
                    live.stop()
                    console.print()
                    return choices[selected][1]
                elif key in {"ESC", "CTRL_C"}:
                    live.stop()
                    return "abort"
                elif key.isdigit():
                    digit = int(key)
                    if 1 <= digit <= len(choices):
                        selected = digit - 1
                        live.stop()
                        console.print()
                        return choices[selected][1]
            except KeyboardInterrupt:
                live.stop()
                return "abort"


def _normalize_scope_root(user_path: Path, *, base_directory: Path) -> str:
    abs_path = (base_directory / user_path).resolve()
    try:
        rel = abs_path.relative_to(base_directory)
    except ValueError:
        raise ValueError(f"Scope root {user_path!s} must be inside {base_directory}")
    scope = str(rel).replace("\\", "/").strip("/")
    return scope or "."


def _build_like_prefix(scope_root: str) -> str | None:
    scope = str(scope_root or ".").strip().replace("\\", "/").strip("/")
    if not scope or scope == ".":
        return None
    escaped = escape_like_pattern(scope)
    return f"{escaped}/%"


def _collapse_scope_roots(scope_roots: list[str]) -> list[str]:
    cleaned: list[str] = []
    for root in scope_roots:
        normalized = str(root or ".").replace("\\", "/").strip("/") or "."
        if normalized not in cleaned:
            cleaned.append(normalized)

    cleaned.sort()
    collapsed: list[str] = []
    for root in cleaned:
        if root == ".":
            return ["."]
        if any(root == parent or root.startswith(parent + "/") for parent in collapsed):
            continue
        collapsed.append(root)
    return collapsed


def _common_scope_root(scope_roots: list[str]) -> str:
    if not scope_roots:
        return "."
    if any(r == "." for r in scope_roots):
        return "."
    parts = [r.split("/") for r in scope_roots]
    prefix: list[str] = []
    for idx in range(min(len(p) for p in parts)):
        token = parts[0][idx]
        if all(p[idx] == token for p in parts[1:]):
            prefix.append(token)
        else:
            break
    return "/".join(prefix) if prefix else "."


def _try_get_scope_git_head_shas(
    *, base_directory: Path, scope_roots: list[str]
) -> dict[str, str]:
    out: dict[str, str] = {}
    base_resolved = base_directory.resolve()
    for scope_root in scope_roots:
        key = str(scope_root or ".")
        probe = (base_resolved / key).resolve() if key != "." else base_resolved
        candidates = [probe, base_resolved]
        seen: set[Path] = set()
        sha: str | None = None
        for cwd in candidates:
            if cwd in seen:
                continue
            seen.add(cwd)
            if not cwd.exists():
                continue
            try:
                proc = run_git(["rev-parse", "HEAD"], cwd=cwd, timeout_s=5)
            except Exception:
                continue
            if int(proc.returncode) != 0:
                continue
            candidate = str(proc.stdout or "").strip().lower()
            if not _GIT_HEAD_SHA_PATTERN.match(candidate):
                continue
            sha = candidate
            break
        if sha is not None:
            out[key] = sha
    return out


def _sanitize_llm_synthesis_configured(llm_config: object | None) -> dict[str, object] | None:
    if llm_config is None:
        return None
    try:
        get_provider_configs = getattr(llm_config, "get_provider_configs")
        _utility_cfg, synthesis_cfg = get_provider_configs()
    except Exception:
        return None
    if not isinstance(synthesis_cfg, dict):
        return None
    provider = synthesis_cfg.get("provider", "openai")
    model = synthesis_cfg.get("model", "gpt-5-nano")
    if not isinstance(provider, str) or not isinstance(model, str):
        return None

    out: dict[str, object] = {"provider": provider, "model": model}
    effort = synthesis_cfg.get("reasoning_effort")
    if isinstance(effort, str) and effort.strip():
        out["reasoning_effort"] = effort.strip()
    return out


def _refresh_snapshot_run_metadata_for_reuse(*, out_dir: Path, config: object) -> None:
    run_path = out_dir / "snapshot.run.json"
    if not run_path.exists():
        return
    try:
        payload = json.loads(run_path.read_text("utf-8"))
    except Exception:
        return
    if not isinstance(payload, dict):
        return
    if str(payload.get("schema_version") or "") != _RUN_SCHEMA_VERSION:
        return

    config_snapshot = payload.get("config_snapshot")
    if not isinstance(config_snapshot, dict):
        config_snapshot = {}
        payload["config_snapshot"] = config_snapshot

    config_snapshot["llm_synthesis_configured"] = _sanitize_llm_synthesis_configured(
        getattr(config, "llm", None)
    )

    base_directory_obj = payload.get("base_directory")
    base_directory = Path(str(base_directory_obj or ".")).resolve()
    scope_root_obj = payload.get("scope_root") or []
    scope_roots: list[str] = []
    if isinstance(scope_root_obj, list):
        scope_roots = [str(x) for x in scope_root_obj if str(x)]
    elif isinstance(scope_root_obj, str) and scope_root_obj.strip():
        scope_roots = [scope_root_obj.strip()]
    if not scope_roots:
        scope_roots = ["."]

    scope_git_head_shas = _try_get_scope_git_head_shas(
        base_directory=base_directory,
        scope_roots=scope_roots,
    )
    unique_shas = {sha for sha in scope_git_head_shas.values() if sha}
    payload["scope_git_head_sha"] = next(iter(unique_shas), None) if len(unique_shas) == 1 else None
    payload["scope_git_head_shas"] = dict(scope_git_head_shas)

    try:
        _write_json(run_path, payload)
    except Exception:
        return


def _duckdb_scope_clause(
    *, column: str, scope_like_prefixes: list[str | None]
) -> tuple[str, list[object]]:
    valid = [p for p in scope_like_prefixes if p]
    if not valid:
        return "", []
    placeholders = " OR ".join([f"{column} LIKE ? ESCAPE '\\'" for _ in valid])
    return f" AND ({placeholders})", list(valid)


def _duckdb_scope_where(
    *, column: str, scope_like_prefixes: list[str | None]
) -> tuple[str, list[object]]:
    clause, params = _duckdb_scope_clause(
        column=column,
        scope_like_prefixes=scope_like_prefixes,
    )
    if not clause:
        return "", []
    return f"WHERE {clause.removeprefix(' AND ')}", params


def _get_selector(args: argparse.Namespace, config: Config) -> _SnapshotSelector:
    provider = getattr(args, "embedding_provider", None)
    model = getattr(args, "embedding_model", None)
    dims = getattr(args, "embedding_dims", 1536)

    if provider is None and config.embedding is not None:
        provider = str(config.embedding.provider)
    if model is None and config.embedding is not None:
        model = str(config.embedding.get_default_model())

    if not provider or not model:
        raise RuntimeError(
            "Snapshot requires embedding selector. Provide --embedding-provider and "
            "--embedding-model (or configure embedding selector in config)."
        )

    try:
        dims_i = int(dims)
    except Exception as exc:
        raise RuntimeError("--embedding-dims must be an integer") from exc
    if dims_i <= 0:
        raise RuntimeError("--embedding-dims must be positive")

    return _SnapshotSelector(provider=str(provider), model=str(model), dims=dims_i)


def _code_chunk_type_values() -> list[str]:
    return [chunk_type.value for chunk_type in ChunkType if chunk_type.is_code]


def _approx_tokens(text: str) -> int:
    # Deterministic, tokenizer-free estimate suitable for unit tests.
    # Mirrors the common ~4 chars/token heuristic.
    return max(1, int(len(text) // 4))


_DUnder_RE = re.compile(r"^__[^_].*__?$")
_CLASS_DEF_RE = re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\b")


def _is_cryptic_symbol(symbol: str) -> bool:
    s = str(symbol or "").strip()
    if not s or s in {"<unknown>", "<none>"}:
        return True
    if s.startswith("__") and s.endswith("__"):
        return True
    if s in {"__repr__", "__str__", "__init__", "__call__", "__iter__", "__getitem__"}:
        return True
    if len(s) <= 2:
        return True
    if all(ch in "_." for ch in s):
        return True
    if _DUnder_RE.match(s) is not None:
        return True
    return False


def _cluster_is_cryptic(
    *,
    cluster: dict,
    item_by_chunk_id: dict[int, _SnapshotItem] | None = None,
    chunk_hints_by_id: dict[int, _ChunkNameHints] | None = None,
) -> bool:
    top_symbols = cluster.get("top_symbols") or []
    if isinstance(top_symbols, list) and top_symbols:
        pairs: list[tuple[str, int]] = []
        for row in top_symbols[:16]:
            if not isinstance(row, dict):
                continue
            sym = row.get("symbol")
            cnt = row.get("count")
            try:
                cnt_i = int(cnt or 0)
            except Exception:
                cnt_i = 0
            pairs.append((str(sym) if sym is not None else "", max(0, cnt_i)))

        total = sum(cnt for _, cnt in pairs)
        if total > 0:
            cryptic = sum(cnt for sym, cnt in pairs if _is_cryptic_symbol(sym))
            ratio = cryptic / float(total)
            if ratio >= 0.55:
                return True
            if cryptic >= 6:
                return True

    ids = cluster.get("chunk_ids") or []
    if (
        isinstance(ids, list)
        and ids
        and item_by_chunk_id is not None
        and chunk_hints_by_id is not None
    ):
        checked = 0
        cryptic = 0
        for raw_id in ids[:20]:
            try:
                chunk_id = int(raw_id)
            except Exception:
                continue
            item = item_by_chunk_id.get(chunk_id)
            if item is None:
                continue
            hint = chunk_hints_by_id.get(chunk_id)
            sym = hint.symbol if hint is not None else (item.symbol or "")
            checked += 1
            cryptic += 1 if _is_cryptic_symbol(sym) else 0
        if checked and (cryptic / float(checked)) >= 0.6:
            return True

    return False


_PATHY_LABEL_RE = re.compile(
    r"[/\\].*\.(py|js|ts|tsx|jsx|go|java|rs|c|cc|cpp|h|hpp)\b",
    re.I,
)


def _label_is_generic(label: str) -> bool:
    s = str(label or "").strip()
    if not s:
        return True
    lower = s.lower()
    if "__repr__" in lower or "__str__" in lower:
        return True
    if _PATHY_LABEL_RE.search(s) is not None:
        return True
    if "chunk system" in lower or "chunk-system" in lower or "cluster" in lower:
        return True
    if lower in {"misc", "miscellaneous", "other", "unknown", "general"}:
        return True
    if len(lower.split()) <= 2 and lower in {"config", "utils", "helpers", "common"}:
        return True
    return False


def _is_test_path(path: str) -> bool:
    p = str(path or "").replace("\\", "/")
    if not p:
        return False
    if "/tests/" in p or p.startswith("tests/") or p.startswith("chunkhound/tests/"):
        return True
    if "/test_" in p or p.endswith("_test.py"):
        return True
    return False


def _cluster_test_evidence_ratio(cluster: dict) -> tuple[float, int]:
    top_files = cluster.get("top_files") or []
    if not isinstance(top_files, list) or not top_files:
        return 0.0, 0

    total = 0
    test_total = 0
    for row in top_files:
        if not isinstance(row, dict):
            continue
        path = row.get("path")
        cnt = row.get("count")
        try:
            cnt_i = int(cnt or 0)
        except Exception:
            cnt_i = 0
        if cnt_i <= 0:
            continue
        total += cnt_i
        if isinstance(path, str) and _is_test_path(path):
            test_total += cnt_i

    if total <= 0:
        return 0.0, 0
    return (test_total / float(total)), int(total)


def _cluster_is_test_only(cluster: dict) -> bool:
    ratio, total = _cluster_test_evidence_ratio(cluster)
    return bool(total >= 10 and ratio >= 0.95)


def _cluster_is_test_heavy(cluster: dict) -> bool:
    ratio, total = _cluster_test_evidence_ratio(cluster)
    return bool(total >= 10 and ratio >= TEST_HEAVY_EVIDENCE_RATIO)


_TESTS_PREFIX_RE = re.compile(r"^\s*tests\s*:\s*(.*)$", re.I)
_LEADING_TEST_PHRASE_RE = re.compile(
    r"^\s*(testing\b|tests\b|test\s*suite\b|test\s*fixtures\b)\s*[:\-\u2013]?\s*(.*)$",
    re.I,
)


def _normalize_tests_prefix(label: str) -> str:
    raw = str(label or "").strip()
    if not raw:
        return "Tests:"

    m = _TESTS_PREFIX_RE.match(raw)
    if m is not None:
        rest = (m.group(1) or "").strip()
        return f"Tests: {rest}".rstrip() if rest else "Tests:"

    m2 = _LEADING_TEST_PHRASE_RE.match(raw)
    if m2 is not None:
        rest = (m2.group(2) or "").strip()
        if rest:
            return f"Tests: {rest}"

    return f"Tests: {raw}"


def _build_chunk_name_hints(
    *,
    chunk_ids: set[int],
    item_by_chunk_id: dict[int, _SnapshotItem],
) -> dict[int, _ChunkNameHints]:
    out: dict[int, _ChunkNameHints] = {}
    token_re = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")
    for chunk_id in chunk_ids:
        item = item_by_chunk_id.get(int(chunk_id))
        if item is None:
            continue
        symbol = str(item.symbol or "").strip()
        tokens = token_re.findall(symbol)[:24] if symbol else []
        out[int(chunk_id)] = _ChunkNameHints(
            symbol=symbol or "<unknown>",
            signature_line=None,
            identifier_tokens=tokens,
        )
    return out


def _available_selectors_in_scope(
    *,
    conn: duckdb.DuckDBPyConnection,
    scope_like_prefixes: list[str | None],
    allowed_chunk_types: list[str],
) -> list[_AvailableSelector]:
    tables = conn.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_name LIKE 'embeddings_%'
        ORDER BY table_name
        """
    ).fetchall()

    out: list[_AvailableSelector] = []
    for (table_name,) in tables:
        dims_raw = str(table_name).removeprefix("embeddings_")
        try:
            dims = int(dims_raw)
        except ValueError:
            continue

        scope_sql, scope_params = _duckdb_scope_clause(
            column="f.path", scope_like_prefixes=scope_like_prefixes
        )
        placeholders = ", ".join(["?" for _ in allowed_chunk_types])
        params: list[object] = []
        params.extend(scope_params)
        params.extend(allowed_chunk_types)

        rows = conn.execute(
            f"""
            SELECT e.provider, e.model, COUNT(*) AS cnt
            FROM {table_name} e
            JOIN chunks c ON c.id = e.chunk_id
            JOIN files f ON f.id = c.file_id
            WHERE 1=1
            {scope_sql}
              AND c.chunk_type IN ({placeholders})
            GROUP BY e.provider, e.model
            ORDER BY cnt DESC, e.provider, e.model
            """,
            params,
        ).fetchall()

        for provider, model, count in rows:
            out.append(
                _AvailableSelector(
                    provider=str(provider),
                    model=str(model),
                    dims=int(dims),
                    count=int(count),
                )
            )
    return out


def _format_available_selectors(selectors: list[_AvailableSelector]) -> str:
    if not selectors:
        return "  (none)"
    lines = []
    for selector in sorted(
        selectors,
        key=lambda s: (-int(s.count), str(s.provider), str(s.model), int(s.dims)),
    )[:20]:
        lines.append(
            "  "
            f"provider={selector.provider} model={selector.model} "
            f"dims={selector.dims} count={selector.count}"
        )
    return "\n".join(lines)


def _collect_chunk_system_prompt_chunk_ids(
    *, chunk_systems_payload: dict, item_by_chunk_id: dict[int, _SnapshotItem]
) -> set[int]:
    clusters = chunk_systems_payload.get("clusters") or []
    out: set[int] = set()
    if not isinstance(clusters, list):
        return out

    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        ids = cluster.get("chunk_ids") or []
        if not isinstance(ids, list):
            continue
        for raw in ids[:8]:
            try:
                chunk_id = int(raw)
            except Exception:
                continue
            if chunk_id in item_by_chunk_id:
                out.add(chunk_id)
    return out


def _batch_label_schema(*, allowed_ids: list[str] | None = None) -> dict[str, Any]:
    item_schema: dict[str, Any] = {
        "type": "object",
        "required": ["id", "label", "confidence"],
        "properties": {
            "id": {"type": ["string", "integer"]},
            "label": {
                "type": "string",
                "maxLength": int(LABEL_MAX_CHARS),
                "pattern": _LABEL_NO_NEWLINES_PATTERN,
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "additionalProperties": False,
    }
    if allowed_ids:
        item_schema["properties"]["id"] = {"type": "string", "enum": list(allowed_ids)}

    results_schema: dict[str, Any] = {
        "type": "array",
        "items": item_schema,
    }
    if allowed_ids:
        results_schema["minItems"] = int(len(allowed_ids))
        results_schema["maxItems"] = int(len(allowed_ids))

    return {
        "type": "object",
        "required": ["results"],
        "properties": {
            "results": {
                **results_schema,
            }
        },
        "additionalProperties": False,
    }


def _render_label_batch_prompt(*, kind: str, items: list[_BatchLabelRequest]) -> str:
    lines: list[str] = []
    lines.append(f"# Label batch ({kind})")
    lines.append("")
    lines.append("Return JSON with: {\"results\": [{\"id\",\"label\",\"confidence\"}]}")
    lines.append("Use concise labels suitable for operator-facing reports.")
    if kind == "chunk_systems":
        lines.append("")
        lines.append("Guidance (chunk systems):")
        lines.append(
            "- A label describes behavior/responsibility (what it does), not an "
            "artifact/stage (what exists)."
        )
        lines.append(
            "- Prefer a short, natural noun phrase naming the responsibility "
            "(2–7 words)."
            " 'X system' / 'X subsystem' is allowed but not required "
            "(use only if X is specific)."
        )
        lines.append(
            "- Avoid awkward forced gerunds/verb-phrases "
            "('Generating...', 'Orchestrating...')."
        )
        lines.append(
            "- Avoid filler endings like handling/processing/dispatching/writing/"
            "formatting/limiting/searching; prefer concrete nouns "
            "('... registry', '... resolver', '... writer', '... quota enforcement', "
            "'... search')."
        )
        lines.append(
            "- Avoid method-name labels like '__repr__/__str__ methods' unless intent is truly unclear."
        )
        lines.append(
            "- Avoid artifact/stage-only labels like '... content' or 'seed data'."
            " If evidence is fixtures/data, name their role "
            "(e.g. 'Tests: fixtures for <behavior>')."
            " Avoid file-path-ish labels."
        )
        lines.append(
            "- If evidence is mostly tests/fixtures/experiments/docs, label the behavior being exercised/supported."
        )
        lines.append(
            "- If the cluster is overwhelmingly tests/fixtures, the label must start with 'Tests:'."
        )
        lines.append("- If evidence is weak or conflicting, lower confidence.")
        lines.append(
            "- Confidence rubric: ~0.9 strong evidence, ~0.7 moderate, ~0.4 weak/guess."
        )
        lines.append(
            "- Self-check: if your draft label names an artifact/stage, rewrite it as a responsibility."
        )
    elif kind == "chunk_system_groups":
        lines.append("")
        lines.append("Guidance (chunk system groups):")
        lines.append(
            "- You label a group of systems (a higher-level responsibility/theme)."
        )
        lines.append(
            "- Prefer a short, natural noun phrase naming the shared responsibility (2–7 words)."
        )
        lines.append(
            "- If members span multiple subtopics, choose an umbrella label anchored on the dominant hub/subsystem "
            "(often the most central/representative member), not a two-topic mashup."
        )
        lines.append(
            "- Self-check: the label should plausibly apply to most member systems; if it fits only 1–2, broaden."
        )
        lines.append(
            "- Avoid listing member names/paths; name what the group *does*."
        )
        lines.append(
            "- Avoid artifact/stage-only labels like '... content' or 'seed data'."
        )
        lines.append(
            "- If the group is overwhelmingly tests/fixtures, the label must start with 'Tests:'."
        )
        lines.append("- If evidence is weak or conflicting, lower confidence.")
    lines.append("")
    for item in items:
        lines.append(f"## id: {item.item_id}")
        lines.append(item.content.strip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _pack_label_batches(
    *,
    kind: str,
    items: list[_BatchLabelRequest],
    estimate_tokens: Callable[[str], int],
    max_tokens: int,
    max_items: int,
) -> list[list[_BatchLabelRequest]]:
    del kind

    max_tokens_i = max(1, int(max_tokens))
    max_items_i = max(1, int(max_items))

    batches: list[list[_BatchLabelRequest]] = []
    current: list[_BatchLabelRequest] = []
    current_tokens = 0

    for item in items:
        item_tokens = max(1, int(estimate_tokens(item.content)))

        would_exceed_items = len(current) >= max_items_i
        would_exceed_tokens = bool(current) and (current_tokens + item_tokens > max_tokens_i)

        if would_exceed_items or would_exceed_tokens:
            batches.append(current)
            current = []
            current_tokens = 0

        current.append(item)
        current_tokens += item_tokens

    if current:
        batches.append(current)

    return batches


def _take_label_batch(
    *,
    items: list[_BatchLabelRequest],
    estimate_tokens: Callable[[str], int],
    max_tokens: int,
    max_items: int,
) -> tuple[list[_BatchLabelRequest], list[_BatchLabelRequest]]:
    if not items:
        return [], []

    max_tokens_i = max(1, int(max_tokens))
    max_items_i = max(1, int(max_items))

    batch: list[_BatchLabelRequest] = []
    batch_tokens = 0
    for item in items:
        item_tokens = max(1, int(estimate_tokens(item.content)))
        would_exceed_items = len(batch) >= max_items_i
        would_exceed_tokens = bool(batch) and (batch_tokens + item_tokens > max_tokens_i)
        if would_exceed_items or would_exceed_tokens:
            break
        batch.append(item)
        batch_tokens += item_tokens

    if not batch:
        batch = [items[0]]

    rest = items[len(batch) :]
    return batch, rest


async def _run_structured_label_batches_parallel(
    *,
    provider: Any,
    kind: str,
    items: list[_BatchLabelRequest],
    out_dir: Path,
    batch_prefix: str,
    estimate_tokens: Callable[[str], int],
    max_tokens: int,
    max_items: int,
    concurrency: int,
    system: str,
    max_completion_tokens: int,
    progress: Any | None = None,
    progress_task_id: Any | None = None,
    on_batch_complete: Callable[[list[_BatchLabelRequest], _StructuredBatchOutcome, int], None]
    | None = None,
) -> int:
    if not items:
        return 0

    max_items_i = max(1, int(max_items))
    max_tokens_i = max(1, int(max_tokens))
    concurrency_i = max(1, int(concurrency))

    pending = list(items)
    adaptive_max_items = int(max_items_i)
    batch_idx = 0
    llm_calls_total = 0

    api_semaphore = asyncio.Semaphore(concurrency_i)

    async def _run_one(batch: list[_BatchLabelRequest], idx: int) -> tuple[int, list[_BatchLabelRequest], _StructuredBatchOutcome]:
        prompt_path = out_dir / f"{batch_prefix}_{idx:04d}.md"
        outcome = await _complete_structured_label_batch_with_replay(
            provider=provider,
            kind=kind,
            items=batch,
            prompt_path=prompt_path,
            system=system,
            max_completion_tokens=int(max_completion_tokens),
            api_semaphore=api_semaphore,
        )
        return int(idx), batch, outcome

    in_flight: dict[asyncio.Task[tuple[int, list[_BatchLabelRequest], _StructuredBatchOutcome]], int] = {}

    while pending or in_flight:
        while pending and len(in_flight) < concurrency_i:
            batch_idx += 1
            batch, pending = _take_label_batch(
                items=pending,
                estimate_tokens=estimate_tokens,
                max_tokens=max_tokens_i,
                max_items=adaptive_max_items,
            )
            if progress is not None and progress_task_id is not None:
                try:
                    progress.update(
                        progress_task_id,
                        info=f"{batch_prefix} {batch_idx} (n={len(batch)}; in_flight={len(in_flight)+1}/{concurrency_i})",
                    )
                except Exception:
                    pass
            task = asyncio.create_task(_run_one(batch, int(batch_idx)))
            in_flight[task] = int(batch_idx)

        if not in_flight:
            continue

        done, _pending_tasks = await asyncio.wait(
            in_flight.keys(), return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            idx = int(in_flight.pop(task, 0) or 0)
            try:
                _idx, batch, outcome = task.result()
            except Exception:
                for other in list(in_flight.keys()):
                    other.cancel()
                await asyncio.gather(*list(in_flight.keys()), return_exceptions=True)
                raise

            llm_calls_total += int(outcome.llm_calls)
            if outcome.min_successful_batch_items > 0:
                adaptive_max_items = min(
                    int(adaptive_max_items), int(outcome.min_successful_batch_items)
                )
            if on_batch_complete is not None:
                try:
                    on_batch_complete(batch, outcome, idx)
                except Exception:
                    # Caller callback is best-effort; errors shouldn't break labeling.
                    pass

    return int(llm_calls_total)


def _parse_validate_batch_results(
    *, requested_ids: list[str], response: object
) -> dict[str, tuple[str, float]]:
    if not isinstance(response, dict):
        raise RuntimeError("Structured response must be an object")

    results = response.get("results")
    if not isinstance(results, list):
        raise RuntimeError("Structured response must contain results[]")

    allowed = {str(x) for x in requested_ids}
    parsed: dict[str, tuple[str, float]] = {}

    for row in results:
        if not isinstance(row, dict):
            raise RuntimeError("Each result item must be an object")

        raw_id = row.get("id")
        item_id = str(raw_id).strip()
        if not item_id:
            raise RuntimeError("Result id is missing/empty")

        if item_id not in allowed:
            raise RuntimeError(f"Unexpected result id: {item_id}")

        label = row.get("label")
        if not isinstance(label, str) or not label.strip():
            raise RuntimeError(f"Result label missing for id={item_id}")

        confidence = row.get("confidence")
        try:
            conf_f = float(confidence)
        except Exception as exc:
            raise RuntimeError(f"Result confidence invalid for id={item_id}") from exc

        conf_f = min(1.0, max(0.0, conf_f))
        parsed[item_id] = (label.strip(), conf_f)

    missing = [item_id for item_id in requested_ids if item_id not in parsed]
    if missing:
        raise RuntimeError(f"Missing results for ids: {missing}")

    return parsed


_TRUNCATION_ERROR_RE = re.compile(
    r"(structured completion truncated|token limit exceeded|finish_reason.*\blength\b|context length|output truncated)",
    re.I,
)


def _is_llm_truncation_error(exc: BaseException) -> bool:
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if _TRUNCATION_ERROR_RE.search(str(cur) or "") is not None:
            return True
        nxt = cur.__cause__ or cur.__context__
        cur = nxt if isinstance(nxt, BaseException) else None
    return False


def _prompt_path_with_suffix(path: Path, suffix: str) -> Path:
    return path.with_name(path.stem + suffix + path.suffix)


_LLM_TRANSIENT_ERROR_MARKERS = (
    "timeout",
    "timed out",
    "connection reset",
    "connection aborted",
    "temporarily unavailable",
    "service unavailable",
    "overloaded",
    "rate limit",
    "429",
    "500",
    "502",
    "503",
    "504",
)


def _is_transient_llm_error(exc: BaseException) -> bool:
    if _is_llm_truncation_error(exc):
        return False

    error_type = type(exc).__name__
    if error_type in {
        "TimeoutError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "APIConnectionError",
        "APITimeoutError",
    }:
        return True

    if isinstance(exc, (TimeoutError, ConnectionError, OSError, asyncio.TimeoutError)):
        return True

    error_str = str(exc).lower()
    return any(marker in error_str for marker in _LLM_TRANSIENT_ERROR_MARKERS)


def _retry_sleep_seconds(
    *,
    attempt: int,
    base_delay_seconds: float,
    max_delay_seconds: float,
    jitter_ratio: float,
) -> float:
    base = float(base_delay_seconds) * (2**int(attempt))
    delay = min(float(max_delay_seconds), base)
    jitter = delay * random.uniform(0.0, float(jitter_ratio))
    return max(0.0, float(delay + jitter))


@dataclass(frozen=True)
class _StructuredBatchOutcome:
    parsed: dict[str, tuple[str, float]]
    llm_calls: int
    min_successful_batch_items: int


async def _complete_structured_label_batch_with_replay(
    *,
    provider: Any,
    kind: str,
    items: list[_BatchLabelRequest],
    prompt_path: Path,
    system: str,
    max_completion_tokens: int,
    api_semaphore: asyncio.Semaphore | None = None,
    retry_max_attempts: int = 5,
    retry_base_delay_seconds: float = 1.0,
    retry_max_delay_seconds: float = 30.0,
    retry_jitter_ratio: float = 0.2,
    replay_depth: int = 0,
    replay_max_depth: int = 12,
) -> _StructuredBatchOutcome:
    if not items:
        return _StructuredBatchOutcome(parsed={}, llm_calls=0, min_successful_batch_items=0)

    batch_ids = [item.item_id for item in items]
    prompt = _render_label_batch_prompt(kind=kind, items=items)
    prompt_path.write_text(prompt, encoding="utf-8")

    llm_calls = 0
    schema = _batch_label_schema(allowed_ids=batch_ids)

    async def _call_once() -> object:
        if api_semaphore is None:
            return await provider.complete_structured(
                prompt=prompt,
                json_schema=schema,
                system=system,
                max_completion_tokens=int(max_completion_tokens),
            )
        async with api_semaphore:
            return await provider.complete_structured(
                prompt=prompt,
                json_schema=schema,
                system=system,
                max_completion_tokens=int(max_completion_tokens),
            )

    last_exc: Exception | None = None
    max_attempts = max(1, int(retry_max_attempts))

    for attempt in range(max_attempts):
        try:
            llm_calls += 1
            response = await _call_once()
            parsed = _parse_validate_batch_results(requested_ids=batch_ids, response=response)
            return _StructuredBatchOutcome(
                parsed=parsed,
                llm_calls=int(llm_calls),
                min_successful_batch_items=int(len(items)),
            )
        except Exception as exc:
            last_exc = exc
            if (
                _is_llm_truncation_error(exc)
                and len(items) > 1
                and int(replay_depth) < int(replay_max_depth)
            ):
                mid = max(1, len(items) // 2)
                left = items[:mid]
                right = items[mid:]
                logger.warning(
                    "Structured label batch truncated; retrying with smaller batches "
                    f"(n={len(items)} -> {len(left)}+{len(right)})."
                )
                left_coro = _complete_structured_label_batch_with_replay(
                    provider=provider,
                    kind=kind,
                    items=left,
                    prompt_path=_prompt_path_with_suffix(
                        prompt_path, f"_split{replay_depth+1}a"
                    ),
                    system=system,
                    max_completion_tokens=max_completion_tokens,
                    api_semaphore=api_semaphore,
                    retry_max_attempts=retry_max_attempts,
                    retry_base_delay_seconds=retry_base_delay_seconds,
                    retry_max_delay_seconds=retry_max_delay_seconds,
                    retry_jitter_ratio=retry_jitter_ratio,
                    replay_depth=int(replay_depth) + 1,
                    replay_max_depth=replay_max_depth,
                )
                right_coro = _complete_structured_label_batch_with_replay(
                    provider=provider,
                    kind=kind,
                    items=right,
                    prompt_path=_prompt_path_with_suffix(
                        prompt_path, f"_split{replay_depth+1}b"
                    ),
                    system=system,
                    max_completion_tokens=max_completion_tokens,
                    api_semaphore=api_semaphore,
                    retry_max_attempts=retry_max_attempts,
                    retry_base_delay_seconds=retry_base_delay_seconds,
                    retry_max_delay_seconds=retry_max_delay_seconds,
                    retry_jitter_ratio=retry_jitter_ratio,
                    replay_depth=int(replay_depth) + 1,
                    replay_max_depth=replay_max_depth,
                )
                left_out, right_out = await asyncio.gather(left_coro, right_coro)
                combined = dict(left_out.parsed)
                combined.update(right_out.parsed)
                return _StructuredBatchOutcome(
                    parsed=combined,
                    llm_calls=int(llm_calls + left_out.llm_calls + right_out.llm_calls),
                    min_successful_batch_items=int(
                        min(
                            left_out.min_successful_batch_items,
                            right_out.min_successful_batch_items,
                        )
                    ),
                )

            if not _is_transient_llm_error(exc) or attempt >= max_attempts - 1:
                raise

            sleep_seconds = _retry_sleep_seconds(
                attempt=attempt,
                base_delay_seconds=float(retry_base_delay_seconds),
                max_delay_seconds=float(retry_max_delay_seconds),
                jitter_ratio=float(retry_jitter_ratio),
            )
            logger.warning(
                "Transient LLM error during structured labeling; retrying in "
                f"{sleep_seconds:.2f}s (attempt {attempt + 1}/{max_attempts}). Error: {exc}"
            )
            if sleep_seconds > 0:
                await asyncio.sleep(sleep_seconds)

    raise RuntimeError(
        f"Structured label batch failed after {max_attempts} attempts: {last_exc}"
    )


def _chunk_system_prompt_body_rich(
    *,
    cluster: dict,
    item_by_chunk_id: dict[int, _SnapshotItem],
    chunk_hints_by_id: dict[int, _ChunkNameHints],
    is_test_only: bool = False,
) -> list[str]:
    lines: list[str] = []
    cluster_id = int(cluster.get("cluster_id") or 0)
    size = int(cluster.get("size") or 0)
    lines.append(f"Cluster ID: {cluster_id}")
    lines.append(f"Size: {size}")
    lines.append(
        "Label style: behavior/responsibility (short noun phrase). "
        "'X system' is allowed but not required. Prefer concrete component nouns "
        "(e.g. engine/parser/registry/harness) over filler '-ing' endings "
        "(handling/processing/etc). Avoid artifact/stage labels like 'content' or 'seed data'."
    )
    ratio, total = _cluster_test_evidence_ratio(cluster)
    if bool(is_test_only):
        lines.append("Hint: test-only cluster (>=95% test evidence). Label must start with 'Tests:'.")
    elif bool(total >= 10 and ratio >= TEST_HEAVY_EVIDENCE_RATIO):
        pct = int(round(100.0 * ratio))
        lines.append(
            f"Hint: mostly tests/fixtures (~{pct}% test evidence). "
            "Label must start with 'Tests:' and name the behavior being exercised."
        )

    top_files = cluster.get("top_files") or []
    if isinstance(top_files, list) and top_files:
        lines.append("Top files:")
        for row in top_files[:8]:
            if not isinstance(row, dict):
                continue
            lines.append(f"- {row.get('path')}: {row.get('count')}")

    top_symbols = cluster.get("top_symbols") or []
    if isinstance(top_symbols, list) and top_symbols:
        lines.append("Top symbols:")
        for row in top_symbols[:8]:
            if not isinstance(row, dict):
                continue
            symbol = row.get("symbol")
            lines.append(f"- {symbol if symbol is not None else '<none>'}: {row.get('count')}")

    ids = cluster.get("chunk_ids") or []
    if isinstance(ids, list) and ids:
        lines.append("Representative chunks:")
        for raw_id in ids[:5]:
            try:
                chunk_id = int(raw_id)
            except Exception:
                continue
            item = item_by_chunk_id.get(chunk_id)
            if item is None:
                continue
            hint = chunk_hints_by_id.get(chunk_id)
            symbol = hint.symbol if hint is not None else (item.symbol or "<none>")
            lines.append(
                f"- {item.path}#L{item.start_line}-L{item.end_line} "
                f"{item.chunk_type} symbol={symbol}"
            )

    return lines


def _chunk_system_excerpt_candidate_ids(
    *,
    cluster: dict,
    item_by_chunk_id: dict[int, _SnapshotItem],
    chunk_hints_by_id: dict[int, _ChunkNameHints],
    max_candidates: int,
) -> list[int]:
    ids = cluster.get("chunk_ids") or []
    if not isinstance(ids, list) or not ids:
        return []

    top_files = cluster.get("top_files") or []
    file_rank: dict[str, int] = {}
    if isinstance(top_files, list):
        for idx, row in enumerate(top_files[:64]):
            if not isinstance(row, dict):
                continue
            path = row.get("path")
            if isinstance(path, str) and path and path not in file_rank:
                file_rank[path] = idx

    chunk_type_rank = {
        "class": 0,
        "function": 1,
        "method": 2,
    }

    candidates: list[tuple[tuple[int, int, int, int, int], int]] = []
    for raw_id in ids:
        try:
            chunk_id = int(raw_id)
        except Exception:
            continue
        item = item_by_chunk_id.get(chunk_id)
        if item is None:
            continue
        hint = chunk_hints_by_id.get(chunk_id)
        sym = hint.symbol if hint is not None else (item.symbol or "")
        cryptic = 1 if _is_cryptic_symbol(sym) else 0
        fr = file_rank.get(item.path, 10**9)
        ctr = chunk_type_rank.get(str(item.chunk_type or "").strip().lower(), 50)
        key = (cryptic, fr, ctr, int(item.start_line or 0), int(chunk_id))
        candidates.append((key, chunk_id))

    if not candidates:
        return []

    candidates.sort(key=lambda pair: pair[0])

    # Diversity pass: take one "best" candidate per file first (in top_files order).
    chosen: list[int] = []
    seen_ids: set[int] = set()
    best_by_file: dict[str, int] = {}
    for _, chunk_id in candidates:
        item = item_by_chunk_id.get(chunk_id)
        if item is None:
            continue
        if item.path not in best_by_file:
            best_by_file[item.path] = chunk_id

    file_order = sorted(
        best_by_file.keys(), key=lambda p: (file_rank.get(p, 10**9), p)
    )
    for path in file_order:
        chunk_id = best_by_file[path]
        if chunk_id not in seen_ids:
            chosen.append(chunk_id)
            seen_ids.add(chunk_id)
            if len(chosen) >= max_candidates:
                return chosen

    for _, chunk_id in candidates:
        if chunk_id in seen_ids:
            continue
        chosen.append(chunk_id)
        seen_ids.add(chunk_id)
        if len(chosen) >= max_candidates:
            break

    return chosen


def _truncate_code_to_token_budget(
    *,
    code: str,
    estimate_tokens: Callable[[str], int],
    max_tokens: int,
) -> str:
    max_tokens_i = max(1, int(max_tokens))
    raw = str(code or "").rstrip("\n")
    if not raw:
        return ""
    if estimate_tokens(raw) <= max_tokens_i:
        return raw

    lines = raw.splitlines()
    if not lines:
        return ""

    head_n = min(len(lines), 80)
    tail_n = min(max(0, len(lines) - head_n), 30)

    for _ in range(10):
        out_lines = list(lines[:head_n])
        if tail_n > 0:
            out_lines.append("… (truncated) …")
            out_lines.extend(lines[-tail_n:])
        candidate = "\n".join(out_lines).rstrip("\n")
        if estimate_tokens(candidate) <= max_tokens_i:
            return candidate
        if head_n <= 10 and tail_n <= 5:
            break
        head_n = max(10, head_n // 2)
        tail_n = max(5, tail_n // 2) if tail_n > 0 else 0

    # Fallback: crude char budget (~4 chars/token).
    char_budget = max(64, max_tokens_i * 4)
    if len(raw) <= char_budget:
        return raw
    head_chars = max(32, int(char_budget * 0.7))
    tail_chars = max(0, char_budget - head_chars - 20)
    head = raw[:head_chars].rstrip("\n")
    tail = raw[-tail_chars:].lstrip("\n") if tail_chars > 0 else ""
    if tail:
        return (head + "\n… (truncated) …\n" + tail).rstrip("\n")
    return head.rstrip("\n")


def _pack_chunk_code_excerpts(
    *,
    cluster: dict,
    item_by_chunk_id: dict[int, _SnapshotItem],
    chunk_hints_by_id: dict[int, _ChunkNameHints],
    code_by_chunk_id: dict[int, str],
    max_content_tokens: int,
    estimate_tokens: Callable[[str], int],
    enclosing_class_by_chunk_id: dict[int, str] | None = None,
) -> list[str]:
    max_tokens_i = max(1, int(max_content_tokens))

    candidate_ids = _chunk_system_excerpt_candidate_ids(
        cluster=cluster,
        item_by_chunk_id=item_by_chunk_id,
        chunk_hints_by_id=chunk_hints_by_id,
        max_candidates=250,
    )
    if not candidate_ids:
        return []

    lines: list[str] = []
    used_tokens = 0

    for chunk_id in candidate_ids:
        item = item_by_chunk_id.get(chunk_id)
        if item is None:
            continue
        code = code_by_chunk_id.get(chunk_id, "")
        if not isinstance(code, str) or not code.strip():
            continue

        hint = chunk_hints_by_id.get(chunk_id)
        symbol = hint.symbol if hint is not None else (item.symbol or "<none>")
        class_hint = ""
        if enclosing_class_by_chunk_id is not None:
            cls = enclosing_class_by_chunk_id.get(chunk_id)
            if isinstance(cls, str) and cls.strip():
                class_hint = f" enclosing_class={cls.strip()}"

        header = (
            f"### {item.path}#L{item.start_line}-L{item.end_line} "
            f"{item.chunk_type} symbol={symbol}{class_hint} (chunk_id={chunk_id})"
        )

        remaining = max_tokens_i - used_tokens
        if remaining <= 50:
            break

        overhead = estimate_tokens(header) + 10
        code_budget = max(1, remaining - overhead)
        code_budget = min(code_budget, 1400)

        snippet = _truncate_code_to_token_budget(
            code=code,
            estimate_tokens=estimate_tokens,
            max_tokens=code_budget,
        )
        if not snippet.strip():
            continue

        excerpt_text = header + "\n```" + "\n" + snippet.rstrip("\n") + "\n```\n"
        excerpt_tokens = estimate_tokens(excerpt_text)
        if used_tokens and (used_tokens + excerpt_tokens > max_tokens_i):
            break
        if excerpt_tokens > remaining:
            break

        lines.extend(excerpt_text.rstrip("\n").splitlines())
        lines.append("")
        used_tokens += excerpt_tokens

    if lines and not lines[-1].strip():
        while lines and not lines[-1].strip():
            lines.pop()
    return lines


def _chunk_system_prompt_body_deep(
    *,
    cluster: dict,
    item_by_chunk_id: dict[int, _SnapshotItem],
    chunk_hints_by_id: dict[int, _ChunkNameHints],
    code_by_chunk_id: dict[int, str],
    estimate_tokens: Callable[[str], int],
    enclosing_class_by_chunk_id: dict[int, str] | None = None,
    is_test_only: bool = False,
) -> list[str]:
    lines = _chunk_system_prompt_body_rich(
        cluster=cluster,
        item_by_chunk_id=item_by_chunk_id,
        chunk_hints_by_id=chunk_hints_by_id,
        is_test_only=is_test_only,
    )
    lines.append("")
    lines.append(
        "This cluster looks cryptic/generic. Use the code excerpts below to infer the *purpose*."
    )
    lines.append(
        "Avoid method-name-driven labels (e.g. '__repr__ methods'); prefer intent (e.g. redaction, summary)."
    )
    lines.append(
        "Avoid artifact/stage labels (e.g. '... content', 'seed data'); name the responsibility."
    )
    lines.append("")
    lines.append(
        f"Code excerpts (token-bounded; <= {MAX_CHUNK_CONTENT_TOKENS_PER_CLUSTER} tokens):"
    )
    excerpt_lines = _pack_chunk_code_excerpts(
        cluster=cluster,
        item_by_chunk_id=item_by_chunk_id,
        chunk_hints_by_id=chunk_hints_by_id,
        code_by_chunk_id=code_by_chunk_id,
        max_content_tokens=MAX_CHUNK_CONTENT_TOKENS_PER_CLUSTER,
        estimate_tokens=estimate_tokens,
        enclosing_class_by_chunk_id=enclosing_class_by_chunk_id,
    )
    if excerpt_lines:
        lines.extend(excerpt_lines)
    else:
        lines.append("<no code excerpts available>")
    return lines


def _chunk_system_prompt(
    *,
    cluster_id: int,
    body_lines: list[str] | None = None,
) -> str:
    lines: list[str] = []
    lines.append("# Chunk-system labeling")
    lines.append("")
    lines.append("You label chunk-level clusters for an operator-facing snapshot report.")
    lines.append("Use the filenames/paths, chunk types, and symbols below.")
    lines.append("Prefer a short label that captures the concept/subsystem.")
    lines.append("")
    lines.append(f"Cluster: {cluster_id}")
    lines.extend(body_lines or [])
    lines.append("")
    lines.append("Return JSON with keys: label (string), confidence (number 0..1).")
    return "\n".join(lines).strip() + "\n"


def _resolution_key(resolution: float) -> str:
    s = f"{float(resolution):.6g}"
    s = s.replace("-", "m").replace(".", "p")
    return s


def _chunk_system_group_prompt(
    *,
    resolution: float,
    group_id: int,
    body_lines: list[str] | None = None,
) -> str:
    lines: list[str] = []
    lines.append("# Chunk-system group labeling")
    lines.append("")
    lines.append(
        "You label *groups of chunk-systems* for an operator-facing snapshot report."
    )
    lines.append(
        "A group represents a higher-level responsibility/theme shared by multiple systems."
    )
    lines.append("Prefer a short label that captures the shared behavior/subsystem.")
    lines.append("")
    lines.append(f"Resolution: {float(resolution)}")
    lines.append(f"Group ID: {int(group_id)}")
    lines.extend(body_lines or [])
    lines.append("")
    lines.append("Return JSON with keys: label (string), confidence (number 0..1).")
    return "\n".join(lines).strip() + "\n"


def _chunk_system_group_prompt_body(
    *,
    resolution: float,
    group_id: int,
    member_systems: list[dict[str, object]],
    system_label_by_id: dict[int, str],
    is_test_only: bool,
    is_test_heavy: bool,
    max_member_systems: int,
    max_files: int,
    max_symbols: int,
    prompt_mode: str,
) -> list[str]:
    def is_tests_prefix(label: str) -> bool:
        return str(label or "").strip().lower().startswith("tests:")

    def _system_display_name(label: str, *, cluster_id: int) -> str:
        raw = str(label or "").strip()
        if not raw:
            return f"System #{int(cluster_id)}"

        tests_prefix = ""
        m = _TESTS_PREFIX_RE.match(raw)
        if m is not None:
            tests_prefix = "Tests:"
            raw = (m.group(1) or "").strip()

        if " · " in raw:
            raw = raw.split(" · ", 1)[0].strip()

        raw = raw.replace("\\", "/")
        if "/" in raw:
            raw = raw.rsplit("/", 1)[-1].strip()

        if "." in raw:
            stem, ext = raw.rsplit(".", 1)
            if ext.lower() in {
                "py",
                "md",
                "txt",
                "json",
                "yaml",
                "yml",
                "toml",
                "ini",
                "cfg",
                "sh",
                "bash",
                "js",
                "ts",
                "tsx",
                "jsx",
            }:
                raw = stem.strip()

        raw = raw.replace("_", " ").replace("-", " ").replace(".", " ")
        raw = re.sub(r"\s+", " ", raw).strip()
        if not raw:
            raw = f"System #{int(cluster_id)}"

        if tests_prefix:
            return f"{tests_prefix} {raw}".rstrip()
        return raw

    def _system_is_testy(system: dict[str, object], label: str) -> bool:
        if is_tests_prefix(label):
            return True
        tf = system.get("top_files") or []
        if isinstance(tf, list):
            for row in tf:
                if not isinstance(row, dict):
                    continue
                path = row.get("path")
                if isinstance(path, str) and _is_test_path(path):
                    return True
        return False

    def member_sort_key(system: dict[str, object]) -> tuple[int, int, int]:
        cluster_id = int(system.get("cluster_id") or 0)
        label = system_label_by_id.get(cluster_id) or str(system.get("label") or "")
        return (
            1 if _system_is_testy(system, label) else 0,
            -int(system.get("size") or 0),
            cluster_id,
        )

    members_sorted = sorted(member_systems, key=member_sort_key)
    lines: list[str] = []
    mode = str(prompt_mode or "").strip().lower()
    if mode not in {"systems_only", "full"}:
        mode = "systems_only"

    if mode == "systems_only":
        lines.append("Member systems:")
        for s in members_sorted:
            cid = int(s.get("cluster_id") or 0)
            lab_raw = system_label_by_id.get(cid) or str(s.get("label") or "")
            lab = _system_display_name(lab_raw, cluster_id=cid)
            lines.append(f"- #{cid}: {lab}")
        return lines

    lines.append(f"Systems: {int(len(member_systems))}")
    lines.append("")
    lines.append(
        "Label style: behavior/responsibility (short noun phrase). "
        "'X system' is allowed but not required. Prefer concrete component nouns "
        "(engine/parser/registry/harness) over filler '-ing' endings."
    )
    lines.append(
        "Avoid artifact/stage labels like 'content' or 'seed data'; name the responsibility."
    )
    if is_test_only:
        lines.append(
            "Hint: test-only group (>=95% test evidence). Label must start with 'Tests:'."
        )
    elif is_test_heavy:
        lines.append(
            "Hint: test-heavy group (>=90% test evidence). If appropriate, label should start with 'Tests:'."
        )
    lines.append("")

    lines.append(f"Member systems (top {int(max_member_systems)}):")
    shown = members_sorted[: max(1, int(max_member_systems))]
    for s in shown:
        cid = int(s.get("cluster_id") or 0)
        lab_raw = system_label_by_id.get(cid) or str(s.get("label") or "")
        lab = _system_display_name(lab_raw, cluster_id=cid)
        lines.append(f"- #{cid}: {lab}")
    if len(member_systems) > len(shown):
        lines.append(f"- +{int(len(member_systems) - len(shown))} more")
    lines.append("")

    # Aggregate top files across member systems.
    file_counts: dict[str, int] = {}
    seen_test_files = False
    seen_non_test_files = False
    for s in member_systems:
        tf = s.get("top_files") or []
        if not isinstance(tf, list):
            continue
        for row in tf:
            if not isinstance(row, dict):
                continue
            path = row.get("path")
            cnt = row.get("count")
            if not isinstance(path, str) or not path:
                continue
            if _is_test_path(path):
                seen_test_files = True
            else:
                seen_non_test_files = True
            try:
                cnt_i = int(cnt or 0)
            except Exception:
                cnt_i = 0
            if cnt_i <= 0:
                continue
            file_counts[path] = int(file_counts.get(path, 0)) + int(cnt_i)
    if file_counts:
        if seen_test_files and seen_non_test_files:
            lines.append(
                "Hint: mixed group. Use an umbrella label anchored on the dominant non-test subsystem "
                "(see Top files), not a narrow label that fits only a single member."
            )
            lines.append("")
        lines.append(f"Top files (aggregated; top {int(max_files)}):")
        for path, cnt in sorted(
            file_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))
        )[: max(1, int(max_files))]:
            lines.append(f"- {path} ({int(cnt)})")
        lines.append("")

    sym_counts: dict[str, int] = {}
    for s in member_systems:
        ts = s.get("top_symbols") or []
        if not isinstance(ts, list):
            continue
        for row in ts:
            if not isinstance(row, dict):
                continue
            sym = row.get("symbol")
            cnt = row.get("count")
            if not isinstance(sym, str) or not sym:
                continue
            try:
                cnt_i = int(cnt or 0)
            except Exception:
                cnt_i = 0
            if cnt_i <= 0:
                continue
            sym_counts[sym] = int(sym_counts.get(sym, 0)) + int(cnt_i)
    if sym_counts:
        lines.append(f"Top symbols (aggregated; top {int(max_symbols)}):")
        for sym, cnt in sorted(
            sym_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))
        )[: max(1, int(max_symbols))]:
            lines.append(f"- {sym} ({int(cnt)})")
        lines.append("")

    return lines


async def _label_chunk_system_groups_llm(
    *,
    system_groups_payload: dict[str, object],
    out_dir: Path,
    llm_manager: LLMManager | None,
    llm_dry_run: bool,
    llm_label_batching: bool,
    llm_label_batch_max_items: int,
    llm_label_batch_max_tokens: int,
    llm_label_concurrency: int | None = None,
    system_label_by_id: dict[int, str],
    prompt_mode: str,
    progress: Any | None = None,
    progress_task_id: Any | None = None,
) -> tuple[dict[str, object], int, int]:
    systems_obj = system_groups_payload.get("systems")
    parts_obj = system_groups_payload.get("partitions")
    params_obj = system_groups_payload.get("params") or {}
    if not isinstance(systems_obj, list) or not isinstance(parts_obj, list):
        raise RuntimeError("system groups payload missing systems/partitions")

    systems: list[dict[str, object]] = [s for s in systems_obj if isinstance(s, dict)]
    partitions_in: list[dict[str, object]] = [
        p for p in parts_obj if isinstance(p, dict)
    ]

    max_member_systems = 12
    max_files = 12
    max_symbols = 12

    estimate_tokens: Callable[[str], int]
    if llm_manager is not None:
        estimate_tokens = llm_manager.get_synthesis_provider().estimate_tokens
    else:
        estimate_tokens = _approx_tokens

    group_test_heavy: set[tuple[int, int]] = set()
    group_test_only: set[tuple[int, int]] = set()

    per_item: list[tuple[tuple[int, int], str, _BatchLabelRequest]] = []
    id_to_group: dict[str, tuple[int, int, str]] = {}
    inherited: dict[tuple[int, int], _LLMChunkSystemGroupLabel] = {}

    def part_key(resolution: float) -> int:
        # Use milliresolution int key to avoid float dict pitfalls (0.25 -> 250).
        return int(round(float(resolution) * 1000.0))

    for part in partitions_in:
        res = float(part.get("resolution") or 0.0)
        membership_obj = part.get("membership")
        if not isinstance(membership_obj, list):
            continue
        membership = [int(x or 0) for x in membership_obj]
        if len(membership) != len(systems):
            continue

        members_by_gid: dict[int, list[dict[str, object]]] = {}
        for idx, gid in enumerate(membership):
            if gid <= 0:
                continue
            members_by_gid.setdefault(int(gid), []).append(systems[idx])

        # Aggregate top_files once per group for test-heuristic.
        for gid, member_systems in members_by_gid.items():
            file_counts: dict[str, int] = {}
            for s in member_systems:
                tf = s.get("top_files") or []
                if not isinstance(tf, list):
                    continue
                for row in tf:
                    if not isinstance(row, dict):
                        continue
                    path = row.get("path")
                    cnt = row.get("count")
                    if not isinstance(path, str) or not path:
                        continue
                    try:
                        cnt_i = int(cnt or 0)
                    except Exception:
                        cnt_i = 0
                    if cnt_i <= 0:
                        continue
                    file_counts[path] = int(file_counts.get(path, 0)) + int(cnt_i)
            top_files = [
                {"path": p, "count": c}
                for p, c in sorted(
                    file_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))
                )
            ]
            cluster_like = {"top_files": top_files}
            if _cluster_is_test_only(cluster_like):
                group_test_only.add((part_key(res), int(gid)))
            if _cluster_is_test_heavy(cluster_like):
                group_test_heavy.add((part_key(res), int(gid)))

            if len(member_systems) == 1:
                only = member_systems[0]
                cid = int(only.get("cluster_id") or 0)
                base_label = system_label_by_id.get(cid) or str(only.get("label") or "")
                lab = base_label.strip() or f"System #{cid}"
                if (part_key(res), int(gid)) in group_test_heavy:
                    lab = _normalize_tests_prefix(lab)
                inherited[(part_key(res), int(gid))] = _LLMChunkSystemGroupLabel(
                    resolution=float(res),
                    group_id=int(gid),
                    label=lab,
                    confidence=None,
                    prompt_path=None,
                    kind="inherited",
                )
                continue

            is_test_only = (part_key(res), int(gid)) in group_test_only
            is_test_heavy = (part_key(res), int(gid)) in group_test_heavy
            body_lines = _chunk_system_group_prompt_body(
                resolution=res,
                group_id=int(gid),
                member_systems=member_systems,
                system_label_by_id=system_label_by_id,
                is_test_only=is_test_only,
                is_test_heavy=is_test_heavy,
                max_member_systems=max_member_systems,
                max_files=max_files,
                max_symbols=max_symbols,
                prompt_mode=str(prompt_mode),
            )
            prompt = _chunk_system_group_prompt(
                resolution=res,
                group_id=int(gid),
                body_lines=body_lines,
            )
            res_key = _resolution_key(res)
            prompt_path = out_dir / f"llm_chunk_system_group_r{res_key}_g{int(gid)}.md"
            prompt_path.write_text(prompt, encoding="utf-8")

            item_id = f"chunk_system_group:r{res_key}:g{int(gid)}"
            req = _BatchLabelRequest(item_id=item_id, content="\n".join(body_lines))
            per_item.append(((part_key(res), int(gid)), prompt_path.name, req))
            id_to_group[item_id] = (part_key(res), int(gid), prompt_path.name)

    labelable = [req for (_k, _name, req) in per_item]
    if progress is not None and progress_task_id is not None:
        try:
            progress.update(
                progress_task_id,
                total=int(len(labelable) + len(inherited)),
                completed=0,
                info=("dry-run" if llm_dry_run else "starting"),
                speed="",
            )
        except Exception:
            pass

    max_items = int(llm_label_batch_max_items) if bool(llm_label_batching) else 1
    max_tokens = int(llm_label_batch_max_tokens) if bool(llm_label_batching) else 10**9

    results: dict[tuple[int, int], _LLMChunkSystemGroupLabel] = dict(inherited)
    llm_calls = 0

    provider = llm_manager.get_synthesis_provider() if llm_manager is not None else None
    effective_concurrency: int | None = None
    if llm_dry_run:
        if llm_label_concurrency is not None:
            try:
                effective_concurrency = max(1, int(llm_label_concurrency))
            except Exception:
                effective_concurrency = None
        pending = list(labelable)
        batch_idx = 0
        while pending:
            batch_idx += 1
            batch, pending = _take_label_batch(
                items=pending,
                estimate_tokens=estimate_tokens,
                max_tokens=max_tokens,
                max_items=int(max_items),
            )
            batch_prompt_path = out_dir / f"llm_chunk_system_group_batch_{batch_idx:04d}.md"
            batch_prompt = _render_label_batch_prompt(kind="chunk_system_groups", items=batch)
            batch_prompt_path.write_text(batch_prompt, encoding="utf-8")
            for item in batch:
                item_id = item.item_id
                res_k, gid, prompt_name = id_to_group[item_id]
                placeholder = f"SYSTEM_GROUP_r{item_id.split(':')[1][1:]}_g{gid}"
                if (res_k, gid) in group_test_heavy:
                    placeholder = _normalize_tests_prefix(placeholder)
                results[(res_k, gid)] = _LLMChunkSystemGroupLabel(
                    resolution=float(res_k) / 1000.0,
                    group_id=int(gid),
                    label=placeholder,
                    confidence=0.0,
                    prompt_path=prompt_name,
                    kind="llm",
                )
                if progress is not None and progress_task_id is not None:
                    try:
                        progress.update(
                            progress_task_id,
                            advance=1,
                            info=f"llm_chunk_system_group_batch {batch_idx} (n={len(batch)})",
                        )
                    except Exception:
                        pass
        pending = []
    else:
        if provider is None:
            raise RuntimeError("LLM manager not configured")

        concurrency_i = int(llm_label_concurrency) if llm_label_concurrency is not None else int(provider.get_synthesis_concurrency())
        concurrency_i = max(1, int(concurrency_i))
        effective_concurrency = int(concurrency_i)

        system_prompt = (
            "You label system groups for an operator-facing snapshot report. "
            "A label describes shared behavior/responsibility (what the group does), "
            "not an artifact or stage. Prefer a short natural noun phrase (2–7 words). "
            "If the group spans multiple subtopics, choose an umbrella label anchored on the dominant hub/subsystem, "
            "not a two-topic mashup. "
            f"Labels must be <= {int(LABEL_MAX_CHARS)} chars. "
            "Use 'Tests:' prefix when the group is overwhelmingly tests/fixtures."
        )

        def _on_batch_complete(
            batch: list[_BatchLabelRequest], outcome: _StructuredBatchOutcome, batch_idx: int
        ) -> None:
            parsed = outcome.parsed
            for item in batch:
                item_id = item.item_id
                res_k, gid, prompt_name = id_to_group[item_id]
                label, confidence = parsed[item_id]
                if (res_k, gid) in group_test_heavy:
                    label = _normalize_tests_prefix(label)
                results[(res_k, gid)] = _LLMChunkSystemGroupLabel(
                    resolution=float(res_k) / 1000.0,
                    group_id=int(gid),
                    label=str(label),
                    confidence=float(confidence),
                    prompt_path=prompt_name,
                    kind="llm",
                )
                if progress is not None and progress_task_id is not None:
                    try:
                        progress.update(
                            progress_task_id,
                            advance=1,
                            info=f"llm_chunk_system_group_batch {batch_idx} (n={len(batch)})",
                        )
                    except Exception:
                        pass

        llm_calls = await _run_structured_label_batches_parallel(
            provider=provider,
            kind="chunk_system_groups",
            items=list(labelable),
            out_dir=out_dir,
            batch_prefix="llm_chunk_system_group_batch",
            estimate_tokens=estimate_tokens,
            max_tokens=max_tokens,
            max_items=max_items,
            concurrency=concurrency_i,
            system=system_prompt,
            max_completion_tokens=400,
            progress=progress,
            progress_task_id=progress_task_id,
            on_batch_complete=_on_batch_complete,
        )

    # Write inherited progress advances (counted as labeled without LLM).
    if inherited and progress is not None and progress_task_id is not None:
        try:
            progress.update(progress_task_id, advance=int(len(inherited)))
        except Exception:
            pass

    if progress is not None and progress_task_id is not None:
        try:
            progress.update(progress_task_id, info="done")
        except Exception:
            pass

    # Build output payload aligned with input partitions.
    out_partitions: list[dict[str, object]] = []
    for part in partitions_in:
        res = float(part.get("resolution") or 0.0)
        res_k = part_key(res)
        group_count = int(part.get("group_count") or 0)
        labels: list[dict[str, object] | None] = [None] * max(0, group_count)
        for gid in range(1, 1 + max(0, group_count)):
            row = results.get((res_k, int(gid)))
            if row is None:
                continue
            labels[int(gid) - 1] = {
                "label": str(row.label),
                "confidence": row.confidence if row.confidence is not None else None,
                "kind": str(row.kind),
                "prompt_file": row.prompt_path,
            }
        out_partitions.append(
            {
                "resolution": float(res),
                "group_count": int(group_count),
                "labels": labels,
            }
        )

    payload: dict[str, object] = {
        "schema_version": _SYSTEM_GROUP_LABELS_SCHEMA_VERSION,
        "schema_revision": "2026-02-20",
        "source": "snapshot.chunk_systems.system_groups.json",
        "params": {
            "labeler": "llm",
            "seed": int(params_obj.get("seed") or 0) if isinstance(params_obj, dict) else 0,
            "resolutions": [
                float(p.get("resolution") or 0.0)
                for p in partitions_in
                if isinstance(p, dict)
            ],
            "batching": bool(llm_label_batching),
            "batch_max_items": int(llm_label_batch_max_items),
            "batch_max_tokens": int(llm_label_batch_max_tokens),
            "concurrency": int(effective_concurrency) if effective_concurrency is not None else None,
            "prompt_mode": str(prompt_mode),
            "caps": {
                "max_member_systems": int(max_member_systems),
                "max_files": int(max_files),
                "max_symbols": int(max_symbols),
            },
        },
        "partitions": out_partitions,
    }

    labeled_count = int(len(results))
    return payload, int(llm_calls), int(labeled_count)


async def _label_chunk_systems_llm(
    *,
    chunk_systems_payload: dict,
    db_file: Path,
    base_directory: Path,
    scope_roots: list[str],
    out_dir: Path,
    llm_manager: LLMManager | None,
    llm_dry_run: bool,
    item_by_chunk_id: dict[int, _SnapshotItem],
    chunk_hints_by_id: dict[int, _ChunkNameHints],
    llm_label_batching: bool,
    llm_label_batch_max_items: int,
    llm_label_batch_max_tokens: int,
    llm_label_concurrency: int | None = None,
    progress: Any | None = None,
    progress_task_id: Any | None = None,
) -> tuple[list[_LLMChunkSystemLabel], int]:
    clusters = chunk_systems_payload.get("clusters") or []
    if not isinstance(clusters, list):
        raise ValueError("chunk systems payload missing clusters list")

    estimate_tokens: Callable[[str], int]
    if llm_manager is not None:
        estimate_tokens = llm_manager.get_synthesis_provider().estimate_tokens
    else:
        estimate_tokens = _approx_tokens

    cryptic_cluster_ids: set[int] = set()
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        cluster_id = cluster.get("cluster_id")
        if not isinstance(cluster_id, int):
            continue
        size = int(cluster.get("size") or 0)
        if size < 2:
            continue
        if _cluster_is_cryptic(
            cluster=cluster,
            item_by_chunk_id=item_by_chunk_id,
            chunk_hints_by_id=chunk_hints_by_id,
        ):
            cryptic_cluster_ids.add(int(cluster_id))

    pass1_labels, pass1_calls = await _label_chunk_systems_llm_one_pass(
        clusters=clusters,
        out_dir=out_dir,
        llm_manager=llm_manager,
        llm_dry_run=llm_dry_run,
        item_by_chunk_id=item_by_chunk_id,
        chunk_hints_by_id=chunk_hints_by_id,
        llm_label_batching=llm_label_batching,
        llm_label_batch_max_items=llm_label_batch_max_items,
        llm_label_batch_max_tokens=llm_label_batch_max_tokens,
        llm_label_concurrency=llm_label_concurrency,
        db_file=db_file,
        base_directory=base_directory,
        scope_roots=scope_roots,
        deep_cluster_ids=cryptic_cluster_ids,
        prompt_suffix="",
        batch_prefix="llm_chunk_system_batch",
        estimate_tokens=estimate_tokens,
        progress=progress,
        progress_task_id=progress_task_id,
        reset_progress=True,
    )

    if llm_dry_run:
        return pass1_labels, pass1_calls

    relabel_ids: set[int] = set()
    for label in pass1_labels:
        if float(label.confidence) < float(LOW_CONF_THRESHOLD):
            relabel_ids.add(int(label.cluster_id))
            continue
        if int(label.cluster_id) in cryptic_cluster_ids and _label_is_generic(label.label):
            relabel_ids.add(int(label.cluster_id))

    if not relabel_ids:
        return pass1_labels, pass1_calls

    pass2_clusters: list[object] = []
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        cluster_id = cluster.get("cluster_id")
        if not isinstance(cluster_id, int):
            continue
        size = int(cluster.get("size") or 0)
        if size < 2:
            continue
        if int(cluster_id) in relabel_ids:
            pass2_clusters.append(cluster)

    if progress is not None and progress_task_id is not None:
        try:
            progress.update(
                progress_task_id,
                total=int(len(pass1_labels) + len(pass2_clusters)),
                info=f"pass2: relabel {len(pass2_clusters)}",
            )
        except Exception:
            pass

    pass2_labels, pass2_calls = await _label_chunk_systems_llm_one_pass(
        clusters=pass2_clusters,
        out_dir=out_dir,
        llm_manager=llm_manager,
        llm_dry_run=False,
        item_by_chunk_id=item_by_chunk_id,
        chunk_hints_by_id=chunk_hints_by_id,
        llm_label_batching=llm_label_batching,
        llm_label_batch_max_items=llm_label_batch_max_items,
        llm_label_batch_max_tokens=llm_label_batch_max_tokens,
        llm_label_concurrency=llm_label_concurrency,
        db_file=db_file,
        base_directory=base_directory,
        scope_roots=scope_roots,
        deep_cluster_ids={int(c.get("cluster_id")) for c in pass2_clusters if isinstance(c, dict) and isinstance(c.get("cluster_id"), int)},
        prompt_suffix="pass2",
        batch_prefix="llm_chunk_system_batch_pass2",
        estimate_tokens=estimate_tokens,
        progress=progress,
        progress_task_id=progress_task_id,
        reset_progress=False,
    )

    by_cluster: dict[int, _LLMChunkSystemLabel] = {int(l.cluster_id): l for l in pass1_labels}
    for l in pass2_labels:
        by_cluster[int(l.cluster_id)] = l

    combined = [by_cluster[k] for k in sorted(by_cluster)]
    return combined, int(pass1_calls + pass2_calls)


async def _label_chunk_systems_llm_one_pass(
    *,
    clusters: list[object],
    out_dir: Path,
    llm_manager: LLMManager | None,
    llm_dry_run: bool,
    item_by_chunk_id: dict[int, _SnapshotItem],
    chunk_hints_by_id: dict[int, _ChunkNameHints],
    llm_label_batching: bool,
    llm_label_batch_max_items: int,
    llm_label_batch_max_tokens: int,
    llm_label_concurrency: int | None = None,
    db_file: Path,
    base_directory: Path,
    scope_roots: list[str],
    deep_cluster_ids: set[int],
    prompt_suffix: str,
    batch_prefix: str,
    estimate_tokens: Callable[[str], int],
    progress: Any | None = None,
    progress_task_id: Any | None = None,
    reset_progress: bool = False,
) -> tuple[list[_LLMChunkSystemLabel], int]:
    prompt_suffix_norm = str(prompt_suffix or "").strip()
    suffix = f"_{prompt_suffix_norm}" if prompt_suffix_norm else ""

    labelable_clusters: list[dict] = []
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        cluster_id = cluster.get("cluster_id")
        if not isinstance(cluster_id, int):
            continue
        size = int(cluster.get("size") or 0)
        if size < 2:
            continue
        labelable_clusters.append(cluster)

    labelable_count = int(len(labelable_clusters))
    if reset_progress and progress is not None and progress_task_id is not None:
        try:
            progress.update(
                progress_task_id,
                total=labelable_count,
                completed=0,
                info=("dry-run" if llm_dry_run else "starting"),
                speed="",
            )
        except Exception:
            pass

    test_only_cluster_ids: set[int] = set()
    test_heavy_cluster_ids: set[int] = set()
    for cluster in labelable_clusters:
        cluster_id = cluster.get("cluster_id")
        if not isinstance(cluster_id, int):
            continue
        if _cluster_is_test_heavy(cluster):
            test_heavy_cluster_ids.add(int(cluster_id))
        if _cluster_is_test_only(cluster):
            test_only_cluster_ids.add(int(cluster_id))

    deep_chunk_ids: set[int] = set()
    if deep_cluster_ids:
        for cluster in labelable_clusters:
            cluster_id = cluster.get("cluster_id")
            if not isinstance(cluster_id, int) or int(cluster_id) not in deep_cluster_ids:
                continue
            deep_chunk_ids.update(
                _chunk_system_excerpt_candidate_ids(
                    cluster=cluster,
                    item_by_chunk_id=item_by_chunk_id,
                    chunk_hints_by_id=chunk_hints_by_id,
                    max_candidates=250,
                )
            )

    code_by_chunk_id: dict[int, str] = {}
    if deep_chunk_ids:
        try:
            code_by_chunk_id = _read_chunk_code_map(db_file=db_file, chunk_ids=deep_chunk_ids)
        except Exception as exc:
            logger.warning(
                "Chunk-system deep labeling: unable to read code from duckdb ({}). Falling back to disk excerpts.",
                exc,
            )
            code_by_chunk_id = {}

        missing_ids = [int(cid) for cid in deep_chunk_ids if int(cid) not in code_by_chunk_id]
        if missing_ids:
            disk_code = _read_chunk_code_map_from_disk(
                base_directory=base_directory,
                scope_roots=scope_roots,
                item_by_chunk_id=item_by_chunk_id,
                chunk_ids=missing_ids,
            )
            code_by_chunk_id.update(disk_code)

    enclosing_class_by_chunk_id: dict[int, str] = {}
    if deep_chunk_ids:
        scope_paths = _scope_root_paths(base_directory=base_directory, scope_roots=scope_roots)
        file_cache: dict[str, list[str] | None] = {}
        for chunk_id in deep_chunk_ids:
            item = item_by_chunk_id.get(int(chunk_id))
            if item is None:
                continue
            if not str(item.path).lower().endswith(".py"):
                continue
            hint = chunk_hints_by_id.get(int(chunk_id))
            symbol = hint.symbol if hint is not None else (item.symbol or "")
            if symbol not in {"__repr__", "__str__", "__init__"}:
                continue
            code = code_by_chunk_id.get(int(chunk_id), "")
            if isinstance(code, str) and "class " in code:
                continue
            abs_path = (base_directory / str(item.path)).resolve()
            try:
                abs_path.relative_to(base_directory)
            except ValueError:
                continue
            if not _is_under_any_scope_root(path=abs_path, scope_root_paths=scope_paths):
                continue
            cache_key = str(abs_path)
            if cache_key not in file_cache:
                file_cache[cache_key] = _read_text_lines_best_effort(abs_path=abs_path)
            file_lines = file_cache[cache_key]
            if not file_lines:
                continue
            cls = _infer_python_enclosing_class(
                file_lines=file_lines,
                start_line=int(item.start_line or 0),
            )
            if cls:
                enclosing_class_by_chunk_id[int(chunk_id)] = cls

    per_item: list[tuple[int, str, _BatchLabelRequest]] = []
    id_to_cluster: dict[str, tuple[int, str]] = {}

    for cluster in labelable_clusters:
        cluster_id = cluster.get("cluster_id")
        if not isinstance(cluster_id, int):
            continue
        is_deep = int(cluster_id) in deep_cluster_ids
        is_test_only = int(cluster_id) in test_only_cluster_ids
        if is_deep:
            body_lines = _chunk_system_prompt_body_deep(
                cluster=cluster,
                item_by_chunk_id=item_by_chunk_id,
                chunk_hints_by_id=chunk_hints_by_id,
                code_by_chunk_id=code_by_chunk_id,
                estimate_tokens=estimate_tokens,
                enclosing_class_by_chunk_id=enclosing_class_by_chunk_id,
                is_test_only=is_test_only,
            )
        else:
            body_lines = _chunk_system_prompt_body_rich(
                cluster=cluster,
                item_by_chunk_id=item_by_chunk_id,
                chunk_hints_by_id=chunk_hints_by_id,
                is_test_only=is_test_only,
            )

        prompt = _chunk_system_prompt(cluster_id=cluster_id, body_lines=body_lines)
        prompt_path = out_dir / f"llm_chunk_system_{cluster_id}{suffix}.md"
        prompt_path.write_text(prompt, encoding="utf-8")

        item_id = f"chunk_system:{cluster_id}"
        req = _BatchLabelRequest(item_id=item_id, content="\n".join(body_lines))
        per_item.append((cluster_id, prompt_path.name, req))
        id_to_cluster[item_id] = (cluster_id, prompt_path.name)

    max_items = int(llm_label_batch_max_items) if bool(llm_label_batching) else 1
    max_tokens = int(llm_label_batch_max_tokens) if bool(llm_label_batching) else 10**9

    provider = llm_manager.get_synthesis_provider() if llm_manager is not None else None
    llm_calls = 0
    labeled_by_cluster: dict[int, _LLMChunkSystemLabel] = {}

    pending_items = [req for (_cluster, _name, req) in per_item]
    if llm_dry_run:
        pending = list(pending_items)
        batch_idx = 0
        while pending:
            batch_idx += 1
            batch, pending = _take_label_batch(
                items=pending,
                estimate_tokens=estimate_tokens,
                max_tokens=max_tokens,
                max_items=int(max_items),
            )
            batch_prompt_path = out_dir / f"{batch_prefix}_{batch_idx:04d}.md"
            batch_prompt = _render_label_batch_prompt(kind="chunk_systems", items=batch)
            batch_prompt_path.write_text(batch_prompt, encoding="utf-8")
            for item in batch:
                item_id = item.item_id
                cluster_id, prompt_name = id_to_cluster[item_id]
                labeled_by_cluster[int(cluster_id)] = _LLMChunkSystemLabel(
                    cluster_id=cluster_id,
                    label=f"CHUNK_SYSTEM_{cluster_id}",
                    confidence=0.0,
                    prompt_path=prompt_name,
                )
                if progress is not None and progress_task_id is not None:
                    try:
                        progress.update(
                            progress_task_id,
                            advance=1,
                            info=f"{batch_prefix} {batch_idx} (n={len(batch)})",
                        )
                    except Exception:
                        pass
    else:
        if provider is None:
            raise RuntimeError("LLM manager not configured")

        concurrency_i = (
            int(llm_label_concurrency)
            if llm_label_concurrency is not None
            else int(provider.get_synthesis_concurrency())
        )
        concurrency_i = max(1, int(concurrency_i))

        system_prompt = (
            _CHUNK_SYSTEM_LABEL_SYSTEM_PROMPT
            + f" Labels must be <= {int(LABEL_MAX_CHARS)} chars."
        )

        def _on_batch_complete(
            batch: list[_BatchLabelRequest], outcome: _StructuredBatchOutcome, batch_idx: int
        ) -> None:
            parsed = outcome.parsed
            for item in batch:
                item_id = item.item_id
                cluster_id, prompt_name = id_to_cluster[item_id]
                label, confidence = parsed[item_id]
                if int(cluster_id) in test_heavy_cluster_ids:
                    label = _normalize_tests_prefix(label)
                labeled_by_cluster[int(cluster_id)] = _LLMChunkSystemLabel(
                    cluster_id=cluster_id,
                    label=label,
                    confidence=confidence,
                    prompt_path=prompt_name,
                )
                if progress is not None and progress_task_id is not None:
                    try:
                        progress.update(
                            progress_task_id,
                            advance=1,
                            info=f"{batch_prefix} {batch_idx} (n={len(batch)})",
                        )
                    except Exception:
                        pass

        llm_calls = await _run_structured_label_batches_parallel(
            provider=provider,
            kind="chunk_systems",
            items=list(pending_items),
            out_dir=out_dir,
            batch_prefix=str(batch_prefix),
            estimate_tokens=estimate_tokens,
            max_tokens=max_tokens,
            max_items=max_items,
            concurrency=concurrency_i,
            system=system_prompt,
            max_completion_tokens=400,
            progress=progress,
            progress_task_id=progress_task_id,
            on_batch_complete=_on_batch_complete,
        )

    if progress is not None and progress_task_id is not None:
        try:
            progress.update(progress_task_id, info="done")
        except Exception:
            pass

    results = [labeled_by_cluster[k] for k in sorted(labeled_by_cluster)]
    return results, int(llm_calls)


def _chunk_systems_payload_with_label_overlay(
    *, chunk_systems_payload: dict, chunk_system_labels: dict[int, str]
) -> dict:
    if not chunk_system_labels:
        return chunk_systems_payload
    out = dict(chunk_systems_payload)
    clusters = out.get("clusters") or []
    if not isinstance(clusters, list):
        return out

    new_clusters: list[object] = []
    for cluster in clusters:
        if not isinstance(cluster, dict):
            new_clusters.append(cluster)
            continue
        cluster_id = cluster.get("cluster_id")
        if not isinstance(cluster_id, int):
            new_clusters.append(cluster)
            continue
        label = chunk_system_labels.get(cluster_id)
        if not isinstance(label, str) or not label.strip():
            new_clusters.append(cluster)
            continue
        cluster_copy = dict(cluster)
        cluster_copy["label"] = label.strip()
        new_clusters.append(cluster_copy)

    out["clusters"] = new_clusters
    return out


def _read_scoped_items(
    *,
    db_file: Path,
    selector: _SnapshotSelector,
    scope_like_prefixes: list[str | None],
) -> tuple[list[tuple[_SnapshotItem, list[float]]], dict[str, int]]:
    table = f"embeddings_{selector.dims}"
    allowed_chunk_types = _code_chunk_type_values()

    conn = duckdb.connect(str(db_file), read_only=True)
    try:
        exists = conn.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = ?",
            [table],
        ).fetchone()
        if not exists:
            raise RuntimeError(
                "Embeddings table not found for requested dims. "
                f"Expected table: {table}"
            )

        scope_sql, scope_params = _duckdb_scope_clause(
            column="f.path", scope_like_prefixes=scope_like_prefixes
        )
        type_placeholders = ", ".join(["?" for _ in allowed_chunk_types])

        params: list[object] = [selector.provider, selector.model]
        params.extend(scope_params)
        params.extend(allowed_chunk_types)

        rows = conn.execute(
            f"""
            SELECT
                e.chunk_id,
                c.chunk_type,
                c.symbol,
                c.start_line,
                c.end_line,
                f.path,
                e.embedding
            FROM {table} e
            JOIN chunks c ON c.id = e.chunk_id
            JOIN files f ON f.id = c.file_id
            WHERE e.provider = ? AND e.model = ?
            {scope_sql}
              AND c.chunk_type IN ({type_placeholders})
            ORDER BY f.path, c.start_line, e.chunk_id
            """,
            params,
        ).fetchall()

        out: list[tuple[_SnapshotItem, list[float]]] = []
        for chunk_id, chunk_type, symbol, start_line, end_line, path, embedding in rows:
            item = _SnapshotItem(
                chunk_id=int(chunk_id),
                path=str(path),
                symbol=str(symbol) if symbol is not None else None,
                chunk_type=str(chunk_type),
                start_line=int(start_line or 0),
                end_line=int(end_line or 0),
            )
            vector = [float(x) for x in list(embedding or [])]
            out.append((item, vector))

        scope_where, count_params = _duckdb_scope_where(
            column="f.path", scope_like_prefixes=scope_like_prefixes
        )

        chunk_filter = "c.chunk_type IN (" + ", ".join(["?" for _ in allowed_chunk_types]) + ")"
        files_count_row = conn.execute(
            f"""
            SELECT COUNT(DISTINCT f.path)
            FROM chunks c
            JOIN files f ON f.id = c.file_id
            {scope_where}
              {('AND' if scope_where else 'WHERE')} {chunk_filter}
            """,
            count_params + allowed_chunk_types,
        ).fetchone()

        chunks_count_row = conn.execute(
            f"""
            SELECT COUNT(*)
            FROM chunks c
            JOIN files f ON f.id = c.file_id
            {scope_where}
              {('AND' if scope_where else 'WHERE')} {chunk_filter}
            """,
            count_params + allowed_chunk_types,
        ).fetchone()

        counts = {
            "files": int(files_count_row[0] if files_count_row else 0),
            "chunks": int(chunks_count_row[0] if chunks_count_row else 0),
            "chunks_with_embeddings": int(len(out)),
        }
        return out, counts
    finally:
        conn.close()


def _read_chunk_code_map(*, db_file: Path, chunk_ids: Iterable[int]) -> dict[int, str]:
    ids = sorted({int(x) for x in chunk_ids})
    if not ids:
        return {}

    placeholders = ", ".join(["?" for _ in ids])
    conn = duckdb.connect(str(db_file), read_only=True)
    try:
        rows = conn.execute(
            f"SELECT id, code FROM chunks WHERE id IN ({placeholders})",
            list(ids),
        ).fetchall()
    finally:
        conn.close()

    out: dict[int, str] = {}
    for chunk_id, code in rows:
        try:
            chunk_id_i = int(chunk_id)
        except Exception:
            continue
        if isinstance(code, str):
            out[chunk_id_i] = code
    return out


def _scope_root_paths(*, base_directory: Path, scope_roots: list[str]) -> list[Path]:
    if not scope_roots:
        return [base_directory.resolve()]
    resolved: list[Path] = []
    for root in scope_roots:
        r = str(root or ".").replace("\\", "/").strip("/") or "."
        abs_root = (base_directory / r).resolve()
        if abs_root not in resolved:
            resolved.append(abs_root)
    return resolved or [base_directory.resolve()]


def _is_under_any_scope_root(*, path: Path, scope_root_paths: list[Path]) -> bool:
    for root in scope_root_paths:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _infer_python_enclosing_class(*, file_lines: list[str], start_line: int) -> str | None:
    # start_line is 1-based.
    idx = max(0, int(start_line or 0) - 1)
    start = min(idx, max(0, len(file_lines) - 1))
    scanned = 0
    for j in range(start, -1, -1):
        scanned += 1
        if scanned > 250:
            break
        m = _CLASS_DEF_RE.match(file_lines[j])
        if m is not None:
            name = m.group(1)
            return name.strip() if name else None
    return None


def _read_text_lines_best_effort(*, abs_path: Path, max_bytes: int = 512 * 1024) -> list[str] | None:
    try:
        if not abs_path.exists() or not abs_path.is_file():
            return None
        if abs_path.stat().st_size > int(max_bytes):
            return None
        raw = abs_path.read_text(encoding="utf-8", errors="replace")
        return raw.splitlines()
    except Exception:
        return None


def _read_chunk_code_map_from_disk(
    *,
    base_directory: Path,
    scope_roots: list[str],
    item_by_chunk_id: dict[int, _SnapshotItem],
    chunk_ids: Iterable[int],
) -> dict[int, str]:
    ids = sorted({int(x) for x in chunk_ids})
    if not ids:
        return {}

    scope_root_paths = _scope_root_paths(base_directory=base_directory, scope_roots=scope_roots)

    out: dict[int, str] = {}
    file_cache: dict[str, list[str] | None] = {}

    for chunk_id in ids:
        item = item_by_chunk_id.get(int(chunk_id))
        if item is None:
            continue

        abs_path = (base_directory / str(item.path)).resolve()
        try:
            abs_path.relative_to(base_directory)
        except ValueError:
            continue
        if not _is_under_any_scope_root(path=abs_path, scope_root_paths=scope_root_paths):
            continue

        cache_key = str(abs_path)
        if cache_key not in file_cache:
            file_cache[cache_key] = _read_text_lines_best_effort(abs_path=abs_path)
        file_lines = file_cache[cache_key]
        if not file_lines:
            continue

        start_line = int(item.start_line or 0)
        end_line = int(item.end_line or 0)

        if start_line <= 0:
            continue

        start_idx = max(0, start_line - 1)
        if end_line <= 0 or end_line < start_line:
            end_idx = min(len(file_lines), start_idx + 200)
        else:
            end_idx = min(len(file_lines), end_line)

        snippet_lines = file_lines[start_idx:end_idx]
        code = "\n".join(snippet_lines).rstrip("\n")
        if code.strip():
            out[int(chunk_id)] = code

    return out


async def snapshot_command(args: argparse.Namespace, config: Config) -> None:
    """Execute the snapshot chunk-systems command."""
    out_dir = Path(getattr(args, "out_dir")).resolve()
    if out_dir.exists() and not out_dir.is_dir():
        logger.error(f"--out-dir must be a directory: {out_dir}")
        sys.exit(2)
    out_dir.mkdir(parents=True, exist_ok=True)

    want_chunk_systems = bool(getattr(args, "chunk_systems", False))
    if not want_chunk_systems:
        logger.error("Isolated snapshot branch supports only --chunk-systems mode")
        sys.exit(2)

    formatter = RichOutputFormatter(verbose=bool(getattr(args, "verbose", False)))
    tui_effective = _tui_is_effective(args=args, formatter=formatter)

    out_dir_mode = str(getattr(args, "out_dir_mode", "prompt") or "prompt").strip().lower()
    if out_dir_mode not in {"prompt", "reuse", "force"}:
        logger.error(
            f"Invalid --out-dir-mode: {out_dir_mode!r} (expected: prompt, reuse, force)"
        )
        sys.exit(2)

    refresh_run_metadata = bool(getattr(args, "refresh_run_metadata", False))
    out_dir_non_empty = _out_dir_is_non_empty(out_dir)
    if out_dir_non_empty and out_dir_mode in {"prompt", "reuse"}:
        errors = validate_chunk_systems_tui_assets(out_dir)
        if out_dir_mode == "reuse":
            if errors:
                joined = "\n".join(f"  - {e}" for e in errors)
                logger.error(
                    "Requested --out-dir-mode reuse but required snapshot artifacts are missing/invalid:\n"
                    f"{joined}"
                )
                sys.exit(1)
            if refresh_run_metadata:
                _refresh_snapshot_run_metadata_for_reuse(out_dir=out_dir, config=config)
            if tui_effective:
                run_chunk_systems_tui(out_dir)
            return

        # prompt
        if not _rich_prompt_supported(formatter=formatter):
            logger.error(
                "Non-interactive run with non-empty --out-dir and --out-dir-mode=prompt.\n"
                "Use --out-dir-mode force|reuse."
            )
            sys.exit(2)
        console = formatter.console
        assert console is not None

        if not errors:
            choice = _rich_select_out_dir_mode(
                question=f"--out-dir already contains snapshot artifacts:\n  {out_dir}\n\nSelect an action:",
                choices=[
                    ("Reuse existing run (open TUX now)", "reuse"),
                    ("Force new run", "force"),
                    ("Abort", "abort"),
                ],
                default_index=0,
                console=console,
            )
            if choice == "reuse":
                if refresh_run_metadata:
                    _refresh_snapshot_run_metadata_for_reuse(out_dir=out_dir, config=config)
                if tui_effective:
                    run_chunk_systems_tui(out_dir)
                return
            if choice == "abort":
                return
            # choice == "force" -> fall through to compute
        else:
            choice = _rich_select_out_dir_mode(
                question=(
                    "--out-dir is non-empty but required TUX assets are missing/invalid.\n\n"
                    f"{chr(10).join(f'- {e}' for e in errors)}\n\n"
                    "Select an action:"
                ),
                choices=[
                    ("Force new run", "force"),
                    ("Abort", "abort"),
                ],
                default_index=1,
                console=console,
            )
            if choice == "abort":
                return
            # choice == "force" -> fall through to compute

    llm_dry_run = bool(getattr(args, "llm_dry_run", False))
    system_group_labels_requested = bool(
        getattr(args, "chunk_systems_system_groups_labels", False)
    )
    system_group_labels_prompt_mode = str(
        getattr(args, "chunk_systems_system_groups_labels_prompt_mode", "systems_only")
        or "systems_only"
    ).strip().lower()
    if system_group_labels_prompt_mode not in {"systems_only", "full"}:
        logger.error(
            "Invalid --chunk-systems-system-groups-labels-prompt-mode: "
            f"{system_group_labels_prompt_mode!r} (expected: systems_only, full)"
        )
        sys.exit(2)
    labeler_mode = str(getattr(args, "labeler", "llm") or "llm").strip().lower()
    if labeler_mode not in {"llm", "heuristic"}:
        logger.error(f"Invalid --labeler: {labeler_mode!r} (expected: llm, heuristic)")
        sys.exit(2)

    explicit_labeler = any(
        arg == "--labeler" or str(arg).startswith("--labeler=") for arg in sys.argv
    )
    explicit_adjacency_mode = any(
        arg == "--chunk-systems-adjacency-mode"
        or str(arg).startswith("--chunk-systems-adjacency-mode=")
        for arg in sys.argv
    )

    if explicit_labeler is False and labeler_mode == "llm":
        labeler_mode = "heuristic"
        # Preserve --llm-dry-run for system-group labeling when explicitly requested.
        if not system_group_labels_requested:
            llm_dry_run = False

    md_labels_mode = str(getattr(args, "md_labels", "heuristic") or "heuristic").strip().lower()
    if md_labels_mode not in {"heuristic", "llm"}:
        logger.error(f"Invalid --md-labels: {md_labels_mode!r} (expected: heuristic, llm)")
        sys.exit(2)

    llm_label_batching = bool(getattr(args, "llm_label_batching", True))
    llm_label_batch_max_items = getattr(args, "llm_label_batch_max_items", 16)
    llm_label_batch_max_tokens = getattr(args, "llm_label_batch_max_tokens", 20000)
    llm_label_concurrency_arg = getattr(args, "llm_label_concurrency", None)

    try:
        llm_label_batch_max_items_i = int(llm_label_batch_max_items)
        llm_label_batch_max_tokens_i = int(llm_label_batch_max_tokens)
    except Exception:
        logger.error("--llm-label-batch-max-items and --llm-label-batch-max-tokens must be integers")
        sys.exit(2)

    if llm_label_batch_max_items_i <= 0 or llm_label_batch_max_tokens_i <= 0:
        logger.error("--llm-label-batch-max-items and --llm-label-batch-max-tokens must be positive")
        sys.exit(2)

    llm_label_concurrency_i: int | None = None
    if llm_label_concurrency_arg is not None:
        try:
            llm_label_concurrency_i = int(llm_label_concurrency_arg)
        except Exception:
            logger.error("--llm-label-concurrency must be an integer")
            sys.exit(2)
        if llm_label_concurrency_i <= 0:
            logger.error("--llm-label-concurrency must be positive")
            sys.exit(2)

    try:
        selector = _get_selector(args, config)
    except Exception as exc:
        logger.error(str(exc))
        sys.exit(2)

    matryoshka_dims_arg = getattr(args, "matryoshka_dims", None)
    matryoshka_dims: int | None = None
    if matryoshka_dims_arg is not None:
        try:
            matryoshka_dims = int(matryoshka_dims_arg)
        except Exception:
            logger.error("--matryoshka-dims must be an integer")
            sys.exit(2)
        if matryoshka_dims <= 0:
            logger.error("--matryoshka-dims must be positive")
            sys.exit(2)
        if matryoshka_dims > selector.dims:
            logger.error(
                "--matryoshka-dims must be <= --embedding-dims "
                f"({matryoshka_dims} > {selector.dims})"
            )
            sys.exit(2)

    needs_llm_calls = bool(
        (labeler_mode == "llm" and explicit_labeler) or system_group_labels_requested
    ) and not llm_dry_run

    if needs_llm_calls:
        if config.llm is None or not config.llm.is_provider_configured():
            logger.error(
                "LLM labeling requires `llm` configuration. Configure `llm` in "
                ".chunkhound.json, or pass --labeler heuristic, or pass --llm-dry-run."
            )
            sys.exit(1)

    utility_config: dict[str, Any] = {}
    synthesis_config: dict[str, Any] = {}
    if config.llm is not None and config.llm.is_provider_configured():
        utility_config, synthesis_config = config.llm.get_provider_configs()

    llm_manager = LLMManager(utility_config, synthesis_config) if needs_llm_calls else None
    llm_label_concurrency_effective: int | None = None
    if llm_label_concurrency_i is not None:
        llm_label_concurrency_effective = int(llm_label_concurrency_i)
    elif llm_manager is not None:
        try:
            llm_label_concurrency_effective = int(
                llm_manager.get_synthesis_provider().get_synthesis_concurrency()
            )
        except Exception:
            llm_label_concurrency_effective = None

    base_directory = (config.target_dir or Path.cwd()).resolve()
    scope_roots_arg = getattr(args, "scope_roots", None)
    pos_scope_path = Path(getattr(args, "scope_root"))

    try:
        if scope_roots_arg:
            pos_norm = _normalize_scope_root(pos_scope_path, base_directory=base_directory)
            if pos_norm != ".":
                logger.error(
                    "Provide either a positional scope_root or repeatable --scope-root, not both."
                )
                sys.exit(2)
            scope_roots = [
                _normalize_scope_root(Path(path), base_directory=base_directory)
                for path in scope_roots_arg
            ]
        else:
            scope_roots = [_normalize_scope_root(pos_scope_path, base_directory=base_directory)]
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(2)

    scope_roots = _collapse_scope_roots(scope_roots)
    scope_root_common = _common_scope_root(scope_roots)
    scope_like_prefixes = [_build_like_prefix(root) for root in scope_roots]

    db_file = config.database.get_db_path()
    if not db_file.exists():
        logger.error(f"Database file not found: {db_file}")
        sys.exit(1)

    with formatter.create_progress_display() as progress_manager:
        progress = progress_manager.get_progress_instance()
        started_total = time.perf_counter()

        db_task_id = progress.add_task(
            "Snapshot: DB fetch",
            total=1,
            info="",
            speed="",
        )

        started_fetch = time.perf_counter()
        try:
            if config.database.provider != "duckdb":
                raise RuntimeError(
                    "Snapshot chunk-systems currently supports duckdb only "
                    f"(got: {config.database.provider})"
                )
            item_vectors, counts = _read_scoped_items(
                db_file=db_file,
                selector=selector,
                scope_like_prefixes=scope_like_prefixes,
            )
        except Exception as exc:
            logger.error(str(exc))
            sys.exit(1)

        stage_db_fetch_ms = int(round((time.perf_counter() - started_fetch) * 1000.0))
        try:
            progress.update(
                db_task_id,
                completed=1,
                info=f"{counts.get('chunks_with_embeddings', 0)} embeddings",
            )
        except Exception:
            pass

        if not item_vectors:
            selectors: list[_AvailableSelector] = []
            try:
                conn = duckdb.connect(str(db_file), read_only=True)
                try:
                    selectors = _available_selectors_in_scope(
                        conn=conn,
                        scope_like_prefixes=scope_like_prefixes,
                        allowed_chunk_types=_code_chunk_type_values(),
                    )
                finally:
                    conn.close()
            except Exception:
                selectors = []

            logger.error(
                "No embeddings found for selected scope/selector.\n"
                f"  requested: provider={selector.provider} model={selector.model} dims={selector.dims}\n"
                f"  scope_root: {scope_roots}\n"
                "Available selectors in this scope:\n"
                f"{_format_available_selectors(selectors)}"
            )
            sys.exit(1)

        item_vectors_sorted = sorted(
            item_vectors,
            key=lambda pair: (
                pair[0].path,
                pair[0].chunk_type,
                pair[0].symbol or "",
                pair[0].chunk_id,
            ),
        )
        if matryoshka_dims is not None:
            limit = int(matryoshka_dims)
            item_vectors_sorted = [
                (item, list(vector[:limit])) for item, vector in item_vectors_sorted
            ]

        items_sorted = [item for item, _ in item_vectors_sorted]
        embeddings_sorted = [vector for _, vector in item_vectors_sorted]
        item_by_chunk_id = {int(item.chunk_id): item for item in items_sorted}

        scope_hash = _xxh3_64_hexdigest(
            "\n".join(
                [
                    _RUN_SCHEMA_VERSION,
                    _SCHEMA_REVISION,
                    str(base_directory),
                    "scope_roots",
                    *scope_roots,
                    f"{selector.provider}:{selector.model}:{selector.dims}",
                    f"labeler:{labeler_mode}",
                    f"labeler_dry_run:{bool(llm_dry_run)}",
                    f"system_group_labels:{bool(system_group_labels_requested)}",
                    f"system_group_labels_prompt_mode:{system_group_labels_prompt_mode}",
                    "chunk_types:code",
                ]
            )
        )

        scope_git_head_shas = _try_get_scope_git_head_shas(
            base_directory=base_directory,
            scope_roots=list(scope_roots),
        )
        unique_shas = {sha for sha in scope_git_head_shas.values() if sha}
        scope_git_head_sha = next(iter(unique_shas), None) if len(unique_shas) == 1 else None

        llm_synthesis_used: dict[str, object] | None = None
        if llm_manager is not None:
            try:
                synthesis_provider = llm_manager.get_synthesis_provider()
                llm_synthesis_used = {
                    "provider": str(synthesis_provider.name),
                    "model": str(synthesis_provider.model),
                }
                effort = synthesis_config.get("reasoning_effort")
                if isinstance(effort, str) and effort.strip():
                    llm_synthesis_used["reasoning_effort"] = effort.strip()
            except Exception:
                llm_synthesis_used = None
        llm_synthesis_configured = _sanitize_llm_synthesis_configured(config.llm)

        run_payload: dict[str, Any] = {
            "schema_version": _RUN_SCHEMA_VERSION,
            "schema_revision": _SCHEMA_REVISION,
            "base_directory": str(base_directory),
            "scope_root": list(scope_roots),
            "scope_hash": scope_hash,
            "scope_git_head_sha": scope_git_head_sha,
            "scope_git_head_shas": dict(scope_git_head_shas),
            "counts": counts,
            "embedding": {
                "provider": selector.provider,
                "model": selector.model,
                "dims": selector.dims,
                "matryoshka_dims": int(matryoshka_dims)
                if matryoshka_dims is not None
                else None,
            },
            "config_snapshot": {
                "chunk_types": "code",
                "labeler": (
                    {"mode": "llm", "dry_run": bool(llm_dry_run)}
                    if labeler_mode == "llm"
                    else {"mode": "heuristic"}
                ),
                "llm_synthesis_used": llm_synthesis_used,
                "llm_synthesis_configured": llm_synthesis_configured,
                "system_group_labels": {
                    "enabled": bool(system_group_labels_requested),
                    "dry_run": bool(llm_dry_run) if system_group_labels_requested else False,
                    "prompt_mode": str(system_group_labels_prompt_mode),
                },
                "md_labels": md_labels_mode,
                "llm_label_batching": bool(llm_label_batching),
                "llm_label_batch_max_items": int(llm_label_batch_max_items_i),
                "llm_label_batch_max_tokens": int(llm_label_batch_max_tokens_i),
                "llm_label_concurrency_requested": int(llm_label_concurrency_i)
                if llm_label_concurrency_i is not None
                else None,
                "llm_label_concurrency_effective": int(llm_label_concurrency_effective)
                if llm_label_concurrency_effective is not None
                else None,
            },
            "metrics": {
                "chunk_systems": None,
                "llm": None,
            },
            "stages": {
                "db_fetch_ms": stage_db_fetch_ms,
                "chunk_systems_ms": None,
                "llm_label_ms": None,
                "system_group_labels_ms": None,
                "total_ms": None,
                "chunk_systems_labeled": 0,
                "system_groups_labeled": 0,
                "llm_calls_chunk_system_batches": 0,
                "llm_calls_system_group_batches": 0,
                "llm_calls_total": 0,
                "llm_usage": None,
            },
        }
        _write_json(out_dir / "snapshot.run.json", run_payload)

        k = getattr(args, "chunk_systems_k", 30)
        tau = getattr(args, "chunk_systems_tau", 0.25)
        min_degree = getattr(args, "chunk_systems_min_degree", 0)
        fallback_tau = getattr(args, "chunk_systems_fallback_tau", None)
        fallback_path_mode = getattr(args, "chunk_systems_fallback_path_mode", "any")
        max_nodes = getattr(args, "chunk_systems_max_nodes", 20000)
        min_cluster_size = getattr(args, "chunk_systems_min_cluster_size", 1)
        partitioner = getattr(args, "chunk_systems_partitioner", "auto")
        leiden_resolution = getattr(args, "chunk_systems_leiden_resolution", 1.0)
        leiden_seed = getattr(args, "chunk_systems_leiden_seed", 0)
        resolutions_csv = getattr(args, "chunk_systems_leiden_resolutions", None)
        auto_selector_arg = getattr(args, "chunk_systems_auto_selector", "legacy")
        auto_seeds_csv = getattr(args, "chunk_systems_auto_stability_seeds", "0,1,2")
        auto_stability_min_arg = getattr(args, "chunk_systems_auto_stability_min", 0.90)
        auto_min_avg_size_arg = getattr(args, "chunk_systems_auto_min_avg_size", 10)
        auto_max_avg_size_arg = getattr(args, "chunk_systems_auto_max_avg_size", 80)
        auto_max_largest_frac_arg = getattr(args, "chunk_systems_auto_max_largest_frac", 0.20)
        auto_max_singleton_frac_arg = getattr(args, "chunk_systems_auto_max_singleton_frac", 0.02)
        auto_write_sweep = bool(getattr(args, "chunk_systems_auto_write_sweep", False))
        write_graph = bool(getattr(args, "chunk_systems_write_graph", False))
        write_adjacency = bool(getattr(args, "chunk_systems_write_adjacency", False))
        chunk_systems_viz = bool(getattr(args, "chunk_systems_viz", False))
        system_groups = bool(getattr(args, "chunk_systems_system_groups", False))
        system_groups_resolutions_csv = getattr(
            args, "chunk_systems_system_groups_resolutions", None
        )
        system_groups_seed = getattr(args, "chunk_systems_system_groups_seed", 0)
        system_groups_weight = getattr(args, "chunk_systems_system_groups_weight", "weight_sum")
        adjacency_evidence_k_arg = getattr(args, "chunk_systems_adjacency_evidence_k", 5)
        adjacency_max_neighbors_arg = getattr(args, "chunk_systems_adjacency_max_neighbors", 20)
        adjacency_mode_arg = getattr(args, "chunk_systems_adjacency_mode", "mutual")
        write_adjacency_effective = bool(write_adjacency or chunk_systems_viz or tui_effective)
        write_system_groups_effective = bool(
            system_groups
            or chunk_systems_viz
            or system_group_labels_requested
            or tui_effective
        )

        try:
            k_i = int(k)
        except Exception:
            logger.error("--chunk-systems-k must be an integer")
            sys.exit(2)
        if k_i <= 0:
            logger.error("--chunk-systems-k must be positive")
            sys.exit(2)

        try:
            tau_f = float(tau)
        except Exception:
            logger.error("--chunk-systems-tau must be a float")
            sys.exit(2)
        if not (0.0 <= tau_f <= 1.0):
            logger.error("--chunk-systems-tau must be between 0.0 and 1.0")
            sys.exit(2)

        try:
            min_degree_i = int(min_degree)
        except Exception:
            logger.error("--chunk-systems-min-degree must be an integer")
            sys.exit(2)
        if min_degree_i < 0:
            logger.error("--chunk-systems-min-degree must be >= 0")
            sys.exit(2)

        fallback_tau_f: float | None
        if fallback_tau is None:
            fallback_tau_f = None
        else:
            try:
                fallback_tau_f = float(fallback_tau)
            except Exception:
                logger.error("--chunk-systems-fallback-tau must be a float")
                sys.exit(2)
            if not (0.0 <= float(fallback_tau_f) <= 1.0):
                logger.error("--chunk-systems-fallback-tau must be between 0.0 and 1.0")
                sys.exit(2)

        fallback_path_mode_s = str(fallback_path_mode or "any").strip().lower()
        if fallback_path_mode_s not in {"any", "same_file", "same_dir"}:
            logger.error(
                "--chunk-systems-fallback-path-mode must be one of: any, same_file, same_dir"
            )
            sys.exit(2)

        try:
            max_nodes_i = int(max_nodes)
        except Exception:
            logger.error("--chunk-systems-max-nodes must be an integer")
            sys.exit(2)
        if max_nodes_i <= 0:
            logger.error("--chunk-systems-max-nodes must be positive")
            sys.exit(2)

        try:
            min_cluster_size_i = int(min_cluster_size)
        except Exception:
            logger.error("--chunk-systems-min-cluster-size must be an integer")
            sys.exit(2)
        if min_cluster_size_i < 1:
            logger.error("--chunk-systems-min-cluster-size must be >= 1")
            sys.exit(2)

        part = str(partitioner or "auto").strip().lower()
        if part not in {"auto", "cc", "leiden"}:
            logger.error("--chunk-systems-partitioner must be one of: auto, cc, leiden")
            sys.exit(2)

        auto_selector = str(auto_selector_arg or "legacy").strip().lower()
        if auto_selector not in {"legacy", "objective_stable"}:
            logger.error(
                "--chunk-systems-auto-selector must be one of: legacy, objective_stable"
            )
            sys.exit(2)
        if part != "auto" and auto_selector != "legacy":
            logger.error(
                "--chunk-systems-auto-selector is only supported when "
                "--chunk-systems-partitioner=auto"
            )
            sys.exit(2)
        if auto_write_sweep and auto_selector != "objective_stable":
            logger.error(
                "--chunk-systems-auto-write-sweep requires "
                "--chunk-systems-auto-selector=objective_stable"
            )
            sys.exit(2)

        adjacency_evidence_k_i = 5
        adjacency_max_neighbors_i = 20
        adjacency_mode = str(adjacency_mode_arg or "mutual").strip().lower()
        if adjacency_mode not in {"mutual", "directed"}:
            logger.error("--chunk-systems-adjacency-mode must be one of: mutual, directed")
            sys.exit(2)
        if tui_effective and not explicit_adjacency_mode:
            adjacency_mode = "directed"

        system_groups_resolutions: list[float] = []
        if write_system_groups_effective:
            raw = str(system_groups_resolutions_csv or "")
            pieces = [p.strip() for p in raw.split(",") if p.strip()]
            if not pieces:
                logger.error(
                    "--chunk-systems-system-groups-resolutions must be a non-empty CSV when enabled"
                )
                sys.exit(2)
            try:
                system_groups_resolutions = [float(p) for p in pieces]
            except Exception:
                logger.error("--chunk-systems-system-groups-resolutions must be a CSV of floats")
                sys.exit(2)
            if any(float(x) <= 0.0 for x in system_groups_resolutions):
                logger.error(
                    "--chunk-systems-system-groups-resolutions values must be positive"
                )
                sys.exit(2)
            try:
                system_groups_seed_i = int(system_groups_seed)
            except Exception:
                logger.error("--chunk-systems-system-groups-seed must be an integer")
                sys.exit(2)
            system_groups_seed = system_groups_seed_i

            system_groups_weight_s = str(system_groups_weight or "weight_sum").strip().lower()
            if system_groups_weight_s not in {"weight_sum", "weight_max"}:
                logger.error(
                    "--chunk-systems-system-groups-weight must be one of: weight_sum, weight_max"
                )
                sys.exit(2)
            system_groups_weight = system_groups_weight_s
        if write_adjacency:
            try:
                adjacency_evidence_k_i = int(adjacency_evidence_k_arg)
            except Exception:
                logger.error("--chunk-systems-adjacency-evidence-k must be an integer")
                sys.exit(2)
            if adjacency_evidence_k_i <= 0:
                logger.error("--chunk-systems-adjacency-evidence-k must be positive")
                sys.exit(2)
            try:
                adjacency_max_neighbors_i = int(adjacency_max_neighbors_arg)
            except Exception:
                logger.error("--chunk-systems-adjacency-max-neighbors must be an integer")
                sys.exit(2)
            if adjacency_max_neighbors_i <= 0:
                logger.error("--chunk-systems-adjacency-max-neighbors must be positive")
                sys.exit(2)

        try:
            leiden_resolution_f = float(leiden_resolution)
        except Exception:
            logger.error("--chunk-systems-leiden-resolution must be a float")
            sys.exit(2)
        if leiden_resolution_f <= 0.0:
            logger.error("--chunk-systems-leiden-resolution must be positive")
            sys.exit(2)

        try:
            leiden_seed_i = int(leiden_seed)
        except Exception:
            logger.error("--chunk-systems-leiden-seed must be an integer")
            sys.exit(2)

        resolutions_auto: list[float] | None = None
        if resolutions_csv is not None:
            raw = str(resolutions_csv or "")
            parts = [piece.strip() for piece in raw.split(",") if piece.strip()]
            if not parts:
                logger.error("--chunk-systems-leiden-resolutions must be a non-empty CSV")
                sys.exit(2)
            try:
                resolutions_auto = [float(piece) for piece in parts]
            except Exception:
                logger.error("--chunk-systems-leiden-resolutions must be a CSV of floats")
                sys.exit(2)
            if any(float(value) <= 0.0 for value in resolutions_auto):
                logger.error("--chunk-systems-leiden-resolutions values must be positive")
                sys.exit(2)

        auto_stability_seeds: list[int] | None = None
        auto_stability_min_f = 0.90
        auto_min_avg_size_i = 10
        auto_max_avg_size_i = 80
        auto_max_largest_frac_f = 0.20
        auto_max_singleton_frac_f = 0.02

        if auto_selector == "objective_stable":
            raw = str(auto_seeds_csv or "")
            pieces = [piece.strip() for piece in raw.split(",") if piece.strip()]
            if not pieces:
                logger.error("--chunk-systems-auto-stability-seeds must be a non-empty CSV")
                sys.exit(2)
            try:
                seeds_raw = [int(piece) for piece in pieces]
            except Exception:
                logger.error("--chunk-systems-auto-stability-seeds must be a CSV of integers")
                sys.exit(2)
            # De-dupe while preserving order.
            seen_seeds: set[int] = set()
            seeds: list[int] = []
            for s in seeds_raw:
                if int(s) in seen_seeds:
                    continue
                seen_seeds.add(int(s))
                seeds.append(int(s))
            auto_stability_seeds = seeds

            try:
                auto_stability_min_f = float(auto_stability_min_arg)
            except Exception:
                logger.error("--chunk-systems-auto-stability-min must be a float")
                sys.exit(2)
            if not (0.0 <= auto_stability_min_f <= 1.0):
                logger.error("--chunk-systems-auto-stability-min must be between 0.0 and 1.0")
                sys.exit(2)

            try:
                auto_min_avg_size_i = int(auto_min_avg_size_arg)
                auto_max_avg_size_i = int(auto_max_avg_size_arg)
            except Exception:
                logger.error("--chunk-systems-auto-min-avg-size and --chunk-systems-auto-max-avg-size must be integers")
                sys.exit(2)
            if auto_min_avg_size_i <= 0 or auto_max_avg_size_i <= 0:
                logger.error("--chunk-systems-auto-min-avg-size and --chunk-systems-auto-max-avg-size must be positive")
                sys.exit(2)
            if auto_min_avg_size_i > auto_max_avg_size_i:
                logger.error("--chunk-systems-auto-min-avg-size must be <= --chunk-systems-auto-max-avg-size")
                sys.exit(2)

            try:
                auto_max_largest_frac_f = float(auto_max_largest_frac_arg)
                auto_max_singleton_frac_f = float(auto_max_singleton_frac_arg)
            except Exception:
                logger.error("--chunk-systems-auto-max-largest-frac and --chunk-systems-auto-max-singleton-frac must be floats")
                sys.exit(2)
            if not (0.0 <= auto_max_largest_frac_f <= 1.0):
                logger.error("--chunk-systems-auto-max-largest-frac must be between 0.0 and 1.0")
                sys.exit(2)
            if not (0.0 <= auto_max_singleton_frac_f <= 1.0):
                logger.error("--chunk-systems-auto-max-singleton-frac must be between 0.0 and 1.0")
                sys.exit(2)

        if len(items_sorted) > max_nodes_i:
            logger.error(
                "Chunk systems safety cap exceeded.\n"
                f"  nodes: {len(items_sorted)}\n"
                f"  max_nodes: {max_nodes_i}\n"
                "Consider increasing --chunk-systems-max-nodes or narrowing scope_root."
            )
            sys.exit(2)

        if part == "cc":
            partition_total_steps = 1
        elif part == "leiden":
            partition_total_steps = 2
        else:
            auto_resolutions = (
                list(resolutions_auto)
                if resolutions_auto is not None
                else [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
            )
            if auto_selector == "objective_stable":
                seed_count = int(len(auto_stability_seeds or []))
                if seed_count <= 0:
                    seed_count = 1
                partition_total_steps = 1 + max(1, len(auto_resolutions)) * seed_count
            else:
                partition_total_steps = 1 + max(1, len(auto_resolutions))

        knn_total_nodes = int(len(items_sorted))
        knn_task_id = progress.add_task(
            "Snapshot: chunk systems kNN",
            total=max(1, knn_total_nodes),
            info=f"kNN 0/{knn_total_nodes}",
            speed="",
        )
        partition_task_id = progress.add_task(
            "Snapshot: chunk systems partition",
            total=max(1, int(partition_total_steps)),
            info="pending",
            speed="",
        )

        def _on_knn_progress(done: int, total: int) -> None:
            try:
                progress.update(
                    knn_task_id,
                    total=max(1, knn_total_nodes),
                    completed=min(int(done), knn_total_nodes),
                    info=f"kNN {int(done)}/{int(total)}",
                )
            except Exception:
                pass

        def _on_partition_progress(done: int, total: int, detail: str) -> None:
            try:
                progress.update(
                    partition_task_id,
                    total=max(1, int(total)),
                    completed=min(max(0, int(done)), max(1, int(total))),
                    info=detail,
                )
            except Exception:
                pass

        graph_context: ChunkSystemsGraphContext | None = None

        def _on_graph_context(ctx: ChunkSystemsGraphContext) -> None:
            nonlocal graph_context
            graph_context = ctx

        started_chunk = time.perf_counter()
        try:
            chunk_systems_payload = compute_chunk_systems(
                items=items_sorted,
                embeddings=embeddings_sorted,
                k=k_i,
                tau=tau_f,
                max_nodes=max_nodes_i,
                partitioner=part,
                leiden_resolution=leiden_resolution_f,
                leiden_seed=leiden_seed_i,
                leiden_resolutions_auto=resolutions_auto,
                capture_directed_arcs=bool(
                    (write_adjacency_effective and adjacency_mode == "directed")
                    or (write_system_groups_effective and adjacency_mode == "directed")
                ),
                leiden_auto_selector=str(auto_selector),
                leiden_auto_stability_seeds=auto_stability_seeds,
                leiden_auto_stability_min=float(auto_stability_min_f),
                leiden_auto_min_avg_size=int(auto_min_avg_size_i),
                leiden_auto_max_avg_size=int(auto_max_avg_size_i),
                leiden_auto_max_largest_frac_nodes=float(auto_max_largest_frac_f),
                leiden_auto_max_singleton_frac_nodes=float(auto_max_singleton_frac_f),
                record_partition_sweep=bool(auto_write_sweep),
                edge_min_degree=min_degree_i,
                edge_fallback_tau=fallback_tau_f,
                edge_fallback_path_mode=fallback_path_mode_s,
                knn_progress_callback=_on_knn_progress,
                partition_progress_callback=_on_partition_progress,
                graph_context_callback=_on_graph_context
                if (write_graph or write_adjacency_effective or chunk_systems_viz)
                else None,
            )
        except Exception as exc:
            logger.error(str(exc))
            sys.exit(1)

        chunk_systems_ms = int(round((time.perf_counter() - started_chunk) * 1000.0))
        try:
            progress.update(
                knn_task_id,
                completed=max(1, knn_total_nodes),
                info="done",
            )
            part_total = max(1, int(partition_total_steps))
            try:
                part_task = progress.tasks[partition_task_id]
                if getattr(part_task, "total", None) is not None:
                    part_total = max(1, int(part_task.total))
            except Exception:
                pass
            progress.update(
                partition_task_id,
                completed=part_total,
                info=f"done ({chunk_systems_ms}ms)",
            )
        except Exception:
            pass

        run_obj = chunk_systems_payload.get("run")
        if isinstance(run_obj, dict):
            run_obj["scope_hash"] = scope_hash
            run_obj["embedding"] = {
                "provider": selector.provider,
                "model": selector.model,
                "dims": selector.dims,
                "matryoshka_dims": int(matryoshka_dims)
                if matryoshka_dims is not None
                else None,
            }

        sweep_payload: object | None = None
        if isinstance(chunk_systems_payload, dict):
            sweep_payload = chunk_systems_payload.pop("_partition_sweep", None)
        if auto_write_sweep and sweep_payload is not None:
            _write_json(
                out_dir / "snapshot.chunk_systems.partition_sweep.json",
                sweep_payload,
            )

        # Raw compute payload is always written for audit/debugging.
        _write_json(out_dir / "snapshot.chunk_systems.json", chunk_systems_payload)

        report_payload = chunk_systems_payload
        if min_cluster_size_i > 1:
            pruned, dropped = build_chunk_systems_views(
                payload=chunk_systems_payload,
                min_cluster_size=min_cluster_size_i,
            )
            _write_json(out_dir / "snapshot.chunk_systems.pruned.json", pruned)
            _write_json(out_dir / "snapshot.chunk_systems.dropped.json", dropped)
            report_payload = pruned

            counts_view = pruned.get("counts_view") or {}
            if isinstance(counts_view, dict):
                run_payload["metrics"]["chunk_systems_view"] = dict(counts_view)

        if tui_effective and isinstance(report_payload, dict):
            try:
                _write_jsonl(
                    out_dir / "snapshot.chunk_systems.items.jsonl",
                    _chunk_systems_items_jsonl_rows(
                        report_payload=report_payload,
                        item_by_chunk_id=item_by_chunk_id,
                    ),
                )
            except Exception as exc:
                logger.warning(
                    "Unable to write snapshot.chunk_systems.items.jsonl ({}). Evidence browsing will be limited.",
                    exc,
                )

        if write_graph or write_adjacency_effective or chunk_systems_viz:
            adjacency_payload_for_viz: dict[str, object] | None = None
            system_metrics_for_viz: dict[int, dict[str, float | int | None]] | None = None
            system_groups_payload_for_viz: dict[str, object] | None = None

            if graph_context is None:
                logger.error("Chunk-systems graph context was not captured")
                sys.exit(1)
            kept_cluster_ids: set[int] = set()
            clusters_obj = report_payload.get("clusters") if isinstance(report_payload, dict) else None
            if isinstance(clusters_obj, list):
                for c in clusters_obj:
                    if not isinstance(c, dict):
                        continue
                    cid = c.get("cluster_id")
                    if cid is None:
                        continue
                    try:
                        kept_cluster_ids.add(int(cid))
                    except Exception:
                        continue

            if write_graph:
                _write_jsonl(
                    out_dir / "snapshot.chunk_systems.graph.nodes.jsonl",
                    iter_graph_nodes_jsonl(
                        items=items_sorted,
                        chunk_id_to_system=graph_context.chunk_id_to_system,
                        kept_cluster_ids=kept_cluster_ids,
                    ),
                )
                _write_jsonl(
                    out_dir / "snapshot.chunk_systems.graph.edges.jsonl",
                    iter_graph_edges_jsonl(
                        edges=graph_context.edges,
                        strict_edge_keys=graph_context.strict_edge_keys,
                        chunk_id_to_system=graph_context.chunk_id_to_system,
                        kept_cluster_ids=kept_cluster_ids,
                    ),
                )

            report_payload_dict: dict[str, object] = (
                report_payload if isinstance(report_payload, dict) else {}
            )

            adjacency_payload: dict[str, object] | None = None
            if write_adjacency_effective:
                if adjacency_mode == "mutual":
                    adjacency_payload = build_system_adjacency_json(
                        report_payload=report_payload_dict,
                        edges=graph_context.edges,
                        strict_edge_keys=graph_context.strict_edge_keys,
                        chunk_id_to_system=graph_context.chunk_id_to_system,
                        item_by_chunk_id=item_by_chunk_id,
                        kept_cluster_ids=kept_cluster_ids,
                        evidence_k=int(adjacency_evidence_k_i),
                        max_neighbors_per_system=int(adjacency_max_neighbors_i),
                        min_cluster_size=int(min_cluster_size_i),
                    )
                else:
                    if graph_context.directed_arcs is None:
                        logger.error("Chunk-systems directed arcs were not captured")
                        sys.exit(1)
                    adjacency_payload = build_system_adjacency_json_directed(
                        report_payload=report_payload_dict,
                        directed_arcs=graph_context.directed_arcs,
                        chunk_id_to_system=graph_context.chunk_id_to_system,
                        item_by_chunk_id=item_by_chunk_id,
                        kept_cluster_ids=kept_cluster_ids,
                        evidence_k=int(adjacency_evidence_k_i),
                        max_neighbors_per_system=int(adjacency_max_neighbors_i),
                        min_cluster_size=int(min_cluster_size_i),
                        k=int(k_i),
                        tau=float(tau_f),
                    )
                _write_json(
                    out_dir / "snapshot.chunk_systems.system_adjacency.json",
                    adjacency_payload,
                )

            if write_system_groups_effective:
                groups_payload: dict[str, object]
                if adjacency_mode == "mutual":
                    groups_payload = build_system_groups_json_from_chunk_edges(
                        report_payload=report_payload_dict,
                        edges=graph_context.edges,
                        chunk_id_to_system=graph_context.chunk_id_to_system,
                        kept_cluster_ids=kept_cluster_ids,
                        resolutions=list(system_groups_resolutions),
                        seed=int(system_groups_seed),
                        edge_weight=str(system_groups_weight),
                    )
                else:
                    if graph_context.directed_arcs is None:
                        logger.error("Chunk-systems directed arcs were not captured")
                        sys.exit(1)
                    groups_payload = build_system_groups_json_from_directed_arcs(
                        report_payload=report_payload_dict,
                        directed_arcs=graph_context.directed_arcs,
                        chunk_id_to_system=graph_context.chunk_id_to_system,
                        kept_cluster_ids=kept_cluster_ids,
                        resolutions=list(system_groups_resolutions),
                        seed=int(system_groups_seed),
                        edge_weight=str(system_groups_weight),
                    )
                _write_json(
                    out_dir / "snapshot.chunk_systems.system_groups.json",
                    groups_payload,
                )
                system_groups_payload_for_viz = groups_payload

            if chunk_systems_viz:
                if adjacency_payload is None:
                    logger.error("Chunk-systems adjacency payload was not built")
                    sys.exit(1)

                # Per-system metrics for operators. External metrics are based on the
                # (possibly truncated) adjacency links, while internal cohesion is
                # derived from the chunk-level graph.
                internal_weight_sum: dict[int, float] = {
                    int(cid): 0.0 for cid in kept_cluster_ids
                }
                internal_edge_count: dict[int, int] = {
                    int(cid): 0 for cid in kept_cluster_ids
                }
                external_weight_sum: dict[int, float] = {
                    int(cid): 0.0 for cid in kept_cluster_ids
                }
                external_weight_sum_out: dict[int, float] = {
                    int(cid): 0.0 for cid in kept_cluster_ids
                }
                external_weight_sum_in: dict[int, float] = {
                    int(cid): 0.0 for cid in kept_cluster_ids
                }
                out_degree: dict[int, int] = {int(cid): 0 for cid in kept_cluster_ids}
                in_degree: dict[int, int] = {int(cid): 0 for cid in kept_cluster_ids}
                degree_undirected: dict[int, int] = {
                    int(cid): 0 for cid in kept_cluster_ids
                }

                for a, b, w in graph_context.edges:
                    ca = graph_context.chunk_id_to_system.get(int(a))
                    cb = graph_context.chunk_id_to_system.get(int(b))
                    if ca is None or cb is None:
                        continue
                    ca_i = int(ca)
                    cb_i = int(cb)
                    if ca_i != cb_i:
                        continue
                    if ca_i not in kept_cluster_ids:
                        continue
                    internal_weight_sum[ca_i] = float(
                        internal_weight_sum.get(ca_i, 0.0)
                    ) + float(w)
                    internal_edge_count[ca_i] = int(internal_edge_count.get(ca_i, 0)) + 1

                links_obj = (
                    adjacency_payload.get("links")
                    if isinstance(adjacency_payload, dict)
                    else None
                )
                is_directed_adjacency = False
                if isinstance(adjacency_payload, dict):
                    is_directed_adjacency = bool(adjacency_payload.get("directed"))
                if isinstance(links_obj, list):
                    for link in links_obj:
                        if not isinstance(link, dict):
                            continue
                        if not is_directed_adjacency:
                            try:
                                u = int(link.get("a"))  # type: ignore[arg-type]
                                v = int(link.get("b"))  # type: ignore[arg-type]
                            except Exception:
                                continue
                            if u not in kept_cluster_ids or v not in kept_cluster_ids:
                                continue
                            wsum = float(link.get("weight_sum") or 0.0)
                            external_weight_sum[u] = (
                                float(external_weight_sum.get(u, 0.0)) + float(wsum)
                            )
                            external_weight_sum[v] = (
                                float(external_weight_sum.get(v, 0.0)) + float(wsum)
                            )
                            degree_undirected[u] = int(degree_undirected.get(u, 0)) + 1
                            degree_undirected[v] = int(degree_undirected.get(v, 0)) + 1
                        else:
                            try:
                                src = int(link.get("source"))  # type: ignore[arg-type]
                                dst = int(link.get("target"))  # type: ignore[arg-type]
                            except Exception:
                                continue
                            if src not in kept_cluster_ids or dst not in kept_cluster_ids:
                                continue
                            wsum = float(link.get("weight_sum") or 0.0)
                            external_weight_sum[src] = (
                                float(external_weight_sum.get(src, 0.0)) + float(wsum)
                            )
                            external_weight_sum[dst] = (
                                float(external_weight_sum.get(dst, 0.0)) + float(wsum)
                            )
                            external_weight_sum_out[src] = (
                                float(external_weight_sum_out.get(src, 0.0))
                                + float(wsum)
                            )
                            external_weight_sum_in[dst] = (
                                float(external_weight_sum_in.get(dst, 0.0))
                                + float(wsum)
                            )
                            out_degree[src] = int(out_degree.get(src, 0)) + 1
                            in_degree[dst] = int(in_degree.get(dst, 0)) + 1

                system_metrics: dict[int, dict[str, float | int | None]] = {}
                for cid in sorted(kept_cluster_ids):
                    iw = float(internal_weight_sum.get(int(cid), 0.0))
                    ew = float(external_weight_sum.get(int(cid), 0.0))
                    ratio = (ew / iw) if iw > 0.0 else None
                    od = int(out_degree.get(int(cid), 0))
                    idg = int(in_degree.get(int(cid), 0))
                    internal_edge_count_i = int(internal_edge_count.get(int(cid), 0))
                    external_weight_sum_out_f = float(
                        external_weight_sum_out.get(int(cid), 0.0)
                    )
                    external_weight_sum_in_f = float(
                        external_weight_sum_in.get(int(cid), 0.0)
                    )
                    ratio_f: float | None = float(ratio) if ratio is not None else None
                    system_metrics[int(cid)] = {
                        "internal_weight_sum": float(iw),
                        "internal_edge_count": int(internal_edge_count_i),
                        "external_weight_sum": float(ew),
                        "external_weight_sum_out": float(external_weight_sum_out_f),
                        "external_weight_sum_in": float(external_weight_sum_in_f),
                        "out_degree": int(od),
                        "in_degree": int(idg),
                        "degree": (
                            int(od + idg)
                            if is_directed_adjacency
                            else int(degree_undirected.get(int(cid), 0))
                        ),
                        "external_internal_ratio": ratio_f,
                    }

                adjacency_payload_for_viz = adjacency_payload
                system_metrics_for_viz = system_metrics

                _write_markdown(
                    out_dir / "snapshot.chunk_systems.viz.html",
                    render_chunk_systems_viz_html(
                        adjacency_payload=adjacency_payload,
                        system_metrics=system_metrics,
                        system_groups_payload=system_groups_payload_for_viz,
                    ),
                )

        _write_markdown(
            out_dir / "snapshot.chunk_systems.md",
            render_chunk_systems_markdown(report_payload),
        )

        counts_obj = chunk_systems_payload.get("counts") or {}
        if isinstance(counts_obj, dict):
            run_payload["metrics"]["chunk_systems"] = {
                "nodes": int(counts_obj.get("nodes") or 0),
                "mutual_edges": int(counts_obj.get("mutual_edges") or 0),
                "edges": int(counts_obj.get("edges") or 0),
                "fallback_edges": int(counts_obj.get("fallback_edges") or 0),
                "degree_zero_nodes": int(counts_obj.get("degree_zero_nodes") or 0),
                "clusters": int(counts_obj.get("clusters") or 0),
                "singletons": int(counts_obj.get("singletons") or 0),
                "largest_cluster": int(counts_obj.get("largest_cluster") or 0),
            }

        labels_payload: dict[str, Any] | None = None
        chunk_systems_labeled = 0
        llm_calls_chunk_system_batches = 0
        llm_label_ms = 0
        system_groups_labeled = 0
        llm_calls_system_group_batches = 0
        system_group_labels_ms = 0
        system_group_labels_payload_for_viz: dict[str, object] | None = None

        if labeler_mode == "llm":
            labels_payload = {
                "schema_version": _LABELS_SCHEMA_VERSION,
                "schema_revision": _SCHEMA_REVISION,
                "source_scope_hash": scope_hash,
                "mode": "llm",
                "dry_run": bool(llm_dry_run),
                "params": {
                    "batching": bool(llm_label_batching),
                    "batch_max_items": int(llm_label_batch_max_items_i),
                    "batch_max_tokens": int(llm_label_batch_max_tokens_i),
                    "concurrency_requested": int(llm_label_concurrency_i)
                    if llm_label_concurrency_i is not None
                    else None,
                    "concurrency_effective": int(llm_label_concurrency_effective)
                    if llm_label_concurrency_effective is not None
                    else None,
                },
                "llm": {
                    "utility_provider": utility_config.get("provider"),
                    "utility_model": utility_config.get("model"),
                    "synthesis_provider": synthesis_config.get("provider"),
                    "synthesis_model": synthesis_config.get("model"),
                },
                "chunk_systems": {},
            }

        should_label = bool(
            labeler_mode == "llm" and explicit_labeler and labels_payload is not None
        )
        if should_label:
            label_task_id = progress.add_task(
                "Snapshot: label chunk systems",
                total=0,
                info="",
                speed="",
            )

            started_llm = time.perf_counter()
            prompt_chunk_ids = _collect_chunk_system_prompt_chunk_ids(
                chunk_systems_payload=report_payload,
                item_by_chunk_id=item_by_chunk_id,
            )
            chunk_hints = _build_chunk_name_hints(
                chunk_ids=prompt_chunk_ids,
                item_by_chunk_id=item_by_chunk_id,
            )

            chunk_labels, llm_calls_chunk_system_batches = (
                await _label_chunk_systems_llm(
                    chunk_systems_payload=report_payload,
                    db_file=db_file,
                    base_directory=base_directory,
                    scope_roots=scope_roots,
                    out_dir=out_dir,
                    llm_manager=llm_manager,
                    llm_dry_run=llm_dry_run,
                    item_by_chunk_id=item_by_chunk_id,
                    chunk_hints_by_id=chunk_hints,
                    llm_label_batching=llm_label_batching,
                    llm_label_batch_max_items=llm_label_batch_max_items_i,
                    llm_label_batch_max_tokens=llm_label_batch_max_tokens_i,
                    llm_label_concurrency=llm_label_concurrency_effective,
                    progress=progress,
                    progress_task_id=label_task_id,
                )
            )

            llm_label_ms = int(round((time.perf_counter() - started_llm) * 1000.0))
            chunk_systems_labeled = int(len(chunk_labels))
            labels_payload["chunk_systems"] = {
                str(label.cluster_id): {
                    "label": label.label,
                    "confidence": label.confidence,
                    "prompt_file": label.prompt_path,
                }
                for label in chunk_labels
            }

        if system_group_labels_requested:
            if system_groups_payload_for_viz is None:
                logger.error("Chunk-systems system groups were not built (required for labels)")
                sys.exit(1)

            group_label_task_id = progress.add_task(
                "Snapshot: label system groups",
                total=0,
                info="",
                speed="",
            )

            # Prefer LLM-labeled system names when present, otherwise fall back to heuristic.
            system_label_by_id: dict[int, str] = {}
            sys_rows = system_groups_payload_for_viz.get("systems")
            if isinstance(sys_rows, list):
                for row in sys_rows:
                    if not isinstance(row, dict):
                        continue
                    cid = row.get("cluster_id")
                    if cid is None:
                        continue
                    try:
                        cid_i = int(cid)
                    except Exception:
                        continue
                    lab = row.get("label")
                    if isinstance(lab, str) and lab.strip():
                        system_label_by_id[cid_i] = lab.strip()

            if labels_payload is not None and isinstance(labels_payload.get("chunk_systems"), dict):
                for cid_s, row in labels_payload["chunk_systems"].items():
                    if not isinstance(row, dict):
                        continue
                    lab = row.get("label")
                    if not isinstance(lab, str) or not lab.strip():
                        continue
                    try:
                        cid_i = int(cid_s)
                    except Exception:
                        continue
                    system_label_by_id[cid_i] = lab.strip()

            started_groups_llm = time.perf_counter()
            system_group_labels_payload_for_viz, llm_calls_system_group_batches, system_groups_labeled = (
                await _label_chunk_system_groups_llm(
                    system_groups_payload=system_groups_payload_for_viz,
                    out_dir=out_dir,
                    llm_manager=llm_manager,
                    llm_dry_run=llm_dry_run,
                    llm_label_batching=llm_label_batching,
                    llm_label_batch_max_items=llm_label_batch_max_items_i,
                    llm_label_batch_max_tokens=llm_label_batch_max_tokens_i,
                    llm_label_concurrency=llm_label_concurrency_effective,
                    system_label_by_id=system_label_by_id,
                    prompt_mode=str(system_group_labels_prompt_mode),
                    progress=progress,
                    progress_task_id=group_label_task_id,
                )
            )
            system_group_labels_ms = int(
                round((time.perf_counter() - started_groups_llm) * 1000.0)
            )
            _write_json(
                out_dir / "snapshot.chunk_systems.system_group_labels.json",
                system_group_labels_payload_for_viz,
            )

        if (
            chunk_systems_viz
            and adjacency_payload_for_viz is not None
            and system_metrics_for_viz is not None
        ):
            viz_chunk_system_labels: dict[int, dict[str, object]] = {}
            if labels_payload is not None and isinstance(
                labels_payload.get("chunk_systems"),
                dict,
            ):
                for cluster_id, row in labels_payload["chunk_systems"].items():
                    if not isinstance(row, dict):
                        continue
                    label = row.get("label")
                    if not isinstance(label, str) or not label.strip():
                        continue
                    confidence = row.get("confidence")
                    conf_obj: object | None = None
                    if confidence is not None:
                        try:
                            conf_obj = float(confidence)  # type: ignore[arg-type]
                        except Exception:
                            conf_obj = None
                    try:
                        cluster_id_i = int(cluster_id)
                    except Exception:
                        continue
                    viz_chunk_system_labels[cluster_id_i] = {
                        "label": label.strip(),
                        "confidence": conf_obj,
                    }

            _write_markdown(
                out_dir / "snapshot.chunk_systems.viz.html",
                render_chunk_systems_viz_html(
                    adjacency_payload=adjacency_payload_for_viz,
                    system_metrics=system_metrics_for_viz,
                    chunk_system_labels=viz_chunk_system_labels,
                    system_groups_payload=system_groups_payload_for_viz,
                    system_group_labels_payload=system_group_labels_payload_for_viz,
                ),
            )

        if isinstance(report_payload, dict):
            overlay_labels: dict[int, str] = {}
            if (
                labels_payload is not None
                and md_labels_mode == "llm"
                and labeler_mode == "llm"
                and isinstance(labels_payload.get("chunk_systems"), dict)
            ):
                for cluster_id, row in labels_payload["chunk_systems"].items():
                    if not isinstance(row, dict):
                        continue
                    label = row.get("label")
                    if not isinstance(label, str) or not label.strip():
                        continue
                    try:
                        cluster_id_i = int(cluster_id)
                    except Exception:
                        continue
                    overlay_labels[cluster_id_i] = label.strip()

            payload_for_md = _chunk_systems_payload_with_label_overlay(
                chunk_systems_payload=report_payload,
                chunk_system_labels=overlay_labels,
            )
            _write_markdown(
                out_dir / "snapshot.chunk_systems.md",
                render_chunk_systems_markdown(payload_for_md),
            )

        if labels_payload is not None:
            _write_json(out_dir / "snapshot.labels.json", labels_payload)

        if llm_manager is not None:
            run_payload["metrics"]["llm"] = llm_manager.get_usage_stats()

        run_payload["stages"]["chunk_systems_ms"] = int(chunk_systems_ms)
        run_payload["stages"]["llm_label_ms"] = int(llm_label_ms)
        run_payload["stages"]["system_group_labels_ms"] = int(system_group_labels_ms)
        run_payload["stages"]["chunk_systems_labeled"] = int(chunk_systems_labeled)
        run_payload["stages"]["system_groups_labeled"] = int(system_groups_labeled)
        run_payload["stages"]["llm_calls_chunk_system_batches"] = int(
            llm_calls_chunk_system_batches
        )
        run_payload["stages"]["llm_calls_system_group_batches"] = int(
            llm_calls_system_group_batches
        )
        run_payload["stages"]["llm_calls_total"] = int(
            llm_calls_chunk_system_batches + llm_calls_system_group_batches
        )
        run_payload["stages"]["llm_usage"] = run_payload["metrics"]["llm"]
        run_payload["stages"]["total_ms"] = int(
            round((time.perf_counter() - started_total) * 1000.0)
        )

        _write_json(out_dir / "snapshot.run.json", run_payload)

    formatter.success(
        "Snapshot chunk-systems complete: "
        f"scope_root={scope_roots} common={scope_root_common} out_dir={out_dir}"
    )
    if tui_effective:
        run_chunk_systems_tui(out_dir)


__all__ = [
    "_BatchLabelRequest",
    "_pack_label_batches",
    "_parse_validate_batch_results",
    "snapshot_command",
]
