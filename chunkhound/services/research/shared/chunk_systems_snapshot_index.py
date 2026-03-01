"""Snapshot chunk-systems membership loader (consumer-only).

This module is intentionally pure: it reads JSON artifacts from disk and builds
an in-memory membership index (chunk_id -> system_id). It does not touch the DB
and does not import LLM-related modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

_POINTER_SCHEMA_VERSION = "snapshot.pointer.v1"
_CHUNK_SYSTEMS_SCHEMA_VERSION = "snapshot.chunk_systems.v1"

_POINTER_FILENAME = "snapshot.latest.json"
_CHUNK_SYSTEMS_FILENAME = "snapshot.chunk_systems.json"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        return path.is_relative_to(root)
    except AttributeError:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False


def _read_json_object(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object in {path.name}")
    return raw


def _resolve_run_dir(snapshot_dir: Path) -> Path:
    """Resolve snapshot root dir -> run dir via pointer, else treat as run dir."""
    pointer_path = snapshot_dir / _POINTER_FILENAME
    if not pointer_path.exists():
        return snapshot_dir

    try:
        pointer = _read_json_object(pointer_path)
    except Exception:
        return snapshot_dir

    if str(pointer.get("schema_version") or "") != _POINTER_SCHEMA_VERSION:
        return snapshot_dir

    run_dir_obj = pointer.get("run_dir")
    if not isinstance(run_dir_obj, str):
        return snapshot_dir

    run_dir_rel = Path(run_dir_obj)
    if run_dir_rel.is_absolute():
        return snapshot_dir

    root = snapshot_dir.resolve(strict=False)
    candidate = (snapshot_dir / run_dir_rel).resolve(strict=False)
    if not _is_relative_to(candidate, root):
        return snapshot_dir

    return candidate


def load_chunk_id_to_system_id(
    *, snapshot_dir: Path, explicit: bool
) -> dict[int, int] | None:
    """Load snapshot chunk-systems membership index (chunk_id -> system_id).

    Args:
        snapshot_dir: Directory containing snapshot artifacts. May be either:
            - a snapshot root directory containing snapshot.latest.json + runs/...
            - a run directory containing snapshot.chunk_systems.json directly
        explicit: True if snapshot_dir was explicitly set by the user. When True,
            loader failures will be logged as warnings. When False, failures are
            silent (callers should fall back).

    Returns:
        Mapping from chunk_id to system_id, or None on any failure.
    """
    try:
        snapshot_dir = Path(snapshot_dir)
        if not snapshot_dir.exists() or not snapshot_dir.is_dir():
            raise FileNotFoundError(f"Snapshot dir not found or not a directory: {snapshot_dir}")

        run_dir = _resolve_run_dir(snapshot_dir)
        chunk_systems_path = run_dir / _CHUNK_SYSTEMS_FILENAME
        if not chunk_systems_path.exists() or not chunk_systems_path.is_file():
            raise FileNotFoundError(f"Missing artifact: {chunk_systems_path}")

        payload = _read_json_object(chunk_systems_path)
        schema = str(payload.get("schema_version") or "")
        if schema != _CHUNK_SYSTEMS_SCHEMA_VERSION:
            raise ValueError(
                f"Invalid schema_version in {_CHUNK_SYSTEMS_FILENAME}: {schema!r} "
                f"(expected {_CHUNK_SYSTEMS_SCHEMA_VERSION!r})"
            )

        clusters_obj = payload.get("clusters")
        if not isinstance(clusters_obj, list):
            raise ValueError("Invalid chunk-systems payload: 'clusters' must be a list")

        chunk_id_to_system_id: dict[int, int] = {}
        for cluster in clusters_obj:
            if not isinstance(cluster, dict):
                raise ValueError("Invalid chunk-systems payload: cluster entry must be an object")

            try:
                system_id = int(cluster.get("cluster_id"))
            except Exception as exc:
                raise ValueError("Invalid chunk-systems payload: cluster_id must be int-like") from exc

            chunk_ids_obj = cluster.get("chunk_ids")
            if not isinstance(chunk_ids_obj, list):
                raise ValueError("Invalid chunk-systems payload: chunk_ids must be a list")

            for raw_chunk_id in chunk_ids_obj:
                try:
                    chunk_id = int(raw_chunk_id)
                except Exception as exc:
                    raise ValueError("Invalid chunk-systems payload: chunk_id must be int-like") from exc

                if chunk_id in chunk_id_to_system_id:
                    raise ValueError(f"Duplicate chunk_id across clusters: {chunk_id}")
                chunk_id_to_system_id[chunk_id] = system_id

        return chunk_id_to_system_id
    except Exception as exc:
        if explicit:
            logger.warning(f"Failed to load chunk-systems snapshot index from {snapshot_dir}: {exc}")
        return None

