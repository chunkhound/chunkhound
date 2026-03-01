import json

from chunkhound.services.research.shared.chunk_systems_snapshot_index import (
    load_chunk_id_to_system_id,
)


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_inverts_membership_from_run_dir(tmp_path):
    _write_json(
        tmp_path / "snapshot.chunk_systems.json",
        {
            "schema_version": "snapshot.chunk_systems.v1",
            "clusters": [
                {"cluster_id": 1, "chunk_ids": [10, 11]},
                {"cluster_id": 2, "chunk_ids": [20]},
            ],
        },
    )

    idx = load_chunk_id_to_system_id(snapshot_dir=tmp_path, explicit=True)
    assert idx == {10: 1, 11: 1, 20: 2}


def test_resolves_latest_pointer_to_run_dir(tmp_path):
    root = tmp_path / "root"
    run_dir = root / "runs" / "run1"
    run_dir.mkdir(parents=True)

    _write_json(
        run_dir / "snapshot.chunk_systems.json",
        {
            "schema_version": "snapshot.chunk_systems.v1",
            "clusters": [{"cluster_id": 7, "chunk_ids": [101]}],
        },
    )

    _write_json(
        root / "snapshot.latest.json",
        {
            "schema_version": "snapshot.pointer.v1",
            "run_id": "run1",
            "run_dir": "runs/run1",
            "updated_at": "2026-02-27T00:00:00Z",
        },
    )

    idx = load_chunk_id_to_system_id(snapshot_dir=root, explicit=True)
    assert idx == {101: 7}


def test_pointer_path_traversal_rejected(tmp_path):
    root = tmp_path / "root"
    root.mkdir()

    _write_json(
        root / "snapshot.latest.json",
        {"schema_version": "snapshot.pointer.v1", "run_dir": "../escape"},
    )

    idx = load_chunk_id_to_system_id(snapshot_dir=root, explicit=False)
    assert idx is None


def test_missing_payload_returns_none(tmp_path):
    idx = load_chunk_id_to_system_id(snapshot_dir=tmp_path, explicit=False)
    assert idx is None

