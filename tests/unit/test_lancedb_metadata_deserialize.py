"""Lance metadata must always surface as a dict to research consumers."""

from __future__ import annotations

import json

from chunkhound.providers.database.lancedb_provider import _deserialize_metadata
from chunkhound.services.research.shared.evidence_ledger.ledger import (
    EvidenceLedger,
    _coerce_chunk_metadata,
)


def test_deserialize_metadata_from_json_string() -> None:
    raw = '{"node_type": "assignment", "kind": "variable"}'
    meta = _deserialize_metadata(raw)
    assert isinstance(meta, dict)
    assert meta["kind"] == "variable"


def test_deserialize_metadata_already_dict() -> None:
    assert _deserialize_metadata({"kind": "function"}) == {"kind": "function"}


def test_deserialize_metadata_double_encoded_string() -> None:
    inner = {"kind": "class"}
    double = json.dumps(json.dumps(inner))
    assert _deserialize_metadata(double) == inner


def test_deserialize_metadata_nullish() -> None:
    assert _deserialize_metadata(None) == {}
    assert _deserialize_metadata("") == {}


def test_deserialize_metadata_invalid_and_non_object_json() -> None:
    assert _deserialize_metadata("not-json") == {}
    assert _deserialize_metadata("[1, 2]") == {}
    assert _deserialize_metadata("null") == {}


def test_format_chunks_by_file_id_preserves_embedding_and_deserializes() -> None:
    """Simulate as_model=False reformatting without a live Lance connection."""
    # Mirror the provider loop: raw Lance-like row → dict with dict metadata
    result = {
        "id": 1,
        "file_id": 2,
        "name": "foo",
        "content": "x = 1",
        "chunk_type": "function",
        "start_line": 1,
        "end_line": 2,
        "language": "python",
        "metadata": '{"kind": "function"}',
        "embedding": [0.1, 0.2, 0.3],
        "provider": "voyageai",
        "model": "voyage-code-3",
        "created_time": 1.0,
    }
    row = dict(result)
    row["metadata"] = _deserialize_metadata(row.get("metadata"))
    if "name" in row and "symbol" not in row:
        row["symbol"] = row.get("name") or ""
    if "content" in row and "code" not in row:
        row["code"] = row.get("content") or ""
    if "id" in row and "chunk_id" not in row:
        row["chunk_id"] = row["id"]

    assert isinstance(row["metadata"], dict)
    assert row["metadata"]["kind"] == "function"
    assert row["embedding"] == [0.1, 0.2, 0.3]
    assert row["symbol"] == "foo"
    assert row["code"] == "x = 1"
    assert row["chunk_id"] == 1
    # embedding_count style check used by get_file_stats
    emb = row.get("embedding")
    assert emb is not None and len(emb) > 0


def test_evidence_ledger_from_chunks_accepts_string_metadata() -> None:
    """Regression: import-resolution chunks had JSON-string metadata."""
    chunks = [
        {
            "file_path": "a.py",
            "metadata": '{"constants": [{"name": "FOO", "value": "1"}]}',
        },
        {
            "file_path": "b.py",
            "metadata": {"constants": [{"name": "BAR", "value": "2"}]},
        },
        {
            "file_path": "c.py",
            "metadata": "not-json",
        },
    ]
    ledger = EvidenceLedger.from_chunks(chunks)
    names = set(ledger.constants.keys())
    # keys may be namespaced; assert names appear
    assert any("FOO" in k for k in names)
    assert any("BAR" in k for k in names)
    assert ledger.constants_count == 2


def test_coerce_chunk_metadata() -> None:
    assert _coerce_chunk_metadata('{"kind": "x"}') == {"kind": "x"}
    assert _coerce_chunk_metadata(None) == {}
