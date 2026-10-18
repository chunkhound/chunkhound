"""Contract test: the persisted-gate artifact stays env-free end to end.

``snapshot_for_persisted_gate`` suppresses nested BaseSettings env sources so
a standing ``CHUNKHOUND_*`` var cannot mask a hazard entering the global JSON
(chunkhound/core/config/remote/pipeline.py builds it fresh before
``persistence.backup_and_write``). Construction-time tests prove the snapshot
ignores env; this test proves the artifact itself does: serialize the
snapshot, write it to disk, reload it as the next run's global layer, and
assert the file's value survives — never the env value.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chunkhound.core.config.config import Config


def test_persisted_gate_snapshot_roundtrip_stays_env_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHUNKHOUND_RESEARCH_TARGET_TOKENS", "12345")

    # The hazard path: a rule sets a value the standing env var could mask.
    working: dict[str, object] = {"research": {"target_tokens": 20000}}

    post = Config.snapshot_for_persisted_gate(working)
    assert post.research.target_tokens == 20000, (
        "snapshot must carry the file value, not the env value"
    )

    # Round-trip the artifact exactly as persistence does: dump, write, reload
    # as the next run's global layer.
    persisted = tmp_path / ".chunkhound.global.json"
    persisted.write_text(post.model_dump_json(), encoding="utf-8")
    reloaded = json.loads(persisted.read_text(encoding="utf-8"))

    next_run = Config.snapshot_for_persisted_gate(reloaded)
    assert next_run.research.target_tokens == 20000, (
        "the persisted artifact must reload with the file value — an env value "
        "(12345) would prove a hazard leaked into the JSON on disk"
    )
