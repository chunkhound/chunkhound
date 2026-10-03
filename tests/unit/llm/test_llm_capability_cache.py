"""Persistence contracts for learned LLM request capabilities."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

from chunkhound.providers.llm import capability_cache

ENV_VAR = "CHUNKHOUND_LLM_CAPABILITY_CACHE"
CACHE_NAME = "llm-capabilities.json"
KEY = "openrouter:poolside/laguna-xs-2.1"


@pytest.fixture
def overridden_cache_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    cache_path = tmp_path / "nested" / "capabilities.json"
    monkeypatch.setenv(ENV_VAR, str(cache_path))
    return cache_path


@pytest.mark.parametrize("state", ["accepted", "rejected"])
def test_capability_cache_env_override_round_trips_both_states(
    overridden_cache_path: Path,
    state: str,
) -> None:
    """The explicit file target persists learned state across instances."""
    capability_cache.LLMCapabilityStore().set(
        "openrouter", "poolside/laguna-xs-2.1", state
    )

    assert overridden_cache_path.is_file()
    assert (
        capability_cache.LLMCapabilityStore().get(
            "openrouter", "poolside/laguna-xs-2.1"
        )
        == state
    )


def test_capability_cache_write_uses_atomic_replace(
    monkeypatch: pytest.MonkeyPatch,
    overridden_cache_path: Path,
) -> None:
    """A write replaces the target atomically instead of editing it in place."""
    replace_calls: list[tuple[Path, Path]] = []
    original_replace = os.replace

    def recording_replace(source: str | Path, destination: str | Path) -> None:
        replace_calls.append((Path(source), Path(destination)))
        original_replace(source, destination)

    monkeypatch.setattr(os, "replace", recording_replace)

    capability_cache.LLMCapabilityStore().set("openrouter", "model", "accepted")

    assert len(replace_calls) == 1
    temporary_path, destination_path = replace_calls[0]
    assert destination_path == overridden_cache_path
    assert temporary_path != overridden_cache_path
    assert not temporary_path.exists()


@pytest.mark.parametrize(
    ("state", "accepted"),
    [("accepted", True), ("rejected", False)],
)
def test_capability_cache_json_schema_is_versioned_per_model(
    overridden_cache_path: Path,
    state: str,
    accepted: bool,
) -> None:
    """Disk entries use the locked provider:model value schema."""
    before = time.time()
    capability_cache.LLMCapabilityStore().set(
        "openrouter", "poolside/laguna-xs-2.1", state
    )
    after = time.time()

    payload = json.loads(overridden_cache_path.read_text(encoding="utf-8"))
    assert set(payload) == {KEY}
    assert payload[KEY]["accepted"] is accepted
    assert before <= payload[KEY]["ts"] <= after
    assert payload[KEY]["format_version"] == 1
    assert set(payload[KEY]) == {"accepted", "ts", "format_version"}


@pytest.mark.parametrize(
    ("platform", "cache_root_kind"),
    [
        pytest.param("linux", "xdg", id="xdg-cache-home"),
        pytest.param("linux", "home", id="unix-home-fallback"),
        pytest.param("win32", "local-appdata", id="windows-local-appdata"),
    ],
)
def test_capability_cache_default_path_follows_platform_conventions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    platform: str,
    cache_root_kind: str,
) -> None:
    """Without an override, the user-scoped cache follows ChunkHound precedent."""
    home = tmp_path / "home"
    xdg = tmp_path / "xdg"
    local_appdata = tmp_path / "local-appdata"
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    if cache_root_kind == "xdg":
        monkeypatch.setenv("XDG_CACHE_HOME", str(xdg))
        expected = xdg / "chunkhound" / CACHE_NAME
    elif cache_root_kind == "local-appdata":
        monkeypatch.setenv("LOCALAPPDATA", str(local_appdata))
        expected = local_appdata / "ChunkHound" / CACHE_NAME
    else:
        expected = home / ".cache" / "chunkhound" / CACHE_NAME

    capability_cache.LLMCapabilityStore().set("openrouter", "model", "accepted")

    assert expected.is_file()


def test_capability_cache_write_failure_is_tolerated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv(ENV_VAR, str(blocker / "cap.json"))
    store = capability_cache.LLMCapabilityStore()

    store.set("openrouter", "model", "accepted")

    assert store.get("openrouter", "model") == "unknown"


def test_capability_cache_cleans_temp_file_when_replace_fails(
    monkeypatch: pytest.MonkeyPatch,
    overridden_cache_path: Path,
) -> None:
    def failing_replace(source: str | Path, destination: str | Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", failing_replace)

    capability_cache.LLMCapabilityStore().set("openrouter", "model", "accepted")

    assert list(overridden_cache_path.parent.glob("*.tmp")) == []
    assert not overridden_cache_path.exists()


def test_capability_cache_merges_keys_across_writes(
    overridden_cache_path: Path,
) -> None:
    first_store = capability_cache.LLMCapabilityStore()
    second_store = capability_cache.LLMCapabilityStore()

    first_store.set("openrouter", "first-model", "accepted")
    second_store.set("openrouter", "second-model", "rejected")

    payload = json.loads(overridden_cache_path.read_text(encoding="utf-8"))
    assert set(payload) == {"openrouter:first-model", "openrouter:second-model"}


def test_cache_rejects_wrong_typed_or_versioned_entries_and_keeps_valid_siblings(
    overridden_cache_path: Path,
) -> None:
    overridden_cache_path.parent.mkdir(parents=True)
    overridden_cache_path.write_text(
        json.dumps(
            {
                "openrouter:bool-version": {
                    "accepted": True,
                    "ts": time.time(),
                    "format_version": True,
                },
                "openrouter:future-version": {
                    "accepted": True,
                    "ts": time.time(),
                    "format_version": 2,
                },
                "openrouter:non-bool-accepted": {
                    "accepted": 1,
                    "ts": time.time(),
                    "format_version": 1,
                },
                "openrouter:string-ts": {
                    "accepted": True,
                    "ts": "now",
                    "format_version": 1,
                },
                "openrouter:bool-ts": {
                    "accepted": True,
                    "ts": True,
                    "format_version": 1,
                },
                "openrouter:valid": {
                    "accepted": False,
                    "ts": time.time(),
                    "format_version": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    store = capability_cache.LLMCapabilityStore()

    for model in (
        "bool-version",
        "future-version",
        "non-bool-accepted",
        "string-ts",
        "bool-ts",
    ):
        assert store.get("openrouter", model) == "unknown"
    assert store.get("openrouter", "valid") == "rejected"


def test_capability_cache_isolates_same_model_across_providers(
    overridden_cache_path: Path,
) -> None:
    """One model name under two providers keeps independent decisions."""
    store = capability_cache.LLMCapabilityStore()

    store.set("openrouter", "shared-model", "accepted")
    store.set("other", "shared-model", "rejected")

    assert store.get("openrouter", "shared-model") == "accepted"
    assert store.get("other", "shared-model") == "rejected"


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param(
            {"accepted": True, "ts": time.time()},
            id="missing-format-version",
        ),
        pytest.param(
            {"accepted": True, "ts": time.time(), "format_version": None},
            id="null-format-version",
        ),
        pytest.param(
            {"accepted": True, "ts": time.time(), "format_version": "1"},
            id="string-format-version",
        ),
        pytest.param(
            {"accepted": True, "ts": time.time(), "format_version": 1.0},
            id="float-format-version",
        ),
    ],
)
def test_capability_cache_rejects_unusable_format_versions(
    overridden_cache_path: Path,
    entry: dict[str, object],
) -> None:
    """Entries without an integer format version never supply a decision."""
    overridden_cache_path.parent.mkdir(parents=True)
    overridden_cache_path.write_text(json.dumps({KEY: entry}), encoding="utf-8")

    assert (
        capability_cache.LLMCapabilityStore().get(
            "openrouter", "poolside/laguna-xs-2.1"
        )
        == "unknown"
    )


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param(
            {"ts": time.time(), "format_version": 1},
            id="missing-accepted",
        ),
        pytest.param(
            {"accepted": None, "ts": time.time(), "format_version": 1},
            id="null-accepted",
        ),
        pytest.param(
            {"accepted": "true", "ts": time.time(), "format_version": 1},
            id="string-accepted",
        ),
        pytest.param(
            {"accepted": True, "format_version": 1},
            id="missing-ts",
        ),
        pytest.param(
            {"accepted": True, "ts": None, "format_version": 1},
            id="null-ts",
        ),
    ],
)
def test_capability_cache_rejects_missing_or_null_required_fields(
    overridden_cache_path: Path,
    entry: dict[str, object],
) -> None:
    """Missing or non-conforming required fields read as unknown without raising."""
    overridden_cache_path.parent.mkdir(parents=True)
    overridden_cache_path.write_text(json.dumps({KEY: entry}), encoding="utf-8")

    assert (
        capability_cache.LLMCapabilityStore().get(
            "openrouter", "poolside/laguna-xs-2.1"
        )
        == "unknown"
    )


@pytest.mark.parametrize("accepted", [True, False], ids=["accepted", "rejected"])
@pytest.mark.parametrize(
    ("seconds_inside_boundary", "expected"),
    [(0, "unknown"), (1, None)],
    ids=["at-boundary", "inside-boundary"],
)
def test_capability_cache_ttl_boundary(
    monkeypatch: pytest.MonkeyPatch,
    overridden_cache_path: Path,
    accepted: bool,
    seconds_inside_boundary: int,
    expected: str | None,
) -> None:
    fixed = 2_000_000_000.0
    monkeypatch.setattr(capability_cache.time, "time", lambda: fixed)
    state = "accepted" if accepted else "rejected"
    overridden_cache_path.parent.mkdir(parents=True)
    overridden_cache_path.write_text(
        json.dumps(
            {
                KEY: {
                    "accepted": accepted,
                    "ts": fixed
                    - capability_cache.state_ttl_seconds(state)
                    + seconds_inside_boundary,
                    "format_version": 1,
                }
            }
        ),
        encoding="utf-8",
    )

    expected_state = expected or state
    assert (
        capability_cache.LLMCapabilityStore().get(
            "openrouter", "poolside/laguna-xs-2.1"
        )
        == expected_state
    )


@pytest.mark.parametrize(
    "timestamp",
    [float("inf"), float("nan")],
    ids=["infinite", "not-a-number"],
)
def test_capability_cache_rejects_non_finite_timestamps(
    overridden_cache_path: Path,
    timestamp: float,
) -> None:
    """Non-finite timestamps never yield a live decision."""
    overridden_cache_path.parent.mkdir(parents=True)
    overridden_cache_path.write_text(
        json.dumps(
            {
                KEY: {
                    "accepted": True,
                    "ts": timestamp,
                    "format_version": 1,
                }
            }
        ),
        encoding="utf-8",
    )

    assert (
        capability_cache.LLMCapabilityStore().get(
            "openrouter", "poolside/laguna-xs-2.1"
        )
        == "unknown"
    )


@pytest.mark.parametrize(
    "payload",
    [[], None, "text", {}, {"openrouter:model": "not-a-dict"}],
)
def test_cache_reads_unknown_for_unexpected_top_level_shapes(
    overridden_cache_path: Path,
    payload: object,
) -> None:
    overridden_cache_path.parent.mkdir(parents=True)
    overridden_cache_path.write_text(json.dumps(payload), encoding="utf-8")

    assert capability_cache.LLMCapabilityStore().get("openrouter", "model") == "unknown"


def test_capability_cache_set_rejects_unknown_state(
    overridden_cache_path: Path,
) -> None:
    with pytest.raises(ValueError, match="accepted.*rejected"):
        capability_cache.LLMCapabilityStore().set("openrouter", "model", "unknown")

    assert not overridden_cache_path.exists()


def test_capability_cache_missing_or_corrupt_file_reads_unknown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Unavailable persistence never prevents an LLM request."""
    cache_path = tmp_path / "capabilities.json"
    monkeypatch.setenv(ENV_VAR, str(cache_path))

    assert capability_cache.LLMCapabilityStore().get("openrouter", "model") == "unknown"

    cache_path.write_text("{not valid json", encoding="utf-8")
    assert capability_cache.LLMCapabilityStore().get("openrouter", "model") == "unknown"


def test_capability_cache_expires_rejected_after_one_day_and_accepted_after_30_days(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Rejected leases expire after one day while accepted leases last 30 days."""
    fixed = 2_000_000_000.0
    monkeypatch.setattr(capability_cache.time, "time", lambda: fixed)
    cache_path = tmp_path / "capabilities.json"
    monkeypatch.setenv(ENV_VAR, str(cache_path))

    def state_for(accepted: bool, age: float) -> str:
        cache_path.write_text(
            json.dumps(
                {
                    KEY: {
                        "accepted": accepted,
                        "ts": fixed - age,
                        "format_version": 1,
                    }
                }
            ),
            encoding="utf-8",
        )
        return capability_cache.LLMCapabilityStore().get(
            "openrouter", "poolside/laguna-xs-2.1"
        )

    assert state_for(False, 24 * 60 * 60 + 1) == "unknown"
    assert state_for(False, 23 * 60 * 60) == "rejected"
    assert state_for(True, 30 * 24 * 60 * 60 + 1) == "unknown"


@pytest.mark.parametrize(
    "corrupt_content",
    [
        pytest.param(b"\xff\xfe\x00", id="invalid-utf-8"),
        pytest.param(("[" * 2000 + "]" * 2000).encode(), id="deeply-nested-json"),
    ],
)
def test_capability_cache_tolerates_unreadable_cache_files(
    overridden_cache_path: Path,
    corrupt_content: bytes,
) -> None:
    """Unreadable persisted bytes never raise and are replaced on the next write."""
    overridden_cache_path.parent.mkdir(parents=True)
    overridden_cache_path.write_bytes(corrupt_content)
    store = capability_cache.LLMCapabilityStore()

    assert store.get("openrouter", "model") == "unknown"

    store.set("openrouter", "model", "accepted")

    payload = json.loads(overridden_cache_path.read_text(encoding="utf-8"))
    assert set(payload) == {"openrouter:model"}
    assert store.get("openrouter", "model") == "accepted"


def test_capability_cache_set_prunes_malformed_and_expired_entries(
    overridden_cache_path: Path,
) -> None:
    """A write uses the decision's TTL while preserving newer formats."""
    now = time.time()
    old_but_accepted = now - 2 * 24 * 60 * 60
    overridden_cache_path.parent.mkdir(parents=True)
    overridden_cache_path.write_text(
        json.dumps(
            {
                "openrouter:expired": {
                    "accepted": True,
                    "ts": now - (30 * 24 * 60 * 60) - 1,
                    "format_version": 1,
                },
                "openrouter:expired-rejected": {
                    "accepted": False,
                    "ts": old_but_accepted,
                    "format_version": 1,
                },
                "openrouter:old-accepted": {
                    "accepted": True,
                    "ts": old_but_accepted,
                    "format_version": 1,
                },
                "openrouter:old-version": {
                    "accepted": True,
                    "ts": now,
                    "format_version": 0,
                },
                "openrouter:junk": "not-a-dict",
                "openrouter:future-version": {
                    "accepted": True,
                    "ts": now,
                    "format_version": 2,
                },
                "openrouter:valid": {
                    "accepted": False,
                    "ts": now,
                    "format_version": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    store = capability_cache.LLMCapabilityStore()

    store.set("openrouter", "new", "accepted")

    payload = json.loads(overridden_cache_path.read_text(encoding="utf-8"))
    assert set(payload) == {
        "openrouter:old-accepted",
        "openrouter:future-version",
        "openrouter:valid",
        "openrouter:new",
    }
    assert store.get("openrouter", "old-accepted") == "accepted"
    assert store.get("openrouter", "valid") == "rejected"
    assert store.get("openrouter", "new") == "accepted"
    assert store.get("openrouter", "future-version") == "unknown"


@pytest.mark.parametrize("state", ["accepted", "rejected"])
def test_capability_cache_set_returns_and_reads_persisted_deadline(
    monkeypatch: pytest.MonkeyPatch,
    overridden_cache_path: Path,
    state: str,
) -> None:
    """Writes and reads agree on each decision's persisted deadline."""
    fixed = 2_000_000_000.0
    monkeypatch.setattr(capability_cache.time, "time", lambda: fixed)
    store = capability_cache.LLMCapabilityStore()

    deadline = store.set("openrouter", "model", state)

    assert deadline == fixed + capability_cache.state_ttl_seconds(state)
    assert store.get_with_expiry("openrouter", "model") == (state, deadline)

    overridden_cache_path.write_text(
        json.dumps(
            {
                "openrouter:dead": {
                    "accepted": state == "accepted",
                    "ts": fixed - capability_cache.state_ttl_seconds(state),
                    "format_version": 1,
                }
            }
        ),
        encoding="utf-8",
    )
    assert store.get_with_expiry("openrouter", "dead") == ("unknown", None)


def test_capability_cache_tolerates_overflowing_sibling_timestamps(
    overridden_cache_path: Path,
) -> None:
    """An astronomically large sibling ts never breaks reads or writes."""
    now = time.time()
    overridden_cache_path.parent.mkdir(parents=True)
    overridden_cache_path.write_text(
        json.dumps(
            {
                "openrouter:huge": {
                    "accepted": True,
                    "ts": 10**1000,
                    "format_version": 1,
                },
                "openrouter:valid": {
                    "accepted": True,
                    "ts": now,
                    "format_version": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    store = capability_cache.LLMCapabilityStore()

    assert store.get("openrouter", "huge") == "unknown"

    store.set("openrouter", "new", "accepted")

    payload = json.loads(overridden_cache_path.read_text(encoding="utf-8"))
    assert set(payload) == {"openrouter:valid", "openrouter:new"}
    assert store.get("openrouter", "valid") == "accepted"
