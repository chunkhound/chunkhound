"""Contract tests for global-config target resolution.

Locks the invariant that the remote-config writer and the loader's reader
agree on which file is "the global JSON":

- `CHUNKHOUND_GLOBAL_CONFIG_FILE` wins over auto-discovered candidates for
  both sides.
- With env var unset, the reader/writer fall back to the same candidate
  list, with `~/.chunkhound.json` as the write-target fallback when none
  exist yet.
- A pinned env-var target that does not yet exist is not a fatal reader
  error — the remote pipeline creates it on first apply.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chunkhound.core.config.config import Config
from chunkhound.core.config.remote import persistence


def test_resolve_target_honors_env_var_when_set(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A candidate file exists at the default HOME location — the resolver
    # must still prefer the env-var-pinned target.
    (_isolate / ".chunkhound.json").write_text("{}")
    pinned = _isolate / "elsewhere" / "global.json"
    monkeypatch.setenv("CHUNKHOUND_GLOBAL_CONFIG_FILE", str(pinned))

    assert persistence.resolve_target() == pinned


def test_resolve_target_uses_candidates_when_env_unset(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CHUNKHOUND_GLOBAL_CONFIG_FILE", raising=False)

    # None exist → last candidate is the write-target fallback.
    fallback = _isolate / ".chunkhound.json"
    assert persistence.resolve_target() == fallback

    # First existing candidate wins over the last-candidate fallback.
    higher_priority = _isolate / ".config" / "chunkhound" / "chunkhound.json"
    higher_priority.parent.mkdir(parents=True)
    higher_priority.write_text("{}")
    assert persistence.resolve_target() == higher_priority


def test_env_pinned_missing_does_not_fall_back_to_candidates(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A candidate exists at the default location, but env var pins a
    # missing path. Env var wins — no silent fallback to the candidate,
    # since the pinned path expresses operator intent.
    (_isolate / ".chunkhound.json").write_text("{}")
    pinned = _isolate / "not-yet" / "global.json"
    monkeypatch.setenv("CHUNKHOUND_GLOBAL_CONFIG_FILE", str(pinned))

    assert persistence.resolve_target() == pinned
    cfg = Config()
    assert cfg.global_config_file is None


def test_reader_treats_missing_env_target_as_empty_base(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Env var pins a path that does not yet exist. Reader must not raise;
    # the remote pipeline is responsible for creating it on first apply.
    pinned = _isolate / "not-yet" / "global.json"
    assert not pinned.exists()
    monkeypatch.setenv("CHUNKHOUND_GLOBAL_CONFIG_FILE", str(pinned))

    # Construction succeeds and reflects defaults + env only — no crash.
    cfg = Config()
    # global_config_file stays unset (symmetric with "no candidate exists"
    # under env-var-unset) — the write target lives in resolve_target, not
    # on the Config field.
    assert cfg.global_config_file is None
