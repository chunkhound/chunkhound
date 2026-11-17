"""Consent and fallback contracts for automatic hub use by ``index``."""

import argparse
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

import chunkhound_native as native
from chunkhound.api.cli.commands import run
from chunkhound.api.cli.parsers.run_parser import add_run_subparser
from chunkhound.core.config.config import Config
from chunkhound.hub.errors import (
    HubInstallationInterrupted,
    HubLocalStateError,
    HubUnavailable,
)
from chunkhound.hub.trust import HubTrustError

pytestmark = pytest.mark.asyncio


class RecordingFormatter:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def info(self, message: str) -> None:
        self.messages.append(("info", message))

    def warning(self, message: str) -> None:
        self.messages.append(("warning", message))

    def error(self, message: str) -> None:
        self.messages.append(("error", message))

    @property
    def text(self) -> str:
        return "\n".join(message for _, message in self.messages)


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(
        target_dir=tmp_path,
        database={"provider": "duckdb", "path": tmp_path / ".chunkhound" / "db"},
        embedding={
            "provider": "openai",
            "model": "test-model",
            "output_dims": 8,
            "api_key": "test",
        },
        hub={"url": "https://hub.invalid/manifest?secret=hidden"},
        skip_layers={"global", "env", "local_config", "config_file", "cli"},
    )


@pytest.fixture
def args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        path=tmp_path,
        trust=False,
        simulate=False,
        check_ignores=False,
        no_embeddings=False,
    )


@pytest.fixture
def formatter() -> RecordingFormatter:
    return RecordingFormatter()


@pytest.fixture
def absent_target(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    target = tmp_path / ".chunkhound" / "db" / "chunks.db"
    monkeypatch.setattr(run, "installation_target", lambda _config: target)
    monkeypatch.setattr(run, "automatic_target_absent", lambda _target: True)
    return target


def _forbid(*_args, **_kwargs):
    raise AssertionError("forbidden boundary was contacted")


async def test_index_parser_accepts_trust() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_run_subparser(subparsers)

    parsed = parser.parse_args(["index", "--trust"])

    assert parsed.trust is True


@pytest.mark.parametrize("excluded", ["simulate", "check_ignores", "no_embeddings"])
async def test_subtools_skip_target_trust_prompt_and_hub(
    config, args, formatter, monkeypatch, excluded
):
    setattr(args, excluded, True)
    monkeypatch.setattr(run, "installation_target", _forbid)
    monkeypatch.setattr(run, "is_trusted", _forbid)
    monkeypatch.setattr(run.prompts, "prompt_choice", _forbid)
    monkeypatch.setattr(run, "pull", _forbid)

    decision = await run._prepare_hub_index(args, config, formatter)

    assert decision is run.HubIndexDecision.CONTINUE_LOCAL


async def test_mcp_mode_skips_target_trust_prompt_and_hub(
    config, args, formatter, monkeypatch
):
    monkeypatch.setenv("CHUNKHOUND_MCP_MODE", "1")
    monkeypatch.setattr(run, "installation_target", _forbid)
    monkeypatch.setattr(run, "is_trusted", _forbid)
    monkeypatch.setattr(run.prompts, "prompt_choice", _forbid)
    monkeypatch.setattr(run, "pull", _forbid)

    decision = await run._prepare_hub_index(args, config, formatter)

    assert decision is run.HubIndexDecision.CONTINUE_LOCAL


@pytest.mark.parametrize("hub_url", [None, ""])
async def test_no_url_continues_before_target_trust_or_hub(
    config, args, formatter, monkeypatch, hub_url
):
    config.hub.url = hub_url
    monkeypatch.setattr(run, "installation_target", _forbid)
    monkeypatch.setattr(run, "is_trusted", _forbid)
    monkeypatch.setattr(run, "pull", _forbid)

    decision = await run._prepare_hub_index(args, config, formatter)

    assert decision is run.HubIndexDecision.CONTINUE_LOCAL


@pytest.mark.parametrize("hub_url", [None, ""])
async def test_explicit_trust_without_url_stops_before_target_with_guidance(
    config, args, formatter, monkeypatch, hub_url
):
    config.hub.url = hub_url
    args.trust = True
    monkeypatch.setattr(run, "installation_target", _forbid)
    monkeypatch.setattr(run, "pull", _forbid)

    decision = await run._prepare_hub_index(args, config, formatter)

    assert decision is run.HubIndexDecision.STOP
    assert "hub" in formatter.text.lower()
    assert "config" in formatter.text.lower()


@pytest.mark.parametrize("aligned", [True, False])
async def test_existing_database_never_reads_trust_prompts_or_contacts_hub(
    config, args, formatter, monkeypatch, tmp_path, aligned
):
    target = tmp_path / ("aligned.db" if aligned else "misaligned.db")
    monkeypatch.setattr(run, "installation_target", lambda _config: target)
    monkeypatch.setattr(run, "automatic_target_absent", lambda _target: False)
    monkeypatch.setattr(run, "is_trusted", _forbid)
    monkeypatch.setattr(run.prompts, "prompt_choice", _forbid)
    monkeypatch.setattr(run, "pull", _forbid)

    decision = await run._prepare_hub_index(args, config, formatter)

    assert decision is run.HubIndexDecision.CONTINUE_LOCAL


async def test_unsupported_target_is_safe_failure_without_implicit_fallback(
    config, args, formatter, monkeypatch
):
    monkeypatch.setattr(
        run,
        "installation_target",
        lambda _config: (_ for _ in ()).throw(HubUnavailable("unsupported target")),
    )
    monkeypatch.setattr(run.prompts, "is_interactive", lambda: False)
    monkeypatch.setattr(run.prompts, "prompt_choice", _forbid)
    monkeypatch.setattr(run, "pull", _forbid)

    decision = await run._prepare_hub_index(args, config, formatter)

    assert decision is run.HubIndexDecision.STOP
    assert "unsupported target" in formatter.text


@pytest.mark.parametrize(
    ("failure", "expected_type"),
    [
        (native.DatabaseBusyError(), HubLocalStateError),
        (native.DatabaseLayoutError(), HubLocalStateError),
        (native.DatabaseFilesystemError(), HubLocalStateError),
        (native.DatabaseInterruptedError(), HubInstallationInterrupted),
        (OSError("raw local path"), HubLocalStateError),
    ],
)
async def test_target_inspection_failures_are_normalized_as_fatal_local_state(
    config, args, formatter, monkeypatch, tmp_path, failure, expected_type
):
    target = tmp_path / "chunks.db"
    monkeypatch.setattr(run, "installation_target", lambda _config: target)
    monkeypatch.setattr(
        run,
        "automatic_target_absent",
        lambda _target: (_ for _ in ()).throw(failure),
    )
    monkeypatch.setattr(run.prompts, "prompt_choice", _forbid)
    monkeypatch.setattr(run, "pull", _forbid)

    with pytest.raises(expected_type):
        await run._prepare_hub_index(args, config, formatter)


async def test_explicit_trust_existing_database_validates_and_saves_without_pull(
    config, args, formatter, monkeypatch, tmp_path
):
    args.trust = True
    target = tmp_path / "chunks.db"
    source = object()
    monkeypatch.setattr(run, "installation_target", lambda _config: target)
    monkeypatch.setattr(run, "automatic_target_absent", lambda _target: False)
    fetch = AsyncMock(return_value=source)
    grant = Mock()
    monkeypatch.setattr(run, "fetch_validated_manifest", fetch)
    monkeypatch.setattr(run, "grant_trust", grant)
    monkeypatch.setattr(run, "pull", _forbid)

    decision = await run._prepare_hub_index(args, config, formatter)

    assert decision is run.HubIndexDecision.CONTINUE_LOCAL
    fetch.assert_awaited_once_with(config)
    grant.assert_called_once_with(config.target_dir, config.hub.url)


async def test_existing_database_safe_validation_failure_offers_local_fallback(
    config, args, formatter, monkeypatch, tmp_path
):
    args.trust = True
    target = tmp_path / "chunks.db"
    monkeypatch.setattr(run, "installation_target", lambda _config: target)
    monkeypatch.setattr(run, "automatic_target_absent", lambda _target: False)
    monkeypatch.setattr(run.prompts, "is_interactive", lambda: True)
    monkeypatch.setattr(run.prompts, "prompt_choice", lambda *_args, **_kwargs: "local")
    monkeypatch.setattr(
        run,
        "fetch_validated_manifest",
        AsyncMock(side_effect=HubUnavailable("sanitized unavailable")),
    )
    monkeypatch.setattr(run, "pull", _forbid)

    decision = await run._prepare_hub_index(args, config, formatter)

    assert decision is run.HubIndexDecision.CONTINUE_LOCAL
    assert "sanitized unavailable" in formatter.text


async def test_untrusted_interactive_trust_validates_saves_and_pulls(
    config, args, formatter, monkeypatch, absent_target
):
    source = object()
    monkeypatch.setattr(run, "is_trusted", lambda *_args: False)
    monkeypatch.setattr(run.prompts, "is_interactive", lambda: True)
    monkeypatch.setattr(run.prompts, "prompt_choice", lambda *_args, **_kwargs: "trust")
    fetch = AsyncMock(return_value=source)
    grant = Mock()
    pull = AsyncMock(return_value=True)
    monkeypatch.setattr(run, "fetch_validated_manifest", fetch)
    monkeypatch.setattr(run, "grant_trust", grant)
    monkeypatch.setattr(run, "pull", pull)

    decision = await run._prepare_hub_index(args, config, formatter)

    assert decision is run.HubIndexDecision.HUB_INSTALLED
    fetch.assert_awaited_once_with(config)
    grant.assert_called_once_with(config.target_dir, config.hub.url)
    pull.assert_awaited_once_with(config, automatic=True, source=source)


@pytest.mark.parametrize(
    ("choice", "expected"),
    [("local", "CONTINUE_LOCAL"), ("cancel", "STOP")],
)
async def test_untrusted_interactive_local_or_cancel_never_contacts_hub(
    config, args, formatter, monkeypatch, absent_target, choice, expected
):
    monkeypatch.setattr(run, "is_trusted", lambda *_args: False)
    monkeypatch.setattr(run.prompts, "is_interactive", lambda: True)
    monkeypatch.setattr(
        run.prompts, "prompt_choice", lambda *_args, **_kwargs: choice
    )
    monkeypatch.setattr(run, "fetch_validated_manifest", _forbid)
    monkeypatch.setattr(run, "pull", _forbid)

    decision = await run._prepare_hub_index(args, config, formatter)

    assert decision is getattr(run.HubIndexDecision, expected)


async def test_untrusted_noninteractive_stops_before_hub_or_local(
    config, args, formatter, monkeypatch, absent_target
):
    monkeypatch.setattr(run, "is_trusted", lambda *_args: False)
    monkeypatch.setattr(run.prompts, "is_interactive", lambda: False)
    monkeypatch.setattr(run, "pull", _forbid)

    decision = await run._prepare_hub_index(args, config, formatter)

    assert decision is run.HubIndexDecision.STOP
    assert "index --trust" in formatter.text
    assert "pull --trust" in formatter.text
    assert "interactiv" in formatter.text.lower()


async def test_explicit_trust_authorizes_without_prompt(
    config, args, formatter, monkeypatch, absent_target
):
    args.trust = True
    source = object()
    monkeypatch.setattr(run.prompts, "prompt_choice", _forbid)
    monkeypatch.setattr(run, "is_trusted", _forbid)
    monkeypatch.setattr(run, "fetch_validated_manifest", AsyncMock(return_value=source))
    monkeypatch.setattr(run, "grant_trust", Mock())
    pull = AsyncMock(return_value=True)
    monkeypatch.setattr(run, "pull", pull)

    decision = await run._prepare_hub_index(args, config, formatter)

    assert decision is run.HubIndexDecision.HUB_INSTALLED
    pull.assert_awaited_once_with(config, automatic=True, source=source)


async def test_matching_saved_trust_pulls_without_prevalidated_source(
    config, args, formatter, monkeypatch, absent_target
):
    monkeypatch.setattr(run, "is_trusted", lambda *_args: True)
    monkeypatch.setattr(run.prompts, "prompt_choice", _forbid)
    pull = AsyncMock(return_value=True)
    monkeypatch.setattr(run, "pull", pull)

    decision = await run._prepare_hub_index(args, config, formatter)

    assert decision is run.HubIndexDecision.HUB_INSTALLED
    pull.assert_awaited_once_with(config, automatic=True)


async def test_trust_read_failure_is_fail_closed_and_can_choose_local(
    config, args, formatter, monkeypatch, absent_target
):
    def fail_read(*_args):
        raise HubTrustError("raw trust path secret")

    monkeypatch.setattr(run, "is_trusted", fail_read)
    monkeypatch.setattr(run.prompts, "is_interactive", lambda: True)
    monkeypatch.setattr(run.prompts, "prompt_choice", lambda *_args, **_kwargs: "local")
    monkeypatch.setattr(run, "pull", _forbid)

    decision = await run._prepare_hub_index(args, config, formatter)

    assert decision is run.HubIndexDecision.CONTINUE_LOCAL
    assert "raw trust path secret" not in formatter.text
    assert "trust" in formatter.text.lower()


async def test_trust_save_failure_warns_but_current_authorized_attempt_proceeds(
    config, args, formatter, monkeypatch, absent_target
):
    args.trust = True
    source = object()
    monkeypatch.setattr(run, "fetch_validated_manifest", AsyncMock(return_value=source))

    def fail_save(*_args):
        raise HubTrustError("raw save secret")

    monkeypatch.setattr(run, "grant_trust", fail_save)
    pull = AsyncMock(return_value=True)
    monkeypatch.setattr(run, "pull", pull)

    decision = await run._prepare_hub_index(args, config, formatter)

    assert decision is run.HubIndexDecision.HUB_INSTALLED
    assert "future automatic hub use was not enabled" in formatter.text
    assert "raw save secret" not in formatter.text
    pull.assert_awaited_once_with(config, automatic=True, source=source)


@pytest.mark.parametrize("interactive", [False, True])
async def test_authorized_safe_failure_requires_explicit_local_fallback(
    config, args, formatter, monkeypatch, absent_target, interactive
):
    args.trust = True
    monkeypatch.setattr(run.prompts, "is_interactive", lambda: interactive)
    monkeypatch.setattr(
        run.prompts,
        "prompt_choice",
        (lambda *_args, **_kwargs: "local") if interactive else _forbid,
    )
    monkeypatch.setattr(
        run,
        "fetch_validated_manifest",
        AsyncMock(side_effect=HubUnavailable("sanitized unavailable")),
    )

    decision = await run._prepare_hub_index(args, config, formatter)

    expected = (
        run.HubIndexDecision.CONTINUE_LOCAL
        if interactive
        else run.HubIndexDecision.STOP
    )
    assert decision is expected
    assert "sanitized unavailable" in formatter.text
    assert config.hub.url not in formatter.text


async def test_authorized_pull_failure_can_fall_back_only_interactively(
    config, args, formatter, monkeypatch, absent_target
):
    monkeypatch.setattr(run, "is_trusted", lambda *_args: True)
    monkeypatch.setattr(run.prompts, "is_interactive", lambda: True)
    monkeypatch.setattr(run.prompts, "prompt_choice", lambda *_args, **_kwargs: "local")
    monkeypatch.setattr(run, "pull", AsyncMock(return_value=False))

    decision = await run._prepare_hub_index(args, config, formatter)

    assert decision is run.HubIndexDecision.CONTINUE_LOCAL


async def test_fatal_local_failure_never_offers_fallback(
    config, args, formatter, monkeypatch, absent_target
):
    monkeypatch.setattr(run, "is_trusted", lambda *_args: True)
    monkeypatch.setattr(run.prompts, "prompt_choice", _forbid)
    monkeypatch.setattr(
        run, "pull", AsyncMock(side_effect=HubLocalStateError("unsafe local state"))
    )

    with pytest.raises(HubLocalStateError, match="unsafe local state"):
        await run._prepare_hub_index(args, config, formatter)
