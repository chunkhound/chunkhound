"""Consent and fallback contracts for automatic hub use by ``index``."""

import argparse
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

import chunkhound_native as native
from chunkhound.api.cli.commands import run
from chunkhound.api.cli.parsers.run_parser import add_run_subparser
from chunkhound.core.config.config import Config
from chunkhound.hub.client import ValidatedHubSource
from chunkhound.hub.errors import (
    HUB_PULL_FAILURE_GUIDANCE,
    HubDatabaseMismatch,
    HubInstallationInterrupted,
    HubLocalStateError,
    HubUnavailable,
)
from chunkhound.hub.trust import HubTrustError, checkout_digest, manifest_url_digest

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
    monkeypatch.setattr(
        run, "automatic_target_appears_cold", lambda _target, _backend: True
    )
    monkeypatch.setattr(
        run, "automatic_target_absent", lambda _target, _backend: True
    )
    return target


def _forbid(*_args, **_kwargs):
    raise AssertionError("forbidden boundary was contacted")


def _validated_source(checkout: Path, url: str) -> ValidatedHubSource:
    return ValidatedHubSource(
        manifest=Mock(),
        checkout_digest=checkout_digest(checkout),
        url_digest=manifest_url_digest(url),
    )


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
async def test_trusted_existing_index_never_fetches_or_pulls(
    config, args, formatter, monkeypatch, tmp_path, aligned
):
    target = tmp_path / ("aligned.db" if aligned else "misaligned.db")
    monkeypatch.setattr(run, "is_trusted", lambda *_args: True)
    monkeypatch.setattr(run, "installation_target", lambda _config: target)
    monkeypatch.setattr(
        run, "automatic_target_absent", lambda _target, _backend: False
    )
    monkeypatch.setattr(run.prompts, "prompt_choice", _forbid)
    monkeypatch.setattr(run, "fetch_validated_manifest", _forbid)
    monkeypatch.setattr(run, "pull", _forbid)
    monkeypatch.setattr(run, "_hub_authorized_failure_guidance", _forbid)

    decision = await run._prepare_hub_index(args, config, formatter)

    assert decision is run.HubIndexDecision.CONTINUE_LOCAL


async def test_untrusted_noninteractive_cold_continues_local_without_hub(
    config, args, formatter, monkeypatch
):
    monkeypatch.setattr(run, "is_trusted", lambda *_args: False)
    monkeypatch.setattr(run.prompts, "is_interactive", lambda: False)
    monkeypatch.setattr(run, "installation_target", _forbid)
    monkeypatch.setattr(run, "automatic_target_absent", _forbid)
    monkeypatch.setattr(run, "fetch_validated_manifest", _forbid)
    monkeypatch.setattr(run, "pull", _forbid)
    monkeypatch.setattr(run, "grant_trust", _forbid)
    monkeypatch.setattr(run, "_hub_authorized_failure_guidance", _forbid)

    decision = await run._prepare_hub_index(args, config, formatter)

    assert decision is run.HubIndexDecision.CONTINUE_LOCAL


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("local", run.HubIndexDecision.CONTINUE_LOCAL),
        ("cancel", run.HubIndexDecision.STOP),
        (None, run.HubIndexDecision.STOP),
        ("", run.HubIndexDecision.STOP),
        ("invalid", run.HubIndexDecision.STOP),
    ],
)
async def test_untrusted_interactive_cold_offers_local_or_cancel_once(
    config, args, formatter, monkeypatch, tmp_path, answer, expected
):
    target = tmp_path / ".chunkhound" / "db" / "chunks.db"
    monkeypatch.setattr(run, "is_trusted", lambda *_args: False)
    monkeypatch.setattr(run.prompts, "is_interactive", lambda: True)
    monkeypatch.setattr(run, "installation_target", lambda _config: target)
    monkeypatch.setattr(
        run, "automatic_target_appears_cold", lambda _target, _backend: True
    )
    prompt = Mock(return_value=answer)
    monkeypatch.setattr(run.prompts, "prompt_text", prompt)
    monkeypatch.setattr(run, "fetch_validated_manifest", _forbid)
    monkeypatch.setattr(run, "pull", _forbid)
    monkeypatch.setattr(run, "grant_trust", _forbid)

    decision = await run._prepare_hub_index(args, config, formatter)

    assert decision is expected
    prompt.assert_called_once()
    assert "index --trust" in formatter.text
    assert "secret=hidden" not in formatter.text


async def test_untrusted_interactive_non_cold_skips_prompt_and_continues_local(
    config, args, formatter, monkeypatch, tmp_path
):
    target = tmp_path / ".chunkhound" / "db" / "chunks.db"
    monkeypatch.setattr(run, "is_trusted", lambda *_args: False)
    monkeypatch.setattr(run.prompts, "is_interactive", lambda: True)
    monkeypatch.setattr(run, "installation_target", lambda _config: target)
    monkeypatch.setattr(
        run, "automatic_target_appears_cold", lambda _target, _backend: False
    )
    monkeypatch.setattr(run.prompts, "prompt_text", _forbid)
    monkeypatch.setattr(run, "automatic_target_absent", _forbid)
    monkeypatch.setattr(run, "fetch_validated_manifest", _forbid)
    monkeypatch.setattr(run, "pull", _forbid)
    monkeypatch.setattr(run, "grant_trust", _forbid)
    monkeypatch.setattr(run, "_hub_authorized_failure_guidance", _forbid)

    decision = await run._prepare_hub_index(args, config, formatter)

    assert decision is run.HubIndexDecision.CONTINUE_LOCAL


async def test_prompts_disabled_untrusted_index_continues_local_without_hub(
    config, args, formatter, monkeypatch
):
    monkeypatch.setenv("CHUNKHOUND_NO_PROMPTS", "1")
    monkeypatch.setattr(run, "is_trusted", lambda *_args: False)
    monkeypatch.setattr(run, "installation_target", _forbid)
    monkeypatch.setattr(run, "automatic_target_appears_cold", _forbid)
    monkeypatch.setattr(run, "fetch_validated_manifest", _forbid)
    monkeypatch.setattr(run, "pull", _forbid)
    monkeypatch.setattr(run, "grant_trust", _forbid)

    decision = await run._prepare_hub_index(args, config, formatter)

    assert decision is run.HubIndexDecision.CONTINUE_LOCAL


async def test_unsupported_target_is_safe_failure_without_implicit_fallback(
    config, args, formatter, monkeypatch
):
    args.trust = True
    source = _validated_source(config.target_dir, config.hub.url)
    monkeypatch.setattr(
        run,
        "installation_target",
        lambda _config: (_ for _ in ()).throw(HubUnavailable("unsupported target")),
    )
    monkeypatch.setattr(
        run, "fetch_validated_manifest", AsyncMock(return_value=source)
    )
    monkeypatch.setattr(run.prompts, "is_interactive", lambda: False)
    monkeypatch.setattr(run.prompts, "prompt_choice", _forbid)
    monkeypatch.setattr(run, "pull", _forbid)

    decision = await run._prepare_hub_index(args, config, formatter)

    assert decision is run.HubIndexDecision.STOP
    assert "unsupported target" in formatter.text
    assert HUB_PULL_FAILURE_GUIDANCE not in formatter.text


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
    args.trust = True
    target = tmp_path / "chunks.db"
    source = _validated_source(config.target_dir, config.hub.url)
    monkeypatch.setattr(run, "installation_target", lambda _config: target)
    monkeypatch.setattr(
        run, "fetch_validated_manifest", AsyncMock(return_value=source)
    )
    monkeypatch.setattr(
        run,
        "automatic_target_absent",
        lambda _target, _backend: (_ for _ in ()).throw(failure),
    )
    monkeypatch.setattr(run.prompts, "prompt_choice", _forbid)
    monkeypatch.setattr(run, "pull", _forbid)

    with pytest.raises(expected_type):
        await run._prepare_hub_index(args, config, formatter)
    assert "raw local path" not in formatter.text


async def test_checkout_identity_resolution_failure_is_normalized(
    config, args, formatter, monkeypatch
):
    raw = "raw checkout path?token=secret"

    def fail_resolve(_checkout):
        raise OSError(raw)

    monkeypatch.setattr(run, "checkout_digest", fail_resolve)

    with pytest.raises(HubLocalStateError) as caught:
        await run._prepare_hub_index(args, config, formatter)

    assert str(caught.value) == "Hub checkout identity is unavailable"
    assert raw not in str(caught.value)


async def test_run_command_formats_checkout_failure_once_without_raw_cause(
    config, args, monkeypatch
):
    raw = "raw checkout path?token=secret"
    formatter = Mock()

    def fail_digest(_checkout):
        raise OSError(raw)

    monkeypatch.setattr(run, "checkout_digest", fail_digest)
    monkeypatch.setattr(run, "RichOutputFormatter", lambda verbose: formatter)
    args.verbose = False
    args.db = config.database.path
    monkeypatch.setattr(run, "_validate_run_arguments", lambda *_args: True)

    with pytest.raises(SystemExit) as stopped:
        await run.run_command(args, config)

    assert stopped.value.code == 1
    formatter.error.assert_called_once_with("Hub checkout identity is unavailable")
    assert raw not in str(formatter.method_calls)


async def test_run_command_does_not_duplicate_preformatted_fatal_error(
    config, args, monkeypatch
):
    message = "Hub installation was interrupted; retry explicit pull"
    failure = HubInstallationInterrupted(message)
    formatter = Mock()
    monkeypatch.setattr(run, "RichOutputFormatter", lambda verbose: formatter)
    monkeypatch.setattr(
        run,
        "_prepare_hub_index",
        AsyncMock(side_effect=failure),
    )
    args.verbose = False
    args.db = config.database.path
    monkeypatch.setattr(run, "_validate_run_arguments", lambda *_args: True)

    with pytest.raises(SystemExit) as stopped:
        await run.run_command(args, config)

    assert stopped.value.code == 1
    formatter.error.assert_not_called()


async def test_index_trust_existing_validates_refreshes_trust_without_artifact_pull(
    config, args, formatter, monkeypatch, tmp_path
):
    args.trust = True
    target = tmp_path / "chunks.db"
    source = _validated_source(config.target_dir, config.hub.url)
    monkeypatch.setattr(run, "installation_target", lambda _config: target)
    monkeypatch.setattr(
        run, "automatic_target_absent", lambda _target, _backend: False
    )
    fetch = AsyncMock(return_value=source)
    grant = Mock()
    monkeypatch.setattr(run, "fetch_validated_manifest", fetch)
    monkeypatch.setattr(run, "grant_trust", grant)
    monkeypatch.setattr(run, "pull", _forbid)
    monkeypatch.setattr(run, "_hub_authorized_failure_guidance", _forbid)

    decision = await run._prepare_hub_index(args, config, formatter)

    assert decision is run.HubIndexDecision.CONTINUE_LOCAL
    fetch.assert_awaited_once_with(config)
    grant.assert_called_once_with(config.target_dir, config.hub.url)


@pytest.mark.parametrize("change", ["checkout", "url"])
async def test_interactive_binding_change_stops_without_grant_or_local_fallback(
    config, args, formatter, monkeypatch, tmp_path, change
):
    args.trust = True
    target = tmp_path / "chunks.db"
    authorized_checkout = config.target_dir.resolve()
    authorized_url = config.hub.url
    source = _validated_source(authorized_checkout, authorized_url)
    changed_checkout = tmp_path / "changed-checkout"
    changed_checkout.mkdir()

    async def mutate_config_during_fetch(_config):
        if change == "checkout":
            config.target_dir = changed_checkout
        else:
            config.hub.url = "https://changed.invalid/manifest?secret=changed"
        return source

    monkeypatch.setattr(run, "installation_target", lambda _config: target)
    monkeypatch.setattr(
        run, "automatic_target_absent", lambda _target, _backend: False
    )
    fetch = AsyncMock(side_effect=mutate_config_during_fetch)
    grant = Mock()
    monkeypatch.setattr(run, "fetch_validated_manifest", fetch)
    monkeypatch.setattr(run, "grant_trust", grant)
    monkeypatch.setattr(run.prompts, "is_interactive", lambda: True)
    monkeypatch.setattr(run.prompts, "prompt_choice", _forbid)
    monkeypatch.setattr(run, "pull", _forbid)

    decision = await run._prepare_hub_index(args, config, formatter)

    assert decision is run.HubIndexDecision.STOP
    fetch.assert_awaited_once_with(config)
    grant.assert_not_called()
    assert formatter.messages == [
        (
            "error",
            "Validated hub source no longer matches authorized configuration",
        )
    ]
    assert authorized_url not in formatter.text


async def test_interactive_invalid_manifest_stops_without_grant_or_local_fallback(
    config, args, formatter, monkeypatch
):
    args.trust = True
    grant = Mock()
    monkeypatch.setattr(
        run,
        "fetch_validated_manifest",
        AsyncMock(
            side_effect=HubUnavailable("Hub manifest is invalid or unsupported")
        ),
    )
    monkeypatch.setattr(run, "grant_trust", grant)
    monkeypatch.setattr(run.prompts, "is_interactive", lambda: True)
    monkeypatch.setattr(run.prompts, "prompt_choice", _forbid)
    monkeypatch.setattr(run, "pull", _forbid)

    decision = await run._prepare_hub_index(args, config, formatter)

    assert decision is run.HubIndexDecision.STOP
    grant.assert_not_called()
    assert formatter.messages == [
        ("error", "Hub manifest is invalid or unsupported")
    ]


async def test_index_trust_ambiguous_state_retains_validated_trust_but_aborts(
    config, args, formatter, monkeypatch, tmp_path
):
    args.trust = True
    target = tmp_path / "chunks.db"
    source = _validated_source(config.target_dir, config.hub.url)
    grant = Mock()
    monkeypatch.setattr(
        run, "fetch_validated_manifest", AsyncMock(return_value=source)
    )
    monkeypatch.setattr(run, "grant_trust", grant)
    monkeypatch.setattr(run, "installation_target", lambda _config: target)
    monkeypatch.setattr(
        run,
        "automatic_target_absent",
        lambda _target, _backend: (_ for _ in ()).throw(
            HubLocalStateError("ambiguous local state")
        ),
    )
    monkeypatch.setattr(run, "pull", _forbid)

    with pytest.raises(HubLocalStateError, match="ambiguous local state"):
        await run._prepare_hub_index(args, config, formatter)

    grant.assert_called_once_with(config.target_dir, config.hub.url)


async def test_mismatched_validated_capability_stops_before_grant_or_pull(
    config, args, formatter, monkeypatch, tmp_path
):
    args.trust = True
    target = tmp_path / "chunks.db"
    mismatched = _validated_source(
        config.target_dir, "https://other.invalid/manifest?secret=other"
    )
    grant = Mock()
    monkeypatch.setattr(run, "installation_target", lambda _config: target)
    monkeypatch.setattr(
        run, "automatic_target_absent", lambda _target, _backend: False
    )
    monkeypatch.setattr(
        run, "fetch_validated_manifest", AsyncMock(return_value=mismatched)
    )
    monkeypatch.setattr(run, "grant_trust", grant)
    monkeypatch.setattr(run.prompts, "is_interactive", lambda: False)
    monkeypatch.setattr(run.prompts, "prompt_choice", _forbid)
    monkeypatch.setattr(run, "pull", _forbid)

    decision = await run._prepare_hub_index(args, config, formatter)

    assert decision is run.HubIndexDecision.STOP
    grant.assert_not_called()
    assert "no longer matches" in formatter.text
    assert config.hub.url not in formatter.text


async def test_post_fetch_checkout_failure_is_formatted_once(
    config, args, formatter, monkeypatch, tmp_path
):
    args.trust = True
    target = tmp_path / "chunks.db"
    source = _validated_source(config.target_dir, config.hub.url)
    digest_calls = 0

    def fail_revalidation(checkout):
        nonlocal digest_calls
        digest_calls += 1
        if digest_calls == 2:
            raise OSError("raw post-fetch checkout path?token=secret")
        return checkout_digest(checkout)

    monkeypatch.setattr(run, "installation_target", lambda _config: target)
    monkeypatch.setattr(
        run, "automatic_target_absent", lambda _target, _backend: False
    )
    monkeypatch.setattr(
        run,
        "fetch_validated_manifest",
        AsyncMock(return_value=source),
    )
    monkeypatch.setattr(run, "checkout_digest", fail_revalidation)

    with pytest.raises(HubLocalStateError):
        await run._prepare_hub_index(args, config, formatter)

    assert formatter.messages == [
        ("error", "Hub checkout identity is unavailable")
    ]
    assert "raw post-fetch checkout path" not in formatter.text


async def test_manifest_binding_fatal_failure_is_formatted_before_run_exit(
    config, args, formatter, monkeypatch, tmp_path
):
    args.trust = True
    target = tmp_path / "chunks.db"
    failure = HubLocalStateError("Hub checkout identity is unavailable")
    monkeypatch.setattr(run, "installation_target", lambda _config: target)
    monkeypatch.setattr(
        run, "automatic_target_absent", lambda _target, _backend: False
    )
    monkeypatch.setattr(
        run,
        "fetch_validated_manifest",
        AsyncMock(side_effect=failure),
    )

    with pytest.raises(HubLocalStateError):
        await run._prepare_hub_index(args, config, formatter)

    assert formatter.messages == [
        ("error", "Hub checkout identity is unavailable")
    ]


async def test_postauthorization_classification_failure_offers_local_fallback(
    config, args, formatter, monkeypatch, tmp_path
):
    args.trust = True
    target = tmp_path / "chunks.db"
    source = _validated_source(config.target_dir, config.hub.url)
    monkeypatch.setattr(run, "installation_target", lambda _config: target)
    monkeypatch.setattr(
        run,
        "automatic_target_absent",
        lambda _target, _backend: (_ for _ in ()).throw(
            HubUnavailable("sanitized unavailable")
        ),
    )
    monkeypatch.setattr(run.prompts, "is_interactive", lambda: True)
    monkeypatch.setattr(run.prompts, "prompt_choice", lambda *_args, **_kwargs: "local")
    monkeypatch.setattr(
        run, "fetch_validated_manifest", AsyncMock(return_value=source)
    )
    monkeypatch.setattr(run, "grant_trust", Mock())
    monkeypatch.setattr(run, "pull", _forbid)

    decision = await run._prepare_hub_index(args, config, formatter)

    assert decision is run.HubIndexDecision.CONTINUE_LOCAL
    assert "sanitized unavailable" in formatter.text


async def test_index_trust_cold_validates_refreshes_trust_and_pulls_source(
    config, args, formatter, monkeypatch, absent_target
):
    args.trust = True
    source = _validated_source(config.target_dir, config.hub.url)
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


async def test_explicit_trust_authorizes_without_prompt(
    config, args, formatter, monkeypatch, absent_target
):
    args.trust = True
    source = _validated_source(config.target_dir, config.hub.url)
    monkeypatch.setattr(run.prompts, "prompt_choice", _forbid)
    monkeypatch.setattr(run, "is_trusted", _forbid)
    monkeypatch.setattr(run, "fetch_validated_manifest", AsyncMock(return_value=source))
    monkeypatch.setattr(run, "grant_trust", Mock())
    pull = AsyncMock(return_value=True)
    monkeypatch.setattr(run, "pull", pull)

    decision = await run._prepare_hub_index(args, config, formatter)

    assert decision is run.HubIndexDecision.HUB_INSTALLED
    pull.assert_awaited_once_with(config, automatic=True, source=source)


async def test_trusted_cold_index_calls_automatic_pull(
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
    config, args, formatter, monkeypatch
):
    def fail_read(*_args):
        raise HubTrustError("raw trust path secret")

    target = config.target_dir / ".chunkhound" / "db" / "chunks.db"
    monkeypatch.setattr(run, "is_trusted", fail_read)
    monkeypatch.setattr(run, "installation_target", lambda _config: target)
    monkeypatch.setattr(
        run, "automatic_target_appears_cold", lambda _target, _backend: True
    )
    monkeypatch.setattr(run.prompts, "is_interactive", lambda: True)
    monkeypatch.setattr(run.prompts, "prompt_text", lambda *_args, **_kwargs: "local")
    monkeypatch.setattr(run, "pull", _forbid)

    decision = await run._prepare_hub_index(args, config, formatter)

    assert decision is run.HubIndexDecision.CONTINUE_LOCAL
    assert "raw trust path secret" not in formatter.text
    assert "trust" in formatter.text.lower()


async def test_trust_save_failure_warns_but_current_authorized_attempt_proceeds(
    config, args, formatter, monkeypatch, absent_target
):
    args.trust = True
    source = _validated_source(config.target_dir, config.hub.url)
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
async def test_attempted_index_pull_unavailable_always_stops_with_local_retry_guidance(
    config, args, formatter, monkeypatch, absent_target, interactive
):
    args.trust = True
    source = _validated_source(config.target_dir, config.hub.url)
    monkeypatch.setattr(run.prompts, "is_interactive", lambda: interactive)
    monkeypatch.setattr(run.prompts, "prompt_choice", _forbid)
    monkeypatch.setattr(
        run, "fetch_validated_manifest", AsyncMock(return_value=source)
    )
    monkeypatch.setattr(run, "grant_trust", Mock())
    monkeypatch.setattr(
        run, "pull", AsyncMock(side_effect=HubUnavailable("sanitized unavailable"))
    )

    decision = await run._prepare_hub_index(args, config, formatter)

    assert decision is run.HubIndexDecision.STOP
    assert formatter.messages == [
        ("error", "sanitized unavailable"),
        ("info", HUB_PULL_FAILURE_GUIDANCE),
    ]
    assert "--forget-trust" in formatter.text
    assert "disable hub.url" in formatter.text
    assert config.hub.url not in formatter.text


async def test_attempted_index_pull_cancellation_stops_with_exact_guidance(
    config, args, formatter, monkeypatch, absent_target
):
    monkeypatch.setattr(run, "is_trusted", lambda *_args: True)
    monkeypatch.setattr(run, "pull", AsyncMock(side_effect=asyncio.CancelledError()))
    monkeypatch.setattr(run.prompts, "prompt_choice", _forbid)

    decision = await run._prepare_hub_index(args, config, formatter)

    assert decision is run.HubIndexDecision.STOP
    assert formatter.messages == [
        ("error", "The attempted hub pull was cancelled before completion."),
        ("info", HUB_PULL_FAILURE_GUIDANCE),
    ]
    assert config.hub.url not in formatter.text


@pytest.mark.parametrize(
    "failure",
    [
        HubDatabaseMismatch("mismatch"),
        HubInstallationInterrupted("interrupted"),
        HubLocalStateError("unsafe local state"),
    ],
)
async def test_attempted_index_pull_fatal_error_stops_with_guidance(
    config, args, formatter, monkeypatch, absent_target, failure
):
    monkeypatch.setattr(run, "is_trusted", lambda *_args: True)
    monkeypatch.setattr(run, "pull", AsyncMock(side_effect=failure))
    monkeypatch.setattr(run.prompts, "prompt_choice", _forbid)

    with pytest.raises(type(failure), match=str(failure)):
        await run._prepare_hub_index(args, config, formatter)

    assert formatter.messages == [("info", HUB_PULL_FAILURE_GUIDANCE)]


async def test_race_created_valid_database_continues_without_failure_prompt(
    config, args, formatter, monkeypatch, tmp_path
):
    target = tmp_path / "chunks.db"
    states = iter([True, False])
    monkeypatch.setattr(run, "is_trusted", lambda *_args: True)
    monkeypatch.setattr(run, "installation_target", lambda _config: target)
    monkeypatch.setattr(
        run,
        "automatic_target_absent",
        lambda _target, _backend: next(states),
    )
    monkeypatch.setattr(run, "pull", AsyncMock(return_value=False))
    monkeypatch.setattr(run, "_choose_hub_failure_action", _forbid)
    monkeypatch.setattr(run, "_hub_authorized_failure_guidance", _forbid)

    decision = await run._prepare_hub_index(args, config, formatter)

    assert decision is run.HubIndexDecision.CONTINUE_LOCAL
    assert formatter.messages == []


@pytest.mark.parametrize("interactive", [False, True])
async def test_still_cold_after_unsuccessful_pull_always_stops_with_guidance(
    config, args, formatter, monkeypatch, tmp_path, interactive
):
    target = tmp_path / "chunks.db"
    monkeypatch.setattr(run, "is_trusted", lambda *_args: True)
    monkeypatch.setattr(run, "installation_target", lambda _config: target)
    monkeypatch.setattr(
        run, "automatic_target_absent", lambda _target, _backend: True
    )
    monkeypatch.setattr(run, "pull", AsyncMock(return_value=False))
    monkeypatch.setattr(run.prompts, "is_interactive", lambda: interactive)
    monkeypatch.setattr(run.prompts, "prompt_choice", _forbid)

    decision = await run._prepare_hub_index(args, config, formatter)

    assert decision is run.HubIndexDecision.STOP
    assert formatter.messages == [("info", HUB_PULL_FAILURE_GUIDANCE)]


@pytest.mark.parametrize(
    ("failure", "expected_type", "diagnostic"),
    [
        (
            HubLocalStateError("ambiguous local state"),
            HubLocalStateError,
            "ambiguous local state",
        ),
        (
            native.DatabaseBusyError(),
            HubLocalStateError,
            "Hub destination is busy; stop database users and retry",
        ),
        (
            native.DatabaseInterruptedError(),
            HubInstallationInterrupted,
            "Hub installation was interrupted; run explicit chunkhound pull again. "
            "Local work may require re-indexing",
        ),
        (
            native.DatabaseLayoutError(),
            HubLocalStateError,
            "Hub database installation layout is unsafe",
        ),
        (
            native.DatabaseFilesystemError(),
            HubLocalStateError,
            "Hub installation filesystem is unavailable",
        ),
        (
            OSError("raw path?token=secret"),
            HubLocalStateError,
            "Hub local filesystem is unavailable",
        ),
    ],
)
async def test_race_created_ambiguous_state_aborts_before_registry(
    config,
    args,
    formatter,
    monkeypatch,
    tmp_path,
    failure,
    expected_type,
    diagnostic,
):
    target = tmp_path / "chunks.db"
    states = iter([True, failure])

    def classify(_target, _backend):
        state = next(states)
        if isinstance(state, Exception):
            raise state
        return state

    monkeypatch.setattr(run, "is_trusted", lambda *_args: True)
    monkeypatch.setattr(run, "installation_target", lambda _config: target)
    monkeypatch.setattr(run, "automatic_target_absent", classify)
    monkeypatch.setattr(run, "pull", AsyncMock(return_value=False))
    monkeypatch.setattr(run.prompts, "prompt_choice", _forbid)
    monkeypatch.setattr(run, "_choose_hub_failure_action", _forbid)

    with pytest.raises(expected_type, match=diagnostic):
        await run._prepare_hub_index(args, config, formatter)

    assert formatter.messages == [
        ("error", diagnostic),
        ("info", HUB_PULL_FAILURE_GUIDANCE),
    ]
    assert "raw path" not in formatter.text


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
