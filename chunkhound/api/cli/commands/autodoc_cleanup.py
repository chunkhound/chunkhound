from __future__ import annotations

import os
from argparse import Namespace

from loguru import logger

from chunkhound.api.cli.utils.rich_output import RichOutputFormatter
from chunkhound.autodoc.docsite import CleanupConfig
from chunkhound.core.config.config import Config
from chunkhound.core.config.llm_config import LLMConfig
from chunkhound.llm_manager import LLMManager
from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider

from .autodoc_errors import AutoDocCLIExitError


def _has_llm_env() -> bool:
    return any(key.startswith("CHUNKHOUND_LLM_") for key in os.environ)


def _build_cleanup_provider_configs(
    llm_config: LLMConfig,
) -> tuple[dict[str, object], dict[str, object]]:
    utility_config, synthesis_config = llm_config.get_provider_configs()

    cleanup_provider = llm_config.autodoc_cleanup_provider
    cleanup_model = llm_config.autodoc_cleanup_model
    cleanup_effort = llm_config.autodoc_cleanup_reasoning_effort

    if cleanup_provider:
        synthesis_config = synthesis_config.copy()
        synthesis_config["provider"] = cleanup_provider

    if cleanup_model:
        synthesis_config = synthesis_config.copy()
        synthesis_config["model"] = cleanup_model

    provider = synthesis_config.get("provider")
    if (
        cleanup_effort
        and isinstance(provider, str)
        and provider in ("codex-cli", "openai")
    ):
        synthesis_config = synthesis_config.copy()
        synthesis_config["reasoning_effort"] = cleanup_effort

    return utility_config, synthesis_config


def _try_load_llm_config_from_env(
    *, formatter: RichOutputFormatter
) -> LLMConfig | None:
    if not _has_llm_env():
        return None
    try:
        return LLMConfig()
    except Exception as exc:
        formatter.warning(f"Failed to load LLM config from environment: {exc}")
        return None


def _resolve_llm_config_for_cleanup(
    *, config: Config, formatter: RichOutputFormatter
) -> LLMConfig | None:
    llm_config = config.llm
    if llm_config is None:
        llm_config = _try_load_llm_config_from_env(formatter=formatter)

    if llm_config is None:
        return None

    if not llm_config.is_provider_configured():
        return None

    return llm_config


def _log_cleanup_model_selection(
    *,
    formatter: RichOutputFormatter,
    llm_config: LLMConfig,
    synthesis_config: dict[str, object],
) -> None:
    provider = synthesis_config.get("provider", "unknown")
    model = synthesis_config.get("model", "unknown")
    effort = synthesis_config.get("reasoning_effort")

    override_notes = [
        label
        for label, enabled in (
            ("cleanup provider", llm_config.autodoc_cleanup_provider),
            ("cleanup model", llm_config.autodoc_cleanup_model),
            ("cleanup reasoning effort", llm_config.autodoc_cleanup_reasoning_effort),
        )
        if enabled
    ]
    suffix = f" ({', '.join(override_notes)} override)" if override_notes else ""

    if isinstance(provider, str) and provider == "codex-cli":
        resolved_model, _model_source = CodexCLIProvider.describe_model_resolution(
            model if isinstance(model, str) else None
        )
        resolved_effort, _effort_source = (
            CodexCLIProvider.describe_reasoning_effort_resolution(
                effort if isinstance(effort, str) else None
            )
        )
        formatter.info(
            "Cleanup model selection: "
            f"provider={provider}, model={model} (resolved={resolved_model}), "
            f"reasoning_effort={resolved_effort}{suffix}"
        )
        return

    effort_display = f", reasoning_effort={effort}" if effort else ""
    formatter.info(
        f"Cleanup model selection: provider={provider}, model={model}"
        f"{effort_display}{suffix}"
    )


def _build_llm_manager_for_cleanup(
    *,
    llm_config: LLMConfig,
    formatter: RichOutputFormatter,
) -> LLMManager | None:
    try:
        utility_config, synthesis_config = _build_cleanup_provider_configs(llm_config)
        _log_cleanup_model_selection(
            formatter=formatter,
            llm_config=llm_config,
            synthesis_config=synthesis_config,
        )
        return LLMManager(utility_config, synthesis_config)
    except Exception as exc:
        formatter.warning(f"Failed to configure LLM provider: {exc}")
        logger.exception("LLM configuration error")
        return None


def resolve_llm_manager(
    *,
    config: Config,
    formatter: RichOutputFormatter,
) -> LLMManager | None:
    llm_config = _resolve_llm_config_for_cleanup(config=config, formatter=formatter)
    if llm_config is None:
        return None
    return _build_llm_manager_for_cleanup(llm_config=llm_config, formatter=formatter)


def resolve_cleanup_config_and_llm_manager(
    *,
    args: Namespace,
    config: Config,
    formatter: RichOutputFormatter,
) -> tuple[CleanupConfig, LLMManager]:
    cleanup_mode = getattr(args, "cleanup_mode", "llm")
    if cleanup_mode != "llm":
        raise AutoDocCLIExitError(
            exit_code=2,
            errors=(
                "Unsupported AutoDoc cleanup mode: "
                f"{cleanup_mode!r}. AutoDoc cleanup now requires an LLM.",
            ),
        )

    llm_manager = resolve_llm_manager(config=config, formatter=formatter)
    if llm_manager is None:
        raise AutoDocCLIExitError(
            exit_code=2,
            errors=(
                "AutoDoc cleanup requires an LLM provider, but none is configured. "
                "Configure `llm` in your config/environment, or run with --assets-only "
                "to update UI assets without regenerating topic pages.",
            ),
        )

    cleanup_config = CleanupConfig(
        mode=cleanup_mode,
        batch_size=max(1, int(getattr(args, "cleanup_batch_size", 4))),
        max_completion_tokens=max(512, int(getattr(args, "cleanup_max_tokens", 4096))),
        audience=getattr(args, "audience", "balanced"),
    )
    return cleanup_config, llm_manager
