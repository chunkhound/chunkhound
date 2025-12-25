"""AutoDoc site generator command module."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from loguru import logger

from chunkhound.api.cli.utils.rich_output import RichOutputFormatter
from chunkhound.autodoc.docsite import CleanupConfig, generate_docsite
from chunkhound.core.config.config import Config
from chunkhound.core.config.llm_config import LLMConfig
from chunkhound.llm_manager import LLMManager
from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider


def _default_out_dir(input_dir: Path) -> Path:
    return input_dir / "autodoc"


def _is_gitignored_path(path: Path) -> bool:
    return ".gitignored" in path.parts


def _has_llm_env() -> bool:
    return any(key.startswith("CHUNKHOUND_LLM_") for key in os.environ)


def _codex_available() -> bool:
    return shutil.which("codex") is not None


def _build_cleanup_provider_configs(
    llm_config: LLMConfig,
) -> tuple[dict[str, object], dict[str, object]]:
    utility_config, synthesis_config = llm_config.get_provider_configs()

    assembly_provider = llm_config.assembly_provider
    assembly_model = llm_config.assembly_model or llm_config.assembly_synthesis_model
    assembly_effort = llm_config.assembly_reasoning_effort

    if assembly_provider:
        synthesis_config = synthesis_config.copy()
        synthesis_config["provider"] = assembly_provider

    if assembly_model:
        synthesis_config = synthesis_config.copy()
        synthesis_config["model"] = assembly_model

    provider = synthesis_config.get("provider")
    if (
        assembly_effort
        and isinstance(provider, str)
        and provider in ("codex-cli", "openai")
    ):
        synthesis_config = synthesis_config.copy()
        synthesis_config["reasoning_effort"] = assembly_effort

    return utility_config, synthesis_config


def _resolve_llm_manager(
    *,
    config: Config,
    cleanup_mode: str,
    formatter: RichOutputFormatter,
) -> LLMManager | None:
    if cleanup_mode != "llm":
        return None

    llm_config = config.llm
    if llm_config is None and _has_llm_env():
        try:
            llm_config = LLMConfig()
        except Exception as exc:
            formatter.warning(f"Failed to load LLM config from environment: {exc}")
            return None

    if llm_config is None and _codex_available():
        formatter.info("Using codex-cli provider for cleanup.")
        llm_config = LLMConfig(
            provider="codex-cli",
            utility_provider="codex-cli",
            synthesis_provider="codex-cli",
        )

    if llm_config is None:
        formatter.warning(
            "No LLM provider configured; cleanup will fall back to minimal."
        )
        return None

    if not llm_config.is_provider_configured():
        formatter.warning(
            "LLM provider is not fully configured; cleanup will fall back to minimal."
        )
        return None

    try:
        utility_config, synthesis_config = _build_cleanup_provider_configs(llm_config)
        provider = synthesis_config.get("provider", "unknown")
        model = synthesis_config.get("model", "unknown")
        effort = synthesis_config.get("reasoning_effort")
        override_notes: list[str] = []
        if llm_config.assembly_provider:
            override_notes.append("assembly provider")
        if llm_config.assembly_model or llm_config.assembly_synthesis_model:
            override_notes.append("assembly model")
        if llm_config.assembly_reasoning_effort:
            override_notes.append("assembly reasoning effort")
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
        elif effort:
            formatter.info(
                f"Cleanup model selection: provider={provider}, model={model}, "
                f"reasoning_effort={effort}{suffix}"
            )
        else:
            formatter.info(
                f"Cleanup model selection: provider={provider}, model={model}{suffix}"
            )
        return LLMManager(utility_config, synthesis_config)
    except Exception as exc:
        formatter.warning(f"Failed to configure LLM provider: {exc}")
        logger.exception("LLM configuration error")
        return None


async def autodoc_command(args, config: Config) -> None:
    """Generate an Astro docs site from AutoDoc outputs."""
    formatter = RichOutputFormatter(verbose=getattr(args, "verbose", False))

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        formatter.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    out_dir_arg = getattr(args, "out_dir", None)
    if out_dir_arg:
        output_dir = Path(out_dir_arg).resolve()
    else:
        output_dir = _default_out_dir(input_dir)

    if not _is_gitignored_path(output_dir):
        formatter.warning(
            "Output directory is not inside .gitignored/. "
            "Generated docs may be picked up by git."
        )

    cleanup_mode = getattr(args, "cleanup_mode", "llm")
    llm_manager = _resolve_llm_manager(
        config=config,
        cleanup_mode=cleanup_mode,
        formatter=formatter,
    )

    cleanup_config = CleanupConfig(
        mode=cleanup_mode,
        batch_size=max(1, int(getattr(args, "cleanup_batch_size", 4))),
        max_completion_tokens=max(512, int(getattr(args, "cleanup_max_tokens", 4096))),
    )

    index_patterns = getattr(args, "index_patterns", None)

    try:
        result = await generate_docsite(
            input_dir=input_dir,
            output_dir=output_dir,
            llm_manager=llm_manager,
            cleanup_config=cleanup_config,
            site_title=getattr(args, "site_title", None),
            site_tagline=getattr(args, "site_tagline", None),
            index_patterns=index_patterns,
            log_info=formatter.info,
            log_warning=formatter.warning,
        )
    except FileNotFoundError as exc:
        formatter.error(str(exc))
        sys.exit(1)
    except Exception as exc:
        formatter.error(f"AutoDoc generation failed: {exc}")
        logger.exception("AutoDoc generation failed")
        sys.exit(1)

    formatter.success("AutoDoc generation complete.")
    formatter.info(f"Output directory: {result.output_dir}")
    formatter.info(f"Pages generated: {len(result.pages)}")
    if result.missing_topics:
        formatter.warning(
            "Missing topic files referenced in index: "
            + ", ".join(result.missing_topics)
        )
