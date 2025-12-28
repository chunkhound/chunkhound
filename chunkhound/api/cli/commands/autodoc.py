"""AutoDoc site generator command module."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from chunkhound.api.cli.utils.rich_output import RichOutputFormatter
from chunkhound.autodoc.docsite import (
    CleanupConfig,
    generate_docsite,
    write_astro_assets_only,
)
from chunkhound.core.config.config import Config
from chunkhound.core.config.llm_config import LLMConfig
from chunkhound.llm_manager import LLMManager
from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider


def _default_out_dir(input_dir: Path) -> Path:
    return input_dir / "autodoc"


def _nearest_existing_dir(path: Path) -> Path | None:
    current = path
    try:
        while True:
            if current.exists():
                return current if current.is_dir() else current.parent
            if current.parent == current:
                return None
            current = current.parent
    except OSError:
        return None


def _git_repo_root(start_dir: Path) -> Path | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(start_dir),
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    root = result.stdout.strip()
    if not root:
        return None
    return Path(root)


def _git_is_ignored(*, repo_root: Path, path: Path) -> bool:
    try:
        rel = path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return False
    try:
        result = subprocess.run(
            ["git", "check-ignore", "-q", str(rel)],
            cwd=str(repo_root),
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def _maybe_warn_git_output_dir(
    output_dir: Path, formatter: RichOutputFormatter
) -> None:
    start_dir = _nearest_existing_dir(output_dir)
    if start_dir is None:
        return

    repo_root = _git_repo_root(start_dir)
    if repo_root is None:
        return

    try:
        output_dir.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return

    if _git_is_ignored(repo_root=repo_root, path=output_dir):
        return

    formatter.warning(
        "Output directory appears to be inside a git repo and is not ignored; "
        "generated docs may show up in `git status`. Consider adding it to "
        ".gitignore or writing to a git-ignored directory."
    )


def _has_llm_env() -> bool:
    return any(key.startswith("CHUNKHOUND_LLM_") for key in os.environ)


def _codex_available() -> bool:
    return shutil.which("codex") is not None


def _is_interactive() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:
        return False


def _prompt_yes_no(question: str, *, default: bool = False) -> bool:
    if not _is_interactive():
        return False

    prompt = " [Y/n]: " if default else " [y/N]: "
    while True:
        try:
            answer = input(question + prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False


def _prompt_text(question: str, *, default: str | None = None) -> str | None:
    if not _is_interactive():
        return default

    suffix = f" (default: {default})" if default else ""
    while True:
        try:
            answer = input(f"{question}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            return default
        if not answer:
            return default
        return answer


def _prompt_choice(
    question: str,
    *,
    choices: tuple[str, ...],
    default: str,
) -> str:
    if not _is_interactive():
        return default

    choices_str = "/".join(choices)
    while True:
        answer = _prompt_text(
            f"{question} ({choices_str})",
            default=default,
        )
        resolved = (answer or "").strip().lower()
        if resolved in choices:
            return resolved


@dataclass(frozen=True)
class _AutoMapPlan:
    map_out_dir: Path
    map_scope: Path
    comprehensiveness: str


def _build_auto_map_plan(
    *,
    output_dir: Path,
    map_out_dir: Path | None = None,
    comprehensiveness: str | None = None,
) -> _AutoMapPlan:
    default_map_out_dir = output_dir.with_name(f"map_{output_dir.name}")
    map_scope = Path.cwd().resolve()
    return _AutoMapPlan(
        map_out_dir=map_out_dir or default_map_out_dir,
        map_scope=map_scope,
        comprehensiveness=comprehensiveness or "medium",
    )


async def _run_code_mapper_for_autodoc(
    *,
    config: Config,
    formatter: RichOutputFormatter,
    output_dir: Path,
    verbose: bool,
    config_path: Path | None,
    map_out_dir: Path | None,
    comprehensiveness: str | None,
) -> _AutoMapPlan:
    from argparse import Namespace

    from chunkhound.api.cli.commands.code_mapper import code_mapper_command

    plan = _build_auto_map_plan(
        output_dir=output_dir,
        map_out_dir=map_out_dir,
        comprehensiveness=comprehensiveness,
    )

    map_args = Namespace(
        command="map",
        verbose=verbose,
        debug=False,
        config=config_path,
        path=plan.map_scope,
        out=plan.map_out_dir,
        overview_only=False,
        comprehensiveness=plan.comprehensiveness,
        combined=False,
    )

    try:
        await code_mapper_command(map_args, config)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        formatter.error("Map generation failed; aborting AutoDoc.")
        sys.exit(code)

    return plan


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

    output_dir = Path(getattr(args, "out_dir")).resolve()

    map_in_arg = getattr(args, "map_in", None)
    map_dir: Path | None
    if map_in_arg is None:
        map_dir = None
    else:
        map_dir = Path(map_in_arg).resolve()
        if not map_dir.exists():
            formatter.error(f"Map outputs directory not found: {map_dir}")
            sys.exit(1)

    _maybe_warn_git_output_dir(output_dir, formatter)

    if bool(getattr(args, "assets_only", False)):
        if not output_dir.exists():
            formatter.error(
                "Output directory not found for --assets-only: "
                f"{output_dir}. Run a full `chunkhound autodoc` first."
            )
            sys.exit(1)
        write_astro_assets_only(output_dir=output_dir)
        formatter.success("AutoDoc assets update complete.")
        formatter.info(f"Output directory: {output_dir}")
        return

    if map_dir is None:
        if not _prompt_yes_no(
            "No `map-in` provided. Generate the codemap first by running "
            "`chunkhound map`, then continue with AutoDoc?",
            default=False,
        ):
            formatter.error(
                "Missing required input: map-in (Code Mapper outputs directory)."
            )
            sys.exit(2)

        map_out_dir_arg = getattr(args, "map_out_dir", None)
        map_comprehensiveness_arg = getattr(args, "map_comprehensiveness", None)

        default_plan = _build_auto_map_plan(output_dir=output_dir)
        map_out_dir_hint = (
            Path(map_out_dir_arg).expanduser()
            if map_out_dir_arg is not None
            else default_plan.map_out_dir
        )

        map_out_dir = Path(map_out_dir_arg).expanduser() if map_out_dir_arg else None
        if map_out_dir is None:
            raw = _prompt_text(
                "Where should Code Mapper write its outputs",
                default=str(map_out_dir_hint),
            )
            map_out_dir = Path(raw).expanduser() if raw else map_out_dir_hint
        if not map_out_dir.is_absolute():
            map_out_dir = (Path.cwd() / map_out_dir).resolve()

        comprehensiveness = (
            map_comprehensiveness_arg
            if isinstance(map_comprehensiveness_arg, str)
            else None
        )
        if comprehensiveness is None:
            comprehensiveness = _prompt_choice(
                "Code Mapper comprehensiveness",
                choices=("minimal", "low", "medium", "high", "ultra"),
                default="medium",
            )

        formatter.info(f"Generating maps via Code Mapper: {map_out_dir}")
        plan = await _run_code_mapper_for_autodoc(
            config=config,
            formatter=formatter,
            output_dir=output_dir,
            verbose=getattr(args, "verbose", False),
            config_path=getattr(args, "config", None),
            map_out_dir=map_out_dir,
            comprehensiveness=comprehensiveness,
        )
        map_dir = plan.map_out_dir

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
        taint=getattr(args, "taint", "balanced"),
    )

    index_patterns = getattr(args, "index_patterns", None)

    try:
        result = await generate_docsite(
            input_dir=map_dir,
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
        formatter.warning(str(exc))

        if not _prompt_yes_no(
            "Generate the codemap first by running `chunkhound map`, "
            "then retry AutoDoc?",
            default=False,
        ):
            formatter.error(str(exc))
            sys.exit(1)

        map_out_dir_arg = getattr(args, "map_out_dir", None)
        map_comprehensiveness_arg = getattr(args, "map_comprehensiveness", None)

        default_plan = _build_auto_map_plan(output_dir=output_dir)
        map_out_dir_hint = (
            Path(map_out_dir_arg).expanduser()
            if map_out_dir_arg is not None
            else default_plan.map_out_dir
        )

        map_out_dir = Path(map_out_dir_arg).expanduser() if map_out_dir_arg else None
        if map_out_dir is None:
            raw = _prompt_text(
                "Where should Code Mapper write its outputs",
                default=str(map_out_dir_hint),
            )
            map_out_dir = Path(raw).expanduser() if raw else map_out_dir_hint
        if not map_out_dir.is_absolute():
            map_out_dir = (Path.cwd() / map_out_dir).resolve()

        comprehensiveness = (
            map_comprehensiveness_arg
            if isinstance(map_comprehensiveness_arg, str)
            else None
        )
        if comprehensiveness is None:
            comprehensiveness = _prompt_choice(
                "Code Mapper comprehensiveness",
                choices=("minimal", "low", "medium", "high", "ultra"),
                default="medium",
            )

        formatter.info(f"Generating maps via Code Mapper: {map_out_dir}")
        plan = await _run_code_mapper_for_autodoc(
            config=config,
            formatter=formatter,
            output_dir=output_dir,
            verbose=getattr(args, "verbose", False),
            config_path=getattr(args, "config", None),
            map_out_dir=map_out_dir,
            comprehensiveness=comprehensiveness,
        )

        try:
            result = await generate_docsite(
                input_dir=plan.map_out_dir,
                output_dir=output_dir,
                llm_manager=llm_manager,
                cleanup_config=cleanup_config,
                site_title=getattr(args, "site_title", None),
                site_tagline=getattr(args, "site_tagline", None),
                index_patterns=index_patterns,
                log_info=formatter.info,
                log_warning=formatter.warning,
            )
        except FileNotFoundError as exc2:
            formatter.error(str(exc2))
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
