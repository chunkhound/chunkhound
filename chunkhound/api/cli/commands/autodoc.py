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
    audience: str


@dataclass(frozen=True)
class _AutoMapOptions:
    map_out_dir: Path
    comprehensiveness: str
    audience: str
    map_context: Path | None


def _confirm_autorun_and_validate_prereqs(
    *,
    config: Config,
    config_path: Path | None,
    formatter: RichOutputFormatter,
    question: str,
    decline_error: str,
    decline_exit_code: int,
    prereq_failure_exit_code: int = 1,
    default: bool = False,
) -> None:
    preflight_ok, missing, _details = _code_mapper_autorun_prereq_summary(
        config=config,
        config_path=config_path,
    )
    warning_suffix = ""
    if not preflight_ok:
        warning_suffix = (
            "\n\n"
            "Note: Code Mapper prerequisites appear missing "
            f"({', '.join(missing)})."
        )

    if not _prompt_yes_no(
        f"{question}{warning_suffix}",
        default=default,
    ):
        formatter.error(decline_error)
        sys.exit(decline_exit_code)

    preflight_ok, _missing, details = _code_mapper_autorun_prereq_summary(
        config=config,
        config_path=config_path,
    )
    if not preflight_ok:
        _exit_autodoc_autorun_prereq_failure(
            formatter=formatter,
            details=details,
            exit_code=prereq_failure_exit_code,
        )


def _build_auto_map_plan(
    *,
    output_dir: Path,
    map_out_dir: Path | None = None,
    comprehensiveness: str | None = None,
    audience: str | None = None,
) -> _AutoMapPlan:
    default_map_out_dir = output_dir.with_name(f"map_{output_dir.name}")
    map_scope = Path.cwd().resolve()
    return _AutoMapPlan(
        map_out_dir=map_out_dir or default_map_out_dir,
        map_scope=map_scope,
        comprehensiveness=comprehensiveness or "medium",
        audience=audience or "balanced",
    )


def _resolve_auto_map_options(*, args, output_dir: Path) -> _AutoMapOptions:
    map_out_dir_arg = getattr(args, "map_out_dir", None)
    map_comprehensiveness_arg = getattr(args, "map_comprehensiveness", None)
    map_context_arg = getattr(args, "map_context", None)

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

    map_audience_arg = getattr(args, "map_audience", None)
    map_audience = map_audience_arg if isinstance(map_audience_arg, str) else None
    if map_audience is None:
        default_audience = getattr(args, "audience", "balanced")
        map_audience = _prompt_choice(
            "Code Mapper audience (map generation)",
            choices=("technical", "balanced", "end-user"),
            default=default_audience,
        )

    map_context: Path | None = (
        Path(map_context_arg).expanduser() if map_context_arg is not None else None
    )
    if map_context is None:
        raw = _prompt_text(
            "Optional Code Mapper context file (--map-context, leave blank for none)",
            default=None,
        )
        map_context = Path(raw).expanduser() if raw else None

    return _AutoMapOptions(
        map_out_dir=map_out_dir,
        comprehensiveness=comprehensiveness,
        audience=map_audience,
        map_context=map_context,
    )


def _effective_config_for_code_mapper_autorun(
    *,
    config: Config,
    config_path: Path | None,
) -> Config:
    from argparse import Namespace

    from chunkhound.api.cli.utils import apply_code_mapper_workspace_overrides

    effective = config.model_copy(deep=True)
    args = Namespace(
        config=config_path,
        db=None,
        database_path=None,
    )
    apply_code_mapper_workspace_overrides(config=effective, args=args)
    return effective


def _code_mapper_autorun_prereq_summary(
    *,
    config: Config,
    config_path: Path | None,
) -> tuple[bool, list[str], list[str]]:
    """Return (ok, missing_labels, detail_lines) for Code Mapper auto-run."""
    from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory

    effective = _effective_config_for_code_mapper_autorun(
        config=config,
        config_path=config_path,
    )

    missing: list[str] = []
    details: list[str] = []

    db_path: Path | None = None
    try:
        db_path = effective.database.get_db_path()
    except ValueError:
        missing.append("database")
        details.append("- Database path is not configured.")
    else:
        if not db_path.exists():
            missing.append("database")
            details.append(f"- Database not found at: {db_path}")

    if effective.embedding is None:
        missing.append("embeddings")
        details.append("- Embedding provider is not configured.")
    else:
        try:
            provider = EmbeddingProviderFactory.create_provider(effective.embedding)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            missing.append("embeddings")
            details.append(f"- Embedding provider setup failed: {exc}")
        else:
            supports_reranking = False
            try:
                if hasattr(provider, "supports_reranking") and callable(
                    provider.supports_reranking
                ):
                    supports_reranking = bool(provider.supports_reranking())
            except Exception:
                supports_reranking = False
            if not supports_reranking:
                missing.append("reranking")
                details.append(
                    "- Embedding provider does not support reranking with current "
                    "config (configure reranking; typically `embedding.rerank_model`)."
                )

    if effective.llm is None:
        missing.append("llm")
        details.append("- LLM provider is not configured.")
    elif not effective.llm.is_provider_configured():
        missing.append("llm")
        details.append("- LLM provider is not fully configured.")

    # De-dup while preserving order
    missing_dedup: list[str] = []
    for item in missing:
        if item not in missing_dedup:
            missing_dedup.append(item)

    return not missing_dedup, missing_dedup, details


def _exit_autodoc_autorun_prereq_failure(
    *,
    formatter: RichOutputFormatter,
    details: list[str],
    exit_code: int,
) -> None:
    formatter.error(
        "AutoDoc can auto-run Code Mapper, but required prerequisites are missing."
    )
    for line in details:
        formatter.error(line)
    formatter.info("To fix:")
    formatter.info("- Run `chunkhound index <directory>` to create the database.")
    formatter.info(
        "- Configure embeddings with reranking support "
        "(e.g. set `embedding.rerank_model`)."
    )
    formatter.info("- Configure an LLM provider (e.g. `CHUNKHOUND_LLM_API_KEY`).")
    sys.exit(exit_code)


async def _run_code_mapper_for_autodoc(
    *,
    config: Config,
    formatter: RichOutputFormatter,
    output_dir: Path,
    verbose: bool,
    config_path: Path | None,
    map_out_dir: Path | None,
    map_context: Path | None,
    comprehensiveness: str | None,
    audience: str | None,
) -> _AutoMapPlan:
    from argparse import Namespace

    from chunkhound.api.cli.commands.code_mapper import code_mapper_command

    plan = _build_auto_map_plan(
        output_dir=output_dir,
        map_out_dir=map_out_dir,
        comprehensiveness=comprehensiveness,
        audience=audience,
    )

    map_args = Namespace(
        command="map",
        verbose=verbose,
        debug=False,
        config=config_path,
        path=plan.map_scope,
        out=plan.map_out_dir,
        context=map_context,
        overview_only=False,
        comprehensiveness=plan.comprehensiveness,
        combined=False,
        audience=plan.audience,
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

    if llm_config is None:
        formatter.warning(
            "No LLM provider configured; AutoDoc cleanup cannot run."
        )
        return None

    if not llm_config.is_provider_configured():
        formatter.warning(
            "LLM provider is not fully configured; AutoDoc cleanup cannot run."
        )
        return None

    try:
        utility_config, synthesis_config = _build_cleanup_provider_configs(llm_config)
        provider = synthesis_config.get("provider", "unknown")
        model = synthesis_config.get("model", "unknown")
        effort = synthesis_config.get("reasoning_effort")
        override_notes: list[str] = []
        if llm_config.autodoc_cleanup_provider:
            override_notes.append("cleanup provider")
        if llm_config.autodoc_cleanup_model:
            override_notes.append("cleanup model")
        if llm_config.autodoc_cleanup_reasoning_effort:
            override_notes.append("cleanup reasoning effort")
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


async def _call_generate_docsite(
    *,
    formatter: RichOutputFormatter,
    input_dir: Path,
    output_dir: Path,
    llm_manager: LLMManager | None,
    cleanup_config: CleanupConfig,
    allow_delete_topics_dir: bool,
    index_patterns: list[str] | None,
    site_title: str | None,
    site_tagline: str | None,
):
    return await generate_docsite(
        input_dir=input_dir,
        output_dir=output_dir,
        llm_manager=llm_manager,
        cleanup_config=cleanup_config,
        site_title=site_title,
        site_tagline=site_tagline,
        allow_delete_topics_dir=allow_delete_topics_dir,
        index_patterns=index_patterns,
        log_info=formatter.info,
        log_warning=formatter.warning,
    )


async def _autorun_code_mapper_for_autodoc(
    *,
    args,
    config: Config,
    formatter: RichOutputFormatter,
    output_dir: Path,
    question: str,
    decline_error: str,
    decline_exit_code: int,
) -> Path:
    config_path = getattr(args, "config", None)
    _confirm_autorun_and_validate_prereqs(
        config=config,
        config_path=config_path,
        formatter=formatter,
        question=question,
        decline_error=decline_error,
        decline_exit_code=decline_exit_code,
    )

    map_options = _resolve_auto_map_options(args=args, output_dir=output_dir)
    formatter.info(f"Generating maps via Code Mapper: {map_options.map_out_dir}")
    plan = await _run_code_mapper_for_autodoc(
        config=config,
        formatter=formatter,
        output_dir=output_dir,
        verbose=getattr(args, "verbose", False),
        config_path=config_path,
        map_out_dir=map_options.map_out_dir,
        map_context=map_options.map_context,
        comprehensiveness=map_options.comprehensiveness,
        audience=map_options.audience,
    )
    return plan.map_out_dir


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
        if not _is_interactive():
            formatter.error(
                "Missing required input: map-in (Code Mapper outputs directory). "
                "Non-interactive mode cannot prompt to auto-generate maps."
            )
            sys.exit(2)

        map_dir = await _autorun_code_mapper_for_autodoc(
            args=args,
            config=config,
            formatter=formatter,
            output_dir=output_dir,
            question=(
                "No `map-in` provided. Generate the codemap first by running "
                "`chunkhound map`, then continue with AutoDoc?"
            ),
            decline_error=(
                "Missing required input: map-in (Code Mapper outputs directory)."
            ),
            decline_exit_code=2,
        )

    cleanup_mode = getattr(args, "cleanup_mode", "llm")
    if cleanup_mode != "llm":
        formatter.error(
            "Unsupported AutoDoc cleanup mode: "
            f"{cleanup_mode!r}. AutoDoc cleanup now requires an LLM."
        )
        sys.exit(2)
    llm_manager = _resolve_llm_manager(
        config=config,
        cleanup_mode=cleanup_mode,
        formatter=formatter,
    )
    if llm_manager is None:
        formatter.error(
            "AutoDoc cleanup requires an LLM provider, but none is configured. "
            "Configure `llm` in your config/environment, or run with --assets-only "
            "to update UI assets without regenerating topic pages."
        )
        sys.exit(2)

    cleanup_config = CleanupConfig(
        mode=cleanup_mode,
        batch_size=max(1, int(getattr(args, "cleanup_batch_size", 4))),
        max_completion_tokens=max(512, int(getattr(args, "cleanup_max_tokens", 4096))),
        audience=getattr(args, "audience", "balanced"),
    )

    index_patterns = getattr(args, "index_patterns", None)
    site_title = getattr(args, "site_title", None)
    site_tagline = getattr(args, "site_tagline", None)

    allow_delete_topics_dir = False
    topics_dir = output_dir / "src" / "pages" / "topics"
    if topics_dir.exists():
        force = bool(getattr(args, "force", False))
        if force:
            allow_delete_topics_dir = True
        elif not _is_interactive():
            formatter.error(
                "Output directory already contains topic pages at "
                f"{topics_dir}. Re-generating will delete them. "
                "Re-run with `--force` to allow deletion."
            )
            sys.exit(2)
        else:
            if not _prompt_yes_no(
                "Output directory already contains generated topic pages at "
                f"{topics_dir}. Delete and re-generate them?",
                default=False,
            ):
                formatter.error("Aborted.")
                sys.exit(2)
            allow_delete_topics_dir = True

    try:
        result = await _call_generate_docsite(
            formatter=formatter,
            input_dir=map_dir,
            output_dir=output_dir,
            llm_manager=llm_manager,
            cleanup_config=cleanup_config,
            allow_delete_topics_dir=allow_delete_topics_dir,
            index_patterns=index_patterns,
            site_title=site_title,
            site_tagline=site_tagline,
        )
    except FileNotFoundError as exc:
        formatter.warning(str(exc))

        if not _is_interactive():
            formatter.error(
                "AutoDoc index not found in map-in directory, and non-interactive "
                "mode cannot prompt to auto-generate maps. Run `chunkhound map` "
                "first (then re-run `chunkhound autodoc` with map-in), or ensure "
                "the map-in folder contains a `*_code_mapper_index.md`."
            )
            sys.exit(1)

        map_dir = await _autorun_code_mapper_for_autodoc(
            args=args,
            config=config,
            formatter=formatter,
            output_dir=output_dir,
            question=(
                "Generate the codemap first by running `chunkhound map`, then retry "
                "AutoDoc?"
            ),
            decline_error=str(exc),
            decline_exit_code=1,
        )

        try:
            result = await _call_generate_docsite(
                formatter=formatter,
                input_dir=map_dir,
                output_dir=output_dir,
                llm_manager=llm_manager,
                cleanup_config=cleanup_config,
                allow_delete_topics_dir=allow_delete_topics_dir,
                index_patterns=index_patterns,
                site_title=site_title,
                site_tagline=site_tagline,
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
