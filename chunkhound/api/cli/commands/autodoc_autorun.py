from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

from chunkhound.api.cli.utils.rich_output import RichOutputFormatter
from chunkhound.core.config.config import Config

from . import autodoc_prompts as prompts
from .autodoc_errors import AutoDocCLIExitError


@dataclass(frozen=True)
class AutoMapPlan:
    map_out_dir: Path
    map_scope: Path
    comprehensiveness: str
    audience: str


@dataclass(frozen=True)
class AutoMapOptions:
    map_out_dir: Path
    comprehensiveness: str
    audience: str
    map_context: Path | None


def _dedup_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _build_auto_map_plan(
    *,
    output_dir: Path,
    map_out_dir: Path | None = None,
    comprehensiveness: str | None = None,
    audience: str | None = None,
) -> AutoMapPlan:
    default_map_out_dir = output_dir.with_name(f"map_{output_dir.name}")
    map_scope = Path.cwd().resolve()
    return AutoMapPlan(
        map_out_dir=map_out_dir or default_map_out_dir,
        map_scope=map_scope,
        comprehensiveness=comprehensiveness or "medium",
        audience=audience or "balanced",
    )


def _resolve_map_out_dir(*, args: Namespace, output_dir: Path) -> Path:
    map_out_dir_arg = getattr(args, "map_out_dir", None)
    default_plan = _build_auto_map_plan(output_dir=output_dir)
    map_out_dir_hint = (
        Path(map_out_dir_arg).expanduser()
        if map_out_dir_arg is not None
        else default_plan.map_out_dir
    )

    map_out_dir = Path(map_out_dir_arg).expanduser() if map_out_dir_arg else None
    if map_out_dir is None:
        raw = prompts.prompt_text(
            "Where should Code Mapper write its outputs",
            default=str(map_out_dir_hint),
        )
        map_out_dir = Path(raw).expanduser() if raw else map_out_dir_hint
    if not map_out_dir.is_absolute():
        map_out_dir = (Path.cwd() / map_out_dir).resolve()
    return map_out_dir


def _resolve_map_comprehensiveness(*, args: Namespace) -> str:
    map_comprehensiveness_arg = getattr(args, "map_comprehensiveness", None)
    comprehensiveness = (
        map_comprehensiveness_arg
        if isinstance(map_comprehensiveness_arg, str)
        else None
    )
    if comprehensiveness is None:
        return prompts.prompt_choice(
            "Code Mapper comprehensiveness",
            choices=("minimal", "low", "medium", "high", "ultra"),
            default="medium",
        )
    return comprehensiveness


def _resolve_map_audience(*, args: Namespace) -> str:
    map_audience_arg = getattr(args, "map_audience", None)
    map_audience = map_audience_arg if isinstance(map_audience_arg, str) else None
    if map_audience is None:
        default_audience = getattr(args, "audience", "balanced")
        return prompts.prompt_choice(
            "Code Mapper audience (map generation)",
            choices=("technical", "balanced", "end-user"),
            default=default_audience,
        )
    return map_audience


def _resolve_map_context(*, args: Namespace) -> Path | None:
    map_context_arg = getattr(args, "map_context", None)
    map_context: Path | None = (
        Path(map_context_arg).expanduser() if map_context_arg is not None else None
    )
    if map_context is None:
        raw = prompts.prompt_text(
            "Optional Code Mapper context file (--map-context, leave blank for none)",
            default=None,
        )
        map_context = Path(raw).expanduser() if raw else None
    return map_context


def resolve_auto_map_options(*, args: Namespace, output_dir: Path) -> AutoMapOptions:
    return AutoMapOptions(
        map_out_dir=_resolve_map_out_dir(args=args, output_dir=output_dir),
        comprehensiveness=_resolve_map_comprehensiveness(args=args),
        audience=_resolve_map_audience(args=args),
        map_context=_resolve_map_context(args=args),
    )


def _effective_config_for_code_mapper_autorun(
    *,
    config: Config,
    config_path: Path | None,
) -> Config:
    from chunkhound.api.cli.utils import apply_code_mapper_workspace_overrides

    effective = config.model_copy(deep=True)
    args = Namespace(
        config=config_path,
        db=None,
        database_path=None,
    )
    apply_code_mapper_workspace_overrides(config=effective, args=args)
    return effective


def _code_mapper_autorun_database_prereqs(
    effective: Config,
) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    details: list[str] = []
    try:
        db_path = effective.database.get_db_path()
    except ValueError:
        missing.append("database")
        details.append("- Database path is not configured.")
        return missing, details

    if not db_path.exists():
        missing.append("database")
        details.append(f"- Database not found at: {db_path}")

    return missing, details


def _provider_supports_reranking(provider: object) -> bool:
    try:
        supports = getattr(provider, "supports_reranking", None)
        if callable(supports):
            return bool(supports())
    except Exception:
        return False
    return False


def _code_mapper_autorun_embedding_prereqs(
    effective: Config,
) -> tuple[list[str], list[str]]:
    from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory

    missing: list[str] = []
    details: list[str] = []

    if effective.embedding is None:
        missing.append("embeddings")
        details.append("- Embedding provider is not configured.")
        return missing, details

    try:
        provider = EmbeddingProviderFactory.create_provider(effective.embedding)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        missing.append("embeddings")
        details.append(f"- Embedding provider setup failed: {exc}")
        return missing, details

    if not _provider_supports_reranking(provider):
        missing.append("reranking")
        details.append(
            "- Embedding provider does not support reranking with current "
            "config (configure reranking; typically `embedding.rerank_model`)."
        )

    return missing, details


def _code_mapper_autorun_llm_prereqs(effective: Config) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    details: list[str] = []

    if effective.llm is None:
        missing.append("llm")
        details.append("- LLM provider is not configured.")
        return missing, details

    if not effective.llm.is_provider_configured():
        missing.append("llm")
        details.append("- LLM provider is not fully configured.")

    return missing, details


def code_mapper_autorun_prereq_summary(
    *,
    config: Config,
    config_path: Path | None,
) -> tuple[bool, list[str], list[str]]:
    """Return (ok, missing_labels, detail_lines) for Code Mapper auto-run."""
    effective = _effective_config_for_code_mapper_autorun(
        config=config,
        config_path=config_path,
    )

    missing: list[str] = []
    details: list[str] = []

    db_missing, db_details = _code_mapper_autorun_database_prereqs(effective)
    missing.extend(db_missing)
    details.extend(db_details)

    embed_missing, embed_details = _code_mapper_autorun_embedding_prereqs(effective)
    missing.extend(embed_missing)
    details.extend(embed_details)

    llm_missing, llm_details = _code_mapper_autorun_llm_prereqs(effective)
    missing.extend(llm_missing)
    details.extend(llm_details)

    missing_dedup = _dedup_preserve_order(missing)
    return not missing_dedup, missing_dedup, details


def _autorun_prereq_failure_exit(*, details: list[str], exit_code: int) -> None:
    raise AutoDocCLIExitError(
        exit_code=exit_code,
        errors=(
            "AutoDoc can auto-run Code Mapper, but required prerequisites are missing.",
            *tuple(details),
        ),
        infos=(
            "To fix:",
            "- Run `chunkhound index <directory>` to create the database.",
            (
                "- Configure embeddings with reranking support "
                "(e.g. set `embedding.rerank_model`)."
            ),
            "- Configure an LLM provider (e.g. `CHUNKHOUND_LLM_API_KEY`).",
        ),
    )


def confirm_autorun_and_validate_prereqs(
    *,
    config: Config,
    config_path: Path | None,
    question: str,
    decline_error: str,
    decline_exit_code: int,
    prereq_failure_exit_code: int = 1,
    default: bool = False,
) -> None:
    preflight_ok, missing, _details = code_mapper_autorun_prereq_summary(
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

    if not prompts.prompt_yes_no(
        f"{question}{warning_suffix}",
        default=default,
    ):
        raise AutoDocCLIExitError(exit_code=decline_exit_code, errors=(decline_error,))

    preflight_ok, _missing, details = code_mapper_autorun_prereq_summary(
        config=config,
        config_path=config_path,
    )
    if not preflight_ok:
        _autorun_prereq_failure_exit(
            details=details, exit_code=prereq_failure_exit_code
        )


async def run_code_mapper_for_autodoc(
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
) -> AutoMapPlan:
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
        raise AutoDocCLIExitError(
            exit_code=code,
            errors=("Map generation failed; aborting AutoDoc.",),
        )

    return plan


async def autorun_code_mapper_for_autodoc(
    *,
    args: Namespace,
    config: Config,
    formatter: RichOutputFormatter,
    output_dir: Path,
    question: str,
    decline_error: str,
    decline_exit_code: int,
) -> Path:
    config_path = getattr(args, "config", None)
    confirm_autorun_and_validate_prereqs(
        config=config,
        config_path=config_path,
        question=question,
        decline_error=decline_error,
        decline_exit_code=decline_exit_code,
    )

    map_options = resolve_auto_map_options(args=args, output_dir=output_dir)
    formatter.info(f"Generating maps via Code Mapper: {map_options.map_out_dir}")
    plan = await run_code_mapper_for_autodoc(
        config=config,
        formatter=formatter,
        output_dir=output_dir,
        verbose=bool(getattr(args, "verbose", False)),
        config_path=config_path,
        map_out_dir=map_options.map_out_dir,
        map_context=map_options.map_context,
        comprehensiveness=map_options.comprehensiveness,
        audience=map_options.audience,
    )
    return plan.map_out_dir


async def ensure_map_dir(
    *,
    args: Namespace,
    config: Config,
    formatter: RichOutputFormatter,
    output_dir: Path,
) -> Path:
    map_in_arg = getattr(args, "map_in", None)
    if map_in_arg is None:
        if not prompts.is_interactive():
            raise AutoDocCLIExitError(
                exit_code=2,
                errors=(
                    "Missing required input: map-in (Code Mapper outputs directory). "
                    "Non-interactive mode cannot prompt to auto-generate maps.",
                ),
            )

        return await autorun_code_mapper_for_autodoc(
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

    map_dir = Path(map_in_arg).resolve()
    if not map_dir.exists():
        raise AutoDocCLIExitError(
            exit_code=1,
            errors=(f"Map outputs directory not found: {map_dir}",),
        )
    return map_dir


def resolve_allow_delete_topics_dir(
    *,
    args: Namespace,
    output_dir: Path,
) -> bool:
    topics_dir = output_dir / "src" / "pages" / "topics"
    if not topics_dir.exists():
        return False

    force = bool(getattr(args, "force", False))
    if force:
        return True

    if not prompts.is_interactive():
        raise AutoDocCLIExitError(
            exit_code=2,
            errors=(
                "Output directory already contains topic pages at "
                f"{topics_dir}. Re-generating will delete them. "
                "Re-run with `--force` to allow deletion.",
            ),
        )

    if not prompts.prompt_yes_no(
        "Output directory already contains generated topic pages at "
        f"{topics_dir}. Delete and re-generate them?",
        default=False,
    ):
        raise AutoDocCLIExitError(exit_code=2, errors=("Aborted.",))

    return True
