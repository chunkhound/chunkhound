from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

from chunkhound.api.cli.utils.rich_output import RichOutputFormatter
from chunkhound.autodoc.docsite import CleanupConfig, generate_docsite
from chunkhound.autodoc.models import DocsiteResult
from chunkhound.core.config.config import Config
from chunkhound.llm_manager import LLMManager

from . import autodoc_autorun as autorun
from . import autodoc_prompts as prompts
from .autodoc_errors import AutoDocCLIExitError


@dataclass(frozen=True)
class DocsiteGenerationInputs:
    map_dir: Path
    output_dir: Path
    llm_manager: LLMManager
    cleanup_config: CleanupConfig
    allow_delete_topics_dir: bool
    index_patterns: list[str] | None
    site_title: str | None
    site_tagline: str | None


async def call_generate_docsite(
    *,
    formatter: RichOutputFormatter,
    input_dir: Path,
    output_dir: Path,
    llm_manager: LLMManager,
    cleanup_config: CleanupConfig,
    allow_delete_topics_dir: bool,
    index_patterns: list[str] | None,
    site_title: str | None,
    site_tagline: str | None,
) -> DocsiteResult:
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


async def generate_docsite_with_optional_autorun(
    *,
    args: Namespace,
    config: Config,
    formatter: RichOutputFormatter,
    inputs: DocsiteGenerationInputs,
) -> DocsiteResult:
    current_map_dir = inputs.map_dir
    for attempt in range(2):
        try:
            return await call_generate_docsite(
                formatter=formatter,
                input_dir=current_map_dir,
                output_dir=inputs.output_dir,
                llm_manager=inputs.llm_manager,
                cleanup_config=inputs.cleanup_config,
                allow_delete_topics_dir=inputs.allow_delete_topics_dir,
                index_patterns=inputs.index_patterns,
                site_title=inputs.site_title,
                site_tagline=inputs.site_tagline,
            )
        except FileNotFoundError as exc:
            if attempt == 1:
                raise AutoDocCLIExitError(exit_code=1, errors=(str(exc),))

            formatter.warning(str(exc))
            if not prompts.is_interactive():
                raise AutoDocCLIExitError(
                    exit_code=1,
                    errors=(
                        "AutoDoc index not found in map-in directory, and "
                        "non-interactive mode cannot prompt to auto-generate maps. "
                        "Run `chunkhound map` first (then re-run `chunkhound autodoc` "
                        "with map-in), or ensure the map-in folder contains a "
                        "`*_code_mapper_index.md`.",
                    ),
                )

            current_map_dir = await autorun.autorun_code_mapper_for_autodoc(
                args=args,
                config=config,
                formatter=formatter,
                output_dir=inputs.output_dir,
                question=(
                    "Generate the codemap first by running `chunkhound map`, "
                    "then retry AutoDoc?"
                ),
                decline_error=str(exc),
                decline_exit_code=1,
            )

    raise RuntimeError("AutoDoc autorun retry loop exhausted without returning.")
