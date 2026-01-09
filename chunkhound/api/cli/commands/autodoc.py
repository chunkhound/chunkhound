"""AutoDoc site generator command module."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from chunkhound.api.cli.utils.rich_output import RichOutputFormatter
from chunkhound.autodoc.docsite import write_astro_assets_only
from chunkhound.core.config.config import Config

from . import autodoc_autorun as autorun
from . import autodoc_cleanup as cleanup
from . import autodoc_generate as generate
from . import autodoc_git as git_utils
from .autodoc_errors import AutoDocCLIExitError


def _render_exit(formatter: RichOutputFormatter, exc: AutoDocCLIExitError) -> None:
    for message in exc.infos:
        formatter.info(message)
    for message in exc.warnings:
        formatter.warning(message)
    for message in exc.errors:
        formatter.error(message)


async def autodoc_command(args: argparse.Namespace, config: Config) -> None:
    """Generate an Astro docs site from AutoDoc outputs."""
    formatter = RichOutputFormatter(verbose=bool(getattr(args, "verbose", False)))
    output_dir = Path(getattr(args, "out_dir")).resolve()

    try:
        git_utils.maybe_warn_git_output_dir(output_dir, formatter)

        if bool(getattr(args, "assets_only", False)):
            if not output_dir.exists():
                raise AutoDocCLIExitError(
                    exit_code=1,
                    errors=(
                        "Output directory not found for --assets-only: "
                        f"{output_dir}. Run a full `chunkhound autodoc` first.",
                    ),
                )
            write_astro_assets_only(output_dir=output_dir)
            formatter.success("AutoDoc assets update complete.")
            formatter.info(f"Output directory: {output_dir}")
            return

        map_dir = await autorun.ensure_map_dir(
            args=args,
            config=config,
            formatter=formatter,
            output_dir=output_dir,
        )

        cleanup_config, llm_manager = cleanup.resolve_cleanup_config_and_llm_manager(
            args=args,
            config=config,
            formatter=formatter,
        )

        allow_delete_topics_dir = autorun.resolve_allow_delete_topics_dir(
            args=args,
            output_dir=output_dir,
        )

        inputs = generate.DocsiteGenerationInputs(
            map_dir=map_dir,
            output_dir=output_dir,
            llm_manager=llm_manager,
            cleanup_config=cleanup_config,
            allow_delete_topics_dir=allow_delete_topics_dir,
            index_patterns=getattr(args, "index_patterns", None),
            site_title=getattr(args, "site_title", None),
            site_tagline=getattr(args, "site_tagline", None),
        )

        result = await generate.generate_docsite_with_optional_autorun(
            args=args,
            config=config,
            formatter=formatter,
            inputs=inputs,
        )

    except AutoDocCLIExitError as exc:
        _render_exit(formatter, exc)
        sys.exit(exc.exit_code)
    except Exception as exc:  # noqa: BLE001
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
