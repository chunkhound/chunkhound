"""Websearch command for ChunkHound CLI."""

from __future__ import annotations

import argparse
import asyncio
import os
import shlex
import subprocess
import sys
import tempfile
import urllib.error
from pathlib import Path

from chunkhound.core.config.config import Config
from chunkhound.utils.websearch_core import (
    build_quickresearch_argv_core,
    fetch_and_save,
    search_multi,
    websearch_timeout,
)
from chunkhound.utils.websearch_expansion import expand_web_queries
from chunkhound.utils.websearch_postprocess import replace_paths_with_urls

from ..utils.provider_setup import setup_llm_manager
from ..utils.rich_output import RichOutputFormatter


def _build_quickresearch_argv(
    args: argparse.Namespace,
    tmpdir: Path,
    config: Config,
    query: str,
    previous_query: str | None,
) -> list[str]:
    """Build argv to invoke _quickresearch as a subprocess, forwarding relevant args.

    ``query`` is the raw user query — the LLM-normalized ``search_query`` is
    reserved for the DDG variants and footer, since lossy normalization would
    silently narrow the research prompt (only outright empty/whitespace/LLM
    failures fall back to raw). ``previous_query`` is pre-coerced by the caller.
    """
    from ..parsers.common_arguments import build_forwarded_argv
    from ..parsers.quickresearch_parser import add_quickresearch_subparser

    _tmp = argparse.ArgumentParser(add_help=False)
    qr_parser = add_quickresearch_subparser(_tmp.add_subparsers())

    cmd = build_quickresearch_argv_core(
        query, tmpdir, config,
        parent_pid=os.getpid(),
        previous_query=previous_query,
    )
    cmd.extend(build_forwarded_argv(
        qr_parser,
        args,
        # path_filter: defensive — forwarding would zero out results (flat tmpdir).
        # parent_pid: emitted explicitly above; skip to avoid double-forwarding.
        # previous_query: emitted explicitly above; skip to avoid double-forwarding.
        skip_dests={"help", "path_filter", "config", "parent_pid", "previous_query"},
    ))
    return cmd


async def websearch_command(args: argparse.Namespace, config: Config) -> None:
    """Fetch DuckDuckGo results for the given query."""
    # Empty-string → None coercion at the surface boundary, so every
    # downstream consumer sees the two-valued "real string or None" contract.
    previous_query = args.previous_query or None
    formatter = RichOutputFormatter(verbose=getattr(args, "verbose", False))
    llm_manager = setup_llm_manager(formatter, config)

    def _on_query_failure(q: str, e: urllib.error.URLError) -> None:
        formatter.warning(
            f"DDG query failed ({q!r}): {e.reason}; "
            "continuing with remaining queries"
        )

    try:
        expansion = await expand_web_queries(
            args.query, llm_manager, previous_query=previous_query
        )
        results = await search_multi(
            expansion.queries,
            args.limit,
            formatter.progress_indicator,
            failure_callback=_on_query_failure,
        )
    except urllib.error.URLError as e:
        formatter.error(f"Web search failed: {e.reason}")
        sys.exit(1)
    if not results:
        formatter.error(
            f"No results found for {args.query!r} — DDG HTML structure may have changed"
        )
        # 10 = empty results (distinct from 1=fetch error, 124=timeout, and
        # the _quickresearch returncode passthrough below).
        sys.exit(10)
    formatter.progress_indicator(
        f"Found {len(results)} results, fetching content..."
    )
    # ignore_cleanup_errors: subprocess may briefly retain handles after exit
    # on Windows/network FS; a cleanup OSError would shadow sys.exit(returncode).
    # The OS reaps $TMPDIR anyway.
    with tempfile.TemporaryDirectory(
        prefix="chunkhound_websearch_", ignore_cleanup_errors=True
    ) as td:
        tmpdir = Path(td)
        mapping: dict[str, str] = {}
        await fetch_and_save(
            [url for _, url, _ in results],
            tmpdir,
            formatter.progress_indicator,
            formatter.warning,
            mapping=mapping,
        )
        # Invoke _quickresearch as a subprocess rather than calling
        # quickresearch_command() directly. On the MCP path, chunkhound's
        # process-global registry singleton (registry/__init__.py) would
        # race — configure_registry() mutates it and registers a database
        # provider as a singleton, so an in-process call could hand this
        # command's DB connection to _quickresearch instead of a fresh
        # :memory: one. The CLI doesn't share the registry concern, but the
        # subprocess wrapper is still the cleanest way to capture stdout for
        # path-rewriting and enforce the wall-clock timeout — using the same
        # path on both surfaces avoids divergence.
        # Positional query is the RAW user input; the LLM-normalized
        # ``search_query`` steers only DDG variants + the follow-up footer.
        # Feeding a lossy normalization to deep research would silently
        # degrade the answer with no signal.
        cmd = _build_quickresearch_argv(
            args, tmpdir, config,
            query=args.query,
            previous_query=previous_query,
        )
        # QUIET routes the child's progress display to stderr (inherited here, so
        # the user still sees it live) and frees stdout to carry only the answer
        # we need to capture and rewrite.
        env = {
            **os.environ,
            "CHUNKHOUND_NO_PROMPTS": "1",
            "CHUNKHOUND_QUICKRESEARCH_QUIET": "1",
        }
        timeout_s = websearch_timeout()
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                check=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                env=env,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            formatter.error(f"websearch timed out after {timeout_s:.0f}s")
            sys.exit(124)
        except subprocess.CalledProcessError as e:
            formatter.error(f"Research failed (exit {e.returncode})")
            sys.exit(e.returncode)
    answer = replace_paths_with_urls(result.stdout, mapping)
    current_query = expansion.search_query
    # Surface-scoped chain footer: emit only the CLI incantation so users
    # never see a tool-call form that isn't runnable at the shell.
    # shlex.quote protects embedded double-quotes so quoted phrases in the
    # search_query survive copy-paste — the whole point of _preserves_quotes.
    footer = (
        "\n\n---\n"
        f'**Continue exploring:** Run `chunkhound websearch "[follow-up]" '
        f'--previous-query {shlex.quote(current_query)}`'
    )
    formatter.text_block(answer + footer)
