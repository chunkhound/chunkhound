"""Fetchurl command for ChunkHound CLI."""

from __future__ import annotations

import argparse
import asyncio
import ssl
import sys
import urllib.error

from chunkhound.core.config.config import Config
from chunkhound.utils.fetchurl import FetchUrlError, run_fetchurl

from ..utils.provider_setup import setup_embedding_manager, setup_llm_manager
from ..utils.rich_output import RichOutputFormatter


async def fetchurl_command(args: argparse.Namespace, config: Config) -> None:
    """Fetch a URL and produce a focused Markdown answer."""
    formatter = RichOutputFormatter(verbose=getattr(args, "verbose", False))

    llm_manager = setup_llm_manager(formatter, config)
    if llm_manager is None:
        formatter.error("fetchurl requires an LLM provider (configure llm.provider).")
        sys.exit(1)

    embedding_manager = setup_embedding_manager(formatter, config)
    embedding_provider = embedding_manager.get_default_provider()
    if embedding_provider is None or not embedding_provider.supports_reranking():
        formatter.error(
            "fetchurl requires a configured reranker "
            "(e.g., VoyageAI with a rerank model, or a TEI/Cohere HTTP reranker)."
        )
        sys.exit(1)

    try:
        answer = await run_fetchurl(
            args.url,
            args.query,
            config,
            embedding_provider,
            llm_manager,
            warning_callback=formatter.warning,
            verbose_log=formatter.verbose_info,
        )
    # asyncio.TimeoutError is a distinct class on Python 3.10 (not an alias
    # for the builtin TimeoutError until 3.11) — keep both until
    # requires-python >= 3.11.
    except (TimeoutError, asyncio.TimeoutError):
        formatter.error("fetchurl timed out")
        sys.exit(124)
    except FetchUrlError as e:
        formatter.error(f"fetchurl failed: {e}")
        sys.exit(1)
    except urllib.error.URLError as e:
        # Covers HTTPError (subclass).
        formatter.error(f"fetchurl failed: {e.reason}")
        sys.exit(1)
    except (ssl.SSLError, ValueError) as e:
        # Fetch-layer non-retryable from _classify_and_raise_if_terminal:
        # ssl.SSLError, and ValueError from fetch_url_to_content
        # (content-type / empty-body).
        formatter.error(f"fetchurl failed: {e}")
        sys.exit(1)
    # Browser transport death on retry-exhaustion is wrapped into FetchUrlError
    # inside `_fetch_with_retry` (caught above). Other unexpected errors
    # (LLM/reranker provider failures, OSError, non-transport CDP errors)
    # intentionally propagate — surfacing the traceback is more useful than a
    # generic "fetchurl failed" swallow.

    formatter.text_block(answer)
