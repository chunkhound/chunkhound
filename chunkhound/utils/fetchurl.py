"""Single-URL fetch pipeline shared by CLI and MCP entry points.

Provides `run_fetchurl` — the sole dispatcher into either the truncate
option (token-truncate + one LLM call) or the chunk-rerank option
(chunk → rerank → elbow → LLM call). Fetch, SSRF validation, in-memory
extraction, header annotation, title resolution, and dispatch all live
in this module. Neither entry point calls `_fetch_with_retry` /
`extract` / `option_truncate` / `option_chunk_rerank` directly — they
collapse to a single `run_fetchurl(...)` call.

Placement invariant: (a) CLI must not import from `mcp_server/` and
vice-versa — hence `utils/`, mirroring `websearch_core.py`; (b) this
module imports upward from `parsers/` (PDFMapping, parser_factory,
UniversalParser) and `services/research/` (elbow_filter) — accepted as
a service-shaped exception in an otherwise leaf-layer directory. Do
not grow the pattern further without discussion.
"""

from __future__ import annotations

import asyncio
import ipaddress
import random
import re
import socket
import ssl
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse

from websockets.exceptions import ConnectionClosed as _WSConnectionClosed

from chunkhound.core.config.config import Config
from chunkhound.core.models.chunk import Chunk
from chunkhound.core.types.common import FileId, Language
from chunkhound.core.utils.token_utils import (
    LLM_CHARS_PER_TOKEN,
    estimate_tokens_llm,
)
from chunkhound.parsers.mappings.pdf import PDFMapping
from chunkhound.parsers.parser_factory import create_parser_for_language
from chunkhound.parsers.universal_parser import UniversalParser
from chunkhound.services.prompts import fetchurl as fetchurl_prompts
from chunkhound.services.research.shared.exploration.elbow_filter import (
    filter_chunks_by_elbow,
)
from chunkhound.utils.websearch_core import (
    _managed_browser,
    fetch_url_to_content,
)

if TYPE_CHECKING:
    from chunkhound.interfaces.embedding_provider import EmbeddingProvider
    from chunkhound.llm_manager import LLMManager

__all__ = [
    "FetchUrlError",
    "FetchExtract",
    "extract",
    "run_fetchurl",
]


class FetchUrlError(Exception):
    """Raised by fetchurl for SSRF rejection, missing PyMuPDF, or unexpected kind."""


def _wrap(url: str, answer: str) -> str:
    return f"Source: {url}\n{'=' * 60}\n{answer}\n{'=' * 60}"


def _wrap_no_content(url: str, reason: str) -> str:
    """Diagnostic short-circuit for fetches that produced no usable content."""
    return _wrap(
        url,
        f"Fetched {url}, but no usable content was available to answer "
        f"the query.\nReason: {reason}",
    )


# Ceiling for LLM Markdown-extraction outputs. Input can be up to
# fetchurl.truncate_tokens (default 15k); 4096 output leaves headroom
# for full-page Markdown while staying within provider free-tier caps.
_FETCHURL_OUTPUT_TOKENS = 4096


async def _complete_and_wrap(
    llm_manager: LLMManager, user_prompt: str, url: str
) -> str:
    """Run one utility-LLM completion and wrap the result."""
    provider = llm_manager.get_utility_provider()
    response = await provider.complete(
        prompt=user_prompt,
        system=fetchurl_prompts.SYSTEM_MESSAGE,
        max_completion_tokens=_FETCHURL_OUTPUT_TOKENS,
    )
    return _wrap(url, response.content)


_H1_RE = re.compile(r"^#\s+(.+?)\s*#*\s*$", re.MULTILINE)


def _normalize_title(raw: str) -> str:
    return " ".join(raw.split())[:100]


def _derive_page_title(
    url: str,
    kind: str,
    source_metadata: dict[str, str | None],
    md_content: str | None,
) -> str:
    candidate = source_metadata.get("title")
    if not candidate and kind == ".md" and md_content:
        m = _H1_RE.search(md_content)
        if m:
            candidate = m.group(1)
    if not candidate:
        parsed = urlparse(url)
        candidate = (parsed.netloc + parsed.path) if parsed.path else parsed.netloc
    return _normalize_title(candidate)


async def _validate_url_and_resolve(url: str) -> None:
    """SSRF pre-check. Rejects non-http(s), missing host, and private IPs.

    DNS lookup runs in a worker thread — cold-cache getaddrinfo can
    block for hundreds of ms and would stall the event loop if called
    directly from this coroutine.

    Out of scope: full DNS-rebind defence. urllib and Chrome re-resolve the
    hostname independently on the actual fetch, so a resolver that returned
    a public IP here could still return a private IP at fetch time. Closing
    that gap requires pinning the resolved IP through the fetch, which the
    fetch stack does not currently support.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise FetchUrlError(f"Unsupported URL scheme: {parsed.scheme!r}")
    hostname = parsed.hostname
    if hostname is None:
        raise FetchUrlError(f"URL has no hostname: {url!r}")

    try:
        addrs = await asyncio.to_thread(socket.getaddrinfo, hostname, None)
    except socket.gaierror as e:
        raise FetchUrlError(f"DNS resolution failed for {hostname!r}: {e}") from e

    for family, _type, _proto, _canon, sockaddr in addrs:
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise FetchUrlError(
                f"Blocked address {ip_str!r} for {hostname!r} "
                f"(loopback/private/link-local/reserved/multicast/unspecified)"
            )


def _classify_and_raise_if_terminal(e: BaseException) -> None:
    """Re-raise ``e`` if non-retryable; return silently if retryable.

    Order matters — HTTPError is a subclass of URLError, and URLError may
    wrap ssl.SSLError.
    """
    # 1. FetchUrlError — never retry (SSRF, sentinel, unexpected kind).
    if isinstance(e, FetchUrlError):
        raise e

    # 2. HTTPError — retryable iff 5xx or 429.
    if isinstance(e, HTTPError):
        if e.code >= 500 or e.code == 429:
            return
        raise e

    # 3. SSL errors — never retry, including URLError wrapping SSLError.
    if isinstance(e, ssl.SSLError):
        raise e
    if isinstance(e, URLError) and isinstance(e.reason, ssl.SSLError):
        raise e

    # 4. ValueError — dispatch on message prefix.
    if isinstance(e, ValueError):
        msg = str(e)
        if msg.startswith("Unsupported content-type:"):
            raise e
        if "body rendered empty" in msg:
            raise e
        if msg.startswith("Navigation failed:"):
            return
        raise e  # Unknown ValueError shape — safer to surface than to loop.

    # 5. Transient network / timeout errors — retryable.
    # asyncio.TimeoutError is a distinct class on Python 3.10 (not an alias
    # for the builtin TimeoutError until 3.11) — keep both until
    # requires-python >= 3.11.
    if isinstance(e, (URLError, TimeoutError, asyncio.TimeoutError, socket.timeout)):
        return

    # 6. Anything else — fail closed.
    raise e


def _is_browser_fatal(e: BaseException) -> bool:
    """True iff ``e`` indicates the CDP WebSocket transport has died.

    Signals that the current browser handle is unusable for the rest of the
    retry loop — the caller should rebind ``browser`` to ``None`` so
    subsequent attempts take the urllib branch in ``fetch_url_to_content``.

    Deliberately narrow: ``asyncio.TimeoutError`` is NOT included. A timeout
    on a live browser during a slow page load is legitimate and should
    retry against the browser. Only positive signals of transport death
    (websocket close) trigger the downgrade.

    Also walks ``__cause__``/``__context__`` because zendriver has been
    observed to wrap CDP transport errors in higher-level exceptions.
    ``__context__`` risks false positives (auto-attached for any raise
    inside an except), but CDP death is one-way — the WebSocket stays
    closed — so downgrading to urllib remains the correct action.
    """
    seen: set[int] = set()
    cur: BaseException | None = e
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, _WSConnectionClosed):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


async def _fetch_with_retry(
    url: str,
    config: Config,
    warning_callback: Callable[[str], None] | None = None,
) -> tuple[str, str | bytes, dict[str, str | None]]:
    """Fetch one URL with SSRF pre-check, retry, and Chrome/urllib fallback.

    Returns the 3-tuple from `fetch_url_to_content` on the final successful
    attempt: ``(kind, payload, source_metadata)``.

    Retry budget: ``config.fetchurl.max_retries`` bounds total attempts, not
    attempts-per-transport. A browser-death attempt consumes one slot before
    the loop downgrades to urllib, so ``max_retries=3`` with an attempt-0
    browser death yields two remaining urllib attempts, not three.
    """
    await _validate_url_and_resolve(url)

    async with _managed_browser(warning_callback) as managed_browser:
        # Rebindable local: on a browser-fatal exception we set this to None
        # so subsequent attempts take the urllib branch. The `async with`
        # cleanup still stops the original browser via its own reference.
        browser = managed_browser
        max_attempts = config.fetchurl.max_retries
        last_exc: BaseException | None = None
        for attempt in range(max_attempts):
            if attempt > 0:
                delay = random.uniform(0, min(8.0, 0.5 * (2**attempt)))
                await asyncio.sleep(delay)
            try:
                return await fetch_url_to_content(url, browser)
            except Exception as e:
                if browser is not None and _is_browser_fatal(e):
                    if warning_callback:
                        has_remaining = attempt + 1 < max_attempts
                        tail = (
                            "falling back to urllib for remaining attempts."
                            if has_remaining
                            else "no retries remain."
                        )
                        warning_callback(
                            f"Browser died mid-fetch ({type(e).__name__}: {e}); "
                            f"{tail}"
                        )
                    browser = None
                    last_exc = e
                    continue
                _classify_and_raise_if_terminal(e)  # Re-raises if non-retryable.
                last_exc = e
        # Exhausted retries — surface the last transient exception.
        assert last_exc is not None
        if _is_browser_fatal(last_exc):
            # Browser died on the final attempt so no urllib fallback ran.
            # Wrap into FetchUrlError so the CLI's exception whitelist matches
            # and MCP callers see a semantic type instead of a raw
            # websockets exception name.
            raise FetchUrlError(
                f"Browser transport died mid-fetch and retries exhausted "
                f"({type(last_exc).__name__}: {last_exc})"
            ) from last_exc
        raise last_exc


def _is_markdown_heading(chunk: Chunk) -> bool:
    """True iff this chunk is an ATX heading emitted by MarkdownMapping.

    Accepts ``atx_heading`` only. Setext headings are out of scope: they
    would require synthesizing the ``#`` prefix from
    ``metadata["heading_level"]``, and the reranker/LLM pipeline already
    reads the raw source text where setext underlines are visible.

    The ``chunk.metadata or {}`` guard covers PDF chunks: ``pdf.py``
    constructs Chunk without ``metadata=``, so ``metadata is None``.
    """
    return (chunk.metadata or {}).get("node_type") == "atx_heading"


def _annotate_parent_headers(chunks: list[Chunk]) -> list[Chunk]:
    """Populate ``parent_header`` on Markdown chunks by walking document order.

    Heading chunks keep ``parent_header=None`` (a heading is not its own
    parent). Non-heading chunks are stamped with the most recently seen
    heading's first non-blank line, ``.strip()``ed, verbatim (leading ``#``s
    intact so the LLM sees a proper Markdown header when the chunk is
    emitted under it later).
    """
    ordered = sorted(chunks, key=lambda c: int(c.start_line))
    result: list[Chunk] = []
    last_header: str | None = None
    for chunk in ordered:
        if _is_markdown_heading(chunk):
            first_line = next(
                (ln for ln in chunk.code.splitlines() if ln.strip()), ""
            )
            last_header = first_line.strip()
            result.append(chunk)  # heading itself: parent_header stays None
        else:
            if last_header is not None:
                result.append(replace(chunk, parent_header=last_header))
            else:
                result.append(chunk)
    return result


def _raise_if_pdf_unavailable(chunks: list[Chunk]) -> None:
    """Raise FetchUrlError with install guidance if PyMuPDF is missing.

    ``PDFMapping.parse_pdf_content`` never raises — it emits a single
    sentinel chunk on failure. The ``pdf_unavailable`` sentinel means
    PyMuPDF is not importable (a broken environment, not a fetch problem),
    so every PDF will fail until the user installs it. Surface an
    actionable message instead of the raw sentinel ``code``.

    The ``pdf_parse_error`` sentinel is handled downstream in
    ``run_fetchurl`` — see ``_is_pdf_parse_error``.
    """
    if len(chunks) == 1 and chunks[0].symbol == "pdf_unavailable":
        raise FetchUrlError(
            "PDF parsing requires PyMuPDF, which is not installed. "
            "Install it with: uv add pymupdf"
        )


@dataclass(frozen=True)
class FetchExtract:
    """Single-pass extraction result from a fetched URL."""

    kind: str            # ".md" | ".pdf"
    text: str            # raw Markdown or joined PDF chunk-code
    chunks: list[Chunk]  # annotated on .md, verbatim on .pdf
    title: str           # resolved via _derive_page_title


def extract(
    kind: str,
    payload: str | bytes,
    url: str,
    source_metadata: dict[str, str | None],
) -> FetchExtract:
    """Fetch a single ``(kind, payload)`` into a `FetchExtract`.

    Called from `run_fetchurl` immediately after `_fetch_with_retry`. All
    four fields are populated in one call — the truncate option reads
    ``text``, the chunk-rerank option reads ``chunks``, both read ``title``.
    """
    if kind == ".md":
        assert isinstance(payload, str)
        # parser_factory returns the LanguageParser protocol; MARKDOWN's
        # concrete instance is a UniversalParser (parser_factory.py
        # branch — MARKDOWN has ts_markdown). Cast so mypy sees the
        # concrete API (file_id kwarg, list[Chunk] return).
        parser = cast(UniversalParser, create_parser_for_language(Language.MARKDOWN))
        chunks = _annotate_parent_headers(
            parser.parse_content(payload, Path("dummy.md"), FileId(0))
        )
        title = _derive_page_title(url, kind, source_metadata, payload)
        # text := payload keeps the raw fetched Markdown for the truncate
        # option; substituting joined chunk-code would introduce chunk-
        # boundary artifacts the LLM would not otherwise see.
        return FetchExtract(kind, payload, chunks, title)

    if kind == ".pdf":
        assert isinstance(payload, bytes)
        chunks = PDFMapping().parse_pdf_content(payload, Path("dummy.pdf"), FileId(0))
        _raise_if_pdf_unavailable(chunks)
        text = "\n\n".join(c.code for c in chunks)
        title = _derive_page_title(url, kind, source_metadata, md_content=None)
        return FetchExtract(kind, text, chunks, title)

    raise FetchUrlError(f"Unexpected kind: {kind}")


def _is_pdf_parse_error(fx: FetchExtract) -> bool:
    """True if extract() returned a single ``pdf_parse_error`` sentinel chunk.

    Signals that the PDF was fetched successfully but could not be parsed
    (encrypted, password-protected, corrupt, or PyMuPDF internal error).
    ``run_fetchurl`` short-circuits on this to emit a diagnostic answer
    without calling the reranker or LLM over a single error-message chunk.
    """
    return (
        fx.kind == ".pdf"
        and len(fx.chunks) == 1
        and fx.chunks[0].symbol == "pdf_parse_error"
    )


async def run_fetchurl(
    url: str,
    query: str,
    config: Config,
    embedding_provider: EmbeddingProvider,
    llm_manager: LLMManager,
    *,
    warning_callback: Callable[[str], None] | None = None,
    verbose_log: Callable[[str], None] | None = None,
) -> str:
    """Fetch a URL, extract, dispatch to the chosen option, return wrapped Markdown.

    The reranker capability gate is enforced at the CLI and MCP entry
    points — ``run_fetchurl`` assumes a reranker-capable provider.

    ``warning_callback`` is reserved for signals not already present in the
    returned body (e.g. reranker degradation); do not re-emit URL/detail
    strings that ``_wrap`` / ``_wrap_no_content`` already surface, or callers
    will render them twice.
    """
    kind, payload, source_metadata = await _fetch_with_retry(
        url, config, warning_callback=warning_callback,
    )
    fx = extract(kind, payload, url, source_metadata)
    if verbose_log:
        verbose_log(f"title={fx.title}")

    if _is_pdf_parse_error(fx):
        detail = fx.chunks[0].code
        return _wrap(
            url,
            f"Fetched PDF at {url}, but could not extract text content.\n"
            f"Detail: {detail}\n\n"
            f"The PDF may be encrypted, password-protected, or corrupt. "
            f"No content is available to answer the query."
        )

    # Fetch succeeded but produced nothing usable to dispatch: 0-chunk PDF
    # (image-only, empty, unsupported layout) or an .md whose parser output
    # is whitespace-only. fetch_url_to_content already rejects pre-parse
    # empty bodies (websearch_core.py `body rendered empty`); this catches
    # the residual post-parse case.
    if not fx.chunks or not fx.text.strip():
        reason = (
            "PDF extracted zero chunks "
            "(empty PDF, image-only without OCR, or unsupported layout)"
            if fx.kind == ".pdf"
            else "page rendered to empty content"
        )
        return _wrap_no_content(url, reason)

    total_tokens = estimate_tokens_llm(fx.text)
    threshold = config.fetchurl.rerank_threshold_tokens

    if not query or total_tokens <= threshold:
        if verbose_log:
            verbose_log(f"option=truncate, total_tokens={total_tokens}")
        return await option_truncate(fx.text, query, url, fx.title, llm_manager, config)

    if verbose_log:
        verbose_log(f"option=chunk-rerank, chunks_before={len(fx.chunks)}")
    return await option_chunk_rerank(
        fx.chunks, query, url, fx.title,
        embedding_provider, llm_manager, config,
        warning_callback=warning_callback,
    )


async def option_truncate(
    text: str,
    query: str,
    url: str,
    title: str,
    llm_manager: LLMManager,
    config: Config,
) -> str:
    """One LLM call over the raw extracted text.

    Char-ratio slice matches ``estimate_tokens_llm`` in ``token_utils.py``:
    deterministic + cheap, no tokenizer round-trip.
    """
    truncated = text[: config.fetchurl.truncate_tokens * LLM_CHARS_PER_TOKEN]

    if query:
        user = fetchurl_prompts.FOCUSED_USER_TEMPLATE.format(
            url=url, title=title, query=query, content=truncated,
        )
    else:
        user = fetchurl_prompts.GENERIC_USER_TEMPLATE.format(
            url=url, title=title, content=truncated,
        )

    return await _complete_and_wrap(llm_manager, user, url)


_PDF_PAGE_RE = re.compile(r"^page_(\d+)_")


def _chunk_locator(c: dict[str, Any]) -> str:
    """Locator marker prefixed on each chunk in the LLM body.

    Markdown chunks: ``[L{start}-{end}]`` — real source line span.
    PDF chunks: ``[P{page}]`` — page number parsed from ``symbol``.
    Empty string on the rare PDF chunk whose symbol does not match
    ``^page_(\\d+)_`` — the assembly loop tolerates an empty marker
    without emitting a bare leading newline.
    """
    if c.get("language") == Language.PDF:
        m = _PDF_PAGE_RE.match(c.get("symbol") or "")
        return f"[P{m.group(1)}]" if m else ""
    return f"[L{c['start_line']}-{c['end_line']}]"


async def option_chunk_rerank(
    chunks: list[Chunk],
    query: str,
    url: str,
    title: str,
    embedding_provider: EmbeddingProvider,
    llm_manager: LLMManager,
    config: Config,
    *,
    warning_callback: Callable[[str], None] | None = None,
) -> str:
    # chunk → dict conversion (elbow filter + rerank attach want dicts).
    # ``is_heading`` is precomputed via _is_markdown_heading so the body-
    # assembly self-detection below is a boolean lookup, not a second
    # metadata pass.
    chunk_dicts: list[dict[str, Any]] = [
        {
            "content": c.code,
            "start_line": int(c.start_line),
            "end_line": int(c.end_line),
            "parent_header": c.parent_header,
            "is_heading": _is_markdown_heading(c),
            "rerank_score": 0.0,
            "symbol": c.symbol,
            "language": c.language,
        }
        for c in chunks
    ]

    # Reranker document builder. Front-load topical scope (title, then
    # section header) so the model can attend to it. Line markers are
    # deliberately NOT injected here — the reranker does not semantically
    # use them; body assembly below emits them on the already-filtered
    # chunks the LLM actually reads.
    def _rerank_document(d: dict[str, Any]) -> str:
        header = d.get("parent_header")
        if header:
            return f"{title}\n{header}\n\n{d['content']}"
        return f"{title}\n\n{d['content']}"

    documents = [_rerank_document(d) for d in chunk_dicts]

    # Batched rerank. Template: bfs_exploration_strategy.py
    # (per-batch slice indexing); flattened here via abs_idx.
    max_batch = embedding_provider.get_max_rerank_batch_size()
    total_rerank_results = 0
    for start in range(0, len(documents), max_batch):
        end = min(start + max_batch, len(documents))
        batch = documents[start:end]
        batch_results = await embedding_provider.rerank(
            query=query, documents=batch,
        )
        total_rerank_results += len(batch_results)
        for r in batch_results:
            abs_idx = r.index + start
            if 0 <= abs_idx < len(chunk_dicts):
                chunk_dicts[abs_idx]["rerank_score"] = r.score

    # Surface degenerate rerank output. fetchurl has no upstream score to fall
    # back to (unlike unified_search's HNSW similarity), so a silent rerank
    # failure would flow through elbow's uniform-score branch and return
    # chunks in original document order — indistinguishable from a healthy
    # result. Warn so operators see the degradation; the elbow filter still
    # runs and callers still get an answer.
    if warning_callback is not None:
        if total_rerank_results == 0:
            warning_callback(
                f"Reranker returned no results for {url}; "
                f"falling back to unranked chunk order."
            )
        elif total_rerank_results != len(documents):
            warning_callback(
                f"Reranker returned {total_rerank_results}/{len(documents)} "
                f"results for {url}; unscored chunks retain default 0.0."
            )
        elif len(chunk_dicts) > 3:
            unique_scores = {c["rerank_score"] for c in chunk_dicts}
            if len(unique_scores) == 1:
                warning_callback(
                    f"Reranker returned identical scores for all "
                    f"{len(chunk_dicts)} chunks on {url}; ranking signal is flat."
                )

    # Elbow filter (handles <3 chunks / no-elbow / uniform edge cases).
    filtered, _stats = filter_chunks_by_elbow(chunk_dicts, score_key="rerank_score")

    # Restore document order. filter_chunks_by_elbow returns score-sorted
    # output on the ``elbow`` and ``no_elbow_detected`` branches;
    # unconditionally sorting here is correct for all three elbow-filter
    # branches.
    filtered.sort(key=lambda c: c["start_line"])

    # Body assembly with per-chunk locator + header dedup.
    body_parts: list[str] = []
    last_header: str | None = None
    for c in filtered:
        header = c.get("parent_header")
        marker = _chunk_locator(c)
        prefix = f"{marker}\n" if marker else ""
        if header and header != last_header:
            body_parts.append(f"{prefix}{header}\n\n{c['content']}")
            last_header = header
        else:
            body_parts.append(f"{prefix}{c['content']}")
            # parent_header=None is ambiguous between "top-of-document" and
            # "chunk IS a heading". Use ``is_heading`` (precomputed above)
            # to disambiguate: prime last_header on heading chunks so the
            # following chunk suppresses the duplicate header line; otherwise
            # reset to None so a later chunk under a re-encountered header
            # will re-emit it.
            if header is None:
                if c["is_heading"]:
                    first_line = next(
                        (ln for ln in c["content"].splitlines() if ln.strip()), ""
                    )
                    last_header = first_line.strip()
                else:
                    last_header = None
    body = "\n\n---\n\n".join(body_parts)

    user = fetchurl_prompts.FOCUSED_USER_TEMPLATE.format(
        url=url, title=title, query=query, content=body,
    )
    return await _complete_and_wrap(llm_manager, user, url)
