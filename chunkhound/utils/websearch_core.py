"""Protocol-neutral websearch helpers shared by the CLI and MCP entry points.

Both ``chunkhound/api/cli/commands/websearch.py`` (CLI) and
``chunkhound/mcp_server/tools.py`` (``websearch_impl``) consume these helpers.
Keeping them here breaks the prior MCP→CLI import direction so both call
sites depend on this neutral module instead of each other.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import html.parser
import os
import re
import sys
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.async_api import BrowserContext

from chunkhound.core.config.config import Config

_MAX_FETCH_CONCURRENCY = 5

WEBSEARCH_LIMIT_MAX = 100


def clamp_limit(limit: int) -> int:
    """Silently clamp result-count limit to [1, WEBSEARCH_LIMIT_MAX].

    Used by MCP (LLM-supplied values); CLI validates via argparse instead.
    """
    return max(1, min(limit, WEBSEARCH_LIMIT_MAX))


def websearch_timeout() -> float:
    """Overall wall-clock timeout (seconds) for the websearch subprocess.

    Reads CHUNKHOUND_WEBSEARCH_TIMEOUT_SECONDS; falls back to 600.0 on
    unset or malformed values.
    """
    raw = os.environ.get("CHUNKHOUND_WEBSEARCH_TIMEOUT_SECONDS")
    if raw is None:
        return 600.0
    try:
        return float(raw)
    except ValueError:
        return 600.0


class _NextFormParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._form: list[dict[str, str | None]] | None = None
        self._forms: list[list[dict[str, str | None]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        d: dict[str, str | None] = dict(attrs)
        if tag == "form":
            self._form = []
        elif tag == "input" and self._form is not None:
            self._form.append(d)

    def handle_endtag(self, tag: str) -> None:
        if tag == "form" and self._form is not None:
            self._forms.append(self._form)
            self._form = None

    def next_params(self) -> dict[str, str] | None:
        for form in self._forms:
            if any(
                a.get("type") == "submit" and a.get("value") == "Next" for a in form
            ):
                return {
                    name: a.get("value") or ""
                    for a in form
                    if a.get("type") == "hidden" and (name := a.get("name")) is not None
                }
        return None


class _ResultParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[tuple[str, str, str]] = []
        self._capture: str | None = None
        self._title = ""
        self._url = ""
        self._desc = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        d: dict[str, str | None] = dict(attrs)
        cls = d.get("class") or ""
        if tag == "a" and "result__a" in cls:
            self._url = d.get("href") or ""
            self._title = ""
            self._capture = "title"
        elif tag == "a" and "result__snippet" in cls:
            self._desc = ""
            self._capture = "desc"

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._capture == "title":
            self._capture = None
        elif tag == "a" and self._capture == "desc":
            self._capture = None
            if self._title and self._url:
                self.results.append(
                    (
                        html.unescape(self._title),
                        self._url,
                        html.unescape(self._desc),
                    )
                )
            self._title = self._url = self._desc = ""

    def handle_data(self, data: str) -> None:
        if self._capture == "title":
            self._title += data
        elif self._capture == "desc":
            self._desc += data


def _fetch(params: dict[str, str]) -> str:
    data = urllib.parse.urlencode(params).encode()
    req = urllib.request.Request(
        "https://html.duckduckgo.com/html/",
        data=data,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode()


def _url_to_filename(url: str, max_length: int = 100) -> str:
    # Append a short stable hash of the full URL so distinct URLs cannot
    # collide via the lossy [^\w.-]→_ substitution or via truncation when
    # two URLs share a long common prefix.
    name = re.sub(r"^https?://", "", url)
    name = re.sub(r"[^\w.-]", "_", name)
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"{name[: max(0, max_length - 9)]}_{digest}"[:max_length]


def _html_to_markdown(html_text: str) -> str:
    # Lazy import: defers loading markdownify + deps (beautifulsoup4, soupsieve, six)
    from markdownify import MarkdownConverter

    class _Converter(MarkdownConverter):
        # strip=... only drops the wrapper tag, not its children, so raw
        # JS/CSS/SVG source still leaks. Override to discard the body.
        def convert_script(self, el, text, parent_tags):
            return ""

        def convert_style(self, el, text, parent_tags):
            return ""

        def convert_svg(self, el, text, parent_tags):
            return ""

    return _Converter(
        strip=[
            "head",
            "nav", "footer", "header", "aside",
            "form", "button", "iframe", "noscript",
        ],
        heading_style="ATX",
    ).convert(html_text)


def _normalize_ct(raw: str | None) -> str:
    """Parse the bare MIME type out of a Content-Type header.

    Returns ``"text/html"`` when the header is missing. ``urllib``'s
    ``get_content_type()`` synthesizes ``"text/plain"`` for header-less
    responses, which would reject real HTML served by misconfigured
    servers; defaulting to ``text/html`` here matches the browser path,
    which has always rendered header-less responses as HTML.
    """
    if not raw:
        return "text/html"
    return raw.split(";", 1)[0].strip().lower()


def _decode_pdf_or_fallback_html(body: bytes, charset: str) -> tuple[str, str | bytes]:
    """Handle a body whose Content-Type claimed application/pdf.

    Some endpoints (paywalls, error pages, auth walls) return HTML under an
    application/pdf Content-Type. Trust the magic bytes, not the header.
    """
    if body.startswith(b"%PDF-"):
        return ".pdf", body
    return ".md", _html_to_markdown(body.decode(charset, errors="replace"))


def _fetch_url(url: str) -> tuple[str, bytes, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        ct = _normalize_ct(resp.headers.get("Content-Type"))
        return ct, resp.read(), resp.headers.get_content_charset() or "utf-8"


async def _fetch_page(context: BrowserContext, url: str) -> tuple[str, bytes, str]:
    """Fetch a single URL.

    HTML is fetched through Chrome's network stack. PDFs go through
    Playwright's APIRequestContext because headless Chrome cannot reliably
    expose PDF bytes to page.content()/response.body() — see Playwright #6342
    and scrapy-playwright #243.
    """
    page = await context.new_page()
    try:
        # commit resolves once response headers are parsed — before Chrome
        # hands a PDF to its viewer or starts rendering HTML.
        response = await page.goto(url, timeout=30000, wait_until="commit")
        if response is None:
            raise ValueError("Navigation failed: no response")

        ct = _normalize_ct(response.headers.get("content-type"))

        if ct == "application/pdf":
            await page.close()
            direct = await context.request.get(url)
            try:
                if not direct.ok:
                    raise ValueError(f"PDF fetch failed: HTTP {direct.status}")
                return "application/pdf", await direct.body(), "utf-8"
            finally:
                await direct.dispose()

        if ct != "text/html":
            raise ValueError(f"Unsupported content-type: {ct!r}")

        await page.wait_for_load_state("load")
        return ct, (await page.content()).encode("utf-8"), "utf-8"
    finally:
        if not page.is_closed():
            await page.close()


async def _fetch_one(
    url: str,
    tmpdir: Path,
    context: BrowserContext | None,
    progress_callback: Callable[[str], None] | None,
    warning_callback: Callable[[str], None] | None,
    semaphore: asyncio.Semaphore,
    mapping: dict[str, str] | None,
) -> None:
    async with semaphore:
        if progress_callback:
            progress_callback(f"Fetching {url}...")
        try:
            if context is not None:
                ct, body, charset = await _fetch_page(context, url)
            else:
                ct, body, charset = await asyncio.to_thread(_fetch_url, url)
            if ct == "application/pdf":
                ext, content = _decode_pdf_or_fallback_html(body, charset)
            elif ct == "text/html":
                ext, content = ".md", _html_to_markdown(
                    body.decode(charset, errors="replace")
                )
            else:
                raise ValueError(f"Unsupported content-type: {ct!r}")
            # Auth walls and error pages often render to whitespace-only
            # markdown; surface that as a fetch failure rather than writing
            # an empty file that consumes a result slot.
            if not content.strip():
                raise ValueError(f"{ct!r} body rendered empty ({len(body)} bytes)")
            path = tmpdir / (_url_to_filename(url) + ext)
            if isinstance(content, bytes):
                path.write_bytes(content)
            else:
                path.write_text(content, encoding="utf-8")
            if mapping is not None:
                mapping[path.name] = url
        except Exception as e:
            if warning_callback:
                warning_callback(f"Failed to fetch {url}: {type(e).__name__}: {e}")


async def fetch_and_save(
    urls: list[str],
    tmpdir: Path,
    progress_callback: Callable[[str], None] | None = None,
    warning_callback: Callable[[str], None] | None = None,
    mapping: dict[str, str] | None = None,
) -> None:
    """Fetch each URL concurrently (bounded) and save content to tmpdir."""
    semaphore = asyncio.Semaphore(_MAX_FETCH_CONCURRENCY)

    async def _run(context: BrowserContext | None) -> None:
        tasks = [
            _fetch_one(
                url, tmpdir, context, progress_callback, warning_callback,
                semaphore, mapping,
            )
            for url in urls
        ]
        await asyncio.gather(*tasks)

    try:
        from playwright.async_api import Error as PlaywrightError
        from playwright.async_api import async_playwright
    except ImportError:
        await _run(None)
        return

    async with async_playwright() as pw:
        # channel="chrome" drives the system-installed Google Chrome instead
        # of Playwright's bundled Chromium. The `chunkhound[browser]` extra
        # only installs the Playwright Python package; users must also have
        # Google Chrome installed system-wide — otherwise launch() fails and
        # we fall through to the urllib path below.
        #
        # New headless mode is required for the PDF path: legacy --headless
        # hands PDFs to an internal viewer and never exposes the response to
        # _fetch_page. --headless=new keeps the navigation visible so we can
        # branch on content-type at commit time.
        try:
            browser = await pw.chromium.launch(
                channel="chrome",
                args=["--headless=new"],
                ignore_default_args=["--headless"],
            )
        except PlaywrightError as e:
            if warning_callback:
                warning_callback(
                    f"Browser launch failed: {e}. Falling back to urllib."
                    " (If Google Chrome is not installed, install it to"
                    " enable rich page fetches.)"
                )
            await _run(None)
            return
        try:
            context = await browser.new_context()
            try:
                await _run(context)
            finally:
                await context.close()
        finally:
            await browser.close()


def search(
    query: str,
    limit: int = 30,
    progress_callback: Callable[[str], None] | None = None,
) -> list[tuple[str, str, str]]:
    results: list[tuple[str, str, str]] = []
    params: dict[str, str] = {"q": query, "b": ""}  # b = submit button name
    page_num = 0
    while True:
        page_num += 1
        if progress_callback:
            progress_callback(f"Fetching page {page_num}...")
        page = _fetch(params)
        rp = _ResultParser()
        rp.feed(page)
        if not rp.results:
            break
        results.extend(rp.results)
        if len(results) >= limit:
            break
        nfp = _NextFormParser()
        nfp.feed(page)
        next_params = nfp.next_params()
        if not next_params:
            break
        params = next_params
    return results[:limit]


def build_quickresearch_argv_core(
    query: str,
    tmpdir: Path,
    path_filter: str | None,
    config: Config,
) -> list[str]:
    """Build argv to invoke _quickresearch as a subprocess.

    Forwards the config source file as an absolute path so the child process
    does not need to re-run config discovery (which would otherwise fall back
    to env vars / defaults under the MCP server's working directory).
    """
    cmd: list[str] = [
        sys.executable,
        "-m", "chunkhound.api.cli.main",
        "_quickresearch",
        query,
        str(tmpdir),
    ]
    if path_filter is not None:
        cmd.extend(["--path-filter", path_filter])
    source = config.config_file or config.local_config_file
    if source is not None:
        cmd.extend(["--config", str(Path(source).resolve())])
    return cmd
