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
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    from http.client import HTTPMessage
    from typing import Any

    import zendriver as zd

from chunkhound.core.config.config import Config

_MAX_FETCH_CONCURRENCY = 5

WEBSEARCH_LIMIT_MAX = 100

# Probe these paths before zendriver's auto-discovery. zendriver picks the
# shortest-named binary from [google-chrome, chromium, chromium-browser,
# chrome, google-chrome-stable], so `chromium` wins over `google-chrome`
# when both are installed — masking the "Chrome not installed" failure that
# the urllib fallback is designed to catch. Mirror Playwright's
# channel="chrome" preference by checking known Chrome paths first.
_CHROME_PATHS = [
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/opt/google/chrome/chrome",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]


_late_completion_guard_installed = False


def _install_late_completion_guard() -> None:
    """Patch zendriver's Transaction.__call__ to drop late completions.

    TODO: remove once zendriver fixes Connection.send to pop cancelled
    transactions from connection.mapper (or guards Transaction.__call__
    upstream). Revisit on every zendriver bump — the version-mismatch
    warning below is the in-process signal that this may be stale.

    zendriver 0.15.3's Connection.send (connection.py:561-572) does not pop the
    Transaction from connection.mapper when its awaiter is cancelled — e.g. when
    we wrap tab.send(cdp.page.navigate(...)) in asyncio.wait_for(..., timeout=30)
    at _fetch_page. The Transaction Future transitions to CANCELLED but stays
    registered. When Chrome later delivers the response, Listener.listener_loop
    (connection.py:780) pops the orphan and calls tx(**message), which calls
    set_result() on the cancelled Future and raises InvalidStateError. That
    exception is uncaught in listener_loop and kills the listener task — the
    asyncio "Task exception was never retrieved" log is the visible symptom; the
    real damage is that the websocket recv stops draining and the browser
    session becomes unusable.

    Guarding __call__ with a Future.done() check makes the late completion a
    no-op, mirroring the local ``on_response`` idempotency guard inside
    ``_fetch_page``. Discarding the late result is correct: the only consumer
    was the awaiter inside Connection.send, which already received
    CancelledError.

    Idempotent via module-level flag — safe to call from every fetch_and_save.
    """
    global _late_completion_guard_installed
    if _late_completion_guard_installed:
        return
    import warnings

    import zendriver
    from zendriver.core.connection import Transaction

    if zendriver.__version__ != "0.15.3":
        warnings.warn(
            f"zendriver {zendriver.__version__} != pinned 0.15.3; "
            "late-completion guard may now be redundant or broken — revisit "
            "whether upstream fixed Connection.send or Transaction.__call__",
            RuntimeWarning,
            stacklevel=2,
        )

    _orig_call = Transaction.__call__

    def _safe_call(self: Transaction, **response: Any) -> None:
        if self.done():
            return
        _orig_call(self, **response)

    Transaction.__call__ = _safe_call  # type: ignore[method-assign]
    _late_completion_guard_installed = True


def _check_chrome_version(chrome_path: str) -> None:
    """Raise ``RuntimeError`` if Chrome at ``chrome_path`` is unverifiable or <124.

    zendriver 0.15.3's CDP binding declares ``Network.Response.charset`` as a
    required str field. Chrome only began emitting ``charset`` on
    ``Network.responseReceived`` around v124. On older Chrome every event
    silently fails to parse in zendriver's listener loop (logged at INFO,
    no handler fires) — navigations time out with no informative error.
    """
    try:
        out = subprocess.check_output(
            [chrome_path, "--version"], text=True, timeout=5
        ).strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        raise RuntimeError(
            f"--version probe at {chrome_path!r} failed: "
            f"{type(e).__name__}: {e}"
        ) from e
    major = next(
        (int(tok.split(".")[0]) for tok in out.split() if tok.split(".")[0].isdigit()),
        None,
    )
    if major is None:
        raise RuntimeError(
            f"unparseable --version output from {chrome_path!r}: {out!r}"
        )
    if major < 124:
        raise RuntimeError(
            f"Chrome at {chrome_path!r} is v{major} (from {out!r}); "
            f"need >=124 for zendriver 0.15.3 to emit Response.charset "
            f"on Network.responseReceived"
        )


def _resolve_chrome_path(
    warning_callback: Callable[[str], None] | None = None,
) -> str | None:
    """Locate a Chrome binary >=124; collapse every failure mode to ``None``.

    Probes ``_CHROME_PATHS`` first; falls back to zendriver's auto-discovery
    if no listed path verifies. Per-candidate failures (version <124,
    unparseable output, hung/failing --version probe) are deferred and only
    emitted via ``warning_callback`` when no usable Chrome is ultimately
    found — a later candidate succeeding silences earlier failures.
    """
    deferred: dict[str, RuntimeError] = {}

    for p in _CHROME_PATHS:
        if os.path.exists(p):
            try:
                _check_chrome_version(p)
                return p
            except RuntimeError as e:
                deferred[p] = e
    try:
        from zendriver.core.config import find_executable
        resolved = find_executable("auto")
    except Exception:
        # Broad catch: zendriver's launch-failure surface is documented as
        # generic exceptions, and the spike observed platform-dependent
        # OSError variants when no Chrome/Chromium binary exists. A broader
        # catch keeps the urllib fallback reachable when find_executable
        # raises anything unexpected.
        resolved = None
    if resolved:
        auto_path = str(resolved)
        if auto_path not in deferred:
            try:
                _check_chrome_version(auto_path)
                return auto_path
            except RuntimeError as e:
                deferred[auto_path] = e

    if warning_callback:
        if deferred:
            bullets = "\n  - " + "\n  - ".join(str(e) for e in deferred.values())
            warning_callback(
                f"Chrome verification failed; falling back to urllib:{bullets}"
                "\n(Upgrade Google Chrome to >=124 to enable rich page fetches.)"
            )
        else:
            warning_callback(
                "No Chrome binary found. Falling back to urllib."
                " (Install Google Chrome to enable rich page fetches.)"
            )
    return None


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


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Fail-loud redirect blocker for the cookie-bearing urllib refetch.

    Chrome already resolved redirects before we re-fetched. Any redirect
    here would cause urllib to forward our Cookie header to a domain
    other than the one we extracted cookies for — silently leaking
    credentials. Raise instead.
    """

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        raise urllib.error.HTTPError(
            req.full_url,
            code,
            f"Redirect to {newurl} blocked on cookie-bearing fetch",
            headers,
            fp,
        )


def _fetch_url(
    url: str,
    extra_headers: dict[str, str] | None = None,
    follow_redirects: bool = True,
) -> tuple[str, bytes, str]:
    # Header values are passed verbatim. http.client blocks CRLF, but
    # nothing else is validated — never pass attacker-influenced values.
    headers = {"User-Agent": "Mozilla/5.0"}
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(url, headers=headers)
    opener = (
        urllib.request.build_opener()
        if follow_redirects
        else urllib.request.build_opener(_NoRedirect)
    )
    with opener.open(req, timeout=30) as resp:
        ct = _normalize_ct(resp.headers.get("Content-Type"))
        return ct, resp.read(), resp.headers.get_content_charset() or "utf-8"


async def _close_tab_quietly(tab: zd.Tab) -> None:
    """Close a tab with a bounded wait.

    tab.close() can hang on the PDF viewer's 10s ack timeout and has been
    observed to take >3s even on the HTML path. Cap at 5s and rely on
    ``browser.stop()`` to reap anything left behind.
    """
    try:
        await asyncio.wait_for(tab.close(), timeout=5)
    except asyncio.TimeoutError:
        pass


async def _fetch_page(browser: zd.Browser, url: str) -> tuple[str, bytes, str]:
    """Fetch a single URL via raw CDP.

    Replicates Playwright's ``wait_until="commit"`` semantics by subscribing
    to ``Network.responseReceived`` for the main frame and branching on the
    content-type before Chrome's PDF viewer engages or HTML rendering starts.
    PDFs are re-fetched via urllib because headless Chrome cannot reliably
    expose PDF bytes through the DOM.
    """
    from zendriver import cdp

    # ``new_tab=True`` is mandatory. Plain ``browser.get("about:blank")``
    # reuses the active tab, so concurrent fetches under the 5-way semaphore
    # would clobber each other's handlers and navigation state.
    tab = await browser.get("about:blank", new_tab=True)
    tab_closed = False
    try:
        # Buffer full ``ResponseReceived`` events (not ``event.response``):
        # ``loader_id`` lives on the event, not on ``Response``. We need it
        # to match buffered events against the navigation once its loader_id
        # is known.
        nav_loader_id: cdp.network.LoaderId | None = None
        candidates: list[cdp.network.ResponseReceived] = []
        main_response: asyncio.Future[cdp.network.Response] = (
            asyncio.get_running_loop().create_future()
        )

        # Handler MUST be ``async def``. zendriver 0.15.3's sync-dispatch
        # path reassigns ``event`` in the listener scope before the
        # thread-pool callback runs — sync handlers see stale events. The
        # idempotency guard is load-bearing: zendriver occasionally routes
        # async handlers through the sync branch and discards the returned
        # coroutine, so the body must tolerate dropped invocations.
        async def on_response(event: cdp.network.ResponseReceived) -> None:
            if main_response.done():
                return
            if nav_loader_id is None:
                candidates.append(event)
                return
            if event.loader_id == nav_loader_id:
                main_response.set_result(event.response)

        # Register the handler BEFORE Network.enable. Events that fire
        # during the enable round-trip would otherwise land before the
        # handler exists and be silently dropped.
        tab.add_handler(cdp.network.ResponseReceived, on_response)
        await tab.send(cdp.network.enable())

        # Non-blocking navigate — does NOT wait for load. Bound the send
        # itself so a stalled websocket cannot hang here indefinitely.
        nav_result = await asyncio.wait_for(
            tab.send(cdp.page.navigate(url=url)), timeout=30
        )
        # cdp.page.navigate returns (frameId, loaderId, errorText). Chrome
        # populates errorText for network-stack failures it can classify
        # synchronously (e.g. net::ERR_CONNECTION_CLOSED, ERR_NAME_NOT_RESOLVED
        # on some platforms); other failure modes leave it None and fall
        # through to the response-wait timeout below. This branch is the
        # only failure surface for the former class — do not remove.
        if nav_result[2]:
            raise ValueError(f"Navigation failed: {nav_result[2]}")
        nav_loader_id = nav_result[1]
        # Drain anything that arrived before nav_loader_id was set. Match
        # on the buffered event's loader_id (Response does not carry
        # loader_id; the ResponseReceived event does).
        for ev in candidates:
            if ev.loader_id == nav_loader_id and not main_response.done():
                main_response.set_result(ev.response)
                break

        response = await asyncio.wait_for(main_response, timeout=30)
        ct = _normalize_ct(
            response.headers.get("content-type")
            or response.headers.get("Content-Type")
        )

        if ct == "application/pdf":
            # Forward Chrome's cookies to the urllib refetch so cookie-
            # gated PDFs (signed-URL CDNs, session-token-protected docs)
            # still resolve. Two defenses keep urllib's static Cookie
            # header from leaking credentials to a different origin:
            #   1. Scope to response.url (Chrome's final URL after
            #      redirects), so get_cookies returns only cookies
            #      applicable to that domain.
            #   2. Block redirects in the refetch (follow_redirects=False)
            #      — Chrome already resolved them, and any redirect now
            #      would re-send our Cookie header to a new origin.
            pdf_url = response.url or url
            # 10s cap matches the other bounded CDP sends in this function.
            # On timeout, fall back to a no-cookie refetch (still
            # redirect-blocked) rather than failing the whole fetch — the
            # request may still succeed for non-cookie-gated PDFs.
            try:
                cookies = await asyncio.wait_for(
                    tab.send(cdp.network.get_cookies(urls=[pdf_url])),
                    timeout=10,
                )
                cookie_header = "; ".join(
                    f"{c.name}={c.value}" for c in cookies
                )
            except asyncio.TimeoutError:
                cookie_header = ""
            # Close the tab *before* urllib so Chrome stops holding the
            # in-flight PDF download. _close_tab_quietly only swallows
            # TimeoutError; any other failure (e.g. dead-connection error
            # mid-close) still propagates, so set tab_closed in a finally
            # to keep the outer finally from double-closing.
            try:
                await _close_tab_quietly(tab)
            finally:
                tab_closed = True
            # Force "application/pdf" so downstream routes via the PDF
            # branch even if urllib's Content-Type differs (redirect / CDN
            # strips the header). Hard-code "utf-8" to match the original's
            # behavior: the charset is only consulted by
            # _decode_pdf_or_fallback_html when the body is NOT a real PDF
            # (e.g. paywall HTML served under application/pdf); preserving
            # the literal avoids a silent decode-behavior change there.
            extra = {"Cookie": cookie_header} if cookie_header else None
            _, body, _ = await asyncio.to_thread(
                _fetch_url, pdf_url, extra, False  # follow_redirects=False
            )
            return "application/pdf", body, "utf-8"

        if ct != "text/html":
            raise ValueError(f"Unsupported content-type: {ct!r}")

        # Bounded load wait — replaces page.wait_for_load_state("load")'s
        # implicit 30s timeout.
        await asyncio.wait_for(tab.wait(), timeout=30)
        html_str = await tab.get_content()
        return ct, html_str.encode("utf-8"), "utf-8"
    finally:
        if not tab_closed:
            await _close_tab_quietly(tab)


async def _fetch_one(
    url: str,
    tmpdir: Path,
    browser: zd.Browser | None,
    progress_callback: Callable[[str], None] | None,
    warning_callback: Callable[[str], None] | None,
    semaphore: asyncio.Semaphore,
    mapping: dict[str, str] | None,
) -> None:
    async with semaphore:
        if progress_callback:
            progress_callback(f"Fetching {url}...")
        try:
            if browser is not None:
                ct, body, charset = await _fetch_page(browser, url)
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

    async def _run(browser: zd.Browser | None) -> None:
        tasks = [
            _fetch_one(
                url, tmpdir, browser, progress_callback, warning_callback,
                semaphore, mapping,
            )
            for url in urls
        ]
        await asyncio.gather(*tasks)

    # Lazy import: pulls in websockets + CDP binding modules. Wasted cost
    # for CLI commands that never touch websearch (e.g. `chunkhound index`).
    import zendriver as zd

    # Must run before any tab.send() creates a Transaction.
    _install_late_completion_guard()

    # _resolve_chrome_path returns None for every "no usable Chrome >=124"
    # case (not installed, too old, --version probe failed) and emits its
    # own warning describing the cause. urllib is the unified fallback —
    # we never hand a bad binary to zendriver, so the silent
    # Response.charset parse-failure loop is never reached.
    chrome_path = _resolve_chrome_path(warning_callback)
    if chrome_path is None:
        await _run(None)
        return

    # --headless=new is required for the PDF path: legacy --headless hands
    # PDFs to Chrome's internal viewer and never exposes the response to
    # _fetch_page. Pass both headless=True and the explicit flag — the
    # relationship between zendriver's Config flag and the explicit arg is
    # undocumented; belt-and-braces.
    # --disable-dev-shm-usage: containers default /dev/shm to 64MB, which
    # 5-way concurrent navigation of JS-heavy SPAs exhausts — Chrome dies
    # and every in-flight tab raises ConnectionClosedError on its CDP
    # WebSocket. --disable-gpu drops the unused GPU process to free RAM.
    try:
        browser = await zd.start(
            headless=True,
            browser_args=[
                "--headless=new",
                "--disable-dev-shm-usage",
                "--disable-gpu",
            ],
            browser_executable_path=chrome_path,
        )
    except Exception as e:
        if warning_callback:
            warning_callback(
                f"Browser launch failed: {e}. Falling back to urllib."
                " (If Google Chrome is not installed, install it to"
                " enable rich page fetches.)"
            )
        await _run(None)
        return
    try:
        await _run(browser)
    finally:
        # Bounded best-effort stop. browser.stop() can wedge on a stuck
        # websocket close or a Chrome process ignoring SIGTERM; the subprocess
        # process-group reaps any orphans when _quickresearch exits.
        try:
            await asyncio.wait_for(browser.stop(), timeout=10)
        except asyncio.TimeoutError:
            pass


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
    source = config.config_file or config.local_config_file
    if source is not None:
        cmd.extend(["--config", str(Path(source).resolve())])
    return cmd
