"""
Web tools — SearXNG search + page fetch + deep research.

SearXNG is a self-hosted meta-search engine (no API key, privacy-friendly). Point
PLEIADES at your instance via config `searxng_url` or env SEARXNG_URL; defaults to
a local instance on :8888.

Tool guide:
  web_search    — get ranked titles/urls/snippets for a query.
  http_fetch    — fetch one URL and return its readable text.
  deep_research — search + fetch N sources in one call; returns a structured
                  multi-source report ready for the model to synthesize.
                  Start here for any research task.
"""

from __future__ import annotations

import html as _htmllib
import http.client
import ipaddress
import json
import os
import re
import socket
import threading
import urllib.parse
import urllib.request

from ..tools import tool

# SSRF guard for http_fetch/deep_research: these tools are safe=True (no
# approval prompt) and take a model-supplied URL, which is reachable via
# untrusted content (a web page, an email, a Discord message) through prompt
# injection with no user in the loop. Without this, a fetch to
# 127.0.0.1:9000 (Anamnesis's unauthenticated control API) or the webui's
# loopback-guarded API (whose guard only checks Host/Origin -- a server-side
# fetch sends no Origin) hands another character's vault secrets/API keys to
# the model. Resolve-then-pin-the-validated-IP closes the DNS-rebind gap
# (checking the hostname alone isn't enough -- a second resolution at
# connect time could return a different, unsafe address).
_MAX_FETCH_BYTES = 2_000_000


class _SSRFBlocked(Exception):
    pass


def _assert_public_ip(ip_text: str, host: str) -> None:
    ip = ipaddress.ip_address(ip_text)
    if (ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
        raise _SSRFBlocked(f"refusing to fetch {host} — resolves to non-public address {ip}")


def _resolve_pinned(host: str) -> str:
    """Resolve `host`, reject any non-public address, return one validated IP."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise _SSRFBlocked(f"could not resolve host {host!r}: {e}") from e
    if not infos:
        raise _SSRFBlocked(f"could not resolve host {host!r}: no addresses")
    for _family, _type, _proto, _canon, sockaddr in infos:
        _assert_public_ip(sockaddr[0], host)
    return infos[0][4][0]


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """Connects to a pre-resolved, pre-validated IP instead of re-resolving
    `host` — re-resolving would reopen the DNS-rebind gap the check above
    exists to close."""
    def __init__(self, host, pinned_ip, *a, **kw):
        super().__init__(host, *a, **kw)
        self._pinned_ip = pinned_ip

    def connect(self):
        self.sock = socket.create_connection((self._pinned_ip, self.port), self.timeout)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host, pinned_ip, *a, **kw):
        super().__init__(host, *a, **kw)
        self._pinned_ip = pinned_ip

    def connect(self):
        sock = socket.create_connection((self._pinned_ip, self.port), self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)

# SearXNG endpoint resolves from the unified project config (Settings.searxng_url,
# which honors PLEIADES_SEARXNG_URL). SEARXNG_URL stays supported for back-compat,
# and bind_searxng() lets a run override it explicitly.
_BOUND: dict[str, str | None] = {"searxng": None}


def bind_searxng(url: str) -> None:
    """Pin the SearXNG base URL for this run (overrides config/env)."""
    _BOUND["searxng"] = url.rstrip("/") if url else None


def _searxng_base() -> str:
    if _BOUND["searxng"]:
        return _BOUND["searxng"]
    env = os.environ.get("SEARXNG_URL")
    if env:
        return env.rstrip("/")
    try:
        from ..config import Config  # unified Settings
        return Config.load().searxng_url.rstrip("/")
    except Exception:
        return "http://localhost:8888"
_UA = "Pleiades/0.1 (+agent workspace)"
# Whole blocks that are almost never content — drop them outright (with contents).
_DROP_RE = re.compile(
    r"<(script|style|noscript|template|svg|nav|header|footer|aside|form)\b[^>]*>.*?</\1>",
    re.S | re.I)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
# Closing block tags become line breaks so structure survives the strip.
_BLOCK_RE = re.compile(r"</(p|div|li|ul|ol|h[1-6]|section|article|tr|table|blockquote)\s*>", re.I)
_BR_RE = re.compile(r"<br\s*/?>", re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_BLANKS_RE = re.compile(r"\n{3,}")


def _get(url: str, timeout: int = 30) -> str:
    """Fetch a trusted, admin-configured endpoint (SearXNG). NOT for
    model-supplied URLs -- see _get_external for that (SSRF-guarded).
    SearXNG is legitimately loopback by default, which is exactly what
    _get_external exists to block."""
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read(_MAX_FETCH_BYTES + 1)
        return body[:_MAX_FETCH_BYTES].decode("utf-8", errors="replace")


def _get_external(url: str, timeout: int = 30, _redirects: int = 0) -> str:
    """Fetch a model-supplied URL (http_fetch/deep_research). SSRF-guarded:
    resolves the host, rejects any non-public address, and connects to the
    validated IP directly rather than letting the connection re-resolve
    (which would reopen a DNS-rebind gap between check and connect).
    Redirects are followed manually so each hop is re-validated the same
    way -- otherwise a URL that passes validation could 302 straight to an
    internal address."""
    if _redirects > 5:
        raise _SSRFBlocked("too many redirects")
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise _SSRFBlocked(f"unsupported scheme: {parts.scheme!r}")
    host = parts.hostname
    if not host:
        raise _SSRFBlocked("URL has no host")
    pinned_ip = _resolve_pinned(host)
    port = parts.port or (443 if parts.scheme == "https" else 80)
    conn_cls = _PinnedHTTPSConnection if parts.scheme == "https" else _PinnedHTTPConnection
    conn = conn_cls(host, pinned_ip, port=port, timeout=timeout)
    try:
        path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        conn.request("GET", path, headers={"User-Agent": _UA, "Host": host})
        resp = conn.getresponse()
        if resp.status in (301, 302, 303, 307, 308):
            location = resp.headers.get("Location")
            resp.read()  # drain so the connection can close cleanly
            if not location:
                raise _SSRFBlocked("redirect with no Location header")
            next_url = urllib.parse.urljoin(url, location)
            return _get_external(next_url, timeout=timeout, _redirects=_redirects + 1)
        body = resp.read(_MAX_FETCH_BYTES + 1)
        return body[:_MAX_FETCH_BYTES].decode("utf-8", errors="replace")
    finally:
        conn.close()


def _strip_html(html: str) -> str:
    """HTML -> readable text, keeping paragraph structure and dropping the usual
    chrome (scripts, styles, nav/header/footer/aside/forms). Decodes entities so
    the model reads "don't" not "don&#39;t". Much higher signal-per-token than a
    flat tag strip — which is what http_fetch and deep_research feed the model."""
    html = _COMMENT_RE.sub(" ", html)
    html = _DROP_RE.sub(" ", html)
    html = _BR_RE.sub("\n", html)
    html = _BLOCK_RE.sub("\n", html)
    text = _TAG_RE.sub(" ", html)
    text = _htmllib.unescape(text)
    # Tidy per line, drop empties, and cap runs of blank lines.
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    return _BLANKS_RE.sub("\n\n", text).strip()


@tool(safe=True, tags=("web", "read"))
def web_search(query: str, count: int = 6) -> str:
    """Search the web via SearXNG and return ranked title/url/snippet results.

    query: the search query.
    count: how many results to return (default 6).

    Resilience ladder (appended 2026-08-26 after live failure): agentic
    research fires searches in bursts, and upstream engines answer bursts
    with CAPTCHAs / 429s ("Suspended" in SearXNG's unresponsive_engines) —
    after which EVERY query returns zero and a bare "(no results)" reads to
    a model as "no evidence exists", which is a lie. The tool therefore:
      1. self-throttles (min gap between SearXNG hits) to avoid earning new
         suspensions;
      2. on an empty result set, retries progressively simplified variants
         of the query (de-quoted → keyword-core) with short backoffs —
         quoted phrases alone often zero out niche queries;
      3. reports infrastructure state HONESTLY on final failure: which
         engines are suspended and that this is transient throttling, NOT
         evidence about the topic. Works identically for any model.
    """
    results, infra, served = _searxng_search(query)
    if not results:
        # Empty even after the retry ladder. Distinguish "the web has
        # nothing" from "our search engines are temporarily blinded".
        if infra:
            return (
                f'(no results for {query!r} — SEARCH INFRASTRUCTURE IMPAIRED, '
                f'not proof of absence: {infra}. Wait ~60s and retry, or '
                f'rephrase shorter without quotes; do not conclude the topic '
                f'is undocumented.)'
            )
        return f"(no results for: {query!r})"
    trimmed = results[:count]
    out = []
    for i, r in enumerate(trimmed, 1):
        eng = (r.get("engine") or "").strip()
        tag = f" [{eng}]" if eng else ""
        out.append(f"{i}.{tag} {r.get('title','')}\n   {r.get('url','')}\n"
                   f"   {(r.get('content','') or '')[:200]}")
    # Degraded-coverage detector. Two live failure shapes, one message:
    #   a) only auxiliary engines answered (all majors suspended), or
    #   b) a major engine answered, but with source-collapsed noise —
    #      e.g. bing serving four wikipedia/YouTube hits for a WTC query
    #      while its real index was throttled.
    # Signal for (b): the whole result set collapses onto <=2 domains.
    # Either way the model must know coverage is broken, or it treats the
    # noise as "the web's answer" (seen live: 'Larry Page / Larry the cat'
    # for a Silverstein/WTC query).
    engines_used = {(r.get("engine") or "").lower() for r in trimmed}
    domains = set()
    for r in trimmed:
        try:
            d = urllib.parse.urlparse(r.get("url", "")).netloc.lower()
            if d.startswith("www."):
                d = d[4:]
            if d:
                domains.add(d)
        except Exception:
            continue
    coverage_broken = bool(infra) and (
        not (engines_used & _MAJOR_ENGINES) or len(domains) <= 2
    )
    if coverage_broken:
        out.append(
            f'[DEGRADED COVERAGE: results collapsed onto {len(domains)} '
            f'source(s) ({", ".join(sorted(domains)) or "?"}) because major '
            f'web engines are temporarily blocked: {infra}. These hits may '
            f'not address your actual question even when they look '
            f'confident. Retry the same search in ~2 minutes for full-web '
            f'results before drawing conclusions.]'
        )
    elif served.strip() != query.strip():
        # A simplified variant won. Tell the model exactly what was searched:
        # if these hits look off-topic, the fix is a fuller rephrase (add the
        # disambiguating words), not concluding the original question is
        # unanswerable.
        out.append(f'[searched as: "{served}" — if these results are '
                   f'off-topic, rephrase with more specific terms]')
    if infra:
        # Results came through despite partial suspension — say so once,
        # briefly, so the model knows coverage may be narrower than usual.
        out.append(f"[note: some search engines temporarily unavailable ({infra}); "
                   f"coverage may be partial]")
    return "\n".join(out)


# ── SearXNG resilience core ──────────────────────────────────────────────────
# Shared by web_search and deep_research. Process-wide throttle: agents fire
# search bursts; every major engine rate-limits bursts; one process-level
# minimum gap is the cheapest prevention (works for any model/tool caller).
_SEARXNG_MIN_GAP = 1.0          # seconds between instance hits, process-wide
_last_searxng_hit = [0.0]
_throttle_lock = threading.Lock()

_RETRY_BACKOFF = (1.5, 3.0)     # sleep before 2nd/3rd attempt


def _throttle() -> None:
    import time as _t
    with _throttle_lock:
        now = _t.monotonic()
        wait = _last_searxng_hit[0] + _SEARXNG_MIN_GAP - now
        if wait > 0:
            _t.sleep(wait)
        _last_searxng_hit[0] = _t.monotonic()


_QUOTES_RE = re.compile(r'["\u201c\u201d]')
_WS_RE = re.compile(r"\s+")
# Only true filler grammar gets dropped from the keyword-core variant.
# Numbers and short tokens are NEVER dropped: "building 7" losing its "7"
# turned a 9/11 query into a Canadian post-hardcore band's discography
# (seen live). Entity-bearing nouns stay; only these connectives go.
_STOPWORDS = frozenset((
    "the a an of in on for to and or is are was were what why how does did "
    "do its it's his her their about at as by from with that this these "
    "those it he she they them you i we us our your my me be been being"
).split())


def _keyword_core(query: str, max_words: int = 8) -> str:
    words = [w for w in _QUOTES_RE.sub(" ", query).split()
             if w.lower() not in _STOPWORDS]
    return _WS_RE.sub(" ", " ".join(words[:max_words])).strip()


def _query_variants(query: str) -> list[str]:
    """Progressively simpler variants of the same information need.

    Attempt order: verbatim → de-quoted (quoted phrases are the single most
    common cause of legitimate-looking zero-result sets) → keyword core
    (filler grammar dropped; numbers and entities preserved). Deduped,
    original first, max 3."""
    seen: list[str] = [query.strip()]
    deq = _WS_RE.sub(" ", _QUOTES_RE.sub(" ", query)).strip()
    if deq and deq.lower() != seen[0].lower():
        seen.append(deq)
    core = _keyword_core(deq or query)
    if core and all(core.lower() != s.lower() for s in seen):
        seen.append(core)
    return seen[:3]


def _unresponsive_note(data: dict) -> str:
    """Human/model-readable summary of engines SearXNG could not use this
    round ('Suspended: CAPTCHA', 'timeout', ...). Empty string when all
    engines answered."""
    bad = data.get("unresponsive_engines") or []
    if not isinstance(bad, list):
        return ""
    parts = []
    for entry in bad:
        try:
            name, reason = str(entry[0]), str(entry[1])[:40]
        except Exception:
            continue
        parts.append(f"{name}: {reason}")
    return "; ".join(parts)


def _searxng_search(query: str, timeout: int = 20) -> tuple[list, str, str]:
    """Run the resilience ladder. Returns (results, infra_note, served_query)
    where infra_note summarizes suspended engines from the LAST attempt
    (empty when everything answered) and served_query is the variant that
    produced the returned results (== query unless a simplified variant won,
    which callers surface so the model knows its search was rewritten).
    Never raises for query-level failures."""
    import time as _t

    # Repeat-query cache: agents re-search near-identical strings within a
    # turn ("Silverstein pull it", "Silverstein pull it building 7", ...).
    # Every redundant hit feeds the upstream rate-limiters that blind the
    # whole pool (the 2026-08-26 incident). 5-minute TTL, small cap — this
    # is dedup, not a search history.
    cache_key = _WS_RE.sub(" ", query.strip().lower())
    now = _t.monotonic()
    hit = _QUERY_CACHE.get(cache_key)
    if hit and now - hit[0] < _QUERY_CACHE_TTL:
        return hit[1], hit[2], hit[3]

    last_results: list = []
    last_note = ""
    variants = _query_variants(query)
    attempts = len(variants)

    def _run(q: str):
        _throttle()
        params = urllib.parse.urlencode({"q": q, "format": "json"})
        try:
            return json.loads(_get(f"{_searxng_base()}/search?{params}",
                                   timeout=timeout))
        except Exception:
            return None

    data = None
    served = variants[-1]
    for i, q in enumerate(variants):
        if i:
            _t.sleep(_RETRY_BACKOFF[min(i, len(_RETRY_BACKOFF)) - 1])
        data = _run(q)
        if data is None:
            last_note = "instance unreachable or errored this attempt"
            continue
        last_results = data.get("results", []) or []
        last_note = _unresponsive_note(data)
        if last_results:
            served = q
            break

    # Patient final attempt: engine suspensions observed live are short
    # (~180s windows) but our ladder so far spans only seconds. When the
    # pool is blinded AND every variant came back empty, one last try after
    # a real pause often catches a suspend expiry — bounded cost paid only
    # in the already-degraded case.
    if not last_results and last_note:
        _t.sleep(_PATIENT_RETRY_WAIT)
        data = _run(variants[0])
        if data is not None:
            last_results = data.get("results", []) or []
            last_note = _unresponsive_note(data) or last_note
            if last_results:
                served = variants[0]

    if not last_results and not last_note:
        _QUERY_CACHE[cache_key] = (now, [], "", served)   # genuine-empty too
        return [], "", served
    if last_results:
        _QUERY_CACHE[cache_key] = (_t.monotonic(), last_results,
                                   last_note, served)
    elif len(_QUERY_CACHE) > 64:
        _QUERY_CACHE.clear()
    return last_results, last_note, served


_QUERY_CACHE: dict[str, tuple] = {}
_QUERY_CACHE_TTL = 300.0       # seconds
_PATIENT_RETRY_WAIT = 20.0     # seconds; only on empty+impaired

# General-web crawlers: if NONE of these contributed to a result set, the
# results are auxiliary-source noise (see web_search's degraded-coverage
# detector), not evidence about the query.
_MAJOR_ENGINES = frozenset((
    "google", "bing", "brave", "duckduckgo", "startpage", "qwant",
    "mojeek", "marginalia", "presearch", "yep",
))


@tool(safe=True, tags=("web", "read"))
def http_fetch(url: str, max_chars: int = 8000) -> str:
    """Fetch a URL and return its readable text (HTML tags stripped).

    url: the page to fetch (http/https).
    max_chars: cap on returned characters.
    """
    if not url.startswith(("http://", "https://")):
        return "Error: url must start with http:// or https://"
    try:
        html = _get_external(url)
    except Exception as e:
        return f"Error fetching {url}: {e}"
    text = _strip_html(html)
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


@tool(safe=True, tags=("web", "read"))
def deep_research(query: str, num_sources: int = 4, chars_per_page: int = 3000) -> str:
    """Research a topic by searching the web and fetching multiple sources. Returns a multi-source report.

    Use this as the starting point for any research task — it replaces the
    manual loop of web_search → http_fetch → http_fetch → ...

    query: the research question or topic.
    num_sources: number of pages to fetch and include (default 4).
    chars_per_page: max characters to extract from each page (default 3000).
    """
    # 1. Search — same resilience ladder as web_search (retries, honest
    # infra reporting); deep research is exactly where bursts hurt most.
    results, infra, _served = _searxng_search(query)
    if not results:
        if infra:
            return (
                f'(no sources found for {query!r} — SEARCH INFRASTRUCTURE '
                f'IMPAIRED ({infra}). This is temporary throttling upstream, '
                f'not proof the topic is undocumented. Wait ~60s and retry.)'
            )
        return f"(no search results for: {query!r})"

    # 2. Fetch top N sources
    out: list[str] = [f"# Deep research: {query}\n"]
    fetched = 0
    for r in results:
        if fetched >= num_sources:
            break
        url = r.get("url", "")
        title = r.get("title", url)
        snippet = (r.get("content") or "")[:200]

        if not url.startswith(("http://", "https://")):
            continue

        out.append(f"---\n## Source {fetched + 1}: {title}\n{url}")
        if snippet:
            out.append(f"*Snippet:* {snippet}")

        try:
            html = _get_external(url, timeout=20)
            text = _strip_html(html)
            body = text[:chars_per_page]
            if len(text) > chars_per_page:
                body += f"\n... [{len(text) - chars_per_page} more chars — "
                body += "use http_fetch for the full page]"
            out.append(body)
        except Exception as e:
            out.append(f"*(could not fetch: {e})*")

        fetched += 1

    if fetched == 0:
        return f"(search returned results but no pages could be fetched for: {query!r})"

    out.append(f"\n---\n*{fetched} source(s) fetched. Synthesize the above to answer the query.*")
    return "\n\n".join(out)
