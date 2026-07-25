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
    """
    params = urllib.parse.urlencode({"q": query, "format": "json"})
    try:
        data = json.loads(_get(f"{_searxng_base()}/search?{params}"))
    except Exception as e:
        return (f"Error reaching SearXNG at {_searxng_base()} ({e}). "
                "Set SEARXNG_URL or start a SearXNG instance.")
    results = data.get("results", [])[:count]
    if not results:
        return "(no results)"
    out = []
    for i, r in enumerate(results, 1):
        out.append(f"{i}. {r.get('title','')}\n   {r.get('url','')}\n"
                   f"   {(r.get('content','') or '')[:200]}")
    return "\n".join(out)


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
    # 1. Search
    params = urllib.parse.urlencode({"q": query, "format": "json"})
    try:
        data = json.loads(_get(f"{_searxng_base()}/search?{params}"))
    except Exception as e:
        return (f"Error reaching SearXNG at {_searxng_base()} ({e}). "
                "Set SEARXNG_URL or start a SearXNG instance.")

    results = data.get("results", [])
    if not results:
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
