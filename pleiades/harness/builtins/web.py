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

import json
import os
import re
import urllib.parse
import urllib.request

from ..tools import tool

_SEARXNG = os.environ.get("SEARXNG_URL", "http://localhost:8888")
_UA = "Pleiades/0.1 (+agent workspace)"
_TAG_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
_HTML_RE = re.compile(r"<[^>]+>")


def _get(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def _strip_html(html: str) -> str:
    text = _TAG_RE.sub(" ", html)
    text = _HTML_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


@tool(safe=True, tags=("web", "read"))
def web_search(query: str, count: int = 6) -> str:
    """Search the web via SearXNG and return ranked title/url/snippet results.

    query: the search query.
    count: how many results to return (default 6).
    """
    params = urllib.parse.urlencode({"q": query, "format": "json"})
    try:
        data = json.loads(_get(f"{_SEARXNG}/search?{params}"))
    except Exception as e:
        return (f"Error reaching SearXNG at {_SEARXNG} ({e}). "
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
        html = _get(url)
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
        data = json.loads(_get(f"{_SEARXNG}/search?{params}"))
    except Exception as e:
        return (f"Error reaching SearXNG at {_SEARXNG} ({e}). "
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
            html = _get(url, timeout=20)
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
