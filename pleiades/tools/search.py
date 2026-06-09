"""Web search via a local SearXNG instance (JSON API).

The SearXNG instance MUST have JSON output enabled (`search.formats: [html, json]`)
or the API returns 403. See services/searxng/settings.yml.
"""

from __future__ import annotations

from . import Tool, ToolContext


class SearchTool(Tool):
    name = "search"
    description = (
        "Search the web and return the most relevant results (title, URL, snippet). "
        "Use this to find current information, documentation, or pages to open in the browser."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "num_results": {
                "type": "integer",
                "description": "How many results to return (default 5, max 10).",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    def run(self, ctx: ToolContext, query: str, num_results: int = 5) -> str:
        num_results = max(1, min(int(num_results), 10))
        url = ctx.settings.searxng_url.rstrip("/") + "/search"
        try:
            resp = ctx.http.get(
                url,
                params={"q": query, "format": "json"},
                headers={"Accept": "application/json"},
            )
        except Exception as e:
            return f"[search error] could not reach SearXNG at {url}: {e}"

        if resp.status_code == 403:
            return (
                "[search error] SearXNG returned 403. Enable JSON output in settings.yml "
                "(`search.formats: [html, json]`) and restart the service."
            )
        if resp.status_code != 200:
            return f"[search error] SearXNG returned {resp.status_code}."

        try:
            results = resp.json().get("results", [])
        except ValueError:
            return "[search error] SearXNG did not return JSON (is the json format enabled?)."

        if not results:
            return f"No results for: {query}"

        lines = []
        for i, r in enumerate(results[:num_results], 1):
            title = r.get("title", "(no title)")
            link = r.get("url", "")
            snippet = (r.get("content") or "").strip().replace("\n", " ")
            lines.append(f"{i}. {title}\n   {link}\n   {snippet}")
        return "\n".join(lines)
