"""Web search via a local SearXNG instance (JSON API).

The SearXNG instance MUST have JSON output enabled (`search.formats: [html, json]`)
or the API returns 403. See services/searxng/settings.yml.
"""

from __future__ import annotations

from . import Tool, ToolContext


class SearchTool(Tool):
    safe = True   # read-only
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
        # Single implementation: delegate to the harness web_search so the chat path
        # and the work path share one SearXNG search. The harness resolves the
        # endpoint from the unified config (Settings.searxng_url).
        num_results = max(1, min(int(num_results), 10))
        try:
            from ..harness.builtins.web import web_search as _web_search, bind_searxng
            # Make sure the shared impl points at this run's configured instance.
            bind_searxng(ctx.settings.searxng_url)
            return _web_search(query, count=num_results)
        except Exception as e:  # never crash the agent loop
            return f"[search error] {e}"
