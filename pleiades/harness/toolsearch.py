"""
toolsearch.py — scale to large tool fleets without drowning the model.

When Pleiades mounts dozens of MCP tools, sending every schema on every turn
bloats context and scatters the model's attention. Tool-search fixes that with
three small, always-present tools and on-demand schema loading:

  find_tools(query)        -> ranked tools + their argument schemas (discover)
  list_catalog()           -> the full inventory, name — description
  call_tool(name, args)    -> invoke any tool by name, through the SAME gate

The model discovers what it needs, then calls it — so context holds ~3 tool
schemas instead of 30+. Ranking is pure stdlib keyword overlap (no embedding
model, no service) — Pleiades stays independent and works offline.
"""

from __future__ import annotations

import json
import re

from .tools import tool, registry, Tool

_WORD = re.compile(r"[a-z0-9]{2,}")
_CTX: dict[str, object] = {"approve": None, "allowed": None}


def bind_dispatch(approve, allowed: "set[str] | None" = None) -> None:
    """Bind the permission gate (and optional allow-list) for this run.

    allowed: tool names this agent may discover/call. None = no restriction.
    A restricted subagent passes its own tool set so call_tool can't reach
    tools outside that sandbox, even when tool-search is active.
    """
    _CTX["approve"] = approve
    _CTX["allowed"] = set(allowed) if allowed is not None else None


def _discoverable() -> list[Tool]:
    # everything except the discovery plumbing itself, restricted to the
    # active agent's allow-list when one is bound.
    allowed = _CTX.get("allowed")
    return [t for t in registry.all()
            if "search" not in t.tags
            and (allowed is None or t.name in allowed)]


def _rank(query: str) -> list[Tool]:
    q = set(_WORD.findall(query.lower()))
    if not q:
        return _discoverable()
    scored: list[tuple[int, Tool]] = []
    for t in _discoverable():
        hay = f"{t.name} {t.description} {' '.join(t.tags)}".lower()
        words = set(_WORD.findall(hay))
        score = len(q & words)
        score += sum(2 for w in q if w in t.name.lower())  # name match weighs more
        if score:
            scored.append((score, t))
    scored.sort(key=lambda x: -x[0])
    return [t for _, t in scored]


@tool(safe=True, tags=("search",))
def find_tools(query: str, k: int = 8) -> str:
    """Search the full tool catalog for tools relevant to a goal. Call this FIRST to discover capabilities, then run one with call_tool.

    query: what you want to accomplish (keywords or a short phrase).
    k: maximum tools to return (default 8).
    """
    hits = _rank(query)[:max(1, int(k))]
    if not hits:
        return (f"No tools matched '{query}'. Try broader keywords, "
                "or call list_catalog for the full inventory.")
    out = []
    for t in hits:
        out.append(json.dumps({
            "tool": t.name,
            "description": t.description,
            "arguments": t.schema.get("properties", {}),
            "required": t.schema.get("required", []),
            "gated": not t.safe,
        }))
    return "\n".join(out)


@tool(safe=True, tags=("search",))
def list_catalog() -> str:
    """List every available tool as 'name — description', one per line. Use when find_tools misses or you want the whole inventory."""
    return "\n".join(f"{t.name} — {t.description}" for t in _discoverable())


@tool(safe=True, tags=("search",))
def call_tool(name: str, arguments: dict) -> str:
    """Invoke a tool discovered via find_tools by its exact name. Gated tools still pass through the permission policy.

    name: the tool name, e.g. "mcp.nyzkh-web.web_search".
    arguments: an object of argument name -> value matching the tool's schema.
    """
    from .agent import execute_tool  # shared permission gate + error handling

    allowed = _CTX.get("allowed")
    if allowed is not None and name not in allowed:
        return (f"Error: tool '{name}' is not available to this agent. "
                "Use find_tools to see what you can call.")
    t = registry.get(name)
    if t is None:
        return (f"Error: no such tool '{name}'. "
                "Use find_tools to discover exact names.")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except Exception:
            arguments = {}
    out, err = execute_tool(t, arguments or {}, _CTX["approve"])
    return out
