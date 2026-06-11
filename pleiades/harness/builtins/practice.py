"""Self-directed practice — the character decides when to get better.

Tool calling is a skill, and skills improve with deliberate practice. These
tools let the MODEL choose to train: check its own track record, study a tool
it fumbles, try it, and write the lesson to long-term memory — where anamnesis
recall will surface it the next time a similar task comes up. Learning from
experience, stored where experience already lives.

Stats come from execute_tool (agent.py), which records every invocation's
outcome to ~/.pleiades/tool_stats.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..tools import registry, tool

STATS_PATH = Path.home() / ".pleiades" / "tool_stats.json"


def _load_stats() -> dict:
    try:
        return json.loads(STATS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def record_outcome(tool_name: str, ok: bool, error: str = "") -> None:
    """Called by execute_tool after every invocation. Must never raise."""
    try:
        stats = _load_stats()
        s = stats.setdefault(tool_name, {"ok": 0, "err": 0, "last_errors": []})
        if ok:
            s["ok"] += 1
        else:
            s["err"] += 1
            s["last_errors"] = (s.get("last_errors", []) + [error[:200]])[-3:]
        STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATS_PATH.write_text(json.dumps(stats, indent=1), encoding="utf-8")
    except Exception:
        pass


@tool(safe=True, tags=("practice",))
def practice_status() -> str:
    """Review your own tool-calling track record: which tools you use well and which you fumble. Call this when you feel unsure about your tools, or just periodically to see where practice would help."""
    stats = _load_stats()
    if not stats:
        return ("No tool usage recorded yet. Your record builds as you work. "
                "You can still study any tool with study_tool(name).")
    rows = []
    for name, s in sorted(stats.items(),
                          key=lambda kv: -(kv[1]["err"] / max(1, kv[1]["ok"] + kv[1]["err"]))):
        total = s["ok"] + s["err"]
        rate = s["ok"] / total
        flag = "  ← practice this" if s["err"] >= 2 and rate < 0.8 else ""
        rows.append(f"- {name}: {s['ok']}/{total} ok ({rate:.0%}){flag}")
    return ("Your tool track record (worst first):\n" + "\n".join(rows)
            + "\n\nTo improve: study_tool(<name>), try a harmless real call, "
              "then record_lesson(<name>, <what you learned>).")


@tool(safe=True, tags=("practice",))
def study_tool(tool_name: str) -> str:
    """Study one tool before using or practicing it: full parameter schema, your recent mistakes with it, and how to drill it. Call this whenever a tool call of yours failed, or before using an unfamiliar tool.

    tool_name: the tool to study.
    """
    t = registry.get(tool_name)
    if t is None:
        names = [x.name for x in registry.all() if tool_name.lower() in x.name.lower()]
        return (f"No tool named '{tool_name}'."
                + (f" Close matches: {', '.join(names[:5])}" if names else ""))
    s = _load_stats().get(tool_name, {})
    mistakes = ""
    if s.get("last_errors"):
        mistakes = "\nYour recent errors with it:\n" + "\n".join(
            f"  - {e}" for e in s["last_errors"])
    return (f"{t.name}: {t.description}\n"
            f"Parameters (JSON schema): {json.dumps(t.schema)}\n"
            f"Record: {s.get('ok', 0)} ok / {s.get('err', 0)} failed.{mistakes}\n\n"
            f"Drill: invoke {t.name} now with a small, harmless, real example. "
            f"Check the result, then save what you learned with "
            f"record_lesson('{t.name}', ...) so future-you recalls it.")


@tool(tags=("practice", "memory"))
def record_lesson(tool_name: str, lesson: str) -> str:
    """Save a lesson you learned about using a tool. The lesson goes to long-term memory, so you will recall it whenever similar work comes up — this is how you get permanently better.

    tool_name: which tool the lesson is about.
    lesson: the concrete takeaway, e.g. the argument format that works, a pitfall, a pattern that succeeded.
    """
    from .memory import _store

    store = _store()
    if store is None:
        return "Error: memory not bound — lesson not saved."
    slug = store.write(f"tool-lesson-{tool_name}", lesson, kind="reference")
    return (f"Lesson saved as '{slug}'. It will surface via recall when you "
            f"work with {tool_name} again.")
