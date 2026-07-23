"""
subagent.py - let the main agent dispatch focused workers.

dispatch_subagent: sequential; one focused sub-task, cheapest model that can do it.
dispatch_subagents_parallel: fan-out; multiple tasks run concurrently via threads.

Each subagent runs the same Agent loop with a narrower tool set and its own tier,
then returns only its final answer - keeping the parent's context clean.
"""

from __future__ import annotations

import concurrent.futures

from .config import Config
from .tools import tool, registry


ROLE_TOOLS: dict[str, dict] = {
    "research": {"tier": "research", "tags": ["web", "read"]},
    "coder":    {"tier": "coder",    "tags": ["file", "shell", "read"]},
    "fast":     {"tier": "fast",     "tags": ["read", "web"]},
    "general":  {"tier": "chat",     "tags": None},
}

_CTX: dict[str, object] = {"cfg": None, "depth": 0, "approve": None}


def bind_context(cfg: Config, depth: int = 0, approve=None) -> None:
    """Called by the parent Agent so subagents inherit config and grants."""
    _CTX["cfg"] = cfg
    _CTX["depth"] = depth
    _CTX["approve"] = approve


@tool(safe=True, tags=("meta",))
def dispatch_subagent(role: str, task: str) -> str:
    """Delegate a self-contained sub-task to a focused subagent and return its result.

    role: one of "research" (web + read tools), "coder" (files + shell),
          "fast" (read + web, smallest tier), or "general" (full tools).
          Each role runs on the like-named tier (configurable in config.json;
          local by default).
    task: complete standalone description - the subagent has none of this conversation's context.
    """
    from .agent import Agent

    cfg: Config = _CTX["cfg"]  # type: ignore[assignment]
    depth: int = _CTX["depth"]  # type: ignore[assignment]
    if cfg is None:
        return "Error: subagent context not bound."
    if depth >= cfg.max_subagent_depth:
        return (f"Error: max subagent depth ({cfg.max_subagent_depth}) reached; "
                "do this work directly instead of delegating.")

    spec = ROLE_TOOLS.get(role, ROLE_TOOLS["general"])
    tools = registry.all() if spec["tags"] is None else (
        registry.select(tags=spec["tags"]) + registry.select(names=["dispatch_subagent"])
    )

    sub = Agent(cfg, tools=tools, tier_name=spec["tier"], depth=depth + 1,
                approve=_CTX["approve"])  # type: ignore[arg-type]
    result = sub.run(task)
    return f"[subagent:{role}] {result.answer}"


@tool(safe=True, tags=("meta",))
def dispatch_subagents_parallel(tasks: list) -> str:
    """Fan out multiple sub-tasks to subagents running concurrently, then return all results.

    Dramatically faster than sequential dispatch_subagent calls for tasks that need
    information from multiple independent sources simultaneously.

    tasks: list of dicts, each with 'role' and 'task' keys.
           Pick the cheapest role that can do each job.
    """
    from .agent import Agent

    cfg: Config = _CTX["cfg"]  # type: ignore[assignment]
    depth: int = _CTX["depth"]  # type: ignore[assignment]
    if cfg is None:
        return "Error: subagent context not bound."
    if depth >= cfg.max_subagent_depth:
        return f"Error: max subagent depth ({cfg.max_subagent_depth}) reached."
    if not isinstance(tasks, list) or not tasks:
        return "Error: tasks must be a non-empty list of dicts with 'role' and 'task' keys."

    # Task-list tools (builtins/tasks.py) scope by job id via thread-local
    # storage, so a fan-out ThreadPoolExecutor -- new OS threads -- would
    # otherwise lose that binding for every worker. Capture it here (the
    # calling thread) and re-bind it inside each worker before it runs.
    from .builtins.tasks import bind_job, current_job_id
    _job_id = current_job_id()

    def _run_one(item: dict) -> str:
        if _job_id:
            bind_job(_job_id)
        role = item.get("role", "general")
        task = item.get("task", "")
        if not task:
            return "[skipped - empty task]"
        spec = ROLE_TOOLS.get(role, ROLE_TOOLS["general"])
        tools = registry.all() if spec["tags"] is None else (
            registry.select(tags=spec["tags"]) +
            registry.select(names=["dispatch_subagent"])
        )
        sub = Agent(cfg, tools=tools, tier_name=spec["tier"], depth=depth + 1,
                    approve=_CTX["approve"])  # type: ignore[arg-type]
        result = sub.run(task)
        return f"[{role}] {result.answer}"

    max_workers = min(len(tasks), cfg.max_subagent_depth * 2, 8)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_run_one, t) for t in tasks]
        results = []
        for i, fut in enumerate(futures):
            try:
                results.append(f"### Task {i+1}\n{fut.result(timeout=300)}")
            except Exception as e:
                results.append(f"### Task {i+1}\n[error: {e}]")

    return "\n\n".join(results)
