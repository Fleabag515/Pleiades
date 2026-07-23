"""
subagent.py - let the main agent dispatch focused workers.

dispatch_subagent: sequential; one focused sub-task, cheapest model that can do it.
dispatch_subagents_parallel: fan-out; multiple tasks run concurrently via threads.

Each subagent runs the same Agent loop with a narrower tool set and its own tier,
then returns only its final answer - keeping the parent's context clean.
"""

from __future__ import annotations

import concurrent.futures
import time

from .config import Config
from .tools import tool, registry


ROLE_TOOLS: dict[str, dict] = {
    "research": {"tier": "research", "tags": ["web", "read"]},
    "coder":    {"tier": "coder",    "tags": ["file", "shell", "read"]},
    "fast":     {"tier": "fast",     "tags": ["read", "web"]},
    "general":  {"tier": "chat",     "tags": None},
}

# Overall wall-clock budget for one dispatch_subagents_parallel call. Bounds
# two separate things:
#   1. each worker's own Agent.run() loop, via should_stop (polled once per
#      step -- same cooperative cancellation as every other Agent.run() call
#      in this codebase; can't interrupt a single already-in-flight blocking
#      LLM/tool call, same documented caveat as elsewhere).
#   2. how long THIS call can block the calling agent's own turn -- see
#      dispatch_subagents_parallel's use of pool.shutdown(wait=False, ...)
#      below. A plain `with ThreadPoolExecutor(...) as pool:` block calls
#      shutdown(wait=True) on exit, which blocks the CALLING thread (the
#      chat turn's own belt.dispatch()) until every worker thread finishes --
#      even a wedged one -- regardless of should_stop or any per-future
#      result() timeout in the loop above it. Bound (1) reduces how often
#      bound (2) actually matters, but (2) is what makes the calling turn's
#      unblock time deterministic even when a worker never notices
#      should_stop (a single hung network call, for instance).
PARALLEL_SUBAGENT_TIMEOUT_SECONDS = 300

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

    deadline = time.monotonic() + PARALLEL_SUBAGENT_TIMEOUT_SECONDS

    def _run_one(item: dict) -> str:
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
        result = sub.run(task, should_stop=lambda: time.monotonic() >= deadline)
        return f"[{role}] {result.answer}"

    max_workers = min(len(tasks), cfg.max_subagent_depth * 2, 8)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures = [pool.submit(_run_one, t) for t in tasks]
        results = []
        for i, fut in enumerate(futures):
            remaining = max(0.0, deadline - time.monotonic())
            try:
                results.append(f"### Task {i+1}\n{fut.result(timeout=remaining)}")
            except Exception as e:
                results.append(
                    f"### Task {i+1}\n[error: subagent did not finish within "
                    f"{PARALLEL_SUBAGENT_TIMEOUT_SECONDS}s ({e}); it may still "
                    "be running in the background but its result is not "
                    "included here]")
        return "\n\n".join(results)
    finally:
        # wait=False -- never block the CALLING turn on a wedged/slow worker
        # (see PARALLEL_SUBAGENT_TIMEOUT_SECONDS' docstring above for why the
        # default `with ThreadPoolExecutor(...) as pool:` context-manager exit
        # is unsafe here). cancel_futures drops anything not yet started;
        # should_stop (wired into each sub.run() above) makes well-behaved
        # workers already in flight wind down close to the same deadline
        # instead of running to their own step ceiling unattended.
        pool.shutdown(wait=False, cancel_futures=True)
