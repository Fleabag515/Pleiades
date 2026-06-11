"""
claude_backend.py — run the harness with Claude itself as the brain, billed to
your Claude **subscription** (Pro/Max) via the Claude Agent SDK.

Architecture (see CLAUDE_BACKEND.md): *Claude drives, Pleiades' tools are
exposed to it.* We publish the harness `registry` as an in-process MCP server,
hand the loop to the Agent SDK, and let Claude operate through the real tool
belt. Auth is whatever `claude login` holds locally — so as long as
ANTHROPIC_API_KEY is unset, usage lands on your subscription, not per-token API.

This is deliberately a *separate* path from `llm.py`'s `backend="anthropic"`
tier, which is the raw Messages API (per-token). Subscription usage must never
go through that tier (ToS: OAuth creds are for Claude Code / Claude.ai only).
"""

from __future__ import annotations

import asyncio
import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import Config
from .tools import Tool, registry
from .agent import execute_tool


# Claude Code built-in tools that are read-only — safe to auto-allow so the
# permission gate only ever stops on genuine side effects.
_SAFE_BUILTINS = {
    "Read", "Glob", "Grep", "LS", "NotebookRead", "TodoRead",
    "WebFetch", "WebSearch",
}

MCP_SERVER = "pleiades"


@dataclass
class ClaudeResult:
    answer: str
    turns: int = 0
    cost_usd: float = 0.0
    usage: dict = field(default_factory=dict)
    error: str = ""


# --------------------------------------------------------------------------- #
# Build the in-process MCP server from the Pleiades tool registry
# --------------------------------------------------------------------------- #
def _make_handler(t: Tool) -> Callable:
    """Async MCP handler that runs a sync Pleiades tool off the event loop."""

    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        # Gate is enforced upstream in can_use_tool; run unguarded here.
        out, err = await asyncio.to_thread(execute_tool, t, args, None)
        return {
            "content": [{"type": "text", "text": out or "(no output)"}],
            "is_error": err,
        }

    return handler


def build_pleiades_server(tools: list[Tool] | None = None):
    """Wrap the harness tool belt as an Agent-SDK in-process MCP server."""
    from claude_agent_sdk import tool as sdk_tool, create_sdk_mcp_server

    belt = tools if tools is not None else registry.all()
    sdk_tools = [
        sdk_tool(t.name, t.description, t.schema)(_make_handler(t))
        for t in belt
    ]
    return create_sdk_mcp_server(name=MCP_SERVER, version="0.1.0", tools=sdk_tools)


def _qualified(name: str) -> str:
    return f"mcp__{MCP_SERVER}__{name}"


# --------------------------------------------------------------------------- #
# Permission gate — mirror Pleiades exec_policy onto the SDK
# --------------------------------------------------------------------------- #
def _make_can_use_tool(cfg: Config, belt: list[Tool],
                       on_event: Callable[[str, Any], None]):
    safe_pleiades = {_qualified(t.name) for t in belt if t.safe}
    policy = cfg.exec_policy

    async def can_use_tool(tool_name: str, tool_input: dict, context: Any):
        from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

        is_safe = tool_name in safe_pleiades or tool_name in _SAFE_BUILTINS
        if is_safe or policy == "allow":
            return PermissionResultAllow()
        if policy == "deny":
            return PermissionResultDeny(message="Denied by Pleiades policy (deny).")
        # policy == "ask": prompt on the console without blocking the loop.
        short = (str(tool_input)[:120]) if tool_input else ""
        try:
            ans = await asyncio.to_thread(
                input, f"  [approve] {tool_name}({short}) [y/N] ")
        except EOFError:
            ans = ""
        if ans.strip().lower() in ("y", "yes"):
            return PermissionResultAllow()
        return PermissionResultDeny(message="Denied by user.")

    return can_use_tool


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def _check_auth(on_event: Callable[[str, Any], None]) -> None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        on_event("warn",
                 "ANTHROPIC_API_KEY is set — Claude Code may bill the API instead "
                 "of your subscription. Unset it to use subscription credit.")
    if not shutil.which("claude"):
        on_event("warn", "`claude` CLI not found on PATH; the Agent SDK needs it.")


async def run_claude_async(
    task: str,
    cfg: Config,
    *,
    tools: list[Tool] | None = None,
    system: str | None = None,
    tier_name: str = "claude",
    add_dirs: list[str] | None = None,
    max_budget_usd: float | None = None,
    on_event: Callable[[str, Any], None] | None = None,
) -> ClaudeResult:
    from claude_agent_sdk import (
        query, ClaudeAgentOptions,
        AssistantMessage, ResultMessage, TextBlock, ThinkingBlock, ToolUseBlock,
    )

    emit = on_event or (lambda *_: None)
    _check_auth(emit)

    belt = tools if tools is not None else registry.all()
    server = build_pleiades_server(belt)
    tier = cfg.tier(tier_name)
    model = tier.model if tier.model and tier.model != "local" else None

    options = ClaudeAgentOptions(
        mcp_servers={MCP_SERVER: server},
        system_prompt=system or _WORK_SYSTEM,
        cwd=cfg.workspace_root,
        permission_mode="default",
        can_use_tool=_make_can_use_tool(cfg, belt, emit),
        max_turns=cfg.max_steps,
        add_dirs=[str(d) for d in (add_dirs or [])],
        model=model,
        effort=getattr(tier, "effort", "high"),
        max_budget_usd=max_budget_usd,
    )

    async def _prompt_stream():
        # can_use_tool requires streaming-input mode (AsyncIterable of messages).
        yield {"type": "user", "message": {"role": "user", "content": task}}

    answer, turns, cost, usage = "", 0, 0.0, {}
    try:
        async for msg in query(prompt=_prompt_stream(), options=options):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        answer = block.text
                        emit("text", block.text)
                    elif isinstance(block, ToolUseBlock):
                        emit("tool_call", block)
                    elif isinstance(block, ThinkingBlock):
                        emit("thinking", getattr(block, "thinking", ""))
            elif isinstance(msg, ResultMessage):
                turns = getattr(msg, "num_turns", 0) or 0
                cost = getattr(msg, "total_cost_usd", 0.0) or 0.0
                usage = getattr(msg, "usage", {}) or {}
                if getattr(msg, "result", None):
                    answer = msg.result
                emit("done", answer)
    except Exception as e:  # noqa: BLE001
        emit("warn", f"claude backend error: {e}")
        return ClaudeResult(answer=answer, turns=turns, cost_usd=cost,
                            usage=usage, error=str(e))

    return ClaudeResult(answer=answer, turns=turns, cost_usd=cost, usage=usage)


def run_claude(task: str, cfg: Config, **kw) -> ClaudeResult:
    """Synchronous wrapper around `run_claude_async`."""
    return asyncio.run(run_claude_async(task, cfg, **kw))


_WORK_SYSTEM = (
    "You are operating INSIDE Pleiades, a workspace on the user's own machine, "
    "running on the user's Claude subscription. You have access to Pleiades' own "
    "tool belt via the MCP server 'pleiades' (tools named mcp__pleiades__*): file "
    "I/O, code navigation, git, shell/process control, web search, a browser, an "
    "encrypted vault, email, subagents, and working memory.\n\n"
    "PREFER the mcp__pleiades__* tools over your built-in equivalents — exercising "
    "them is part of the point. Use your built-in tools only when no Pleiades tool "
    "fits. Work toward the goal directly, verify your work, and finish with a short "
    "plain summary of what you did and what changed."
)


# --------------------------------------------------------------------------- #
# Stress-test / self-audit charter
# --------------------------------------------------------------------------- #
def audit_charter(fix: bool = False, tool_names: list[str] | None = None,
                  report_path: str = "TOOL_AUDIT.md") -> tuple[str, str]:
    """Return (system_prompt, task) for a tool-belt stress test.

    Without `fix` it is read-only triage: exercise tools, log failures, write a
    report. With `fix` it may branch, patch source, and add regression tests.
    """
    belt = registry.select(names=tool_names) if tool_names else registry.all()
    catalog = "\n".join(f"- {t.name}{' (safe)' if t.safe else ''}: {t.description}"
                        for t in belt)

    system = (
        "You are Claude, running on the user's subscription INSIDE the Pleiades "
        "repo, auditing Pleiades' own tool belt from the inside. You reach every "
        "tool through the MCP server 'pleiades' (mcp__pleiades__*). You also have "
        "your built-in tools and full access to the repo source.\n\n"
        "Your job is to be a rigorous, skeptical stress-tester. Local 7B models "
        "use these exact tools and schemas every day; your job is to surface where "
        "they are fragile, ambiguous, under-documented, or simply broken — the "
        "things only become visible when you actually drive them."
    )

    steps = [
        f"You are auditing these tools:\n{catalog}\n",
        "For each tool: (1) read its source in pleiades/harness/builtins/ to "
        "understand intent; (2) actually CALL it via mcp__pleiades__<name> with "
        "realistic and with deliberately adversarial inputs (empty, wrong type, "
        "missing file, huge input, path traversal); (3) note crashes, confusing "
        "errors, silent wrong results, schema/description mismatches, and unsafe "
        "behaviour.",
        "Prefer the mcp__pleiades__* tools — exercising them is the point. Be "
        "economical with turns; batch related checks.",
        f"Write findings to {report_path}: one section per tool with severity "
        "(blocker/bug/smell/nit), what you did, what happened, and a concrete fix.",
    ]
    if fix:
        steps.append(
            "Then FIX what you can: create a git branch, make minimal patches to "
            "the tool source and/or schemas, and ADD regression tests under tests/ "
            "that fail before and pass after. Run the tests. Do not commit; leave "
            "the working tree for the user to review.")
    else:
        steps.append("Do NOT modify source in this run — triage only.")

    task = "\n\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
    return system, task
