"""
agent.py - the main agent loop.

One loop, both backends. The agent receives a task, calls the model with the
available tools, executes any tool calls (through the permission gate), feeds
results back, and repeats until the model stops calling tools or the step cap
is hit.

    agent = Agent(cfg)
    print(agent.run("List the python files here and summarize the largest one"))

Subagents reuse this exact class with a narrower tool set and a cheaper tier.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import Config
from .llm import LLM, ToolCall, Reply
from .tools import registry, Tool
from .anamnesis import Anamnesis


SYSTEM = (
    "You are an autonomous agent running inside Pleiades, a workspace on "
    "the user's own machine. You have real tools to read and write files, "
    "run code, search the web, manage git, dispatch subagents, and more.\n\n"
    "## Operating principles\n"
    "- Work toward the goal directly. Use tools rather than guessing.\n"
    "- Prefer the smallest action that makes progress; verify your work.\n"
    "- Side-effecting actions may require user approval -- if denied, adapt.\n"
    "- When done, give a short plain summary of what you did and what changed.\n\n"
    "## Memory -- use it actively\n"
    "- Use set_section to maintain structured working memory: Goal, Plan, Findings, Open.\n"
    "  Update sections as your understanding evolves, not just at the start.\n"
    "- Use add_finding whenever you discover something relevant mid-task.\n"
    "- Use remember for durable facts that should survive across sessions.\n"
    "- Call read_working_memory at the start of any resumed task.\n\n"
    "## Code navigation -- ordered by efficiency\n"
    "1. get_symbols(file) -- what's in a file without reading it all\n"
    "2. grep_search(pattern, path, glob) -- find where things are used or defined\n"
    "3. read_file(path, start_line, end_line) -- read only the section you need\n"
    "4. preview_edit before edit_file -- always check the diff first\n\n"
    "## After writing code -- close the loop\n"
    "- format_code(path), then lint_code(path), then run_tests(path) -- don't\n"
    "  declare code done until lint and tests confirm it.\n\n"
    "## Documents\n"
    "- read_pdf / read_docx / read_xlsx / read_image for non-text files --\n"
    "  never read_file binary formats.\n\n"
    "## Long-running work\n"
    "- start_process + read_process_output for dev servers, builds, watchers\n"
    "- dispatch_subagents_parallel for research needing multiple sources at once\n"
    "- notify when a long background task finishes so the user knows\n\n"
    "## Context awareness\n"
    "- Call context_status() before heavy tasks or when unsure how full the window is.\n"
    "  When >70% full, summarize findings into remember, clear working memory, continue.\n\n"
    "## Git workflow\n"
    "- git_status / git_diff before touching files -- understand the current state\n"
    "- git_add then git_commit after verified changes (both gated)\n"
)


@dataclass
class AgentResult:
    answer: str
    steps: int
    transcript: list[dict] = field(default_factory=list)


_COMPRESS_NOTICE = (
    "\n\n# Context notice\nYour context window is filling up. Older tool "
    "outputs in this conversation have been compressed. Save anything "
    "important with remember/set_section now; prefer concise tool use."
)
_WARN_NOTICE = (
    "\n\n# Context notice\nYour context window is over 60% full. Summarize "
    "key findings into working memory (set_section) or long-term memory "
    "(remember) soon, and keep tool outputs small (use line ranges/filters)."
)


def squeeze_messages(messages: list[dict], keep_last: int = 4,
                     clip: int = 400) -> int:
    """Truncate bulky content in all but the last `keep_last` messages.

    Clips plain string contents and tool_result blocks (where the bulk lives).
    Backend-native assistant blocks are left untouched. Returns #clipped.
    """
    n = 0
    for m in messages[:max(0, len(messages) - keep_last)]:
        c = m.get("content")
        if isinstance(c, str) and len(c) > clip:
            m["content"] = c[:clip] + f"\n... [compressed, {len(c) - clip} chars hidden]"
            n += 1
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    bc = b.get("content")
                    if isinstance(bc, str) and len(bc) > clip:
                        b["content"] = bc[:clip] + "\n... [compressed]"
                        n += 1
    return n


def execute_tool(tool: Tool, args: dict,
                 approve: Callable[[Tool, dict], bool] | None = None
                 ) -> tuple[str, bool]:
    """Run one tool through the permission gate. Returns (output, is_error).

    Shared by the agent loop and the call_tool dispatcher so every invocation
    passes the same gate and error handling.
    """
    if not tool.safe and approve is not None and not approve(tool, args):
        return ("Denied by user/policy. Do not retry this exact action; "
                "choose a different approach or ask the user."), True
    try:
        result = tool.func(**args)
        return (result if isinstance(result, str) else json.dumps(result)), False
    except Exception as e:
        return f"Error running {tool.name}: {e}", True


class Agent:
    def __init__(self, cfg: Config | None = None, *,
                 tools: list[Tool] | None = None,
                 tier_name: str | None = None,
                 system: str | None = None,
                 depth: int = 0,
                 approve: Callable[[ToolCall, Tool], bool] | None = None) -> None:
        self.cfg = cfg or Config.load()
        self.llm = LLM(self.cfg)
        self.tools = tools if tools is not None else registry.all()
        self.tier_name = tier_name or self.cfg.default_tier
        self.system = system or SYSTEM
        self.depth = depth
        self.approve = approve or self._default_approve
        # Reuse the store bound at startup (shared vector cache); else build one.
        from .builtins.memory import _store          # lazy: avoids import cycle
        self.memory = _store() or Anamnesis.from_config(self.cfg)

    def run(self, task: str, *, on_event: Callable[[str, Any], None] | None = None
            ) -> AgentResult:
        emit = on_event or (lambda *_: None)
        tier = self.cfg.tier(self.tier_name)

        active = self._active_tools()
        tool_index = {t.name: t for t in registry.all()}

        # inject memory up front
        system = self.system
        working = self.memory.working()
        if working:
            system += ("\n\n# Working memory (your live scratchpad -- keep it "
                       "current with set_section / add_finding as you work)\n" + working)
        recalled = self.memory.recall(task)
        if recalled:
            system += "\n\n# Relevant long-term memory\n" + recalled

        messages: list[dict] = [{"role": "user", "content": task}]
        answer = ""
        warned = False

        for step in range(self.cfg.max_steps):
            reply = self.llm.chat(messages, active, tier, system=system)

            # --- context manager: warn at 60%, squeeze old output at 75% ---
            frac = reply.usage.get("input_tokens", 0) / max(1, self.cfg.context_budget)
            if frac >= 0.75:
                self.memory.checkpoint_save("pre-compress")
                clipped = squeeze_messages(messages)
                if clipped:
                    emit("compress", {"fraction": round(frac, 2), "clipped": clipped})
                if _COMPRESS_NOTICE not in system:
                    system += _COMPRESS_NOTICE
            elif frac >= 0.60 and not warned:
                warned = True
                system += _WARN_NOTICE
                emit("context_warning", {"fraction": round(frac, 2)})

            if reply.text:
                emit("text", reply.text)
                answer = reply.text

            if not reply.tool_calls:
                emit("done", answer)
                return AgentResult(answer=answer, steps=step + 1,
                                   transcript=messages)

            messages.append(self._assistant_turn(reply))

            results: list[tuple[ToolCall, str, bool]] = []
            for call in reply.tool_calls:
                emit("tool_call", call)
                out, err = self._dispatch(call, tool_index)
                emit("tool_result", (call, out, err))
                results.append((call, out, err))

            messages.extend(self._result_turns(reply.backend, results))

        emit("done", answer)
        return AgentResult(answer=answer or "(step limit reached)",
                           steps=self.cfg.max_steps, transcript=messages)

    def _active_tools(self) -> list[Tool]:
        """Which tools to expose this turn. In search mode, only discovery tools
        (tag 'search') + always-on meta; the rest are reached via call_tool."""
        non_search = [t for t in self.tools if "search" not in t.tags]
        mode = self.cfg.tool_mode
        big = len(non_search) > self.cfg.tool_search_threshold
        if mode == "search" or (mode == "auto" and big):
            from . import toolsearch
            allowed = {t.name for t in self.tools}
            toolsearch.bind_dispatch(self.approve, allowed)
            # expose discovery/meta tools that are within this agent's set
            sm = registry.select(tags=["search", "meta"])
            return [t for t in sm if t.name in allowed] or sm
        return non_search

    def _dispatch(self, call: ToolCall,
                  index: dict[str, Tool]) -> tuple[str, bool]:
        tool = index.get(call.name)
        if tool is None:
            return f"Error: no such tool '{call.name}'.", True
        return execute_tool(tool, call.args, self.approve)

    def _default_approve(self, tool: Tool, args: dict) -> bool:
        policy = self.cfg.exec_policy
        if policy == "allow":
            return True
        if policy == "deny":
            return False
        try:
            short = json.dumps(args)[:120]
            ans = input(f"  [approve] {tool.name}({short}) [y/N] ").strip().lower()
        except EOFError:
            return False
        return ans in ("y", "yes")

    @staticmethod
    def _assistant_turn(reply: Reply) -> dict:
        if reply.backend == "anthropic":
            return {"role": "assistant", "content": reply.raw_content}
        return {"role": "assistant",
                "content": reply.raw_content.get("content", ""),
                "tool_calls": reply.raw_content.get("tool_calls", [])}

    @staticmethod
    def _result_turns(backend: str,
                      results: list[tuple[ToolCall, str, bool]]) -> list[dict]:
        if backend == "anthropic":
            blocks = [
                {
                    "type": "tool_result",
                    "tool_use_id": c.id,
                    "content": out,
                    **({"is_error": True} if err else {}),
                }
                for c, out, err in results
            ]
            return [{"role": "user", "content": blocks}]
        # ollama / openai: one role:"tool" message per result. The
        # tool_call_id is REQUIRED by the OpenAI spec (and llama.cpp / the
        # Anamnesis proxy) to bind a result to the call that produced it.
        return [{"role": "tool", "tool_call_id": c.id, "name": c.name,
                 "content": out}
                for c, out, _ in results]
