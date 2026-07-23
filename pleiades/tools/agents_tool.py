"""Chat-belt tool: spawn ephemeral background sub-agents.

Delegates one or more focused tasks to background agents that run
concurrently, on the SAME model/llama-server already serving this chat (zero
extra VRAM), and returns immediately with their ids -- results/progress show
up in the desktop app's right panel "Agents" section, not inline in this
turn. See pleiades/agents.py for the manager and the safety properties
(depth-capped at 1, ephemeral memory, safe-tools-only, wall-clock + step
ceilings, concurrency cap).

Gated behind Profile.agents_enabled (see profiles.py) -- build_default_belt
only adds this tool for characters that have the feature turned on, mirroring
how EmailTool is only added when ctx.profile.has_email.
"""

from __future__ import annotations

from . import Tool, ToolContext


class AgentsTool(Tool):
    name = "spawn_agents"
    description = (
        "Delegate one or more self-contained tasks to background sub-agents that run "
        "concurrently while this conversation continues. Each agent is ephemeral -- it "
        "has no memory of this conversation and no persistent identity of its own -- "
        "and only has read-only tools (files, git, search, docs), not the ability to "
        "write files, run shell commands, or spawn further agents. Use this for "
        "independent research/investigation sub-tasks, not anything that needs to "
        "modify the user's system or that depends on this conversation's context. "
        "Returns immediately with agent ids; you do not see their results in this turn "
        "-- the user watches their progress in the Agents panel and can tell you what "
        "they found in a later message, or you can check back with the same tool later "
        "if a mechanism for that is available."
    )
    parameters = {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "One or more complete, standalone task descriptions. Each runs on "
                    "its own agent with none of this conversation's context, so include "
                    "everything it needs to know."
                ),
            },
        },
        "required": ["tasks"],
    }

    def run(self, ctx: ToolContext, tasks: list) -> str:
        if not getattr(ctx.profile, "agents_enabled", False):
            return ("[spawn_agents error] Agents are disabled for this character. "
                    "Enable them in the desktop app's Agents panel first.")
        if isinstance(tasks, str):
            tasks = [tasks]
        if not isinstance(tasks, list) or not tasks:
            return "[spawn_agents error] tasks must be a non-empty list of strings."

        from ..agents import get_manager

        model = getattr(ctx.profile, "agents_model", "") or ""
        try:
            ids = get_manager().spawn_agents(
                tasks, character=ctx.profile.name, model=model
            )
        except ValueError as e:
            return f"[spawn_agents error] {e}"
        return (f"Spawned {len(ids)} background agent(s): {', '.join(ids)}. "
                "They run concurrently; watch the Agents panel for progress.")
