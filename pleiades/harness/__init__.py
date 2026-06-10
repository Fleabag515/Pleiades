"""
Pleiades — an agent-first workspace.

The PC is the agent's home. A main model receives a task, thinks, and acts
through a registry of tools; it can dispatch tiered subagents for sub-tasks.
The brain is hybrid: local Ollama for cheap/private work, the Anthropic API
(Claude) for hard reasoning — chosen per task.

Everything the agent can do is a `@tool`. New capabilities (SearXNG deep
research, Discord control, browser-use, ...) are added by dropping a new
`@tool` into `pleiades/builtins/` — the core never changes.

Public surface:
    from pleiades import Agent, Config, tool, registry
"""

from .config import Config
from .tools import tool, registry, Tool
from .agent import Agent

__all__ = ["Agent", "Config", "tool", "registry", "Tool"]
__version__ = "0.1.0"
