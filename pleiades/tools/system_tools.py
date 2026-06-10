"""Chat-path tools for hardware, models, profile, and characters.

These make Pleiades self-building from the model's side of the table: a
character can inspect the machine it lives on, search/download/start models,
reconfigure its own profile, and manage characters — the same operations the
CLI and web UI expose, via pleiades.system_ops.
"""

from __future__ import annotations

from .. import system_ops
from . import Tool, ToolContext


class HardwareTool(Tool):
    name = "hardware"
    description = (
        "Inspect the machine you run on: GPU/VRAM, RAM, CPU threads, and how "
        "each registered model will be split between GPU and CPU. Use before "
        "choosing or downloading a model."
    )
    parameters = {"type": "object", "properties": {}}

    def run(self, ctx: ToolContext) -> str:
        return system_ops.hardware_report()


class ModelsTool(Tool):
    name = "models"
    description = (
        "Manage the local models you can run on. Actions: 'list' registered models, "
        "'search' Hugging Face for GGUF repos, 'fetch' (download the best quant for "
        "this machine from a repo and register it), 'start'/'stop' a model server, "
        "'assign' a model to a character."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string",
                       "enum": ["list", "search", "fetch", "start", "stop", "assign"]},
            "name": {"type": "string",
                     "description": "Model name (start/stop/assign; optional for fetch)."},
            "repo": {"type": "string",
                     "description": "Hugging Face repo id (for 'fetch')."},
            "query": {"type": "string", "description": "Search text (for 'search')."},
            "quant": {"type": "string",
                      "description": "Force a quant for 'fetch' (e.g. Q4_K_M); "
                                     "default auto-picks for the hardware."},
            "character": {"type": "string", "description": "Character (for 'assign')."},
        },
        "required": ["action"],
    }

    def run(self, ctx: ToolContext, action: str, name: str = "", repo: str = "",
            query: str = "", quant: str = "", character: str = "") -> str:
        if action == "assign" and not character:
            character = ctx.profile.name  # default: assign to yourself
        return system_ops.models_op(action, name=name, repo=repo, query=query,
                                    quant=quant, character=character)


class ProfileTool(Tool):
    name = "profile"
    description = (
        "Read or update your own configuration: email settings (address, "
        "imap_host, imap_port, smtp_host, smtp_port), persona_source, and "
        "your assigned model. Actions: 'get', 'set'."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["get", "set"]},
            "field": {"type": "string", "description": "Field to set."},
            "value": {"type": "string", "description": "New value."},
        },
        "required": ["action"],
    }

    def run(self, ctx: ToolContext, action: str, field: str = "", value: str = "") -> str:
        result = system_ops.profile_op(ctx.profile, action, field=field, value=value)
        if action == "set" and not result.startswith("["):
            # keep the in-memory profile in sync for later tools in this run
            ctx.profile = system_ops.ProfileManager().get(ctx.profile.name)
        return result


class CharactersTool(Tool):
    name = "characters"
    description = (
        "Manage characters on this workstation. Actions: 'list', 'create' a new "
        "character (own memory, vault, browser), 'adopt' an existing Anamnesis "
        "character, 'delete' (requires confirm='delete <name>'; permanent)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string",
                       "enum": ["list", "create", "adopt", "delete"]},
            "name": {"type": "string", "description": "Character name."},
            "confirm": {"type": "string",
                        "description": "For 'delete': must be exactly 'delete <name>'."},
        },
        "required": ["action"],
    }

    def run(self, ctx: ToolContext, action: str, name: str = "", confirm: str = "") -> str:
        if action == "delete" and name == ctx.profile.name:
            return "[characters error] you cannot delete yourself while running."
        return system_ops.characters_op(action, name=name, confirm=confirm)
