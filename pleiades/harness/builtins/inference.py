"""
inference.py — the agent manages its own brain.

Harness-side registration of the shared system operations (pleiades.system_ops):
hardware report, model search/fetch/start/stop/assign, profile self-config, and
character management. Side-effecting actions are NOT marked safe, so they pass
through the exec policy ("ask" by default) like shell or file writes.
"""

from __future__ import annotations

from ...system_ops import (characters_op, hardware_report, models_op,
                           profile_op)
from .. import identity
from ..tools import tool


@tool(safe=True, tags=("system", "models"))
def hardware(action: str = "report") -> str:
    """Inspect this machine: GPU/VRAM, RAM, CPU, and how each model maps onto it.

    action: only 'report' for now.
    """
    return hardware_report()


@tool(tags=("models",))
def models(action: str, name: str = "", repo: str = "", query: str = "",
           quant: str = "", character: str = "") -> str:
    """Manage local models: list, search Hugging Face, fetch the best quant for this machine, start/stop servers, assign to characters.

    action: list | search | fetch | start | stop | assign.
    name: model name (start/stop/assign; optional custom name for fetch).
    repo: Hugging Face repo id (for fetch).
    query: search text (for search).
    quant: force a quant for fetch, e.g. Q4_K_M (default: auto for hardware).
    character: character name (for assign; defaults to the bound character).
    """
    p = identity.active_profile()
    if action == "assign" and not character and p is not None:
        character = p.name
    return models_op(action, name=name, repo=repo, query=query, quant=quant,
                     character=character)


@tool(tags=("system",))
def profile(action: str, field: str = "", value: str = "") -> str:
    """Read or update your own character profile (email settings, persona_source, assigned model).

    action: get | set.
    field: field to set (email_address, imap_host, imap_port, smtp_host, smtp_port, persona_source, model).
    value: new value (for set).
    """
    return profile_op(identity.active_profile(), action, field=field, value=value)


@tool(tags=("system",))
def characters(action: str, name: str = "", confirm: str = "") -> str:
    """Manage characters on this workstation: list, create, adopt, or delete (delete needs confirm='delete <name>').

    action: list | create | adopt | delete.
    name: character name.
    confirm: for delete, exactly 'delete <name>'.
    """
    p = identity.active_profile()
    if action == "delete" and p is not None and name == p.name:
        return "[characters error] you cannot delete yourself while running."
    return characters_op(action, name=name, confirm=confirm)
