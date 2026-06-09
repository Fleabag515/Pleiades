"""Model-facing wrapper over the per-character Vault.

This is the ONLY path by which a stored secret can reach the model: it must explicitly
call `vault` with action `get`. Storing and listing never reveal other secrets' values.
Use this to save credentials for accounts the character creates and retrieve them later.
"""

from __future__ import annotations

from . import Tool, ToolContext
from ..vault import RESERVED_KEYS, Vault


class VaultTool(Tool):
    name = "vault"
    description = (
        "Your private password manager. Save credentials for accounts you create, "
        "retrieve them later, or list what you've stored. Secrets are encrypted and "
        "are only ever shown to you when you explicitly 'get' them."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["store", "get", "list"],
                "description": "store a secret, get one back, or list saved names.",
            },
            "name": {
                "type": "string",
                "description": (
                    "Identifier for the secret, e.g. a domain like 'github.com' or a "
                    "label like 'site:example.com'. Required for store/get."
                ),
            },
            "secret": {
                "type": "string",
                "description": "The secret value to store. Required for action 'store'.",
            },
            "note": {
                "type": "string",
                "description": "Optional non-secret note (e.g. the username/email used).",
            },
        },
        "required": ["action"],
    }

    @staticmethod
    def _normalize(name: str) -> str:
        name = name.strip()
        # Keep reserved keys as-is; namespace bare domains under site:.
        if name in RESERVED_KEYS or name.startswith("site:"):
            return name
        if "." in name and " " not in name:
            return Vault.site_key(name)
        return name

    def run(self, ctx: ToolContext, action: str, name: str = "", secret: str = "", note: str = "") -> str:
        vault = ctx.vault
        action = action.lower().strip()

        if action == "list":
            entries = vault.list()
            if not entries:
                return "No secrets stored yet."
            lines = []
            for e in entries:
                label = e["key"]
                meta_note = (e.get("meta") or {}).get("note") if e.get("meta") else None
                lines.append(f"- {label}" + (f"  ({meta_note})" if meta_note else ""))
            return "Stored secrets:\n" + "\n".join(lines)

        if action == "store":
            if not name or not secret:
                return "[vault error] 'store' needs both 'name' and 'secret'."
            key = self._normalize(name)
            vault.set(key, secret, meta={"note": note} if note else None)
            return f"Stored secret under '{key}'."

        if action == "get":
            if not name:
                return "[vault error] 'get' needs 'name'."
            key = self._normalize(name)
            value = vault.get(key)
            if value is None:
                return f"No secret stored under '{key}'."
            return f"{key} = {value}"

        return f"[vault error] unknown action '{action}'. Use store, get, or list."
