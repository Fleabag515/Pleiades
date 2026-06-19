"""Tool framework: Tool base class, ToolContext, and the ToolBelt registry/dispatcher.

Tools are model-facing capabilities exposed via OpenAI function-calling. Every tool is
bound to a single character through the ToolContext (profile + vault + settings), so a
character's email, vault, and browser are always its own — never shared.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # avoid heavy imports at module load
    from ..config import Settings
    from ..profiles import Profile
    from ..vault import Vault


@dataclass
class ToolContext:
    """Everything a tool needs, scoped to one character. Resources are lazy."""

    profile: "Profile"
    vault: "Vault"
    settings: "Settings"
    _browser: Any = field(default=None, repr=False)
    _http: Any = field(default=None, repr=False)

    @property
    def http(self):
        if self._http is None:
            import httpx

            self._http = httpx.Client(timeout=30.0)
        return self._http

    def browser(self):
        """Lazily create (and reuse) the per-character browser session."""
        if self._browser is None:
            from .browser import BrowserSession

            self._browser = BrowserSession(self.profile)
        return self._browser

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
            self._http = None
        if self._browser is not None:
            try:
                self._browser.close()
            finally:
                self._browser = None


class Tool:
    """Base class for a model-facing tool."""

    name: str = ""
    description: str = ""
    parameters: dict = {}  # JSON schema for the arguments object
    safe: bool = False      # True = read-only, never gated by exec_policy

    def run(self, ctx: ToolContext, **kwargs: Any) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
                or {"type": "object", "properties": {}},
            },
        }


class ToolBelt:
    """Holds tools, exposes their schemas, and dispatches calls."""

    def __init__(self, tools: Optional[list[Tool]] = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.add(t)

    def add(self, tool: Tool) -> None:
        if not tool.name:
            raise ValueError("Tool is missing a name.")
        self._tools[tool.name] = tool

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return list(self._tools)

    def openai_schema(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    def dispatch(self, name: str, args: Any, ctx: ToolContext) -> str:
        """Run a tool; errors come back as TEACHING results, not dead ends.

        A malformed call is a learning opportunity: include the tool's
        parameter schema in the error so the model can correct itself on the
        very next iteration instead of flailing or giving up.
        """
        tool = self._tools.get(name)
        if tool is None:
            close = [n for n in self._tools if name.lower() in n.lower()
                     or n.lower() in name.lower()]
            hint = f" Did you mean: {', '.join(close[:3])}?" if close else ""
            return f"[error] unknown tool: {name}.{hint}"
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                return (f"[error] could not parse arguments for {name} as JSON: "
                        f"{args!r}. Expected parameters: "
                        f"{json.dumps(tool.parameters)}. Retry with valid JSON.")
        if not isinstance(args, dict):
            return (f"[error] arguments for {name} must be a JSON object. "
                    f"Expected parameters: {json.dumps(tool.parameters)}")
        # exec_policy gate for side-effecting tools. No ctx/settings (e.g. a
        # trusted internal caller or a unit test) means no policy to enforce —
        # fall through and run the tool rather than crash.
        if not tool.safe:
            policy = getattr(getattr(ctx, "settings", None), "exec_policy", None)
            if policy == "deny":
                return f"[error] tool '{name}' blocked by exec_policy=deny"
            if policy == "ask":
                short = json.dumps(args)[:120]
                try:
                    ans = input(f"  [approve] {name}({short}) [y/N] ").strip().lower()
                except EOFError:
                    ans = ""
                if ans not in ("y", "yes"):
                    return f"[cancelled] {name}"
        try:
            return tool.run(ctx, **args)
        except TypeError as e:
            return (f"[error] bad arguments for {name}: {e}. "
                    f"Expected parameters: {json.dumps(tool.parameters)}. "
                    f"Retry with corrected arguments.")
        except Exception as e:  # tools must never crash the agent loop
            return f"[error] {name} failed: {e}"


def build_default_belt(ctx: ToolContext) -> ToolBelt:
    """Assemble the standard tools for a character, gated by what's configured."""
    from .search import SearchTool
    from .system_tools import (CharactersTool, HardwareTool, ModelsTool,
                               ProfileTool)
    from .vault_tool import VaultTool

    belt = ToolBelt([SearchTool(), VaultTool(), HardwareTool(), ModelsTool(),
                     ProfileTool(), CharactersTool()])

    if ctx.profile.has_email:
        from .email_box import EmailTool

        belt.add(EmailTool())

    # Browser is optional (requires the [browser] extra + `camoufox fetch`).
    try:
        import camoufox  # noqa: F401

        from .browser import BrowserTool

        belt.add(BrowserTool())
    except ImportError:
        pass

    return belt
