"""
identity.py — run the agent harness *as a character*.

This is the bridge between the two halves of Pleiades:

  • the repo's identity layer — a profile *is* an Anamnesis character, with its own
    encrypted vault, email inbox, browser profile, and Discord token; and
  • the harness — the Claude-Code-style agent loop with its 60+ tool belt.

`bind_character(name, cfg)` makes a character the active identity for a harness run:

  1. loads the Profile + opens its encrypted Vault,
  2. scopes the harness workspace to the character's own workspace dir
     (so file/shell/git tools operate in the character's space, not the cwd),
  3. routes inference through the character's Anamnesis proxy when available
     (memory injection + retrieval, upstreaming to our local llama.cpp engine),
     so "local inference" for a character always passes through its memory.

It also registers two identity-scoped tools onto the harness registry — `vault`
and `email` — that delegate to the repo's curated implementations, bound to
whichever character is active. They no-op gracefully when no character is bound
(e.g. a plain workspace run), so importing this module is always safe.
"""

from __future__ import annotations

from typing import Optional

from .tools import tool
from .. import config as repo_config
from ..profiles import Profile, ProfileManager
from ..tools import ToolContext
from ..tools.vault_tool import VaultTool
from ..tools.email_box import EmailTool


# Active identity for the current harness run (set by bind_character).
_ACTIVE: dict[str, object] = {"profile": None, "vault": None, "settings": None}

_vault_impl = VaultTool()
_email_impl = EmailTool()


def bind_character(
    name: str,
    cfg=None,
    *,
    settings: Optional["repo_config.Settings"] = None,
    route_inference: bool = True,
) -> Profile:
    """Make character `name` the active identity. Returns its Profile.

    cfg: the harness Config for this run. If given, its workspace_root is pointed
         at the character's workspace and (when route_inference) openai_host is
         pointed at the character's Anamnesis proxy.
    """
    settings = settings or repo_config.Settings.load()
    pm = ProfileManager(settings)
    profile = pm.get(name)                 # raises FileNotFoundError if missing
    vault = pm.open_vault(name)
    _ACTIVE.update(profile=profile, vault=vault, settings=settings)

    # Share ONE persistent browser profile between the work-path (harness) browser
    # and the chat-path (engine) browser — both use the character's browser_dir.
    try:
        from .builtins.browser import bind_browser_profile
        bind_browser_profile(profile.browser_dir)
    except Exception:
        pass

    # Scope the workspace to the character's own dir.
    workspace = repo_config.profile_dir(name) / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    if cfg is not None:
        cfg.workspace_root = str(workspace)
        # Scope the in-process working/long-term memory tier to this character, so a
        # profile's memory IS the character's memory (the canonical cross-session
        # context manager remains the Anamnesis daemon — see pleiades.anamnesis).
        mem = repo_config.profile_dir(name) / "memory"
        mem.mkdir(parents=True, exist_ok=True)
        cfg.memory_dir = str(mem)

    # Route inference through the character's Anamnesis proxy (memory + llama.cpp).
    if cfg is not None and route_inference:
        try:
            proxy_url = pm.anamnesis.ensure_running(name)  # http://127.0.0.1:<port>/v1
            if proxy_url:
                cfg.openai_host = proxy_url
        except Exception:
            pass  # Anamnesis daemon not up — fall back to direct inference server.
    return profile


def active_profile() -> Optional[Profile]:
    return _ACTIVE.get("profile")  # type: ignore[return-value]


def _ctx() -> ToolContext:
    profile = _ACTIVE.get("profile")
    vault = _ACTIVE.get("vault")
    settings = _ACTIVE.get("settings")
    if profile is None or vault is None:
        raise RuntimeError(
            "No active character. Run the harness with a bound character "
            "(bind_character(name, cfg)) to use identity tools."
        )
    return ToolContext(profile=profile, vault=vault, settings=settings)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Identity-scoped tools (delegate to the repo's curated implementations).
# --------------------------------------------------------------------------- #
@tool(name="vault", tags=("identity",))
def vault(action: str, name: str = "", secret: str = "", note: str = "") -> str:
    """Your private password manager — store/get/list your own credentials, encrypted at rest.

    action: one of store | get | list.
    name: identifier for the secret (e.g. a domain like 'github.com'). Required for store/get.
    secret: the secret value to store. Required for action 'store'.
    note: optional non-secret note (e.g. the username/email used).
    """
    return _vault_impl.run(_ctx(), action=action, name=name, secret=secret, note=note)


@tool(name="email", tags=("identity",))
def email(action: str, to: str = "", subject: str = "", body: str = "",
          query: str = "", id: str = "", limit: int = 10) -> str:
    """Your own email inbox — read, search, and send mail from the character's account.

    action: one of list_unread | search | read | send.
    to/subject/body: recipient and content for 'send'.
    query: search text for 'search'. id: message id for 'read'. limit: max messages to list.
    """
    ctx = _ctx()
    if not ctx.profile.has_email:
        return ("[email] This character has no email configured. Add IMAP/SMTP "
                "details and an app password to its profile first.")
    return _email_impl.run(ctx, action=action, to=to, subject=subject,
                           body=body, query=query, id=id, limit=limit)
