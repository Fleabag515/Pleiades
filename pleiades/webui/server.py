"""FastAPI backend for the Pleiades control panel.

This module is a thin, careful wrapper around the managers Pleiades already ships:

    * config.Settings / EMAIL_PRESETS   – global engine + harness settings
    * profiles.ProfileManager           – the unifying per-character layer
    * models.ModelManager               – local GGUF registry + lifecycle
    * vault.Vault                        – encrypted per-profile secret store
    * anamnesis.Anamnesis               – the memory-proxy control API

Design rules honoured here (see CLAUDE.md §8):

    * Secrets are never logged. Vault values are only returned by the explicit
      ``GET /api/profiles/{name}/vault/{key}`` reveal endpoint and never appear
      in any list response.
    * Every path/vault/inbox/token is namespaced by character.
    * The UI never re-implements memory or inference; it only configures them.

The whole thing binds to 127.0.0.1 by default — it is a local admin tool.
"""

from __future__ import annotations

import mimetypes
import os
import re
import socket
import asyncio
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Optional

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None  # type: ignore

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import config
from .. import scheduler
from ..profiles import ProfileManager, Profile
from ..models import ModelManager, ModelError
from ..scheduler import ScheduledTaskManager, ScheduleError
from ..vault import RESERVED_KEYS
from ..anamnesis import Anamnesis

STATIC_DIR = Path(__file__).parent / "static"


# --------------------------------------------------------------------------- #
# Request bodies
# --------------------------------------------------------------------------- #
class ProfileCreate(BaseModel):
    name: str
    email_address: str = ""
    imap_host: str = ""
    imap_port: int = 993
    smtp_host: str = ""
    smtp_port: int = 587
    email_password: Optional[str] = None
    discord_token: Optional[str] = None
    persona_source: str = "auto"


class ProfileUpdate(BaseModel):
    email_address: Optional[str] = None
    imap_host: Optional[str] = None
    imap_port: Optional[int] = None
    smtp_host: Optional[str] = None
    smtp_port: Optional[int] = None
    persona_source: Optional[str] = None
    discord_enabled: Optional[bool] = None
    discord_require_mention: Optional[bool] = None
    discord_respond_to_bots: Optional[bool] = None
    discord_allowed_channels: Optional[str] = None
    # Per-character tool-call approval override (composer's Always Ask/
    # Preset/Skip Approvals dropdown). None/omitted leaves it untouched;
    # "allow"/"ask"/"deny"/"preset" sets it. See profiles.Profile.exec_policy
    # and ToolBelt.dispatch's gate. "preset" defers each tool call to
    # profiles.Profile.tool_policies (see ToolPolicyUpdate/set_tool_policy
    # below for how those per-tool entries get set).
    exec_policy: Optional[str] = None
    # Agents panel toggle + model pick (see profiles.Profile.agents_enabled/
    # agents_model, pleiades/agents.py). None/omitted leaves untouched.
    agents_enabled: Optional[bool] = None
    agents_model: Optional[str] = None


class ToolPolicyUpdate(BaseModel):
    # "ask"/"allow" sets this tool's override for THIS character, consulted
    # only while exec_policy=="preset" (see profiles.Profile.tool_policies
    # and ToolBelt.dispatch's gate in pleiades/tools/__init__.py). None/null
    # clears the override, reverting the tool to the default "allow".
    policy: Optional[str] = None


class EmailConfig(BaseModel):
    email_address: str = ""
    imap_host: str = ""
    imap_port: int = 993
    smtp_host: str = ""
    smtp_port: int = 587
    password: Optional[str] = None  # only written if provided


class DiscordConfig(BaseModel):
    token: Optional[str] = None  # only written if provided
    enabled: bool = False
    require_mention: Optional[bool] = None
    respond_to_bots: Optional[bool] = None
    allowed_channels: Optional[str] = None


class EmailSend(BaseModel):
    to: str
    subject: str = ""
    body: str = ""


class VaultEntry(BaseModel):
    key: str
    value: str
    note: str = ""


class ModelCreate(BaseModel):
    name: str
    path: str
    n_ctx: "int | str" = "auto"          # "auto" = planned from hardware at launch
    n_gpu_layers: "int | str" = "auto"   # "auto" = planned from hardware at launch
    chat_format: str = ""
    # Vision: path to a sibling mmproj GGUF. Blank = auto-detect a
    # mmproj-*.gguf next to `path` (see hardware.find_mmproj_sibling(),
    # models.ModelManager.add()); set explicitly only when the sidecar lives
    # somewhere auto-detection won't find it.
    mmproj: str = ""
    capabilities: str = ""


class ModelUpdate(BaseModel):
    path: Optional[str] = None
    n_ctx: "Optional[int | str]" = None
    n_gpu_layers: "Optional[int | str]" = None
    chat_format: Optional[str] = None
    # UI-only display name override; "" clears it back to showing `name`.
    display_name: Optional[str] = None
    mmproj: Optional[str] = None
    capabilities: Optional[str] = None


class AvatarBody(BaseModel):
    data: str   # data URL (image/png or image/jpeg), max ~2 MB decoded


class ChatCreate(BaseModel):
    character: str


class ChatMessage(BaseModel):
    message: str
    system: Optional[str] = None
    # Ids returned by POST /api/chats/{chat_id}/attachments (see
    # pleiades/attachments.py) -- files the user attached to THIS message.
    # Resolved against THIS chat's own cache dir in chats_message(), never
    # trusted as a raw filesystem path.
    attachments: list[str] = []


class ChatInterject(BaseModel):
    """A message woven into an ALREADY-IN-FLIGHT turn (see chats_interject) —
    distinct from ChatMessage, which always starts a brand-new turn."""
    message: str


class ModelFetch(BaseModel):
    repo: str
    name: str = ""
    quant: str = ""   # blank = auto-pick for this machine


class AssignModel(BaseModel):
    model: str  # "" clears the assignment (back to default engine)


class AnamnesisUpstreamUpdate(BaseModel):
    base_url: str
    api_key: Optional[str] = None
    model_name: Optional[str] = None


class ChatBody(BaseModel):
    message: str
    system: Optional[str] = None


class WorkStart(BaseModel):
    task: str
    character: str = ""
    tier: str = ""
    policy: str = ""


class WorkApprove(BaseModel):
    approve: bool = False


class AgentsSettings(BaseModel):
    """POST /api/agents/settings body -- the Agents panel's toggle + model
    picker, both per-character (see profiles.Profile.agents_enabled/
    agents_model). All fields optional; omitted ones are left untouched."""
    character: str
    enabled: Optional[bool] = None
    model: Optional[str] = None


class ScheduledTaskCreate(BaseModel):
    task: str
    character: str = ""
    cron: str = ""
    fire_at: float = 0.0     # epoch seconds; 0 means "not one-time" (use cron)
    tier: str = ""
    policy: str = ""


class ScheduledTaskUpdate(BaseModel):
    task: Optional[str] = None
    character: Optional[str] = None
    cron: Optional[str] = None
    fire_at: Optional[float] = None
    tier: Optional[str] = None
    policy: Optional[str] = None
    enabled: Optional[bool] = None


class BrowserViewStart(BaseModel):
    url: str = ""


class BrowserViewNavigate(BaseModel):
    url: str


class BrowserViewResize(BaseModel):
    width: int
    height: int


class CloudModelBody(BaseModel):
    source: str
    model: str
    # Manual vision/capability override, comma-separated (e.g. "vision").
    # Blank = best-effort auto-populate from OpenRouter's own model listing
    # (architecture.modality) when source == "openrouter"; ollama-cloud has
    # no equivalent public listing, so it stays blank there unless the
    # caller sets it.
    capabilities: str = ""


class ApiKeyBody(BaseModel):
    provider: str   # "openrouter" | "ollama-cloud"
    key: str


class SettingsUpdate(BaseModel):
    # engine
    model_path: Optional[str] = None
    inference_host: Optional[str] = None
    inference_port: Optional[int] = None
    n_ctx: Optional[int] = None
    n_gpu_layers: "Optional[int | str]" = None
    chat_format: Optional[str] = None
    # services
    anamnesis_control_url: Optional[str] = None
    searxng_url: Optional[str] = None
    # harness
    default_tier: Optional[str] = None
    exec_policy: Optional[str] = None
    max_steps: Optional[int] = None
    eval_interval: Optional[int] = None
    flash_attn: Optional[str] = None
    kv_cache_type: Optional[str] = None
    n_batch: Optional[int] = None
    n_ubatch: Optional[int] = None
    mlock: Optional[bool] = None
    draft_model_path: Optional[str] = None
    ollama_cloud_url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _reachable(url: str, path: str = "", timeout: float = 1.2) -> bool:
    if not httpx or not url:
        return False
    try:
        r = httpx.get(url.rstrip("/") + path, timeout=timeout)
        return r.status_code < 500
    except Exception:
        return False


# Multi-key cloud providers: env var name + Settings field per provider id,
# shared by the add/remove key endpoints below.
_KEY_PROVIDERS: dict[str, dict[str, str]] = {
    "openrouter": {"env": "OPENROUTER_API_KEYS", "field": "openrouter_api_keys"},
    "ollama-cloud": {"env": "OLLAMA_CLOUD_API_KEYS", "field": "ollama_cloud_api_keys"},
}


def _env_path() -> Path:
    return Path(os.environ.get("PLEIADES_ROOT") or Path.cwd()) / ".env"


def _write_env(updates: dict[str, str]) -> None:
    """Merge updates into the project .env file, preserving comments/order."""
    path = _env_path()
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    seen: set[str] = set()
    out: list[str] = []
    for raw in lines:
        m = re.match(r"\s*([A-Z0-9_]+)\s*=", raw)
        if m and m.group(1) in updates:
            key = m.group(1)
            out.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            out.append(raw)
    for key, val in updates.items():
        if key not in seen:
            out.append(f"{key}={val}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    # Reflect immediately into the running process so reads are consistent.
    for key, val in updates.items():
        os.environ[key] = val


def _model_mgr_update(mm: ModelManager, name: str, body: ModelUpdate) -> dict:
    """ModelManager has no public update; mutate models.json in place safely."""
    reg = mm._load()  # noqa: SLF001 — same package, documented internal state
    if name not in reg:
        raise ModelError(f"unknown model '{name}'")
    entry = reg[name]
    if body.path is not None:
        p = Path(body.path).expanduser()
        if not p.is_file():
            raise ModelError(f"No such model file: {p}")
        entry["path"] = str(p)
    if body.n_ctx is not None:
        entry["n_ctx"] = body.n_ctx
    if body.n_gpu_layers is not None:
        entry["n_gpu_layers"] = body.n_gpu_layers
    if body.chat_format is not None:
        entry["chat_format"] = body.chat_format
    if body.display_name is not None:
        entry["display_name"] = body.display_name
    if body.mmproj is not None:
        entry["mmproj"] = body.mmproj
    if body.capabilities is not None:
        entry["capabilities"] = body.capabilities
    mm._save(reg)  # noqa: SLF001
    entry = dict(entry)
    entry["running"] = mm.is_running(name)
    return entry


def _profile_view(p: Profile) -> dict:
    d = p.to_json()
    d["has_email"] = p.has_email
    return d


# --------------------------------------------------------------------------- #
# Anamnesis helpers (config lives in ~/.anamnesis/characters/{name}/)
# --------------------------------------------------------------------------- #

def _anamnesis_config_path(name: str) -> Path:
    if not re.match(r"^[A-Za-z0-9_\-]+$", name):
        raise HTTPException(400, "Invalid character name")
    return Path.home() / ".anamnesis" / "characters" / name / "config.json"


def _read_anamnesis_cfg(name: str) -> dict:
    import json as _json
    p = _anamnesis_config_path(name)
    if not p.is_file():
        raise HTTPException(404, f"No Anamnesis config found for '{name}'")
    try:
        return _json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(500, f"Could not read Anamnesis config: {exc}") from exc


def _anamnesis_db_path(name: str) -> Optional[Path]:
    p = Path.home() / ".anamnesis" / "characters" / name / "history.db"
    return p if p.is_file() else None


def _memory_stats(name: str) -> dict:
    db = _anamnesis_db_path(name)
    if not db:
        return {"turns": 0, "scenes": 0, "engrams": 0, "observations": 0}
    try:
        import sqlite3
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
        cur = con.cursor()

        def _count(table: str) -> int:
            try:
                return cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except Exception:
                return 0

        stats = {
            "turns": _count("turns"),
            "scenes": _count("memscenes"),
            "engrams": _count("engrams"),
            "observations": _count("character_observations"),
        }
        con.close()
        return stats
    except Exception:
        return {"turns": 0, "scenes": 0, "engrams": 0, "observations": 0}


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #
def create_app() -> FastAPI:
    app = FastAPI(title="Pleiades Control Panel", version="2.0.0")
    # Desktop app (Electron renderer, served from a Vite dev-server origin or
    # file://, which browsers report as the opaque Origin "null") talks to
    # this loopback-only server cross-origin. Scope CORS to localhost/
    # 127.0.0.1/null/file:// only — this server never leaves the machine
    # anyway, and webui/__init__.py's loopback guard (installed as an outer
    # middleware around this app, so it runs first) is the actual
    # authorization layer; it rejects "null"/foreign Origins itself
    # (hostname("null") isn't in its loopback allowlist). allow_credentials
    # is deliberately False, not True: nothing in this app uses cookies, so
    # there's nothing for a credentialed cross-origin request to carry — no
    # reason to combine "any null-origin page" with "send credentials" even
    # as defense-in-depth, in case the guard is ever disabled (an explicit
    # wildcard-bind opt-out) or removed later.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^(http://(localhost|127\.0\.0\.1)(:\d+)?|file://|null)$",
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    settings = config.Settings.load()
    pm = ProfileManager(settings)
    mm = ModelManager()
    an = Anamnesis(settings.anamnesis_control_url)

    # Mount any configured MCP tool sources (e.g. a claude-mcp sidecar) into the
    # global tool registry once at process startup -- every Agent(...) built later
    # (see _launch_work_job below) snapshots that same registry at construction.
    from ..harness.mcp_client import mount_configured_servers
    mount_configured_servers(settings)

    # ----------------------------- dashboard ------------------------------- #
    @app.get("/api/status")
    def status() -> dict:
        s = config.Settings.load()
        anamnesis_up = _reachable(s.anamnesis_control_url, "/status")
        try:
            chars = an.list_characters()
        except Exception:
            chars = []
        inference_up = _reachable(s.inference_base_url, "/models")
        from pathlib import Path as _P
        has_default = bool(s.model_path) and _P(s.model_path).expanduser().is_file()
        has_registry = bool(mm.list())
        if inference_up:
            inference_state = "running"
        elif has_default or has_registry:
            inference_state = "on_demand"   # starts automatically on first chat
        else:
            inference_state = "no_model"
        searxng_up = _reachable(s.searxng_url, "/healthz") or _reachable(s.searxng_url)
        models = mm.list()
        running = [m for m in models if m.get("running")]
        return {
            "services": {
                "anamnesis": {"up": anamnesis_up, "url": s.anamnesis_control_url,
                              "characters": len(chars)},
                "inference": {"up": inference_up, "url": s.inference_base_url,
                              "model_path": s.model_path, "state": inference_state},
                "searxng": {"up": searxng_up, "url": s.searxng_url},
            },
            "counts": {
                "profiles": len(pm.list()),
                "models": len(models),
                "models_running": len(running),
                "orphans": len(pm.orphan_characters()),
            },
            "running_models": [
                {"name": m["name"], "port": m["port"], "n_gpu_layers": m["n_gpu_layers"]}
                for m in running
            ],
        }

    # ----------------------------- settings -------------------------------- #
    @app.get("/api/settings")
    def get_settings() -> dict:
        s = config.Settings.load()
        d = s.to_dict()
        # Master-key status without ever exposing the key itself.
        env_key = bool(os.environ.get("PLEIADES_MASTER_KEY", "").strip())
        key_file = config.MASTER_KEY_PATH.is_file()
        return {
            "settings": d,
            "email_presets": config.EMAIL_PRESETS,
            "env_file": str(_env_path()),
            "env_file_exists": _env_path().is_file(),
            "master_key": {
                "from_env": env_key,
                "keyfile_present": key_file,
                "keyfile_path": str(config.MASTER_KEY_PATH),
                "configured": env_key or key_file,
            },
            "pleiades_home": str(config.PLEIADES_HOME),
        }

    @app.put("/api/settings")
    def put_settings(body: SettingsUpdate) -> dict:
        env_map = {
            "model_path": "PLEIADES_MODEL_PATH",
            "inference_host": "PLEIADES_INFERENCE_HOST",
            "inference_port": "PLEIADES_INFERENCE_PORT",
            "n_ctx": "PLEIADES_N_CTX",
            "n_gpu_layers": "PLEIADES_N_GPU_LAYERS",
            "chat_format": "PLEIADES_CHAT_FORMAT",
            "anamnesis_control_url": "PLEIADES_ANAMNESIS_CONTROL_URL",
            "searxng_url": "PLEIADES_SEARXNG_URL",
            "exec_policy": "PLEIADES_EXEC_POLICY",
            "eval_interval": "PLEIADES_EVAL_INTERVAL",
            "flash_attn": "PLEIADES_FLASH_ATTN",
            "kv_cache_type": "PLEIADES_KV_CACHE_TYPE",
            "n_batch": "PLEIADES_N_BATCH",
            "n_ubatch": "PLEIADES_N_UBATCH",
            "draft_model_path": "PLEIADES_DRAFT_MODEL",
            "ollama_cloud_url": "OLLAMA_CLOUD_URL",
        }
        updates: dict[str, str] = {}
        for field_name, env_name in env_map.items():
            val = getattr(body, field_name)
            if val is not None:
                updates[env_name] = str(val)
        if body.mlock is not None:
            updates["PLEIADES_MLOCK"] = "true" if body.mlock else "false"
        if updates:
            _write_env(updates)
        return get_settings()

    @app.post("/api/settings/keys")
    def add_settings_key(body: ApiKeyBody) -> dict:
        info = _KEY_PROVIDERS.get(body.provider)
        if not info:
            raise HTTPException(400, f"Unknown provider '{body.provider}'")
        key = (body.key or "").strip()
        if not key:
            raise HTTPException(400, "Key is empty")
        s = config.Settings.load()
        keys = list(getattr(s, info["field"]))
        if key in keys:
            raise HTTPException(409, "That key is already stored for this provider")
        keys.append(key)
        _write_env({info["env"]: ",".join(keys)})
        return get_settings()

    @app.delete("/api/settings/keys")
    def remove_settings_key(provider: str, index: int) -> dict:
        info = _KEY_PROVIDERS.get(provider)
        if not info:
            raise HTTPException(400, f"Unknown provider '{provider}'")
        s = config.Settings.load()
        keys = list(getattr(s, info["field"]))
        if not (0 <= index < len(keys)):
            raise HTTPException(404, "No key at that index")
        keys.pop(index)
        _write_env({info["env"]: ",".join(keys)})
        return get_settings()

    # ----------------------------- profiles -------------------------------- #
    @app.get("/api/profiles")
    def list_profiles() -> dict:
        return {
            "profiles": [_profile_view(p) for p in pm.list()],
            "orphans": pm.orphan_characters(),
        }

    @app.post("/api/profiles")
    def create_profile(body: ProfileCreate) -> dict:
        try:
            p = pm.create(
                body.name,
                email_address=body.email_address,
                imap_host=body.imap_host,
                imap_port=body.imap_port,
                smtp_host=body.smtp_host,
                smtp_port=body.smtp_port,
                email_password=body.email_password,
                discord_token=body.discord_token,
                persona_source=body.persona_source,
            )
        except Exception as e:
            raise HTTPException(400, str(e))
        return _profile_view(p)

    @app.post("/api/adopt")
    def adopt_profile(body: dict) -> dict:
        name = (body or {}).get("name", "")
        try:
            p = pm.adopt(name)
        except Exception as e:
            raise HTTPException(400, str(e))
        return _profile_view(p)

    @app.get("/api/profiles/{name}")
    def get_profile(name: str) -> dict:
        try:
            p = pm.get(name)
        except FileNotFoundError:
            raise HTTPException(404, f"No profile '{name}'")
        d = _profile_view(p)
        d["browser_dir"] = p.browser_dir
        # vault entry metadata (never values)
        with pm.open_vault(name) as v:
            d["vault"] = v.list()
        # is the assigned model registered / running?
        if p.model:
            m = mm.get(p.model)
            d["model_info"] = {
                "registered": bool(m),
                "running": mm.is_running(p.model) if m else False,
            }
        return d

    @app.get("/api/profiles/{name}/tools")
    def get_profile_tools(name: str) -> dict:
        """The character's tool belt: what's available, whether calls need
        approval or are blocked right now, and (where real data exists)
        when each tool was last actually used.

        Availability here is not a separate, invented notion of "enabled" --
        it's read straight off the same two gates the engine itself uses:
          - which tools EXIST in the belt at all: build_default_belt() only
            adds EmailTool when the profile has_email, and BrowserTool when
            the [browser] extra (camoufox) is importable -- a tool this
            character hasn't configured, or that isn't installed, simply
            isn't in the list.
          - whether a listed tool can run RIGHT NOW: exec_policy (this
            profile's override, else the process-wide default) -- "deny"
            blocks it, "ask" means every call needs approval, "allow" means
            it runs immediately, and "preset" defers to THIS tool's own
            entry in profile.tool_policies (default "allow" if unset).
            Tool.safe (read-only tools) always bypasses this gate, exactly
            like ToolBelt.dispatch does.

        `last_used`/`use_count` come from chats.tool_usage(), which mines
        real completed tool calls out of this character's persisted chat
        transcripts -- see that function's docstring for why `last_used` is
        left null for calls made before per-call timestamps existed, rather
        than guessed from a chat file's mtime.
        """
        try:
            p = pm.get(name)
        except FileNotFoundError:
            raise HTTPException(404, f"No profile '{name}'")

        from .. import chats as _chats
        from ..tools import ToolContext, build_default_belt

        with pm.open_vault(name) as vault:
            ctx = ToolContext(profile=p, vault=vault, settings=settings)
            belt = build_default_belt(ctx)

        policy = getattr(p, "exec_policy", None) or settings.exec_policy
        usage = _chats.tool_usage(name)

        tools = []
        for tool_name in belt.names():
            tool = belt.get(tool_name)
            # Per-tool override; only consulted (below) when policy=="preset",
            # but always reported so the sidebar can render the right toggle
            # state even before the character is switched into preset mode.
            tool_policy = p.tool_policies.get(tool_name, "allow")
            if tool.safe:
                status = "available"
            elif policy == "preset":
                status = "needs_approval" if tool_policy == "ask" else "available"
            elif policy == "allow":
                status = "available"
            elif policy == "deny":
                status = "blocked"
            else:  # "ask"
                status = "needs_approval"
            u = usage.get(tool_name, {"count": 0, "last_used": None})
            tools.append({
                "name": tool.name,
                "description": tool.description,
                "safe": tool.safe,
                "status": status,
                "tool_policy": tool_policy,
                "use_count": u["count"],
                "last_used": u["last_used"],
            })

        # How many MORE tools this character can reach on demand through the
        # harness tool-search bridge (find_tools/list_catalog/call_tool --
        # see engine.py's _bridge_harness_tools and harness/toolsearch.py).
        # Those tools aren't enumerated here: they're a much larger, more
        # volatile registry (files/shell/git/web/memory/subagents/MCP...)
        # not scoped per-character the way the belt is, and the whole point
        # of tool-search is that their schemas are loaded lazily rather than
        # always resident -- a real count is useful context, a fake per-tool
        # breakdown here would not be.
        bridge_count = 0
        try:
            from ..harness.tools import registry as _harness_registry
            bridge_count = len([t for t in _harness_registry.all()
                                if "search" not in t.tags])
        except Exception:
            bridge_count = 0

        return {
            "character": name,
            "exec_policy": policy,
            "tools": tools,
            "bridge_count": bridge_count,
        }

    @app.post("/api/profiles/{name}/tools/{tool_name}/policy")
    def set_tool_policy(name: str, tool_name: str, body: ToolPolicyUpdate) -> dict:
        """Set (or clear) THIS character's per-tool ask/allow override --
        the sidebar's per-tool toggle in RightPanel.tsx's ToolsSection.

        Only takes effect on tool-call gating while exec_policy=="preset"
        (see ToolBelt.dispatch); harmless to set at any other time, it just
        sits unused in profile.tool_policies until the character is switched
        into preset mode. `policy: null` (or omitted) clears the override,
        reverting the tool to the implicit "allow" default.
        """
        try:
            p = pm.get(name)
        except FileNotFoundError:
            raise HTTPException(404, f"No profile '{name}'")
        if body.policy is not None and body.policy not in ("ask", "allow"):
            raise HTTPException(400, "policy must be one of: ask, allow (or omit/null to clear)")

        from ..tools import ToolContext, build_default_belt

        with pm.open_vault(name) as vault:
            ctx = ToolContext(profile=p, vault=vault, settings=settings)
            belt = build_default_belt(ctx)
        if tool_name not in belt:
            raise HTTPException(404, f"No tool '{tool_name}' in {name}'s tool belt")

        if body.policy is None:
            p.tool_policies.pop(tool_name, None)
        else:
            p.tool_policies[tool_name] = body.policy
        pm._save(p)  # noqa: SLF001
        return {
            "character": name,
            "tool": tool_name,
            "policy": p.tool_policies.get(tool_name, "allow"),
        }

    @app.put("/api/profiles/{name}")
    def update_profile(name: str, body: ProfileUpdate) -> dict:
        try:
            p = pm.get(name)
        except FileNotFoundError:
            raise HTTPException(404, f"No profile '{name}'")
        if body.exec_policy is not None and body.exec_policy not in ("allow", "ask", "deny", "preset"):
            raise HTTPException(400, "exec_policy must be one of: allow, ask, deny, preset")
        for f in ("email_address", "imap_host", "imap_port", "smtp_host",
                  "smtp_port", "persona_source", "discord_enabled",
                  "discord_require_mention", "discord_respond_to_bots",
                  "discord_allowed_channels", "exec_policy",
                  "agents_enabled", "agents_model"):
            val = getattr(body, f)
            if val is not None:
                setattr(p, f, val)
        pm._save(p)  # noqa: SLF001
        return _profile_view(p)

    @app.delete("/api/profiles/{name}")
    def delete_profile(name: str, delete_anamnesis: bool = True) -> dict:
        try:
            pm.delete(name, delete_anamnesis=delete_anamnesis)
        except ValueError as e:  # invalid / path-traversal name
            raise HTTPException(400, str(e))
        return {"ok": True}

    @app.post("/api/profiles/{name}/model")
    def assign_model(name: str, body: AssignModel) -> dict:
        try:
            p = pm.get(name)
        except FileNotFoundError:
            raise HTTPException(404, f"No profile '{name}'")
        if body.model and not mm.get(body.model):
            raise HTTPException(400, f"Model '{body.model}' is not registered.")
        p.model = body.model
        pm._save(p)  # noqa: SLF001
        if body.model:
            pm.assign_model(name, body.model)  # repoints upstream if running
        return _profile_view(p)

    # email convenience: write profile fields + vault password in one shot
    @app.post("/api/profiles/{name}/email")
    def set_email(name: str, body: EmailConfig) -> dict:
        try:
            p = pm.get(name)
        except FileNotFoundError:
            raise HTTPException(404, f"No profile '{name}'")
        p.email_address = body.email_address
        p.imap_host = body.imap_host
        p.imap_port = body.imap_port
        p.smtp_host = body.smtp_host
        p.smtp_port = body.smtp_port
        pm._save(p)  # noqa: SLF001
        with pm.open_vault(name) as v:
            if body.password:
                v.set("email.password", body.password, meta={"reserved": True})
            v.delete("email.address")  # legacy duplicate; profile.json owns the address
        return _profile_view(p)

    @app.post("/api/profiles/{name}/discord")
    def set_discord(name: str, body: DiscordConfig) -> dict:
        try:
            p = pm.get(name)
        except FileNotFoundError:
            raise HTTPException(404, f"No profile '{name}'")
        with pm.open_vault(name) as v:
            if body.token:
                v.set("discord.token", body.token, meta={"reserved": True})
        p.discord_enabled = body.enabled
        if body.require_mention is not None:
            p.discord_require_mention = body.require_mention
        if body.respond_to_bots is not None:
            p.discord_respond_to_bots = body.respond_to_bots
        if body.allowed_channels is not None:
            p.discord_allowed_channels = body.allowed_channels
        pm._save(p)  # noqa: SLF001
        return _profile_view(p)

    @app.get("/api/profiles/{name}/discord/info")
    def discord_info(name: str) -> dict:
        """Validate the saved token and list the servers the bot is in."""
        with pm.open_vault(name) as v:
            token = v.get("discord.token")
        if not token:
            return {"configured": False}
        if not httpx:
            raise HTTPException(500, "httpx unavailable")
        headers = {"Authorization": f"Bot {token}"}
        try:
            me = httpx.get("https://discord.com/api/v10/users/@me",
                           headers=headers, timeout=8.0)
            if me.status_code == 401:
                return {"configured": True, "valid": False,
                        "error": "Discord rejected the token (401)."}
            me.raise_for_status()
            guilds = httpx.get("https://discord.com/api/v10/users/@me/guilds",
                               headers=headers, timeout=8.0)
            guilds.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(502, f"Discord API error: {e}")
        bot = me.json()
        return {"configured": True, "valid": True,
                "bot": {"username": bot.get("username", ""), "id": bot.get("id", "")},
                "guilds": [{"id": g["id"], "name": g["name"]} for g in guilds.json()]}

    # ----------------------------- email inbox ------------------------------ #
    def _email_conn(name: str):
        try:
            p = pm.get(name)
        except FileNotFoundError:
            raise HTTPException(404, f"No profile '{name}'")
        with pm.open_vault(name) as v:
            password = v.get("email.password")
        if not (p.has_email and password):
            raise HTTPException(400, "Email is not fully configured for this character.")
        from ..tools import email_box
        try:
            return p, password, email_box.imap_connect(p.imap_host, p.imap_port,
                                                       p.email_address, password)
        except Exception as e:
            raise HTTPException(502, f"IMAP connection failed: {e}")

    @app.get("/api/profiles/{name}/email/inbox")
    def email_inbox(name: str, limit: int = 25, unread: bool = False) -> dict:
        from ..tools import email_box
        p, _pw, conn = _email_conn(name)
        try:
            msgs = email_box.list_messages(conn, "UNSEEN" if unread else "ALL",
                                           limit=min(limit, 100))
        finally:
            try:
                conn.logout()
            except Exception:
                pass
        return {"address": p.email_address, "messages": msgs}

    @app.get("/api/profiles/{name}/email/message/{mid}")
    def email_message(name: str, mid: str) -> dict:
        from ..tools import email_box
        _p, _pw, conn = _email_conn(name)
        try:
            return email_box.read_message(conn, mid)
        except RuntimeError as e:
            raise HTTPException(404, str(e))
        finally:
            try:
                conn.logout()
            except Exception:
                pass

    @app.post("/api/profiles/{name}/email/send")
    def email_send(name: str, body: EmailSend) -> dict:
        from ..tools import email_box
        p, password, conn = _email_conn(name)
        try:
            conn.logout()
        except Exception:
            pass
        try:
            email_box.send_message(p.smtp_host, p.smtp_port, p.email_address,
                                   password, body.to, body.subject, body.body)
        except Exception as e:
            raise HTTPException(502, f"send failed: {e}")
        return {"ok": True}

    # ----------------------------- avatar ----------------------------------- #
    _AVATAR_MIME = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}

    def _avatar_path(name: str, ext: str):
        return config.profile_dir(name) / f"avatar.{ext}"

    def _avatar_existing(name: str):
        d = config.profile_dir(name)
        for ext in ("png", "jpg", "jpeg", "webp"):
            p = d / f"avatar.{ext}"
            if p.is_file():
                return p
        return None

    @app.get("/api/profiles/{name}/avatar")
    def avatar_get(name: str):
        try:
            p = _avatar_existing(name)
        except ValueError:
            raise HTTPException(400, "bad name")
        if not p:
            raise HTTPException(404, "no avatar")
        mt = _AVATAR_MIME.get(p.suffix.lstrip("."), "image/png")
        return FileResponse(str(p), media_type=mt,
                            headers={"Cache-Control": "no-cache"})

    @app.post("/api/profiles/{name}/avatar")
    def avatar_set(name: str, body: AvatarBody) -> dict:
        import base64
        try:
            pm.get(name)
        except FileNotFoundError:
            raise HTTPException(404, f"No profile '{name}'")
        m = re.match(r"data:image/(png|jpeg|jpg|webp);base64,(.+)", body.data, re.S)
        if not m:
            raise HTTPException(400, "Expected a data URL (png/jpeg/webp).")
        fmt = m.group(1).lower()
        try:
            blob = base64.b64decode(m.group(2))
        except Exception:
            raise HTTPException(400, "Invalid base64 image data.")
        if len(blob) > 2 * 1024 * 1024:
            raise HTTPException(400, "Image too large (max 2 MB).")
        old = _avatar_existing(name)
        if old:
            old.unlink()
        _avatar_path(name, fmt).write_bytes(blob)
        return {"ok": True}

    @app.delete("/api/profiles/{name}/avatar")
    def avatar_delete(name: str) -> dict:
        try:
            p = _avatar_existing(name)
        except ValueError:
            raise HTTPException(400, "bad name")
        if p:
            p.unlink()
        return {"ok": True}

    # ----------------------------- vault ----------------------------------- #
    @app.get("/api/profiles/{name}/vault")
    def vault_list(name: str) -> dict:
        with pm.open_vault(name) as v:
            return {"entries": v.list()}

    @app.post("/api/profiles/{name}/vault")
    def vault_set(name: str, body: VaultEntry) -> dict:
        meta = {"reserved": True} if body.key in RESERVED_KEYS else None
        if body.note:
            meta = {**(meta or {}), "note": body.note}
        with pm.open_vault(name) as v:
            v.set(body.key, body.value, meta=meta)
            return {"entries": v.list()}

    @app.get("/api/profiles/{name}/vault/{key:path}")
    def vault_reveal(name: str, key: str) -> dict:
        """Explicit, on-demand decrypt of a single value (user clicked 'reveal')."""
        with pm.open_vault(name) as v:
            val = v.get(key)
        if val is None:
            raise HTTPException(404, f"No vault entry '{key}'")
        return {"key": key, "value": val}

    @app.delete("/api/profiles/{name}/vault/{key:path}")
    def vault_delete(name: str, key: str) -> dict:
        with pm.open_vault(name) as v:
            ok = v.delete(key)
        return {"ok": ok}

    # ----------------------------- anamnesis ------------------------------ #

    @app.get("/api/profiles/{name}/anamnesis")
    def get_anamnesis(name: str) -> dict:
        """Character Anamnesis config + memory stats + running model list."""
        try:
            pm.get(name)
        except FileNotFoundError:
            raise HTTPException(404, f"No profile '{name}'")
        cfg = _read_anamnesis_cfg(name)
        stats = _memory_stats(name)
        running_models = [
            {"name": m["name"], "port": m["port"],
             "url": f"http://127.0.0.1:{m['port']}/v1"}
            for m in mm.list() if m.get("running")
        ]
        try:
            char = an.get_character(name)
            char_status = "running" if (char.get("running") or char.get("active")) else "stopped"
        except Exception:
            char_status = "unknown"
        return {"config": cfg, "stats": stats,
                "running_models": running_models, "char_status": char_status}

    @app.put("/api/profiles/{name}/anamnesis/upstream")
    def set_anamnesis_upstream(name: str, body: AnamnesisUpstreamUpdate) -> dict:
        """Update upstream.baseUrl in config.json and restart the character proxy."""
        import json as _json
        try:
            pm.get(name)
        except FileNotFoundError:
            raise HTTPException(404, f"No profile '{name}'")
        cfg_path = _anamnesis_config_path(name)
        cfg = _read_anamnesis_cfg(name)
        upstream = cfg.get("upstream", {})
        upstream["baseUrl"] = body.base_url.rstrip("/")
        if body.api_key is not None:
            upstream["apiKey"] = body.api_key
        if body.model_name is not None:
            upstream["model"] = body.model_name
        cfg["upstream"] = upstream
        cfg_path.write_text(_json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        # Restart via Anamnesis control API
        try:
            an.stop(name)
        except Exception:
            pass
        try:
            an.start(name)
        except Exception as exc:
            raise HTTPException(502, f"Config saved but restart failed: {exc}") from exc
        return {"ok": True, "upstream": upstream}

    @app.get("/api/profiles/{name}/anamnesis/memory")
    def get_anamnesis_memory(name: str, kind: str = "scenes",
                             page: int = 0, per_page: int = 20) -> dict:
        """Paginated memory scenes, engrams, or observations for a character."""
        try:
            pm.get(name)
        except FileNotFoundError:
            raise HTTPException(404, f"No profile '{name}'")
        db = _anamnesis_db_path(name)
        if not db:
            return {"items": [], "total": 0, "kind": kind, "page": page, "per_page": per_page}
        per_page = min(max(per_page, 1), 100)
        offset = page * per_page
        try:
            import sqlite3
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            if kind == "scenes":
                total = cur.execute("SELECT COUNT(*) FROM memscenes").fetchone()[0]
                rows = cur.execute(
                    "SELECT id, title, summary, recall_count, avg_importance, "
                    "created_at, updated_at FROM memscenes "
                    "ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                    (per_page, offset)).fetchall()
            elif kind == "engrams":
                total = cur.execute("SELECT COUNT(*) FROM engrams").fetchone()[0]
                rows = cur.execute(
                    "SELECT id, content, category, decay_score, importance, "
                    "recall_count, created_at FROM engrams "
                    "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (per_page, offset)).fetchall()
            elif kind == "observations":
                total = cur.execute("SELECT COUNT(*) FROM character_observations").fetchone()[0]
                rows = cur.execute(
                    "SELECT * FROM character_observations "
                    "ORDER BY rowid DESC LIMIT ? OFFSET ?",
                    (per_page, offset)).fetchall()
            else:
                raise HTTPException(400, "kind must be scenes, engrams, or observations")
            items = [dict(r) for r in rows]
            con.close()
            return {"items": items, "total": total, "kind": kind,
                    "page": page, "per_page": per_page}
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(500, f"Could not read memory: {exc}") from exc

    # ----------------------------- hardware -------------------------------- #
    @app.get("/api/profiles/{name}/anamnesis/inspect")
    def anamnesis_inspect(name: str, limit: int = 15) -> dict:
        """Browse a character's self-organized memory: scenes, engrams, observations."""
        db = _anamnesis_db_path(name)
        if not db:
            return {"scenes": [], "engrams": [], "observations": []}
        import sqlite3
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
            cur = con.cursor()

            def rows(sql, args=()):
                try:
                    return cur.execute(sql, args).fetchall()
                except Exception:
                    return []

            scenes = [{"title": t, "summary": (s or "")[:400], "recall": rc,
                       "importance": round(ai or 0, 2)}
                      for (t, s, rc, ai) in rows(
                      "SELECT title, summary, recall_count, avg_importance FROM memscenes "
                      "ORDER BY recall_count DESC, updated_at DESC LIMIT ?", (limit,))]
            engrams = [{"content": (c or "")[:300], "category": cat,
                        "importance": round(imp or 0, 2), "recall": rc}
                       for (c, cat, imp, rc) in rows(
                       "SELECT content, category, importance, recall_count FROM engrams "
                       "ORDER BY importance DESC, recall_count DESC LIMIT ?", (limit,))]
            obs = [{"type": ot, "detail": (d or "")[:300]} for (ot, d) in rows(
                   "SELECT obs_type, detail FROM character_observations "
                   "ORDER BY observed_at DESC LIMIT ?", (limit,))]
            con.close()
            return {"scenes": scenes, "engrams": engrams, "observations": obs}
        except Exception as e:
            return {"scenes": [], "engrams": [], "observations": [], "error": str(e)}

    @app.get("/api/hardware")
    def hardware_info() -> dict:
        from .. import hardware, runtime
        from ..autofit import place
        det = hardware.detect()
        cps = runtime.caps()
        engine_bin = runtime.find_native_cpp_engine()
        engine_info = None
        if engine_bin:
            caps_json = runtime.query_native_cpp_engine_caps(engine_bin) or {}
            engine_info = {"path": engine_bin,
                           "backend": caps_json.get("backend"),
                           "moe_offload": caps_json.get("moe_offload"),
                           "vision": caps_json.get("vision")}
        plans = []
        for m in mm.list():
            meta = hardware.read_gguf_meta(m["path"])
            try:
                ctx_for_plan = int(m.get("n_ctx", 8192))
            except (TypeError, ValueError):
                # "auto" (the default since this session's ctx-planning fix) —
                # show the placement for what actually launches: the elastic
                # ceiling plan_context() already verified fits, same value
                # launch.py uses for the native runtime.
                ctx_for_plan = hardware.plan_context(meta, det, caps=cps).n_ctx_max
            p = place(meta, ctx_for_plan, det, cps)
            plans.append({"model": m["name"],
                          "n_gpu_layers": p.n_gpu_layers,
                          "n_layers": meta.n_layers,
                          "strategy": p.strategy, "est_tps": p.est_tps,
                          "moe": meta.is_moe, "n_cpu_moe": p.n_cpu_moe,
                          "reason": p.reason,
                          "fits_fully": p.strategy == "full_gpu"})
        return {
            "gpus": [{"vendor": g.vendor, "name": g.name,
                      "vram_total": g.vram_total, "vram_free": g.vram_free}
                     for g in det.gpus],
            "ram_total": det.ram_total, "ram_available": det.ram_available,
            "cpu_threads": det.cpu_threads, "unified_memory": det.unified_memory,
            "summary": det.describe(), "plans": plans,
            "runtime": runtime.status(),
            "engine": engine_info,
        }

    @app.get("/api/browser/status")
    def browser_status_api() -> dict:
        """Camoufox/Playwright backend, profile dir, and last launch error —
        the "logs and settings" the browser tool otherwise has none of."""
        from ..harness.builtins.browser import browser_status
        return browser_status()

    @app.get("/api/discord/status")
    def discord_status_api() -> dict:
        """Per-character Discord bot service state, token presence, and last log
        line — the "logs and online status" the connector otherwise has none of."""
        from ..connectors.discord_bot import discord_status
        return {"characters": discord_status()}

    @app.get("/api/memory/{name}")
    def memory_api(name: str) -> dict:
        """Read-only window into a character's Anamnesis memory: scenes,
        active foresights, engram mix, and the two health signals that cause
        real drift — fragmented session buckets and mixed embedding vector
        spaces. Reads history.db directly (read-only URI): the proxy needn't
        be running and nothing here can mutate state."""
        import sqlite3
        db = Path.home() / ".anamnesis" / "characters" / name / "history.db"
        if not db.is_file():
            return {"exists": False}
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            def q(sql: str, *a: Any) -> list[dict]:
                return [dict(r) for r in con.execute(sql, a).fetchall()]

            def one(sql: str) -> int:
                return con.execute(sql).fetchone()[0]

            scenes = q(
                """SELECT id, title, summary, avg_importance, recall_count, updated_at,
                          json_array_length(engram_ids) AS engrams, embedding_model
                   FROM episodes ORDER BY updated_at DESC LIMIT 60"""
            )
            foresights = q(
                """SELECT id, intention, target, timeframe, confidence, created_at
                   FROM foresights WHERE fulfilled=0 ORDER BY created_at DESC LIMIT 60"""
            )
            return {
                "exists": True,
                "now": int(time.time()),
                "turns": one("SELECT COUNT(*) FROM turns"),
                "engrams": one("SELECT COUNT(*) FROM engrams"),
                "scenes_total": one("SELECT COUNT(*) FROM episodes"),
                "scenes": scenes,
                "foresights": foresights,
                "categories": q(
                    "SELECT category, COUNT(*) AS n FROM engrams GROUP BY category ORDER BY n DESC"
                ),
                "vector_spaces": q(
                    """SELECT COALESCE(embedding_model,'(none)') AS model, COUNT(*) AS n
                       FROM turns GROUP BY 1 ORDER BY n DESC"""
                ),
                "sessions": q(
                    """SELECT session_key, COUNT(*) AS turns FROM turns
                       GROUP BY 1 ORDER BY turns DESC"""
                ),
            }
        finally:
            con.close()

    @app.post("/api/memory/{name}/foresight/{fid}/dismiss")
    def memory_foresight_dismiss(name: str, fid: str) -> dict:
        """Manually retire a foresight (fulfilled=3 — dismissed by owner) so
        it stops being injected. Sets one flag; nothing is deleted."""
        import sqlite3
        db = Path.home() / ".anamnesis" / "characters" / name / "history.db"
        if not db.is_file():
            raise HTTPException(404, "No memory store for this character.")
        con = sqlite3.connect(db)
        try:
            cur = con.execute(
                "UPDATE foresights SET fulfilled=3 WHERE id=? AND fulfilled=0", (int(fid),)
            )
            con.commit()
            n = cur.rowcount
        finally:
            con.close()
        if not n:
            raise HTTPException(404, "No such active foresight.")
        return {"ok": True}

    @app.post("/api/runtime/install")
    def runtime_install_api() -> dict:
        from .. import runtime
        try:
            path = runtime.install()
        except Exception as e:
            raise HTTPException(502, str(e))
        return {"ok": True, "path": path}

    # ------------------------- fetch from Hugging Face ---------------------- #
    fetch_state: dict[str, Any] = {}

    @app.post("/api/models/fetch")
    def models_fetch(body: ModelFetch) -> dict:
        if fetch_state.get("status") == "downloading":
            raise HTTPException(409, "A download is already in progress.")
        from ..fetch import FetchError, fetch_model

        def progress(fname: str, done: int, total: int) -> None:
            fetch_state.update(file=fname, done=done, total=total)

        def work() -> None:
            fetch_state.update(status="downloading", repo=body.repo, error="",
                               file="", done=0, total=0)
            try:
                entry = fetch_model(body.repo, name=body.name, quant=body.quant,
                                    progress=progress)
                fetch_state.update(status="done", result=entry)
            except (FetchError, Exception) as e:  # surface anything to the UI
                fetch_state.update(status="error", error=str(e))

        threading.Thread(target=work, daemon=True).start()
        return {"ok": True}

    @app.get("/api/models/fetch/status")
    def models_fetch_status() -> dict:
        return fetch_state or {"status": "idle"}

    # ----------------------------- models ---------------------------------- #
    @app.get("/api/models")
    def models_list() -> dict:
        return {"models": mm.list(), "running": mm.running()}

    @app.post("/api/models")
    def models_add(body: ModelCreate) -> dict:
        try:
            entry = mm.add(body.name, body.path, n_ctx=body.n_ctx,
                           n_gpu_layers=body.n_gpu_layers, chat_format=body.chat_format,
                           mmproj=body.mmproj, capabilities=body.capabilities)
        except ModelError as e:
            raise HTTPException(400, str(e))
        entry = dict(entry)
        entry["running"] = False
        return entry

    @app.put("/api/models/{name}")
    def models_update(name: str, body: ModelUpdate) -> dict:
        try:
            return _model_mgr_update(mm, name, body)
        except ModelError as e:
            raise HTTPException(400, str(e))

    @app.delete("/api/models/{name}")
    def models_remove(name: str) -> dict:
        return {"ok": mm.remove(name)}

    @app.post("/api/models/{name}/start")
    def models_start(name: str) -> dict:
        try:
            url = mm.start(name, wait=False)   # card shows 'loading' until healthy
        except ModelError as e:
            raise HTTPException(400, str(e))
        return {"ok": True, "base_url": url, "state": mm.state(name)}

    @app.get("/api/models/{name}/logs")
    def models_logs(name: str, lines: int = 250) -> dict:
        return {"name": name, "state": mm.state(name),
                "log": mm.log_tail(name, lines=min(lines, 2000))}

    @app.get("/api/models/hf-search")
    def models_hf_search(q: str, limit: int = 8) -> dict:
        from ..fetch import FetchError, search_gguf_repos
        try:
            return {"results": search_gguf_repos(q, limit=min(limit, 20))}
        except FetchError as e:
            raise HTTPException(502, str(e))

    @app.get("/api/models/quant-options")
    def models_quant_options(repo: str, n_ctx: int = 8192,
                             preference: str = "balanced") -> dict:
        """For an HF repo, list GGUF quants with autofit placement for THIS machine.

        Powers the foundry: each candidate quant gets size + VRAM fit + GPU layer
        split + estimated tok/s, and the recommendation for the chosen preference.
        """
        from ..fetch import FetchError, list_repo_ggufs, remote_gguf_meta
        from .. import hardware as hw_mod, runtime, autofit
        try:
            files = list_repo_ggufs(repo)
        except FetchError as e:
            raise HTTPException(502, str(e))
        grouped = hw_mod.group_split_ggufs(files)
        by_quant: dict[str, dict] = {}
        for g in grouped:
            q = hw_mod.quant_of(g["name"])
            if q and q in autofit.QUALITY and (q not in by_quant or g["size"] < by_quant[q]["size"]):
                by_quant[q] = g
        if not by_quant:
            raise HTTPException(404, f"No recognized GGUF quants in '{repo}'.")
        hw = hw_mod.detect()
        caps = runtime.caps()
        smallest = min(by_quant.values(), key=lambda g: g["size"])
        meta_hint = remote_gguf_meta(repo, smallest["name"])
        opts = []
        for q, entry in sorted(by_quant.items(), key=lambda kv: autofit.QUALITY[kv[0]]):
            meta = autofit._scaled_meta(meta_hint, entry["size"])
            pl = autofit.place(meta, n_ctx, hw, caps)
            opts.append({"quant": q, "file": entry["name"], "size": entry["size"],
                         "quality": autofit.QUALITY[q], "strategy": pl.strategy,
                         "n_gpu_layers": pl.n_gpu_layers, "est_tps": pl.est_tps,
                         "vram_used": pl.vram_used, "fits_fully": pl.strategy == "full_gpu",
                         "feasible": pl.feasible, "reason": pl.reason})
        try:
            rec = autofit.choose_quant(files, hw, n_ctx, preference, caps, meta_hint)
        except Exception:
            rec = None
        gpu = hw.gpu
        return {"repo": repo, "preference": preference,
                "n_layers": getattr(meta_hint, "n_layers", 0) or 0,
                "is_moe": bool(getattr(meta_hint, "is_moe", False)),
                "hardware": {"summary": hw.describe(),
                             "gpu": gpu.name if gpu else "", "vendor": gpu.vendor if gpu else "",
                             "vram_total": gpu.vram_total if gpu else 0,
                             "vram_free": gpu.vram_free if gpu else 0,
                             "ram_available": hw.ram_available},
                "recommended": rec.quant if rec else None,
                "options": opts}

    @app.get("/api/models/cloud-search")
    def cloud_search(source: str, q: str = "", limit: int = 25) -> dict:
        import httpx
        s = config.Settings.load()
        ql = (q or "").lower()
        if source == "openrouter":
            try:
                r = httpx.get("https://openrouter.ai/api/v1/models", timeout=15)
                data = r.json().get("data", [])
            except Exception as e:
                raise HTTPException(502, f"OpenRouter unreachable: {e}")
            out = []
            for m in data:
                mid = m.get("id", "")
                if ql and ql not in mid.lower() and ql not in (m.get("name", "").lower()):
                    continue
                pr = m.get("pricing", {}) or {}
                is_free = (pr.get("prompt") in ("0", 0, "0.0") and pr.get("completion") in ("0", 0, "0.0"))
                # architecture.modality (e.g. "text->text", "text+image->text")
                # is OpenRouter's own declared capability listing -- surfaced
                # here so a caller (desktop app model picker, or
                # assign_cloud_model below) can tell whether a cloud model
                # takes image input without us having any local file to sniff.
                modality = ((m.get("architecture") or {}).get("modality") or "")
                out.append({"id": mid, "name": m.get("name", mid),
                            "context": m.get("context_length"),
                            "prompt_price": pr.get("prompt"),
                            "completion_price": pr.get("completion"),
                            "is_free": is_free,
                            "modality": modality,
                            "vision": "image" in modality.lower()})
            out.sort(key=lambda x: (0 if x.get("is_free") else 1, x["id"]))
            return {"source": "openrouter", "results": out[:limit]}
        if source == "ollama":
            base = (s.ollama_cloud_url or "https://ollama.com/v1").rstrip("/")
            _ok = s.ollama_cloud_api_keys[0] if getattr(s, "ollama_cloud_api_keys", None) else ""
            headers = {"Authorization": f"Bearer {_ok}"} if _ok else {}
            try:
                r = httpx.get(base + "/models", headers=headers, timeout=15)
                data = r.json().get("data", [])
                out = [{"id": m.get("id"), "name": m.get("id")} for m in data
                       if not ql or ql in (m.get("id", "") or "").lower()]
                return {"source": "ollama", "results": out[:limit]}
            except Exception as e:
                return {"source": "ollama", "results": [],
                        "error": "Set an Ollama Cloud key/URL in Settings (or endpoint unreachable).",
                        "detail": str(e)}
        raise HTTPException(400, "source must be 'openrouter' or 'ollama'")

    @app.post("/api/profiles/{name}/cloud-model")
    def assign_cloud_model(name: str, body: CloudModelBody) -> dict:
        try:
            p = pm.get(name)
        except FileNotFoundError:
            raise HTTPException(404, f"No profile '{name}'")
        prefix = "openrouter:" if body.source == "openrouter" else "ollama-cloud:"
        p.model = prefix + body.model
        capabilities = body.capabilities
        if not capabilities and body.source == "openrouter":
            # Best-effort auto-populate from OpenRouter's own model listing
            # (architecture.modality) -- see cloud_search above for the same
            # extraction. Never fatal: an unreachable/slow OpenRouter here
            # just means the character keeps model_capabilities blank (=
            # text-only assumed), the same as it always used to be.
            try:
                import httpx as _httpx
                r = _httpx.get("https://openrouter.ai/api/v1/models", timeout=8)
                for m in r.json().get("data", []):
                    if m.get("id") == body.model:
                        modality = ((m.get("architecture") or {}).get("modality") or "")
                        if "image" in modality.lower():
                            capabilities = "vision"
                        break
            except Exception:
                pass
        p.model_capabilities = capabilities
        pm._save(p)  # noqa: SLF001
        return _profile_view(p)

    @app.post("/api/models/{name}/stop")
    def models_stop(name: str) -> dict:
        return {"ok": mm.stop(name)}

    class ResizeBody(BaseModel):
        n_ctx: int

    @app.post("/api/models/{name}/resize")
    def models_resize(name: str, body: ResizeBody) -> dict:
        """Resize a running elastic server's KV cache in place (no model reload)."""
        try:
            return mm.resize(name, body.n_ctx)
        except ModelError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/email/presets")
    def email_presets() -> dict:
        return config.EMAIL_PRESETS

    # ----------------------------- chat sessions ---------------------------- #
    @app.get("/api/chats")
    def chats_list() -> dict:
        from .. import chats
        return {"chats": chats.list_chats()}

    @app.post("/api/chats")
    def chats_create(body: ChatCreate) -> dict:
        from .. import chats
        try:
            pm.get(body.character)
        except FileNotFoundError:
            raise HTTPException(404, f"No profile '{body.character}'")
        return chats.create(body.character)

    @app.get("/api/chats/{chat_id}")
    def chats_get(chat_id: str) -> dict:
        from .. import chats
        try:
            return chats.load(chat_id)
        except (FileNotFoundError, ValueError):
            raise HTTPException(404, "No such chat.")

    @app.post("/api/chats/{chat_id}/attachments")
    async def chats_attachment_upload(chat_id: str, file: UploadFile = File(...)) -> dict:
        """Save one uploaded file into THIS chat's own attachment cache dir
        (see pleiades/attachments.py) and return {id, filename, path, mime,
        size}. The composer sends the returned `id` back in a later POST
        .../message's `attachments` list.

        This is deliberately dumb about file *content* -- no image
        captioning, no vision content-parts, no audio transcription happens
        here (those are separate, parallel efforts). All this endpoint does
        is give the model a REAL path on disk it can point its existing
        file/shell tools at (see engine.py's `_attachments_note`) -- which is
        exactly what makes "drop a logo, then say 'use it on the site'" work
        even for a text-only model.

        Scoped by chat_id, not a flat shared workspace dir -- see
        attachments.py's module docstring for why (the existing character
        workspace is shared across every chat AND every background/
        scheduled-task job for that character, so same-named uploads could
        collide there).
        """
        from .. import chats
        from .. import attachments as chat_attachments
        try:
            chat = chats.load(chat_id)
        except (FileNotFoundError, ValueError):
            raise HTTPException(404, "No such chat.")
        data = await file.read()
        try:
            return chat_attachments.save_attachment(
                chat["character"], chat_id, file.filename or "file", data)
        except (ValueError, OSError) as e:
            raise HTTPException(400, f"Could not save attachment: {e}")

    @app.get("/api/chats/{chat_id}/attachments/{filename}")
    def chats_attachment_get(chat_id: str, filename: str):
        """Stream one previously-uploaded attachment's raw bytes back (see
        POST .../attachments above + pleiades/attachments.py) so the desktop
        UI can render a persisted attachment chip's real thumbnail/`<audio>`
        player against a real URL (see MessageBubble.tsx) -- the upload
        endpoint only ever returned a path on THIS machine, which is no use
        to an `<img src=...>` in the renderer process.

        Same path-traversal guard as the upload endpoint, just applied to
        the filename coming back out instead of going in -- see
        attachments.attachment_path / sanitize_filename.
        """
        from .. import chats
        from .. import attachments as chat_attachments
        try:
            chat = chats.load(chat_id)
        except (FileNotFoundError, ValueError):
            raise HTTPException(404, "No such chat.")
        path = chat_attachments.attachment_path(chat["character"], chat_id, filename)
        if not path:
            raise HTTPException(404, "No such attachment.")
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return FileResponse(str(path), media_type=mime, headers={"Cache-Control": "no-cache"})

    @app.delete("/api/chats/{chat_id}")
    def chats_delete(chat_id: str) -> dict:
        from .. import chats
        from .. import attachments as chat_attachments
        # Cache-lifecycle decision (see attachments.py's module docstring):
        # the attachment cache is scoped by chat_id and is ONLY ever deleted
        # here, on a real chat delete -- never on new-chat, because
        # startNewChat() archives the outgoing chat into History rather than
        # deleting it, and wiping a shared cache at that point would break
        # image references still visible in that archived chat. Read the
        # character BEFORE deleting the chat file so we know which
        # profile's chat-cache/<chat_id>/ directory to remove.
        character = None
        try:
            character = chats.load(chat_id).get("character")
        except (FileNotFoundError, ValueError):
            pass
        try:
            ok = chats.delete(chat_id)
        except ValueError:
            raise HTTPException(400, "Bad chat id.")
        if character:
            chat_attachments.delete_chat_cache(character, chat_id)
        return {"ok": ok}

    # Pending tool approvals for chat turns. The NDJSON stream is stalled
    # while the engine waits inside the approve callback, so the frontend
    # polls GET /approval and answers via POST /approval.
    chat_approvals: dict[str, dict] = {}
    # Per-chat stop flags for the emergency-stop button: set() to cut a
    # streaming run short between rounds/chunks (see Engine.stream_events).
    chat_stops: dict[str, threading.Event] = {}
    # Mid-turn user interjections (see chats_interject / Engine.stream_events'
    # poll_injections): one Queue[str] per chat that currently has a turn
    # in flight. Registered/cleared in chats_message()'s try/finally exactly
    # like chat_approvals/chat_stops above.
    import queue as _queue_mod
    chat_inboxes: dict[str, "_queue_mod.Queue[str]"] = {}
    # Per-chat in-flight lock -- fixes a real pre-existing bug: nothing used
    # to stop two concurrent POST /message calls for the SAME chat_id from
    # both registering into chat_stops/chat_approvals (the second clobbers
    # the first's entries) and both driving the engine at once, interleaving
    # two separate turns into one transcript via append_turn. A plain
    # non-blocking Lock per chat_id turns the second concurrent request into
    # a clean 409 instead of silent corruption. Guarded by a small mutex so
    # two threads creating the SAME chat's lock for the first time can't
    # race and hand out two different Lock objects.
    chat_locks: dict[str, threading.Lock] = {}
    _chat_locks_guard = threading.Lock()

    def _lock_for(chat_id: str) -> threading.Lock:
        with _chat_locks_guard:
            lock = chat_locks.get(chat_id)
            if lock is None:
                lock = threading.Lock()
                chat_locks[chat_id] = lock
            return lock

    def _chat_approve_cb(chat_id: str, profile: Optional[Profile] = None):
        import threading

        ctl = {"pending": None, "event": threading.Event(), "answer": {"ok": False}}
        chat_approvals[chat_id] = ctl

        def approve(tool, args) -> bool:
            # Per-character override wins over the global default: a fresh
            # ProfileManager read here (not the `profile` snapshot taken at
            # stream-start) so a dropdown change lands on the very next tool
            # call of an in-flight turn, not just the next message.
            policy = None
            if profile is not None:
                try:
                    policy = getattr(pm.get(profile.name), "exec_policy", None)
                except Exception:
                    policy = getattr(profile, "exec_policy", None)
            if not policy:
                try:
                    from ..harness import Config as HarnessConfig
                    policy = HarnessConfig.load().exec_policy
                except Exception:
                    policy = "ask"
            if policy == "allow":
                return True
            if policy == "deny":
                return False
            import json as _json
            ctl["pending"] = {"tool": getattr(tool, "name", str(tool)),
                              "args": _json.dumps(args)[:300]}
            ctl["event"].clear()
            answered = ctl["event"].wait(timeout=600)  # 10 min, then deny
            ctl["pending"] = None
            return bool(ctl["answer"]["ok"]) if answered else False

        return approve

    @app.get("/api/chats/{chat_id}/approval")
    def chats_approval_get(chat_id: str) -> dict:
        ctl = chat_approvals.get(chat_id)
        return {"pending": ctl["pending"] if ctl else None}

    @app.post("/api/chats/{chat_id}/approval")
    def chats_approval_post(chat_id: str, body: WorkApprove) -> dict:
        ctl = chat_approvals.get(chat_id)
        if not ctl or not ctl["pending"]:
            raise HTTPException(409, "Nothing awaiting approval.")
        ctl["answer"]["ok"] = bool(body.approve)
        ctl["event"].set()
        return {"ok": True}

    @app.post("/api/chats/{chat_id}/stop")
    def chats_stop(chat_id: str) -> dict:
        """Emergency stop: cut a streaming run short at the next round or chunk."""
        evt = chat_stops.get(chat_id)
        if not evt:
            raise HTTPException(409, "No run in progress for this chat.")
        evt.set()
        ctl = chat_approvals.get(chat_id)
        if ctl and ctl["pending"]:  # unblock a tool stuck waiting on approval
            ctl["answer"]["ok"] = False
            ctl["event"].set()
        return {"ok": True}

    @app.post("/api/chats/{chat_id}/message")
    def chats_message(chat_id: str, body: ChatMessage) -> StreamingResponse:
        """One unified chat/agent turn, streamed as NDJSON events.

        Lines: {"type":"token","text"} | {"type":"tool_call","name","args"} |
               {"type":"tool_result","name","output","ok"} |
               {"type":"user_injected","text"} | {"type":"undelivered","text"} |
               {"type":"done","tokens","seconds","tps"} | {"type":"error","error"}
        The model has the FULL tool belt (chat + harness) and uses tools only
        when the prompt needs them. The turn is persisted to the chat file.

        Concurrency: a per-chat lock (see _lock_for) is acquired here, BEFORE
        the streaming generator is even returned to Starlette, and released
        in gen()'s finally. Two POSTs racing for the same chat_id used to
        both register into chat_stops/chat_approvals (the second clobbering
        the first's entries) and both drive the engine at once, interleaving
        two turns into one transcript via append_turn -- a real correctness
        bug independent of mid-turn interjection. The second racer now gets a
        clean 409 instead. A genuinely NEW message while a turn is already
        in flight for this chat should go through POST .../interject instead
        (see chats_interject) -- it weaves into the running turn rather than
        starting a second one.
        """
        from .. import chats
        from .. import attachments as chat_attachments
        try:
            chat = chats.load(chat_id)
            p = pm.get(chat["character"])
        except (FileNotFoundError, ValueError):
            raise HTTPException(404, "No such chat (or its character is gone).")

        # Resolve attachment ids (see POST .../attachments) against THIS
        # chat's own cache dir up front -- never trust the ids as raw paths.
        # Silently drops stale/bad ids rather than 400ing the whole turn
        # (see attachments.resolve_attachments).
        resolved_attachments = (
            chat_attachments.resolve_attachments(chat["character"], chat_id, body.attachments)
            if body.attachments else []
        )

        lock = _lock_for(chat_id)
        if not lock.acquire(blocking=False):
            raise HTTPException(409, "A turn is already in progress for this chat "
                                     "-- use .../interject to send a message into it.")

        def gen():
            import json as _json
            import queue as _q
            items: list[dict] = []
            text_acc = ""
            reasoning_acc = ""
            meta: dict = {}
            stop_evt = threading.Event()
            chat_stops[chat_id] = stop_evt
            # Mid-turn interjections land here (see chats_interject). Queued
            # BEFORE the try block so it's visible to callers the instant
            # this turn starts, and so the finally block below can still
            # drain anything left over even if engine construction itself
            # blows up.
            inbox: "_q.Queue[str]" = _q.Queue()
            chat_inboxes[chat_id] = inbox

            def _poll_injections() -> list[str]:
                out: list[str] = []
                while True:
                    try:
                        out.append(inbox.get_nowait())
                    except _q.Empty:
                        break
                return out

            try:
                from ..engine import Engine
                engine = Engine(settings=config.Settings.load(),
                                manager=pm, anamnesis=an,
                                approve=_chat_approve_cb(chat_id, p))
                # Pump the engine on a thread and emit a {"type":"ping"}
                # whenever it's been quiet for 10s: slow local models (MoE
                # with CPU experts, long thinking phases) legitimately sit
                # silent for a while, and without traffic some link in the
                # browser←uvicorn chain gives up — the user then sees a raw
                # Firefox "Error in input stream" instead of their reply.
                evq: _q.Queue = _q.Queue(maxsize=512)
                _END = object()

                def _pump():
                    try:
                        # Recent turns from THIS chat's own transcript, resent
                        # alongside the new message so Anamnesis's recency-
                        # window mechanism has real prior context to work with
                        # (see chats.recent_messages docstring — fixes losing
                        # the subject on short pronoun follow-ups like
                        # "message him again").
                        hist = chats.recent_messages(chat)
                        stream_kwargs = dict(
                            system=body.system,
                            history=hist,
                            should_stop=stop_evt.is_set,
                            poll_injections=_poll_injections,
                        )
                        if resolved_attachments:
                            # Only ever passed when there's something to say --
                            # keeps the kwarg absent for the common no-attachment
                            # case, which is also what existing tests that
                            # monkeypatch Engine.stream_events with a narrower
                            # fake signature (no attachments_note param) expect.
                            stream_kwargs["attachments_note"] = Engine._attachments_note(
                                resolved_attachments)
                            # Real image bytes for a vision-capable model (see
                            # Engine._build_user_content) -- composes with the
                            # text note above rather than replacing it, since
                            # the note's real file path is still needed for
                            # the character's own file tools regardless of
                            # whether it can also literally see the image.
                            stream_kwargs["attachments"] = resolved_attachments
                            # First audio attachment (if any) drives the
                            # audio fallback path (see Engine._audio_fallback_note)
                            # -- transcription-only today, see
                            # pleiades/tools/audio_transcribe.py's docstring.
                            _audio = next((a for a in resolved_attachments
                                          if (a.get("mime") or "").lower().startswith("audio/")), None)
                            if _audio:
                                stream_kwargs["audio_path"] = _audio.get("path")
                        for _evt in engine.stream_events(p, body.message, **stream_kwargs):
                            evq.put(_evt)
                        evq.put(_END)
                    except Exception as ex:  # re-raised on the response thread
                        evq.put(ex)

                threading.Thread(target=_pump, daemon=True).start()
                while True:
                    try:
                        item = evq.get(timeout=10)
                    except _q.Empty:
                        yield '{"type":"ping"}\n'
                        continue
                    if item is _END:
                        break
                    if isinstance(item, Exception):
                        raise item
                    evt = item
                    if evt["type"] == "reasoning":
                        # Bug: reasoning used to be forwarded live and then
                        # dropped -- never folded into the saved turn's
                        # `items`, so the block vanished the moment the chat
                        # was reloaded/refetched. Persist it like text/tool
                        # output: flush whatever text was building (so a
                        # reasoning burst always starts its own item, never
                        # gets glued onto a preceding text run), then
                        # accumulate this burst's chunks together. The next
                        # non-reasoning event flushes THIS accumulator before
                        # doing its own thing, so a second, later burst of
                        # reasoning becomes its own distinct item afterward
                        # instead of merging into the first one.
                        if text_acc:
                            items.append({"t": "text", "text": text_acc})
                            text_acc = ""
                        reasoning_acc += evt["text"]
                    elif evt["type"] == "token":
                        if reasoning_acc:
                            items.append({"t": "reasoning", "text": reasoning_acc})
                            reasoning_acc = ""
                        text_acc += evt["text"]
                    elif evt["type"] in ("tool_call", "tool_result"):
                        if reasoning_acc:
                            items.append({"t": "reasoning", "text": reasoning_acc})
                            reasoning_acc = ""
                        if text_acc:
                            items.append({"t": "text", "text": text_acc})
                            text_acc = ""
                        if evt["type"] == "tool_call":
                            # "ts" is the real wall-clock moment this call was
                            # made -- the only honest source for a per-tool
                            # "last used" display (see chats.tool_usage and
                            # the Tools right-panel section). The chat
                            # file's own "updated" field is NOT a
                            # substitute: it's overwritten by every later
                            # turn in the same chat, so it can't tell you
                            # when an EARLIER turn's tool call happened.
                            items.append({"t": "tool", "name": evt["name"],
                                          "args": evt["args"], "output": None,
                                          "ok": None, "ts": time.time()})
                        else:
                            for it in reversed(items):
                                if it["t"] == "tool" and it["output"] is None:
                                    it["output"] = evt["output"][:4000]
                                    it["ok"] = evt["ok"]
                                    break
                    elif evt["type"] == "user_injected":
                        # A mid-turn interjection the engine wove into this
                        # SAME turn (see Engine.stream_events' poll_injections)
                        # — persist it as its own item, interleaved in place,
                        # so reloading the chat and chats.recent_messages()
                        # both see it exactly where it happened.
                        if reasoning_acc:
                            items.append({"t": "reasoning", "text": reasoning_acc})
                            reasoning_acc = ""
                        if text_acc:
                            items.append({"t": "text", "text": text_acc})
                            text_acc = ""
                        items.append({"t": "user_injected", "text": evt["text"]})
                    elif evt["type"] == "done":
                        meta = {k: evt[k] for k in ("tokens", "seconds", "tps")}
                    elif evt["type"] == "stopped":
                        meta["stopped"] = True
                    yield _json.dumps(evt, ensure_ascii=False) + "\n"
                if reasoning_acc:
                    items.append({"t": "reasoning", "text": reasoning_acc})
                if text_acc:
                    items.append({"t": "text", "text": text_acc})
                try:
                    chats.append_turn(chat_id, body.message, items, meta,
                                      attachments=resolved_attachments)
                except Exception:
                    pass  # persistence must never kill the stream
            except Exception as e:
                msg = str(e) or e.__class__.__name__
                low = msg.lower()
                if any(k in low for k in ("connection", "peer closed", "disconnect",
                                          "incomplete chunked", "remoteprotocol",
                                          "socket hang", "reset by peer")):
                    msg = ("lost the model server mid-generation (was it restarted "
                           "or did it crash?) — partial reply kept · " + msg)
                yield _json.dumps({"type": "error", "error": msg}) + "\n"
            finally:
                # Edge case: an interjection can arrive in the tiny window
                # between the engine's last poll_injections() drain and the
                # turn actually ending (e.g. the user types a correction just
                # as the model produces its final answer with no more tool
                # rounds left to apply it at). Never silently drop it -- surface
                # it so the frontend can offer to resend it as a fresh message.
                leftover: list[str] = []
                while True:
                    try:
                        leftover.append(inbox.get_nowait())
                    except _q.Empty:
                        break
                for _text in leftover:
                    try:
                        yield _json.dumps({"type": "undelivered", "text": _text},
                                          ensure_ascii=False) + "\n"
                    except Exception:
                        pass
                chat_approvals.pop(chat_id, None)  # no stale approval cards
                chat_stops.pop(chat_id, None)
                chat_inboxes.pop(chat_id, None)
                lock.release()

        return StreamingResponse(gen(), media_type="application/x-ndjson")

    @app.post("/api/chats/{chat_id}/interject")
    def chats_interject(chat_id: str, body: ChatInterject) -> dict:
        """Weave a message into an ALREADY-IN-FLIGHT turn for this chat,
        instead of starting a second concurrent one.

        The engine only drains this queue at safe round/tool boundaries
        (see Engine.stream_events' poll_injections) -- never mid-stream, since
        re-prefilling context mid-token-generation is a 150-200s round trip on
        the daily-driver model (see engine.py's _UPSTREAM_TIMEOUT comment).
        409s if there's no turn in flight for this chat_id -- the caller
        should fall back to POST .../message (a brand-new turn) in that case.
        """
        inbox = chat_inboxes.get(chat_id)
        if inbox is None:
            raise HTTPException(409, "No turn in progress for this chat to interject into.")
        inbox.put(body.message)
        return {"ok": True}

    # ----------------------------- chat ------------------------------------ #
    @app.post("/api/profiles/{name}/chat")
    def profile_chat(name: str, body: ChatBody) -> StreamingResponse:
        """Stream a chat turn with a character as NDJSON lines.

        Each line is one of:
            {"type": "chunk", "text": "..."}   - assistant text
            {"type": "done"}                    - turn complete
            {"type": "error", "error": "..."}  - something went wrong
        Memory injection/storage is Anamnesis's job; we send only the new turn.
        """
        try:
            p = pm.get(name)
        except FileNotFoundError:
            raise HTTPException(404, f"No profile '{name}'")

        def gen():
            import json as _json
            try:
                from ..engine import Engine

                def _policy_only(tool, args) -> bool:
                    # No approval UI on this legacy endpoint: 'ask' fails safe.
                    # Per-character override (see profiles.Profile.exec_policy)
                    # still wins over the global default here, same as the
                    # live chat endpoint above.
                    policy = getattr(p, "exec_policy", None)
                    if not policy:
                        try:
                            from ..harness import Config as HarnessConfig
                            policy = HarnessConfig.load().exec_policy
                        except Exception:
                            policy = "ask"
                    return policy == "allow"

                engine = Engine(settings=config.Settings.load(),
                                manager=pm, anamnesis=an, approve=_policy_only)
                for chunk in engine.stream(p, body.message, system=body.system):
                    if chunk:
                        yield _json.dumps({"type": "chunk", "text": chunk}) + "\n"
                yield _json.dumps({"type": "done"}) + "\n"
            except Exception as e:  # surface anything to the UI, never log secrets
                yield _json.dumps({"type": "error", "error": str(e)}) + "\n"

        return StreamingResponse(gen(), media_type="application/x-ndjson")

    # ----------------------------- agent work ------------------------------- #
    # Background jobs for the workspace harness ("pleiades work" in the UI).
    work_jobs: dict[str, dict] = {}            # id -> serializable job state
    work_ctl: dict[str, dict] = {}             # id -> {event, answer} (not serialized)

    def _job_view(job: dict, since: int = 0) -> dict:
        d = {k: v for k, v in job.items() if k != "events"}
        d["events"] = job["events"][since:]
        d["event_count"] = len(job["events"])
        return d

    @app.get("/api/work")
    def work_list() -> dict:
        return {"jobs": [
            {"id": j["id"], "task": j["task"], "character": j["character"],
             "status": j["status"], "started": j["started"],
             "pending_approval": j["pending_approval"]}
            for j in sorted(work_jobs.values(), key=lambda x: -x["started"])
        ]}

    def _launch_work_job(task_text: str, character: str = "", tier: str = "",
                         policy: str = "") -> str:
        """Start a work job and return its id. The shared core behind both
        POST /api/work (a human/UI-triggered run) and the scheduler (a
        cron/one-time-triggered run) -- one job-execution code path, two
        triggers. Caller validates task_text is non-empty."""
        import uuid
        job_id = uuid.uuid4().hex[:12]
        job = {"id": job_id, "task": task_text, "character": character,
               "tier": tier, "policy": policy, "status": "running",
               "started": time.time(), "finished": None, "events": [],
               "result": "", "error": "", "steps": 0,
               "pending_approval": None, "cancel": False}
        ctl = {"event": threading.Event(), "answer": {"ok": False}}
        work_jobs[job_id] = job
        work_ctl[job_id] = ctl

        def emit(kind: str, payload: Any) -> None:
            if job["cancel"]:
                raise RuntimeError("cancelled by user")
            if kind == "tool_call":
                job["events"].append({"kind": "tool_call", "ts": time.time(),
                                      "name": getattr(payload, "name", str(payload))})
            elif kind == "tool_result":
                _call, out, is_err = payload
                job["events"].append({"kind": "tool_result", "ts": time.time(),
                                      "name": getattr(_call, "name", ""),
                                      "ok": not is_err,
                                      "output": (out or "")[:2000]})
            elif kind == "text":
                job["events"].append({"kind": "text", "ts": time.time(),
                                      "text": str(payload)})

        def runner() -> None:
            try:
                from ..harness import Agent, Config as HarnessConfig
                from ..harness import builtins as _builtins      # noqa: F401
                from ..harness import identity as _identity      # noqa: F401
                from ..harness.subagent import bind_context
                from ..harness.builtins.memory import bind_memory
                from ..harness.builtins.tasks import bind_job
                from ..harness.anamnesis import Anamnesis as WorkingMemory

                bind_job(job_id)  # scope create_task/update_task/list_tasks to this job
                cfg = HarnessConfig.load()
                if policy:
                    cfg.exec_policy = policy
                elif character:
                    # No explicit job-level policy: fall back to the
                    # character's own exec_policy override (composer's
                    # Approve/Ask dropdown) before the process-wide default.
                    try:
                        char_policy = getattr(pm.get(character), "exec_policy", None)
                        if char_policy:
                            cfg.exec_policy = char_policy
                    except Exception:
                        pass
                if character:
                    _identity.bind_character(character, cfg=cfg)
                bind_memory(WorkingMemory.from_config(cfg))

                def approve(tool, args) -> bool:
                    """Permission gate; 'ask' surfaces an approval card in the UI."""
                    p = cfg.exec_policy
                    if p == "allow":
                        return True
                    if p == "deny":
                        return False
                    import json as _json
                    job["pending_approval"] = {
                        "tool": getattr(tool, "name", str(tool)),
                        "args": _json.dumps(args)[:300],
                    }
                    ctl["event"].clear()
                    answered = ctl["event"].wait(timeout=600)   # 10 min, then deny
                    job["pending_approval"] = None
                    return bool(ctl["answer"]["ok"]) if answered else False

                agent = Agent(cfg, tier_name=tier or None, approve=approve)
                bind_context(cfg, depth=0, approve=agent.approve)
                result = agent.run(task_text, on_event=emit)
                job["result"] = getattr(result, "answer", str(result))
                job["steps"] = getattr(result, "steps", 0)
                job["status"] = "done"
            except Exception as e:
                job["status"] = "cancelled" if job["cancel"] else "error"
                job["error"] = "" if job["cancel"] else str(e)
            finally:
                job["finished"] = time.time()

        threading.Thread(target=runner, daemon=True).start()
        return job_id

    # Scheduler fires jobs through the exact same path a human-triggered
    # POST /api/work does -- see the startup hook below for where this is
    # actually wired in (scheduler.set_job_launcher).
    def _scheduled_job_launcher(task_text: str, character: str, tier: str,
                                policy: str) -> str:
        return _launch_work_job(task_text, character, tier, policy)

    @app.post("/api/work")
    def work_start(body: WorkStart) -> dict:
        task_text = (body.task or "").strip()
        if not task_text:
            raise HTTPException(400, "Task must not be empty.")
        job_id = _launch_work_job(task_text, body.character, body.tier, body.policy)
        return {"id": job_id}

    @app.get("/api/work/{job_id}")
    def work_get(job_id: str, since: int = 0) -> dict:
        job = work_jobs.get(job_id)
        if not job:
            raise HTTPException(404, "No such job.")
        return _job_view(job, since)

    @app.get("/api/work/{job_id}/tasks")
    def work_tasks(job_id: str) -> dict:
        """Live task list for a work job -- the desktop app's Progress panel
        polls this. Backed by pleiades.harness.builtins.tasks, the same
        create_task/update_task/list_tasks/delete_task tools a character
        calls mid-job; 404 only if the job itself doesn't exist (a job that
        never called create_task still returns an empty list, same as
        work_get does for a job with no events yet)."""
        from ..harness.builtins.tasks import get_tasks
        if job_id not in work_jobs:
            raise HTTPException(404, "No such job.")
        return {"tasks": get_tasks(job_id)}

    @app.post("/api/work/{job_id}/approve")
    def work_approve(job_id: str, body: WorkApprove) -> dict:
        job, ctl = work_jobs.get(job_id), work_ctl.get(job_id)
        if not job or not ctl:
            raise HTTPException(404, "No such job.")
        if not job["pending_approval"]:
            raise HTTPException(409, "Nothing awaiting approval.")
        ctl["answer"]["ok"] = bool(body.approve)
        ctl["event"].set()
        return {"ok": True}

    @app.post("/api/work/{job_id}/cancel")
    def work_cancel(job_id: str) -> dict:
        job, ctl = work_jobs.get(job_id), work_ctl.get(job_id)
        if not job:
            raise HTTPException(404, "No such job.")
        job["cancel"] = True
        if ctl and job["pending_approval"]:        # unblock a waiting approval
            ctl["answer"]["ok"] = False
            ctl["event"].set()
        return {"ok": True}

    # ----------------------------- agents (background sub-agents) ----------- #
    # Ephemeral, depth-capped, concurrency-capped agents spawned from chat via
    # the spawn_agents tool (pleiades/tools/agents_tool.py) -- backed by the
    # single process-wide AgentManager (pleiades/agents.py, mirrors the
    # work_jobs pattern above but is its own module since it's also imported
    # directly by the chat-belt tool, not just this webui layer).
    from ..agents import agent_view, get_manager

    @app.get("/api/agents")
    def agents_list() -> dict:
        return {"agents": get_manager().list()}

    @app.get("/api/agents/{agent_id}")
    def agents_get(agent_id: str, since: int = 0) -> dict:
        rec = get_manager().get(agent_id)
        if not rec:
            raise HTTPException(404, "No such agent.")
        return agent_view(rec, since)

    @app.post("/api/agents/{agent_id}/stop")
    def agents_stop(agent_id: str) -> dict:
        if not get_manager().stop(agent_id):
            raise HTTPException(404, "No such agent.")
        return {"ok": True}

    @app.post("/api/agents/settings")
    def agents_settings(body: AgentsSettings) -> dict:
        try:
            p = pm.get(body.character)
        except FileNotFoundError:
            raise HTTPException(404, f"No profile '{body.character}'")
        if body.enabled is not None:
            p.agents_enabled = bool(body.enabled)
        if body.model is not None:
            p.agents_model = body.model
        pm._save(p)  # noqa: SLF001
        return _profile_view(p)

    # ----------------------------- scheduled tasks --------------------------- #
    # Same manager the schedule harness tools (pleiades/harness/builtins/
    # schedule.py) use -- a character's own create_scheduled_task and this
    # desktop-app CRUD surface are two front doors onto one JSON store
    # (~/.pleiades/scheduled_tasks.json), never two sources of truth.
    stm = ScheduledTaskManager()

    @app.get("/api/scheduled-tasks")
    def scheduled_tasks_list() -> dict:
        return {"tasks": stm.list()}

    @app.post("/api/scheduled-tasks")
    def scheduled_tasks_create(body: ScheduledTaskCreate) -> dict:
        try:
            return stm.create(body.task, character=body.character, cron=body.cron,
                              fire_at=body.fire_at, tier=body.tier, policy=body.policy)
        except ScheduleError as e:
            raise HTTPException(400, str(e))

    @app.patch("/api/scheduled-tasks/{task_id}")
    def scheduled_tasks_update(task_id: str, body: ScheduledTaskUpdate) -> dict:
        patch = {k: v for k, v in body.model_dump().items() if v is not None}
        try:
            return stm.update(task_id, **patch)
        except ScheduleError as e:
            raise HTTPException(400, str(e))

    @app.delete("/api/scheduled-tasks/{task_id}")
    def scheduled_tasks_delete(task_id: str) -> dict:
        ok = stm.delete(task_id)
        if not ok:
            raise HTTPException(404, "No such scheduled task.")
        return {"ok": True}

    @app.post("/api/scheduled-tasks/{task_id}/run")
    def scheduled_tasks_run_now(task_id: str) -> dict:
        """Manual "Run now" trigger (desktop UI button) -- fires the task
        immediately through the same job-launcher the background tick thread
        uses, out of band from its schedule. A one-time task still disables
        itself afterward; a recurring task's next_run is untouched by this
        (still computed from its cron, not shifted by the manual run)."""
        try:
            return stm.run_now(task_id)
        except ScheduleError as e:
            raise HTTPException(404, str(e))

    # ----------------------------- browser view (Playwright/Chromium) ------- #
    # Live, embeddable browser-use view for the desktop app's right panel.
    # Separate capability from harness/builtins/browser.py's Camoufox tool —
    # see browser_view.py's module docstring for why this is Chromium-only.
    @app.post("/api/profiles/{name}/browser-view/start")
    async def browser_view_start(name: str, body: BrowserViewStart) -> dict:
        try:
            pm.get(name)
        except FileNotFoundError:
            raise HTTPException(404, f"No profile '{name}'")
        from .browser_view import get_session
        session = get_session(name)
        try:
            await session.start(body.url)
        except Exception as e:
            raise HTTPException(502, str(e))
        return session.snapshot()

    @app.post("/api/profiles/{name}/browser-view/stop")
    async def browser_view_stop(name: str) -> dict:
        from .browser_view import get_session
        session = get_session(name)
        await session.stop()
        return session.snapshot()

    @app.get("/api/profiles/{name}/browser-view/status")
    def browser_view_status(name: str) -> dict:
        from .browser_view import get_session
        return get_session(name).snapshot()

    @app.post("/api/profiles/{name}/browser-view/navigate")
    async def browser_view_navigate(name: str, body: BrowserViewNavigate) -> dict:
        from .browser_view import get_session
        session = get_session(name)
        try:
            url = await session.navigate(body.url)
        except Exception as e:
            raise HTTPException(502, str(e))
        return {"ok": True, "url": url}

    @app.post("/api/profiles/{name}/browser-view/open-separate")
    async def browser_view_open_separate(name: str) -> dict:
        """Explicit, human-triggered escape hatch: switch this character's
        headless panel session to a real headed OS window (owner brief item
        5 -- never a real window by default, only on this explicit action).
        See BrowserViewSession.open_separate()'s docstring for why this
        re-homes the SAME profile/session headed rather than opening a
        separate plain browser."""
        from .browser_view import get_session
        session = get_session(name)
        try:
            await session.open_separate()
        except Exception as e:
            raise HTTPException(502, str(e))
        return session.snapshot()

    @app.post("/api/profiles/{name}/browser-view/resize")
    async def browser_view_resize(name: str, body: BrowserViewResize) -> dict:
        """Used by the panel's expand/fullscreen view to request a bigger
        rendered viewport (rather than just upscaling a blurry small one)."""
        from .browser_view import get_session
        session = get_session(name)
        try:
            await session.resize(body.width, body.height)
        except Exception as e:
            raise HTTPException(502, str(e))
        return session.snapshot()

    @app.websocket("/api/profiles/{name}/browser-view/ws")
    async def browser_view_ws(websocket: WebSocket, name: str) -> None:
        """Frames out (binary JPEG messages), input events in (JSON: see
        BrowserViewSession.dispatch_input for the event shape). One
        WebSocket per connected panel; multiple panels may watch the same
        character's session concurrently (each gets its own drop-old
        frame queue)."""
        from .browser_view import get_session
        await websocket.accept()
        session = get_session(name)
        await websocket.send_json({"type": "status", **session.snapshot()})
        queue = session.subscribe()

        async def sender() -> None:
            try:
                while True:
                    frame = await queue.get()
                    if frame is None:  # session torn down
                        await websocket.send_json({"type": "status", **session.snapshot()})
                        break
                    await websocket.send_bytes(frame)
            except Exception:
                pass

        sender_task = asyncio.create_task(sender())
        try:
            while True:
                msg = await websocket.receive_json()
                if msg.get("kind") == "ping":
                    continue
                await session.dispatch_input(msg)
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            sender_task.cancel()
            session.unsubscribe(queue)

    @app.on_event("startup")
    async def _start_scheduler() -> None:
        """Wire the scheduler to fire jobs through the same path POST
        /api/work uses, then start its once-a-minute background thread.
        Idempotent (start_scheduler_thread no-ops on a second call), so this
        is safe even though create_app() can in principle run more than
        once in a process (tests do this)."""
        scheduler.set_job_launcher(_scheduled_job_launcher)
        scheduler.start_scheduler_thread()

    @app.on_event("startup")
    async def _ensure_anamnesis_running() -> None:
        """Best-effort: bring up the vendored Anamnesis daemon if nothing is
        already answering at Settings.anamnesis_control_url. Phase 1 of
        docs/specs/2026-07-25-anamnesis-vendoring-and-installer-plan.md --
        see pleiades/anamnesis_runtime.py's module docstring for the full
        story (vendored source, ~/.anamnesis untouched, systemd migration).

        Safe on a pre-phase-1 machine: is_running() checks the real health
        endpoint, not just its own pid file, so this correctly no-ops against
        an existing systemd-managed instance already serving :9000 instead
        of spawning a duplicate -- nothing changes for Minty today until
        `pleiades anamnesis adopt` is actually run.

        Never blocks startup or crashes the webui on failure (no Node
        installed, anamnesis/ not vendored on this machine yet, port
        conflict, etc.) -- Anamnesis has always been optional infrastructure
        pleiades/anamnesis.py's client tolerates being unreachable.
        """
        import sys
        from ..anamnesis_runtime import AnamnesisDaemon, AnamnesisRuntimeError
        try:
            await asyncio.to_thread(AnamnesisDaemon().start, wait=False)
        except AnamnesisRuntimeError as e:
            print(f"[pleiades] Anamnesis daemon not started (non-fatal): {e}", file=sys.stderr)
        except Exception as e:
            print(f"[pleiades] unexpected error starting Anamnesis daemon (non-fatal): {e}", file=sys.stderr)

    @app.on_event("startup")
    async def _bind_browser_view_loop() -> None:
        """Capture this process's running asyncio loop so tools/browser.py
        (the chat-path model tool, which runs synchronously on a per-turn
        worker thread -- see engine.py) can bridge synchronous Tool.run()
        calls into browser_view.py's async Playwright sessions. Only ever
        set inside this webui/desktop backend process -- the Discord bot and
        `pleiades work` jobs never call create_app()/run this, so
        browser_view.main_loop_available() stays False there and
        tools/browser.py falls back to the harness's Camoufox browser
        unchanged."""
        from .browser_view import set_main_loop
        set_main_loop(asyncio.get_running_loop())

    @app.on_event("shutdown")
    async def _shutdown_browser_views() -> None:
        """Never leave an orphaned headed or headless Chromium behind a
        normal app quit (Electron sends SIGTERM -> uvicorn's default signal
        handling runs this before the process exits)."""
        from .browser_view import shutdown_all
        await shutdown_all()

    # ----------------------------- search service --------------------------- #
    search_state: dict[str, Any] = {}

    @app.post("/api/search/{direction}")
    def search_ctl(direction: str) -> dict:
        if direction not in ("up", "down"):
            raise HTTPException(400, "Direction must be 'up' or 'down'.")
        if search_state.get("status") == "working":
            raise HTTPException(409, "A search-service operation is in progress.")
        from ..searxng_runtime import SearxngDaemon, SearxngRuntimeError
        daemon = SearxngDaemon()

        def runner() -> None:
            search_state.update(status="working", direction=direction, error="")
            try:
                if direction == "up":
                    daemon.start()
                else:
                    daemon.stop()
                search_state.update(status="done")
            except SearxngRuntimeError as e:
                search_state.update(status="error", error=str(e))

        threading.Thread(target=runner, daemon=True).start()
        return {"ok": True}

    @app.get("/api/search/status")
    def search_status() -> dict:
        return search_state or {"status": "idle"}

    # ----------------------------- self-update ------------------------------ #
    update_state: dict[str, Any] = {}

    @app.get("/api/update/check")
    def update_check() -> dict:
        from ..update import check
        c = check()
        return {"supported": c.supported, "installed": c.installed,
                "latest": c.latest, "behind": c.behind, "error": c.error}

    @app.post("/api/update")
    def update_run() -> dict:
        if update_state.get("status") in ("updating", "restarting"):
            raise HTTPException(409, "Update already in progress.")
        from ..update import run_update

        def work() -> None:
            update_state.update(status="updating", log=[], error="")
            try:
                changed = run_update(log=lambda m: update_state["log"].append(m))
            except Exception as e:
                update_state.update(status="error", error=str(e))
                return
            if changed:
                update_state.update(status="restarting")
                time.sleep(0.8)  # let the status response flush to the browser
                _restart_server()
            else:
                update_state.update(status="up-to-date")

        threading.Thread(target=work, daemon=True).start()
        return {"ok": True}

    @app.get("/api/update/status")
    def update_status() -> dict:
        return update_state or {"status": "idle"}

    # ----------------------------- static ---------------------------------- #
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index() -> Any:
        # The new workstation UI is now the default; the old panel lives at /classic.
        nx = STATIC_DIR / "next.html"
        if nx.is_file():
            return FileResponse(str(nx))
        idx = STATIC_DIR / "index.html"
        if idx.is_file():
            return FileResponse(str(idx))
        return JSONResponse({"error": "frontend not built"}, status_code=500)

    @app.get("/next")
    def index_next() -> Any:
        return index()

    @app.get("/classic")
    def index_classic() -> Any:
        idx = STATIC_DIR / "index.html"
        if idx.is_file():
            return FileResponse(str(idx))
        return JSONResponse({"error": "classic UI not present"}, status_code=404)

    # ── Anamnesis control ──────────────────────────────────────────────────────
    @app.post("/api/anamnesis/{action}")
    async def anamnesis_control(action: str):
        import psutil
        import subprocess
        import signal as _signal
        if action not in ("start", "stop"):
            raise HTTPException(status_code=400, detail="action must be start or stop")
        procs = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info.get('cmdline') or [])
                if 'node' in (proc.info.get('name') or '').lower() and 'anamnesis' in cmdline:
                    procs.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        if action == "stop":
            for proc in procs:
                try:
                    proc.send_signal(_signal.SIGTERM)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return {"ok": True, "action": "stop", "killed": len(procs)}
        daemon = os.path.expanduser("~/.local/share/anamnesis/src/daemon.js")
        if procs:
            return {"ok": True, "action": "start", "status": "already_running"}
        if not os.path.exists(daemon):
            raise HTTPException(status_code=404, detail=f"daemon not found: {daemon}")
        subprocess.Popen(["node", daemon], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
        return {"ok": True, "action": "start", "status": "starting"}

    # ── Rules & guidelines ─────────────────────────────────────────────────────
    import json as _json
    import uuid as _uuid
    _RULES_FILE = os.path.expanduser("~/.pleiades/rules.json")
    def _load_rules():
        try:
            with open(_RULES_FILE) as _f:
                return _json.load(_f)
        except FileNotFoundError:
            return []
        except Exception:
            return []
    def _save_rules(rules):
        os.makedirs(os.path.dirname(_RULES_FILE), exist_ok=True)
        with open(_RULES_FILE, 'w') as _f:
            _json.dump(rules, _f, indent=2)
    @app.get("/api/rules")
    async def get_rules():
        return {"rules": _load_rules()}
    @app.post("/api/rules")
    async def create_rule(body: dict):
        rules = _load_rules()
        rule = {"id": _uuid.uuid4().hex[:8], "text": body.get("text",""),
                "enabled": True, "pending": bool(body.get("pending", False))}
        rules.append(rule)
        _save_rules(rules)
        return rule
    @app.put("/api/rules/{rule_id}")
    async def update_rule(rule_id: str, body: dict):
        rules = _load_rules()
        for r in rules:
            if r["id"] == rule_id:
                for k in ("text","enabled","pending"):
                    if k in body:
                        r[k] = body[k]
                _save_rules(rules)
                return r
        raise HTTPException(status_code=404, detail="rule not found")
    @app.delete("/api/rules/{rule_id}")
    async def delete_rule_ep(rule_id: str):
        _save_rules([r for r in _load_rules() if r["id"] != rule_id])
        return {"ok": True}


    return app


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def _free_port(preferred: int = 8750) -> int:
    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return preferred


_RUN_ARGS: dict[str, Any] = {}


def _restart_server() -> None:
    """Replace this process with a fresh `python -m pleiades.webui` (post-update)."""
    args = [sys.executable, "-m", "pleiades.webui",
            "--host", str(_RUN_ARGS.get("host", "127.0.0.1")),
            "--port", str(_RUN_ARGS.get("port", 8750)), "--no-browser"]
    if os.name == "posix":
        os.execv(sys.executable, args)
    else:  # Windows: detach a child, then exit so it can take the port
        subprocess.Popen(args, creationflags=0x00000008 | 0x00000200)
        os._exit(0)


def _wait_port_free(host: str, port: int, timeout: float = 8.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((host, port)) != 0:
                return
        time.sleep(0.2)


def run(host: str = "127.0.0.1", port: Optional[int] = None,
        open_browser: bool = True) -> None:
    """Launch the control panel with uvicorn and (optionally) open a browser.

    Cross-platform: ``webbrowser.open`` works on Linux and Windows alike.
    """
    import uvicorn

    if port:
        _wait_port_free(host, port)  # restart case: predecessor may still hold it
    port = port or _free_port()
    _RUN_ARGS.update(host=host, port=port)
    url = f"http://{host}:{port}/"

    if open_browser:
        def _open() -> None:
            time.sleep(1.0)
            try:
                webbrowser.open(url)
            except Exception:
                pass
        threading.Thread(target=_open, daemon=True).start()

    print(f"\n  Pleiades control panel -> {url}\n  (Ctrl-C to stop)\n")
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")
