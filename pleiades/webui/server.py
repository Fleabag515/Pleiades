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

import os
import re
import socket
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

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import config
from ..profiles import ProfileManager, Profile
from ..models import ModelManager, ModelError
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
    n_ctx: int = 8192
    n_gpu_layers: "int | str" = "auto"   # "auto" = planned from hardware at launch
    chat_format: str = ""


class ModelUpdate(BaseModel):
    path: Optional[str] = None
    n_ctx: Optional[int] = None
    n_gpu_layers: "Optional[int | str]" = None
    chat_format: Optional[str] = None


class ModelFetch(BaseModel):
    repo: str
    name: str = ""
    quant: str = ""   # blank = auto-pick for this machine


class AssignModel(BaseModel):
    model: str  # "" clears the assignment (back to default engine)


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
    mm._save(reg)  # noqa: SLF001
    entry = dict(entry)
    entry["running"] = mm.is_running(name)
    return entry


def _profile_view(p: Profile) -> dict:
    d = p.to_json()
    d["has_email"] = p.has_email
    return d


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #
def create_app() -> FastAPI:
    app = FastAPI(title="Pleiades Control Panel", version="2.0.0")

    settings = config.Settings.load()
    pm = ProfileManager(settings)
    mm = ModelManager()
    an = Anamnesis(settings.anamnesis_control_url)

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
        }
        updates: dict[str, str] = {}
        for field_name, env_name in env_map.items():
            val = getattr(body, field_name)
            if val is not None:
                updates[env_name] = str(val)
        if updates:
            _write_env(updates)
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

    @app.put("/api/profiles/{name}")
    def update_profile(name: str, body: ProfileUpdate) -> dict:
        try:
            p = pm.get(name)
        except FileNotFoundError:
            raise HTTPException(404, f"No profile '{name}'")
        for f in ("email_address", "imap_host", "imap_port", "smtp_host",
                  "smtp_port", "persona_source", "discord_enabled",
                  "discord_require_mention", "discord_respond_to_bots",
                  "discord_allowed_channels"):
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

    # ----------------------------- hardware -------------------------------- #
    @app.get("/api/hardware")
    def hardware_info() -> dict:
        from .. import hardware
        det = hardware.detect()
        plans = []
        for m in mm.list():
            meta = hardware.read_gguf_meta(m["path"])
            p = hardware.plan(meta, int(m.get("n_ctx", 8192)), det)
            plans.append({"model": m["name"], "n_gpu_layers": p.n_gpu_layers,
                          "n_layers": p.n_layers, "reason": p.reason,
                          "fits_fully": p.fits_fully})
        return {
            "gpus": [{"vendor": g.vendor, "name": g.name,
                      "vram_total": g.vram_total, "vram_free": g.vram_free}
                     for g in det.gpus],
            "ram_total": det.ram_total, "ram_available": det.ram_available,
            "cpu_threads": det.cpu_threads, "unified_memory": det.unified_memory,
            "summary": det.describe(), "plans": plans,
        }

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
                           n_gpu_layers=body.n_gpu_layers, chat_format=body.chat_format)
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

    @app.post("/api/models/{name}/stop")
    def models_stop(name: str) -> dict:
        return {"ok": mm.stop(name)}

    @app.get("/api/email/presets")
    def email_presets() -> dict:
        return config.EMAIL_PRESETS

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
                engine = Engine(settings=config.Settings.load(),
                                manager=pm, anamnesis=an)
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

    @app.post("/api/work")
    def work_start(body: WorkStart) -> dict:
        import uuid
        task_text = (body.task or "").strip()
        if not task_text:
            raise HTTPException(400, "Task must not be empty.")
        job_id = uuid.uuid4().hex[:12]
        job = {"id": job_id, "task": task_text, "character": body.character,
               "tier": body.tier, "policy": body.policy, "status": "running",
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
                from ..harness.anamnesis import Anamnesis as WorkingMemory

                cfg = HarnessConfig.load()
                if body.policy:
                    cfg.exec_policy = body.policy
                if body.character:
                    _identity.bind_character(body.character, cfg=cfg)
                bind_memory(WorkingMemory.from_config(cfg))

                def approve(tool, args) -> bool:
                    """Permission gate; 'ask' surfaces an approval card in the UI."""
                    policy = cfg.exec_policy
                    if policy == "allow":
                        return True
                    if policy == "deny":
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

                agent = Agent(cfg, tier_name=body.tier or None, approve=approve)
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
        return {"id": job_id}

    @app.get("/api/work/{job_id}")
    def work_get(job_id: str, since: int = 0) -> dict:
        job = work_jobs.get(job_id)
        if not job:
            raise HTTPException(404, "No such job.")
        return _job_view(job, since)

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

    # ----------------------------- search service --------------------------- #
    search_state: dict[str, Any] = {}

    @app.post("/api/search/{direction}")
    def search_ctl(direction: str) -> dict:
        if direction not in ("up", "down"):
            raise HTTPException(400, "Direction must be 'up' or 'down'.")
        if search_state.get("status") == "working":
            raise HTTPException(409, "A search-service operation is in progress.")
        compose = Path(__file__).resolve().parents[2] / "docker-compose.yml"
        if not compose.is_file():
            raise HTTPException(400, f"docker-compose.yml not found at {compose}")
        args = ["docker", "compose", "-f", str(compose)]
        args += ["up", "-d"] if direction == "up" else ["down"]

        def runner() -> None:
            search_state.update(status="working", direction=direction, error="")
            try:
                r = subprocess.run(args, capture_output=True, text=True, timeout=600)
                if r.returncode != 0:
                    search_state.update(status="error",
                                        error=(r.stderr or r.stdout or "")[-800:])
                else:
                    search_state.update(status="done")
            except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
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
        idx = STATIC_DIR / "index.html"
        if idx.is_file():
            return FileResponse(str(idx))
        return JSONResponse({"error": "frontend not built"}, status_code=500)

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

    print(f"\n  Pleiades control panel → {url}\n  (Ctrl-C to stop)\n")
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")
