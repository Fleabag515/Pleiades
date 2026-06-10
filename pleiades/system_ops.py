"""Model-facing system operations — ONE implementation for every surface.

Pleiades' rule: anything the user can do, the model can do. These functions
power the `hardware`, `models`, `profile`, and `characters` tools in both the
chat ToolBelt (pleiades.tools.system_tools) and the agent harness
(pleiades.harness.builtins.inference). They return plain strings the model can
read, never raise, and never print secrets.
"""

from __future__ import annotations

from typing import Optional

from . import config, hardware
from .profiles import Profile, ProfileManager


# --------------------------------------------------------------------------- #
# hardware
# --------------------------------------------------------------------------- #
def hardware_report() -> str:
    """Detected hardware + the offload plan for every registered model."""
    try:
        det = hardware.detect()
        lines = [det.describe()]
        from .models import ModelManager
        for m in ModelManager().list():
            meta = hardware.read_gguf_meta(m["path"])
            p = hardware.plan(meta, int(m.get("n_ctx", 8192)), det)
            running = " (running)" if m.get("running") else ""
            lines.append(f"\nmodel '{m['name']}'{running}: n_gpu_layers={p.n_gpu_layers}"
                         f" — {p.reason}")
        s = config.Settings.load()
        if s.model_path:
            meta = hardware.read_gguf_meta(s.model_path)
            p = hardware.plan(meta, s.n_ctx, det)
            lines.append(f"\ndefault engine ({s.model_path}): "
                         f"n_gpu_layers={p.n_gpu_layers} — {p.reason}")
        return "\n".join(lines)
    except Exception as e:
        return f"[hardware error] {e}"


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def models_list() -> str:
    from .models import ModelManager
    models = ModelManager().list()
    if not models:
        return ("No models registered. Search with models action=search, then "
                "action=fetch to download the best quant for this machine.")
    out = []
    for m in models:
        state = "running" if m["running"] else "stopped"
        out.append(f"- {m['name']}: {state}, port {m['port']}, "
                   f"gpu_layers {m['n_gpu_layers']}, {m['path']}")
    return "\n".join(out)


def models_start(name: str) -> str:
    from .models import ModelError, ModelManager
    try:
        url = ModelManager().start(name, wait=True)
        return f"Model '{name}' is serving at {url}."
    except ModelError as e:
        return f"[models error] {e}"


def models_stop(name: str) -> str:
    from .models import ModelManager
    ok = ModelManager().stop(name)
    return f"Stopped '{name}'." if ok else f"'{name}' was not running."


def models_search(query: str, limit: int = 8) -> str:
    from .fetch import FetchError, search_gguf_repos
    try:
        rows = search_gguf_repos(query, limit=limit)
    except FetchError as e:
        return f"[models error] {e}"
    if not rows:
        return f"No GGUF repos found for '{query}'."
    return "\n".join(f"- {r['id']} ({r['downloads']:,} downloads)" for r in rows) + \
        "\nDownload one with: models action=fetch repo=<id>"


def models_fetch(repo: str, name: str = "", quant: str = "") -> str:
    from .fetch import FetchError, fetch_model
    try:
        entry = fetch_model(repo, name=name, quant=quant)
    except FetchError as e:
        return f"[models error] {e}"
    return (f"Downloaded and registered '{entry['name']}' ({entry['chosen']}, "
            f"{entry['size'] / 1024 ** 3:.1f} GiB). {entry['why']}. "
            f"Start it with models action=start, or assign it with action=assign.")


def models_assign(model: str, character: str) -> str:
    from .models import ModelManager
    if not ModelManager().get(model):
        return f"[models error] no registered model '{model}'."
    try:
        ProfileManager().assign_model(character, model)
    except FileNotFoundError as e:
        return f"[models error] {e}"
    return f"'{character}' now uses model '{model}' (auto-starts on next chat)."


def models_op(action: str, *, name: str = "", repo: str = "", query: str = "",
              quant: str = "", character: str = "") -> str:
    """Dispatch for the model-facing `models` tool."""
    action = action.lower().strip()
    if action == "list":
        return models_list()
    if action == "start":
        return models_start(name) if name else "[models error] 'start' needs 'name'."
    if action == "stop":
        return models_stop(name) if name else "[models error] 'stop' needs 'name'."
    if action == "search":
        return models_search(query) if query else "[models error] 'search' needs 'query'."
    if action == "fetch":
        return models_fetch(repo, name=name, quant=quant) if repo else \
            "[models error] 'fetch' needs 'repo' (a Hugging Face repo id)."
    if action == "assign":
        if not (name and character):
            return "[models error] 'assign' needs 'name' (model) and 'character'."
        return models_assign(name, character)
    return f"[models error] unknown action '{action}'. " \
           "Use list, search, fetch, start, stop, or assign."


# --------------------------------------------------------------------------- #
# profile (self-configuration)
# --------------------------------------------------------------------------- #
_PROFILE_FIELDS = ("email_address", "imap_host", "imap_port", "smtp_host",
                   "smtp_port", "model")


def profile_show(profile: Profile) -> str:
    d = profile.to_json()
    lines = [f"profile '{profile.name}':"]
    lines += [f"  {k} = {d[k]!r}" for k in _PROFILE_FIELDS]
    lines.append(f"  discord_enabled = {profile.discord_enabled}")
    lines.append(f"  email configured: {profile.has_email}")
    return "\n".join(lines)


def profile_op(profile: Optional[Profile], action: str, *, field: str = "",
               value: str = "") -> str:
    """`profile` tool dispatch: get / set on the character's own profile."""
    if profile is None:
        return "[profile error] no character is bound to this session."
    action = action.lower().strip()
    if action == "get":
        return profile_show(profile)
    if action == "set":
        if field not in _PROFILE_FIELDS:
            return (f"[profile error] unknown field '{field}'. "
                    f"Editable: {', '.join(_PROFILE_FIELDS)}")
        val: "str | int" = value
        if field.endswith("_port"):
            try:
                val = int(value)
            except ValueError:
                return f"[profile error] {field} must be an integer."
        if field == "model":
            from .models import ModelManager
            if value and not ModelManager().get(value):
                return f"[profile error] no registered model '{value}'."
        setattr(profile, field, val)
        ProfileManager()._save(profile)  # noqa: SLF001 — same package
        return f"Set {field} = {val!r} on '{profile.name}'."
    return f"[profile error] unknown action '{action}'. Use get or set."


# --------------------------------------------------------------------------- #
# characters
# --------------------------------------------------------------------------- #
def characters_op(action: str, *, name: str = "", confirm: str = "") -> str:
    """`characters` tool dispatch: list / create / adopt / delete."""
    pm = ProfileManager()
    action = action.lower().strip()
    if action == "list":
        profiles = pm.list()
        orphans = pm.orphan_characters()
        lines = [f"- {p.name}" + (f" (model: {p.model})" if p.model else "")
                 for p in profiles] or ["(no characters yet)"]
        if orphans:
            lines.append("adoptable Anamnesis characters: " + ", ".join(orphans))
        return "\n".join(lines)
    if not name:
        return f"[characters error] '{action}' needs 'name'."
    try:
        if action == "create":
            config.validate_name(name)
            pm.create(name)
            return f"Created character '{name}' (memory, vault, browser dir)."
        if action == "adopt":
            pm.adopt(name)
            return f"Adopted existing Anamnesis character '{name}'."
        if action == "delete":
            expected = f"delete {name}"
            if confirm != expected:
                return (f"[characters] deleting '{name}' is permanent (memory, vault, "
                        f"browser). To proceed, call again with confirm='{expected}'.")
            pm.delete(name)
            return f"Deleted character '{name}'."
    except Exception as e:
        return f"[characters error] {e}"
    return f"[characters error] unknown action '{action}'. " \
           "Use list, create, adopt, or delete."
