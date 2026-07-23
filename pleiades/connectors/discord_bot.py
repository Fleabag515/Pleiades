"""Host a Pleiades character as a Discord bot.

One bot process per character. The token comes from the character's vault
(`discord.token`). On a DM or a mention, the message is run through the engine and the
reply is sent back. Requires the [discord] extra and the Message Content intent enabled
in the Discord Developer Portal.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

from ..engine import Engine
from ..profiles import Profile


def _safe_service_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name.lower())


def _systemctl_env() -> dict:
    # webui/CLI are often started from a process with no session D-Bus —
    # same fix as _discord_service_linux() in cli.py. POSIX-only (os.getuid),
    # only ever called from the Linux branch below.
    env = dict(os.environ)
    uid = os.getuid()
    env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{uid}")
    env.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path=/run/user/{uid}/bus")
    return env


def _service_state_linux(name: str) -> tuple[str, str, "int | None", "str | None"]:
    unit = f"pleiades-discord-{_safe_service_name(name)}.service"
    state, substate, restarts, last_log = "unknown", "unknown", None, None
    try:
        r = subprocess.run(
            ["systemctl", "--user", "show", unit,
             "-p", "ActiveState", "-p", "SubState", "-p", "NRestarts"],
            capture_output=True, text=True, timeout=5, env=_systemctl_env())
        vals = dict(line.split("=", 1) for line in r.stdout.splitlines() if "=" in line)
        state = vals.get("ActiveState", "unknown")
        substate = vals.get("SubState", "unknown")
        restarts = int(vals["NRestarts"]) if "NRestarts" in vals else None
    except Exception:
        pass
    try:
        r = subprocess.run(
            ["journalctl", "--user", "-u", unit, "-n", "1", "--no-pager", "-o", "cat"],
            capture_output=True, text=True, timeout=5, env=_systemctl_env())
        last_log = r.stdout.strip() or None
    except Exception:
        pass
    return state, substate, restarts, last_log


def _service_state_macos(name: str) -> tuple[str, str, "int | None", "str | None"]:
    label = f"com.pleiades.discord.{_safe_service_name(name)}"
    domain = f"gui/{os.getuid()}"
    state, substate, last_log = "unknown", "unknown", None
    try:
        r = subprocess.run(["launchctl", "print", f"{domain}/{label}"],
                            capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            state, substate = "inactive", "not-loaded"
        else:
            running = "state = running" in r.stdout.lower()
            state = "active" if running else "inactive"
            substate = "running" if running else "loaded"
    except Exception:
        pass
    log_path = (Path.home() / "Library" / "Logs"
                / f"pleiades-discord-{_safe_service_name(name)}.log")
    try:
        if log_path.is_file():
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            last_log = lines[-1] if lines else None
    except Exception:
        pass
    return state, substate, None, last_log


def _service_state_windows(name: str) -> tuple[str, str, "int | None", "str | None"]:
    task_name = f"Pleiades Discord {name}"
    state, substate = "unknown", "unknown"
    try:
        r = subprocess.run(["schtasks", "/Query", "/TN", task_name, "/FO", "LIST", "/V"],
                            capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            state, substate = "inactive", "not-installed"
        else:
            status_line = next((ln for ln in r.stdout.splitlines() if ln.startswith("Status:")), "")
            running = "running" in status_line.lower()
            state = "active" if running else "inactive"
            substate = status_line.split(":", 1)[-1].strip() or "unknown"
    except Exception:
        pass
    # No stdout redirection wired into the scheduled task XML, so there's no
    # log file to tail here the way Linux/macOS have one.
    return state, substate, None, None


def discord_status() -> list[dict]:
    """Per-character Discord bot status: service installed/running, token present,
    discord.py installed — the "logs and online status" the connector otherwise has
    none of. Best-effort; degrades to 'unknown' if the OS service manager isn't
    reachable. Linux: systemd --user. macOS: launchd. Windows: Task Scheduler."""
    try:
        import discord  # noqa: F401
        installed = True
    except ImportError:
        installed = False

    from ..profiles import ProfileManager
    out: list[dict] = []
    manager = ProfileManager()
    for profile in manager.list():
        if not profile.discord_enabled:
            continue
        with manager.open_vault(profile.name) as vault:
            has_token = bool(vault.get("discord.token"))
        if sys.platform == "darwin":
            state, substate, restarts, last_log = _service_state_macos(profile.name)
        elif os.name == "nt":
            state, substate, restarts, last_log = _service_state_windows(profile.name)
        else:
            state, substate, restarts, last_log = _service_state_linux(profile.name)
        out.append({
            "name": profile.name,
            "discordpy_installed": installed,
            "has_token": has_token,
            "service_state": state,
            "service_substate": substate,
            "restarts": restarts,
            "last_log": last_log,
        })
    return out


def run_discord_bot(name: str, *, engine: Optional[Engine] = None) -> None:
    try:
        import discord
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("discord.py not installed. Install the [discord] extra.") from e

    engine = engine or Engine()
    manager = engine.manager
    profile: Profile = manager.get(name)

    with manager.open_vault(name) as vault:
        token = vault.get("discord.token")
    if not token:
        raise RuntimeError(
            f"No discord.token in the vault for '{name}'. "
            f"Add it with: pleiades vault set {name} discord.token"
        )

    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        # flush=True: under systemd, stdout is a pipe (block-buffered), so this
        # otherwise never reaches `journalctl` — which was the actual "no online
        # status" complaint, since the gateway connects fine but you can't see it.
        print(f"[pleiades] {name} connected to Discord as {client.user}.", flush=True)

    def _channel_allowed(message) -> bool:
        allow = [c.strip().lower() for c in
                 (profile.discord_allowed_channels or "").split(",") if c.strip()]
        if not allow:
            return True
        ch = message.channel
        candidates = {str(ch.id), getattr(ch, "name", "").lower()}
        return bool(candidates & set(allow))

    @client.event
    async def on_message(message: "discord.Message"):
        if message.author == client.user:
            return
        if getattr(message.author, "bot", False) and not profile.discord_respond_to_bots:
            return
        is_dm = message.guild is None
        is_mention = client.user in getattr(message, "mentions", [])
        if is_dm:
            pass  # DMs always work
        else:
            if not _channel_allowed(message):
                return
            if profile.discord_require_mention and not is_mention:
                return

        content = message.content
        if is_mention and client.user is not None:
            content = content.replace(client.user.mention, "").strip()
        if not content:
            return

        async with message.channel.typing():
            # engine.run is blocking; run it off the event loop.
            import asyncio

            reply = await asyncio.to_thread(engine.run, profile, content)

        for chunk in _split(reply or "(no response)"):
            await message.channel.send(chunk)

    client.run(token)


def _split(text: str, limit: int = 1900):
    """Discord caps messages at 2000 chars; chunk safely."""
    for i in range(0, len(text), limit):
        yield text[i : i + limit]
