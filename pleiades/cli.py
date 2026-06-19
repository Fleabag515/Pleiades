"""`pleiades` command-line interface."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from . import config
from .anamnesis import AnamnesisError
from .profiles import ProfileManager

# Windows consoles default to a legacy code page (cp1252) that can't encode the
# arrows/box characters in our help text — Click's --help writes raw stdout, so a
# stray glyph crashes the whole command with UnicodeEncodeError. Force UTF-8 once,
# at import, so the CLI is encoding-safe everywhere. (Rich already does this for
# console.print; this covers click.echo / --help too.)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

console = Console()


def _manager() -> ProfileManager:
    return ProfileManager(config.Settings.load())


@click.group()
@click.version_option()
def cli() -> None:
    """Pleiades — a self-contained inference workstation for LLM agents."""


# --------------------------------------------------------------------------- #
# new
# --------------------------------------------------------------------------- #
@cli.command()
@click.argument("name")
@click.option("--provider", type=click.Choice(sorted(config.EMAIL_PRESETS)), default="generic",
              help="Email provider preset for IMAP/SMTP defaults.")
@click.option("--no-email", is_flag=True, help="Skip email setup.")
@click.option("--discord", "discord_token", default=None, help="Discord bot token (optional).")
def new(name: str, provider: str, no_email: bool, discord_token: str | None) -> None:
    """Create a character: Anamnesis character + Pleiades profile + vault + browser dir."""
    mgr = _manager()
    try:
        if config.profile_json_path(name).is_file():
            console.print(f"[red]Profile '{name}' already exists.[/red]")
            sys.exit(1)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    email_kwargs: dict = {}
    if not no_email:
        preset = config.EMAIL_PRESETS[provider]
        if preset.get("note"):
            console.print(f"[dim]{provider}: {preset['note']}[/dim]")
        address = Prompt.ask("Email address", default="")
        if address:
            email_kwargs.update(
                email_address=address,
                imap_host=Prompt.ask("IMAP host", default=str(preset.get("imap_host", ""))),
                imap_port=int(Prompt.ask("IMAP port", default=str(preset.get("imap_port", 993)))),
                smtp_host=Prompt.ask("SMTP host", default=str(preset.get("smtp_host", ""))),
                smtp_port=int(Prompt.ask("SMTP port", default=str(preset.get("smtp_port", 587)))),
                email_password=Prompt.ask("Email app password", password=True, default=""),
            )

    try:
        mgr.create(name, discord_token=discord_token, **email_kwargs)
    except (AnamnesisError, ValueError) as e:
        console.print(f"[red]Failed to create '{name}': {e}[/red]")
        sys.exit(1)
    console.print(f"[green]Created character '{name}'.[/green] Try: [bold]pleiades chat {name}[/bold]")


# --------------------------------------------------------------------------- #
# list
# --------------------------------------------------------------------------- #
@cli.command(name="list")
def list_cmd() -> None:
    """List characters and their status."""
    mgr = _manager()
    profiles = mgr.list()
    if not profiles:
        console.print("No characters yet. Create one with [bold]pleiades new <name>[/bold].")
        return

    table = Table(title="Pleiades characters")
    table.add_column("name", style="bold")
    table.add_column("anamnesis")
    table.add_column("email")
    table.add_column("discord")
    for p in profiles:
        try:
            running = mgr.anamnesis.is_running(p.name)
            status = "[green]running[/green]" if running else "stopped"
        except AnamnesisError:
            status = "[red]daemon down[/red]"
        table.add_row(
            p.name,
            status,
            p.email_address or "—",
            "enabled" if p.discord_enabled else "—",
        )
    console.print(table)
    orphans = mgr.orphan_characters()
    if orphans:
        console.print(f"\n[dim]Existing Anamnesis characters not yet adopted: "
                      f"{', '.join(orphans)}[/dim]")
        console.print("[dim]Adopt one (keeps its memory) with "
                      "[bold]pleiades adopt <name>[/bold].[/dim]")


# --------------------------------------------------------------------------- #
# chat
# --------------------------------------------------------------------------- #
@cli.command()
@click.argument("name")
@click.option("--no-stream", is_flag=True, help="Disable streaming.")
def chat(name: str, no_stream: bool) -> None:
    """Open a REPL chat with a character (memory + tools wired in)."""
    from .engine import Engine

    engine = Engine()
    try:
        profile = engine.manager.get(name)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    console.print(f"[bold cyan]Chatting with {name}[/bold cyan]  (Ctrl-D or 'exit' to quit)\n")
    while True:
        try:
            user = console.input("[bold green]you ›[/bold green] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            break
        if user.strip().lower() in {"exit", "quit"}:
            break
        if not user.strip():
            continue
        console.print(f"[bold magenta]{name} ›[/bold magenta] ", end="")
        try:
            if no_stream:
                console.print(engine.run(profile, user), markup=False)
            else:
                for chunk in engine.stream(profile, user):
                    console.print(chunk, end="", markup=False)
                console.print()
        except Exception as e:
            console.print(f"\n[red]error: {e}[/red]")


# --------------------------------------------------------------------------- #
# discord
# --------------------------------------------------------------------------- #
@cli.command()
@click.argument("name")
@click.option(
    "--service",
    "service_action",
    type=click.Choice(["install", "uninstall", "status"]),
    default=None,
    help="Manage a persistent systemd user service instead of running in the foreground.",
)
def discord(name: str, service_action: "str | None") -> None:
    """Run the character as a Discord bot (blocks), or manage its systemd service.

    \b
    Run in foreground (manual):
        pleiades discord Mark

    Install as a background service (auto-starts on login):
        pleiades discord Mark --service install

    Check service status / remove:
        pleiades discord Mark --service status
        pleiades discord Mark --service uninstall
    """
    if service_action:
        _discord_service(name, service_action)
        return

    from .connectors.discord_bot import run_discord_bot

    try:
        run_discord_bot(name)
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)


def _discord_service(name: str, action: str) -> None:
    """Create/remove/query the systemd user unit for a character's Discord bot."""
    import os
    import shutil

    # Ensure systemctl --user can reach the session D-Bus even when called
    # from a non-login shell (e.g. a script or MCP tool).
    uid = os.getuid()
    os.environ.setdefault("XDG_RUNTIME_DIR", f"/run/user/{uid}")
    os.environ.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path=/run/user/{uid}/bus")

    unit_name = f"pleiades-discord-{name.lower()}.service"
    unit_dir = Path.home() / ".config" / "systemd" / "user"
    unit_path = unit_dir / unit_name
    pleiades_bin = shutil.which("pleiades") or str(
        Path(sys.executable).parent / "pleiades"
    )

    if action == "install":
        unit_dir.mkdir(parents=True, exist_ok=True)
        unit_path.write_text(
            "[Unit]\n"
            f"Description=Pleiades Discord bot – {name}\n"
            "After=network-online.target\n"
            "Wants=network-online.target\n"
            "\n"
            "[Service]\n"
            "Type=simple\n"
            f"ExecStart={pleiades_bin} discord {name}\n"
            "Restart=on-failure\n"
            "RestartSec=10\n"
            "\n"
            "[Install]\n"
            "WantedBy=default.target\n"
        )
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
        try:
            subprocess.run(
                ["systemctl", "--user", "enable", "--now", unit_name], check=True
            )
        except subprocess.CalledProcessError as e:
            console.print(f"[red]systemctl failed: {e}[/red]")
            sys.exit(1)
        console.print(f"[green]Service '[bold]{unit_name}[/bold]' installed and started.[/green]")
        console.print(f"Logs:   [bold]journalctl --user -u {unit_name} -f[/bold]")
        console.print(f"Stop:   [bold]pleiades discord {name} --service uninstall[/bold]")

    elif action == "uninstall":
        if not unit_path.exists():
            console.print(f"[yellow]No service '[bold]{unit_name}[/bold]' found.[/yellow]")
            return
        subprocess.run(
            ["systemctl", "--user", "disable", "--now", unit_name], check=False
        )
        unit_path.unlink()
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
        console.print(f"[green]Service '[bold]{unit_name}[/bold]' stopped and removed.[/green]")

    elif action == "status":
        result = subprocess.run(
            ["systemctl", "--user", "status", unit_name]
        )
        sys.exit(result.returncode)


# --------------------------------------------------------------------------- #
# work — the Claude-Code-style agent harness (optionally as a character)
# --------------------------------------------------------------------------- #
@cli.command()
@click.argument("task", nargs=-1)
@click.option("--as", "character", default=None,
              help="Run as a character: binds its identity, memory, vault, and email.")
@click.option("--tier", default=None,
              help="Model tier: local (default, our llama.cpp), coder, cloud, ollama, ...")
@click.option("--policy", default=None, type=click.Choice(["ask", "allow", "deny"]),
              help="Permission policy for side-effecting tools.")
def work(task: tuple[str, ...], character: str | None, tier: str | None,
         policy: str | None) -> None:
    """Run the agent harness on a TASK (files, git, shell, web, browser, +more).

    Local llama.cpp inference is the default brain. With --as <character> the agent
    operates inside that character's workspace, memory, vault, and inbox, routing
    inference through its Anamnesis proxy.
    """
    from .harness import Agent, Config
    from .harness import builtins as _builtins      # noqa: F401  registers tools
    from .harness import identity as _identity      # noqa: F401  registers vault/email
    from .harness.subagent import bind_context
    from .harness.builtins.memory import bind_memory
    from .harness.anamnesis import Anamnesis as WorkingMemory

    if not task:
        console.print('[yellow]Give a task, e.g. pleiades work "summarize README.md"[/yellow]')
        sys.exit(1)

    cfg = Config.load()
    if policy:
        cfg.exec_policy = policy

    if character:
        try:
            _identity.bind_character(character, cfg=cfg)
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            sys.exit(1)
        console.print(f"[dim]bound character: {character}[/dim]")

    # In-process working/long-term memory tier (scoped to the character if bound).
    bind_memory(WorkingMemory.from_config(cfg))

    def on_event(kind, payload):
        if kind == "tool_call":
            console.print(f"[cyan]→ {payload.name}[/cyan]")
        elif kind == "tool_result":
            _call, out, err = payload
            mark = "[red]x[/red]" if err else "[green]ok[/green]"
            first = out.splitlines()[0][:100] if out else ""
            console.print(f"  {mark} {first}")
        elif kind == "text":
            console.print(f"\n{payload}")

    agent = Agent(cfg, tier_name=tier)
    bind_context(cfg, depth=0, approve=agent.approve)
    console.print(f"[dim]policy={cfg.exec_policy} · tier={tier or cfg.default_tier} · "
                  f"backend={cfg.tier(tier or cfg.default_tier).backend}[/dim]\n")
    try:
        result = agent.run(" ".join(task), on_event=on_event)
    except Exception as e:
        console.print(f"[red]error: {e}[/red]")
        sys.exit(1)
    console.print(f"\n[dim]- done in {getattr(result, 'steps', '?')} steps -[/dim]")


# --------------------------------------------------------------------------- #
# audit — stress-test the tool belt with a subscription-billed Claude session
# --------------------------------------------------------------------------- #
@cli.command()
@click.option("--fix", is_flag=True,
              help="Let Claude patch source + add tests (default: read-only triage).")
@click.option("--tier", default="claude",
              help="Model tier (default: claude = subscription-billed).")
@click.option("--policy", default=None, type=click.Choice(["ask", "allow", "deny"]),
              help="Permission policy for side-effecting tools.")
@click.option("--tools", "tool_csv", default=None,
              help="Comma-separated subset of tool names to audit (default: all).")
@click.option("--max-budget", "max_budget", type=float, default=None,
              help="Hard USD budget cap for the run.")
def audit(fix: bool, tier: str, policy: str | None,
          tool_csv: str | None, max_budget: float | None) -> None:
    """Stress-test Pleiades' own tool belt from the inside (Claude on your plan).

    Claude exercises each tool with realistic and adversarial inputs, logs failures
    to TOOL_AUDIT.md, and with --fix branches, patches source, and adds tests.
    """
    from .harness import Config
    from .harness import builtins as _builtins        # noqa: F401  registers tools
    from .harness.claude_backend import run_claude, audit_charter

    cfg = Config.load()
    cfg.exec_policy = policy or ("allow" if fix else "ask")
    names = [t.strip() for t in tool_csv.split(",")] if tool_csv else None
    system, task = audit_charter(fix=fix, tool_names=names)

    def ev(kind, payload):
        if kind == "tool_call":
            console.print(f"[cyan]\u2192 {getattr(payload, 'name', payload)}[/cyan]")
        elif kind == "text":
            console.print(f"\n{payload}")
        elif kind == "warn":
            console.print(f"[yellow]! {payload}[/yellow]")

    console.print(f"[dim]audit \u00b7 fix={fix} \u00b7 policy={cfg.exec_policy} \u00b7 tier={tier}[/dim]\n")
    res = run_claude(task, cfg, tier_name=tier, system=system,
                     add_dirs=[cfg.workspace_root], max_budget_usd=max_budget,
                     on_event=ev)
    if res.error:
        console.print(f"[red]error: {res.error}[/red]")
        sys.exit(1)
    console.print(f"\n[dim]- audit done \u00b7 {res.turns} turns \u00b7 "
                  f"${res.cost_usd:.4f} \u00b7 see TOOL_AUDIT.md -[/dim]")


# --------------------------------------------------------------------------- #
# vault
# --------------------------------------------------------------------------- #
@cli.group()
def vault() -> None:
    """Manage a character's secrets."""


@vault.command("set")
@click.argument("name")
@click.argument("key")
@click.option("--value", default=None, help="Value (omit to be prompted securely).")
def vault_set(name: str, key: str, value: str | None) -> None:
    if value is None:
        value = Prompt.ask(f"Value for {key}", password=True)
    with _manager().open_vault(name) as v:
        v.set(key, value)
    console.print(f"[green]Stored '{key}' for {name}.[/green]")


@vault.command("get")
@click.argument("name")
@click.argument("key")
def vault_get(name: str, key: str) -> None:
    with _manager().open_vault(name) as v:
        value = v.get(key)
    console.print(value if value is not None else f"[yellow]No secret '{key}'.[/yellow]")


@vault.command("list")
@click.argument("name")
def vault_list(name: str) -> None:
    with _manager().open_vault(name) as v:
        entries = v.list()
    if not entries:
        console.print("No secrets stored.")
        return
    for e in entries:
        console.print(f"- {e['key']}" + ("  [dim](reserved)[/dim]" if e["reserved"] else ""))


@vault.command("rm")
@click.argument("name")
@click.argument("key")
def vault_rm(name: str, key: str) -> None:
    with _manager().open_vault(name) as v:
        ok = v.delete(key)
    console.print(f"[green]Deleted '{key}'.[/green]" if ok else f"[yellow]No secret '{key}'.[/yellow]")


# --------------------------------------------------------------------------- #
# search (SearXNG via docker compose)
# --------------------------------------------------------------------------- #
@cli.command()
@click.argument("direction", type=click.Choice(["up", "down"]))
def search(direction: str) -> None:
    """Bring the local SearXNG service up or down (docker compose)."""
    compose = Path(__file__).resolve().parent.parent / "docker-compose.yml"
    if not compose.is_file():
        console.print(f"[red]docker-compose.yml not found at {compose}.[/red]")
        sys.exit(1)
    args = ["docker", "compose", "-f", str(compose)]
    args += ["up", "-d"] if direction == "up" else ["down"]
    try:
        subprocess.run(args, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        console.print(f"[red]docker compose failed: {e}[/red]")
        sys.exit(1)


# --------------------------------------------------------------------------- #
# serve (debug: bring our inference server up in the foreground)
# --------------------------------------------------------------------------- #
@cli.command()
def serve() -> None:
    """Start the in-process llama.cpp inference server (foreground; for debugging)."""
    from .inference import InferenceServer

    settings = config.Settings.load()
    if not settings.model_path:
        console.print("[red]Set PLEIADES_MODEL_PATH first.[/red]")
        sys.exit(1)
    server = InferenceServer(settings)
    console.print(f"Starting inference at {server.base_url} (model: {settings.model_path}) …")
    server.start(wait=True)
    console.print("[green]Ready.[/green] Press Ctrl-C to stop.")
    try:
        while True:
            if server._proc and server._proc.poll() is not None:
                break
            import time

            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


# --------------------------------------------------------------------------- #
# runtime — native llama.cpp server (unlocks MoE expert offload)
# --------------------------------------------------------------------------- #
@cli.group()
def runtime() -> None:
    """Manage the native llama.cpp runtime (faster; enables MoE CPU offload)."""


@runtime.command("install")
def runtime_install() -> None:
    """Download the best official llama.cpp build for this machine."""
    from . import runtime as rt
    try:
        path = rt.install(log=lambda m: console.print(f"[dim]{m}[/dim]"))
    except Exception as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    console.print(f"[green]Native runtime installed:[/green] {path}")
    console.print("Models now launch through llama-server with MoE-aware autofit. "
                  "Restart any running models to pick it up.")


@runtime.command("status")
def runtime_status() -> None:
    """Show which inference runtime models will use."""
    from . import runtime as rt
    st = rt.status()
    if st["native"]:
        console.print(f"native llama-server: [green]{st['native']}[/green] "
                      f"{st.get('version', '')}")
        console.print("MoE expert offload: "
                      + ("[green]supported[/green]" if st["moe_offload"]
                         else "[yellow]not supported by this build[/yellow]"))
    else:
        console.print("native llama-server: [yellow]not installed[/yellow] "
                      "(models use the python server)")
        console.print("Install it for faster inference + MoE offload: "
                      "[bold]pleiades runtime install[/bold]")


# --------------------------------------------------------------------------- #
# hw — what this machine has and how models will be placed on it
# --------------------------------------------------------------------------- #
@cli.command()
@click.option("--model", "model_name", default=None,
              help="Also show the offload plan for a registered model.")
def hw(model_name: str | None) -> None:
    """Show detected hardware (GPU/VRAM/RAM) and how models map onto it."""
    from . import hardware
    det = hardware.detect()
    console.print(det.describe())
    from .models import ModelManager
    mm = ModelManager()
    targets = [m for m in mm.list() if not model_name or m["name"] == model_name]
    s = config.Settings.load()
    if s.model_path and not model_name:
        targets.insert(0, {"name": "(default engine)", "path": s.model_path,
                           "n_ctx": s.n_ctx})
    from . import runtime as rt
    from .autofit import place
    cps = rt.caps()
    for m in targets:
        meta = hardware.read_gguf_meta(m["path"])
        raw_ctx = m.get("n_ctx", "auto")
        if str(raw_ctx).strip().lower() in ("", "auto"):
            cp = hardware.plan_context(meta, det, caps=cps)
            n_ctx_val, ctx_line = cp.n_ctx, f"\n  [dim]ctx: {cp.reason}[/dim]"
        else:
            n_ctx_val, ctx_line = int(raw_ctx), ""
        p = place(meta, n_ctx_val, det, cps)
        moe = " · MoE" if meta.is_moe else ""
        console.print(f"\n[bold]{m['name']}[/bold]{moe} → ctx={n_ctx_val}, {p.strategy} "
                      f"(~{p.est_tps} tok/s){ctx_line}\n  [dim]{p.reason}[/dim]")
        if meta.is_moe and not cps.moe_offload:
            console.print("  [yellow]tip: pleiades runtime install → "
                          "expert-offload would speed this up a lot[/yellow]")
    if not targets:
        console.print("\n[dim]No models registered yet — try: "
                      "pleiades model fetch <hf-repo>[/dim]")


# --------------------------------------------------------------------------- #
# ui — local web control panel
# --------------------------------------------------------------------------- #
@cli.command()
@click.option("--host", default="127.0.0.1", show_default=True,
              help="Bind address. Use 0.0.0.0 to reach it from another machine.")
@click.option("--port", type=int, default=None,
              help="Port (default: first free port from 8750).")
@click.option("--no-browser", is_flag=True,
              help="Do not auto-open a browser (headless / remote).")
def ui(host: str, port: int | None, no_browser: bool) -> None:
    """Launch the Pleiades control panel (local web UI). Works on Linux and Windows."""
    try:
        from .webui import run
    except ModuleNotFoundError as e:  # fastapi/uvicorn not installed
        console.print(f"[red]Web UI dependencies missing: {e}[/red]")
        console.print("Install them with: [bold]pip install -e \".[ui]\"[/bold] "
                      "(or: pip install fastapi uvicorn)")
        sys.exit(1)
    run(host=host, port=port, open_browser=not no_browser)


# --------------------------------------------------------------------------- #
# adopt — wrap a pre-existing Anamnesis character as a Pleiades profile
# --------------------------------------------------------------------------- #
@cli.command()
@click.argument("name")
def adopt(name: str) -> None:
    """Adopt an EXISTING Anamnesis character (keeps its memory) as a Pleiades profile."""
    mgr = _manager()
    try:
        mgr.adopt(name)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    console.print(f"[green]Adopted '{name}'.[/green] Its Anamnesis memory is preserved. "
                  f"Try: [bold]pleiades chat {name}[/bold]")


# --------------------------------------------------------------------------- #
# update — pull the latest Pleiades from GitHub and reinstall
# --------------------------------------------------------------------------- #
@cli.command()
@click.option("--no-reinstall", is_flag=True, help="Pull only; skip pip reinstall.")
@click.option("--check", "check_only", is_flag=True, help="Only check; don't update.")
def update(no_reinstall: bool, check_only: bool) -> None:
    """Update Pleiades to the latest version on GitHub (also: the Update button in the UI)."""
    from . import update as upd

    if check_only:
        c = upd.check()
        if c.error:
            console.print(f"[yellow]{c.error}[/yellow]")
        elif c.behind:
            console.print(f"Update available: {c.installed} → [green]{c.latest}[/green]. "
                          "Run [bold]pleiades update[/bold].")
        else:
            console.print(f"[green]Up to date ({c.installed}).[/green]")
        return
    try:
        changed = upd.run_update(reinstall=not no_reinstall,
                                 log=lambda m: console.print(f"[dim]{m}[/dim]"))
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    console.print("[green]Updated.[/green]" if changed
                  else "[green]Already up to date.[/green]")


# --------------------------------------------------------------------------- #
# model — register, run, stop, and assign local GGUF models
# --------------------------------------------------------------------------- #
@cli.group()
def model() -> None:
    """Manage the local models (GGUFs) Pleiades runs."""


@model.command("add")
@click.argument("name")
@click.argument("path")
@click.option("--ctx", "n_ctx", default="auto", show_default=True,
              help="Context window: 'auto' (planned from VRAM at each launch, "
                   "elastic at runtime) or a number to pin it.")
@click.option("--gpu-layers", "n_gpu_layers", default="auto", show_default=True,
              help="GPU offload: 'auto' (planned from detected hardware at launch), "
                   "-1 = all layers, 0 = CPU. NVIDIA/CUDA, AMD/ROCm, Apple/Metal builds.")
@click.option("--chat-format", default="", help="Override chat template (e.g. chatml, llama-3).")
@click.option("--port", default=0, help="Fixed port (0 = auto-assign).")
def model_add(name: str, path: str, n_ctx: str, n_gpu_layers: str,
              chat_format: str, port: int) -> None:
    """Register a GGUF model file under NAME."""
    from .models import ModelManager, ModelError
    try:
        m = ModelManager().add(name, path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
                               chat_format=chat_format, port=port)
    except ModelError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    console.print(f"[green]Added model '{name}'[/green] → {m['path']} "
                  f"(port {m['port']}, gpu_layers {m['n_gpu_layers']}).")
    console.print(f"Start it: [bold]pleiades model start {name}[/bold]")


@model.command("resize")
@click.argument("name")
@click.argument("n_ctx", type=int)
def model_resize(name: str, n_ctx: int) -> None:
    """Resize a RUNNING model's context window in place (elastic engine)."""
    from .models import ModelManager, ModelError
    try:
        out = ModelManager().resize(name, n_ctx)
    except ModelError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    console.print(f"[green]'{name}' now at n_ctx {out.get('n_ctx')}[/green] "
                  f"({out.get('took_ms', '?')} ms)")


@model.command("fetch")
@click.argument("repo")
@click.option("--name", default="", help="Registry name (default: derived from repo).")
@click.option("--quant", default="", help="Force a quant (e.g. Q4_K_M); default: auto-pick for this machine.")
@click.option("--ctx", "n_ctx", default="auto", show_default=True,
              help="Context window to register: 'auto' (planned from VRAM at each "
                   "launch, elastic at runtime) or a number to pin it.")
def model_fetch(repo: str, name: str, quant: str, n_ctx: str) -> None:
    """Download the best GGUF quant for THIS machine from a Hugging Face REPO.

    Example: pleiades model fetch bartowski/Llama-3.2-3B-Instruct-GGUF
    """
    from .fetch import FetchError, fetch_model
    from rich.progress import (BarColumn, DownloadColumn, Progress, TextColumn,
                               TransferSpeedColumn)
    with Progress(TextColumn("[bold blue]{task.fields[fname]}"), BarColumn(),
                  DownloadColumn(), TransferSpeedColumn(), console=console) as bar:
        tasks: dict[str, object] = {}

        def progress(fname: str, done: int, total: int) -> None:
            if fname not in tasks:
                tasks[fname] = bar.add_task("dl", fname=fname, total=total or None)
            bar.update(tasks[fname], completed=done, total=total or None)

        try:
            entry = fetch_model(repo, name=name, quant=quant, n_ctx=n_ctx,
                                progress=progress)
        except FetchError as e:
            console.print(f"[red]{e}[/red]")
            sys.exit(1)
    console.print(f"[green]Registered '{entry['name']}'[/green] — {entry['chosen']}")
    console.print(f"[dim]{entry['why']}[/dim]")
    console.print(f"Chat with it: [bold]pleiades model use <character> {entry['name']}[/bold]")


@model.command("search")
@click.argument("query", nargs=-1, required=True)
@click.option("--limit", default=8, show_default=True)
def model_search(query: tuple[str, ...], limit: int) -> None:
    """Search Hugging Face for GGUF model repos."""
    from .fetch import FetchError, search_gguf_repos
    try:
        rows = search_gguf_repos(" ".join(query), limit=limit)
    except FetchError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    if not rows:
        console.print("No GGUF repos found.")
        return
    for r in rows:
        console.print(f"- [bold]{r['id']}[/bold]  "
                      f"[dim]{r['downloads']:,} downloads · {r['likes']} likes[/dim]")
    console.print("\nFetch one: [bold]pleiades model fetch <repo-id>[/bold]")


@model.command("list")
def model_list() -> None:
    """List registered models and whether each is running."""
    from .models import ModelManager
    models = ModelManager().list()
    if not models:
        console.print("No models yet. Add one: [bold]pleiades model add <name> <path.gguf>[/bold]")
        return
    table = Table(title="Models")
    table.add_column("name", style="bold")
    table.add_column("status")
    table.add_column("port")
    table.add_column("gpu_layers")
    table.add_column("path")
    for m in models:
        table.add_row(
            m["name"],
            "[green]running[/green]" if m["running"] else "stopped",
            str(m["port"]), str(m["n_gpu_layers"]), m["path"],
        )
    console.print(table)


@model.command("rm")
@click.argument("name")
def model_rm(name: str) -> None:
    """Stop (if running) and unregister a model."""
    from .models import ModelManager
    ok = ModelManager().remove(name)
    console.print(f"[green]Removed '{name}'.[/green]" if ok else f"[yellow]No model '{name}'.[/yellow]")


@model.command("start")
@click.argument("name")
@click.option("--no-wait", is_flag=True, help="Return immediately; don't wait for readiness.")
def model_start(name: str, no_wait: bool) -> None:
    """Start a model's inference server (background)."""
    from .models import ModelManager, ModelError
    try:
        url = ModelManager().start(name, wait=not no_wait)
    except ModelError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    console.print(f"[green]Model '{name}' serving at {url}[/green]")


@model.command("stop")
@click.argument("name")
def model_stop(name: str) -> None:
    """Stop a model's inference server."""
    from .models import ModelManager
    ok = ModelManager().stop(name)
    console.print(f"[green]Stopped '{name}'.[/green]" if ok else f"[yellow]'{name}' was not running.[/yellow]")


@model.command("running")
def model_running() -> None:
    """Show currently running model servers."""
    from .models import ModelManager
    rows = ModelManager().running()
    if not rows:
        console.print("No models running.")
        return
    for r in rows:
        mark = "[green]alive[/green]" if r["alive"] else "[red]dead[/red]"
        console.print(f"- {r['name']}  (pid {r['pid']}, port {r['port']})  {mark}")


@model.command("use")
@click.argument("profile")
@click.argument("model_name")
def model_use(profile: str, model_name: str) -> None:
    """Assign MODEL_NAME to a character PROFILE (it'll use that model when chatting)."""
    from .models import ModelManager
    mgr = _manager()
    try:
        mgr.get(profile)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    if ModelManager().get(model_name) is None:
        console.print(f"[red]No model '{model_name}'. Add it: pleiades model add {model_name} <path>[/red]")
        sys.exit(1)
    mgr.assign_model(profile, model_name)
    console.print(f"[green]{profile} now uses model '{model_name}'.[/green] "
                  f"It starts automatically on next chat (or: pleiades model start {model_name}).")


if __name__ == "__main__":
    cli()
