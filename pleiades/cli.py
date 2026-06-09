"""`pleiades` command-line interface."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

from . import config
from .anamnesis import Anamnesis, AnamnesisError
from .profiles import ProfileManager

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
    if config.profile_json_path(name).is_file():
        console.print(f"[red]Profile '{name}' already exists.[/red]")
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
                console.print(engine.run(profile, user))
            else:
                for chunk in engine.stream(profile, user):
                    console.print(chunk, end="")
                console.print()
        except Exception as e:
            console.print(f"\n[red]error: {e}[/red]")


# --------------------------------------------------------------------------- #
# discord
# --------------------------------------------------------------------------- #
@cli.command()
@click.argument("name")
def discord(name: str) -> None:
    """Run the character as a Discord bot (blocks)."""
    from .connectors.discord_bot import run_discord_bot

    try:
        run_discord_bot(name)
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)


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


if __name__ == "__main__":
    cli()
