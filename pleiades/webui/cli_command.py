# ─────────────────────────────────────────────────────────────────────────
# Paste this block into pleiades/cli.py (anywhere among the other @cli.command
# definitions, e.g. right after the `serve` command). It adds `pleiades ui`.
# ─────────────────────────────────────────────────────────────────────────

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
