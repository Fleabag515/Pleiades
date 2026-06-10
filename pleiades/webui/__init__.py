"""Pleiades Web UI — a local control panel for characters, models, and settings.

A FastAPI app that wraps the existing Pleiades managers (ProfileManager,
ModelManager, Vault, Anamnesis, Settings) and serves a polished single-page
frontend. Cross-platform: it runs in the browser, so it behaves identically on
Linux and Windows.

Launch it with::

    pleiades ui                 # added to the CLI
    python -m pleiades.webui    # or directly

Everything it shows is your own local state under ``~/.pleiades`` — no data
leaves the machine.

Security: the control panel exposes powerful routes (``POST /api/work`` runs the
agent's shell/file tools). This package installs a loopback guard around the
FastAPI app so a webpage the user merely visits cannot drive it via DNS
rebinding or CSRF — see ``_guard.py``. The guard is wired here (not in the large
``server.py``) by wrapping ``create_app`` / ``run``.
"""

from __future__ import annotations

from typing import Any, Optional

from . import server as _server
from ._guard import LOOPBACK_HOSTS, allowlist, install_loopback_guard, is_wildcard

__all__ = ["create_app", "run"]

# Bind host for the *next* served app, set by run() so the arg-less create_app()
# call inside server.run() can scope the guard to the right address.
_state: dict[str, Any] = {"host": None, "enforce": True}

_raw_create_app = _server.create_app
_raw_run = _server.run


def create_app(allowed_hosts: "set[str] | None" = None,
               enforce_guard: bool = True) -> Any:
    """Build the control-panel app with the loopback guard installed.

    enforce_guard=False (or a wildcard bind recorded by run()) returns the raw
    app unguarded — the operator's explicit choice to expose it.
    """
    app = _raw_create_app()
    host = _state.get("host")
    guard_on = (enforce_guard and _state.get("enforce", True)
                and not is_wildcard(host))
    if not guard_on:
        return app
    allowed = allowlist(host)
    if allowed_hosts:
        allowed |= {h.lower() for h in allowed_hosts}
    install_loopback_guard(app, allowed)
    return app


# server.run() refers to the module-global name `create_app`; repoint it at the
# guarded wrapper so the served app is always protected, regardless of entry path.
_server.create_app = create_app


def run(host: str = "127.0.0.1", port: "int | None" = None,
        open_browser: bool = True) -> None:
    """Launch the control panel, scoping the loopback guard to `host`."""
    _state["host"] = host
    _state["enforce"] = not is_wildcard(host)
    if is_wildcard(host):
        print("  !  Binding to a wildcard address exposes the control panel — and\n"
              "     POST /api/work (which runs shell/file tools) — to your network.\n"
              "     The loopback guard is OFF for this bind; put it behind a trusted\n"
              "     reverse proxy with auth, or bind to 127.0.0.1 instead.\n")
    _raw_run(host=host, port=port, open_browser=open_browser)
