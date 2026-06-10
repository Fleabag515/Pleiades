"""Loopback guard for the control-panel API (DNS-rebinding + CSRF defense).

The web UI exposes powerful routes — `POST /api/work` runs the full agent tool
belt (shell, files, ...) with a caller-chosen policy. Even bound to 127.0.0.1
the server is reachable by *any* page the user's browser loads, via two classic
attacks against localhost services:

  * DNS rebinding — an attacker domain re-resolves to 127.0.0.1, after which the
    browser sends requests to the local server carrying the *attacker's*
    hostname in the Host header.
  * CSRF — a cross-site page issues requests to 127.0.0.1; those carry a foreign
    Origin/Referer.

Validating Host and Origin against a loopback allow-list defeats both, and needs
no token in the same-origin UI (its own requests always carry a loopback Host
and Origin). When the operator deliberately binds to a non-loopback address that
host is added to the allow-list; a wildcard bind (0.0.0.0 / ::) is treated as an
explicit "expose me" and the guard steps aside (the launcher prints a warning).

This lives in its own module and is wired in from the package __init__, so the
large server.py is untouched.
"""

from __future__ import annotations

from typing import Optional
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_WILDCARD = {"", "0.0.0.0", "::", "[::]", "*"}


def is_wildcard(host: Optional[str]) -> bool:
    """True if `host` is a wildcard bind (explicit network exposure)."""
    return (host or "127.0.0.1").strip().lower() in _WILDCARD


def hostname(value: str) -> Optional[str]:
    """Lowercased host from a Host header or an Origin/Referer URL (port stripped)."""
    if not value:
        return None
    netloc = value if "://" in value else "//" + value
    try:
        return urlsplit(netloc).hostname
    except ValueError:
        return None


def allowlist(bind_host: Optional[str] = None) -> set[str]:
    """Allowed host set: loopback names plus a concrete non-loopback bind host."""
    allowed = set(LOOPBACK_HOSTS)
    if bind_host and not is_wildcard(bind_host):
        h = bind_host.strip().lower()
        if h not in LOOPBACK_HOSTS:
            allowed.add(h)
    return allowed


def install_loopback_guard(app: FastAPI, allowed: set[str]) -> None:
    """Reject requests whose Host or Origin/Referer is outside `allowed`."""

    @app.middleware("http")
    async def _loopback_guard(request: Request, call_next):
        host = hostname(request.headers.get("host", ""))
        if host is not None and host not in allowed:
            return JSONResponse(
                {"error": "Host not allowed. The Pleiades control panel only "
                          "serves loopback requests (DNS-rebinding guard)."},
                status_code=403)
        origin = request.headers.get("origin") or request.headers.get("referer")
        if origin:
            oh = hostname(origin)
            if oh is not None and oh not in allowed:
                return JSONResponse(
                    {"error": "Cross-origin request blocked."}, status_code=403)
        return await call_next(request)
