"""Tor-routed browser tool — a separate, explicit-action browser for
anonymized/Tor-network browsing, alongside (not instead of) ``BrowserTool``
(``pleiades/tools/browser.py``).

Why a SEPARATE tool rather than a mode flag on BrowserTool
------------------------------------------------------------
BrowserTool is tightly coupled to one always-on, per-character session that
starts the instant the model calls ``goto`` and is meant to be the SAME
session the desktop panel shows live (webui/browser_view.py) or the one
persistent Camoufox window Discord/`pleiades work` reuse across calls. Tor
browsing is a different kind of thing on purpose: routing through Tor is
slower, trips CAPTCHA/verification walls far more often (exit nodes are
heavily rate-limited and blocklisted across the web), and should be a
deliberate choice the model makes for a specific reason (an onion service,
wanting a different exit IP, not wanting a site to see the character's usual
address) — not something that silently starts riding a different network
path. A separate tool name (`tor_browser`, called explicitly) makes that
choice visible in the transcript instead of hiding it behind a parameter the
model might pass by accident. It also means an ordinary `browser` session and
a `tor_browser` session can be open at the same time without fighting over
one singleton or one profile directory.

Why headed (a real OS window), not the desktop panel's embedded live view
------------------------------------------------------------------------
"Headed" here means a real OS-level browser window, mirroring
``harness/builtins/browser.py``'s existing headed-by-default Camoufox
session — NOT the desktop app's embedded panel (``webui/browser_view.py``).
Two independent reasons rule out the panel route for this tool specifically:

  1. Panel embedding is Chromium-only. browser_view.py's own docstring is
     explicit about why: the live-frame trick is Chrome DevTools Protocol's
     screencast feature (``Page.startScreencast``), which has no Firefox/
     BiDi equivalent today. Camoufox — the stealth backend this tool prefers
     for the same anti-fingerprint reasons BrowserTool's Camoufox fallback
     uses it — IS Firefox. Panel-embedding this tool would mean giving up
     Camoufox's stealth properties in favor of plain Chromium+Tor, which
     defeats a real chunk of the point (already-suspicious Tor exit traffic
     with an easily-fingerprinted default browser is a worse combination
     than Tor traffic with an anti-fingerprint Firefox).
  2. It matches the actual use case better. harness/builtins/browser.py is
     headed by default for exactly the reason this tool needs it too: "the
     agent doesn't solve CAPTCHAs or auto-defeat verification... solving
     human-verification challenges is your call, made in the visible
     window." Tor makes that wall come up far more often, so a real visible
     window an owner can glance at and step into matters MORE here, not
     less — and this is a deliberate, occasional action rather than an
     always-on per-character session, so it doesn't need the panel's
     watch-anytime-from-the-desktop-app semantics BrowserTool was built for.

So: headed by default (``PLEIADES_TOR_BROWSER_HEADLESS=1`` to hide it),
Camoufox-first with a plain Playwright-Firefox fallback, same shape as
``harness/builtins/browser.py``'s singleton — just proxied through Tor and
kept on its own persistent profile dir (``profiles/<name>/tor_browser/``,
see ``config.tor_browser_dir`` / ``Profile.tor_browser_dir``) so it never
shares cookies/storage with the ordinary browser tool's profile.

Action set / helper reuse
--------------------------
Same actions as BrowserTool where they make sense — goto/click/type/read/
screenshot/elements — for a consistent model-facing surface. Reuses rather
than re-implements:

  * ``_ELEMENTS_JS``, ``_format_elements``, ``_ref_selector`` from
    ``tools/browser.py`` (pure JS string / pure functions — Phase 1 of
    docs/specs/2026-07-22-vision-grounded-browser-use-design.md's
    accessibility-tree extraction — no async dependency, so they drop
    straight into this tool's synchronous Playwright/Camoufox calls).
  * ``_clickables``, ``_input_fields``, ``_hint``, ``_detect_challenge``,
    ``_CHALLENGE_MARKERS`` from ``harness/builtins/browser.py`` (already
    page-argument-based sync helpers, not tied to that module's own
    singleton state) for the same "what's actually on this page" hint-on-
    failure behavior and CAPTCHA/verification-wall detection.

Requires the `tor` daemon/package running locally with its SOCKS5 proxy
reachable (default 127.0.0.1:9050, overridable via
``PLEIADES_TOR_SOCKS_PROXY``, e.g. "socks5://127.0.0.1:9150" for Tor
Browser's own bundled process instead of a system daemon). A clear,
actionable error is returned if it isn't reachable — this tool never tries
to silently fall back to a non-anonymized connection.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from urllib.parse import urlsplit

from . import Tool, ToolContext, format_invalid_choice
from .browser import _ELEMENTS_JS, _format_elements, _ref_selector

_DEFAULT_SOCKS_PROXY = "socks5://127.0.0.1:9050"

# module-level singleton — one Tor-routed window, reused across tool calls,
# entirely separate from harness/builtins/browser.py's own _B singleton.
_TB: dict[str, object] = {"page": None, "ctx": None, "pw": None, "backend": None}
_LAST_PROFILE: dict[str, str | None] = {"dir": None}
_LAST_ERROR: dict[str, str | None] = {"error": None}


def tor_proxy_server() -> str:
    """The SOCKS5 proxy URL Playwright/Camoufox should dial through."""
    return os.environ.get("PLEIADES_TOR_SOCKS_PROXY", _DEFAULT_SOCKS_PROXY)


def _headless() -> bool:
    return os.environ.get("PLEIADES_TOR_BROWSER_HEADLESS", "0") == "1"


def _ensure_display() -> None:
    """Same reasoning as harness/builtins/browser.py's helper of the same
    name: a headed Firefox needs an X display, and processes started via
    nohup/systemd --user often don't inherit the desktop session's DISPLAY."""
    if os.environ.get("DISPLAY"):
        return
    os.environ["DISPLAY"] = os.environ.get("PLEIADES_DISPLAY", ":0")


def socks_host_port(server: str | None = None) -> tuple[str, int]:
    """Parse host/port out of a socks5://host:port URL for a raw TCP reachability check."""
    parsed = urlsplit(server or tor_proxy_server())
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 9050
    return host, port


def tor_socks_reachable(timeout: float = 1.5) -> bool:
    """Quick local TCP connect check to the configured SOCKS proxy — lets
    _ensure_page fail with a clear, specific message instead of a browser
    launch timing out and blaming Playwright."""
    host, port = socks_host_port()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def tor_browser_status() -> dict:
    """Current backend/profile/proxy/last-error — for UI status, not the model."""
    return {
        "backend": _TB["backend"],
        "page_open": _TB["page"] is not None,
        "profile_dir": _LAST_PROFILE["dir"],
        "proxy": tor_proxy_server(),
        "socks_reachable": tor_socks_reachable(),
        "last_error": _LAST_ERROR["error"],
    }


def _profile_dir(ctx: ToolContext) -> str:
    p = ctx.profile.tor_browser_dir
    Path(p).mkdir(parents=True, exist_ok=True)
    return p


def _bind_profile(ctx: ToolContext) -> str:
    target = _profile_dir(ctx)
    if _LAST_PROFILE["dir"] not in (None, target):
        _close()
    _LAST_PROFILE["dir"] = target
    return target


def _close() -> None:
    try:
        if _TB["ctx"] is not None:
            _TB["ctx"].close()  # type: ignore[union-attr]
    except Exception:
        pass
    # Tear down whatever launched the browser -- NOT just the "playwright-tor"
    # fallback. The "camoufox-tor" backend's `pw` is the raw Camoufox
    # (PlaywrightContextManager subclass) instance itself, which only exposes
    # `__exit__`, not `.stop()` -- calling `.stop()` unconditionally on it was
    # skipped by the old `backend == "playwright-tor"` check, so the
    # Camoufox path NEVER released its event loop here. Since the sync
    # Playwright/Camoufox API keeps that thread's asyncio loop marked
    # "running" for as long as the object is alive, a never-closed Camoufox
    # instance means every later goto in this same thread -- for the rest of
    # this process's life -- fails with Playwright's "Sync API inside the
    # asyncio loop" error instead of doing anything useful. Try both shapes,
    # best-effort, regardless of which backend was actually in use.
    pw = _TB["pw"]
    if pw is not None:
        for method in ("stop", "__exit__"):
            fn = getattr(pw, method, None)
            if fn is None:
                continue
            try:
                fn(None, None, None) if method == "__exit__" else fn()
                break
            except Exception:
                continue
    _TB.update(page=None, ctx=None, pw=None, backend=None)


def _ensure_page(ctx: ToolContext):
    """Lazily launch the Tor-proxied browser (Camoufox if present, else
    Playwright Firefox) with a persistent, tool-specific profile."""
    if _TB["page"] is not None:
        return _TB["page"]

    if not tor_socks_reachable():
        raise RuntimeError(
            f"Tor SOCKS proxy not reachable at {tor_proxy_server()}. Is the "
            "`tor` daemon installed and running? Try: sudo apt install tor "
            "&& sudo systemctl start tor  (or set PLEIADES_TOR_SOCKS_PROXY "
            "if Tor's SOCKS port is somewhere else, e.g. Tor Browser's own "
            "bundled process on 127.0.0.1:9150)."
        )

    _ensure_display()
    base = _profile_dir(ctx)
    headless = _headless()
    proxy = {"server": tor_proxy_server()}

    # 1) Camoufox (stealth Firefox) with the Tor proxy wired into its own
    #    launch — preferred, same reasoning as harness/builtins/browser.py.
    try:
        from camoufox.sync_api import Camoufox

        profile = os.path.join(base, "camoufox")
        Path(profile).mkdir(parents=True, exist_ok=True)
        last_exc: Exception | None = None
        for _attempt in range(2):  # first-run profile setup can outrun a 30s launch timeout
            cam = Camoufox(headless=headless, persistent_context=True,
                           user_data_dir=profile, proxy=proxy, timeout=60000)
            try:
                c = cam.start() if hasattr(cam, "start") else cam.__enter__()
                page = c.pages[0] if c.pages else c.new_page()
                _TB.update(page=page, ctx=c, pw=cam, backend="camoufox-tor")
                _LAST_ERROR["error"] = None
                return page
            except ImportError:
                raise
            except Exception as e:
                last_exc = e
                # A failed launch can still leave this thread's asyncio loop
                # marked "running" (Playwright's sync context manager creates
                # and starts pumping it before the failure surfaces) -- if we
                # don't tear it down here, EVERY later call in this same
                # thread (this retry's second pass, and every future 'goto'
                # for the life of this process) hits Playwright's own "Sync
                # API inside the asyncio loop" guard instead of the real
                # error, forever, because the thread never gets a clean
                # loop again. Best-effort close before looping/raising so a
                # transient failure stays transient instead of permanently
                # wedging the tool.
                try:
                    exit_fn = getattr(cam, "__exit__", None) or getattr(cam, "close", None)
                    if exit_fn is not None:
                        try:
                            exit_fn(type(e), e, e.__traceback__)
                        except TypeError:
                            exit_fn()
                except Exception:
                    pass
        raise last_exc  # type: ignore[misc]
    except ImportError:
        pass  # fall through to plain Playwright Firefox

    # 2) Playwright Firefox with a persistent context + proxy.
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise RuntimeError(
            "No browser backend installed. Install one:\n"
            "  pip install playwright && python -m playwright install firefox\n"
            "  (preferred, anti-fingerprint) pip install camoufox[geoip]")
    profile = os.path.join(base, "playwright")
    Path(profile).mkdir(parents=True, exist_ok=True)
    pw = sync_playwright().start()
    c = pw.firefox.launch_persistent_context(user_data_dir=profile, headless=headless, proxy=proxy)
    page = c.pages[0] if c.pages else c.new_page()
    _TB.update(page=page, ctx=c, pw=pw, backend="playwright-tor")
    _LAST_ERROR["error"] = None
    return page


def _elements(page, limit: int = 40) -> list[dict]:
    try:
        items = page.evaluate(_ELEMENTS_JS)
    except Exception:
        return []
    return items[:limit]


def _run(ctx: ToolContext, action: str, **kw) -> str:
    from ..harness.builtins.browser import (_clickables, _detect_challenge,
                                            _hint, _input_fields)

    if action == "goto":
        url = kw.get("url")
        if not url:
            return "[tor_browser error] 'goto' needs a 'url'."
        _bind_profile(ctx)
        try:
            page = _ensure_page(ctx)
        except Exception as e:
            _LAST_ERROR["error"] = str(e)
            return f"[tor_browser error] {e}"
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        try:
            # Longer timeout than the plain browser tool's -- Tor circuits
            # add real latency, especially on the first request of a session.
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            _LAST_ERROR["error"] = None
        except Exception as e:
            _LAST_ERROR["error"] = str(e)
            return f"[tor_browser error] could not open {url} via Tor: {e}"
        ch = _detect_challenge(page)
        note = (f"\n[!] Verification challenge detected ({ch}). Tor exit "
                "nodes trigger these often. I won't try to solve it — please "
                "complete it in the browser window, then tell me to "
                "continue.") if ch else ""
        return (f"Opened {page.url} via Tor (backend: {_TB['backend']}) — "
                f"title: {page.title()!r}{note}")

    page = _TB["page"]
    if page is None:
        return ("[tor_browser error] No page open yet. Call the 'goto' "
                "action with a url first — that starts the Tor-routed "
                "session.")

    try:
        if action == "read":
            ch = _detect_challenge(page)
            if ch:
                return (f"[verification wall: {ch}] This page is gated by a "
                        "human challenge. I won't bypass it — complete it in "
                        "the window, then ask me to continue.")
            try:
                text = page.inner_text("body")
            except Exception as e:
                return f"[tor_browser error] could not read the page: {e}"
            max_chars = int(kw.get("max_chars", 8000))
            head = f"# {page.title()}\n{page.url}\n\n"
            return head + (text[:max_chars] + ("..." if len(text) > max_chars else ""))

        if action == "elements":
            items = _elements(page)
            return _format_elements(items)

        if action == "click":
            ref = kw.get("ref")
            target = _ref_selector(ref) if ref else (kw.get("text") or kw.get("selector"))
            if not target:
                return "[tor_browser error] 'click' needs a 'ref', 'selector', or 'text'."
            try:
                if ref or kw.get("selector"):
                    page.click(target, timeout=8000)
                else:
                    try:
                        page.get_by_text(target, exact=False).first.click(timeout=8000)
                    except Exception:
                        page.click(target, timeout=8000)
                page.wait_for_load_state("domcontentloaded", timeout=20000)
            except Exception as e:
                hint = _hint("Clickable things I can see (retry with one of these):",
                             _clickables(page))
                return f"[tor_browser error] could not click {target!r}: {e}{hint}"
            return f"Clicked {target!r} — now at {page.url}"

        if action == "type":
            ref = kw.get("ref")
            selector = _ref_selector(ref) if ref else kw.get("selector")
            text = kw.get("text")
            if not selector or text is None:
                return "[tor_browser error] 'type' needs ('ref' or 'selector') and 'text'."
            try:
                page.fill(selector, text, timeout=8000)
            except Exception as css_exc:
                if ref:
                    hint = _hint("Input fields on this page (use one of these selectors):",
                                 _input_fields(page))
                    return f"[tor_browser error] could not fill ref {ref!r}: {css_exc}{hint}"
                try:
                    page.get_by_label(selector, exact=False).first.fill(text, timeout=8000)
                except Exception:
                    try:
                        page.get_by_placeholder(selector, exact=False).first.fill(text, timeout=8000)
                    except Exception:
                        hint = _hint("Input fields on this page (use one of these selectors):",
                                     _input_fields(page))
                        return f"[tor_browser error] could not fill {selector!r}: {css_exc}{hint}"
            if kw.get("submit"):
                try:
                    page.keyboard.press("Enter")
                    page.wait_for_load_state("domcontentloaded", timeout=20000)
                except Exception:
                    pass
            return (f"Filled {(ref and f'ref {ref!r}') or repr(selector)}"
                    + (" and pressed Enter" if kw.get("submit") else "")
                    + f" — now at {page.url}")

        if action == "screenshot":
            out_path = os.path.join(_profile_dir(ctx), "tor_screenshot.png")
            try:
                page.screenshot(path=out_path, full_page=False)
            except Exception as e:
                return f"[tor_browser error] could not take screenshot: {e}"
            return f"Screenshot saved to {out_path} (Tor-routed session, backend: {_TB['backend']})."
    except Exception as e:  # never let a bad page state crash the agent loop
        return f"[tor_browser error] {action} failed: {e}"

    return format_invalid_choice(
        "tor_browser", "action", action,
        ["goto", "click", "type", "read", "screenshot", "elements"],
    )


class TorBrowserTool(Tool):
    name = "tor_browser"
    description = (
        "Control a SEPARATE, real (headed) browser window whose traffic is "
        "routed through the Tor network via a local Tor SOCKS5 proxy — use "
        "this deliberately when you need a different exit IP than the "
        "character's ordinary connection, or need to reach an .onion "
        "service. This is NOT the same session as the regular 'browser' "
        "tool and does not share cookies/logins with it; the two can be "
        "open at the same time. A real OS window pops up (not the desktop "
        "app's live panel) so a human can see and step in if a site throws "
        "up a verification challenge, which Tor exit traffic triggers more "
        "often than a normal connection. Actions: 'goto' (open a url — "
        "ALWAYS call this first; starts the Tor-routed session if one "
        "isn't running yet, or navigates the existing page); 'elements' "
        "(lists clickable/fillable things on the page with a stable 'ref' "
        "id); 'click' (ref/selector/text); 'type' (ref or selector, plus "
        "text, plus optional submit=true); 'read' (page text, optionally "
        "scoped by selector); 'screenshot' (saves a PNG, reports its "
        "path). Requires the local `tor` daemon to be installed and "
        "running — if it isn't, this tool reports that clearly instead of "
        "silently browsing without anonymization."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["goto", "click", "type", "read", "screenshot", "elements"],
            },
            "url": {"type": "string", "description": "URL for 'goto' (http/https, or an .onion address)."},
            "ref": {
                "type": "string",
                "description": "Element ref from the 'elements' action — the most reliable way to target something for 'click'/'type'.",
            },
            "selector": {"type": "string", "description": "CSS selector, for 'click'/'type'."},
            "text": {
                "type": "string",
                "description": "For 'click': visible text to click by. For 'type': the text to type into the field.",
            },
            "submit": {"type": "boolean", "description": "For 'type': press Enter after typing.", "default": False},
        },
        "required": ["action"],
    }

    def run(self, ctx: ToolContext, action: str, **kw) -> str:
        action = (action or "").lower().strip()
        return _run(ctx, action, **kw)
