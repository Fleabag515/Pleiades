"""
browser.py — a real browser for the agent. Camoufox-first, Playwright fallback.

The "best of both worlds" that's actually buildable:

  * REALISM  — launches Camoufox (an anti-fingerprint Firefox) when available, so
    pages render like a human's browser and the agent trips far fewer false
    "are you a bot?" flags. Falls back to plain Playwright Firefox if Camoufox
    isn't installed.
  * PERSISTENCE — a persistent user-data profile, so cookies and logins are SAVED
    across runs. The agent stays signed in instead of re-authenticating every time.

Headful by default (PLEIADES_BROWSER_HEADLESS=1 to hide) so YOU can watch and,
crucially, step in at a real challenge.

The one line we hold: this does NOT solve CAPTCHAs or auto-defeat verification.
When `browser_read` detects a challenge wall, it says so and hands off to you —
solving human-verification challenges is your call, made in the visible window,
not the agent's to punch through.

Optional dependency. The core never imports these; tools degrade with a clear
install hint if neither library is present:
    pip install playwright && python -m playwright install firefox
    pip install camoufox[geoip]   # optional, for the stealth Firefox
"""

from __future__ import annotations

import os
from pathlib import Path

from ..tools import tool

# module-level singleton browser state (one window, reused across tool calls)
_B: dict[str, object] = {"page": None, "ctx": None, "pw": None, "backend": None}

_CHALLENGE_MARKERS = (
    "recaptcha", "hcaptcha", "g-recaptcha", "cf-challenge", "challenge-platform",
    "verify you are human", "are you a robot", "press and hold",
    "checking your browser", "/cdn-cgi/challenge",
)


def _profile_dir() -> str:
    p = os.environ.get("PLEIADES_BROWSER_PROFILE") or str(
        Path.cwd() / "browser_profile")
    Path(p).mkdir(parents=True, exist_ok=True)
    return p


def _headless() -> bool:
    return os.environ.get("PLEIADES_BROWSER_HEADLESS", "0") == "1"


def _ensure_page():
    """Lazily launch the browser (Camoufox if present, else Playwright Firefox)
    with a persistent profile, and return a Page. Raises with an install hint."""
    if _B["page"] is not None:
        return _B["page"]

    profile, headless = _profile_dir(), _headless()

    # 1) Camoufox (stealth Firefox) — preferred
    try:
        from camoufox.sync_api import Camoufox
        cam = Camoufox(headless=headless, persistent_context=True,
                       user_data_dir=profile)
        ctx = cam.start() if hasattr(cam, "start") else cam.__enter__()
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        _B.update(page=page, ctx=ctx, pw=cam, backend="camoufox")
        return page
    except ImportError:
        pass  # fall through to Playwright

    # 2) Playwright Firefox with a persistent context
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise RuntimeError(
            "No browser backend installed. Install one:\n"
            "  pip install playwright && python -m playwright install firefox\n"
            "  (optional stealth) pip install camoufox[geoip]")
    pw = sync_playwright().start()
    ctx = pw.firefox.launch_persistent_context(user_data_dir=profile,
                                               headless=headless)
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    _B.update(page=page, ctx=ctx, pw=pw, backend="playwright")
    return page


def _detect_challenge(page) -> str | None:
    try:
        url = (page.url or "").lower()
        html = (page.content() or "").lower()
    except Exception:
        return None
    for m in _CHALLENGE_MARKERS:
        if m in url or m in html:
            return m
    return None


# ---- tools ----------------------------------------------------------------
@tool(tags=("browser", "web"))
def browser_open(url: str) -> str:
    """Open a URL in the persistent browser (cookies/logins are saved across runs).

    url: the page to navigate to (http/https).
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        page = _ensure_page()
        page.goto(url, wait_until="domcontentloaded", timeout=45000)
    except Exception as e:
        return f"Error opening {url}: {e}"
    ch = _detect_challenge(page)
    note = (f"\n[!] Verification challenge detected ({ch}). I won't try to solve "
            "it — please complete it in the browser window, then tell me to "
            "continue.") if ch else ""
    return f"Opened {page.url} — title: {page.title()!r}{note}"


@tool(safe=True, tags=("browser", "web", "read"))
def browser_read(max_chars: int = 8000) -> str:
    """Return the readable text of the current page. If a human-verification challenge is present, says so and hands off (does not attempt to bypass it).

    max_chars: cap on returned characters.
    """
    page = _B["page"]
    if page is None:
        return "No page open. Use browser_open first."
    ch = _detect_challenge(page)
    if ch:
        return (f"[verification wall: {ch}] This page is gated by a human "
                "challenge. I won't bypass it — complete it in the window, then "
                "ask me to continue.")
    try:
        text = page.inner_text("body")
    except Exception as e:
        return f"Error reading page: {e}"
    head = f"# {page.title()}\n{page.url}\n\n"
    return head + (text[:max_chars] + ("..." if len(text) > max_chars else ""))


@tool(tags=("browser", "web"))
def browser_click(target: str) -> str:
    """Click an element by visible text or CSS selector.

    target: a CSS selector, or visible link/button text (tried as text= first).
    """
    page = _B["page"]
    if page is None:
        return "No page open. Use browser_open first."
    try:
        try:
            page.get_by_text(target, exact=False).first.click(timeout=8000)
        except Exception:
            page.click(target, timeout=8000)
        page.wait_for_load_state("domcontentloaded", timeout=15000)
    except Exception as e:
        return f"Error clicking {target!r}: {e}"
    return f"Clicked {target!r} — now at {page.url}"


@tool(tags=("browser", "web"))
def browser_fill(selector: str, value: str) -> str:
    """Type a value into an input field. For passwords or payment fields, stop and let the user fill them in the visible window instead.

    selector: CSS selector of the input.
    value: text to type.
    """
    page = _B["page"]
    if page is None:
        return "No page open. Use browser_open first."
    try:
        page.fill(selector, value, timeout=8000)
    except Exception as e:
        return f"Error filling {selector!r}: {e}"
    return f"Filled {selector!r}."


@tool(safe=True, tags=("browser", "web", "read"))
def browser_screenshot(path: str = "screenshot.png") -> str:
    """Save a screenshot of the current page (useful to show the user a challenge or result).

    path: output PNG path.
    """
    page = _B["page"]
    if page is None:
        return "No page open. Use browser_open first."
    try:
        page.screenshot(path=path, full_page=False)
    except Exception as e:
        return f"Error: {e}"
    return f"Saved screenshot to {path} (backend: {_B['backend']})"


@tool(tags=("browser",))
def browser_close() -> str:
    """Close the browser window (the saved profile/cookies persist)."""
    try:
        if _B["ctx"] is not None:
            _B["ctx"].close()
        if _B["backend"] == "playwright" and _B["pw"] is not None:
            _B["pw"].stop()
    except Exception:
        pass
    _B.update(page=None, ctx=None, pw=None, backend=None)
    return "Browser closed; profile saved."
