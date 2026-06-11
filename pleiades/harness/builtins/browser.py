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


_BOUND_PROFILE: dict[str, str | None] = {"dir": None}


def bind_browser_profile(path: str) -> None:
    """Pin the persistent browser profile dir (e.g. a character's browser_dir),
    so the work-path browser shares one profile with the chat-path browser."""
    _BOUND_PROFILE["dir"] = path or None


def _profile_dir() -> str:
    p = (_BOUND_PROFILE["dir"]
         or os.environ.get("PLEIADES_BROWSER_PROFILE")
         or str(Path.cwd() / "browser_profile"))
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


def _clickables(page, limit: int = 25) -> list[str]:
    """Visible text of links/buttons on the page — so a missed click can retry
    against something that actually exists instead of guessing."""
    try:
        items = page.eval_on_selector_all(
            "a, button, [role=button], input[type=submit], [onclick]",
            "els => els.map(e => (e.innerText||e.value||e.getAttribute('aria-label')||'')"
            ".trim()).filter(t => t.length>0 && t.length<70)")
    except Exception:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for t in items:
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= limit:
            break
    return out


def _input_fields(page, limit: int = 20) -> list[str]:
    """Best-effort CSS selectors for the page's input fields, each with its
    placeholder/label — so browser_fill can be aimed precisely."""
    try:
        items = page.eval_on_selector_all(
            "input:not([type=hidden]), textarea, select",
            "els => els.map(e => {"
            " const id = e.id ? '#'+e.id : '';"
            " const nm = e.name ? e.tagName.toLowerCase()+'[name=\"'+e.name+'\"]' : '';"
            " const sel = id || nm || e.tagName.toLowerCase();"
            " const hint = e.placeholder || e.getAttribute('aria-label') || e.type || '';"
            " return hint ? sel+'  — '+hint : sel; })")
    except Exception:
        return []
    return [t for t in items if t][:limit]


def _hint(label: str, items: list[str]) -> str:
    return ("\n" + label + "\n  - " + "\n  - ".join(items)) if items else ""


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
        hint = _hint("Clickable things I can see (retry with one of these):",
                     _clickables(page))
        return f"Error clicking {target!r}: {e}{hint}"
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
        hint = _hint("Input fields I can see (retry with one of these selectors):",
                     _input_fields(page))
        return f"Error filling {selector!r}: {e}{hint}"
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


@tool(safe=True, tags=("browser", "web", "read"))
def browser_links(limit: int = 25) -> str:
    """List the clickable things on the current page (links/buttons by their visible text). Call this when you're unsure what to click or a click failed.

    limit: max number of items to list (default 25).
    """
    page = _B["page"]
    if page is None:
        return "No page open. Use browser_open first."
    items = _clickables(page, max(1, min(int(limit), 80)))
    if not items:
        return "No clickable elements found (page may still be loading or JS-rendered)."
    return "Clickable on this page:\n  - " + "\n  - ".join(items)


@tool(tags=("browser", "web"))
def browser_press(key: str = "Enter") -> str:
    """Press a keyboard key on the page — most useful as 'Enter' to submit a search box or form after browser_fill.

    key: e.g. 'Enter', 'Tab', 'Escape', 'PageDown'.
    """
    page = _B["page"]
    if page is None:
        return "No page open. Use browser_open first."
    try:
        page.keyboard.press(key)
        page.wait_for_load_state("domcontentloaded", timeout=15000)
    except Exception as e:
        return f"Error pressing {key!r}: {e}"
    return f"Pressed {key!r} — now at {page.url}"


@tool(tags=("browser", "web"))
def browser_back() -> str:
    """Go back to the previous page in this browser's history."""
    page = _B["page"]
    if page is None:
        return "No page open. Use browser_open first."
    try:
        page.go_back(wait_until="domcontentloaded", timeout=30000)
    except Exception as e:
        return f"Error going back: {e}"
    return f"Back to {page.url} — title: {page.title()!r}"


@tool(safe=True, tags=("browser", "web", "read"))
def browser_wait_for(selector: str, timeout_ms: int = 15000) -> str:
    """Wait for an element to appear — use this on JS-heavy pages where content loads after the initial response, before reading or clicking.

    selector: CSS selector to wait for.
    timeout_ms: how long to wait before giving up (default 15000).
    """
    page = _B["page"]
    if page is None:
        return "No page open. Use browser_open first."
    try:
        page.wait_for_selector(selector, timeout=max(1000, int(timeout_ms)))
    except Exception as e:
        return f"{selector!r} did not appear within {timeout_ms}ms: {e}"
    return f"{selector!r} is present."


@tool(tags=("browser", "web"))
def browser_scroll(amount: int = 1) -> str:
    """Scroll the page down (positive) or up (negative) by `amount` viewport-heights — needed to trigger lazy-loaded / infinite-scroll content before reading it.

    amount: number of viewport-heights to scroll (default 1; negative scrolls up).
    """
    page = _B["page"]
    if page is None:
        return "No page open. Use browser_open first."
    try:
        page.evaluate("n => window.scrollBy(0, n * window.innerHeight)", int(amount))
        page.wait_for_timeout(600)
    except Exception as e:
        return f"Error scrolling: {e}"
    return f"Scrolled {amount} viewport-height(s); page is now scrolled to y={page.evaluate('window.scrollY')}."


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
