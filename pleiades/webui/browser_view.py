"""
browser_view.py — a genuinely live, embeddable browser view for the desktop
app's right panel.

This is deliberately a SEPARATE capability from ``harness/builtins/browser.py``
(the Camoufox-first stealth browser the agent itself drives autonomously,
used elsewhere including the Discord bot). That module stays untouched. This
one exists purely so a human, looking at the desktop app's right panel, can
watch (and optionally drive) a real browser window live.

Why Playwright + Chromium here, specifically, and not Camoufox/Firefox: the
embedding trick this needs is Chrome DevTools Protocol's screencast feature
(``Page.startScreencast`` / ``Page.screencastFrame`` events), which streams
actual rendered frames of the page over a CDP session as they're painted —
Firefox (and therefore Camoufox) has no equivalent in CDP/BiDi today. Chromium
is the only backend that can do this.

Architecture
------------
One ``BrowserViewSession`` per character, created lazily and kept in the
module-level ``_SESSIONS`` registry. Each session:

  * Launches a headed, persistent-context Chromium via Playwright's *async*
    API (``playwright.async_api``), so it shares the same asyncio event loop
    FastAPI/uvicorn already run on — no extra thread, no thread-safety
    headaches bridging callbacks back into async code.
  * Profile dir: ``<profile_dir>/browser_chromium`` — a sibling of (never the
    same as) the Camoufox tool's ``<profile_dir>/browser`` dir, so the two
    tools' persistent state never collides.
  * Opens a CDP session against its one page (``context.new_cdp_session(page)``
    — the exact API Playwright's Python docs expose for raw protocol access)
    and issues ``Page.startScreencast``. Each ``Page.screencastFrame`` event
    carries one base64 JPEG frame; we fan it out to every subscribed
    WebSocket client (each gets its own small drop-old queue, so a slow
    client sees the *latest* frame rather than a growing backlog) and ack it
    with ``Page.screencastFrameAck`` (required, or Chromium stops sending
    more).
  * Accepts input back via ``Input.dispatchMouseEvent`` / ``dispatchKeyEvent``
    / ``insertText`` — real CDP input dispatch, not Playwright's higher-level
    ``page.mouse``/``page.keyboard`` (which do extra actionability/visibility
    waiting that fights a raw pixel-coordinate click coming from a canvas).

This module owns no FastAPI routes itself — ``server.py`` wires small REST
endpoints (start/stop/status/navigate) plus one WebSocket endpoint
(frames out, input events in) around it.
"""

from __future__ import annotations

import asyncio
import base64
import os
import time
from pathlib import Path
from typing import Any, Optional

from .. import config

# ----------------------------------------------------------------------- #
# helpers
# ----------------------------------------------------------------------- #

def _ensure_display() -> None:
    """Same reasoning as harness/builtins/browser.py's helper of the same
    name: headed Chromium needs an X display, and the webui backend is
    sometimes started from a process that doesn't inherit the desktop
    session's DISPLAY. Default to :0 (single-seat machine); override with
    PLEIADES_DISPLAY."""
    if os.environ.get("DISPLAY"):
        return
    os.environ["DISPLAY"] = os.environ.get("PLEIADES_DISPLAY", ":0")


def _ensure_browsers_path() -> None:
    """Pin PLAYWRIGHT_BROWSERS_PATH to the real shared install location
    (~/.cache/ms-playwright). Needed specifically for the PyInstaller-
    bundled backend: collect_all("playwright") in backend.spec copies the
    whole playwright package tree (including an empty driver/.../
    .local-browsers/ dir from the build machine's venv), and the mere
    presence of that directory flips Playwright's Node-side driver into
    "local browsers" mode, which then looks for Chromium colocated with
    the bundled driver instead of the shared cache where it's actually
    installed -- "Executable doesn't exist at .../.local-browsers/...".
    Setting this env var explicitly overrides that heuristic. A no-op for
    the normal dev-venv run (same effective path either way)."""
    if "PLAYWRIGHT_BROWSERS_PATH" not in os.environ:
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(Path.home() / ".cache" / "ms-playwright")


def chromium_profile_dir(name: str) -> str:
    """Sibling of config.browser_dir(name) ('.../browser') — never the same
    directory, so this tool's persistent Chromium profile can never collide
    with the Camoufox/Playwright-Firefox profile the agent's own browser
    tool uses."""
    d = Path(config.profile_dir(name)) / "browser_chromium"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


_DEFAULT_VIEWPORT = {"width": 1280, "height": 800}


class BrowserViewSession:
    """One character's live, embeddable Chromium session."""

    def __init__(self, character: str):
        self.character = character
        self.viewport: dict[str, int] = dict(_DEFAULT_VIEWPORT)
        self.status: str = "stopped"     # stopped | starting | running | error
        self.error: Optional[str] = None
        self.url: Optional[str] = None
        self.interactive = True          # bidirectional input is wired up (see snapshot())

        self._playwright = None
        self._context = None
        self._page = None
        self._cdp = None
        self._lock = asyncio.Lock()
        self._subscribers: "set[asyncio.Queue]" = set()
        self._last_frame: Optional[bytes] = None

    # -- lifecycle --------------------------------------------------------
    async def start(self, start_url: str = "") -> None:
        async with self._lock:
            if self.status in ("running", "starting"):
                return
            self.status = "starting"
            self.error = None
            try:
                _ensure_display()
                _ensure_browsers_path()
                from playwright.async_api import async_playwright

                profile_dir = chromium_profile_dir(self.character)
                self._playwright = await async_playwright().start()
                self._context = await self._playwright.chromium.launch_persistent_context(
                    user_data_dir=profile_dir,
                    headless=False,
                    viewport=self.viewport,
                    args=["--window-position=0,0"],
                )
                self._page = (
                    self._context.pages[0] if self._context.pages else await self._context.new_page()
                )
                self._cdp = await self._context.new_cdp_session(self._page)
                await self._cdp.send("Page.enable")
                self._cdp.on("Page.screencastFrame", self._on_frame)
                await self._cdp.send(
                    "Page.startScreencast",
                    {
                        "format": "jpeg",
                        "quality": 60,
                        "maxWidth": self.viewport["width"],
                        "maxHeight": self.viewport["height"],
                        "everyNthFrame": 1,
                    },
                )
                if start_url:
                    if not start_url.startswith(("http://", "https://")):
                        start_url = "https://" + start_url
                    await self._page.goto(start_url, wait_until="domcontentloaded", timeout=45000)
                self.url = self._page.url
                self.status = "running"
            except Exception as e:  # surface to the UI, never raise into uvicorn
                self.status = "error"
                self.error = str(e)
                await self._cleanup()
                raise

    async def stop(self) -> None:
        async with self._lock:
            await self._cleanup()
            self.status = "stopped"
            self.url = None

    async def _cleanup(self) -> None:
        cdp, self._cdp = self._cdp, None
        if cdp is not None:
            try:
                await cdp.send("Page.stopScreencast")
            except Exception:
                pass
            try:
                await cdp.detach()
            except Exception:
                pass
        ctx, self._context = self._context, None
        self._page = None
        if ctx is not None:
            try:
                await ctx.close()
            except Exception:
                pass
        pw, self._playwright = self._playwright, None
        if pw is not None:
            try:
                await pw.stop()
            except Exception:
                pass
        # Wake any subscribers with a sentinel so their websocket loop exits
        # cleanly instead of hanging on an empty queue forever.
        for q in list(self._subscribers):
            try:
                q.put_nowait(None)
            except Exception:
                pass

    async def navigate(self, url: str) -> str:
        if self._page is None:
            raise RuntimeError("Browser session is not running. Start it first.")
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        await self._page.goto(url, wait_until="domcontentloaded", timeout=45000)
        self.url = self._page.url
        return self.url

    # -- frames -------------------------------------------------------------
    def _on_frame(self, params: dict) -> None:
        # Sync callback (pyee emitter, invoked on this same running loop).
        # Ack'ing and fan-out both need `await`, so hand off to a task.
        asyncio.get_running_loop().create_task(self._handle_frame(params))

    async def _handle_frame(self, params: dict) -> None:
        data = params.get("data")
        session_id = params.get("sessionId")
        if data:
            frame = base64.b64decode(data)
            self._last_frame = frame
            dead = []
            for q in self._subscribers:
                try:
                    if q.full():
                        try:
                            q.get_nowait()
                        except Exception:
                            pass
                    q.put_nowait(frame)
                except Exception:
                    dead.append(q)
            for q in dead:
                self._subscribers.discard(q)
        if session_id is not None and self._cdp is not None:
            try:
                await self._cdp.send("Page.screencastFrameAck", {"sessionId": session_id})
            except Exception:
                pass

    def subscribe(self) -> "asyncio.Queue":
        q: "asyncio.Queue" = asyncio.Queue(maxsize=2)
        if self._last_frame is not None:
            try:
                q.put_nowait(self._last_frame)
            except Exception:
                pass
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: "asyncio.Queue") -> None:
        self._subscribers.discard(q)

    # -- input ---------------------------------------------------------------
    async def dispatch_input(self, evt: dict) -> None:
        """Forward one input event from the panel into the real page via raw
        CDP Input.* commands (not page.mouse/page.keyboard — those add
        actionability waits that fight a raw pixel click from a canvas)."""
        if self._cdp is None:
            return
        kind = evt.get("kind")
        try:
            if kind == "mouse":
                params: dict[str, Any] = {
                    "type": evt.get("type", "mouseMoved"),
                    "x": float(evt.get("x", 0)),
                    "y": float(evt.get("y", 0)),
                    "button": evt.get("button", "left"),
                    "clickCount": int(evt.get("clickCount", 1)),
                }
                if evt.get("type") == "mouseWheel":
                    params["deltaX"] = float(evt.get("deltaX", 0))
                    params["deltaY"] = float(evt.get("deltaY", 0))
                await self._cdp.send("Input.dispatchMouseEvent", params)
            elif kind == "key":
                params = {
                    "type": evt.get("type", "keyDown"),
                    "key": evt.get("key", ""),
                    "code": evt.get("code", ""),
                }
                if evt.get("text"):
                    params["text"] = evt["text"]
                    params["unmodifiedText"] = evt["text"]
                if evt.get("keyCode") is not None:
                    params["windowsVirtualKeyCode"] = int(evt["keyCode"])
                    params["nativeVirtualKeyCode"] = int(evt["keyCode"])
                await self._cdp.send("Input.dispatchKeyEvent", params)
            elif kind == "text":
                # Bulk text insertion (e.g. paste) — sidesteps per-keystroke
                # keycode mapping entirely for plain typed content.
                await self._cdp.send("Input.insertText", {"text": evt.get("text", "")})
        except Exception:
            # Never let a bad input event kill the session.
            pass

    # -- reporting ------------------------------------------------------------
    def snapshot(self) -> dict:
        return {
            "character": self.character,
            "status": self.status,
            "error": self.error,
            "url": self.url,
            "viewport": self.viewport,
            "interactive": self.interactive,
            "backend": "playwright-chromium",
        }


_SESSIONS: dict[str, BrowserViewSession] = {}


def get_session(name: str) -> BrowserViewSession:
    s = _SESSIONS.get(name)
    if s is None:
        s = BrowserViewSession(name)
        _SESSIONS[name] = s
    return s


async def shutdown_all() -> None:
    """Close every live session — called from the FastAPI shutdown hook so a
    normal app quit (SIGTERM -> uvicorn graceful shutdown) never leaves an
    orphaned headed Chromium process behind."""
    for s in list(_SESSIONS.values()):
        try:
            await s.stop()
        except Exception:
            pass
