"""
browser_view.py — a genuinely live, embeddable browser view for the desktop
app's right panel.

This is deliberately a SEPARATE capability from ``harness/builtins/browser.py``
(the Camoufox-first stealth browser the agent itself drove autonomously
before the chat path was unified with this module -- see ``tools/browser.py``
for that unification and why the harness module itself stays untouched,
still used as-is by the Discord bot and ``pleiades work`` jobs).

Why Playwright + Chromium here, specifically, and not Camoufox/Firefox: the
embedding trick this needs is Chrome DevTools Protocol's screencast feature
(``Page.startScreencast`` / ``Page.screencastFrame`` events), which streams
actual rendered frames of the page over a CDP session as they're painted —
Firefox (and therefore Camoufox) has no equivalent in CDP/BiDi today. Chromium
is the only backend that can do this.

Headless by default (owner feedback: a real OS browser window popping open
unasked is not wanted). Headless Chromium still runs the full rendering
pipeline and still emits ``Page.screencastFrame`` events over CDP, so the
panel's live view works identically either way -- headless just means there's
no on-screen window. ``BrowserViewSession.open_separate()`` is the explicit,
human-triggered escape hatch: it tears down and relaunches the SAME
persistent-profile context headed (a real OS window), so logins/cookies
carry over exactly, and continues streaming frames to the panel at the same
time. (Chosen over "open a plain system browser" because that would be a
different profile/session entirely -- the user would lose the very state,
and the very "same session as chat" property, this whole feature is for.)

Architecture
------------
One ``BrowserViewSession`` per character, created lazily and kept in the
module-level ``_SESSIONS`` registry. Each session:

  * Launches a persistent-context Chromium via Playwright's *async* API
    (``playwright.async_api``), so it shares the same asyncio event loop
    FastAPI/uvicorn already run on — no extra thread, no thread-safety
    headaches bridging callbacks back into async code. Headless unless
    ``open_separate()`` has been called (see ``self.headed``).
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
    That's for the panel's own pointer/keyboard forwarding; the chat-path
    tool (``tools/browser.py``) instead drives ``self.page`` with normal
    Playwright page-level calls (click/fill/etc.), which is fine for that
    caller and still shows up in the screencast either way (screencast
    captures paints, regardless of what triggered them).

This module owns no FastAPI routes itself — ``server.py`` wires small REST
endpoints (start/stop/status/navigate/resize/open-separate) plus one
WebSocket endpoint (frames out, input events in) around it. ``server.py``
also calls ``set_main_loop()`` once at startup so ``tools/browser.py`` (the
chat-path model tool, which runs synchronously on a per-turn worker thread —
see engine.py) can bridge into this module's async sessions via ``run_sync``.
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
    PLEIADES_DISPLAY. Not needed for the default headless path, but still
    called unconditionally so ``open_separate()`` (which DOES need a
    display) never has to remember to call it."""
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
_EXPANDED_VIEWPORT = {"width": 1600, "height": 1000}

# ----------------------------------------------------------------------- #
# sync/async bridge for tools/browser.py (the chat-path model tool)
# ----------------------------------------------------------------------- #

_MAIN_LOOP: Optional[asyncio.AbstractEventLoop] = None


def set_main_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Called once from server.py's FastAPI startup hook. A chat turn (and
    therefore tools/browser.py's BrowserTool.run) executes synchronously on
    its own worker thread, not on this module's asyncio loop -- run_sync()
    is the one bridge point that lets that sync call drive these async
    sessions. Deliberately left unset in any process that never starts the
    webui/desktop backend (the Discord bot, `pleiades work` jobs): those
    call main_loop_available() and get False, and fall back to the
    harness's Camoufox browser unchanged -- this module never affects them."""
    global _MAIN_LOOP
    _MAIN_LOOP = loop


def main_loop_available() -> bool:
    return _MAIN_LOOP is not None and _MAIN_LOOP.is_running()


def run_sync(coro: Any, timeout: float = 45.0) -> Any:
    """Run a browser_view coroutine from a synchronous caller (tools/browser.py),
    blocking that thread for the result. Raises plainly if no loop has been
    bound yet (main_loop_available() should be checked first by callers that
    want a graceful fallback instead of an exception)."""
    if not main_loop_available():
        raise RuntimeError(
            "browser_view has no running event loop bound -- not running "
            "inside the webui/desktop backend process."
        )
    fut = asyncio.run_coroutine_threadsafe(coro, _MAIN_LOOP)
    return fut.result(timeout=timeout)


class BrowserViewSession:
    """One character's live, embeddable Chromium session."""

    def __init__(self, character: str):
        self.character = character
        self.viewport: dict[str, int] = dict(_DEFAULT_VIEWPORT)
        self.status: str = "stopped"     # stopped | starting | running | error
        self.error: Optional[str] = None
        self.url: Optional[str] = None
        self.interactive = True          # bidirectional input is wired up (see snapshot())
        self.headed = False              # True after open_separate() -- a real OS window

        self._playwright = None
        self._context = None
        self._page = None
        self._cdp = None
        self._lock = asyncio.Lock()
        self._subscribers: "set[asyncio.Queue]" = set()
        self._last_frame: Optional[bytes] = None

    @property
    def page(self):
        """Public accessor for tools/browser.py -- the chat-path tool drives
        this Playwright Page directly with normal page-level calls
        (click/fill/inner_text/...); it's still the exact same page the
        panel's screencast is watching."""
        return self._page

    # -- lifecycle --------------------------------------------------------
    async def start(self, start_url: str = "") -> None:
        async with self._lock:
            if self.status in ("running", "starting"):
                return
            self.status = "starting"
            self.error = None
            try:
                await self._launch(headless=True, start_url=start_url)
                self.status = "running"
            except Exception as e:  # surface to the UI, never raise into uvicorn
                self.status = "error"
                self.error = str(e)
                await self._cleanup()
                raise

    async def open_separate(self) -> None:
        """Switch this running session to a real headed OS window, in place.
        Playwright can't flip headless<->headed on a live browser (it's a
        launch-time option), so this tears down and relaunches the same
        persistent profile dir at the same URL -- cookies/logins carry over,
        and the CDP screencast (and therefore the panel's live view) keeps
        working exactly as before, now alongside a real visible window."""
        async with self._lock:
            if self.status != "running":
                raise RuntimeError("Start the browser session before opening it separately.")
            current_url = self.url
            await self._cleanup()
            self.status = "starting"
            try:
                await self._launch(headless=False, start_url=current_url or "")
                self.status = "running"
            except Exception as e:
                self.status = "error"
                self.error = str(e)
                await self._cleanup()
                raise

    async def resize(self, width: int, height: int) -> None:
        """Resize the page viewport (used by the panel's expand/fullscreen
        view to request a bigger, sharper frame instead of just upscaling a
        blurry small one) and restart the screencast at the new dimensions."""
        async with self._lock:
            if self._page is None:
                raise RuntimeError("Browser session is not running.")
            self.viewport = {"width": max(320, int(width)), "height": max(240, int(height))}
            await self._page.set_viewport_size(self.viewport)
            await self._restart_screencast()

    async def _launch(self, headless: bool, start_url: str = "") -> None:
        _ensure_display()
        _ensure_browsers_path()
        from playwright.async_api import async_playwright

        profile_dir = chromium_profile_dir(self.character)
        self._playwright = await async_playwright().start()
        launch_kwargs: dict[str, Any] = dict(
            user_data_dir=profile_dir,
            headless=headless,
            viewport=self.viewport,
        )
        if not headless:
            launch_kwargs["args"] = ["--window-position=0,0"]
        self._context = await self._playwright.chromium.launch_persistent_context(**launch_kwargs)
        self._page = (
            self._context.pages[0] if self._context.pages else await self._context.new_page()
        )
        self._cdp = await self._context.new_cdp_session(self._page)
        await self._cdp.send("Page.enable")
        self._cdp.on("Page.screencastFrame", self._on_frame)
        await self._start_screencast()
        if start_url:
            if not start_url.startswith(("http://", "https://")):
                start_url = "https://" + start_url
            await self._page.goto(start_url, wait_until="domcontentloaded", timeout=45000)
        self.url = self._page.url
        self.headed = not headless

    async def _start_screencast(self) -> None:
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

    async def _restart_screencast(self) -> None:
        try:
            await self._cdp.send("Page.stopScreencast")
        except Exception:
            pass
        await self._start_screencast()

    async def stop(self) -> None:
        async with self._lock:
            await self._cleanup()
            self.status = "stopped"
            self.url = None
            self.headed = False

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
            "headed": self.headed,
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
    orphaned headed OR headless Chromium process behind."""
    for s in list(_SESSIONS.values()):
        try:
            await s.stop()
        except Exception:
            pass
