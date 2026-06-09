"""Headed browser control via Camoufox (anti-fingerprint Firefox).

Each character gets a persistent browser context at ~/.pleiades/profiles/<name>/browser
so logins and cookies survive across sessions. The browser is headed (headless=False)
so it runs visibly alongside the user.

Requires the [browser] extra and a one-time `camoufox fetch` to download the browser.
The session is kept alive across tool calls within a turn (see ToolContext.browser()).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import Tool, ToolContext
from ..config import browser_dir


class BrowserSession:
    """A lazily-started, persistent, headed Camoufox session for one character."""

    def __init__(self, profile, headless: bool = False):
        self.profile = profile
        self.headless = headless
        self._cm = None       # the Camoufox context manager
        self._browser = None  # BrowserContext (persistent) or Browser
        self._page = None

    def _ensure_page(self):
        if self._page is not None:
            return self._page
        try:
            from camoufox.sync_api import Camoufox
        except ImportError as e:
            raise RuntimeError(
                "Camoufox is not installed. Install the [browser] extra and run `camoufox fetch`."
            ) from e

        user_data = browser_dir(self.profile.name)
        user_data.mkdir(parents=True, exist_ok=True)

        self._cm = Camoufox(
            headless=self.headless,
            persistent_context=True,
            user_data_dir=str(user_data),
        )
        self._browser = self._cm.__enter__()
        # Persistent contexts expose .pages / .new_page directly.
        pages = getattr(self._browser, "pages", None)
        if pages:
            self._page = pages[0]
        else:
            self._page = self._browser.new_page()
        return self._page

    # -- actions ------------------------------------------------------------ #
    def goto(self, url: str) -> str:
        page = self._ensure_page()
        page.goto(url, wait_until="domcontentloaded")
        return f"Loaded {page.url} — title: {page.title()}"

    def click(self, selector: Optional[str] = None, text: Optional[str] = None) -> str:
        page = self._ensure_page()
        if text and not selector:
            page.get_by_text(text, exact=False).first.click()
            return f"Clicked element with text: {text!r}"
        if selector:
            page.click(selector)
            return f"Clicked: {selector}"
        return "[browser error] click needs a selector or text."

    def type(self, selector: str, text: str, submit: bool = False) -> str:
        page = self._ensure_page()
        page.fill(selector, text)
        if submit:
            page.press(selector, "Enter")
        return f"Typed into {selector}" + (" and submitted." if submit else ".")

    def read(self, selector: Optional[str] = None, max_chars: int = 4000) -> str:
        page = self._ensure_page()
        if selector:
            el = page.query_selector(selector)
            text = el.inner_text() if el else ""
        else:
            text = page.inner_text("body")
        text = " ".join(text.split())
        return text[:max_chars]

    def screenshot(self, path: Optional[str] = None) -> str:
        page = self._ensure_page()
        out = path or str(browser_dir(self.profile.name) / "screenshot.png")
        page.screenshot(path=out, full_page=True)
        return f"Saved screenshot to {out}"

    def close(self) -> None:
        if self._cm is not None:
            try:
                self._cm.__exit__(None, None, None)
            finally:
                self._cm = None
                self._browser = None
                self._page = None


class BrowserTool(Tool):
    name = "browser"
    description = (
        "Control a real web browser. Actions: 'goto' (open a URL), 'click' (by CSS "
        "selector or visible text), 'type' (fill a field, optionally submit), 'read' "
        "(get visible page text, optionally for a selector), 'screenshot'. The browser "
        "keeps your logins between calls."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["goto", "click", "type", "read", "screenshot"],
            },
            "url": {"type": "string", "description": "URL for 'goto'."},
            "selector": {"type": "string", "description": "CSS selector."},
            "text": {"type": "string", "description": "Text to click by, or to type."},
            "submit": {"type": "boolean", "description": "Press Enter after typing.", "default": False},
        },
        "required": ["action"],
    }

    def run(self, ctx: ToolContext, action: str, **kw) -> str:
        session = ctx.browser()
        action = action.lower().strip()
        if action == "goto":
            if not kw.get("url"):
                return "[browser error] 'goto' needs a 'url'."
            return session.goto(kw["url"])
        if action == "click":
            return session.click(selector=kw.get("selector"), text=kw.get("text"))
        if action == "type":
            if not kw.get("selector") or kw.get("text") is None:
                return "[browser error] 'type' needs 'selector' and 'text'."
            return session.type(kw["selector"], kw["text"], submit=bool(kw.get("submit")))
        if action == "read":
            return session.read(selector=kw.get("selector"))
        if action == "screenshot":
            return session.screenshot()
        return f"[browser error] unknown action '{action}'."
