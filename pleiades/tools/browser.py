"""Headed browser control — delegates to the harness browser module.

Single implementation: `harness/builtins/browser.py` already does Camoufox
(anti-fingerprint Firefox) with a Playwright fallback, persistent per-profile
cookies, CAPTCHA/challenge detection, and clickable/input-field hints. This
module just binds that shared implementation to the current character's
browser_dir and exposes it as a chat-path Tool — the same delegation
tools/search.py does against harness/builtins/web.py's web_search().

The harness module keeps ONE browser window per process (module-level state,
not per-character) — fine for its usual one-character-per-process callers
(Discord bot, `pleiades work`). The chat path can serve several characters
from one process (e.g. the webui), so `_bind()` force-closes the shared
window whenever the active character changes, ensuring two characters' pages
never overlap. True concurrent browser use by two characters at once isn't
supported here (it isn't really a coherent scenario for a single headed
window anyway) — ponytail: no locking added for that; revisit if a real
multi-character-at-once browser need shows up.

Requires the [browser] extra and a one-time `camoufox fetch` to download the
browser.
"""

from __future__ import annotations

import os

from . import Tool, ToolContext

_LAST_PROFILE: dict[str, str | None] = {"dir": None}


def _bind(ctx: ToolContext) -> None:
    from ..harness.builtins import browser as hb

    target = ctx.profile.browser_dir
    if _LAST_PROFILE["dir"] not in (None, target):
        hb.browser_close()
    _LAST_PROFILE["dir"] = target
    hb.bind_browser_profile(target)


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
        from ..harness.builtins import browser as hb

        _bind(ctx)
        action = action.lower().strip()
        if action == "goto":
            if not kw.get("url"):
                return "[browser error] 'goto' needs a 'url'."
            return hb.browser_open(kw["url"])
        if action == "click":
            target = kw.get("text") or kw.get("selector")
            if not target:
                return "[browser error] 'click' needs a 'selector' or 'text'."
            return hb.browser_click(target)
        if action == "type":
            if not kw.get("selector") or kw.get("text") is None:
                return "[browser error] 'type' needs 'selector' and 'text'."
            out = hb.browser_fill(kw["selector"], kw["text"])
            if kw.get("submit"):
                out = out + " " + hb.browser_press("Enter")
            return out
        if action == "read":
            return hb.browser_read()
        if action == "screenshot":
            return hb.browser_screenshot(os.path.join(ctx.profile.browser_dir, "screenshot.png"))
        return f"[browser error] unknown action '{action}'."
