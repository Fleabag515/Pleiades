"""Chat-path browser tool.

When running inside the webui/desktop backend (the normal case for chat),
this drives the SAME per-character Playwright Chromium session the right
panel displays live (``pleiades/webui/browser_view.py``) -- headless by
default, watchable/interactible in the panel the instant the model calls it.
That's the unification the owner asked for: "the built-in browser the model
uses IN CHAT should be the same session shown in the panel."

When running anywhere else (the Discord bot, or any caller that never starts
the webui backend), ``browser_view.main_loop_available()`` is False and this
tool transparently falls back to the ORIGINAL behavior: delegating to
``harness/builtins/browser.py``'s Camoufox-first singleton, completely
unchanged, one browser window per process. That module and its other callers
(Discord bot, `pleiades work` harness jobs) are not modified by this file at
all -- see its own docstring.

Why a runtime feature-detect instead of two separate Tool classes wired up
by caller: `Engine` (pleiades/engine.py) is shared verbatim by both the webui
chat path and the Discord bot's connector, and both go through the same
`build_default_belt()` -> `BrowserTool`. Branching inside `run()` on whether
this process has a bound browser_view event loop gets the right behavior for
both callers without touching Engine, discord_bot.py, or the harness module
-- the safest way to satisfy "unify chat's browser with the panel" and "don't
break Discord/`pleiades work`" at the same time.
"""

from __future__ import annotations

import os

from . import Tool, ToolContext, format_invalid_choice, vision_caption

# ---------------------------------------------------------------------- #
# Fallback backend: harness/builtins/browser.py's Camoufox singleton.
# Used whenever this process has no browser_view event loop bound (i.e.
# it isn't the webui/desktop backend) -- unchanged from before this file's
# unification, so the Discord bot and any other non-webui caller keep
# exactly their previous behavior.
# ---------------------------------------------------------------------- #

_LAST_PROFILE: dict[str, str | None] = {"dir": None}


def _bind_camoufox(ctx: ToolContext) -> None:
    from ..harness.builtins import browser as hb

    target = ctx.profile.browser_dir
    if _LAST_PROFILE["dir"] not in (None, target):
        hb.browser_close()
    _LAST_PROFILE["dir"] = target
    hb.bind_browser_profile(target)


def _run_camoufox(ctx: ToolContext, action: str, **kw) -> str:
    from ..harness.builtins import browser as hb

    _bind_camoufox(ctx)
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
    return format_invalid_choice(
        "browser", "action", action, ["goto", "click", "type", "read", "screenshot"]
    )


# ---------------------------------------------------------------------- #
# Panel backend: pleiades/webui/browser_view.py's per-character Chromium
# session. Used whenever this process HAS a bound browser_view event loop
# (i.e. it is the webui/desktop backend) -- the normal case for chat.
# ---------------------------------------------------------------------- #

async def _panel_read(page, selector: str | None) -> str:
    try:
        if selector:
            text = await page.locator(selector).first.inner_text(timeout=8000)
        else:
            text = await page.inner_text("body")
    except Exception as e:
        return f"[browser error] could not read the page: {e}"
    title = await page.title()
    text = text or ""
    head = f"# {title}\n{page.url}\n\n"
    return head + (text[:8000] + ("..." if len(text) > 8000 else ""))


async def _panel_clickables(page, limit: int = 25) -> list[str]:
    try:
        items = await page.eval_on_selector_all(
            "a, button, [role=button], input[type=submit], [onclick]",
            "els => els.map(e => (e.innerText||e.value||e.getAttribute('aria-label')||'')"
            ".trim()).filter(t => t.length>0 && t.length<70)",
        )
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


async def _panel_input_fields(page, limit: int = 20) -> list[str]:
    try:
        items = await page.eval_on_selector_all(
            "input:not([type=hidden]), textarea, select, [role=textbox], "
            "[role=searchbox], [role=combobox], "
            "[contenteditable=''], [contenteditable='true'], "
            "[contenteditable='plaintext-only']",
            "els => els.map(e => {"
            " const id = e.id ? '#'+e.id : '';"
            " const nm = e.name ? e.tagName.toLowerCase()+'[name=\\\"'+e.name+'\\\"]' : '';"
            " const sel = id || nm || e.tagName.toLowerCase();"
            " const hint = e.placeholder || e.getAttribute('aria-label') || e.getAttribute('data-placeholder') || e.getAttribute('data-text') || e.type || '';"
            " return hint ? sel+'  — '+hint : sel; })",
        )
    except Exception:
        return []
    return [t for t in items if t][:limit]


def _hint(label: str, items: list[str]) -> str:
    return ("\n" + label + "\n  - " + "\n  - ".join(items)) if items else ""


# ---------------------------------------------------------------------- #
# Accessibility-tree extraction (Phase 1 of vision-grounded browser-use,
# docs/specs/2026-07-22-vision-grounded-browser-use-design.md). Gives ANY
# model (no vision required) a stable, semantic way to refer to an element
# -- "click ref e7" instead of guessing a CSS selector -- covering the same
# ground Claude in Chrome's own accessibility-tree-first design covers
# before it ever falls back to vision. This is additive: 'selector'/'text'
# targeting on click/type is unchanged, 'ref' is a new alternative.
#
# Refs are assigned by tagging the live DOM node with a data-pleiades-ref
# attribute the first time it's seen, so a ref stays valid for that node
# across calls as long as the node itself isn't removed/replaced -- no
# separate selector-generation algorithm to keep in sync with the page.
# ---------------------------------------------------------------------- #

_ELEMENTS_JS = r"""
() => {
  const SEL = "a, button, [role=button], [role=link], [role=checkbox], " +
    "[role=radio], [role=tab], [role=menuitem], [role=textbox], " +
    "[role=searchbox], [role=combobox], input:not([type=hidden]), " +
    "textarea, select, [contenteditable=''], [contenteditable='true'], " +
    "[contenteditable='plaintext-only'], [onclick], " +
    "[tabindex]:not([tabindex='-1'])";
  const els = Array.from(document.querySelectorAll(SEL));
  let n = 0;
  const out = [];
  for (const e of els) {
    const rect = e.getBoundingClientRect();
    const style = getComputedStyle(e);
    const visible = rect.width > 0 && rect.height > 0 &&
      style.visibility !== 'hidden' && style.display !== 'none';
    if (!visible) continue;
    let ref = e.getAttribute('data-pleiades-ref');
    if (!ref) {
      ref = 'e' + (++n) + '_' + Math.random().toString(36).slice(2, 6);
      e.setAttribute('data-pleiades-ref', ref);
    }
    const role = e.getAttribute('role') || e.tagName.toLowerCase();
    const tag = e.tagName.toLowerCase();
    const aria = e.getAttribute('aria-label') || '';
    const placeholder = e.getAttribute('placeholder') ||
      e.getAttribute('data-placeholder') || e.getAttribute('data-text') || '';
    const label = (e.innerText || e.value || aria || placeholder || '')
      .trim().replace(/\s+/g, ' ').slice(0, 80);
    // "editable" = something you can type text INTO (an input/textarea, a
    // contenteditable surface like Discord's Slate message box, or an
    // ARIA text/search/combobox). Surfaced so the model can tell a place it
    // can TYPE from a place it can only CLICK, and pick the RIGHT one when
    // several exist (a message composer vs. a search bar look alike here).
    const ce = e.getAttribute('contenteditable');
    const editableCE = ce === '' || ce === 'true' || ce === 'plaintext-only';
    const textInput = tag === 'textarea' ||
      (tag === 'input' && !['button','submit','checkbox','radio','range',
        'color','file','image','reset'].includes((e.getAttribute('type')||'text')
        .toLowerCase()));
    const editableRole = ['textbox','searchbox','combobox'].includes(role);
    out.push({
      ref, role, label,
      enabled: !e.disabled,
      tag,
      editable: !!(editableCE || textInput || editableRole),
      aria, placeholder,
    });
  }
  return out;
}
"""


async def _panel_elements(page, limit: int = 40) -> list[dict]:
    try:
        items = await page.evaluate(_ELEMENTS_JS)
    except Exception:
        return []
    return items[:limit]


def _format_elements(items: list[dict]) -> str:
    if not items:
        return "No interactive elements found on the current page."
    lines = ["Interactive elements on this page (use action='click'/'type' with ref=<ref>):"]
    for it in items:
        state = "" if it.get("enabled", True) else " [disabled]"
        kind = " [text field — you can type here]" if it.get("editable") else ""
        label = it.get("label") or "(no label)"
        lines.append(
            f"  - ref={it['ref']}  {it.get('role', it.get('tag', '?'))}: "
            f"{label!r}{kind}{state}"
        )
    editable = [it for it in items if it.get("editable")]
    if len(editable) > 1:
        opts = ", ".join(
            f"ref={it['ref']} ({it.get('label') or it.get('role') or '?'})"
            for it in editable
        )
        lines.append(
            "NOTE: this page has more than one place you can type: "
            + opts
            + ". These can look alike (e.g. a message/compose box vs. a search "
            "bar). Before you type, pick the ref whose label matches your actual "
            "goal — a message you want to SEND goes in the field labelled like "
            "'Message ...'/the composer, NOT the one labelled 'Search'. If none "
            "of the labels matches your goal, do NOT just type into the closest "
            "one — read the page again or say you couldn't find the right field."
        )
    return "\n".join(lines)


def _ref_selector(ref: str) -> str:
    return f'[data-pleiades-ref="{ref}"]'


async def _panel_click(page, target: str) -> str:
    try:
        try:
            await page.get_by_text(target, exact=False).first.click(timeout=8000)
        except Exception:
            await page.click(target, timeout=8000)
        await page.wait_for_load_state("domcontentloaded", timeout=15000)
    except Exception as e:
        hint = _hint(
            "Clickable things I can see (retry with one of these):",
            await _panel_clickables(page),
        )
        return f"[browser error] could not click {target!r}: {e}{hint}"
    return f"Clicked {target!r} — now at {page.url}"


# After a type(..., submit=true) we check what ACTUALLY happened before the
# model is allowed to report success. The reported failure mode was: the model
# typed a Discord message into the search bar, pressed Enter, and confidently
# told the user "sent" -- when nothing was sent at all. The single most
# discriminating, low-false-positive signal for a "send a message / submit"
# action is: did the field CLEAR? A real message composer (and most submit
# inputs) empties when its contents are actually submitted; a search box you
# pressed Enter in keeps its text and sends nothing. We report the observed
# state factually and, when the text is still sitting in the field, tell the
# model in no uncertain terms to VERIFY before claiming success.
_VERIFY_TYPED_JS = r"""
(args) => {
  const { selector, text } = args;
  const norm = s => (s || '').replace(/\s+/g, ' ').trim();
  const t = norm(text);
  const el = selector ? document.querySelector(selector) : null;
  const tag = el ? el.tagName.toLowerCase() : '';
  const elText = el
    ? norm((tag === 'input' || tag === 'textarea')
        ? (el.value || '')
        : (el.innerText || el.textContent || el.value || ''))
    : '';
  const inTarget = t.length > 0 && elText.indexOf(t) !== -1;
  let elsewhere = false;
  if (t.length > 0 && document.body) {
    const body = norm(document.body.innerText || '');
    const inBody = body.split(t).length - 1;
    const inTgt = elText.split(t).length - 1;
    elsewhere = inBody > inTgt;   // the text shows up somewhere OTHER than the field
  }
  return { targetFound: !!el, inTarget, elsewhere };
}
"""


async def _verify_typed(page, selector: str, text: str) -> "dict | None":
    if not selector:
        return None
    try:
        return await page.evaluate(_VERIFY_TYPED_JS, {"selector": selector, "text": text})
    except Exception:
        return None


def _submit_outcome(before_url: str, after_url: str, verify: "dict | None") -> str:
    if before_url != after_url:
        return (f" and pressed Enter — the page navigated to {after_url} "
                "(something was submitted).")
    if verify is None:
        return " and pressed Enter."
    if verify.get("inTarget"):
        return (
            " and pressed Enter, but the text is STILL sitting in that field — "
            "pressing Enter did not clear or send it, so it very likely did NOT "
            "go through. This is exactly what happens when you type into a search "
            "box or the wrong field instead of a real message/compose box. DO NOT "
            "tell the user it was sent. Verify first: call 'read' or 'elements' "
            "and confirm the text actually appears where you intended (e.g. as a "
            "new message in the conversation), and that the field you used was "
            "labelled like your goal (a composer, not 'Search'). If it isn't "
            "there, it did not send — find the right field and try again."
        )
    if verify.get("elsewhere"):
        return (" and pressed Enter — the field cleared and the text now appears "
                "elsewhere on the page, consistent with it having been sent/"
                "submitted. (Still worth a quick 'read' to confirm.)")
    return (" and pressed Enter — the field cleared (consistent with the text "
            "being submitted). Confirm with 'read' if this was a message you "
            "needed to actually send.")


async def _panel_type(page, selector: str, text: str, submit: bool) -> str:
    used_selector = selector
    try:
        await page.fill(selector, text, timeout=8000)
    except Exception as css_exc:
        try:
            await page.get_by_label(selector, exact=False).first.fill(text, timeout=8000)
            used_selector = None  # filled via label match, no stable selector to re-check
        except Exception:
            try:
                await page.get_by_placeholder(selector, exact=False).first.fill(text, timeout=8000)
                used_selector = None
            except Exception:
                hint = _hint(
                    "Input fields on this page (use one of these selectors):",
                    await _panel_input_fields(page),
                )
                return f"[browser error] could not fill {selector!r}: {css_exc}{hint}"
    if not submit:
        return f"Filled {selector!r} — now at {page.url}"
    before_url = page.url
    try:
        await page.keyboard.press("Enter")
        await page.wait_for_load_state("domcontentloaded", timeout=15000)
    except Exception:
        pass
    verify = await _verify_typed(page, used_selector, text)
    return f"Filled {selector!r}" + _submit_outcome(before_url, page.url, verify) + f" Now at {page.url}"


async def _panel_screenshot(page, out_path: str) -> str:
    await page.screenshot(path=out_path, full_page=False)
    return f"Screenshot saved to {out_path} (this is the same page visible live in the desktop panel)."


# ---------------------------------------------------------------------- #
# Thin vision-fallback captioning (docs/specs/2026-07-22-vision-grounded-
# browser-use-design.md, "Thin vision-fallback layer for text-only models").
# Deliberately separate from Phases 2-6's vision-escalation-for-clicking --
# this is NOT grounded to any clickable element and cannot itself resolve a
# click. It only runs when 'elements' finds nothing (the cheapest proxy this
# project has for "this is probably canvas/custom-rendered"), and only when
# PLEIADES_VISION_FALLBACK_URL points at a separately-run tiny captioner
# (see pleiades/tools/vision_caption.py) -- unconfigured, this is a total
# no-op and today's "No interactive elements found" message is unchanged.
# ---------------------------------------------------------------------- #

async def _panel_screenshot_bytes(page) -> "bytes | None":
    try:
        return await page.screenshot(full_page=False)
    except Exception:
        return None


async def _panel_vision_fallback_caption(page) -> "str | None":
    if not vision_caption.is_configured():
        return None
    png = await _panel_screenshot_bytes(page)
    if not png:
        return None
    return vision_caption.caption_screenshot(png)


def _with_vision_fallback(elements_text: str, caption: "str | None") -> str:
    if not caption:
        return elements_text
    return (
        elements_text
        + "\n\nVisual description (from a small, separate screenshot-captioning "
        "model -- NOT grounded to any clickable element, purely descriptive; see "
        "docs/specs/2026-07-22-vision-grounded-browser-use-design.md, 'Thin "
        "vision-fallback layer'): " + caption
    )


async def _panel_goto(session, url: str) -> str:
    if session.status == "running":
        await session.navigate(url)
    else:
        await session.start(url)
    return session.url or url


def _run_panel(ctx: ToolContext, action: str, **kw) -> str:
    from ..webui import browser_view

    session = browser_view.get_session(ctx.profile.name)

    if action == "goto":
        url = kw.get("url")
        if not url:
            return "[browser error] 'goto' needs a 'url'."
        try:
            final_url = browser_view.run_sync(_panel_goto(session, url))
        except Exception as e:
            return f"[browser error] could not open {url!r}: {e}"
        return f"Opened {final_url!r} — visible live in the desktop panel now."

    if session.status != "running" or session.page is None:
        return (
            "[browser error] No page open yet. Call the 'goto' action with a "
            "url first — that starts the panel's live browser session."
        )

    page = session.page
    try:
        if action == "read":
            return browser_view.run_sync(_panel_read(page, kw.get("selector")))
        if action == "elements":
            items = browser_view.run_sync(_panel_elements(page))
            out = _format_elements(items)
            if not items:
                caption = browser_view.run_sync(_panel_vision_fallback_caption(page))
                out = _with_vision_fallback(out, caption)
            return out
        if action == "click":
            # 'ref' (from the 'elements' action) is the most specific target --
            # prefer it over selector/text when given. Resolves to the same
            # _panel_click() used for selector/text, so the existing
            # hint-on-failure behavior is unchanged either way.
            ref = kw.get("ref")
            target = _ref_selector(ref) if ref else (kw.get("selector") or kw.get("text"))
            if not target:
                return "[browser error] 'click' needs a 'ref', 'selector', or 'text'."
            return browser_view.run_sync(_panel_click(page, target))
        if action == "type":
            ref = kw.get("ref")
            selector = _ref_selector(ref) if ref else kw.get("selector")
            text = kw.get("text")
            if not selector or text is None:
                return "[browser error] 'type' needs ('ref' or 'selector') and 'text'."
            return browser_view.run_sync(_panel_type(page, selector, text, bool(kw.get("submit"))))
        if action == "screenshot":
            out_path = os.path.join(ctx.profile.browser_dir, "panel_screenshot.png")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            return browser_view.run_sync(_panel_screenshot(page, out_path))
    except Exception as e:
        return f"[browser error] {action} failed: {e}"

    return format_invalid_choice(
        "browser", "action", action, ["goto", "click", "type", "read", "screenshot", "elements"]
    )


class BrowserTool(Tool):
    name = "browser"
    description = (
        "Control your live browser — the SAME session shown live in the desktop "
        "app's right-hand panel, so the user can watch and take over at any time. "
        "It starts hidden (no OS window pops up) the first time you use it. "
        "Actions: 'goto' (open a url — ALWAYS call this first, it starts the "
        "session if one isn't running yet, or navigates the existing page if one "
        "is); 'elements' (lists every clickable/fillable thing on the current "
        "page with a stable 'ref' id — call this when you're not sure what's on "
        "the page, then click/type using that ref instead of guessing a "
        "selector. If nothing is found and a small vision-fallback captioner is "
        "configured, a short plain-text description of the screenshot is appended too "
        "\u2014 informational only, not something you can click by ref); "
        "'click' (ref: an id from 'elements' \u2014 most reliable \u2014 OR "
        "selector: a CSS selector, OR text: visible link/button text); 'type' "
        "(ref or selector, plus text to fill a field, plus optional submit=true "
        "to press Enter afterward. When several text fields exist -- e.g. a "
        "message/compose box AND a search bar, which look alike -- pick the ref "
        "whose label matches your goal: a message to SEND belongs in the field "
        "labelled like 'Message ...', never the one labelled 'Search'. After "
        "submit=true the result tells you whether the text actually left the "
        "field; if it says the text is still sitting in the field it did NOT "
        "send, so verify before telling the user it worked); 'read' (optional selector to scope it, "
        "otherwise the whole visible page text); 'screenshot' (saves a PNG and "
        "reports its path). If a click or type target isn't found, the error "
        "message lists what's actually clickable or fillable on the current page "
        "so you can retry with something that exists instead of guessing again. "
        "The browser keeps your logins between calls."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["goto", "click", "type", "read", "screenshot", "elements"],
            },
            "url": {"type": "string", "description": "URL for 'goto'."},
            "ref": {
                "type": "string",
                "description": "Element ref from the 'elements' action. For 'click'/'type' — the most reliable way to target something, prefer this over selector/text when you've called 'elements'.",
            },
            "selector": {
                "type": "string",
                "description": "CSS selector. For 'click', visible text also works via the 'text' field instead.",
            },
            "text": {
                "type": "string",
                "description": "For 'click': visible text to click by (alternative to 'selector'/'ref'). For 'type': the text to type into the field.",
            },
            "submit": {"type": "boolean", "description": "For 'type': press Enter after typing.", "default": False},
        },
        "required": ["action"],
    }

    def run(self, ctx: ToolContext, action: str, **kw) -> str:
        action = (action or "").lower().strip()

        from ..webui import browser_view

        if browser_view.main_loop_available():
            return _run_panel(ctx, action, **kw)
        return _run_camoufox(ctx, action, **kw)
