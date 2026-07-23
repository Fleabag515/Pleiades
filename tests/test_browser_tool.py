"""Phase 1 of vision-grounded browser-use (docs/specs/2026-07-22-vision-
grounded-browser-use-design.md): accessibility-tree extraction + stable-ref
based click/type, additive to the existing selector/text targeting."""

from __future__ import annotations

import asyncio
import types

from pleiades.tools import browser as bt
from pleiades.webui import browser_view


class FakePage:
    """Duck-types just enough of a Playwright Page for these tests."""

    def __init__(self, elements=None, screenshot_bytes=b"fake-png-bytes"):
        self._elements = elements if elements is not None else []
        self.clicked = []
        self.filled = []
        self.screenshot_calls = 0
        self._screenshot_bytes = screenshot_bytes

    async def evaluate(self, _js):
        return self._elements

    async def screenshot(self, **kw):
        self.screenshot_calls += 1
        return self._screenshot_bytes

    async def click(self, selector, timeout=8000):
        if selector not in ('[data-pleiades-ref="e1_ab12"]', "#real-button"):
            raise Exception(f"no element matches {selector!r}")
        self.clicked.append(selector)

    def get_by_text(self, text, exact=False):
        # Real Playwright returns a chainable Locator synchronously here;
        # this stub isn't exercised by these tests (click() falls straight
        # to the CSS-selector branch below), it just needs to fail the same
        # way a real call would if given garbage.
        raise Exception("not used in these tests")

    async def fill(self, selector, text, timeout=8000):
        if selector not in ('[data-pleiades-ref="e2_cd34"]', "#real-input"):
            raise Exception(f"no element matches {selector!r}")
        self.filled.append((selector, text))

    async def wait_for_load_state(self, *a, **kw):
        return None

    @property
    def url(self):
        return "https://example.test/page"


# ─── pure helpers ────────────────────────────────────────────────────────────


def test_ref_selector_builds_attribute_selector():
    assert bt._ref_selector("e7_ab12") == '[data-pleiades-ref="e7_ab12"]'


def test_format_elements_empty():
    assert "No interactive elements" in bt._format_elements([])


def test_format_elements_lists_ref_role_label_and_disabled_state():
    items = [
        {"ref": "e1_ab12", "role": "button", "label": "Sign in", "enabled": True, "tag": "button"},
        {"ref": "e2_cd34", "role": "textbox", "label": "Search", "enabled": False, "tag": "input"},
    ]
    out = bt._format_elements(items)
    assert "ref=e1_ab12" in out
    assert "Sign in" in out
    assert "ref=e2_cd34" in out
    assert "[disabled]" in out


def test_panel_elements_returns_items_and_respects_limit():
    items = [{"ref": f"e{i}", "role": "button", "label": str(i), "enabled": True, "tag": "button"} for i in range(5)]
    page = FakePage(elements=items)
    got = asyncio.run(bt._panel_elements(page, limit=3))
    assert len(got) == 3
    assert got[0]["ref"] == "e0"


def test_panel_elements_returns_empty_list_on_evaluate_failure():
    class BrokenPage(FakePage):
        async def evaluate(self, _js):
            raise Exception("page navigated away mid-evaluate")

    got = asyncio.run(bt._panel_elements(BrokenPage()))
    assert got == []


# ─── _run_panel dispatch (ref-based click/type + 'elements' action) ─────────


class FakeSession:
    def __init__(self, page):
        self.status = "running"
        self.page = page


def _patch_browser_view(monkeypatch, page):
    monkeypatch.setattr(browser_view, "get_session", lambda name: FakeSession(page))
    monkeypatch.setattr(browser_view, "run_sync", lambda coro, timeout=45.0: asyncio.run(coro))
    monkeypatch.setattr(browser_view, "main_loop_available", lambda: True)


def _ctx():
    return types.SimpleNamespace(profile=types.SimpleNamespace(name="mark", browser_dir="/tmp"))


def test_run_panel_elements_action_lists_page_elements(monkeypatch):
    items = [{"ref": "e1_ab12", "role": "button", "label": "Sign in", "enabled": True, "tag": "button"}]
    page = FakePage(elements=items)
    _patch_browser_view(monkeypatch, page)

    out = bt._run_panel(_ctx(), "elements")
    assert "ref=e1_ab12" in out
    assert "Sign in" in out


def test_run_panel_click_prefers_ref_over_selector_and_text(monkeypatch):
    page = FakePage()
    _patch_browser_view(monkeypatch, page)

    out = bt._run_panel(_ctx(), "click", ref="e1_ab12", selector="#some-other-thing", text="ignored too")
    assert "Clicked" in out
    assert page.clicked == ['[data-pleiades-ref="e1_ab12"]']


def test_run_panel_click_falls_back_to_selector_when_no_ref(monkeypatch):
    page = FakePage()
    _patch_browser_view(monkeypatch, page)

    out = bt._run_panel(_ctx(), "click", selector="#real-button")
    assert "Clicked" in out
    assert page.clicked == ["#real-button"]


def test_run_panel_type_prefers_ref_over_selector(monkeypatch):
    page = FakePage()
    _patch_browser_view(monkeypatch, page)

    out = bt._run_panel(_ctx(), "type", ref="e2_cd34", selector="#some-other-input", text="hello")
    assert "Filled" in out
    assert page.filled == [('[data-pleiades-ref="e2_cd34"]', "hello")]


def test_run_panel_click_requires_ref_selector_or_text(monkeypatch):
    page = FakePage()
    _patch_browser_view(monkeypatch, page)

    out = bt._run_panel(_ctx(), "click")
    assert "[browser error]" in out
    assert "ref" in out


# ─── Thin vision-fallback captioning hook (docs/specs/2026-07-22-vision- ───
# grounded-browser-use-design.md, "Thin vision-fallback layer for
# text-only models") -- only fires when 'elements' finds nothing AND a
# fallback captioner is configured; a total no-op otherwise.


def test_run_panel_elements_nonempty_never_triggers_vision_fallback(monkeypatch):
    items = [{"ref": "e1_ab12", "role": "button", "label": "Sign in", "enabled": True, "tag": "button"}]
    page = FakePage(elements=items)
    _patch_browser_view(monkeypatch, page)

    def _boom(_png):
        raise AssertionError("must not caption a screenshot when elements were found")

    monkeypatch.setattr(bt.vision_caption, "caption_screenshot", _boom)

    out = bt._run_panel(_ctx(), "elements")

    assert "Sign in" in out
    assert page.screenshot_calls == 0


def test_run_panel_elements_empty_unconfigured_leaves_message_unchanged(monkeypatch):
    page = FakePage(elements=[])
    _patch_browser_view(monkeypatch, page)
    monkeypatch.setattr(bt.vision_caption, "is_configured", lambda: False)

    def _boom(_png):
        raise AssertionError("must not caption when the fallback isn't configured")

    monkeypatch.setattr(bt.vision_caption, "caption_screenshot", _boom)

    out = bt._run_panel(_ctx(), "elements")

    assert out == "No interactive elements found on the current page."
    assert page.screenshot_calls == 0


def test_run_panel_elements_empty_caption_returns_none_leaves_message_unchanged(monkeypatch):
    page = FakePage(elements=[])
    _patch_browser_view(monkeypatch, page)
    monkeypatch.setattr(bt.vision_caption, "is_configured", lambda: True)
    monkeypatch.setattr(bt.vision_caption, "caption_screenshot", lambda _png: None)

    out = bt._run_panel(_ctx(), "elements")

    assert out == "No interactive elements found on the current page."
    assert page.screenshot_calls == 1


def test_run_panel_elements_empty_configured_appends_caption(monkeypatch):
    page = FakePage(elements=[])
    _patch_browser_view(monkeypatch, page)
    monkeypatch.setattr(bt.vision_caption, "is_configured", lambda: True)
    monkeypatch.setattr(
        bt.vision_caption, "caption_screenshot",
        lambda _png: "A blue rectangle with a white circle.",
    )

    out = bt._run_panel(_ctx(), "elements")

    assert "No interactive elements found" in out
    assert "A blue rectangle with a white circle." in out
    assert "NOT grounded to any clickable element" in out
    assert page.screenshot_calls == 1


def test_panel_screenshot_bytes_returns_none_on_failure():
    class BrokenPage(FakePage):
        async def screenshot(self, **kw):
            raise Exception("page closed mid-screenshot")

    got = asyncio.run(bt._panel_screenshot_bytes(BrokenPage()))
    assert got is None


def test_with_vision_fallback_passthrough_when_no_caption():
    assert bt._with_vision_fallback("original text", None) == "original text"
    assert bt._with_vision_fallback("original text", "") == "original text"


# ─── Discord-class fix: composer visibility + false-success verification ─────
# (reported bug: model typed a Discord message into the SEARCH bar, pressed
# Enter, and told the user "sent" when nothing was. Root cause was two-fold:
# the message composer -- a role=textbox/contenteditable Slate div -- was
# invisible to element extraction, and nothing verified a submit landed.)


def test_format_elements_marks_editable_fields():
    items = [
        {"ref": "e1", "role": "button", "label": "Send", "enabled": True, "editable": False},
        {"ref": "e2", "role": "textbox", "label": "Message #general", "enabled": True, "editable": True},
    ]
    out = bt._format_elements(items)
    assert "text field — you can type here" in out
    # single editable field -> no disambiguation note
    assert "more than one place you can type" not in out


def test_format_elements_warns_when_multiple_text_fields_exist():
    items = [
        {"ref": "e1", "role": "input", "label": "Search", "enabled": True, "editable": True},
        {"ref": "e2", "role": "textbox", "label": "Message #general", "enabled": True, "editable": True},
    ]
    out = bt._format_elements(items)
    assert "more than one place you can type" in out
    # both candidate fields are named so the model can pick by label
    assert "Search" in out and "Message #general" in out
    assert "NOT the one labelled 'Search'" in out


def test_submit_outcome_navigation_counts_as_submitted():
    out = bt._submit_outcome("https://a.test/x", "https://a.test/y", None)
    assert "navigated" in out
    assert "submitted" in out


def test_submit_outcome_text_still_in_field_warns_not_sent():
    verify = {"targetFound": True, "inTarget": True, "elsewhere": False}
    out = bt._submit_outcome("https://a.test/x", "https://a.test/x", verify)
    assert "STILL sitting in that field" in out
    assert "did NOT" in out
    assert "search box" in out  # names the exact reported failure mode


def test_submit_outcome_field_cleared_and_text_elsewhere_reads_as_sent():
    verify = {"targetFound": True, "inTarget": False, "elsewhere": True}
    out = bt._submit_outcome("https://a.test/x", "https://a.test/x", verify)
    assert "cleared" in out
    assert "sent" in out or "submitted" in out


def test_submit_outcome_no_verify_info_is_neutral():
    out = bt._submit_outcome("https://a.test/x", "https://a.test/x", None)
    assert out == " and pressed Enter."


def test_verify_typed_returns_none_without_selector():
    assert asyncio.run(bt._verify_typed(FakePage(), None, "hi")) is None


def test_verify_typed_returns_none_on_evaluate_failure():
    class BrokenPage(FakePage):
        async def evaluate(self, _js, _arg=None):
            raise Exception("page gone")

    assert asyncio.run(bt._verify_typed(BrokenPage(), "#x", "hi")) is None
