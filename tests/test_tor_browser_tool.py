"""Unit tests for the Tor-routed browser tool (pleiades/tools/tor_browser.py),
following the fake-Playwright-page conventions of test_browser_tool.py. These
never launch a real browser or require a running Tor daemon -- _ensure_page
and the SOCKS reachability check are monkeypatched/stubbed everywhere except
the tests that specifically target that reachability logic."""

from __future__ import annotations

import socket
import types

import pytest

from pleiades.tools import tor_browser as tb
import pleiades.harness.builtins.browser as hb


class FakePage:
    """Duck-types just enough of a sync Playwright Page for these tests."""

    def __init__(self, elements=None, url="https://example.test/page", title="Example"):
        self._elements = elements if elements is not None else []
        self._url = url
        self._title = title
        self.clicked = []
        self.filled = []
        self.screenshots = []

    def evaluate(self, _js):
        return self._elements

    def goto(self, url, wait_until="domcontentloaded", timeout=60000):
        # Real Playwright normalizes/redirects; this fake just simulates a
        # successful navigation without changing the constructor-supplied
        # self._url, so tests can assert on a specific final URL.
        return None

    def click(self, selector, timeout=8000):
        if selector not in ('[data-pleiades-ref="e1_ab12"]', "#real-button"):
            raise Exception(f"no element matches {selector!r}")
        self.clicked.append(selector)

    def get_by_text(self, text, exact=False):
        raise Exception("not used in these tests")

    def fill(self, selector, text, timeout=8000):
        if selector not in ('[data-pleiades-ref="e2_cd34"]', "#real-input"):
            raise Exception(f"no element matches {selector!r}")
        self.filled.append((selector, text))

    def wait_for_load_state(self, *a, **kw):
        return None

    def inner_text(self, _sel):
        return "Hello from the Tor-routed page."

    def title(self):
        return self._title

    def screenshot(self, path, full_page=False):
        self.screenshots.append(path)

    @property
    def url(self):
        return self._url

    @property
    def keyboard(self):
        return types.SimpleNamespace(press=lambda key: None)


def _ctx(tmp_path):
    return types.SimpleNamespace(
        profile=types.SimpleNamespace(name="mark", tor_browser_dir=str(tmp_path))
    )


def _reset_singleton(monkeypatch, page=None, backend="camoufox-tor"):
    monkeypatch.setitem(tb._TB, "page", page)
    monkeypatch.setitem(tb._TB, "backend", backend if page is not None else None)


# ─── pure helpers ────────────────────────────────────────────────────────────


def test_tor_proxy_server_defaults_to_local_tor_socks_port(monkeypatch):
    monkeypatch.delenv("PLEIADES_TOR_SOCKS_PROXY", raising=False)
    assert tb.tor_proxy_server() == "socks5://127.0.0.1:9050"


def test_tor_proxy_server_respects_env_override(monkeypatch):
    monkeypatch.setenv("PLEIADES_TOR_SOCKS_PROXY", "socks5://127.0.0.1:9150")
    assert tb.tor_proxy_server() == "socks5://127.0.0.1:9150"


def test_socks_host_port_parses_default():
    assert tb.socks_host_port("socks5://127.0.0.1:9050") == ("127.0.0.1", 9050)


def test_socks_host_port_parses_custom_host_and_port():
    assert tb.socks_host_port("socks5://10.0.0.5:9150") == ("10.0.0.5", 9150)


def test_socks_host_port_falls_back_to_defaults_on_missing_pieces():
    assert tb.socks_host_port("socks5://") == ("127.0.0.1", 9050)


def test_tor_socks_reachable_true_when_connect_succeeds(monkeypatch):
    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(socket, "create_connection", lambda *a, **kw: FakeSocket())
    assert tb.tor_socks_reachable() is True


def test_tor_socks_reachable_false_when_connect_fails(monkeypatch):
    def _boom(*a, **kw):
        raise OSError("connection refused")

    monkeypatch.setattr(socket, "create_connection", _boom)
    assert tb.tor_socks_reachable() is False


# ─── goto / proxy-unreachable error path ────────────────────────────────────


def test_goto_needs_a_url(tmp_path):
    out = tb._run(_ctx(tmp_path), "goto")
    assert "[tor_browser error]" in out
    assert "url" in out


def test_goto_reports_clear_error_when_tor_not_reachable(monkeypatch, tmp_path):
    _reset_singleton(monkeypatch, page=None)
    monkeypatch.setattr(tb, "tor_socks_reachable", lambda timeout=1.5: False)

    out = tb._run(_ctx(tmp_path), "goto", url="https://example.test")
    assert "[tor_browser error]" in out
    assert "tor" in out.lower()
    assert "9050" in out


def test_goto_opens_page_and_reports_title_and_backend(monkeypatch, tmp_path):
    page = FakePage(url="https://example.test/", title="Example Domain")
    monkeypatch.setattr(tb, "_ensure_page", lambda ctx: page)
    monkeypatch.setattr(hb, "_detect_challenge", lambda _page: None)

    out = tb._run(_ctx(tmp_path), "goto", url="example.test")
    assert "Opened https://example.test/" in out
    assert "Example Domain" in out


def test_goto_surfaces_challenge_wall_note(monkeypatch, tmp_path):
    page = FakePage()
    monkeypatch.setattr(tb, "_ensure_page", lambda ctx: page)
    monkeypatch.setattr(hb, "_detect_challenge", lambda _page: "recaptcha")

    out = tb._run(_ctx(tmp_path), "goto", url="example.test")
    assert "Verification challenge detected (recaptcha)" in out
    assert "won't try to solve it" in out


# ─── no-page-open guard for every non-goto action ───────────────────────────


def test_actions_other_than_goto_require_an_open_page(monkeypatch, tmp_path):
    _reset_singleton(monkeypatch, page=None)
    for action in ("read", "elements", "click", "type", "screenshot"):
        out = tb._run(_ctx(tmp_path), action)
        assert "[tor_browser error]" in out
        assert "goto" in out


# ─── elements / click / type / read / screenshot dispatch ──────────────────


def test_elements_action_lists_page_elements(monkeypatch, tmp_path):
    items = [{"ref": "e1_ab12", "role": "button", "label": "Sign in", "enabled": True, "tag": "button"}]
    page = FakePage(elements=items)
    _reset_singleton(monkeypatch, page=page)

    out = tb._run(_ctx(tmp_path), "elements")
    assert "ref=e1_ab12" in out
    assert "Sign in" in out


def test_click_prefers_ref_over_selector_and_text(monkeypatch, tmp_path):
    page = FakePage()
    _reset_singleton(monkeypatch, page=page)

    out = tb._run(_ctx(tmp_path), "click", ref="e1_ab12", selector="#some-other-thing", text="ignored too")
    assert "Clicked" in out
    assert page.clicked == ['[data-pleiades-ref="e1_ab12"]']


def test_click_falls_back_to_selector_when_no_ref(monkeypatch, tmp_path):
    page = FakePage()
    _reset_singleton(monkeypatch, page=page)

    out = tb._run(_ctx(tmp_path), "click", selector="#real-button")
    assert "Clicked" in out
    assert page.clicked == ["#real-button"]


def test_click_requires_ref_selector_or_text(monkeypatch, tmp_path):
    page = FakePage()
    _reset_singleton(monkeypatch, page=page)

    out = tb._run(_ctx(tmp_path), "click")
    assert "[tor_browser error]" in out
    assert "ref" in out


def test_type_prefers_ref_over_selector(monkeypatch, tmp_path):
    page = FakePage()
    _reset_singleton(monkeypatch, page=page)

    out = tb._run(_ctx(tmp_path), "type", ref="e2_cd34", selector="#some-other-input", text="hello")
    assert "Filled" in out
    assert page.filled == [('[data-pleiades-ref="e2_cd34"]', "hello")]


def test_type_falls_back_to_selector_when_no_ref(monkeypatch, tmp_path):
    page = FakePage()
    _reset_singleton(monkeypatch, page=page)

    out = tb._run(_ctx(tmp_path), "type", selector="#real-input", text="hi")
    assert "Filled" in out
    assert page.filled == [("#real-input", "hi")]


def test_read_returns_page_text(monkeypatch, tmp_path):
    page = FakePage()
    _reset_singleton(monkeypatch, page=page)
    monkeypatch.setattr(hb, "_detect_challenge", lambda _page: None)

    out = tb._run(_ctx(tmp_path), "read")
    assert "Hello from the Tor-routed page." in out
    assert page.url in out


def test_read_returns_challenge_wall_message(monkeypatch, tmp_path):
    page = FakePage()
    _reset_singleton(monkeypatch, page=page)
    monkeypatch.setattr(hb, "_detect_challenge", lambda _page: "hcaptcha")

    out = tb._run(_ctx(tmp_path), "read")
    assert "[verification wall: hcaptcha]" in out


def test_screenshot_action_saves_png(monkeypatch, tmp_path):
    page = FakePage()
    _reset_singleton(monkeypatch, page=page)

    out = tb._run(_ctx(tmp_path), "screenshot")
    assert "Screenshot saved to" in out
    assert len(page.screenshots) == 1
    assert page.screenshots[0].endswith("tor_screenshot.png")


def test_invalid_action_lists_valid_choices(monkeypatch, tmp_path):
    page = FakePage()
    _reset_singleton(monkeypatch, page=page)

    out = tb._run(_ctx(tmp_path), "not-a-real-action")
    assert "[tor_browser error]" in out
    assert "goto" in out and "elements" in out


# ─── TorBrowserTool.run() glue ──────────────────────────────────────────────


def test_tool_run_lowercases_and_strips_action(monkeypatch, tmp_path):
    page = FakePage(elements=[])
    _reset_singleton(monkeypatch, page=page)

    tool = tb.TorBrowserTool()
    out = tool.run(_ctx(tmp_path), action="  ELEMENTS  ")
    assert "No interactive elements" in out


def test_tool_schema_has_expected_name_and_actions():
    tool = tb.TorBrowserTool()
    schema = tool.schema()
    assert schema["function"]["name"] == "tor_browser"
    assert set(schema["function"]["parameters"]["properties"]["action"]["enum"]) == {
        "goto", "click", "type", "read", "screenshot", "elements",
    }
