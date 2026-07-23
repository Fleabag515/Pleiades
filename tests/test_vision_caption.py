"""Tests for the thin vision-fallback captioner (docs/specs/2026-07-22-
vision-grounded-browser-use-design.md, "Thin vision-fallback layer for
text-only models"). Unconfigured (no PLEIADES_VISION_FALLBACK_URL) must be
a complete no-op -- no network call attempted -- since this is an optional
signal layered on top of Phase 1's accessibility-tree extraction, not
something any caller can assume is present.
"""

from __future__ import annotations

import base64
import json

from pleiades.tools import vision_caption as vc


class _FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_disabled_by_default_returns_none_without_network_call(monkeypatch):
    monkeypatch.delenv("PLEIADES_VISION_FALLBACK_URL", raising=False)

    def _boom(*a, **kw):
        raise AssertionError("must not attempt a network call when unconfigured")

    monkeypatch.setattr(vc.urllib.request, "urlopen", _boom)

    assert vc.caption_screenshot(b"fake-png-bytes") is None


def test_is_configured_reflects_env_var(monkeypatch):
    monkeypatch.delenv("PLEIADES_VISION_FALLBACK_URL", raising=False)
    assert vc.is_configured() is False

    monkeypatch.setenv("PLEIADES_VISION_FALLBACK_URL", "http://127.0.0.1:8790")
    assert vc.is_configured() is True

    monkeypatch.setenv("PLEIADES_VISION_FALLBACK_URL", "   ")
    assert vc.is_configured() is False


def test_no_image_bytes_returns_none_without_network_call(monkeypatch):
    monkeypatch.setenv("PLEIADES_VISION_FALLBACK_URL", "http://127.0.0.1:8790")

    def _boom(*a, **kw):
        raise AssertionError("must not attempt a network call with no image bytes")

    monkeypatch.setattr(vc.urllib.request, "urlopen", _boom)

    assert vc.caption_screenshot(b"") is None


def test_successful_call_posts_base64_image_and_returns_stripped_content(monkeypatch):
    monkeypatch.setenv("PLEIADES_VISION_FALLBACK_URL", "http://127.0.0.1:8790/")
    captured = {}

    def _fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse({"content": "  A blue rectangle with a white circle.  "})

    monkeypatch.setattr(vc.urllib.request, "urlopen", _fake_urlopen)

    out = vc.caption_screenshot(b"fake-png-bytes", timeout=3.0)

    assert out == "A blue rectangle with a white circle."
    # trailing slash on the configured base URL must not produce a double slash
    assert captured["url"] == "http://127.0.0.1:8790/completions"
    assert captured["timeout"] == 3.0
    assert captured["body"]["prompt"]["multimodal_data"] == [
        base64.b64encode(b"fake-png-bytes").decode("ascii")
    ]
    assert "<__media__>" in captured["body"]["prompt"]["prompt_string"]


def test_network_error_returns_none(monkeypatch):
    monkeypatch.setenv("PLEIADES_VISION_FALLBACK_URL", "http://127.0.0.1:8790")

    def _fake_urlopen(req, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(vc.urllib.request, "urlopen", _fake_urlopen)

    assert vc.caption_screenshot(b"fake-png-bytes") is None


def test_malformed_json_response_returns_none(monkeypatch):
    monkeypatch.setenv("PLEIADES_VISION_FALLBACK_URL", "http://127.0.0.1:8790")

    class _BadResponse:
        def read(self):
            return b"not json"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(vc.urllib.request, "urlopen", lambda req, timeout=None: _BadResponse())

    assert vc.caption_screenshot(b"fake-png-bytes") is None


def test_empty_response_content_returns_none(monkeypatch):
    monkeypatch.setenv("PLEIADES_VISION_FALLBACK_URL", "http://127.0.0.1:8790")
    monkeypatch.setattr(
        vc.urllib.request, "urlopen",
        lambda req, timeout=None: _FakeResponse({"content": "   "}),
    )

    assert vc.caption_screenshot(b"fake-png-bytes") is None


def test_custom_prompt_template_is_used_when_set(monkeypatch):
    monkeypatch.setenv("PLEIADES_VISION_FALLBACK_URL", "http://127.0.0.1:8790")
    monkeypatch.setenv("PLEIADES_VISION_FALLBACK_PROMPT", "CUSTOM<__media__>PROMPT")
    captured = {}

    def _fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse({"content": "ok"})

    monkeypatch.setattr(vc.urllib.request, "urlopen", _fake_urlopen)

    vc.caption_screenshot(b"fake-png-bytes")

    assert captured["body"]["prompt"]["prompt_string"] == "CUSTOM<__media__>PROMPT"
