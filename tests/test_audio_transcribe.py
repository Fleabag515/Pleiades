"""Tests for the whisper.cpp speech-fallback transcriber (audio-fallback
council review, 2026-07-23). Unconfigured (neither
PLEIADES_AUDIO_FALLBACK_ENABLED nor PLEIADES_AUDIO_FALLBACK_URL set) must be
a complete no-op -- no network call, no process spawned, no download --
exactly like vision_caption's no-op-when-unconfigured contract, since this
is an optional fallback layered under engine.py's audio-capable check, not
something any caller can assume is present.
"""

from __future__ import annotations

import json

from pleiades.tools import audio_transcribe as at


class _FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _clear_env(monkeypatch):
    for var in (
        "PLEIADES_AUDIO_FALLBACK_ENABLED",
        "PLEIADES_AUDIO_FALLBACK_URL",
        "PLEIADES_AUDIO_FALLBACK_MODEL",
        "PLEIADES_AUDIO_FALLBACK_HOST",
        "PLEIADES_AUDIO_FALLBACK_PORT",
        "PLEIADES_WHISPER_BIN",
    ):
        monkeypatch.delenv(var, raising=False)


def test_disabled_by_default_is_a_complete_noop(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    assert at.is_configured() is False

    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fake-wav-bytes")

    def _boom(*a, **kw):
        raise AssertionError("must not attempt a network call when unconfigured")

    monkeypatch.setattr(at.urllib.request, "urlopen", _boom)

    assert at.transcribe(str(audio)) is None


def test_is_configured_reflects_url_override(monkeypatch):
    _clear_env(monkeypatch)
    assert at.is_configured() is False

    monkeypatch.setenv("PLEIADES_AUDIO_FALLBACK_URL", "http://127.0.0.1:8791")
    assert at.is_configured() is True

    monkeypatch.setenv("PLEIADES_AUDIO_FALLBACK_URL", "   ")
    assert at.is_configured() is False


def test_is_configured_reflects_enabled_flag(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("PLEIADES_AUDIO_FALLBACK_ENABLED", "true")
    assert at.is_configured() is True

    monkeypatch.setenv("PLEIADES_AUDIO_FALLBACK_ENABLED", "0")
    assert at.is_configured() is False


def test_missing_audio_file_returns_none_without_network_call(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("PLEIADES_AUDIO_FALLBACK_URL", "http://127.0.0.1:8791")

    def _boom(*a, **kw):
        raise AssertionError("must not attempt a network call for a nonexistent file")

    monkeypatch.setattr(at.urllib.request, "urlopen", _boom)

    assert at.transcribe(str(tmp_path / "does-not-exist.wav")) is None


def test_successful_call_posts_multipart_and_returns_stripped_text(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("PLEIADES_AUDIO_FALLBACK_URL", "http://127.0.0.1:8791/")
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fake-wav-bytes")
    captured = {}

    def _fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["body"] = req.data
        captured["content_type"] = req.headers.get("Content-type", "")
        return _FakeResponse({"text": "  hello from the recording  "})

    monkeypatch.setattr(at.urllib.request, "urlopen", _fake_urlopen)

    out = at.transcribe(str(audio), timeout=5.0)

    assert out == "hello from the recording"
    # trailing slash on the configured base URL must not produce a double slash
    assert captured["url"] == "http://127.0.0.1:8791/inference"
    assert captured["timeout"] == 5.0
    assert b'name="file"; filename="clip.wav"' in captured["body"]
    assert b"fake-wav-bytes" in captured["body"]
    assert "multipart/form-data" in captured["content_type"]


def test_network_error_returns_none(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("PLEIADES_AUDIO_FALLBACK_URL", "http://127.0.0.1:8791")
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fake-wav-bytes")

    def _fake_urlopen(req, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(at.urllib.request, "urlopen", _fake_urlopen)

    assert at.transcribe(str(audio)) is None


def test_malformed_json_response_returns_none(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("PLEIADES_AUDIO_FALLBACK_URL", "http://127.0.0.1:8791")
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fake-wav-bytes")

    class _BadResponse:
        def read(self):
            return b"not json"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(at.urllib.request, "urlopen", lambda req, timeout=None: _BadResponse())

    assert at.transcribe(str(audio)) is None


def test_empty_transcript_returns_none(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("PLEIADES_AUDIO_FALLBACK_URL", "http://127.0.0.1:8791")
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fake-wav-bytes")

    monkeypatch.setattr(
        at.urllib.request, "urlopen",
        lambda req, timeout=None: _FakeResponse({"text": "   "}),
    )

    assert at.transcribe(str(audio)) is None


def test_model_path_uses_configured_model_name(monkeypatch):
    _clear_env(monkeypatch)
    assert at.model_path().name == "ggml-large-v3-turbo-q5_0.bin"

    monkeypatch.setenv("PLEIADES_AUDIO_FALLBACK_MODEL", "small")
    assert at.model_path().name == "ggml-small.bin"


def test_find_binary_prefers_explicit_override(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    fake_bin = tmp_path / "whisper-server"
    fake_bin.write_bytes(b"not a real binary")
    monkeypatch.setenv("PLEIADES_WHISPER_BIN", str(fake_bin))

    assert at.find_binary() == str(fake_bin)


def test_platform_asset_none_on_darwin(monkeypatch):
    monkeypatch.setattr(at.sys, "platform", "darwin")
    assert at._platform_asset() is None


def test_stop_is_noop_when_nothing_running(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setattr(at, "_load_running", lambda: {})
    assert at.stop() is False
