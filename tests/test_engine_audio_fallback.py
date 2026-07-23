"""Engine-side wiring for the audio-fallback transcription note (2026-07-23
audio-fallback design decision). No real whisper.cpp server or model is
involved here -- audio_transcribe.transcribe is monkeypatched -- this only
checks the routing: no attachment -> no note; audio-capable model -> no
note (fallback skipped); not-audio-capable model + configured fallback ->
note injected; unconfigured fallback -> no note.
"""

from __future__ import annotations

from pleiades.engine import Engine
from pleiades.models import ModelManager
from pleiades.profiles import Profile
from pleiades.tools import audio_transcribe


def test_no_note_when_no_audio_path():
    assert Engine._audio_fallback_note(Profile(name="char1"), None) is None


def test_no_note_when_model_is_audio_capable(monkeypatch, tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF-fake")
    mm = ModelManager()
    mm.remove("audiomodel")
    mm.add("audiomodel", str(gguf))
    reg = mm._load()
    reg["audiomodel"]["audio_capable"] = True
    mm._save(reg)

    monkeypatch.setenv("PLEIADES_AUDIO_FALLBACK_URL", "http://127.0.0.1:8791")

    def _boom(*a, **kw):
        raise AssertionError("must not call the fallback transcriber for an audio-capable model")

    monkeypatch.setattr(audio_transcribe, "transcribe", _boom)

    profile = Profile(name="char1", model="audiomodel")
    assert Engine._model_audio_capable(profile) is True
    assert Engine._audio_fallback_note(profile, "/tmp/whatever.wav") is None
    mm.remove("audiomodel")


def test_model_defaults_to_not_audio_capable(tmp_path):
    gguf = tmp_path / "m2.gguf"
    gguf.write_bytes(b"GGUF-fake")
    mm = ModelManager()
    mm.remove("plainmodel")
    mm.add("plainmodel", str(gguf))

    profile = Profile(name="char1", model="plainmodel")
    assert Engine._model_audio_capable(profile) is False
    mm.remove("plainmodel")


def test_no_note_when_fallback_not_configured(monkeypatch):
    monkeypatch.delenv("PLEIADES_AUDIO_FALLBACK_URL", raising=False)
    monkeypatch.delenv("PLEIADES_AUDIO_FALLBACK_ENABLED", raising=False)
    profile = Profile(name="char1", model="")
    assert Engine._audio_fallback_note(profile, "/tmp/whatever.wav") is None


def test_note_injected_when_configured_and_transcript_available(monkeypatch):
    monkeypatch.setenv("PLEIADES_AUDIO_FALLBACK_URL", "http://127.0.0.1:8791")
    monkeypatch.setattr(audio_transcribe, "transcribe", lambda path, **kw: "hello world")

    profile = Profile(name="char1", model="")
    note = Engine._audio_fallback_note(profile, "/tmp/whatever.wav")

    assert note is not None
    assert "hello world" in note
    assert "cannot take audio input directly" in note
    assert "not a description of music" in note


def test_no_note_when_transcription_fails(monkeypatch):
    monkeypatch.setenv("PLEIADES_AUDIO_FALLBACK_URL", "http://127.0.0.1:8791")
    monkeypatch.setattr(audio_transcribe, "transcribe", lambda path, **kw: None)

    profile = Profile(name="char1", model="")
    assert Engine._audio_fallback_note(profile, "/tmp/whatever.wav") is None
