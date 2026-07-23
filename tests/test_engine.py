"""_resolve_upstream's legacy-model-path fallback (no real llama server launched)."""

from pleiades import config
from pleiades.engine import Engine
from pleiades.models import ModelManager
from pleiades.profiles import Profile


def test_resolve_upstream_registers_model_path_via_modelmanager(tmp_path, monkeypatch):
    """PLEIADES_MODEL_PATH should now flow through ModelManager (autofit/native-runtime
    treatment) instead of the old dumb InferenceServer fallback."""
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF-fake")
    mm = ModelManager()
    mm.remove("default")  # leftover from a previous run, if any

    s = config.Settings(model_path=str(gguf))
    eng = Engine(settings=s, manager=object(), anamnesis=object())

    # Stub start() — we're checking *which path* is taken, not actually booting llama.cpp.
    started = {}
    monkeypatch.setattr(ModelManager, "start", lambda self, name, **kw: started.setdefault("name", name))

    upstream = eng._resolve_upstream(Profile(name="char1"))

    assert started["name"] == "default"
    assert mm.get("default")["path"] == str(gguf)
    assert upstream["baseUrl"] == mm.base_url("default")
    mm.remove("default")


def test_resolve_upstream_falls_back_to_legacy_when_no_model_path():
    s = config.Settings(model_path="")
    eng = Engine(settings=s, manager=object(), anamnesis=object())
    # No model_path, no registered/running models -> old InferenceServer path still
    # raises its own clear error (unchanged behavior for the "nothing configured" case).
    import pytest
    from pleiades.inference import InferenceError
    with pytest.raises(InferenceError):
        eng._resolve_upstream(Profile(name="char1"))


# --------------------------------------------------------------------------- #
# Vision routing (docs/specs/2026-07-23-vision-routing-design.md)
# --------------------------------------------------------------------------- #

def test_build_user_content_plain_text_when_no_attachments():
    assert Engine._build_user_content("hello", None, vision_capable=True) == "hello"
    assert Engine._build_user_content("hello", [], vision_capable=True) == "hello"


def test_build_user_content_builds_content_parts_when_vision_capable():
    atts = [{"mime": "image/png", "data_b64": "QUJD", "name": "logo.png"}]
    content = Engine._build_user_content("what is this?", atts, vision_capable=True)
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "what is this?"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == "data:image/png;base64,QUJD"


def test_build_user_content_multiple_attachments():
    atts = [
        {"mime": "image/png", "data_b64": "QUFB", "name": "a.png"},
        {"mime": "image/jpeg", "data_b64": "QkJC", "name": "b.jpg"},
    ]
    content = Engine._build_user_content("compare these", atts, vision_capable=True)
    assert len(content) == 3
    assert content[1]["image_url"]["url"] == "data:image/png;base64,QUFB"
    assert content[2]["image_url"]["url"] == "data:image/jpeg;base64,QkJC"


def test_build_user_content_falls_back_to_text_description_when_not_vision_capable(monkeypatch):
    from pleiades.tools import vision_client
    monkeypatch.setattr(vision_client, "is_configured", lambda: True)
    monkeypatch.setattr(vision_client, "describe_image", lambda raw, **kw: "a red circle")
    atts = [{"mime": "image/png", "data_b64": "QUJD", "name": "logo.png"}]
    content = Engine._build_user_content("what is this?", atts, vision_capable=False)
    assert isinstance(content, str)
    assert "what is this?" in content
    assert "[attached image: logo.png]" in content
    assert "a red circle" in content


def test_build_user_content_notes_unconfigured_fallback_when_not_vision_capable(monkeypatch):
    from pleiades.tools import vision_client
    monkeypatch.setattr(vision_client, "is_configured", lambda: False)
    atts = [{"mime": "image/png", "data_b64": "QUJD", "name": "logo.png"}]
    content = Engine._build_user_content("what is this?", atts, vision_capable=False)
    assert isinstance(content, str)
    assert "no vision-fallback model configured" in content


def test_degrade_history_attachments_is_a_noop_for_plain_string_history():
    history = [{"role": "user", "content": "plain text turn"},
              {"role": "assistant", "content": "a reply"}]
    assert Engine._degrade_history_attachments(history) == history


def test_degrade_history_attachments_strips_image_parts_to_a_placeholder():
    history = [
        {"role": "user", "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}, "name": "logo.png"},
        ]},
    ]
    out = Engine._degrade_history_attachments(history)
    assert out[0]["content"] == "what is this?\n[attached image: logo.png]"
    # No base64 anywhere in the degraded output.
    assert "AAAA" not in out[0]["content"]


def test_base_messages_wires_attachments_only_into_the_fresh_turn_not_history():
    history = [
        {"role": "user", "content": [
            {"type": "text", "text": "earlier question"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,STALE"}, "name": "old.png"},
        ]},
    ]
    atts = [{"mime": "image/png", "data_b64": "RlJFU0g=", "name": "new.png"}]
    messages = Engine._base_messages("new question", None, None, history,
                                     attachments=atts, vision_capable=True)
    # system, degraded history user turn, fresh user turn
    assert messages[0]["role"] == "system"
    assert messages[1]["content"] == "earlier question\n[attached image: old.png]"
    assert "STALE" not in str(messages)
    fresh = messages[-1]
    assert fresh["role"] == "user"
    assert isinstance(fresh["content"], list)
    assert fresh["content"][1]["image_url"]["url"] == "data:image/png;base64,RlJFU0g="


def test_model_vision_capable_local_registered_model(tmp_path):
    gguf = tmp_path / "vlm.gguf"
    gguf.write_bytes(b"fake-model")
    mmproj = tmp_path / "mmproj-vlm-f16.gguf"
    mmproj.write_bytes(b"fake-mmproj")
    mm = ModelManager()
    mm.add("vlm-test-engine", str(gguf))
    try:
        s = config.Settings()
        eng = Engine(settings=s, manager=object(), anamnesis=object())
        p = Profile(name="char-vlm", model="vlm-test-engine")
        assert eng._model_vision_capable(p) is True
    finally:
        mm.remove("vlm-test-engine", delete_file=False)


def test_model_vision_capable_local_text_only_model(tmp_path):
    gguf = tmp_path / "text.gguf"
    gguf.write_bytes(b"fake-model")
    mm = ModelManager()
    mm.add("text-test-engine", str(gguf))
    try:
        s = config.Settings()
        eng = Engine(settings=s, manager=object(), anamnesis=object())
        p = Profile(name="char-text", model="text-test-engine")
        assert eng._model_vision_capable(p) is False
    finally:
        mm.remove("text-test-engine", delete_file=False)


def test_model_vision_capable_cloud_model_manual_override():
    s = config.Settings()
    eng = Engine(settings=s, manager=object(), anamnesis=object())
    p_yes = Profile(name="char-cloud-vision", model="openrouter:some/vlm",
                    model_capabilities="vision")
    p_no = Profile(name="char-cloud-text", model="openrouter:some/text-model")
    assert eng._model_vision_capable(p_yes) is True
    assert eng._model_vision_capable(p_no) is False


def test_model_vision_capable_ollama_cloud_manual_override():
    s = config.Settings()
    eng = Engine(settings=s, manager=object(), anamnesis=object())
    p = Profile(name="char-ollama-vision", model="ollama-cloud:qwen2.5vl",
               model_capabilities="vision")
    assert eng._model_vision_capable(p) is True


def test_build_user_content_reads_real_file_path_attachment(tmp_path):
    """feature/attach-cache's real shape (pleiades/attachments.py,
    resolve_attachments()) is {"name","path","mime"} -- a real file already
    saved on disk, no inline base64 at all. Confirm that shape works too."""
    img = tmp_path / "logo.png"
    img.write_bytes(b"\x89PNG-fake-bytes")
    atts = [{"name": "logo.png", "path": str(img), "mime": "image/png"}]
    content = Engine._build_user_content("what is this?", atts, vision_capable=True)
    assert isinstance(content, list)
    import base64 as b64mod
    expected = b64mod.b64encode(b"\x89PNG-fake-bytes").decode("ascii")
    assert content[1]["image_url"]["url"] == f"data:image/png;base64,{expected}"


def test_build_user_content_ignores_nonimage_attachments():
    atts = [{"name": "notes.pdf", "path": "/tmp/does-not-matter.pdf", "mime": "application/pdf"}]
    content = Engine._build_user_content("check this doc", atts, vision_capable=True)
    assert content == "check this doc"


def test_build_user_content_unreadable_path_falls_back_to_plain_text(tmp_path):
    atts = [{"name": "gone.png", "path": str(tmp_path / "does-not-exist.png"), "mime": "image/png"}]
    content = Engine._build_user_content("what is this?", atts, vision_capable=True)
    assert content == "what is this?"
