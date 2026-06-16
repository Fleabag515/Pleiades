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
