"""Model registry/assignment + Anamnesis adoption (no real llama server launched)."""

import pytest

from pleiades import config
from pleiades.models import ModelManager, ModelError
from pleiades.profiles import ProfileManager


class FakeAnamnesis:
    def __init__(self, names):
        self._names = set(names)

    def exists(self, n):
        return n in self._names

    def list_characters(self):
        return [{"name": n} for n in self._names]

    def update_character(self, n, **kw):
        return {}


def test_model_add_list_remove(tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF-fake")
    mm = ModelManager()
    m = mm.add("t1", str(gguf), n_gpu_layers=-1)
    assert m["path"] == str(gguf)
    assert m["n_gpu_layers"] == -1
    assert m["port"] > 0
    assert "t1" in [x["name"] for x in mm.list()]
    assert mm.is_running("t1") is False
    assert mm.base_url("t1").endswith("/v1")
    assert mm.remove("t1") is True
    assert mm.get("t1") is None


def test_model_add_missing_file_errors():
    with pytest.raises(ModelError):
        ModelManager().add("nope", "/no/such/file.gguf")


def test_model_ports_are_unique(tmp_path):
    a = tmp_path / "a.gguf"
    a.write_bytes(b"x")
    b = tmp_path / "b.gguf"
    b.write_bytes(b"x")
    mm = ModelManager()
    pa = mm.add("pa", str(a))
    pb = mm.add("pb", str(b))
    assert pa["port"] != pb["port"]
    mm.remove("pa")
    mm.remove("pb")


def test_stop_when_not_running():
    assert ModelManager().stop("ghost-model") is False


def test_adopt_existing_anamnesis_character():
    fa = FakeAnamnesis(["Mark", "default"])
    pm = ProfileManager(config.Settings.load(), anamnesis=fa)
    prof = pm.adopt("Mark")
    assert prof.name == "Mark"
    assert config.profile_json_path("Mark").is_file()
    orphans = pm.orphan_characters()
    assert "default" in orphans       # not yet adopted
    assert "Mark" not in orphans      # adopted -> no longer orphan


def test_adopt_missing_character_errors():
    pm = ProfileManager(config.Settings.load(), anamnesis=FakeAnamnesis([]))
    with pytest.raises(ValueError):
        pm.adopt("ghost")


def test_assign_model_persists(tmp_path):
    g = tmp_path / "mm.gguf"
    g.write_bytes(b"x")
    ModelManager().add("mistral", str(g))
    pm = ProfileManager(config.Settings.load(), anamnesis=FakeAnamnesis(["zoe"]))
    pm.adopt("zoe")
    pm.assign_model("zoe", "mistral")
    assert pm.get("zoe").model == "mistral"
    ModelManager().remove("mistral")


def test_state_cleans_up_dead_process(tmp_path):
    g = tmp_path / "dead.gguf"
    g.write_bytes(b"x")
    mm = ModelManager()
    mm.add("deadtest", str(g))
    # forge a running entry pointing at a dead pid and an unused port
    run = mm._load_running()
    run["deadtest"] = {"pid": 99999999, "host": "127.0.0.1", "port": 1}
    mm._save_running(run)
    assert mm.state("deadtest") == "crashed"      # detected + cleaned
    assert mm.state("deadtest") == "stopped"      # stale entry is gone
    mm.remove("deadtest")


def test_log_tail_missing_is_empty():
    assert ModelManager().log_tail("no-such-model") == ""
