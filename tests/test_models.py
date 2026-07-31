"""Model registry/assignment + Anamnesis adoption (no real llama server launched)."""

import os

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


def test_start_relocates_occupied_port(tmp_path, monkeypatch):
    """If another app holds the registered port, start() must move the model."""
    import socket

    import pleiades.models as models_mod

    g = tmp_path / "p.gguf"
    g.write_bytes(b"x")
    mm = ModelManager()
    entry = mm.add("porttest", str(g))
    # squat on the registered port like docker-proxy would
    squatter = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    squatter.bind(("127.0.0.1", entry["port"]))
    squatter.listen(1)
    launched = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, **kw):
        if "--port" not in cmd:        # hardware detection probes etc.
            raise OSError("blocked in test")
        launched["port"] = int(cmd[cmd.index("--port") + 1])
        return FakeProc()

    monkeypatch.setattr(models_mod.subprocess, "Popen", fake_popen)
    try:
        mm.start("porttest", wait=False)
        assert launched["port"] != entry["port"]          # relocated
        assert mm.get("porttest")["port"] == launched["port"]  # persisted
        assert "moved" in mm.log_tail("porttest")
    finally:
        squatter.close()
        run = mm._load_running()
        run.pop("porttest", None)
        mm._save_running(run)
        mm.remove("porttest")


def test_model_add_auto_detects_sibling_mmproj(tmp_path):
    gguf = tmp_path / "vlm-model.gguf"
    gguf.write_bytes(b"fake-model")
    mmproj = tmp_path / "mmproj-vlm-model-f16.gguf"
    mmproj.write_bytes(b"fake-mmproj")
    mm = ModelManager()
    m = mm.add("vlm1", str(gguf))
    assert m["mmproj"] == str(mmproj)
    mm.remove("vlm1", delete_file=False)


def test_model_add_without_sibling_mmproj_leaves_it_blank(tmp_path):
    gguf = tmp_path / "text-model.gguf"
    gguf.write_bytes(b"fake-model")
    mm = ModelManager()
    m = mm.add("text1", str(gguf))
    assert m["mmproj"] == ""
    mm.remove("text1", delete_file=False)


def test_model_add_explicit_mmproj_overrides_autodetect(tmp_path):
    gguf = tmp_path / "vlm-model2.gguf"
    gguf.write_bytes(b"fake-model")
    (tmp_path / "mmproj-vlm-model2-f16.gguf").write_bytes(b"fake-mmproj-auto")
    explicit = tmp_path / "elsewhere-mmproj.gguf"
    explicit.write_bytes(b"fake-mmproj-explicit")
    mm = ModelManager()
    m = mm.add("vlm2", str(gguf), mmproj=str(explicit))
    assert m["mmproj"] == str(explicit)
    mm.remove("vlm2", delete_file=False)


def test_model_add_capabilities_override(tmp_path):
    gguf = tmp_path / "cloud-ish.gguf"
    gguf.write_bytes(b"fake-model")
    mm = ModelManager()
    m = mm.add("caps1", str(gguf), capabilities="vision")
    assert m["capabilities"] == "vision"
    mm.remove("caps1", delete_file=False)


def test_model_vision_dataclass_property():
    from pleiades.models import Model
    assert Model(name="a", path="/x", mmproj="/y/mmproj-a.gguf").vision is True
    assert Model(name="a", path="/x").vision is False
    assert Model(name="a", path="/x", capabilities="vision").vision is True
    assert Model(name="a", path="/x", capabilities="chat,vision").vision is True
    assert Model(name="a", path="/x", capabilities="chat").vision is False


def test_is_vision_capable_helper():
    from pleiades.models import is_vision_capable
    assert is_vision_capable(None) is False
    assert is_vision_capable({}) is False
    assert is_vision_capable({"mmproj": "/x/mmproj-a.gguf"}) is True
    assert is_vision_capable({"mmproj": "", "capabilities": "vision"}) is True
    assert is_vision_capable({"mmproj": "", "capabilities": "chat"}) is False


# --------------------------------------------------------------------------- #
# slot save/restore (2026-07-24): turn a cold restart's full persona/memory
# reprocess into a fast state load, via llama-server's --slot-save-path +
# POST /slots/0?action=save|restore. Both helpers are deliberately best-
# effort/never-raise -- these tests cover that as much as the happy path.
# --------------------------------------------------------------------------- #
class _FakeResponse:
    def __init__(self, status_code=200):
        self.status_code = status_code


def test_slot_save_returns_false_when_nothing_is_listening():
    from pleiades.models import _slot_save
    # An arbitrary local port nothing is bound to -- connection refused.
    assert _slot_save("127.0.0.1", 1, timeout=1.0) is False


def test_slot_restore_skips_http_entirely_when_no_save_file_exists(monkeypatch):
    import pleiades.models as models_mod
    calls = []
    monkeypatch.setattr(models_mod.httpx, "post", lambda *a, **kw: calls.append((a, kw)) or _FakeResponse())
    assert models_mod._slot_restore("127.0.0.1", 12345, "no-such-model-xyz") is False
    assert calls == []  # never even tried -- file-existence check short-circuits


def test_slot_restore_posts_correct_request_when_save_file_exists(monkeypatch, tmp_path):
    import pleiades.models as models_mod
    save_dir = config.PLEIADES_HOME / "slots" / "sometest"
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "latest.bin").write_bytes(b"fake-state")
    calls = []

    def fake_post(url, params=None, json=None, timeout=None):
        calls.append((url, params, json))
        return _FakeResponse(200)

    monkeypatch.setattr(models_mod.httpx, "post", fake_post)
    assert models_mod._slot_restore("127.0.0.1", 12345, "sometest") is True
    assert len(calls) == 1
    url, params, body = calls[0]
    assert url == "http://127.0.0.1:12345/slots/0"
    assert params == {"action": "restore"}
    assert body == {"filename": "latest.bin"}


def test_slot_save_returns_false_on_non_200(monkeypatch):
    import pleiades.models as models_mod
    monkeypatch.setattr(models_mod.httpx, "post", lambda *a, **kw: _FakeResponse(501))
    assert models_mod._slot_save("127.0.0.1", 12345) is False


def test_stop_calls_slot_save_before_sending_the_kill_signal(monkeypatch):
    import pleiades.models as models_mod
    mm = ModelManager()
    mm._save_running({"savetest": {"pid": 999999, "host": "127.0.0.1", "port": 55001}})
    calls = []
    monkeypatch.setattr(models_mod, "_slot_save", lambda host, port: calls.append((host, port)))
    monkeypatch.setattr(models_mod, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(models_mod, "_pid_on_port", lambda host, port: None)
    # os.killpg/os.getpgid only exist on POSIX -- stop() itself branches on
    # os.name (see pleiades/models.py) and only touches them there, so mirror
    # that branch here rather than monkeypatching attributes that don't exist
    # on the os module at all on Windows (monkeypatch.setattr raises
    # AttributeError for a nonexistent target by default).
    if os.name == "posix":
        monkeypatch.setattr(models_mod.os, "killpg", lambda pgid, sig: None)
        monkeypatch.setattr(models_mod.os, "getpgid", lambda pid: pid)
    else:
        monkeypatch.setattr(models_mod.os, "kill", lambda pid, sig: None)
    assert mm.stop("savetest") is True
    assert calls == [("127.0.0.1", 55001)]


def test_start_calls_slot_restore_after_wait_ready(tmp_path, monkeypatch):
    import pleiades.models as models_mod

    g = tmp_path / "r.gguf"
    g.write_bytes(b"x")
    mm = ModelManager()
    mm.add("restoretest", str(g))

    class FakeProc:
        pid = 4243

        def poll(self):
            return None

    def fake_popen(cmd, **kw):
        if "--port" not in cmd:
            raise OSError("blocked in test")
        return FakeProc()

    calls = []
    monkeypatch.setattr(models_mod.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(mm, "_wait_ready", lambda name, proc, timeout: None)
    monkeypatch.setattr(models_mod, "_slot_restore",
                        lambda host, port, name: calls.append((host, port, name)))
    try:
        mm.start("restoretest", wait=True)
        assert len(calls) == 1
        assert calls[0][2] == "restoretest"
    finally:
        run = mm._load_running()
        run.pop("restoretest", None)
        mm._save_running(run)
        mm.remove("restoretest")


def test_start_does_not_call_slot_restore_when_wait_is_false(tmp_path, monkeypatch):
    import pleiades.models as models_mod

    g = tmp_path / "r2.gguf"
    g.write_bytes(b"x")
    mm = ModelManager()
    mm.add("norestoretest", str(g))

    class FakeProc:
        pid = 4244

    def fake_popen(cmd, **kw):
        if "--port" not in cmd:
            raise OSError("blocked in test")
        return FakeProc()

    calls = []
    monkeypatch.setattr(models_mod.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(models_mod, "_slot_restore",
                        lambda host, port, name: calls.append((host, port, name)))
    try:
        mm.start("norestoretest", wait=False)
        assert calls == []
    finally:
        run = mm._load_running()
        run.pop("norestoretest", None)
        mm._save_running(run)
        mm.remove("norestoretest")


def test_concurrent_start_does_not_spawn_duplicate_processes(tmp_path, monkeypatch):
    """Two characters sharing one model both taking their first turn in the
    same instant must not both spawn a llama-server. See HANDOFF's
    2026-07-25 correctness audit, finding #4."""
    import threading
    import time

    import pleiades.models as models_mod

    g = tmp_path / "race.gguf"
    g.write_bytes(b"x")
    mm = ModelManager()
    mm.add("racetest", str(g))

    spawn_count = {"n": 0}
    spawn_lock = threading.Lock()

    class FakeProc:
        def __init__(self, pid):
            self.pid = pid

    def fake_popen(cmd, **kw):
        if "pleiades.inference.server" not in cmd and not any(
                str(c).endswith(("llama-server", "llama-server.exe")) for c in cmd):
            raise OSError("blocked in test")  # hardware-detection probes etc.
        with spawn_lock:
            spawn_count["n"] += 1
            pid = 5000 + spawn_count["n"]
        time.sleep(0.05)  # widen the window a real race would need
        return FakeProc(pid)

    monkeypatch.setattr(models_mod.subprocess, "Popen", fake_popen)
    # A real spawned pid is alive immediately (still loading, not crashed) --
    # FakeProc's made-up pid isn't a real process, so without this the
    # second thread's is_running() check sees state()=="crashed" (correctly,
    # for a pid that really doesn't exist) and legitimately retries. This
    # simulates "the process really did start" the same way the sibling
    # test_stop_calls_slot_save_before_sending_the_kill_signal does.
    monkeypatch.setattr(models_mod, "_pid_alive", lambda pid: True)
    barrier = threading.Barrier(2)
    errors = []

    def worker():
        try:
            barrier.wait(timeout=5)
            mm.start("racetest", wait=False)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    try:
        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert not errors
        assert spawn_count["n"] == 1, "exactly one llama-server should be spawned"
        run = mm._load_running()
        assert run["racetest"]["pid"] == 5001
    finally:
        run = mm._load_running()
        run.pop("racetest", None)
        mm._save_running(run)
        mm.remove("racetest")
