"""Anamnesis daemon supervisor (no real Node process launched, mirrors
test_models.py's "no real llama server launched" convention).

IMPORTANT: this machine may have a REAL Anamnesis daemon already listening on
the default control port (127.0.0.1:9000) -- every test that touches
is_running()/status()/start() must monkeypatch _control_url() to an unused
port first, or it'll observe production instead of a clean slate.
"""

import os

import pytest

from pleiades import anamnesis_runtime as ar

_FAKE_URL = "http://127.0.0.1:19999"  # nothing real ever listens here


@pytest.fixture(autouse=True)
def _isolate_control_url(monkeypatch):
    """Every test in this file gets a control URL nothing is actually
    listening on, so is_running()/_health() can't accidentally observe this
    machine's real production Anamnesis daemon on :9000."""
    monkeypatch.setattr(ar, "_control_url", lambda: _FAKE_URL)


def test_anamnesis_dir_default_is_repo_sibling():
    d = ar.anamnesis_dir()
    assert d.name == "anamnesis"
    assert d.parent.name == "Pleiades" or (d.parent / "pleiades").is_dir()


def test_anamnesis_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("PLEIADES_ANAMNESIS_DIR", str(tmp_path / "custom"))
    assert ar.anamnesis_dir() == tmp_path / "custom"


def test_daemon_script_path(tmp_path, monkeypatch):
    monkeypatch.setenv("PLEIADES_ANAMNESIS_DIR", str(tmp_path))
    assert ar.daemon_script() == tmp_path / "src" / "daemon.js"


def test_pid_alive_true_for_self():
    assert ar._pid_alive(os.getpid()) is True


def test_pid_alive_false_for_none_or_zero():
    assert ar._pid_alive(None) is False
    assert ar._pid_alive(0) is False


def test_pid_alive_false_for_dead_pid():
    # A pid that's (almost certainly) not in use: fork, let it exit, then check.
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)
    assert ar._pid_alive(pid) is False


def test_running_json_roundtrip():
    ar._save_running({"pid": 123, "host": "127.0.0.1", "port": 9000})
    assert ar._load_running() == {"pid": 123, "host": "127.0.0.1", "port": 9000}
    ar._save_running({})
    assert ar._load_running() == {}


def test_load_running_missing_file_returns_empty():
    ar._save_running({})
    ar._running_json().unlink()
    assert ar._load_running() == {}


def test_status_when_nothing_running(tmp_path, monkeypatch):
    monkeypatch.setenv("PLEIADES_ANAMNESIS_DIR", str(tmp_path))
    ar._save_running({})
    st = ar.AnamnesisDaemon().status()
    assert st["running"] is False
    assert st["pid"] is None
    assert st["source_dir"] == str(tmp_path)


def test_is_running_false_initially(tmp_path, monkeypatch):
    monkeypatch.setenv("PLEIADES_ANAMNESIS_DIR", str(tmp_path))
    ar._save_running({})
    assert ar.AnamnesisDaemon().is_running() is False


def test_stop_when_not_running_returns_false():
    ar._save_running({})
    assert ar.AnamnesisDaemon().stop() is False


def test_start_raises_clear_error_when_source_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("PLEIADES_ANAMNESIS_DIR", str(tmp_path / "nope"))
    ar._save_running({})
    with pytest.raises(ar.AnamnesisRuntimeError, match="not found"):
        ar.AnamnesisDaemon().start(wait=False)


def test_start_raises_clear_error_when_deps_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("PLEIADES_ANAMNESIS_DIR", str(tmp_path))
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "daemon.js").write_text("// fake")
    ar._save_running({})
    with pytest.raises(ar.AnamnesisRuntimeError, match="node_modules"):
        ar.AnamnesisDaemon().start(wait=False)


def test_node_bin_raises_clear_error_when_missing(monkeypatch):
    monkeypatch.setattr(ar.shutil, "which", lambda _: None)
    with pytest.raises(ar.AnamnesisRuntimeError, match="node"):
        ar._node_bin()


def test_host_port_parses_control_url(monkeypatch):
    monkeypatch.setattr(ar, "_control_url", lambda: "http://127.0.0.1:9123")
    assert ar._host_port() == ("127.0.0.1", 9123)


def test_adopt_raises_when_source_missing(tmp_path, monkeypatch):
    # No real systemd unit in the test env, so this exercises the same
    # "source not found" path start() itself uses -- adopt's job is to hand
    # off to the supervisor, and it should surface the same clear error
    # rather than silently doing nothing if the vendored copy isn't there.
    monkeypatch.setenv("PLEIADES_ANAMNESIS_DIR", str(tmp_path / "nope"))
    ar._save_running({})
    with pytest.raises(ar.AnamnesisRuntimeError):
        ar.adopt_existing_systemd(keep_systemd=True)
