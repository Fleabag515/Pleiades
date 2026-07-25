"""SearXNG process supervisor (no real SearXNG process launched, mirrors
test_anamnesis_runtime.py's "no real daemon launched" convention).

IMPORTANT: this machine may have a REAL SearXNG process already listening on
the default port (127.0.0.1:8888) -- every test that touches
is_running()/status()/start() must monkeypatch _searxng_url() to an unused
port first, or it'll observe production instead of a clean slate.
"""

import os

import pytest

from pleiades import searxng_runtime as sr

_FAKE_URL = "http://127.0.0.1:19998"  # nothing real ever listens here


@pytest.fixture(autouse=True)
def _isolate_searxng_url(monkeypatch):
    """Every test in this file gets a URL nothing is actually listening on,
    so is_running()/_health() can't accidentally observe this machine's real
    production SearXNG on :8888."""
    monkeypatch.setattr(sr, "_searxng_url", lambda: _FAKE_URL)


def test_searxng_dir_default_is_repo_sibling():
    d = sr.searxng_dir()
    assert d.name == "searxng-src"
    assert d.parent.name == "Pleiades" or (d.parent / "pleiades").is_dir()


def test_searxng_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("PLEIADES_SEARXNG_DIR", str(tmp_path / "custom"))
    assert sr.searxng_dir() == tmp_path / "custom"


def test_venv_python_path(tmp_path, monkeypatch):
    monkeypatch.setenv("PLEIADES_SEARXNG_DIR", str(tmp_path))
    expected = tmp_path / ".venv" / ("bin" if os.name == "posix" else "Scripts") / (
        "python" if os.name == "posix" else "python.exe"
    )
    assert sr._venv_python() == expected


def test_settings_path_default_is_repo_relative():
    p = sr.settings_path()
    assert p.name == "settings.yml"
    assert p.parent.name == "searxng"


def test_settings_path_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("PLEIADES_SEARXNG_SETTINGS", str(tmp_path / "custom.yml"))
    assert sr.settings_path() == tmp_path / "custom.yml"


def test_pid_alive_true_for_self():
    assert sr._pid_alive(os.getpid()) is True


def test_pid_alive_false_for_none_or_zero():
    assert sr._pid_alive(None) is False
    assert sr._pid_alive(0) is False


def test_pid_alive_false_for_dead_pid():
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)
    assert sr._pid_alive(pid) is False


def test_running_json_roundtrip():
    sr._save_running({"pid": 123, "host": "127.0.0.1", "port": 8888})
    assert sr._load_running() == {"pid": 123, "host": "127.0.0.1", "port": 8888}
    sr._save_running({})
    assert sr._load_running() == {}


def test_load_running_missing_file_returns_empty():
    sr._save_running({})
    sr._running_json().unlink()
    assert sr._load_running() == {}


def test_status_when_nothing_running(tmp_path, monkeypatch):
    monkeypatch.setenv("PLEIADES_SEARXNG_DIR", str(tmp_path))
    sr._save_running({})
    st = sr.SearxngDaemon().status()
    assert st["running"] is False
    assert st["pid"] is None
    assert st["source_dir"] == str(tmp_path)


def test_is_running_false_initially(tmp_path, monkeypatch):
    monkeypatch.setenv("PLEIADES_SEARXNG_DIR", str(tmp_path))
    sr._save_running({})
    assert sr.SearxngDaemon().is_running() is False


def test_stop_when_not_running_returns_false():
    sr._save_running({})
    assert sr.SearxngDaemon().stop() is False


def test_start_raises_clear_error_when_venv_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("PLEIADES_SEARXNG_DIR", str(tmp_path / "nope"))
    sr._save_running({})
    with pytest.raises(sr.SearxngRuntimeError, match="venv not found"):
        sr.SearxngDaemon().start(wait=False)


def test_start_raises_clear_error_when_settings_missing(tmp_path, monkeypatch):
    # Fake a venv python existing so we get past the first check and hit
    # the settings.yml check specifically.
    monkeypatch.setenv("PLEIADES_SEARXNG_DIR", str(tmp_path))
    venv_bin = tmp_path / ".venv" / ("bin" if os.name == "posix" else "Scripts")
    venv_bin.mkdir(parents=True)
    fake_python = venv_bin / ("python" if os.name == "posix" else "python.exe")
    fake_python.write_text("#!/bin/sh\n")
    fake_python.chmod(0o755)
    monkeypatch.setenv("PLEIADES_SEARXNG_SETTINGS", str(tmp_path / "nope.yml"))
    sr._save_running({})
    with pytest.raises(sr.SearxngRuntimeError, match="settings.yml not found"):
        sr.SearxngDaemon().start(wait=False)


def test_host_port_parses_settings_yml(tmp_path, monkeypatch):
    settings = tmp_path / "settings.yml"
    settings.write_text("server:\n  bind_address: \"127.0.0.1\"\n  port: 8888\n")
    monkeypatch.setenv("PLEIADES_SEARXNG_SETTINGS", str(settings))
    assert sr._host_port() == ("127.0.0.1", 8888)


def test_host_port_falls_back_to_defaults_when_unreadable(tmp_path, monkeypatch):
    monkeypatch.setenv("PLEIADES_SEARXNG_SETTINGS", str(tmp_path / "nope.yml"))
    assert sr._host_port() == ("127.0.0.1", 8888)
