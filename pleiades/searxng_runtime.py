"""SearXNG lifecycle — Pleiades-owned supervision, no Docker required.

Phase 3 of docs/specs/2026-07-25-anamnesis-vendoring-and-installer-plan.md:
replaces `docker compose up -d searxng` with the same start/stop/health-check
shape `pleiades/models.py`'s ModelManager and `pleiades/anamnesis_runtime.py`
already use, so a fresh install needs nothing pre-installed (no Docker
daemon, no separate download) to get local web search working.

SearXNG's SOURCE is fetched at install time (not committed to git — unlike
Anamnesis, this is a large, independently-developed third-party project, not
a fork Pleiades maintains, so there's no reason to vendor its full history
in-repo; see install.sh's install_searxng()) into <repo_root>/searxng-src/,
pinned to a specific upstream commit for reproducibility (SearXNG has no
git tags -- it's CalVer/rolling-release -- so a commit SHA is the only
stable thing to pin). It gets its OWN dedicated venv
(<repo_root>/searxng-src/.venv), entirely separate from Pleiades' own
.venv, to avoid any dependency-version collision between SearXNG's Flask/
lxml/babel stack and Pleiades' own FastAPI/webui stack.

Config lives at <repo_root>/services/searxng/settings.yml (unchanged
location from the Docker-Compose era -- only its bind_address/port/
secret_key were updated: 127.0.0.1 instead of 0.0.0.0 since there's no NAT
layer anymore, and a real random secret_key instead of a shared
"change-me-please" placeholder). Client side is unchanged: Settings.
searxng_url (default http://127.0.0.1:8888) is exactly what
pleiades/tools/search.py and harness/builtins/web.py already talk to --
this module only starts/stops the process, it never changes that contract.

State (under PLEIADES_HOME, mirroring anamnesis-running.json):
  searxng-running.json   {"pid": int, "host": str, "port": int}
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import httpx
import yaml

from . import config


class SearxngRuntimeError(RuntimeError):
    """Raised when SearXNG can't be located, started, or reached."""


def _running_json() -> Path:
    return config.PLEIADES_HOME / "searxng-running.json"


def searxng_dir() -> Path:
    """Where the fetched SearXNG source lives.

    PLEIADES_SEARXNG_DIR overrides everything (dev against a separate
    checkout, or a packaged build's bundled copy, mirroring
    PLEIADES_ANAMNESIS_DIR). Otherwise: a sibling `searxng-src/` next to
    this file's repo root (this file is pleiades/searxng_runtime.py, so repo
    root is one level up) -- the install-time-fetch location.
    """
    override = os.environ.get("PLEIADES_SEARXNG_DIR")
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parent.parent / "searxng-src"


def _venv_python() -> Path:
    """SearXNG's own dedicated venv -- never Pleiades' own .venv (see
    module docstring for why: independent, potentially-conflicting
    dependency trees, same reasoning as Anamnesis getting its own Node
    process rather than being require()'d into anything else)."""
    if os.name == "posix":
        return searxng_dir() / ".venv" / "bin" / "python"
    return searxng_dir() / ".venv" / "Scripts" / "python.exe"


def settings_path() -> Path:
    override = os.environ.get("PLEIADES_SEARXNG_SETTINGS")
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parent.parent / "services" / "searxng" / "settings.yml"


def _searxng_url() -> str:
    return config.Settings.load().searxng_url


def _host_port() -> tuple[str, int]:
    """Where we tell SearXNG to bind (unlike Anamnesis's daemon, SearXNG DOES
    read its bind address/port from settings.yml, which we control -- see
    settings_path(). This reads settings.yml directly rather than parsing
    Settings.searxng_url, so a mismatch between the two is visible instead
    of silently trusting one over the other)."""
    try:
        data = yaml.safe_load(settings_path().read_text(encoding="utf-8")) or {}
        server = data.get("server", {}) or {}
        return server.get("bind_address", "127.0.0.1"), int(server.get("port", 8888))
    except (OSError, yaml.YAMLError, ValueError):
        return "127.0.0.1", 8888


def _load_running() -> dict:
    p = _running_json()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_running(data: dict) -> None:
    config.PLEIADES_HOME.mkdir(parents=True, exist_ok=True)
    # Atomic write -- a concurrent reader (e.g. CLI status check racing a
    # start()/stop() from another process) must never see a torn/partial body.
    config.atomic_write_json(_running_json(), data)


# Shared implementation: os.kill(pid, 0) is a real CTRL_C_EVENT on Windows,
# not a probe -- see config.pid_alive's docstring.
_pid_alive = config.pid_alive


def _health(timeout: float = 3.0) -> bool:
    """GET /healthz -- SearXNG's own cheap liveness endpoint (plain text
    "OK", no real search performed). Deliberately NOT a real /search?
    format=json call on every poll -- that would hit real upstream search
    engines just to check "is my local process up", which is both slow and
    rude to those engines. A real JSON-search round-trip is worth checking
    once after a fresh install (install.sh does this -- see its
    install_searxng()), not on every health poll."""
    try:
        r = httpx.get(f"{_searxng_url()}/healthz", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


class SearxngDaemon:
    """Start/stop/status for the locally-run SearXNG process."""

    def is_running(self) -> bool:
        run = _load_running()
        return _pid_alive(run.get("pid")) or _health()

    def status(self) -> dict:
        h = _health()
        run = _load_running()
        return {
            "running": h,
            "pid": run.get("pid"),
            "url": _searxng_url(),
            "source_dir": str(searxng_dir()),
        }

    def start(self, *, wait: bool = True, timeout: float = 30.0) -> str:
        if self.is_running():
            return _searxng_url()

        python = _venv_python()
        if not python.exists():
            raise SearxngRuntimeError(
                f"SearXNG venv not found at {python}. Expected the "
                "install-time fetch from phase 3 of docs/specs/2026-07-25-"
                "anamnesis-vendoring-and-installer-plan.md (see install.sh's "
                "install_searxng()), or set PLEIADES_SEARXNG_DIR to point at "
                "an existing checkout+venv."
            )
        settings = settings_path()
        if not settings.exists():
            raise SearxngRuntimeError(f"SearXNG settings.yml not found at {settings}.")

        host, port = _host_port()

        logdir = config.PLEIADES_HOME / "logs"
        logdir.mkdir(parents=True, exist_ok=True)
        logf = open(logdir / "searxng.log", "ab")
        logf.write(f"[pleiades] starting {python} -m searx.webapp (settings={settings})\n".encode())
        logf.flush()

        env = dict(os.environ)
        env["SEARXNG_SETTINGS_PATH"] = str(settings)

        kwargs: dict = {}
        if os.name == "posix":
            kwargs["start_new_session"] = True  # survive the CLI exiting
        else:
            kwargs["creationflags"] = 0x00000200  # CREATE_NEW_PROCESS_GROUP

        try:
            proc = subprocess.Popen(
                [str(python), "-m", "searx.webapp"],
                cwd=str(searxng_dir()), env=env,
                stdout=logf, stderr=subprocess.STDOUT, **kwargs,
            )
        except FileNotFoundError as e:
            raise SearxngRuntimeError(f"Could not launch SearXNG: {e}") from e

        _save_running({"pid": proc.pid, "host": host, "port": port})

        if wait:
            self._wait_ready(proc, timeout)
        return _searxng_url()

    def _wait_ready(self, proc: subprocess.Popen, timeout: float) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if _health():
                return
            if proc.poll() is not None:
                raise SearxngRuntimeError(
                    f"SearXNG exited early (code {proc.returncode}). "
                    f"See {config.PLEIADES_HOME / 'logs' / 'searxng.log'}"
                )
            time.sleep(0.5)
        raise SearxngRuntimeError(f"SearXNG not ready after {timeout:.0f}s.")

    def stop(self) -> bool:
        run = _load_running()
        pid = run.get("pid")
        if not _pid_alive(pid):
            _save_running({})
            return False
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            else:
                os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        deadline = time.time() + 10.0
        while time.time() < deadline and _pid_alive(pid):
            time.sleep(0.2)
        if _pid_alive(pid):
            try:
                # signal.SIGKILL doesn't exist on Windows (no HAVE_SIGKILL) --
                # os.kill(pid, SIGTERM) above already maps to TerminateProcess
                # there, so a still-alive process just gets one more wait cycle.
                if os.name == "posix":
                    os.kill(pid, signal.SIGKILL)
                else:
                    os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        _save_running({})
        return True
