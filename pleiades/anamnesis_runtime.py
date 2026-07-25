"""Anamnesis daemon lifecycle — Pleiades-owned supervision.

Phase 1 of docs/specs/2026-07-25-anamnesis-vendoring-and-installer-plan.md:
replaces the old hand-set-up systemd unit (`/etc/systemd/system/
anamnesis.service`, present on machines set up before this existed) with the
same start/stop/health-check shape `pleiades/models.py`'s ModelManager
already uses for llama-server processes, so Anamnesis gets supervised the
same way everything else in Pleiades is.

Anamnesis's SOURCE is vendored in-repo at <repo_root>/anamnesis/ (git subtree
from Fleabag515/anamnesis — phase 0 of the same plan; see that doc for why
subtree over submodule/copy). Its RUNTIME DATA (history.db, characters/,
registry.json, downloaded models) stays at ~/.anamnesis — untouched by
anything here, by phase 0's vendoring, or by adopt() below. Control API
lives at Settings.anamnesis_control_url (default http://127.0.0.1:9000),
exactly what pleiades/anamnesis.py's Anamnesis client already talks to — this
module only starts/stops the process; it never changes that contract.

State (under PLEIADES_HOME, mirroring models-running.json):
  anamnesis-running.json   {"pid": int, "host": str, "port": int}

Packaged (desktop app) builds: the Electron main process sets
PLEIADES_ANAMNESIS_DIR (see desktop/src/main/index.ts's spawnBackend()) to
point at electron-builder's extraResources copy (<resources>/anamnesis/),
and _node_bin() below looks for a bundled Node runtime as a sibling of
wherever anamnesis_dir() resolves to (<resources>/anamnesis-node-runtime/
node, from a *second* extraResources entry — see build-native/fetch-node.sh)
before falling back to PATH. Dev/CLI installs still just use anamnesis/ at
the repo root and whatever `node` is on PATH, same as phase 0/1. If neither
the bundled dir nor a repo-root checkout exists, start() still raises a
clear AnamnesisRuntimeError rather than pretending to work.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from . import config


class AnamnesisRuntimeError(RuntimeError):
    """Raised when the Anamnesis daemon can't be located, started, or reached."""


def _running_json() -> Path:
    return config.PLEIADES_HOME / "anamnesis-running.json"


def anamnesis_dir() -> Path:
    """Where the vendored Anamnesis source lives.

    PLEIADES_ANAMNESIS_DIR overrides everything (useful for dev against a
    separate checkout, or once phase 2 defines where a packaged build
    extracts its bundled copy to). Otherwise: a sibling `anamnesis/` next to
    this file's repo root (this file is pleiades/anamnesis_runtime.py, so
    repo root is one level up) — the phase-0 git-subtree location.
    """
    override = os.environ.get("PLEIADES_ANAMNESIS_DIR")
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parent.parent / "anamnesis"


def daemon_script() -> Path:
    return anamnesis_dir() / "src" / "daemon.js"


def _node_bin() -> str:
    """Resolve a `node` to run the daemon with.

    Resolution order: PLEIADES_ANAMNESIS_NODE_BIN (explicit override, for
    testing/dev) -> a bundled runtime sibling to anamnesis_dir() (packaged
    desktop builds — see build-native/fetch-node.sh's output layout and
    electron-builder.yml's second extraResources entry) -> whatever `node`
    is on PATH (dev/CLI installs, same as the fork's own pre-vendoring
    setup)."""
    override = os.environ.get("PLEIADES_ANAMNESIS_NODE_BIN")
    if override:
        return override

    bundled = anamnesis_dir().parent / "anamnesis-node-runtime" / (
        "node.exe" if os.name != "posix" else "node"
    )
    if bundled.is_file():
        return str(bundled)

    node = shutil.which("node")
    if not node:
        raise AnamnesisRuntimeError(
            "No 'node' on PATH — Anamnesis needs Node.js >=20 to run. "
            "Install Node, then try again."
        )
    return node


def _control_url() -> str:
    return config.Settings.load().anamnesis_control_url


def _host_port() -> tuple[str, int]:
    """Where we EXPECT the daemon to end up listening.

    NOTE: the daemon does not read PORT/HOST env vars -- it binds whatever
    its own config.json's daemonCfg.controlPort says (hardcoded fallback
    9000 if unset), same as the pre-vendoring setup. This just describes the
    expected address for running.json bookkeeping and the health check; it
    does not and cannot configure the daemon's actual bind address. If
    Settings.anamnesis_control_url and the vendored anamnesis/config.json
    ever disagree, the health check below will correctly report "not
    running" even though a process exists -- that's a real configuration
    mismatch to fix on its own, not something to paper over here."""
    parsed = urlsplit(_control_url())
    return parsed.hostname or "127.0.0.1", parsed.port or 9000


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
    _running_json().write_text(json.dumps(data, indent=2), encoding="utf-8")


def _pid_alive(pid: "int | None") -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just not ours to signal
    except OSError:
        return False


def _health(timeout: float = 3.0) -> "dict | None":
    """GET /status if anything answers — real signal, not just 'a pid exists'
    (the same distinction ModelManager._wait_ready draws for models)."""
    try:
        r = httpx.get(f"{_control_url()}/status", timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


class AnamnesisDaemon:
    """Start/stop/status for the vendored Anamnesis daemon process."""

    def is_running(self) -> bool:
        run = _load_running()
        return _pid_alive(run.get("pid")) or _health() is not None

    def status(self) -> dict:
        h = _health()
        run = _load_running()
        return {
            "running": h is not None,
            "pid": run.get("pid"),
            "control_url": _control_url(),
            "source_dir": str(anamnesis_dir()),
            "health": h,
        }

    def start(self, *, wait: bool = True, timeout: float = 60.0) -> str:
        if self.is_running():
            return _control_url()

        script = daemon_script()
        if not script.exists():
            raise AnamnesisRuntimeError(
                f"Anamnesis source not found at {script}. Expected the "
                "vendored copy from phase 0 of docs/specs/2026-07-25-"
                "anamnesis-vendoring-and-installer-plan.md, a packaged "
                "build's bundled copy, or set PLEIADES_ANAMNESIS_DIR to "
                "point at a checkout."
            )
        node_modules = anamnesis_dir() / "node_modules"
        if not node_modules.is_dir():
            raise AnamnesisRuntimeError(
                f"{anamnesis_dir()} has no node_modules/ — install its deps "
                f"first: (cd {anamnesis_dir()} && npm install --omit=dev)"
            )

        node = _node_bin()
        host, port = _host_port()

        logdir = config.PLEIADES_HOME / "logs"
        logdir.mkdir(parents=True, exist_ok=True)
        logf = open(logdir / "anamnesis-daemon.log", "ab")
        logf.write(f"[pleiades] starting {node} {script}\n".encode())
        logf.flush()

        kwargs: dict = {}
        if os.name == "posix":
            kwargs["start_new_session"] = True  # survive the CLI exiting
            # Preserve the old systemd unit's Nice=10 intent: background
            # memory chores (extraction/consolidation/the helper LLM) must
            # never starve the primary inference server — that unit's own
            # comment documents a real 15->1-2 tok/s regression on Ornith
            # when this ran at normal priority. os.nice() in a preexec_fn
            # runs in the CHILD after fork, before exec, so it only affects
            # the daemon, never this process.
            kwargs["preexec_fn"] = lambda: os.nice(10)
        else:
            kwargs["creationflags"] = 0x00000200  # CREATE_NEW_PROCESS_GROUP

        try:
            proc = subprocess.Popen(
                [node, str(script)], cwd=str(anamnesis_dir()),
                stdout=logf, stderr=subprocess.STDOUT, **kwargs,
            )
        except FileNotFoundError as e:
            raise AnamnesisRuntimeError(f"Could not launch node: {e}") from e

        _save_running({"pid": proc.pid, "host": host, "port": port})

        if wait:
            self._wait_ready(proc, timeout)
        return _control_url()

    def _wait_ready(self, proc: subprocess.Popen, timeout: float) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if _health() is not None:
                return
            if proc.poll() is not None:
                raise AnamnesisRuntimeError(
                    f"Anamnesis daemon exited early (code {proc.returncode}). "
                    f"See {config.PLEIADES_HOME / 'logs' / 'anamnesis-daemon.log'}"
                )
            time.sleep(0.5)
        raise AnamnesisRuntimeError(f"Anamnesis daemon not ready after {timeout:.0f}s.")

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
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        _save_running({})
        return True


def adopt_existing_systemd(*, keep_systemd: bool = False) -> str:
    """Migrate a pre-phase-1 machine (systemd-managed daemon at
    ~/.local/share/anamnesis, per the old hand-written unit) onto this
    supervisor. Never touches ~/.anamnesis (runtime data). Returns a
    human-readable summary of what it did.
    """
    lines: list[str] = []
    unit_path = Path("/etc/systemd/system/anamnesis.service")
    if not keep_systemd and unit_path.exists():
        for cmd in (
            ["sudo", "systemctl", "stop", "anamnesis"],
            ["sudo", "systemctl", "disable", "anamnesis"],
        ):
            subprocess.run(cmd, capture_output=True, check=False)
        lines.append(f"stopped + disabled {unit_path} (left the file — remove it "
                      "by hand once you've confirmed the new supervisor works)")
    elif keep_systemd:
        lines.append(f"--keep-systemd: leaving {unit_path} alone")

    wrapper = Path.home() / ".local" / "bin" / "anamnesis"
    if wrapper.exists():
        lines.append(f"note: {wrapper} (the old fork install.sh's CLI wrapper) "
                      "still exists — harmless, but its own `anamnesis` binary "
                      "is no longer what serves :9000 once this supervisor "
                      "starts the vendored copy")

    lines.append("~/.anamnesis (runtime data: history.db, characters/, "
                  "registry.json, models) left untouched")
    daemon = AnamnesisDaemon()
    daemon.stop()  # in case the systemd-launched process is still alive
    url = daemon.start()
    lines.append(f"vendored Anamnesis now running under Pleiades supervision at {url}")
    return "\n".join(lines)
