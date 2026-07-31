"""Model registry + lifecycle — add GGUFs, run one server per model, assign to characters.

Pleiades is an inference engine: it runs models itself via llama.cpp. This module lets
you register multiple GGUF models by name, start/stop a dedicated OpenAI-compatible
server for each (on its own port), and choose which character uses which model.

State (under ~/.pleiades):
  models.json          registered models: name -> {path, n_ctx, n_gpu_layers, chat_format, host, port}
  models-running.json  live servers: name -> {pid, host, port}
  logs/model-<name>.log per-model server log

GPU: n_gpu_layers controls offload and works for NVIDIA/AMD/Apple Silicon
*provided the runtime that actually ends up running the model was built with
that backend* -- Pleiades' own engine (the default whenever its binary is
present, see launch.py's engine-selection order), the autodetected native
llama-server binary, or llama-cpp-python as the last-resort fallback, each
with their own backend build story. -1 offloads all layers; 0 is CPU-only.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import httpx

from . import config


def _models_json() -> Path:
    return config.PLEIADES_HOME / "models.json"


def _running_json() -> Path:
    return config.PLEIADES_HOME / "models-running.json"


# Re-exported for existing call sites in this module; now shared with
# profiles.py / chats.py / searxng_runtime.py / anamnesis_runtime.py so every
# writer of small state-tracking JSON gets the same torn-read protection.
_atomic_write_json = config.atomic_write_json


@dataclass
class Model:
    name: str
    path: str
    # "auto" (default) plans the launch window + elastic ceiling from the GGUF
    # and detected VRAM at each start; an int pins the window (no elasticity).
    n_ctx: "int | str" = "auto"
    # "auto" plans offload from detected hardware at each launch; int overrides
    # (0 = CPU, -1 = all layers on GPU — CUDA, ROCm, or Metal build).
    n_gpu_layers: "int | str" = "auto"
    chat_format: str = ""     # blank = llama.cpp auto-detect
    # Optional UI-only override for how this model's name renders in the
    # desktop app (composer picker, character model-assign dropdown, Models
    # list) — never touched by launch/registry-key logic, which always keys
    # off `name`. Blank means "show `name` as-is" (existing behavior).
    display_name: str = ""
    # Forward-compatible capability flag: True means this model can take audio
    # input directly (a real --mmproj-style audio encoder, or a native engine
    # that supports it) and engine.py should NOT run it through the
    # transcription fallback (pleiades/tools/audio_transcribe.py). Defaults to
    # False for every model because, as of this writing (2026-07-23 audio-
    # fallback council review), no model registrable here actually has this --
    # nothing sets it True yet. It exists so the day one does, engine.py's
    # `if not model.audio_capable` check needs no changes, just a registry
    # entry with this flipped on.
    audio_capable: bool = False
    host: str = "127.0.0.1"
    port: int = 0             # assigned on add()
    # Vision support (docs/specs/2026-07-23-vision-routing-design.md). Path to
    # a sibling multimodal-projector GGUF, auto-detected at add()/fetch time
    # via hardware.find_mmproj_sibling() (blank = none found / not applicable).
    # Consumed by both native launch paths (launch.py): the autodetected
    # llama-server binary, and (Phase 9.4.2) Pleiades' own engine, which now
    # links libmtmd too. Only the llama_cpp.server python fallback still
    # doesn't implement multimodal serving, so this field is a no-op there.
    mmproj: str = ""
    # Manual capability override, comma-separated (e.g. "vision"). For a local
    # GGUF this is normally redundant with `mmproj` (presence of the sidecar
    # already implies vision) -- it exists mainly for cloud models
    # (openrouter:/ollama-cloud: prefixed profile.model strings), which have
    # no local file to sniff and no entry in this registry at all; see
    # Profile.model_capabilities for where the cloud-model equivalent lives.
    capabilities: str = ""
    # Phase 9.4.4 (docs/specs/2026-07-21-native-inference-engine-design.md):
    # the permanent, backend-agnostic escape hatch for hardware this repo's
    # own engine builds don't (yet, or ever) run on -- points this registry
    # entry at ANY already-running OpenAI-compatible server instead of one
    # Pleiades spawns/manages itself. When set, `path` is not a GGUF file and
    # is never opened; start()/stop() become no-ops and base_url() returns
    # this verbatim. See ModelManager.add()'s external_url param.
    external_url: str = ""

    @property
    def vision(self) -> bool:
        """True if this model can accept image input.

        True when a multimodal-projector sidecar is associated (mmproj set)
        OR "vision" is explicitly listed in the manual capabilities override
        -- the latter matters if mmproj detection ever needs a manual assist
        (e.g. the sidecar lives in a different directory than the model).
        """
        if self.mmproj:
            return True
        return "vision" in {c.strip() for c in self.capabilities.split(",") if c.strip()}


class ModelError(RuntimeError):
    pass


def is_vision_capable(entry: "dict | None") -> bool:
    """Same test as Model.vision, for the plain dicts ModelManager.get()/
    list() hand back (registry entries round-trip through JSON, not the
    dataclass) -- e.g. engine.py resolving whether the character's assigned
    model can take image input."""
    if not entry:
        return False
    if entry.get("mmproj"):
        return True
    caps = entry.get("capabilities") or ""
    return "vision" in {c.strip() for c in caps.split(",") if c.strip()}


def _port_free(host: str, port: int) -> bool:
    """Can we actually bind it? Registry bookkeeping isn't enough — other apps
    (Docker, dev servers) may already hold a port in our 8090+ range.

    No SO_REUSEADDR on Windows: there it means "steal the port", so bind()
    happily succeeds on a port another process is LISTENING on and this probe
    reports occupied ports as free (live servers then get duplicated).
    """
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if os.name == "posix":
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # TIME_WAIT
        elif hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


# Shared implementation: os.kill(pid, 0) is a real CTRL_C_EVENT on Windows,
# not a probe -- see config.pid_alive's docstring. Module-level alias (not a
# call-site rewrite) so tests can keep monkeypatching models._pid_alive.
_pid_alive = config.pid_alive


# Fixed filename: one slot (--parallel 1, see launch.py), one save per model
# -- "the most recent state," not a history of snapshots. See launch.py's
# --slot-save-path comment for why this is always wired (cheap when unused)
# and models.py's stop()/start() for when these actually get called.
_SLOT_FILENAME = "latest.bin"


def _slot_save(host: str, port: int, timeout: float = 120.0) -> bool:
    """Best-effort: ask a still-alive llama-server to persist its slot-0
    state to disk. Returns whether it actually saved something; never
    raises -- a model not started with --slot-save-path, one with no
    active state, or any request failure all just mean "nothing saved,"
    which is exactly the pre-existing behavior (a normal cold start next
    time), not a new failure mode."""
    try:
        r = httpx.post(f"http://{host}:{port}/slots/0",
                       params={"action": "save"},
                       json={"filename": _SLOT_FILENAME}, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _slot_restore(host: str, port: int, name: str, timeout: float = 120.0) -> bool:
    """Best-effort: if stop() previously saved this model's slot state,
    restore it right after boot -- turns a cold restart's full persona/
    memory reprocess into a fast state load. Checks the save file exists
    before even trying (avoids a slow round-trip to be told "no such
    file"); never raises -- a missing, corrupted, or model-mismatched save
    file just means an ordinary cold start, same as before this existed."""
    save_file = config.PLEIADES_HOME / "slots" / name / _SLOT_FILENAME
    if not save_file.is_file():
        return False
    try:
        r = httpx.post(f"http://{host}:{port}/slots/0",
                       params={"action": "restore"},
                       json={"filename": _SLOT_FILENAME}, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _pid_on_port(host: str, port: int) -> Optional[int]:
    """PID of the process LISTENING on host:port, or None.

    Needed because running.json can go stale (crash, manual kill): the
    recorded pid dies but the actual server lives on. Windows: parse
    netstat -ano. POSIX: spawned with start_new_session, killpg covers it.
    """
    if os.name != "nt" or not port:
        return None
    try:
        out = subprocess.run(["netstat", "-ano"], capture_output=True,
                             text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    needle = f":{port}"
    for line in out.splitlines():
        parts = line.split()
        if (len(parts) >= 5 and parts[0] == "TCP" and "LISTENING" in parts
                and parts[1].endswith(needle)):
            try:
                return int(parts[-1])
            except ValueError:
                continue
    return None


# Guards models.json/models-running.json read-modify-write sequences. The
# webui runs sync endpoints in a threadpool (many threads in one process),
# and add()/start() each do read -> compute -> write with no protection --
# two characters starting the same shared model in the same second could
# both see is_running()==False and both spawn a llama-server on the same
# port. MODULE level (not per-instance) because nearly every call site
# (engine.py's per-turn _resolve_upstream, webui endpoints, cli commands)
# constructs its own throwaway ModelManager(), and per-instance locks on
# throwaway instances never contend, guarding nothing -- same reasoning as
# scheduler.py's _registry_lock. RLock (not Lock): start() and add() call
# the already-lock-wrapped _load()/_save() etc. from inside their own
# `with self._lock:` section, on the same thread.
_registry_lock = threading.RLock()


class ModelManager:
    """Register, run, stop, and assign local GGUF models."""

    def __init__(self) -> None:
        config.ensure_home()
        self._lock = _registry_lock

    # -- registry ----------------------------------------------------------- #
    def _load(self) -> dict:
        """Registry, pruned against what's actually on disk.

        models.json is bookkeeping, not truth — if a GGUF was deleted (by a
        previous remove(), or by hand) its entry must not go on appearing in
        the library forever. Every read goes through here so list()/get()/
        add() all see the reconciled state, not a stale in-memory picture.
        """
        with self._lock:
            p = _models_json()
            reg: dict = {}
            if p.is_file():
                try:
                    reg = json.loads(p.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    reg = {}
            if self._prune_missing(reg):
                self._save(reg)
            return reg

    def _prune_missing(self, reg: dict) -> bool:
        """Drop registry entries whose GGUF no longer exists on disk.

        path.is_file() can raise OSError for reasons that aren't "the file
        is gone" -- e.g. WinError 448 ("untrusted mount point"), seen live
        crossing a symlink/junction under a different security context on
        Windows. Treat "can't confirm" as "leave it registered" rather than
        letting one unreachable path's exotic filesystem error crash the
        whole models list (and therefore /api/hardware) for every model.
        """
        changed = False
        for name in list(reg.keys()):
            if reg[name].get("external_url"):
                continue  # no local file to prune against -- see add()'s external_url branch
            path = Path(reg[name].get("path", "")).expanduser()
            try:
                exists = path.is_file()
            except OSError:
                exists = True
            if not exists:
                del reg[name]
                changed = True
        return changed

    def _save(self, data: dict) -> None:
        with self._lock:
            _atomic_write_json(_models_json(), data)

    def add(self, name: str, path: str = "", *, n_ctx: "int | str" = "auto",
            n_gpu_layers: "int | str" = "auto",
            chat_format: str = "", port: int = 0,
            mmproj: str = "", capabilities: str = "",
            external_url: str = "") -> dict:
        if external_url:
            # Escape hatch (Phase 9.4.4): no local file, no port of our own to
            # manage -- base_url()/start()/stop()/state() all special-case
            # external_url below. `path` is deliberately left blank rather
            # than storing the URL there too: every other code path that
            # reads `path` (remove()'s _delete_model_files, log tailing by
            # name, etc.) assumes it names a real file on disk, and an empty
            # string trips their existing "nothing to do" branches instead of
            # a URL tripping something that tries to open/delete it.
            with self._lock:
                reg = self._load()
                reg[name] = asdict(Model(name=name, path="", n_ctx=n_ctx,
                                         n_gpu_layers=n_gpu_layers, chat_format=chat_format,
                                         mmproj=mmproj, capabilities=capabilities,
                                         external_url=external_url))
                self._save(reg)
                return reg[name]
        p = Path(path).expanduser()
        if not p.is_file():
            raise ModelError(f"No such model file: {p}")
        # Locked end to end: _free_port() reads the current registry to pick
        # an unused port, and that choice is only valid until it's committed
        # below. Without the lock, two concurrent add() calls can both read
        # the same "used" set and both pick the same free port.
        with self._lock:
            reg = self._load()
            port = port or self._free_port(reg)
            if not mmproj:
                # Auto-detect a sibling mmproj-*.gguf next to this model (the
                # foundry's download pipeline used to just discard this file --
                # see hardware.is_mmproj()'s docstring and fetch.py's
                # fetch_model(), which now downloads it alongside the model
                # specifically so this detection has something to find).
                from .hardware import find_mmproj_sibling
                mmproj = find_mmproj_sibling(str(p))
            reg[name] = asdict(Model(name=name, path=str(p), n_ctx=n_ctx,
                                     n_gpu_layers=n_gpu_layers, chat_format=chat_format,
                                     port=port, mmproj=mmproj, capabilities=capabilities))
            self._save(reg)
            return reg[name]

    def remove(self, name: str, *, delete_file: bool = True) -> bool:
        with self._lock:
            reg = self._load()
            if name not in reg:
                return False
            self.stop(name)
            path = Path(reg[name].get("path", "")).expanduser()
            del reg[name]
            self._save(reg)
        if delete_file:
            self._delete_model_files(path)
        return True

    def _delete_model_files(self, path: Path) -> None:
        """Best-effort delete of the actual GGUF(s) backing a removed model.

        Downloads made through the foundry live under their own directory in
        ~/.pleiades/models/<repo>/ (one dir per repo, may hold split parts) —
        for those, remove the whole directory so no orphaned parts remain.
        A model registered from an arbitrary path (`pleiades model add`) just
        has that one file unlinked; we never touch sibling files outside our
        managed models dir.
        """
        if not path or not str(path):
            return
        try:
            rp = path.resolve()
        except OSError:
            return
        managed_root = None
        try:
            from .fetch import models_dir
            managed_root = models_dir().resolve()
        except Exception:
            pass
        try:
            if managed_root and (rp.parent == managed_root or managed_root in rp.parents):
                import shutil
                shutil.rmtree(rp.parent, ignore_errors=True)
            elif rp.is_file():
                rp.unlink()
        except OSError:
            pass

    def get(self, name: str) -> Optional[dict]:
        return self._load().get(name)

    def list(self) -> list[dict]:
        out = []
        for name, m in self._load().items():
            m = dict(m)
            m["state"] = self.state(name)
            m["running"] = m["state"] in ("running", "loading")
            out.append(m)
        return out

    def _free_port(self, reg: dict, start: int = 8090, host: str = "127.0.0.1") -> int:
        used = {m.get("port") for m in reg.values()}
        used |= {r.get("port") for r in self._load_running().values()}
        port = start
        while port in used or not _port_free(host, port):
            port += 1
        return port

    # -- running state ------------------------------------------------------ #
    def _load_running(self) -> dict:
        with self._lock:
            p = _running_json()
            if p.is_file():
                try:
                    return json.loads(p.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    return {}
            return {}

    def _save_running(self, data: dict) -> None:
        with self._lock:
            _atomic_write_json(_running_json(), data)

    def base_url(self, name: str) -> str:
        m = self.get(name)
        if not m:
            raise ModelError(f"unknown model '{name}'")
        if m.get("external_url"):
            return m["external_url"]
        return f"http://{m['host']}:{m['port']}/v1"

    def state(self, name: str) -> str:
        """'running' (HTTP healthy) | 'loading' (process up, API not yet) |
        'crashed' (process died while registered) | 'stopped'."""
        m = self.get(name)
        if m and m.get("external_url"):
            # No process of ours to track -- "running" means "answered just
            # now", full stop. Never "loading"/"crashed": those describe a
            # lifecycle we don't own for a server we didn't spawn.
            try:
                r = httpx.get(f"{m['external_url'].rstrip('/')}/models", timeout=3.0)
                return "running" if r.status_code == 200 else "stopped"
            except httpx.HTTPError:
                return "stopped"
        run = self._load_running().get(name)
        if not run:
            return "stopped"
        try:
            r = httpx.get(f"http://{run['host']}:{run['port']}/v1/models", timeout=1.5)
            if r.status_code == 200:
                return "running"
        except httpx.HTTPError:
            pass
        if _pid_alive(run.get("pid")):
            return "loading"
        # Process died without a clean stop: clear the stale entry, report crash.
        all_run = self._load_running()
        all_run.pop(name, None)
        self._save_running(all_run)
        return "crashed"

    def is_running(self, name: str) -> bool:
        return self.state(name) in ("running", "loading")

    def log_tail(self, name: str, lines: int = 200) -> str:
        logf = config.PLEIADES_HOME / "logs" / f"model-{name}.log"
        if not logf.is_file():
            return ""
        try:
            text = logf.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])

    def _build_launch(self, m: dict) -> tuple[list, str, dict]:
        """Assemble the server command via the shared autofit-aware planner
        (same logic the single-model legacy engine uses — see launch.py)."""
        from .launch import build_command

        plan = build_command(
            m["path"], m["host"], m["port"], name=str(m.get("name", "local")),
            n_ctx=m.get("n_ctx", "auto"), n_gpu_layers=m.get("n_gpu_layers", "auto"),
            chat_format=m.get("chat_format", ""), mmproj=m.get("mmproj", ""),
        )
        self._last_ctx = (plan.n_ctx, plan.n_ctx_max)
        self._last_ctx_why = plan.ctx_why
        return plan.cmd, plan.why, plan.env or {}

    def start(self, name: str, *, wait: bool = True, timeout: float = 180.0) -> str:
        # Locked from the is_running() check through committing the spawned
        # process to running.json: that's the exact window where two
        # concurrent start() calls for the same model (two characters
        # sharing it, both taking their first turn at once) could otherwise
        # both see "not running" and both spawn a server on the same port.
        # NOT held through _wait_ready below -- that can take up to
        # `timeout` seconds and has nothing to do with this model's own
        # registry entry, so holding the lock there would serialize starting
        # totally unrelated models too.
        with self._lock:
            m = self.get(name)
            if not m:
                raise ModelError(
                    f"unknown model '{name}'. Add it first: pleiades model add {name} <path.gguf>"
                )
            if m.get("external_url"):
                # Nothing to spawn -- confirm it's actually reachable (a
                # start() that returns a URL nothing answers on is a worse
                # failure mode than an honest error here) and hand back the
                # same URL base_url() would.
                if self.state(name) != "running":
                    raise ModelError(
                        f"external model '{name}' at {m['external_url']} is not reachable"
                    )
                return self.base_url(name)
            if self.is_running(name):
                return self.base_url(name)

            # The registered port may be occupied by an ORPHAN OF OURSELVES: if
            # running.json went stale (crash/manual kill) the old server keeps
            # serving while is_running() says dead — relocating would then spawn
            # a duplicate per chat. If whatever holds the port answers /v1/models
            # with our alias, adopt it instead of spawning another.
            if not _port_free(m["host"], int(m["port"])) and self._is_our_server(m):
                run = self._load_running()
                run[name] = {"pid": _pid_on_port(m["host"], int(m["port"])) or 0,
                             "host": m["host"], "port": m["port"]}
                self._save_running(run)
                return self.base_url(name)

            # The registered port may have been taken by another app since add()
            # (e.g. a Docker service). Detect it now and relocate instead of letting
            # llama.cpp load the whole model and then die on bind.
            port_note = ""
            if not _port_free(m["host"], int(m["port"])):
                reg = self._load()
                old_port = m["port"]
                new_port = self._free_port(reg, host=m["host"])
                reg[name]["port"] = new_port
                self._save(reg)
                m = reg[name]
                port_note = (f"[pleiades] port {old_port} is in use by another "
                             f"application — moved '{name}' to port {new_port}\n")

            cmd, why, plan_env = self._build_launch(m)

            logdir = config.PLEIADES_HOME / "logs"
            logdir.mkdir(parents=True, exist_ok=True)
            logf = open(logdir / f"model-{name}.log", "ab")
            if port_note:
                logf.write(port_note.encode())
            if getattr(self, "_last_ctx_why", ""):
                logf.write(f"[pleiades] ctx plan: {self._last_ctx_why}\n".encode())
            logf.write(f"[pleiades] {why}\n".encode())
            logf.flush()

            kwargs: dict = {}
            if os.name == "posix":
                kwargs["start_new_session"] = True          # survive the CLI exiting
            else:
                kwargs["creationflags"] = 0x00000200        # CREATE_NEW_PROCESS_GROUP

            if plan_env:
                kwargs["env"] = {**os.environ, **plan_env}
            try:
                proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, **kwargs)
            except FileNotFoundError as e:
                raise ModelError(
                    "llama_cpp.server not available. Install with "
                    "`pip install 'llama-cpp-python[server]'`."
                ) from e

            run = self._load_running()
            n_ctx_run, n_ctx_max_run = getattr(self, "_last_ctx", (None, None))
            run[name] = {"pid": proc.pid, "host": m["host"], "port": m["port"],
                         "n_ctx": n_ctx_run, "n_ctx_max": n_ctx_max_run}
            self._save_running(run)

        if wait:
            self._wait_ready(name, proc, timeout)
            # Best-effort: if a previous graceful stop saved this model's
            # slot state (see stop()), restore it now instead of paying a
            # full persona/memory reprocess on the first real request.
            # Silently no-ops for models not started with --slot-save-path,
            # a first-ever start (nothing saved yet), or any restore
            # failure (corrupt/incompatible file) -- falls through to an
            # ordinary cold-context start in every one of those cases,
            # same as before this feature existed.
            _slot_restore(m["host"], int(m["port"]), name)
        return self.base_url(name)

    def _wait_ready(self, name: str, proc: subprocess.Popen, timeout: float) -> None:
        """Block until the server is actually answering HTTP (state == 'running'),
        not merely until the process exists (state == 'loading') — a caller that
        waits expects to be able to send a request the moment this returns."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.state(name) == "running":
                return
            if proc.poll() is not None:
                raise ModelError(
                    f"model '{name}' server exited early (code {proc.returncode}). "
                    f"See {config.PLEIADES_HOME / 'logs' / ('model-' + name + '.log')}"
                )
            time.sleep(0.5)
        raise ModelError(f"model '{name}' not ready after {timeout:.0f}s.")

    def _is_our_server(self, m: dict) -> bool:
        """Does whatever holds the model's port answer as this model?"""
        import httpx

        try:
            r = httpx.get(f"http://{m['host']}:{m['port']}/v1/models", timeout=3.0)
            ids = {x.get("id", "") for x in r.json().get("data", [])}
            return m["name"] in ids
        except Exception:
            return False

    def stop(self, name: str) -> bool:
        run = self._load_running()
        r = run.get(name)
        if not r:
            return False
        pid = r.get("pid")
        if _pid_alive(pid):
            # Best-effort, best BEFORE the kill signal: ask the server to
            # persist its slot state to disk while it's still alive and
            # responsive (see launch.py's --slot-save-path and start()'s
            # matching _slot_restore call). Silently no-ops if this model
            # wasn't started with slot-save-path, has no active state, or
            # the request fails for any reason -- must never block stop().
            _slot_save(r.get("host", "127.0.0.1"), int(r.get("port") or 0))
            try:
                if os.name == "posix":
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                else:
                    os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        # running.json can be stale — also kill whoever still holds the port,
        # or "Stopped" is a lie and orphans accumulate one per restart. Only
        # if it answers as THIS model though; never terminate a foreign app.
        host = r.get("host", "127.0.0.1")
        port = int(r.get("port") or 0)
        port_pid = _pid_on_port(host, port)
        if (port_pid and port_pid != pid
                and self._is_our_server({"host": host, "port": port, "name": name})):
            try:
                os.kill(port_pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        # Re-fetch right before the delete+save rather than reusing the
        # `run` snapshot from the top of this method: _slot_save above can
        # take up to 120s, and holding the lock for that whole window would
        # block unrelated models' start()/stop() calls for no reason. This
        # keeps the actual read-modify-write atomic without penalizing
        # concurrent operations on other models.
        with self._lock:
            run = self._load_running()
            run.pop(name, None)
            self._save_running(run)
        return True

    def resize(self, name: str, n_ctx: int) -> dict:
        """Resize a RUNNING elastic server's context window in place."""
        run = self._load_running().get(name)
        if not run:
            raise ModelError(f"model '{name}' is not running")
        try:
            r = httpx.post(f"http://{run['host']}:{run['port']}/resize",
                           json={"n_ctx": int(n_ctx)}, timeout=600.0)
        except httpx.HTTPError as e:
            raise ModelError(f"resize failed: {e}") from e
        if r.status_code == 404:
            raise ModelError(
                "this model is not running on the elastic backend (native "
                "llama-server or PLEIADES_ENGINE=llama_cpp) — restart to resize")
        if r.status_code != 200:
            raise ModelError(f"resize failed: {r.text}")
        out = r.json()
        allr = self._load_running()
        if name in allr:
            allr[name]["n_ctx"] = out.get("n_ctx", int(n_ctx))
            self._save_running(allr)
        return out

    def running(self) -> list[dict]:
        out = []
        for name, r in self._load_running().items():
            out.append({"name": name, "pid": r.get("pid"),
                        "port": r.get("port"), "alive": self.is_running(name)})
        return out
