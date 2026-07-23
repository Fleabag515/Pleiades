"""Model registry + lifecycle — add GGUFs, run one server per model, assign to characters.

Pleiades is an inference engine: it runs models itself via llama.cpp. This module lets
you register multiple GGUF models by name, start/stop a dedicated OpenAI-compatible
server for each (on its own port), and choose which character uses which model.

State (under ~/.pleiades):
  models.json          registered models: name -> {path, n_ctx, n_gpu_layers, chat_format, host, port}
  models-running.json  live servers: name -> {pid, host, port}
  logs/model-<name>.log per-model server log

GPU: n_gpu_layers controls offload and works for both NVIDIA (CUDA) and AMD (ROCm)
*provided llama-cpp-python was built with that backend* (the installer auto-detects).
-1 offloads all layers; 0 is CPU-only.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
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
    host: str = "127.0.0.1"
    port: int = 0             # assigned on add()
    # Vision support (docs/specs/2026-07-23-vision-routing-design.md). Path to
    # a sibling multimodal-projector GGUF, auto-detected at add()/fetch time
    # via hardware.find_mmproj_sibling() (blank = none found / not applicable).
    # Only ever consumed by the native llama-server launch path (launch.py) --
    # the llama_cpp.server python fallback and PLEIADES_ENGINE=pleiades_native
    # don't implement multimodal serving, so this field is a no-op there.
    mmproj: str = ""
    # Manual capability override, comma-separated (e.g. "vision"). For a local
    # GGUF this is normally redundant with `mmproj` (presence of the sidecar
    # already implies vision) -- it exists mainly for cloud models
    # (openrouter:/ollama-cloud: prefixed profile.model strings), which have
    # no local file to sniff and no entry in this registry at all; see
    # Profile.model_capabilities for where the cloud-model equivalent lives.
    capabilities: str = ""

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


def _pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
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


class ModelManager:
    """Register, run, stop, and assign local GGUF models."""

    def __init__(self) -> None:
        config.ensure_home()

    # -- registry ----------------------------------------------------------- #
    def _load(self) -> dict:
        """Registry, pruned against what's actually on disk.

        models.json is bookkeeping, not truth — if a GGUF was deleted (by a
        previous remove(), or by hand) its entry must not go on appearing in
        the library forever. Every read goes through here so list()/get()/
        add() all see the reconciled state, not a stale in-memory picture.
        """
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
        """Drop registry entries whose GGUF no longer exists on disk."""
        changed = False
        for name in list(reg.keys()):
            path = Path(reg[name].get("path", "")).expanduser()
            if not path.is_file():
                del reg[name]
                changed = True
        return changed

    def _save(self, data: dict) -> None:
        _models_json().write_text(json.dumps(data, indent=2), encoding="utf-8")

    def add(self, name: str, path: str, *, n_ctx: "int | str" = "auto",
            n_gpu_layers: "int | str" = "auto",
            chat_format: str = "", port: int = 0,
            mmproj: str = "", capabilities: str = "") -> dict:
        p = Path(path).expanduser()
        if not p.is_file():
            raise ModelError(f"No such model file: {p}")
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
        p = _running_json()
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_running(self, data: dict) -> None:
        _running_json().write_text(json.dumps(data, indent=2), encoding="utf-8")

    def base_url(self, name: str) -> str:
        m = self.get(name)
        if not m:
            raise ModelError(f"unknown model '{name}'")
        return f"http://{m['host']}:{m['port']}/v1"

    def state(self, name: str) -> str:
        """'running' (HTTP healthy) | 'loading' (process up, API not yet) |
        'crashed' (process died while registered) | 'stopped'."""
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
        m = self.get(name)
        if not m:
            raise ModelError(
                f"unknown model '{name}'. Add it first: pleiades model add {name} <path.gguf>"
            )
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
        del run[name]
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
