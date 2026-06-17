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
import sys
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
    host: str = "127.0.0.1"
    port: int = 0             # assigned on add()


class ModelError(RuntimeError):
    pass


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
        p = _models_json()
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save(self, data: dict) -> None:
        _models_json().write_text(json.dumps(data, indent=2), encoding="utf-8")

    def add(self, name: str, path: str, *, n_ctx: "int | str" = "auto",
            n_gpu_layers: "int | str" = "auto",
            chat_format: str = "", port: int = 0) -> dict:
        p = Path(path).expanduser()
        if not p.is_file():
            raise ModelError(f"No such model file: {p}")
        reg = self._load()
        port = port or self._free_port(reg)
        reg[name] = asdict(Model(name=name, path=str(p), n_ctx=n_ctx,
                                 n_gpu_layers=n_gpu_layers, chat_format=chat_format,
                                 port=port))
        self._save(reg)
        return reg[name]

    def remove(self, name: str) -> bool:
        reg = self._load()
        if name not in reg:
            return False
        self.stop(name)
        del reg[name]
        self._save(reg)
        return True

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

    def _build_launch(self, m: dict) -> tuple[list, str]:
        """Assemble the server command via autofit: MoE-aware on the native
        llama-server runtime, layer-split on the python fallback."""
        from . import runtime
        from .autofit import RuntimeCaps, place
        from .hardware import read_gguf_meta

        from .hardware import resolve_context
        n_ctx, n_ctx_max, ctx_why = resolve_context(m.get("n_ctx", "auto"), m["path"])
        self._last_ctx = (n_ctx, n_ctx_max)
        self._last_ctx_why = ctx_why
        explicit = m.get("n_gpu_layers", "auto")
        meta = read_gguf_meta(m["path"])
        cps = runtime.caps()
        native = runtime.find_native() if cps.native else None
        threads = max((os.cpu_count() or 8) // 2, 4)   # physical cores beat SMT

        pl = place(meta, n_ctx, caps=cps if native else RuntimeCaps())
        forced = None
        if not (isinstance(explicit, str) and str(explicit).strip().lower() in ("", "auto")):
            try:
                forced = int(explicit)
            except (TypeError, ValueError):
                forced = None

        if native:
            ngl = forced if forced is not None else (999 if pl.n_gpu_layers == -1
                                                     else pl.n_gpu_layers)
            cmd = [native, "-m", m["path"],
                   "--host", m["host"], "--port", str(m["port"]),
                   "-c", str(n_ctx), "-ngl", str(ngl), "-t", str(threads),
                   "--alias", m["name"],
                   "--jinja"]
            if forced is None and pl.n_cpu_moe and cps.moe_offload:
                cmd += ["--n-cpu-moe", str(pl.n_cpu_moe)]
            why = (f"runtime=llama-server strategy={pl.strategy} "
                   f"ngl={ngl}" + (f" n_cpu_moe={pl.n_cpu_moe}" if pl.n_cpu_moe else "")
                   + f" est={pl.est_tps} tok/s — {pl.reason}")
            if forced is not None:
                why = f"runtime=llama-server ngl={forced} (explicit override)"
            return cmd, why

        layers = forced if forced is not None else pl.n_gpu_layers
        if os.environ.get("PLEIADES_ENGINE", "elastic").lower() == "llama_cpp":
            # escape hatch: bundled llama-cpp-python server (fixed window)
            cmd = [
                sys.executable, "-m", "llama_cpp.server",
                "--model", m["path"],
                "--model_alias", m["name"],
                "--host", m["host"], "--port", str(m["port"]),
                "--n_ctx", str(n_ctx),
                "--n_gpu_layers", str(layers),
                "--n_threads", str(threads),
                "--flash_attn", "true",
            ]
            if m.get("chat_format"):
                cmd += ["--chat_format", m["chat_format"]]
        else:
            # our elastic server: weights stay loaded, KV resizes in place
            # (POST /resize; prompt overflow auto-upshifts within the ceiling)
            cmd = [
                sys.executable, "-m", "pleiades.inference.server",
                "--model", m["path"],
                "--host", m["host"], "--port", str(m["port"]),
                "--n-ctx", str(n_ctx), "--n-ctx-max", str(n_ctx_max),
                "--n-gpu-layers", str(layers),
                "--alias", str(m.get("name", "local")),
            ]
            if m.get("chat_format"):
                cmd += ["--chat-format", m["chat_format"]]
        why = (f"runtime=python strategy={pl.strategy} n_gpu_layers={layers} "
               f"est={pl.est_tps} tok/s — {pl.reason}")
        if forced is not None:
            why = f"runtime=python n_gpu_layers={forced} (explicit override)"
        if meta.is_moe and not cps.moe_offload:
            why += (" · TIP: `pleiades runtime install` adds the native runtime "
                    "with MoE expert offload — much faster for this model")
        return cmd, why

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

        cmd, why = self._build_launch(m)

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
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_running(name):
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
