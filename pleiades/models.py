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
    n_ctx: int = 8192
    # "auto" plans offload from detected hardware at each launch; int overrides
    # (0 = CPU, -1 = all layers on GPU — CUDA, ROCm, or Metal build).
    n_gpu_layers: "int | str" = "auto"
    chat_format: str = ""     # blank = llama.cpp auto-detect
    host: str = "127.0.0.1"
    port: int = 0             # assigned on add()


class ModelError(RuntimeError):
    pass


def _pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


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

    def add(self, name: str, path: str, *, n_ctx: int = 8192,
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

    def _free_port(self, reg: dict, start: int = 8090) -> int:
        used = {m.get("port") for m in reg.values()}
        used |= {r.get("port") for r in self._load_running().values()}
        port = start
        while port in used:
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

    def start(self, name: str, *, wait: bool = True, timeout: float = 180.0) -> str:
        m = self.get(name)
        if not m:
            raise ModelError(
                f"unknown model '{name}'. Add it first: pleiades model add {name} <path.gguf>"
            )
        if self.is_running(name):
            return self.base_url(name)

        from .hardware import resolve_layers
        layers, why = resolve_layers(m.get("n_gpu_layers", "auto"),
                                     m["path"], int(m.get("n_ctx", 8192)))
        cmd = [
            sys.executable, "-m", "llama_cpp.server",
            "--model", m["path"],
            "--host", m["host"], "--port", str(m["port"]),
            "--n_ctx", str(m["n_ctx"]),
            "--n_gpu_layers", str(layers),
        ]
        if m.get("chat_format"):
            cmd += ["--chat_format", m["chat_format"]]

        logdir = config.PLEIADES_HOME / "logs"
        logdir.mkdir(parents=True, exist_ok=True)
        logf = open(logdir / f"model-{name}.log", "ab")
        logf.write(f"[pleiades] n_gpu_layers={layers} — {why}\n".encode())
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
        run[name] = {"pid": proc.pid, "host": m["host"], "port": m["port"]}
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
        del run[name]
        self._save_running(run)
        return True

    def running(self) -> list[dict]:
        out = []
        for name, r in self._load_running().items():
            out.append({"name": name, "pid": r.get("pid"),
                        "port": r.get("port"), "alive": self.is_running(name)})
        return out
