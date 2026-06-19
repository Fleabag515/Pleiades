"""Shared server-launch planner — the one place that decides HOW to run a GGUF.

Two call sites used to each hand-roll this: the multi-model registry
(`models.py`, backing the webui "model foundry") had the real MoE-aware,
native-runtime-aware logic; the single-model legacy engine
(`inference/__init__.py`, used by `pleiades serve` / the default chat tier)
only ever did a dense layer-split and never looked at the native runtime or
autofit at all. Same job, two implementations, one of them dumb — that's the
"model foundry kinda sucked" / inconsistent-performance bug. Now there's one
implementation; both callers just ask it for a command.

Autodetect/autoconfig, end to end:
  hardware.py  -> GPU/RAM/CPU + GGUF structure (dense facts)
  runtime.py   -> is the native llama-server binary installed, and does it
                  support MoE expert offload (--n-cpu-moe)?
  autofit.py   -> given those facts, the fastest feasible placement: full GPU,
                  MoE hot/cold split, partial dense layers, or CPU-only
  this module  -> turn that placement into an actual command line, on
                  whichever runtime is actually available
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional

from . import runtime
from .autofit import RuntimeCaps, place
from .config import Settings
from .hardware import read_gguf_meta, resolve_context


@dataclass
class LaunchPlan:
    cmd: list[str]
    why: str            # placement strategy + estimated tok/s, for logs/UI
    ctx_why: str         # context-window planning reason, for logs/UI
    n_ctx: int
    n_ctx_max: int


def build_command(model_path: str, host: str, port: int, *, name: str = "local",
                  n_ctx: "int | str" = "auto", n_gpu_layers: "int | str" = "auto",
                  chat_format: str = "", settings: Optional[Settings] = None) -> LaunchPlan:
    """Plan + assemble the server command: MoE-aware on the native llama-server
    runtime (autodetected), layer-split on the bundled python fallback."""
    eff = settings or Settings.load()

    n_ctx_v, n_ctx_max, ctx_why = resolve_context(n_ctx, model_path)
    meta = read_gguf_meta(model_path)
    cps = runtime.caps()
    native = runtime.find_native() if cps.native else None
    threads = max((os.cpu_count() or 8) // 2, 4)   # physical cores beat SMT

    pl = place(meta, n_ctx_v, caps=cps if native else RuntimeCaps())
    forced = None
    if not (isinstance(n_gpu_layers, str) and n_gpu_layers.strip().lower() in ("", "auto")):
        try:
            forced = int(n_gpu_layers)
        except (TypeError, ValueError):
            forced = None

    if native:
        ngl = forced if forced is not None else (999 if pl.n_gpu_layers == -1
                                                  else pl.n_gpu_layers)
        cmd = [native, "-m", model_path,
               "--host", host, "--port", str(port),
               "-c", str(n_ctx_v), "-ngl", str(ngl), "-t", str(threads),
               "--alias", name,
               "--jinja"]
        if eff.flash_attn:
            cmd += ["-fa", eff.flash_attn]
        if eff.kv_cache_type:          # KV-cache compaction (e.g. q8_0)
            cmd += ["-ctk", eff.kv_cache_type, "-ctv", eff.kv_cache_type]
        if getattr(eff, "n_batch", 0):
            cmd += ["-b", str(eff.n_batch)]
        if getattr(eff, "n_ubatch", 0):
            cmd += ["-ub", str(eff.n_ubatch)]
        if getattr(eff, "mlock", False):
            cmd += ["--mlock"]
        if getattr(eff, "draft_model_path", "") and os.path.isfile(eff.draft_model_path):
            cmd += ["-md", eff.draft_model_path]   # speculative decoding draft
        if forced is None and pl.n_cpu_moe and cps.moe_offload:
            cmd += ["--n-cpu-moe", str(pl.n_cpu_moe)]
        why = (f"runtime=llama-server strategy={pl.strategy} "
               f"ngl={ngl}" + (f" n_cpu_moe={pl.n_cpu_moe}" if pl.n_cpu_moe else "")
               + f" est={pl.est_tps} tok/s — {pl.reason}")
        if forced is not None:
            why = f"runtime=llama-server ngl={forced} (explicit override)"
        return LaunchPlan(cmd, why, ctx_why, n_ctx_v, n_ctx_max)

    layers = forced if forced is not None else pl.n_gpu_layers
    if os.environ.get("PLEIADES_ENGINE", "elastic").lower() == "llama_cpp":
        # escape hatch: bundled llama-cpp-python server (fixed window)
        cmd = [
            sys.executable, "-m", "llama_cpp.server",
            "--model", model_path,
            "--model_alias", name,
            "--host", host, "--port", str(port),
            "--n_ctx", str(n_ctx_v),
            "--n_gpu_layers", str(layers),
            "--n_threads", str(threads),
            "--flash_attn", "true",
        ]
        if chat_format:
            cmd += ["--chat_format", chat_format]
    else:
        # our elastic server: weights stay loaded, KV resizes in place
        cmd = [
            sys.executable, "-m", "pleiades.inference.server",
            "--model", model_path,
            "--host", host, "--port", str(port),
            "--n-ctx", str(n_ctx_v), "--n-ctx-max", str(n_ctx_max),
            "--n-gpu-layers", str(layers),
            "--alias", name,
        ]
        if chat_format:
            cmd += ["--chat-format", chat_format]
    why = (f"runtime=python strategy={pl.strategy} n_gpu_layers={layers} "
           f"est={pl.est_tps} tok/s — {pl.reason}")
    if forced is not None:
        why = f"runtime=python n_gpu_layers={forced} (explicit override)"
    if meta.is_moe and not cps.moe_offload:
        why += (" · TIP: `pleiades runtime install` adds the native runtime "
                "with MoE expert offload — much faster for this model")
    return LaunchPlan(cmd, why, ctx_why, n_ctx_v, n_ctx_max)
