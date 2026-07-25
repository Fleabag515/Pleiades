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

from pathlib import Path

from . import config, runtime
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
    env: Optional[dict] = None   # extra environment for the server process


def build_command(model_path: str, host: str, port: int, *, name: str = "local",
                  n_ctx: "int | str" = "auto", n_gpu_layers: "int | str" = "auto",
                  chat_format: str = "", settings: Optional[Settings] = None,
                  mmproj: str = "") -> LaunchPlan:
    """Plan + assemble the server command: MoE-aware on the native llama-server
    runtime (autodetected), layer-split on the bundled python fallback.

    `mmproj` (path to a multimodal-projector GGUF, see models.Model.mmproj) is
    ONLY ever wired into the native llama-server branch below via --mmproj.
    Scoped there deliberately, not to the `llama_cpp.server` python fallback
    or the PLEIADES_ENGINE=pleiades_native custom engine branch above: llama.
    cpp's own bundled server has no first-class multimodal flag in the
    version pinned here, and the from-scratch native C++ engine
    (engine/http_server.cpp) doesn't implement image handling at all yet
    (see docs/specs/2026-07-23-vision-routing-design.md) -- silently
    dropping `mmproj` on those two paths is intentional, not an oversight.
    """
    eff = settings or Settings.load()

    # meta before caps: with two managed runtimes (mainline + the MoE-
    # prefill fork), which binary we prefer depends on whether THIS model
    # is MoE.
    meta = read_gguf_meta(model_path)

    # Feature-flagged, off by default (docs/specs/2026-07-21-native-
    # inference-engine-design.md, Phase 5): the from-scratch libllama-based
    # engine, once it clears the parity bar Phase 4 benchmarked. Kept as
    # its own branch with its own minimal command construction rather than
    # slotting into the native-llama-server branch below -- that branch
    # assumes whatever binary it gets back understands llama-server's full
    # flag set (flash-attention toggle, KV quantization, MoE expert
    # offload, speculative decoding), none of which this engine implements
    # yet. See find_native_cpp_engine()'s own docstring for the reasoning
    # (checked against a second independent opinion, not just assumed).
    if os.environ.get("PLEIADES_ENGINE", "").lower() == "pleiades_native":
        native_cpp = runtime.find_native_cpp_engine()
        if native_cpp:
            n_ctx_v, n_ctx_max, ctx_why = resolve_context(n_ctx, model_path)
            launch_ctx = n_ctx_max
            forced = None
            if not (isinstance(n_gpu_layers, str) and n_gpu_layers.strip().lower() in ("", "auto")):
                try:
                    forced = int(n_gpu_layers)
                except (TypeError, ValueError):
                    forced = None
            ngl = forced if forced is not None else 0  # no autofit/MoE-split support yet -- CPU-only until Phase 6+
            cmd = [native_cpp, model_path, host, str(port), str(launch_ctx), str(ngl), name]
            why = (f"runtime=pleiades-native-engine (Phase 5, experimental) "
                   f"n_ctx={launch_ctx} n_gpu_layers={ngl}")
            return LaunchPlan(cmd, why, ctx_why, launch_ctx, n_ctx_max)
        print("[launch] PLEIADES_ENGINE=pleiades_native requested but "
              "pleiades-engine-server not found (build via `cmake --build "
              "engine/build`, or set PLEIADES_NATIVE_CPP_ENGINE_BIN) -- "
              "falling back to the normal runtime selection.", flush=True)

    cps = runtime.caps(prefer_moe_opts=meta.is_moe)
    native = runtime.find_native(prefer_moe_opts=meta.is_moe) if cps.native else None
    # caps before resolve_context: a MoE model's real ceiling depends on
    # whether THIS runtime actually offloads experts (hardware.py's
    # _moe_ceiling_fits) — without it, ctx planning falls back to the
    # dense-only check and starves a MoE model down to the smallest gear.
    n_ctx_v, n_ctx_max, ctx_why = resolve_context(n_ctx, model_path, caps=cps)
    threads = max((os.cpu_count() or 8) // 2, 4)   # physical cores beat SMT
    # Prefill/batch processing is compute-bound (more threads help, up to
    # physical core count) but decode is memory-bandwidth-bound and can
    # actually get WORSE past a point once threads start contending for the
    # same memory bus instead of adding throughput. Real measurement on a
    # Ryzen 7 5700X (8 physical cores): decode tok/s peaked at 4 threads,
    # +10% over 8, and collapsed 3x at 16 (SMT). Halving the physical count
    # is a reasonable general default (llama-server's own docs recommend
    # starting near physical-core-count/2 for memory-bound decode on
    # desktop parts) but this exact ratio is only verified on this one box
    # -- an explicit --n-gpu-layers-style override doesn't exist for this
    # yet; if per-machine calibration lands (see autofit follow-up), this
    # is the value it should replace.
    decode_threads = max(threads // 2, 2)

    # Placement must be sized against whichever ctx will ACTUALLY launch —
    # native runs fixed at n_ctx_max (see below), so planning layers/expert
    # split against the smaller n_ctx_v would under-provision VRAM for the
    # real KV cost at the real window (the swap-thrash bug fixed earlier
    # this session, reintroduced via a mismatched ctx instead of a bad margin).
    place_ctx = n_ctx_max if native else n_ctx_v
    pl = place(meta, place_ctx, caps=cps if native else RuntimeCaps())
    forced = None
    if not (isinstance(n_gpu_layers, str) and n_gpu_layers.strip().lower() in ("", "auto")):
        try:
            forced = int(n_gpu_layers)
        except (TypeError, ValueError):
            forced = None

    if native:
        # llama-server's context window is fixed at launch — ModelManager.resize()
        # 404s on it, there's no live grow-into-the-ceiling like the python
        # elastic engine. So "start modest, elastic ceiling later" (n_ctx_v)
        # would just be a smaller, permanent window for no benefit: launch
        # directly at n_ctx_max, the ceiling plan_context() already verified
        # fits. (Pinned configs are unaffected — there n_ctx_v == n_ctx_max.)
        launch_ctx = n_ctx_max
        ngl = forced if forced is not None else (999 if pl.n_gpu_layers == -1
                                                  else pl.n_gpu_layers)
        # Persists full sequence state (KV + any recurrent/SSM state — the
        # same llama_state_seq_get/set_data primitives already verified this
        # session to handle hybrid/recurrent models correctly, just with
        # flags=0/"full" instead of the context-checkpoint mechanism's
        # PARTIAL_ONLY) to disk on demand via POST /slots/0?action=save|
        # restore. Directory only, not a flag toggle — costs nothing when
        # unused (ModelManager only calls save/restore around stop/start,
        # see models.py), and having it always available means a save file
        # from a previous run is picked up even if slot support was added
        # after that model was first registered.
        slot_dir = config.PLEIADES_HOME / "slots" / name
        slot_dir.mkdir(parents=True, exist_ok=True)
        cmd = [native, "-m", model_path,
               "--host", host, "--port", str(port),
               "-c", str(launch_ctx), "-ngl", str(ngl),
               "-t", str(decode_threads), "-tb", str(threads),
               "--alias", name,
               "--jinja",
               "--slot-save-path", str(slot_dir) + os.sep,
               # One character = one conversation at a time through its Anamnesis
               # proxy. llama-server's "auto" default spins up 4 parallel slots,
               # each with its own compute-buffer VRAM allocation — wasted memory
               # we'd rather have as headroom (see autofit's CPU-bias margin).
               "--parallel", "1"]
        if mmproj and os.path.isfile(mmproj):
            # Live-verified (2026-07-23, real ggml-org/SmolVLM-500M-Instruct
            # and ggml-org/Qwen2.5-VL-3B-Instruct GGUFs + their mmproj
            # sidecars, this project's pinned llama-server build): --mmproj
            # loads the multimodal projector alongside the model and
            # /v1/chat/completions correctly renders an OpenAI content-parts
            # image_url part through --jinja's chat-template path -- the
            # model's reply demonstrably reflected the actual image content,
            # not a hallucinated generic description. See
            # docs/specs/2026-07-23-vision-routing-design.md for the full
            # verification transcript.
            cmd += ["--mmproj", mmproj]
        fa = eff.flash_attn
        if eff.kv_cache_type and fa == "auto":
            # A quantized V-cache requires flash attention; "auto" can
            # resolve to off for some architectures and then -ctv fails at
            # boot. Forcing it on is safe everywhere -ctv is requested.
            fa = "on"
        if fa:
            cmd += ["-fa", fa]
        if eff.kv_cache_type:          # KV-cache compaction (e.g. q8_0)
            cmd += ["-ctk", eff.kv_cache_type, "-ctv", eff.kv_cache_type]
        if getattr(eff, "cache_reuse", 0):
            # Re-use matching KV prefix chunks across requests. Anamnesis
            # rewrites the memory block every turn, which shifts everything
            # after it — chunk re-use still salvages most of the re-prefill.
            cmd += ["--cache-reuse", str(eff.cache_reuse)]
        if getattr(eff, "n_batch", 0):
            cmd += ["-b", str(eff.n_batch)]
        # Explicit PLEIADES_N_UBATCH always wins; otherwise take autofit's
        # hardware-sized pick (0 = runtime default, no flag passed).
        ub = getattr(eff, "n_ubatch", 0) or pl.n_ubatch
        if not ub and pl.n_cpu_moe:
            # MoE-with-CPU-experts prefill is upload-bound: every ubatch pass
            # re-uploads the layer's touched experts, so bigger physical
            # batches amortize the transfers. Measured on this box, same
            # model/split: pp 112 t/s at the 512 default → 735 t/s at 2048.
            # Autofit's near-tie headroom preference can pick a candidate
            # with ub=0 — restore it whenever experts live on CPU (the hot
            # set is tiny; the larger compute buffer fits comfortably).
            ub = 2048
        if ub:
            cmd += ["-ub", str(ub)]
        if getattr(eff, "mlock", False):
            cmd += ["--mlock"]
        env = dict(runtime.native_env(native))  # relocatable managed builds
        st = str(getattr(eff, "spec_type", "auto")).strip().lower()
        if getattr(eff, "draft_model_path", "") and os.path.isfile(eff.draft_model_path):
            cmd += ["-md", eff.draft_model_path]   # speculative decoding draft
        elif st == "auto":
            # Measured HARMFUL for persona chat on this box: ngram-simple
            # drafted 6.7-token runs at 16.5% acceptance and decode fell
            # 26.8 → 8.3 t/s (the target model verifies-and-discards almost
            # every draft). Creative replies aren't repetitive enough.
            # "auto" therefore means OFF; opt in via PLEIADES_SPEC_TYPE for
            # workloads that ARE repetitive (code edits, quoting tool
            # output), where ngram acceptance — and the speedup — is real.
            pass
        elif st not in ("", "off", "none"):
            cmd += ["--spec-type", str(eff.spec_type).strip()]
        if forced is None and pl.n_cpu_moe and cps.moe_offload:
            cmd += ["--n-cpu-moe", str(pl.n_cpu_moe)]
            if getattr(eff, "moe_prefill_opts", True):
                # Fable's MoE-offload prefill opts (thecodacus/llama.cpp
                # fork): page-lock the host-resident expert weights so
                # uploads run at DMA speed, and prefetch each layer's
                # experts on a second CUDA stream so uploads overlap
                # compute. Token-identical; a mainline build ignores both
                # vars, so setting them is always safe.
                env.update({"GGML_CUDA_REGISTER_HOST": "1",
                            "GGML_SCHED_PREFETCH_EXPERTS": "1"})
        why = (f"runtime=llama-server strategy={pl.strategy} "
               f"ngl={ngl}" + (f" n_cpu_moe={pl.n_cpu_moe}" if pl.n_cpu_moe else "")
               + (f" ub={ub}" if ub else "")
               + (" +moe-prefill-opts" if "GGML_SCHED_PREFETCH_EXPERTS" in env else "")
               + (" +mmproj (vision)" if mmproj and os.path.isfile(mmproj) else "")
               + f" est={pl.est_tps} tok/s — {pl.reason}")
        if forced is not None:
            why = f"runtime=llama-server ngl={forced} (explicit override)"
        # n_ctx == n_ctx_max here: what's recorded/shown is what's really running.
        return LaunchPlan(cmd, why, ctx_why, launch_ctx, n_ctx_max, env)

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
