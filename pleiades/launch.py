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
  runtime.py   -> is Pleiades' own engine binary installed (find_native_cpp_engine),
                  or failing that the autodetected llama-server binary
                  (find_native), and does it support MoE expert offload
                  (--n-cpu-moe)?
  autofit.py   -> given those facts, the fastest feasible placement: full GPU,
                  MoE hot/cold split, partial dense layers, or CPU-only
  this module  -> turn that placement into an actual command line, on
                  whichever runtime is actually available

Engine selection order (docs/specs/2026-07-21-native-inference-engine-design.md,
Phase 9.6 "Release N"): Pleiades' own from-scratch engine
(`engine/src/http_server.cpp`, `runtime.find_native_cpp_engine()`) is the
default whenever its binary resolves — no env var needed. `PLEIADES_ENGINE`
opts OUT of it: `llama_server` forces the autodetected upstream llama-server
binary instead (the sanctioned escape hatch for hardware this engine hasn't
been verified on yet — see the design doc's Risk R1/R2); `llama_cpp` forces
the pip-installed `llama_cpp.server` fallback. `pleiades_native` still works
explicitly too, for back-compat with anything already setting it. Unset +
no engine binary found falls through to the same chain as before (native
llama-server, then the bundled python "elastic" server).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional


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


def _query_engine_caps(binary_path: str) -> RuntimeCaps:
    """`<binary> --caps` -> RuntimeCaps, for the from-scratch C++ engine only.

    Cheap (no model load, see http_server.cpp::print_caps()) and per-BUILD
    accurate -- unlike guessing from a filename/path convention (the pattern
    runtime.caps() still uses for the OTHER native runtime, see that
    function), this asks the actual binary what it was built with. Any
    failure here (older binary predating --caps, unexpected output, a hung
    process) falls back to the previous behavior of this call site
    (moe_offload=True) rather than breaking model placement on an
    introspection hiccup -- see call site's comment. The raw subprocess/JSON
    work now lives in runtime.query_native_cpp_engine_caps() (shared with
    `pleiades runtime status` and the webui's /api/hardware, which want the
    full answer, not just the moe_offload bit this call site narrows down to).
    """
    data = runtime.query_native_cpp_engine_caps(binary_path)
    if data is None:
        return RuntimeCaps(moe_offload=True, native=True)
    return RuntimeCaps(moe_offload=bool(data.get("moe_offload", True)), native=True)


def build_command(model_path: str, host: str, port: int, *, name: str = "local",
                  n_ctx: "int | str" = "auto", n_gpu_layers: "int | str" = "auto",
                  chat_format: str = "", settings: Optional[Settings] = None,
                  mmproj: str = "") -> LaunchPlan:
    """Plan + assemble the server command: MoE-aware on Pleiades' own engine or
    the native llama-server runtime (both autodetected), layer-split on the
    bundled python fallback.

    `mmproj` (path to a multimodal-projector GGUF, see models.Model.mmproj) is
    wired into the native llama-server branch below via --mmproj, and (Phase
    9.4.2, docs/specs/2026-07-21-native-inference-engine-design.md) into the
    default engine branch above the same way -- it links libmtmd now (see
    engine/CMakeLists.txt's LLAMA_BUILD_MTMD). Still NOT wired into the
    `llama_cpp.server` python fallback: that package's own bundled server has
    no first-class multimodal flag in the version pinned here, and that path
    is slated for retirement anyway (Phase 9.6) -- silently dropping `mmproj`
    there only is intentional, not an oversight.
    """
    eff = settings or Settings.load()

    # meta before caps: with two managed runtimes (mainline + the MoE-
    # prefill fork), which binary we prefer depends on whether THIS model
    # is MoE.
    meta = read_gguf_meta(model_path)

    # Phase 9.6 "Release N" (docs/specs/2026-07-21-native-inference-engine-
    # design.md): Pleiades' own from-scratch libllama-based engine is now the
    # DEFAULT whenever its binary resolves -- no env var required. This used
    # to be gated behind `PLEIADES_ENGINE=pleiades_native` ("feature-flagged,
    # off by default", Phase 5) while feature parity was still being closed;
    # 9.4 closed it (slot save/restore, vision, MoE offload, an external-
    # model escape hatch for whatever this engine still can't do). Two
    # explicit opt-OUTs: `PLEIADES_ENGINE=llama_server` skips straight to the
    # autodetected upstream llama-server branch below (the sanctioned escape
    # hatch for hardware this engine hasn't been verified on yet -- see the
    # design doc's Risk R1/R2, still genuinely open for non-CUDA MoE offload
    # and for MSVC/AppleClang builds); `llama_cpp` skips both native branches
    # for the pip-installed pure-python fallback. `PLEIADES_ENGINE=pleiades_native`
    # still works too, purely for back-compat with anything already setting it
    # explicitly -- it's now redundant with the default, not required by it.
    #
    # This engine is kept as its own branch rather than slotting into the
    # native-llama-server branch below: that branch builds flags this engine
    # still doesn't implement (--jinja, speculative decoding via -md/--spec-type,
    # --cache-reuse, --parallel), so pointing it at this binary would pass
    # flags parse_args() rejects. What this engine DOES support --
    # ngl/n-cpu-moe/ub/fa/ctk/ctv/mlock/ctx/threads/slot-save-path/mmproj --
    # is enough to run the same placement, which is what the branch below now
    # does. See find_native_cpp_engine()'s own docstring for the separation
    # rationale (checked against a second independent opinion, not just
    # assumed).
    engine_pref = os.environ.get("PLEIADES_ENGINE", "").strip().lower()
    # llama_server: explicit opt-out, wants the upstream native branch below.
    # llama_cpp: explicit opt-out, wants the pure-python branch at the
    # bottom -- forcing that while silently still running our own engine
    # (because its binary happened to be sitting there) would be exactly the
    # "explicit override silently ignored" footgun Phase 9.2's --caps work
    # was written to avoid elsewhere in this same function.
    if engine_pref not in ("llama_server", "llama_cpp"):
        native_cpp = runtime.find_native_cpp_engine()
        if native_cpp:
            # The from-scratch engine (engine/http_server.cpp) now speaks
            # llama-server's flag CLI and, unlike when this branch first landed
            # ("no autofit/MoE-split support yet"), DOES support GPU + MoE
            # expert offload: --ngl/--n-cpu-moe/--ub/--fa/--ctk/--ctv/--mlock
            # (see that file's parse_args()). So it now goes through the SAME
            # autofit placement as the native-llama-server branch below instead
            # of the old ngl=0 legacy positional mode.
            #
            # A dedicated RuntimeCaps is used rather than probing the (possibly
            # absent) llama-server binary via runtime.caps(): the engine's
            # capabilities are a known subset -- MoE expert offload yes, but not
            # the managed fork's prefill env-opts, speculative decoding, mmproj
            # or slot save/restore. place()'s VRAM cost model applies unchanged
            # because this engine links the very same libllama (identical
            # per-layer weight + KV footprint), and place() already reserves
            # autofit._OVERHEAD off gpu.vram_free and rejects placements that
            # don't fit -- so the OOM margin is inherited, not bypassed, and
            # n_cpu_moe is keyed off THIS model's real MoE structure via `meta`.
            #
            # moe_offload used to be hardcoded True here regardless of what was
            # actually built (Phase 9.2: this binary can now answer for itself
            # via `--caps`, one JSON line, no model touched -- see http_server.cpp's
            # print_caps()). A CPU-only build reports moe_offload=false (nothing
            # to offload between CPU and itself); place() then falls back to
            # dense placement instead of planning an expert split this build
            # can't actually perform -- exactly the degrade-not-OOM behavior
            # Phase 9's Risk R1 asks for. Any failure to query (older binary
            # predating --caps, timeout, bad JSON) keeps the prior
            # assume-offload-capable default rather than breaking a working
            # setup on an introspection hiccup.
            cpp_caps = _query_engine_caps(native_cpp)
            n_ctx_v, n_ctx_max, ctx_why = resolve_context(n_ctx, model_path, caps=cpp_caps)
            # Context is fixed at launch (like llama-server): the engine's
            # /resize exists, but we commit to the ceiling n_ctx_max here, same
            # reasoning as the native branch below.
            launch_ctx = n_ctx_max
            pl = place(meta, launch_ctx, caps=cpp_caps)
            forced = None
            if not (isinstance(n_gpu_layers, str) and n_gpu_layers.strip().lower() in ("", "auto")):
                try:
                    forced = int(n_gpu_layers)
                except (TypeError, ValueError):
                    forced = None
            # "All layers on GPU" is a NEGATIVE ngl for this engine
            # (ModelManager::load -> llama: n_gpu_layers < 0 => n_layer_all + 1),
            # NOT the 999 sentinel the llama-server branch relies on. place()
            # emits -1 for the full-GPU / MoE-split strategies, so it passes
            # straight through; a user override is honored verbatim.
            ngl = forced if forced is not None else pl.n_gpu_layers
            # Phase 9.2 (docs/specs/2026-07-21-native-inference-engine-design.md):
            # this branch used to build its command with none of the
            # thread/batch tuning the native-llama-server branch below computes
            # -- --threads was silently thrown away entirely (the engine then
            # launched pinned at ggml's hardcoded default of 4, see
            # detect_default_threads()'s own comment in http_server.cpp for why
            # that was "the highest-ROI bug an engine review found"). Same
            # halve-physical-cores formula as below, duplicated rather than
            # hoisted above this early-return branch to avoid computing it on
            # every call when this branch isn't taken.
            threads = max((os.cpu_count() or 8) // 2, 4)
            # Phase 9.4: warm-state slot save/restore, same directory/flag
            # shape as the native-llama-server branch below (models.py's
            # _slot_save/_slot_restore don't care which binary answers --
            # both now implement the identical POST /slots/0?action=... wire
            # contract, see http_server.cpp). Directory only, not a toggle --
            # costs nothing when unused, and a save from a previous run is
            # picked up even if this flag was added after that model was
            # first registered (same reasoning as the branch below).
            slot_dir = config.PLEIADES_HOME / "slots" / name
            slot_dir.mkdir(parents=True, exist_ok=True)
            cmd = [native_cpp, "--model", model_path, "--host", host, "--port", str(port),
                   "--ctx", str(launch_ctx), "--ngl", str(ngl), "--alias", name,
                   "--threads", str(threads),
                   "--slot-save-path", str(slot_dir) + os.sep]
            if forced is None and pl.n_cpu_moe:
                cmd += ["--n-cpu-moe", str(pl.n_cpu_moe)]
            ub = getattr(eff, "n_ubatch", 0) or pl.n_ubatch
            if not ub and pl.n_cpu_moe:
                # MoE-with-CPU-experts prefill is upload-bound; a bigger physical
                # batch amortizes the per-ubatch expert re-uploads (same measured
                # win the native branch documents). ub <= 2048 <= the engine's
                # default n_batch, so no --batch bump is needed.
                ub = 2048
            if ub:
                cmd += ["--ub", str(ub)]
            if getattr(eff, "n_batch", 0):
                cmd += ["--batch", str(eff.n_batch)]
            # Settings.cache_reuse does NOT apply here, deliberately: the
            # engine's prefix_cache.cpp does its own exact-longest-common-
            # prefix reuse unconditionally (Phase 6 above -- shipped as a
            # correctness fix, not an opt-in feature, and there is no
            # --cache-reuse flag on this binary to gate it with). That's a
            # different, simpler contract than native llama-server's own
            # chunked --cache-reuse N below; cache_reuse is a no-op knob for
            # this branch, not a bug to fix.
            fa = eff.flash_attn
            if eff.kv_cache_type and fa == "auto":
                fa = "on"  # a quantized V-cache requires flash attention
            if fa:
                cmd += ["--fa", fa]
            if eff.kv_cache_type:
                cmd += ["--ctk", eff.kv_cache_type, "--ctv", eff.kv_cache_type]
            if getattr(eff, "mlock", False):
                cmd += ["--mlock"]
            if mmproj and os.path.isfile(mmproj):
                # Phase 9.4.2: the engine now implements vision (libmtmd,
                # engine/CMakeLists.txt's LLAMA_BUILD_MTMD) -- see this
                # function's own docstring for why this branch was previously
                # scoped out of mmproj entirely and no longer is.
                cmd += ["--mmproj", mmproj]
            why = (f"runtime=pleiades-native-engine (experimental) strategy={pl.strategy} "
                   f"n_ctx={launch_ctx} ngl={ngl}"
                   + (f" n_cpu_moe={pl.n_cpu_moe}" if (forced is None and pl.n_cpu_moe) else "")
                   + (f" ub={ub}" if ub else "")
                   + (" +mmproj (vision)" if mmproj and os.path.isfile(mmproj) else "")
                   + f" est={pl.est_tps} tok/s -- {pl.reason}")
            if forced is not None:
                why = (f"runtime=pleiades-native-engine (experimental) "
                       f"n_ctx={launch_ctx} ngl={forced} (explicit override)")
            return LaunchPlan(cmd, why, ctx_why, launch_ctx, n_ctx_max)
        if engine_pref == "pleiades_native":
            print("[launch] PLEIADES_ENGINE=pleiades_native requested but "
                  "pleiades-engine-server not found (build via `cmake --build "
                  "engine/build`, or set PLEIADES_NATIVE_CPP_ENGINE_BIN) -- "
                  "falling back to the normal runtime selection.", flush=True)
        # Any other case here (unset, or a value we don't recognize) means the
        # engine binary just isn't built/installed yet -- the common case
        # during rollout, not worth a warning on every single launch.

    cps = runtime.caps(prefer_moe_opts=meta.is_moe)
    native = runtime.find_native(prefer_moe_opts=meta.is_moe) if cps.native else None
    if engine_pref == "llama_cpp":
        # Explicit opt-out: treat native llama-server as unavailable for
        # EVERY downstream decision (placement/ctx sizing included, not just
        # the branch check below) -- not doing this left a real bug where
        # `if native:` alone still matched and silently returned a
        # runtime=llama-server plan, ignoring this env var entirely
        # whenever a native binary was actually installed (adversarial
        # review, Phase 9.6 follow-up). `cps` is deliberately left as-is:
        # its moe_offload flag still feeds the "TIP: install the native
        # runtime" message below, which is still good advice here.
        native = None
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
        if engine_pref == "llama_server":
            print("[launch] PLEIADES_ENGINE=llama_server: using the autodetected "
                  "upstream llama-server binary instead of Pleiades' own engine. "
                  "For hardware the built-in engine doesn't support well yet, "
                  "`pleiades model add --external-url` is the longer-term escape "
                  "hatch (Phase 9.4).", flush=True)
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

    if engine_pref == "llama_server":
        print("[launch] PLEIADES_ENGINE=llama_server requested but no "
              "llama-server binary found (`pleiades runtime install`) -- "
              "falling back further.", flush=True)

    layers = forced if forced is not None else pl.n_gpu_layers
    if engine_pref == "llama_cpp":
        # escape hatch: bundled llama-cpp-python server (fixed window)
        print("[launch] PLEIADES_ENGINE=llama_cpp: using the pip-installed "
              "llama_cpp.server fallback instead of Pleiades' own engine.",
              flush=True)
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
