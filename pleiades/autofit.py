"""Autofit — the placement and quantization optimizer.

The whole problem reduces to one physical fact: generating a token reads every
*active* weight exactly once, so

    seconds/token ≈ Σ (bytes touched on device d) / (bandwidth of d)

Dense models touch every weight every token. MoE models touch only the routed
experts (e.g. 8 of 256), so the expert pool is *cold* per-token while
attention, shared experts, embeddings, and the KV cache are *hot*. The optimal
hybrid split therefore keeps the hot dense tensors in VRAM and parks cold
expert tensors in system RAM (llama.cpp `--n-cpu-moe` / `--override-tensor`),
paying CPU bandwidth only for the small active slice.

Autofit models that cost explicitly:

  1. decompose the GGUF into expert vs. non-expert bytes (real header fields:
     expert_count, expert_used_count, expert_feed_forward_length, ...),
  2. estimate device bandwidths (GPU by name/vendor, CPU by DDR heuristics),
  3. enumerate placement strategies the installed runtime supports,
  4. predict tokens/s for each, and
  5. choose quantization + placement under a speed/quality preference:
        speed    — fastest plan with quality ≥ Q4-class
        balanced — best quality that still predicts ≥ 10 tok/s
        quality  — best quality that still predicts ≥ 3 tok/s

Predictions are estimates (±30%), used only to *rank* options — and every
decision carries a human-readable reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .hardware import (GPU, GGUFMeta, Hardware, _gb, detect, group_split_ggufs,
                       quant_of)

# --------------------------------------------------------------------------- #
# Quality scores: relative answer quality per quant (1.0 = fp16), distilled
# from llama.cpp perplexity benchmarks across model families.
# --------------------------------------------------------------------------- #
QUALITY = {
    "F16": 1.0, "BF16": 1.0, "Q8_0": 0.999, "Q6_K": 0.997,
    "Q5_K_M": 0.993, "Q5_K_S": 0.991, "Q5_0": 0.989,
    "Q4_K_M": 0.985, "IQ4_XS": 0.982, "IQ4_NL": 0.982, "Q4_K_S": 0.980,
    "Q4_1": 0.976, "Q4_0": 0.973,
    "Q3_K_L": 0.962, "Q3_K_M": 0.952, "IQ3_S": 0.945, "Q3_K_S": 0.932,
    "Q2_K": 0.880, "IQ2_S": 0.845, "IQ2_XS": 0.820, "IQ2_XXS": 0.790,
}

# Floor used by the 'speed' preference: never trade below Q4-class quality.
_SPEED_QUALITY_FLOOR = 0.973
_MIN_TPS = {"speed": 0.0, "balanced": 10.0, "quality": 3.0}

# Reserved VRAM for compute graph/scratch (same spirit as hardware._OVERHEAD).
# 800 MiB measured too thin in practice (observed <100 MiB free after real loads,
# alongside swap thrash and OOM-adjacent hangs) — the compute graph, KV-cache
# growth, and CUDA/Vulkan allocator fragmentation all eat into this margin.
_OVERHEAD = 1536 * 1024 ** 2
_FIXED_S = 0.004        # per-token fixed cost: kernel launches, sampling, routing

# How much slower (relative) a candidate may be than the fastest one and still
# win on the strength of using less VRAM. Real OOM/swap-thrash protection is
# the hard _OVERHEAD reserve above plus per-candidate feasibility checks, not
# this number — this is only a tie-breaker for near-identical candidates.
# Was 0.15 (could forfeit ~25%+ tok/s on a "safer" choice that was never
# actually closer to the VRAM ceiling — observed picking 23.5 over an
# already-feasible 29.6 tok/s moe_partial candidate on an 11GB 2080 Ti).
# Owner wants max throughput the hardware can actually deliver, so this is
# now just enough to avoid flapping between two candidates of equal speed.
_CPU_BIAS_TOLERANCE = 0.03


# --------------------------------------------------------------------------- #
# Bandwidth estimation
# --------------------------------------------------------------------------- #
# Known GPU memory bandwidths (GB/s). Matched as substrings of the GPU name;
# unknown cards fall back to a conservative vendor default.
_GPU_BW = [
    ("5090", 1790), ("5080", 960), ("5070 ti", 896), ("5070", 672),
    ("4090", 1008), ("4080", 717), ("4070 ti", 672), ("4070", 504),
    ("4060 ti", 288), ("4060", 272),
    ("3090", 936), ("3080", 760), ("3070", 448), ("3060 ti", 448), ("3060", 360),
    ("2080 ti", 616), ("2080", 448), ("2070", 448), ("2060", 336),
    ("1080 ti", 484), ("1080", 320), ("1070", 256), ("1060", 192),
    ("7900 xtx", 960), ("7900 xt", 800), ("7800 xt", 624), ("7700 xt", 432),
    ("6900 xt", 512), ("6800 xt", 512), ("6700 xt", 384),
    ("rx 580", 256), ("vega", 484),
]
_VENDOR_BW = {"nvidia": 400, "amd": 380, "apple": 150}


def gpu_bandwidth(gpu: GPU) -> float:
    """Estimated memory bandwidth in bytes/s."""
    name = gpu.name.lower()
    for token, gbs in _GPU_BW:
        if token in name:
            return gbs * 1e9
    return _VENDOR_BW.get(gpu.vendor, 300) * 1e9


def cpu_bandwidth() -> float:
    """System RAM bandwidth in bytes/s (DDR heuristic, ~70% efficiency)."""
    mts = _ddr_mts()
    if mts:
        # MT/s * 8 bytes/channel * 2 channels * 0.7 efficiency
        return mts * 8 * 2 * 0.7 * 1e6
    return 30e9  # conservative dual-channel DDR4 default


def _ddr_mts() -> int:
    """Configured RAM speed in MT/s from dmidecode (root) — best effort."""
    from .hardware import _run
    for cmd in (["dmidecode", "-t", "memory"], ["sudo", "-n", "dmidecode", "-t", "memory"]):
        out = _run(cmd, timeout=4.0)
        if out:
            m = re.search(r"Configured Memory Speed:\s*(\d+)\s*MT/s", out)
            m = m or re.search(r"Speed:\s*(\d+)\s*MT/s", out)
            if m:
                return int(m.group(1))
    return 0


# --------------------------------------------------------------------------- #
# Weight decomposition
# --------------------------------------------------------------------------- #
_VOCAB_GUESS = 128_000  # only affects the small embeddings share of the split


@dataclass
class Decomposition:
    total: int            # bytes (the file)
    expert: int           # bytes in routed-expert FFN tensors
    rest: int             # everything always-hot: attn, shared, dense FFN, embeds
    active_expert: int    # expert bytes actually read per token
    active: int           # rest + active_expert


def decompose(meta: GGUFMeta) -> Decomposition:
    """Split the model file into hot (always-read) vs cold (expert) bytes.

    Parameter counts come from the architecture math; they are only used as
    *fractions* and then scaled to the real file size, so the quantization
    mix cancels out.
    """
    if not (meta.n_layers and meta.n_embd and meta.n_head):
        # Unknown structure: treat as dense.
        return Decomposition(meta.file_size, 0, meta.file_size, 0, meta.file_size)

    d = meta.n_embd
    head_dim = d // meta.n_head
    kv_dim = head_dim * (meta.n_head_kv or meta.n_head)
    attn = 2 * d * d + 2 * d * kv_dim                       # q,o + k,v projections
    shared = 3 * d * meta.shared_ffn_len if meta.shared_ffn_len else 0
    if meta.is_moe and meta.expert_ffn_len:
        expert_l = meta.expert_count * 3 * d * meta.expert_ffn_len
        dense_ffn = 0
    else:
        expert_l = 0
        dense_ffn = 3 * d * (meta.ffn_len or 4 * d)
    per_layer_rest = attn + shared + dense_ffn
    embeds = 2 * _VOCAB_GUESS * d

    expert_params = meta.n_layers * expert_l
    rest_params = meta.n_layers * per_layer_rest + embeds
    total_params = expert_params + rest_params or 1

    expert_b = int(meta.file_size * expert_params / total_params)
    rest_b = meta.file_size - expert_b
    used_frac = (meta.expert_used / meta.expert_count) if meta.is_moe else 0.0
    active_expert_b = int(expert_b * used_frac)
    return Decomposition(meta.file_size, expert_b, rest_b, active_expert_b,
                         rest_b + active_expert_b)


def kv_bytes(meta: GGUFMeta, n_ctx: int) -> int:
    from .hardware import _kv_bytes
    return _kv_bytes(meta, n_ctx)


# --------------------------------------------------------------------------- #
# Placement strategies
# --------------------------------------------------------------------------- #
@dataclass
class RuntimeCaps:
    """What the installed inference runtime can express."""
    moe_offload: bool = False      # --n-cpu-moe / --override-tensor available
    native: bool = False           # llama.cpp llama-server binary (vs python server)
    moe_prefill_opts: bool = False # env-gated MoE prefill opts (pinned host mem + expert prefetch)


@dataclass
class Placement:
    strategy: str                  # full_gpu | moe_cpu | moe_partial | layers | cpu
    n_gpu_layers: int = -1
    n_cpu_moe: int = 0             # layers whose experts live on CPU (native only)
    n_ubatch: int = 0              # physical batch (-ub); 0 = runtime default (512)
    est_tps: float = 0.0
    quality: float = 0.0
    vram_used: int = 0
    reason: str = ""
    feasible: bool = True


# -ub 2048 roughly doubles prompt-prefill throughput over llama.cpp's default
# 512 (measured: 206->391 tok/s) for ~1.3GiB extra VRAM, no decode-speed or
# quality cost — but that's ONE measurement on ONE 11GiB NVIDIA card. -ub 4096
# tested *worse* on the same box (less headroom left, slower prefill), so this
# never auto-picks past 2048. Only step up when a GPU candidate has real spare
# VRAM beyond what it already plans to use — never just to hit a target batch
# size on hardware this hasn't been measured on.
# ponytail: a one-box gear table, not a fitted curve. Widen it (more steps,
# AMD/Apple-specific constants) once it's been measured on more hardware —
# until then, "no auto step-up" is the safe default everywhere else.
_UBATCH_HEADROOM = int(1.5 * 1024 ** 3)


def _pick_ubatch(vram_budget: int, vram_used: int) -> int:
    """vram_budget is already net of `_OVERHEAD` — this checks for slack
    *beyond* that safety reserve, not just "fits at all"."""
    return 2048 if (vram_budget - vram_used) >= _UBATCH_HEADROOM else 0


def _tps(seconds_per_token: float) -> float:
    return round(1.0 / max(seconds_per_token, 1e-6), 1)


def place(meta: GGUFMeta, n_ctx: int, hw: Optional[Hardware] = None,
          caps: Optional[RuntimeCaps] = None) -> Placement:
    """Choose the fastest feasible placement for this model on this machine."""
    hw = hw or detect()
    caps = caps or RuntimeCaps()
    dec = decompose(meta)
    kv = kv_bytes(meta, n_ctx)
    gpu = hw.gpu
    cpu_bw = cpu_bandwidth()
    ram_budget = int(hw.ram_available * 0.85)
    n_layers = meta.n_layers or 32
    candidates: list[Placement] = []

    # Pure CPU is always feasible (if it fits RAM at all).
    cpu_t = dec.active / cpu_bw + _FIXED_S
    candidates.append(Placement(
        "cpu", n_gpu_layers=0, est_tps=_tps(cpu_t),
        reason=f"CPU only: {_gb(dec.active)} active/token at {cpu_bw / 1e9:.0f} GB/s",
        feasible=dec.total <= ram_budget))

    if gpu and not hw.unified_memory:
        gbw = gpu_bandwidth(gpu)
        vram_budget = gpu.vram_free - _OVERHEAD

        # 1. Everything in VRAM.
        if dec.total + kv <= vram_budget:
            t = dec.active / gbw + _FIXED_S
            vu = dec.total + kv
            candidates.append(Placement(
                "full_gpu", n_gpu_layers=-1, est_tps=_tps(t),
                vram_used=vu, n_ubatch=_pick_ubatch(vram_budget, vu),
                reason=f"whole model + KV fits {_gb(gpu.vram_free)} free VRAM"))

        # 2. MoE: hot path in VRAM, expert pool in RAM.
        if meta.is_moe and caps.moe_offload:
            hot = dec.rest + kv
            if hot <= vram_budget and dec.expert <= ram_budget:
                t = dec.rest / gbw + dec.active_expert / cpu_bw + _FIXED_S
                candidates.append(Placement(
                    "moe_cpu", n_gpu_layers=-1, n_cpu_moe=n_layers,
                    est_tps=_tps(t), vram_used=hot, n_ubatch=_pick_ubatch(vram_budget, hot),
                    reason=(f"MoE split: attention/shared ({_gb(dec.rest)}) on GPU, "
                            f"{meta.expert_count} experts ({_gb(dec.expert)}) in RAM — "
                            f"only {meta.expert_used} experts "
                            f"({_gb(dec.active_expert)}) read per token")))
                # 2b. Pull expert layers back into leftover VRAM where possible.
                # Only claim a fraction of "spare" — filling it to the last byte
                # leaves zero cushion beyond _OVERHEAD for KV growth/fragmentation
                # once real generation starts (this is what was paging into swap
                # and hanging chat requests near the VRAM ceiling).
                spare = (vram_budget - hot) * 0.6
                expert_per_layer = dec.expert / n_layers
                gpu_expert_layers = min(int(spare // max(expert_per_layer, 1)), n_layers)
                if gpu_expert_layers > 0:
                    n_cpu = n_layers - gpu_expert_layers
                    cpu_share = n_cpu / n_layers
                    t = (dec.rest + dec.active_expert * (1 - cpu_share)) / gbw \
                        + dec.active_expert * cpu_share / cpu_bw + _FIXED_S
                    vu = hot + int(gpu_expert_layers * expert_per_layer)
                    candidates.append(Placement(
                        "moe_partial", n_gpu_layers=-1, n_cpu_moe=n_cpu,
                        est_tps=_tps(t),
                        vram_used=vu, n_ubatch=_pick_ubatch(vram_budget, vu),
                        reason=(f"MoE split: experts of {n_cpu}/{n_layers} layers in "
                                f"RAM, the rest fully in VRAM")))

        # 3. Dense (or runtime without MoE offload): whole layers split.
        per_layer = dec.total / (n_layers + 1)
        kv_per_layer = kv / n_layers
        fit = int(vram_budget // max(per_layer + kv_per_layer, 1))
        fit = max(min(fit, n_layers), 0)
        if 0 < fit < n_layers:
            f = fit / n_layers
            t = dec.active * f / gbw + dec.active * (1 - f) / cpu_bw + _FIXED_S
            vu = int(fit * (per_layer + kv_per_layer))
            candidates.append(Placement(
                "layers", n_gpu_layers=fit, est_tps=_tps(t),
                vram_used=vu, n_ubatch=_pick_ubatch(vram_budget, vu),
                reason=f"dense split: {fit}/{n_layers} layers in VRAM",
                feasible=dec.total * (1 - f) <= ram_budget))

    if gpu and hw.unified_memory:
        # Apple Silicon: one memory pool — run fully on the GPU if it fits.
        gbw = gpu_bandwidth(gpu)
        if dec.total + kv <= gpu.vram_free:
            t = dec.active / gbw + _FIXED_S
            candidates.append(Placement(
                "full_gpu", n_gpu_layers=-1, est_tps=_tps(t),
                vram_used=dec.total + kv,
                reason="unified memory: whole model on the Apple GPU"))

    viable = [c for c in candidates if c.feasible]
    if not viable:
        worst = candidates[0]
        worst.reason += " — WARNING: model may not fit this machine at all"
        return worst
    return _pick_cpu_biased(viable)


def _pick_cpu_biased(viable: list[Placement]) -> Placement:
    """Among candidates within _CPU_BIAS_TOLERANCE of the fastest, take the one
    that uses the least VRAM — i.e. push more onto the CPU/RAM whenever the
    speed loss for doing so is small. Maxing out VRAM for a marginal tok/s gain
    just trades a few % speed for OOM/swap risk; that's a bad trade."""
    best_tps = max(c.est_tps for c in viable)
    floor = best_tps * (1 - _CPU_BIAS_TOLERANCE)
    pool = [c for c in viable if c.est_tps >= floor]
    chosen = min(pool, key=lambda c: c.vram_used)
    if chosen.vram_used < max(pool, key=lambda c: c.vram_used).vram_used:
        chosen.reason += (f" — picked over a faster option to leave more VRAM "
                          f"headroom (within {_CPU_BIAS_TOLERANCE:.0%} of best "
                          f"est. {best_tps} tok/s)")
    return chosen


# --------------------------------------------------------------------------- #
# Quant + placement chooser (downloads)
# --------------------------------------------------------------------------- #
@dataclass
class QuantChoice:
    entry: dict                    # the (grouped) HF file entry
    quant: str
    placement: Placement
    quality: float
    why: str = ""


def _scaled_meta(meta_like: Optional[GGUFMeta], size: int) -> GGUFMeta:
    """A candidate's meta: real structure (if known) with this file's size."""
    if meta_like and meta_like.n_layers:
        m = GGUFMeta(**{**meta_like.__dict__})
        m.file_size = size
        return m
    return GGUFMeta(path="", file_size=size)


def choose_quant(files: list[dict], hw: Optional[Hardware] = None,
                 n_ctx: int = 8192, preference: str = "balanced",
                 caps: Optional[RuntimeCaps] = None,
                 meta_hint: Optional[GGUFMeta] = None) -> Optional[QuantChoice]:
    """Pick the best quant for this machine under a speed/quality preference.

    files: [{"name", "size"}] from the HF repo. meta_hint (optional) carries
    the model's structure (layers, MoE shape) so placement math is exact;
    without it, candidates are treated as dense.
    """
    hw = hw or detect()
    preference = preference if preference in _MIN_TPS else "balanced"
    grouped = group_split_ggufs(files)
    by_quant: dict[str, dict] = {}
    for g in grouped:
        q = quant_of(g["name"])
        if q and q in QUALITY and (q not in by_quant or g["size"] < by_quant[q]["size"]):
            by_quant[q] = g
    if not by_quant:
        return None

    scored: list[QuantChoice] = []
    for q, entry in by_quant.items():
        meta = _scaled_meta(meta_hint, entry["size"])
        pl = place(meta, n_ctx, hw, caps)
        scored.append(QuantChoice(entry=entry, quant=q, placement=pl,
                                  quality=QUALITY[q]))

    feasible = [c for c in scored if c.placement.feasible] or scored
    min_tps = _MIN_TPS[preference]

    if preference == "speed":
        pool = [c for c in feasible if c.quality >= _SPEED_QUALITY_FLOOR] or feasible
        best = max(pool, key=lambda c: (c.placement.est_tps, c.quality))
    else:
        pool = [c for c in feasible if c.placement.est_tps >= min_tps]
        if pool:
            best = max(pool, key=lambda c: (c.quality, c.placement.est_tps))
        else:  # nothing meets the target: take the fastest decent option
            best = max(feasible, key=lambda c: (c.placement.est_tps, c.quality))

    best.entry = dict(best.entry)
    best.why = (f"{best.quant} ({preference}): est. {best.placement.est_tps} tok/s "
                f"via {best.placement.strategy.replace('_', ' ')} — "
                f"{best.placement.reason}")
    best.entry["why"] = best.why
    return best
