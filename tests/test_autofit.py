"""Autofit: MoE decomposition, placement strategies, quant choice."""

from pleiades.autofit import (QUALITY, RuntimeCaps, choose_quant, decompose,
                              place)
from pleiades.hardware import GPU, GGUFMeta, Hardware
from pleiades.runtime import pick_asset

GiB = 1024 ** 3


def _hw(vram_gb=11, ram_gb=48, name="RTX 2080 Ti"):
    gpus = [GPU("nvidia", name, int(vram_gb * GiB), int(vram_gb * GiB))] if vram_gb else []
    return Hardware(gpus=gpus, ram_total=int(ram_gb * GiB),
                    ram_available=int(ram_gb * GiB), cpu_threads=16)


def _moe_meta(size_gb=20):
    """Qwen3-35B-A3B-like: 256 experts, 8 used, small expert FFN."""
    return GGUFMeta(path="x", file_size=int(size_gb * GiB), n_layers=40,
                    n_embd=2048, n_head=16, n_head_kv=2,
                    expert_count=256, expert_used=8, expert_ffn_len=512,
                    shared_ffn_len=512)


def _dense_meta(size_gb=8):
    return GGUFMeta(path="x", file_size=int(size_gb * GiB), n_layers=32,
                    n_embd=4096, n_head=32, n_head_kv=8, ffn_len=14336)


# ── decomposition ─────────────────────────────────────────────────────────── #
def test_moe_decomposition_experts_dominate():
    dec = decompose(_moe_meta())
    assert dec.expert > dec.rest                 # expert pool is most of the file
    assert dec.active < dec.total * 0.35         # tiny active slice per token
    assert dec.active_expert == int(dec.expert * 8 / 256)


def test_dense_decomposition_everything_active():
    dec = decompose(_dense_meta())
    assert dec.expert == 0
    assert dec.active == dec.total


# ── placement ─────────────────────────────────────────────────────────────── #
def test_moe_split_chosen_when_runtime_supports_it():
    pl = place(_moe_meta(20), 8192, _hw(vram_gb=11),
               RuntimeCaps(moe_offload=True, native=True))
    assert pl.strategy in ("moe_cpu", "moe_partial")
    assert pl.n_cpu_moe > 0
    assert pl.est_tps > 10            # the whole point: usable speed


def test_moe_without_native_runtime_degrades_to_layers_or_cpu():
    pl = place(_moe_meta(20), 8192, _hw(vram_gb=11), RuntimeCaps())
    assert pl.strategy in ("layers", "cpu")


def test_moe_split_is_faster_than_layer_split():
    hw = _hw(vram_gb=11)
    smart = place(_moe_meta(20), 8192, hw, RuntimeCaps(moe_offload=True, native=True))
    dumb = place(_moe_meta(20), 8192, hw, RuntimeCaps())
    assert smart.est_tps > dumb.est_tps * 1.3


def test_dense_full_gpu_when_it_fits():
    pl = place(_dense_meta(6), 4096, _hw(vram_gb=11))
    assert pl.strategy == "full_gpu" and pl.n_gpu_layers == -1


def test_dense_layer_split_when_too_big():
    pl = place(_dense_meta(20), 4096, _hw(vram_gb=11))
    assert pl.strategy == "layers"
    assert 0 < pl.n_gpu_layers < 32


def test_cpu_when_no_gpu():
    pl = place(_dense_meta(6), 4096, _hw(vram_gb=0))
    assert pl.strategy == "cpu"


def test_ubatch_auto_picked_when_vram_has_headroom():
    """Generous VRAM beyond the chosen placement: autofit steps -ub up."""
    pl = place(_dense_meta(6), 4096, _hw(vram_gb=24))
    assert pl.strategy == "full_gpu"
    assert pl.n_ubatch == 2048


def test_ubatch_default_when_vram_is_tight():
    """Just barely fits: no spare VRAM to risk a bigger batch buffer."""
    pl = place(_dense_meta(10), 4096, _hw(vram_gb=11))
    assert pl.n_ubatch == 0


def test_ubatch_default_on_cpu_only():
    pl = place(_dense_meta(6), 4096, _hw(vram_gb=0))
    assert pl.strategy == "cpu"
    assert pl.n_ubatch == 0


def test_known_gpu_bandwidth_used():
    from pleiades.autofit import gpu_bandwidth
    assert gpu_bandwidth(GPU("nvidia", "NVIDIA GeForce RTX 2080 Ti", 0, 0)) == 616e9
    assert gpu_bandwidth(GPU("nvidia", "Mystery Card 9999", 0, 0)) == 400e9


# ── quant choice ──────────────────────────────────────────────────────────── #
def _files(**sizes_gb):
    return [{"name": f"m-{q}.gguf", "size": int(s * GiB)}
            for q, s in sizes_gb.items()]


def test_balanced_takes_best_quality_meeting_target():
    files = _files(Q8_0=8, Q6_K=6, Q4_K_M=4, Q2_K=2)
    c = choose_quant(files, _hw(vram_gb=11), preference="balanced")
    assert c.quant == "Q8_0"          # everything fits VRAM; max quality wins
    assert "tok/s" in c.why


def test_quality_pref_accepts_slow_moe_split():
    """35B MoE on 11 GiB: quality pref picks a big quant via the MoE split."""
    files = _files(Q8_0=37, Q6_K=29, Q4_K_M=21, Q3_K_M=16)
    c = choose_quant(files, _hw(vram_gb=11, ram_gb=64), preference="quality",
                     caps=RuntimeCaps(moe_offload=True, native=True),
                     meta_hint=_moe_meta())
    assert c.placement.strategy.startswith("moe")
    assert QUALITY[c.quant] >= QUALITY["Q4_K_M"]


def test_speed_pref_never_goes_below_q4_class():
    files = _files(Q8_0=8, Q4_K_M=4, Q2_K=2)
    c = choose_quant(files, _hw(vram_gb=11), preference="speed")
    assert QUALITY[c.quant] >= 0.973


def test_dense_too_big_for_vram_prefers_fitting_quant():
    """Balanced should drop quant size to keep speed usable on dense models."""
    files = _files(Q8_0=30, Q6_K=24, Q4_K_M=18, Q3_K_M=14, Q2_K=10)
    c = choose_quant(files, _hw(vram_gb=11, ram_gb=64), preference="balanced",
                     meta_hint=_dense_meta())
    # nothing fits 11 GiB fully; balanced picks the best quality that still
    # predicts >= 10 tok/s, which forces a smaller quant than Q8
    assert QUALITY[c.quant] < QUALITY["Q8_0"]


# ── runtime asset picker ─────────────────────────────────────────────────── #
def test_pick_asset_prefers_cuda_for_nvidia(monkeypatch):
    import pleiades.runtime as rt
    monkeypatch.setattr(rt, "_platform_tokens", lambda: (["ubuntu", "linux"], "x64"))
    monkeypatch.setattr(rt, "_backend_priority", lambda: ["cuda", "vulkan", ""])
    assets = [{"name": "llama-b1-bin-ubuntu-x64.zip"},
              {"name": "llama-b1-bin-ubuntu-vulkan-x64.zip"},
              {"name": "llama-b1-bin-ubuntu-cuda-x64.zip"},
              {"name": "llama-b1-bin-win-cuda-x64.zip"},
              {"name": "llama-b1-bin-macos-arm64.zip"}]
    assert pick_asset(assets)["name"] == "llama-b1-bin-ubuntu-cuda-x64.zip"


def test_pick_asset_falls_back_to_plain_build(monkeypatch):
    import pleiades.runtime as rt
    monkeypatch.setattr(rt, "_platform_tokens", lambda: (["ubuntu", "linux"], "x64"))
    monkeypatch.setattr(rt, "_backend_priority", lambda: ["cuda", "vulkan", ""])
    assets = [{"name": "llama-b1-bin-ubuntu-x64.zip"},
              {"name": "llama-b1-bin-win-cuda-x64.zip"}]
    assert pick_asset(assets)["name"] == "llama-b1-bin-ubuntu-x64.zip"


def test_pick_asset_cpu_fallback_skips_openvino_and_opencl(monkeypatch):
    # 2026-07-24 fix: the "no GPU -> plain CPU build" fallback excluded
    # cuda/vulkan/hip/rocm/sycl/cann tokens but not openvino/opencl, so a
    # CPU-only box could get handed an OpenVINO (or OpenCL) archive instead
    # of the actual plain build, depending on asset list order.
    import pleiades.runtime as rt
    monkeypatch.setattr(rt, "_platform_tokens", lambda: (["ubuntu", "linux"], "x64"))
    monkeypatch.setattr(rt, "_backend_priority", lambda: ["", "vulkan"])
    assets = [{"name": "llama-b1-bin-ubuntu-openvino-x64.tar.gz"},
              {"name": "llama-b1-bin-ubuntu-x64.zip"},
              {"name": "llama-b1-bin-ubuntu-vulkan-x64.zip"}]
    assert pick_asset(assets)["name"] == "llama-b1-bin-ubuntu-x64.zip"


def test_backend_priority_prefers_vulkan_then_sycl_for_intel(monkeypatch):
    import pleiades.runtime as rt
    hw = Hardware(gpus=[GPU("intel", "Intel Arc A770", 16 * GiB, 16 * GiB)])
    monkeypatch.setattr("pleiades.hardware.detect", lambda: hw)
    assert rt._backend_priority() == ["vulkan", "sycl", ""]


def test_pick_asset_prefers_vulkan_for_intel_gpu(monkeypatch):
    import pleiades.runtime as rt
    hw = Hardware(gpus=[GPU("intel", "Intel Arc A770", 16 * GiB, 16 * GiB)])
    monkeypatch.setattr("pleiades.hardware.detect", lambda: hw)
    monkeypatch.setattr(rt, "_platform_tokens", lambda: (["ubuntu", "linux"], "x64"))
    assets = [{"name": "llama-b1-bin-ubuntu-x64.zip"},
              {"name": "llama-b1-bin-ubuntu-vulkan-x64.zip"},
              {"name": "llama-b1-bin-ubuntu-sycl-x64.zip"},
              {"name": "llama-b1-bin-ubuntu-cuda-x64.zip"}]
    assert pick_asset(assets)["name"] == "llama-b1-bin-ubuntu-vulkan-x64.zip"


def test_pick_asset_cpu_fallback_skips_foreign_architectures(monkeypatch):
    # Regression test for a real bug found 2026-07-24 by dry-running
    # pick_asset against the ACTUAL live ggml-org/llama.cpp b10107 release:
    # "llama-b10107-bin-ubuntu-s390x.tar.gz" has neither an "x64" nor
    # "arm64" token, so the old (arch in n or ("x64" not in n and "arm64"
    # not in n)) filter let it through as if it were a generic x64 build --
    # once the openvino/opencl exclusion fix (above) stopped a CPU-only
    # search from grabbing the openvino asset first, s390x was the very
    # next wrong match in list order. This is the real asset list (trimmed
    # to the relevant subset, names unchanged) that exposed it.
    import pleiades.runtime as rt
    monkeypatch.setattr(rt, "_platform_tokens", lambda: (["ubuntu", "linux"], "x64"))
    monkeypatch.setattr(rt, "_backend_priority", lambda: ["", "vulkan"])
    assets = [{"name": "llama-b10107-bin-ubuntu-arm64.tar.gz"},
              {"name": "llama-b10107-bin-ubuntu-openvino-2026.2.1-x64.tar.gz"},
              {"name": "llama-b10107-bin-ubuntu-rocm-7.2-x64.tar.gz"},
              {"name": "llama-b10107-bin-ubuntu-s390x.tar.gz"},
              {"name": "llama-b10107-bin-ubuntu-sycl-fp16-x64.tar.gz"},
              {"name": "llama-b10107-bin-ubuntu-vulkan-x64.tar.gz"},
              {"name": "llama-b10107-bin-ubuntu-x64.tar.gz"}]
    assert pick_asset(assets)["name"] == "llama-b10107-bin-ubuntu-x64.tar.gz"
