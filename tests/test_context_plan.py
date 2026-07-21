"""pleiades.hardware context planning (virtual context / elastic engine)."""

import struct

from pleiades import hardware
from pleiades.hardware import (GPU, GGUFMeta, Hardware, kv_bytes_per_token,
                               plan_context, resolve_context)

GiB = 1024 ** 3


def _kv_str(key: str, val: str) -> bytes:
    k, v = key.encode(), val.encode()
    return struct.pack("<Q", len(k)) + k + struct.pack("<I", 8) + \
        struct.pack("<Q", len(v)) + v


def _kv_u32(key: str, val: int) -> bytes:
    k = key.encode()
    return struct.pack("<Q", len(k)) + k + struct.pack("<I", 4) + struct.pack("<I", val)


def make_gguf(path, n_layers=32, n_embd=4096, n_head=32, n_head_kv=8,
              context_length=32768, pad_to=4 * 1024 ** 2) -> str:
    kvs = (_kv_str("general.architecture", "llama")
           + _kv_u32("llama.block_count", n_layers)
           + _kv_u32("llama.embedding_length", n_embd)
           + _kv_u32("llama.attention.head_count", n_head)
           + _kv_u32("llama.attention.head_count_kv", n_head_kv)
           + _kv_u32("llama.context_length", context_length)
           + _kv_u32("general.file_type", 15))
    blob = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + \
        struct.pack("<Q", 7) + kvs
    blob += b"\0" * max(0, pad_to - len(blob))
    path.write_bytes(blob)
    return str(path)


def _hw(vram_gb=0, ram_gb=32):
    gpus = [GPU("nvidia", "Fake RTX", int(vram_gb * GiB), int(vram_gb * GiB))] \
        if vram_gb else []
    return Hardware(gpus=gpus, ram_total=int(ram_gb * GiB),
                    ram_available=int(ram_gb * GiB), cpu_threads=16)


def _meta(size_gb=4, n_layers=32, n_ctx_train=32768):
    return GGUFMeta(path="x", file_size=int(size_gb * GiB), n_layers=n_layers,
                    n_embd=4096, n_head=32, n_head_kv=8, n_ctx_train=n_ctx_train)


# --------------------------------------------------------------------------- #
# KV math + parsing
# --------------------------------------------------------------------------- #
def test_kv_bytes_per_token_exact():
    # 32 layers × 8 kv heads × 128 head_dim × 2 (K+V) × 2 bytes = 128 KiB
    assert kv_bytes_per_token(_meta()) == 32 * 8 * 128 * 2 * 2 == 131072


def test_kv_bytes_per_token_fallback_when_unparsed():
    meta = GGUFMeta(path="x", file_size=GiB)  # no header fields
    assert kv_bytes_per_token(meta) == hardware._DEFAULT_LAYERS * 1024


def test_gguf_parser_reads_context_length(tmp_path):
    p = make_gguf(tmp_path / "m.gguf", context_length=131072)
    assert hardware.read_gguf_meta(p).n_ctx_train == 131072


# --------------------------------------------------------------------------- #
# plan_context
# --------------------------------------------------------------------------- #
def test_big_gpu_launches_modest_with_high_ceiling():
    cp = plan_context(_meta(size_gb=4, n_ctx_train=131072), hw=_hw(vram_gb=24))
    assert cp.n_ctx == 16384            # modest launch gear
    assert cp.n_ctx_max >= 32768        # elastic headroom
    assert cp.n_ctx_max <= 131072
    assert "elastic" in cp.reason


def test_trained_context_caps_the_ceiling():
    cp = plan_context(_meta(size_gb=4, n_ctx_train=8192), hw=_hw(vram_gb=24))
    assert cp.n_ctx_max <= 8192
    assert cp.n_ctx <= 8192


def test_tight_vram_caps_the_ceiling_low():
    # 4 GiB model + 0.8 GiB overhead in 6 GiB free: KV is 128 KiB/token, so
    # 8192 ctx (1 GiB) still fully offloads but 16384 (2 GiB) does not --
    # that governs the LAUNCH gear (protect full-offload speed at start).
    cp = plan_context(_meta(size_gb=4), hw=_hw(vram_gb=6))
    assert cp.n_ctx == 8192
    # The ELASTIC CEILING is a separate, more generous question (2026-07-21
    # policy change): with 32 GiB of system RAM available (the _hw() default),
    # the model's full trained context (32768) is reachable via partial CPU
    # offload even though it doesn't fully fit on this tight 6 GiB GPU. The
    # ceiling should reflect that -- it's only ever PAID FOR under real
    # pressure (resize()/self-heal), never at launch.
    assert cp.n_ctx_max == 32768


def test_very_tight_vram_floors_at_smallest_gear():
    # Nothing fully offloads even at the smallest gear on this GPU -- the
    # LAUNCH gear floors out (protect speed/predictability at start).
    cp = plan_context(_meta(size_gb=5), hw=_hw(vram_gb=6))
    assert cp.n_ctx == 4096
    # But the ELASTIC CEILING still reaches the model's trained context via
    # partial CPU offload, since 32 GiB of system RAM easily absorbs the
    # remainder -- a tight GPU no longer means a permanently tiny ceiling.
    assert cp.n_ctx_max == 32768
    assert "elastic" in cp.reason


def test_ceiling_stays_honest_when_ram_also_cant_cover_the_remainder():
    # Tiny GPU AND tiny system RAM: even partial-CPU-offload can't reach the
    # model's full trained context, so the ceiling must NOT overclaim it.
    cp = plan_context(_meta(size_gb=5, n_ctx_train=131072), hw=_hw(vram_gb=1, ram_gb=2))
    assert cp.n_ctx_max < 131072
    assert cp.n_ctx <= cp.n_ctx_max


def test_cpu_plans_from_ram():
    cp = plan_context(_meta(size_gb=4, n_ctx_train=131072), hw=_hw(vram_gb=0, ram_gb=32))
    # 25% of 32 GiB = 8 GiB; 65536 ctx × 128 KiB = 8 GiB fits, 131072 doesn't.
    assert cp.n_ctx_max == 65536
    assert cp.n_ctx == 16384
    assert "CPU" in cp.reason


# --------------------------------------------------------------------------- #
# resolve_context
# --------------------------------------------------------------------------- #
def test_resolve_context_pins_explicit_values(tmp_path):
    assert resolve_context(7000, "x")[:2] == (7000, 7000)
    assert resolve_context("4096", "x")[:2] == (4096, 4096)
    assert resolve_context(7000, "x")[2] == "explicit setting"


def test_resolve_context_auto_plans_from_gguf(tmp_path):
    p = make_gguf(tmp_path / "m.gguf", context_length=32768)
    n_ctx, n_ctx_max, why = resolve_context("auto", p, hw=_hw(vram_gb=24))
    assert n_ctx == 16384
    assert n_ctx_max == 32768
    assert "elastic" in why


# --------------------------------------------------------------------------- #
# elastic server state (no llama import needed)
# --------------------------------------------------------------------------- #
def test_engine_state_gears_and_props(tmp_path):
    from pleiades.inference.server import EngineState
    p = make_gguf(tmp_path / "m.gguf", context_length=32768)
    st = EngineState(p, n_ctx=8192, n_ctx_max=32768, n_gpu_layers=0,
                     chat_format="", alias="t")
    assert st.next_gear() == 16384
    assert st.clamp_gear(999999) == 32768   # ceiling
    assert st.clamp_gear(1) == 1024         # floor
    props = st.props()
    assert props["resizable"] is True
    assert props["n_ctx"] == 8192
    assert props["n_ctx_train"] == 32768
    assert props["default_generation_settings"]["n_ctx"] == 8192
    assert props["kv_bytes_per_token"] == 131072
    st.n_ctx = 32768
    assert st.next_gear() is None           # at ceiling
