"""GGUFMeta.memory_arch / recurrent_state_bytes / kv_bytes_per_token — the
2026-07-21 context-free-model-architecture plan's Phase 0.A truth layer.

See docs/specs/2026-07-21-context-free-model-architecture-design.md.
"""

import struct

from pleiades import hardware
from pleiades.hardware import GGUFMeta, GPU, Hardware, kv_bytes_per_token, plan_context

GiB = 1024 ** 3


def _kv_str(key: str, val: str) -> bytes:
    k, v = key.encode(), val.encode()
    return struct.pack("<Q", len(k)) + k + struct.pack("<I", 8) + \
        struct.pack("<Q", len(v)) + v


def _kv_u32(key: str, val: int) -> bytes:
    k = key.encode()
    return struct.pack("<Q", len(k)) + k + struct.pack("<I", 4) + struct.pack("<I", val)


def make_gguf(path, arch="llama", n_layers=32, n_embd=4096, n_head=32,
              n_head_kv=8, ssm_d_conv=0, ssm_d_inner=0, ssm_d_state=0,
              ssm_n_group=0, pad_to=1024) -> str:
    kvs = _kv_str("general.architecture", arch) \
        + _kv_u32(f"{arch}.block_count", n_layers) \
        + _kv_u32(f"{arch}.embedding_length", n_embd) \
        + _kv_u32(f"{arch}.attention.head_count", n_head) \
        + _kv_u32(f"{arch}.attention.head_count_kv", n_head_kv) \
        + _kv_u32("general.file_type", 15)
    n = 6
    if ssm_d_conv:
        kvs += _kv_u32(f"{arch}.ssm.conv_kernel", ssm_d_conv); n += 1
    if ssm_d_inner:
        kvs += _kv_u32(f"{arch}.ssm.inner_size", ssm_d_inner); n += 1
    if ssm_d_state:
        kvs += _kv_u32(f"{arch}.ssm.state_size", ssm_d_state); n += 1
    if ssm_n_group:
        kvs += _kv_u32(f"{arch}.ssm.group_count", ssm_n_group); n += 1
    blob = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + \
        struct.pack("<Q", n) + kvs
    blob += b"\0" * max(0, pad_to - len(blob))
    path.write_bytes(blob)
    return str(path)


def _hw(vram_gb=11, ram_gb=64):
    gpus = [GPU("nvidia", "Fake RTX 2080 Ti", int(vram_gb * GiB), int(vram_gb * GiB))]
    return Hardware(gpus=gpus, ram_total=int(ram_gb * GiB),
                    ram_available=int(ram_gb * GiB), cpu_threads=16)


def test_ordinary_dense_model_is_unaffected(tmp_path):
    p = make_gguf(tmp_path / "dense.gguf", arch="llama")
    meta = hardware.read_gguf_meta(p)
    assert meta.memory_arch == "dense"
    assert meta.recurrent_state_bytes == 0
    assert kv_bytes_per_token(meta) > 0  # unchanged dense KV formula


def test_pure_recurrent_arch_has_zero_kv_growth(tmp_path):
    p = make_gguf(tmp_path / "mamba2.gguf", arch="mamba2", n_layers=48,
                  ssm_d_conv=4, ssm_d_inner=8192, ssm_d_state=128, ssm_n_group=1)
    meta = hardware.read_gguf_meta(p)
    assert meta.memory_arch == "recurrent"
    assert kv_bytes_per_token(meta) == 0
    assert meta.recurrent_state_bytes > 0  # fixed, sequence-length-independent


def test_hybrid_arch_tagged_but_conservative(tmp_path):
    p = make_gguf(tmp_path / "jamba.gguf", arch="jamba")
    meta = hardware.read_gguf_meta(p)
    assert meta.memory_arch == "hybrid"
    # Conservative on purpose (see design doc): a hybrid model still has
    # unbounded-by-default attention layers we don't yet account for
    # separately, so it keeps the dense KV formula rather than claiming a
    # win we can't verify from the GGUF header alone.
    assert kv_bytes_per_token(meta) > 0


def test_recurrent_model_ceiling_reaches_full_trained_context(tmp_path):
    """The Phase 0.A payoff: a big trained context on modest hardware should
    reach its FULL ceiling for a recurrent model, unlike a same-sized dense
    model which would be capped by KV growth long before then."""
    p = make_gguf(tmp_path / "mamba2-big-ctx.gguf", arch="mamba2", n_layers=48,
                  ssm_d_conv=4, ssm_d_inner=8192, ssm_d_state=128, ssm_n_group=1)
    meta = hardware.read_gguf_meta(p)
    meta.file_size = int(4 * GiB)
    meta.n_ctx_train = 131072
    plan = plan_context(meta, hw=_hw(vram_gb=11))
    assert plan.n_ctx_max == 131072  # nothing constrains it below the trained ceiling
