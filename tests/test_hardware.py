"""pleiades.hardware: GGUF parsing, placement planning, quant selection."""

import struct

from pleiades import hardware
from pleiades.hardware import (GPU, GGUFMeta, Hardware, group_split_ggufs,
                               pick_quant, plan, quant_of, resolve_layers)

GiB = 1024 ** 3


# --------------------------------------------------------------------------- #
# synthetic GGUF
# --------------------------------------------------------------------------- #
def _kv_str(key: str, val: str) -> bytes:
    k, v = key.encode(), val.encode()
    return struct.pack("<Q", len(k)) + k + struct.pack("<I", 8) + \
        struct.pack("<Q", len(v)) + v


def _kv_u32(key: str, val: int) -> bytes:
    k = key.encode()
    return struct.pack("<Q", len(k)) + k + struct.pack("<I", 4) + struct.pack("<I", val)


def make_gguf(path, n_layers=32, n_embd=4096, n_head=32, n_head_kv=8,
              pad_to=1024) -> str:
    kvs = (_kv_str("general.architecture", "llama")
           + _kv_u32("llama.block_count", n_layers)
           + _kv_u32("llama.embedding_length", n_embd)
           + _kv_u32("llama.attention.head_count", n_head)
           + _kv_u32("llama.attention.head_count_kv", n_head_kv)
           + _kv_u32("general.file_type", 15))
    blob = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + \
        struct.pack("<Q", 6) + kvs
    blob += b"\0" * max(0, pad_to - len(blob))
    path.write_bytes(blob)
    return str(path)


def test_gguf_parse(tmp_path):
    p = make_gguf(tmp_path / "m.gguf", n_layers=48, n_embd=8192, n_head=64,
                  n_head_kv=8)
    meta = hardware.read_gguf_meta(p)
    assert meta.n_layers == 48
    assert meta.n_embd == 8192
    assert meta.n_head_kv == 8
    assert meta.quant == "Q4_K_M"
    assert not meta.estimated


def test_gguf_parse_garbage_is_safe(tmp_path):
    p = tmp_path / "junk.gguf"
    p.write_bytes(b"not a gguf at all")
    meta = hardware.read_gguf_meta(p)
    assert meta.estimated and meta.file_size > 0
    meta2 = hardware.read_gguf_meta(tmp_path / "missing.gguf")
    assert meta2.file_size == 0


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #
def _hw(vram_gb=0, ram_gb=32):
    gpus = [GPU("nvidia", "Fake RTX", int(vram_gb * GiB), int(vram_gb * GiB))] \
        if vram_gb else []
    return Hardware(gpus=gpus, ram_total=int(ram_gb * GiB),
                    ram_available=int(ram_gb * GiB), cpu_threads=16)


def _meta(size_gb, n_layers=32):
    return GGUFMeta(path="x", file_size=int(size_gb * GiB), n_layers=n_layers,
                    n_embd=4096, n_head=32, n_head_kv=8)


def test_plan_full_offload_when_it_fits():
    p = plan(_meta(4), n_ctx=8192, hw=_hw(vram_gb=24))
    assert p.n_gpu_layers == -1 and p.fits_fully


def test_plan_partial_offload_when_tight():
    p = plan(_meta(20, n_layers=40), n_ctx=8192, hw=_hw(vram_gb=8))
    assert 0 < p.n_gpu_layers < 40
    assert "partial" in p.reason


def test_plan_cpu_when_no_gpu():
    p = plan(_meta(4), n_ctx=8192, hw=_hw(vram_gb=0))
    assert p.n_gpu_layers == 0 and "no GPU" in p.reason


def test_plan_cpu_when_vram_tiny():
    p = plan(_meta(30, n_layers=40), n_ctx=8192, hw=_hw(vram_gb=1))
    assert p.n_gpu_layers == 0


def test_resolve_layers_passthrough_and_auto(tmp_path):
    assert resolve_layers(-1, "x", 8192)[0] == -1
    assert resolve_layers("7", "x", 8192)[0] == 7
    p = make_gguf(tmp_path / "m.gguf")
    layers, why = resolve_layers("auto", p, 8192, hw=_hw(vram_gb=24))
    assert layers == -1 and why


# --------------------------------------------------------------------------- #
# quant selection
# --------------------------------------------------------------------------- #
def test_quant_of():
    assert quant_of("model-Q4_K_M.gguf") == "Q4_K_M"
    assert quant_of("model.q8_0.gguf") == "Q8_0"
    assert quant_of("model-IQ4_XS.gguf") == "IQ4_XS"
    assert quant_of("model.gguf") == ""


def test_group_split_ggufs():
    files = [{"name": "big-Q4_K_M-00001-of-00002.gguf", "size": 10},
             {"name": "big-Q4_K_M-00002-of-00002.gguf", "size": 5},
             {"name": "small-Q2_K.gguf", "size": 3},
             {"name": "README.md", "size": 1}]
    out = group_split_ggufs(files)
    by_name = {g["name"]: g for g in out}
    assert by_name["big-Q4_K_M.gguf"]["size"] == 15
    assert len(by_name["big-Q4_K_M.gguf"]["parts"]) == 2
    assert by_name["small-Q2_K.gguf"]["parts"] == ["small-Q2_K.gguf"]


def _files(**sizes_gb):
    return [{"name": f"m-{q}.gguf", "size": int(s * GiB)}
            for q, s in sizes_gb.items()]


def test_pick_quant_prefers_largest_that_fits_vram():
    files = _files(Q8_0=8, Q6_K=6, Q4_K_M=4, Q2_K=2)
    # 8 GiB free: Q8_0 (8 GiB) can't fit with KV + margin, Q6_K (6 GiB) can.
    c = pick_quant(files, _hw(vram_gb=8), n_ctx=4096)
    assert quant_of(c["name"]) == "Q6_K"


def test_pick_quant_falls_back_to_ram():
    # CPU path is capped at 50% of available RAM: Q4_K_M (20 GiB) needs 48 GiB free.
    files = _files(Q8_0=30, Q4_K_M=20, Q2_K=10)
    c = pick_quant(files, _hw(vram_gb=8, ram_gb=48), n_ctx=4096)
    assert quant_of(c["name"]) == "Q4_K_M"
    c2 = pick_quant(files, _hw(vram_gb=8, ram_gb=28), n_ctx=4096)
    assert quant_of(c2["name"]) == "Q2_K"


def test_pick_quant_smallest_when_everything_is_too_big():
    files = _files(Q8_0=300, Q2_K=200)
    c = pick_quant(files, _hw(vram_gb=8, ram_gb=16), n_ctx=4096)
    assert quant_of(c["name"]) == "Q2_K"
    assert "smallest" in c["why"]


def test_pick_quant_cpu_fallback_prefers_q4_not_q8():
    """A big Q8 must not be chosen when it would run mostly on CPU."""
    files = _files(Q8_0=8, Q6_K=6, Q4_K_M=4, Q2_K=2)
    c = pick_quant(files, _hw(vram_gb=2, ram_gb=32), n_ctx=4096)
    assert quant_of(c["name"]) == "Q4_K_M"
    assert "speed" in c["why"]
