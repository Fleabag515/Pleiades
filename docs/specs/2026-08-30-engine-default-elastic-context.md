# Pleiades engine: default runtime, elastic context, cross-vendor, efficiency, expert streaming

Status: Phase 1 LANDED (this doc + commits). Owner directive: the from-scratch C++
engine becomes the default runtime; context grows elastically instead of hitting a
fixed ceiling; NVIDIA + AMD; measurable best-in-class efficiency; Colibrì-style
expert streaming for flagship models only.

## Phase 1 — parity so the engine CAN be default (done 2026-08-30)

Landed in `engine/` + `pleiades/launch.py` + `pleiades/runtime.py`:

1. **Chunk-level KV reuse** (`--cache-reuse N`): port of llama-server's
   n_cache_reuse loop over the unified-memory API (`llama_memory_seq_rm` gap +
   `llama_memory_seq_add` re-label). Proven exact at this pin: position shifts
   are corrected at attention time by the graph's k_shift input.
2. **Slot state save/restore** (`--slot-save-path` + `POST /slots/0?action=save|
   restore`): same HTTP contract models.py already drives around stop/start.
   Own "PLST" file format (state blob + resident token list so the PrefixCache
   can rebind). Verified live: save → restart → restore → 696/697 prompt
   tokens served from the restored KV.
3. **mmap/repack control** (`--no-mmap`, `--no-repack` =
   `llama_model_params::use_mmap` / `use_extra_bufts`): the foundation for
   expert streaming (repacking materializes weights and defeats mmap laziness,
   cf. upstream issue #19578).
4. **Request queue + busy reporting**: bounded queue (`--queue-depth`,
   `--queue-timeout-sec`), 429 + Retry-After instead of opaque stalls,
   `busy`/`queued` in `/props`.
5. **launch.py eligibility**: vision (mmproj) models fall back to llama-server
   wholesale; the engine branch now also wires `--slot-save-path` +
   `--cache-reuse` and reuses autofit placement (already did).

### Hard constraints discovered (they shape everything below)

- **llama_decode contiguity rule** (src/llama-batch.cpp): a batch's lowest
  position for a sequence must equal the cache's `seq_pos_max + 1`. The alive
  sequence must ALWAYS be a contiguous `[0, F)` prefix when decoding. This
  kills any "decode a gap, realign a chunk further down, decode again" scheme
  within one turn — chunk moves must always target the fill boundary, and the
  tail beyond it is freed before decoding (upstream's exact structure).
- **Insertion limit (shared with upstream)**: when the rewritten memory block
  GROWS, the shifted tail cannot be salvaged this turn. Real fix is on
  Anamnesis's side: pad the memory block to a fixed token budget (or
  fixed-size slots) so its length changes are rare and bounded — then
  realignment covers nearly every turn. Proposed as an Anamnesis integration
  task, not engine heroics.
- **FA stale-cell leak at this pin**: freeing cells (`seq_rm`) keeps stale
  K/V data; the FA kernel leaks masked cells into attention (empirically
  proven by the Phase 6 work on CPU and CUDA). Whenever the new prompt is
  shorter than the resident sequence and FA may be on, the engine scrubs and
  cold-decodes. Reuse under FA is safe when the prompt >= resident (every
  freed cell is overwritten). watch: an upstream fix lets the guard widen.
- `llama_memory_can_shift()` gates all shifts (recurrent/hybrid models
  cold-decode; matches upstream).

### Verification (2026-08-30, RTX 2080 Ti, Qwen2.5-1.5B Q4_K_M, ngl 999)

- ctest 10/10 (new: test_slot_state, test_cache_reuse — byte-identical greedy
  output vs cold reference on every shape).
- pytest 570 passed (incl. new launch eligibility/flag-wiring tests).
- Parity bench vs upstream llama-server (same pin, same placement):
  prefill 699 tok: 0.123–0.137 s (engine) vs 0.127 s (llama-server);
  decode 128 tok: 0.055–0.058 s vs 0.071 s. Parity or better.
- HTTP smoke: queue served a second request after the in-flight turn; slot
  save/restore roundtrip across process restart reused 696/697 tokens.

## Phase 2 — elastic, unbounded, VRAM-frugal context — LANDED 2026-08-30

- `ContextGovernor::grow_preserving()`: snapshot (`llama_state_seq_get_data`)
  → recreate at the target gear → restore. No re-prefill; the Engine re-binds
  its PrefixCache when the state survived (`Engine::grow_context_to` /
  `ensure_capacity`, called automatically at the top of every request).
- No fixed ceiling: `n_ctx_max == 0` (the default) is UNBOUNDED — gears double
  past the ladder forever (`next_gear_for`), growth is governed by KV budgets,
  not context numbers. `--ctx-max N` remains a hard pin for users who ask for
  one (a need past the pin throws an honest error).
- Auto-YaRN past `n_ctx_train`: power-of-two factor buckets (factor = smallest
  pow2 with factor*train >= n_ctx, `rope_freq_scale = 1/factor` — the same
  recipe llama-server's --rope-scaling yarn applies). Within a bucket growth
  is KV-preserving (scaling identical); crossing a bucket invalidates the
  cached K/V (RoPE changes) and is reported honestly — one re-prefill, then
  elasticity continues. `--no-auto-yarn` opts out (restores the pre-Phase-2
  train ceiling). KNOWN SUBSTRATE QUIRK: tiny-train toy models (stories15M,
  train=128) are yarn'd at every gear and degenerate — mechanics tests opt
  out; text-equality preservation cases run on a real large-train model.
- Launch at the smallest planned gear (launch.py now starts the engine at
  `n_ctx_v`, not the ceiling) with KV q8_0 by default (half the f16 KV
  footprint; requires FA, so "auto" resolves on for the KV types).
- Spill, never truncate: growth past the GPU KV budget (`--kv-budget-mib`,
  default = free device VRAM measured right after model load) recreates the
  context with `offload_kqv=false` — K/V in host RAM, compute on GPU, decode
  slower but the context KEEPS GROWING, and the state roundtrip preserves the
  conversation across the device move. Growth past the RAM budget
  (`--kv-cpu-budget-mib`, default 8192) throws an honest error instead of
  truncating. The budget check prices the target gear STRUCTURALLY
  (`ContextGovernor::kv_bytes_per_token()` — n_layer x n_kv_head x head_dims
  x per-element size from GGUF metadata; currently falls back to full-width
  K/V when the arch's GQA keys are not matched, over-estimating ~6x — spills
  early, never late; refine the key lookup later).
- Verified 2026-08-30: ctest 11/11 (new test_elastic_context: preservation
  within train on Llama-3.2-1B, auto-upshift byte-exact vs cold ref, YaRN
  bucket mechanics + honest invalidation on the toy fixture, spill + budget
  exhaustion); pytest 570 passed; live GPU smoke: boot at 1024 → long turn →
  grew to 4096 with the KV preserved across a CPU spill → completion served;
  /props reports {unbounded, yarn_factor, kv_on_cpu, ctx_pinned}.
- NOTED for later: llama.cpp still GGML_ABORTs on raw allocation failure —
  the budget system above is the guardrail that keeps growth off that path.

## Phase 3 — ship as default on NVIDIA + AMD

CMake: HIP/ROCm + Vulkan alongside CUDA; backend autodetect; `/props` reports
backend + free VRAM. CI builds per-platform binaries; `pleiades runtime install`
fetches them. AMD verified on Ion's RX 9070 XT (standing rules). Then flip the
default: engine (eligible models) → llama-server → python elastic.

## Phase 4 — efficiency program — harness LANDED 2026-09-05; optimization program open

`engine/bench/compare.sh`: boots the engines that can serve a GGUF on this
machine with an identical model/placement and reports wall-clock cold prefill
(1177-token prompt, 1 output) and decode (128 outputs) — the two numbers that
decide perceived single-user latency. Optional engines (ollama) are skipped
when absent. Measured on this box (Qwen2.5-1.5B-Instruct Q4_K_M):

| metric | pleiades engine | llama-server (same pin) | notes |
|---|---|---|---|
| GPU prefill 1177 tok | 0.123–0.137 s | 0.127 s | VRAM-free window, ngl 999 |
| GPU decode 128 tok | 0.055–0.058 s | 0.071 s | engine ~20% faster |
| CPU prefill 1177 tok | 0.0872 s | 0.0922 s | ngl 0, ctx 2048 |
| CPU decode 128 tok | 0.6516 s | 0.7034 s | engine ~7% faster |

(The engine links the same libllama kernels, so parity is the floor; the
deltas come from orchestration — thread defaults, cache policies. The GPU
rows were captured before a co-located training job claimed the VRAM; the
script re-runs both regimes.)

Still open under Phase 4: Anamnesis-realistic KV-reuse metrics (rewritten
memory blocks — needs Anamnesis-side fixed-size memory slots first, see the
Phase-E insertion limit), ik_llama.cpp / vLLM cross-engine rows, CUDA-graph
decode capture, expert prefetch + pinned-host uploads for MoE offload.

## Scope decision (2026-09-05, owner): LINUX ONLY

Windows and macOS support is dropped. Removed: the windows-latest CI matrix,
the Windows engine build + release asset, install.ps1, the desktop dist:win
script, and all install-doc advertising. Kept inert: runtime `os.name` code
branches and install.sh's darwin paths (deleting them buys nothing and risks
churn). The focus is HARDWARE BREADTH on Linux: CUDA (NVIDIA, sm_75→sm_120 via
CUDA 12.8), HIP/ROCm (AMD consumer gfx1100/1101/1102/1151/1200/1201 — built in
CI against AMD's ROCm 6.4.1 apt repo), Vulkan (universal fallback: NVIDIA, AMD,
Intel), CPU. The release fetcher's preference order is cuda → hip → vulkan →
cpu. install_engine_asset() previously shipped a windows-vulkan asset in
v0.1.5; future releases will not.

## Phase 5 — Colibrì-style expert streaming (flagship only)

When autofit finds no fitting placement: dense + KV + hot experts in VRAM,
warm in RAM, cold streamed from NVMe (LRU hot-set, async readahead,
--no-repack semantics so mmap stays lazy). Opt-in per model in the desktop UI,
with down-quant offered as the alternative. Doubles as the generalized
cross-architecture MoE offload the submodule rebase note promised.


---

## Integration note (2026-08-30 evening) — where this plan met the remote Phase 9 series

This doc's Phase 1/2/3 were first implemented on a local main that had diverged
from origin/main, where parallel sessions had independently landed their own
"Phase 9" series: the default flip (Phase 9.6), the backend matrix, slot
save/restore using llama's own session-file format, ResidentMap + compaction
replacing the naive prefix cache, engine vision via mtmd, and desktop bundling
of the engine. The locally-built line was preserved on branch
`backup/engine-phases-2026-08-30` (chunks: chunk-level KV reuse gate, PLST slot
format — superseded by the remote implementations); main was reset to
origin/main (be91d2d) and the UNIQUELY-missing piece — the Phase E elastic
context — was re-ported onto the remote engine:

- `grow_preserving` (state roundtrip; rollback discipline like their resize),
  unbounded gears (`next_gear_for`), auto-YaRN buckets, KV budgets +
  `offload_kqv` spill, structural `kv_bytes_per_token` from GGUF metadata.
- `Engine::ensure_capacity` / `grow_context_to` re-binding their ResidentMap.
- launch.py: engine branch launches at the smallest planned gear with q8_0 KV,
  passes `--ctx-max` (explicit pins) and `--kv-budget-mib` (GPU ceiling).
- `test_elastic_context` (fixture + large-train-ctx model; SKIPs when absent).
- `find_native_cpp_engine()` resolves `~/.pleiades/engine/`; `pleiades runtime
  build-engine` compiles + installs locally; engine-build CI workflow.

The owner's separate uncommitted WIP (tool-call replay for cloud brains, web
search resilience ladder, Anamnesis selector work, desktop cloud-model
settings) is snapshotted on `backup/owner-wip-2026-08-30` and still needs
per-file integration against the newer remote base.
