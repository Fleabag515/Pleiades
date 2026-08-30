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

## Phase 2 — elastic, unbounded, VRAM-frugal context (next)

- KV-preserving grow on upshift: `llama_state_seq_get_data` → recreate ctx at
  the next gear → `set_data` (no re-prefill; the same primitive the slot file
  uses). Auto-upshift when a request's prompt + headroom exceeds n_ctx.
- No fixed ceiling: drop the `n_ctx_train` cap; engage YaRN automatically past
  trained length (computed scale/factor, logged, opt-out). Ceiling becomes the
  VRAM+RAM budget.
- Launch at the smallest gear that fits; grow per turn. KV q8_0 default with FA.
- Spill, never truncate: past the VRAM budget recreate the ctx with KV on CPU
  (`no-kv-offload` semantics); only a RAM-budget breach errors, honestly.
- Failure-path note: llama.cpp currently GGML_ABORTs on allocation failure —
  install an abort callback that throws so resize/grow failures become
  catchable 5xx/downshifts instead of process death.

## Phase 3 — ship as default on NVIDIA + AMD

CMake: HIP/ROCm + Vulkan alongside CUDA; backend autodetect; `/props` reports
backend + free VRAM. CI builds per-platform binaries; `pleiades runtime install`
fetches them. AMD verified on Ion's RX 9070 XT (standing rules). Then flip the
default: engine (eligible models) → llama-server → python elastic.

## Phase 4 — efficiency program

Bench harness (extend bench_ladder): TTFT/prefill/decode/VRAM-at-ctx/KV-reuse
savings on Anamnesis-realistic prompts vs llama-server, ollama, ik_llama.cpp,
vLLM. Optimizations with before/after numbers: expert prefetch + pinned-host
uploads, CUDA graphs, thread/ub calibration, per-slot radix prefix cache.

## Phase 5 — Colibrì-style expert streaming (flagship only)

When autofit finds no fitting placement: dense + KV + hot experts in VRAM,
warm in RAM, cold streamed from NVMe (LRU hot-set, async readahead,
--no-repack semantics so mmap stays lazy). Opt-in per model in the desktop UI,
with down-quant offered as the alternative. Doubles as the generalized
cross-architecture MoE offload the submodule rebase note promised.
