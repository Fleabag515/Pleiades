#pragma once

#include <cstddef>

#include "llama.h"
#include "pleiades_engine/resident_map.h"

namespace pleiades_engine {

// Phase 4 step 2 (docs/specs/2026-07-21-context-free-model-architecture-
// design.md, "bonus lane"): the actual KV-compaction mechanism. NULL POLICY
// -- nothing in this engine decides WHEN to call compact() yet (no
// CascadeLadder/ImportanceScorer/RetentionBudget exist). This is the
// mechanism proven correct in isolation first, per the design's own
// sequencing note, so a future policy layer has a trustworthy primitive to
// call rather than building policy and mechanism at the same time.
struct CompactionResult {
    bool applied = false;
    // Non-empty only when applied == false. Points to a string literal
    // (always valid, never owned) -- no allocation on the common-refusal
    // path.
    const char* reason = "";
};

// Evicts resident_map's tokens [evict_start, evict_end) -- both a token-
// VECTOR-INDEX range and, simultaneously, the identical llama_pos range in
// the live KV (see ResidentMap's own invariant) -- from both the KV
// (mem/seq_id) and resident_map's bookkeeping, in lockstep. Applying one
// without the other breaks the invariant ResidentMap's whole existence
// depends on (mirrors the KV exactly) -- that is the ONE bug class this
// function exists to make structurally impossible to introduce piecemeal.
//
// Mechanism, per the design doc's own "headroom finding": llama_memory_
// seq_rm() removes the evicted span's cells, then llama_memory_seq_add()
// shifts every position after it backward by the evicted span's length --
// this is llama.cpp's OWN position-dependent RoPE remap (verified against
// llama_kv_cache::build_graph_shift in the vendored third_party/llama.cpp --
// confirmed there to re-rotate each surviving cell'S data IN PLACE at its
// own physical slot, not move data between slots, so the evicted span's old
// physical cells become genuinely orphaned, not overwritten), so this
// function never computes a rotation itself; it only chooses the span and
// calls existing primitives correctly, then asserts the resulting KV high-
// water mark moved by exactly the evicted length (a desync here would
// silently corrupt every future decode against this sequence, so it throws
// rather than returning a result the caller might not check -- and scrubs
// + invalidates both sides first so the throw doesn't itself leave a
// half-applied, silently-diverged state behind, matching the scrub-then-
// throw pattern Engine::generate() uses for its own decode-failure paths).
//
// NOT internally synchronized. Nothing calls this concurrently today (it's
// null-policy -- only test_compaction_plan.cpp exercises it, single-
// threaded), but a future policy layer must serialize calls against
// Engine::generate()/complete() itself -- e.g. by holding the same mutex
// http_server.cpp's ServerState already uses to serialize chat/resize/slot
// requests (state.mu) -- before calling this from any other thread.
// ResidentMap's token vector and the live llama_context's KV are both
// unsynchronized mutable state; calling this concurrently with a decode is
// undefined behavior, not merely unsupported.
//
// Refuses (applied=false, no KV/map mutation) rather than corrupting state
// when: the span is empty or extends past what's actually resident; `model`
// reports itself recurrent or hybrid (llama_model_is_recurrent/is_hybrid)
// -- REQUIRED, not optional: traced through the vendored llama-memory-
// hybrid.cpp/llama-memory-recurrent.cpp and confirmed llama_memory_can_shift
// does NOT gate this case (llama_memory_hybrid::get_can_shift() delegates
// only to the attention sub-cache; llama_memory_recurrent::get_can_shift()
// unconditionally returns true), and llama_memory_recurrent::seq_rm()
// SILENTLY RETURNS true for a genuine mid-sequence span without actually
// removing anything from the recurrent state (only a suffix rollback or
// full clear does real work there) -- so without this explicit check,
// compact() would report success on a hybrid model while the DeltaNet-style
// recurrent layers stay permanently contaminated with the "evicted"
// tokens' influence, exactly the silent-misbehavior the design doc's
// "second finding" warns about, with no other signal catching it (the
// position-sync assertion below wouldn't fire either -- both sub-caches'
// position bookkeeping still moves consistently even when the recurrent
// side did nothing real); `fa_maybe_on` is true (see param doc below) --
// EMPIRICALLY CONFIRMED, not just suspected: a real dense fixture model,
// evicting a genuine mid-sequence span under flash attention, produced
// generation that diverged from a true truncation oracle
// (test_compaction_plan.cpp's flash-attention regression section). Root
// cause: seq_rm/seq_add never scrub the evicted span's cell DATA (only
// metadata/position), and this pinned llama.cpp's FLASH-ATTENTION kernel is
// already known (Phase 6, see engine.cpp's own comment on generate()'s
// stale-KV guard) to leak non-zero K/V from masked-but-not-scrubbed cells.
// Unlike generate()'s equivalent hazard, there is no cheap fix here: a full
// llama_memory_clear(data=true) would force a cold re-decode of the ENTIRE
// sequence, defeating compaction's entire purpose (why evict a middle span
// at all if the next call has to re-decode everything anyway?) -- so this
// refuses outright rather than silently producing wrong output OR silently
// costing more than it saves; or the cache type can't shift at all
// (llama_memory_can_shift, e.g. some future non-hybrid, non-shiftable
// cache type); or llama_memory_seq_rm itself refuses the removal (some
// cache types, e.g. SWA, can't truncate mid-sequence -- the same refusal
// Engine::generate() already handles by falling back to a full clear).
//
// `fa_maybe_on`: caller-supplied (compact() has no ContextParams of its
// own) -- pass `ctx_.params().flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED`,
// the exact same check Engine::generate() already uses for its own FA
// guard, so AUTO is conservatively treated as "could be on."
CompactionResult compact(const llama_model* model, llama_memory_t mem, llama_seq_id seq_id,
                          ResidentMap& resident_map, size_t evict_start, size_t evict_end,
                          bool fa_maybe_on);

}  // namespace pleiades_engine
