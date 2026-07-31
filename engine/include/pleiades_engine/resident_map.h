#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "llama.h"

namespace pleiades_engine {

// Tracks the token sequence currently resident in a llama_context's KV
// cache (sequence 0) so that a later request sharing a common prefix with
// it can skip re-decoding that prefix. One responsibility: bookkeeping the
// resident sequence + computing how many leading tokens a new prompt can
// reuse. It performs NO llama_* calls itself -- Engine owns the actual KV
// manipulation (llama_memory_seq_rm) and decode; ResidentMap only tells it
// how much to keep. That split keeps this class trivially unit-testable
// without a model (see tests/test_resident_map.cpp) and keeps the class as
// narrowly scoped as ContextGovernor/ModelManager.
//
// WHY THIS EXISTS (Phase 6): before this, Engine::generate() decoded the
// ENTIRE prompt via llama_batch_get_one() on every request with no KV
// bookkeeping at all. Because llama_batch_get_one() leaves batch.pos ==
// nullptr, llama.cpp auto-assigns each token's position as
// memory->seq_pos_max(seq)+1 -- i.e. it APPENDS after whatever is already
// resident. With nothing ever clearing the KV between requests, a repeated
// system/persona prefix was re-decoded at ever-growing positions, silently
// duplicating content in the KV cache and making turn 2+ attend over stale
// prior-request tokens. See the design doc's Phase 6 section.
//
// RENAMED FROM PrefixCache (Phase 4 "bonus lane" cascading-cache design,
// docs/specs/2026-07-21-context-free-model-architecture-design.md): the
// design's own sequencing note calls for folding PrefixCache into this
// class -- the future metadata-only home for cascade-eviction bookkeeping
// (ResidentMap + CascadeLadder + ImportanceScorer + RetentionBudget +
// CompactionPlan) -- as the FIRST, net-correctness-neutral step, before any
// real eviction policy exists. This pass is exactly that first step and
// nothing more: a rename with the class's behavior otherwise byte-for-byte
// identical to PrefixCache. The cascade itself (multi-segment tracking,
// eviction, the behavioral capability probe that gates it) is NOT
// implemented here -- this is still a single contiguous resident run, same
// as before. Do not assume any cascade machinery exists yet just because
// the name changed.
//
// The resident sequence is tagged with the ContextGovernor epoch it was
// captured under; a resize (free+recreate) bumps that epoch and empties the
// KV, so the cache is dropped rather than trusted against a different
// context.
class ResidentMap {
public:
    // Number of leading tokens of `prompt` that are already resident and
    // can be reused. This is the longest common prefix of the resident
    // sequence and `prompt`, but capped at prompt.size()-1: at least one
    // token must always be decoded fresh so the sampler has valid logits
    // for the new prompt's final token (the resident context's last logits
    // belong to whatever was decoded last -- usually a *generated* token
    // from the previous turn, not this prompt's final token).
    size_t reusable_prefix(const std::vector<llama_token>& prompt) const;

    // Replace the resident sequence wholesale (after decoding a prompt's
    // suffix, the resident sequence is exactly that prompt's tokens).
    void set(std::vector<llama_token> tokens, uint64_t epoch);

    // A single token was just decoded and is now resident -- extend the
    // tracked sequence by one so it keeps mirroring the KV exactly.
    void append(llama_token tok) { tokens_.push_back(tok); }

    // Phase 4 step 2 (docs/specs/2026-07-21-context-free-model-architecture-
    // design.md, "bonus lane"): remove `count` tokens starting at index
    // `start`, keeping this map in sync with a KV eviction applied via
    // compact() (compaction_plan.h). Every index into this vector IS the
    // corresponding token's KV position -- the invariant this whole class
    // exists to maintain -- so `start`/`start+count` here are simultaneously
    // valid vector indices and valid llama_pos values; compact() passes the
    // exact same numbers to both this call and the KV-side seq_rm/seq_add.
    // No bounds checking (matches this class's existing minimal style, e.g.
    // append() above) -- the caller (compact()) validates the span first.
    void evict_span(size_t start, size_t count) {
        tokens_.erase(tokens_.begin() + static_cast<std::ptrdiff_t>(start),
                      tokens_.begin() + static_cast<std::ptrdiff_t>(start + count));
    }

    // Drop everything and (re)bind to `epoch`. Called when the governor
    // epoch changes out from under us (a resize recreated the context) or
    // whenever the KV was hard-cleared.
    void invalidate(uint64_t epoch) {
        tokens_.clear();
        epoch_ = epoch;
    }

    uint64_t epoch() const { return epoch_; }
    size_t size() const { return tokens_.size(); }
    bool empty() const { return tokens_.empty(); }
    const std::vector<llama_token>& tokens() const { return tokens_; }

private:
    std::vector<llama_token> tokens_;
    uint64_t epoch_ = 0;
};

}  // namespace pleiades_engine
