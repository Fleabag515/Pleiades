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
// manipulation (llama_memory_seq_rm) and decode; PrefixCache only tells it
// how much to keep. That split keeps this class trivially unit-testable
// without a model (see tests/test_prefix_cache.cpp) and keeps the class as
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
// The resident sequence is tagged with the ContextGovernor epoch it was
// captured under; a resize (free+recreate) bumps that epoch and empties the
// KV, so the cache is dropped rather than trusted against a different
// context.
class PrefixCache {
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
