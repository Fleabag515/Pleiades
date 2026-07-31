#include "pleiades_engine/compaction_plan.h"

#include <stdexcept>
#include <string>

namespace pleiades_engine {

CompactionResult compact(const llama_model* model, llama_memory_t mem, llama_seq_id seq_id,
                          ResidentMap& resident_map, size_t evict_start, size_t evict_end,
                          bool fa_maybe_on) {
    if (evict_end <= evict_start || evict_end > resident_map.size()) {
        return {false, "empty or out-of-range span"};
    }
    // REQUIRED, not a nice-to-have -- see this function's own header comment
    // for why llama_memory_can_shift() does not catch this case on its own.
    if (llama_model_is_recurrent(model) || llama_model_is_hybrid(model)) {
        return {false, "recurrent/hybrid models are not supported yet -- "
                       "the recurrent memory can silently no-op a mid-sequence "
                       "eviction (see design doc's 'second finding')"};
    }
    // EMPIRICALLY CONFIRMED data leak, not a theoretical precaution -- see
    // this function's own header comment. No cheap mitigation exists here
    // (unlike generate()'s equivalent FA hazard), so refuse outright.
    if (fa_maybe_on) {
        return {false, "flash attention may be active -- compaction is not "
                       "safe under FA on this build yet (confirmed data leak "
                       "from evicted cells, see compaction_plan.h)"};
    }
    if (!llama_memory_can_shift(mem)) {
        return {false, "this cache type cannot shift positions"};
    }

    const llama_pos p0 = static_cast<llama_pos>(evict_start);
    const llama_pos p1 = static_cast<llama_pos>(evict_end);
    const llama_pos pos_max_before = llama_memory_seq_pos_max(mem, seq_id);

    if (!llama_memory_seq_rm(mem, seq_id, p0, p1)) {
        // Cache refused the partial removal (e.g. SWA) -- state is
        // untouched (seq_rm is all-or-nothing per its own contract), so it
        // is safe to just report the refusal.
        return {false, "cache refused partial removal (e.g. SWA)"};
    }
    llama_memory_seq_add(mem, seq_id, p1, -1, -(p1 - p0));
    resident_map.evict_span(evict_start, evict_end - evict_start);

    // Sync assertion: the new high-water mark must be exactly the old one
    // minus the evicted span's length. A mismatch here means the KV and
    // resident_map have silently diverged -- every future reusable_prefix()
    // call would then be reasoning about a sequence that no longer matches
    // what the KV actually holds, corrupting generation with no other
    // signal. Fail loud rather than let that happen quietly -- and leave
    // both sides in a matching, safe (if unhelpful) empty state before
    // doing so, same scrub-then-throw pattern Engine::generate() uses for
    // its own decode-failure paths (see engine.cpp), rather than throwing
    // from a state that's already half-applied and still silently diverged
    // for whatever caller ends up catching this.
    const llama_pos pos_max_after = llama_memory_seq_pos_max(mem, seq_id);
    const llama_pos expected = pos_max_before - (p1 - p0);
    if (pos_max_after != expected) {
        llama_memory_clear(mem, /*data=*/true);
        resident_map.invalidate(resident_map.epoch());
        throw std::runtime_error(
            "pleiades_engine: compaction desynced KV position (expected max=" +
            std::to_string(expected) + " got=" + std::to_string(pos_max_after) + ")");
    }
    return {true, ""};
}

}  // namespace pleiades_engine
