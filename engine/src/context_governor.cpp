#include "pleiades_engine/context_governor.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace pleiades_engine {

ContextGovernor::~ContextGovernor() {
    if (ctx_) {
        llama_free(ctx_);
    }
}

void ContextGovernor::create_ctx(int n_ctx) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = static_cast<uint32_t>(n_ctx);

    // Context-param parity knobs (see docs/specs/2026-07-21-native-
    // inference-engine-design.md's GPU/context-parity pass). params_ is
    // set once in create() and reapplied verbatim on every resize() so a
    // resize can't silently drop these back to library defaults.
    params.n_batch = static_cast<uint32_t>(params_.n_batch);
    params.n_ubatch = static_cast<uint32_t>(params_.n_ubatch);
    if (params_.n_threads > 0) {
        // Matches common/arg.cpp's -t/--threads handler: n_threads and
        // n_threads_batch both follow the same override (there's no
        // separate governor-level knob for threads-batch yet).
        params.n_threads = params_.n_threads;
        params.n_threads_batch = params_.n_threads;
    }
    params.flash_attn_type = params_.flash_attn_type;
    params.type_k = params_.type_k;
    params.type_v = params_.type_v;

    ctx_ = llama_init_from_model(model_, params);
    if (!ctx_) {
        throw std::runtime_error(
            "pleiades_engine: failed to create llama_context at n_ctx=" + std::to_string(n_ctx));
    }
    n_ctx_ = n_ctx;
    // Every fresh context starts with an empty KV -- bump the epoch so any
    // prefix cache tied to the previous context invalidates itself (see
    // context_governor.h::epoch()).
    ++epoch_;
}

void ContextGovernor::create(llama_model* model, int n_ctx, int n_ctx_max, const ContextParams& params) {
    model_ = model;
    n_ctx_max_ = n_ctx_max;
    params_ = params;
    create_ctx(n_ctx);
}

int ContextGovernor::clamp_gear(int requested) const {
    int train_ctx = model_ ? llama_model_n_ctx_train(model_) : 0;
    int ceiling = n_ctx_max_ ? n_ctx_max_ : (train_ctx ? train_ctx : CONTEXT_GEARS.back());
    return std::max(1024, std::min(requested, ceiling));
}

std::optional<int> ContextGovernor::next_gear() const {
    for (int g : CONTEXT_GEARS) {
        if (g > n_ctx_ && (n_ctx_max_ == 0 || g <= n_ctx_max_)) {
            return g;
        }
    }
    return std::nullopt;
}

int ContextGovernor::resize(int requested) {
    int target = clamp_gear(requested);
    if (target == n_ctx_) {
        return n_ctx_;
    }
    // Free-BEFORE-create is deliberate: the grow path must never need the old
    // and new KV resident at once, or growing on a memory-tight box would fail
    // even when the new size alone fits. The price is paid below: if the new
    // context can't be created (llama_init_from_model returns null on any
    // failed allocation -- a routine outcome when stepping up the gear ladder
    // on low-VRAM hardware, not a can't-happen), recreate at the size that was
    // live a moment ago. Its memory was just freed, so that near-certainly
    // succeeds -- leaving the governor serviceable at the old n_ctx instead of
    // holding a null ctx_ that still reports the old size and turns every
    // subsequent decode into a process-killing null deref.
    int prev = n_ctx_;
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    try {
        create_ctx(target);
    } catch (...) {
        // prev == 0 means there was no prior working size (an earlier double
        // failure); skip rollback then -- llama treats n_ctx=0 as "use
        // n_ctx_train", which would silently desync n_ctx_ from the real
        // context. The rollback context starts with an empty KV; create_ctx
        // bumps epoch_ on success, which is what invalidates any prefix cache
        // tied to the freed context.
        if (prev > 0) {
            try {
                create_ctx(prev);
            } catch (...) {
                // Even the previously-working size failed (e.g. another
                // process grabbed the memory in between). No live context:
                // report 0, never a stale size a caller could act on.
                n_ctx_ = 0;
            }
        }
        throw;  // surface the original resize failure whether or not rollback worked
    }
    return n_ctx_;
}

}  // namespace pleiades_engine
