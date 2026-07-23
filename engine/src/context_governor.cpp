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
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    create_ctx(target);
    return n_ctx_;
}

}  // namespace pleiades_engine
