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
    ctx_ = llama_init_from_model(model_, params);
    if (!ctx_) {
        throw std::runtime_error(
            "pleiades_engine: failed to create llama_context at n_ctx=" + std::to_string(n_ctx));
    }
    n_ctx_ = n_ctx;
}

void ContextGovernor::create(llama_model* model, int n_ctx, int n_ctx_max) {
    model_ = model;
    n_ctx_max_ = n_ctx_max;
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
