#include "pleiades_engine/context_governor.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

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

    // Phase E — KV placement: offload_kqv=false keeps the K/V cache in host
    // RAM while compute stays on the GPU (the spill configuration).
    params.offload_kqv = !kv_on_cpu_active_;

    // Phase E — elastic RoPE: past the trained context, engage YaRN with a
    // power-of-two factor bucket (factor = smallest pow2 with
    // factor * n_ctx_train >= n_ctx; rope_freq_scale = 1/factor — the same
    // recipe llama-server's --rope-scaling yarn applies). Within a bucket
    // the scaling is IDENTICAL across gears, so cached K/V stays valid when
    // growing 8k -> 16k -> 32k; entering the next bucket changes RoPE and
    // invalidates the cache (grow_preserving reports that honestly).
    const int train = model_ ? llama_model_n_ctx_train(model_) : 0;
    yarn_factor_ = 1.0f;
    if (params_.auto_yarn && train > 0 && n_ctx > train) {
        float factor = 1.0f;
        while (static_cast<double>(factor) * static_cast<double>(train) <
               static_cast<double>(n_ctx)) {
            factor *= 2.0f;
        }
        params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN;
        params.yarn_orig_ctx = static_cast<uint32_t>(train);
        params.rope_freq_scale = 1.0f / factor;
        yarn_factor_ = factor;
        std::fprintf(stderr,
                     "[pleiades-engine] context %d > trained %d: YaRN engaged (factor %.0fx, orig_ctx %d)\n",
                     n_ctx, train, factor, train);
    }

    ctx_ = llama_init_from_model(model_, params);
    if (!ctx_) {
        throw std::runtime_error(
            "pleiades_engine: failed to create llama_context at n_ctx=" + std::to_string(n_ctx) +
            (kv_on_cpu_active_ ? " (KV on CPU)" : ""));
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
    kv_on_cpu_active_ = params.kv_on_cpu;
    create_ctx(n_ctx);
}

int ContextGovernor::clamp_gear(int requested) const {
    // Phase E: the ceiling is a HARD PIN only (--ctx-max). With no pin the
    // context is unbounded — growth is governed by the KV budgets, and past
    // n_ctx_train auto-YaRN extends the positional range (create_ctx).
    int g = std::max(1024, requested);
    if (n_ctx_max_ > 0) {
        g = std::min(g, n_ctx_max_);
    }
    return g;
}

std::optional<int> ContextGovernor::next_gear() const {
    for (int g : CONTEXT_GEARS) {
        if (g > n_ctx_ && (n_ctx_max_ == 0 || g <= n_ctx_max_)) {
            return g;
        }
    }
    // Phase E: past the ladder, keep doubling (unbounded when no pin).
    int g = n_ctx_;
    while (g <= n_ctx_) {
        g *= 2;
        if (n_ctx_max_ > 0 && g > n_ctx_max_) {
            return std::nullopt;
        }
    }
    return g;
}

int ContextGovernor::next_gear_for(size_t need) const {
    for (int g : CONTEXT_GEARS) {
        if (static_cast<size_t>(g) >= need && (n_ctx_max_ == 0 || g <= n_ctx_max_)) {
            return g;
        }
    }
    int g = CONTEXT_GEARS.back();
    while (static_cast<size_t>(g) < need) {
        g *= 2;
        if (n_ctx_max_ > 0 && g > n_ctx_max_) {
            g = n_ctx_max_;
            break;
        }
    }
    if (n_ctx_max_ > 0 && need > static_cast<size_t>(n_ctx_max_)) {
        throw std::runtime_error(
            "prompt needs " + std::to_string(need) + " context tokens but the context is pinned at " +
            std::to_string(n_ctx_max_) + " (--ctx-max). Drop the pin or shorten the request.");
    }
    return g;
}

size_t ContextGovernor::kv_bytes_per_token() const {
    // Structural K/V cost of one context token: n_layer x n_kv_head x
    // (key_length + value_length) x per-element bytes, GQA-aware — the same
    // formula hardware.py::kv_bytes_per_token uses on the Python side. The
    // geometry comes from the GGUF metadata (arch-namespaced keys, read
    // arch-agnostically by suffix). Best-effort: a missing key falls back to
    // the hidden size (over-estimates for GQA models — the safe direction
    // for a spill decision).
    if (!model_) {
        return 0;
    }
    auto meta_long = [&](const char* suffix) -> long {
        char key[128];
        char buf[64];
        for (uint32_t i = 0; i < llama_model_meta_count(model_); ++i) {
            if (llama_model_meta_key_by_index(model_, i, key, sizeof(key)) < 0) {
                continue;
            }
            const char* dot = std::strstr(key, suffix);
            if (dot && std::strlen(dot) == std::strlen(suffix)) {
                if (llama_model_meta_val_str(model_, key, buf, sizeof(buf)) >= 0) {
                    return std::strtol(buf, nullptr, 10);
                }
            }
        }
        return -1;
    };
    const long n_layer = llama_model_n_layer(model_);
    const long n_embd = llama_model_n_embd(model_);
    long kv_heads = meta_long(".attention.head_count_kv");
    long key_len = meta_long(".attention.key_length");
    long val_len = meta_long(".attention.value_length");
    if (n_layer <= 0 || n_embd <= 0) {
        return 0;
    }
    if (kv_heads < 0 || key_len < 0 || val_len < 0) {
        // No GQA metadata: assume full-width K/V (n_head = n_embd / head_dim
        // unknown -> use n_embd per layer per side; conservative).
        kv_heads = 1;
        key_len = val_len = n_embd;
    }
    // Per-ELEMENT bytes: ggml_type_size() returns the BLOCK size (e.g. 34
    // bytes per 32-element Q8_0 block), so divide by the block size. Using
    // the raw block size over-estimated the K/V cost ~32x and triggered
    // false budget exhaustion.
    const size_t bytes =
        static_cast<size_t>(ggml_type_size(params_.type_k) / std::max<size_t>(1, ggml_blck_size(params_.type_k))) +
        static_cast<size_t>(ggml_type_size(params_.type_v) / std::max<size_t>(1, ggml_blck_size(params_.type_v)));
    return static_cast<size_t>(n_layer) * static_cast<size_t>(kv_heads) *
           static_cast<size_t>(key_len + val_len) * bytes;
}

void ContextGovernor::set_budgets(size_t kv_budget_bytes, size_t cpu_budget_bytes) {
    kv_budget_ = kv_budget_bytes;
    cpu_budget_ = cpu_budget_bytes;
}

ContextGovernor::GrowResult ContextGovernor::grow_preserving(int requested) {
    int target = clamp_gear(requested);
    GrowResult r;
    r.n_ctx = target;
    r.preserved = false;
    if (target == n_ctx_) {
        r.preserved = true;
        return r;
    }

    // Budget check BEFORE touching anything, with a STRUCTURAL estimate: the
    // K/V cost of the TARGET context straight from the model geometry. A
    // residency-based estimate (current state size extrapolated) reads zero
    // on a fresh context and would never spill the first, often largest,
    // growth.
    const double est = static_cast<double>(kv_bytes_per_token()) * static_cast<double>(target);
    const double est_mib = est / (1024.0 * 1024.0);
    if (!kv_on_cpu_active_ && kv_budget_ && est > static_cast<double>(kv_budget_)) {
        // GPU KV budget exhausted: spill. The cache moves to host RAM; decode
        // slows (host bandwidth) but the context KEEPS GROWING.
        params_.kv_on_cpu = true;
        kv_on_cpu_active_ = true;
        r.spilled = true;
        std::fprintf(stderr,
                     "[pleiades-engine] KV spill: growing to %d needs ~%.1f MiB of K/V (budget %.1f MiB) — "
                     "K/V cache moves to CPU RAM\n",
                     target, est_mib, kv_budget_ / (1024.0 * 1024.0));
    }
    if (kv_on_cpu_active_ && cpu_budget_ && est > static_cast<double>(cpu_budget_)) {
        // The context was NOT recreated yet -- roll the spill flag back so
        // the governor's state stays consistent with the live (GPU-KV)
        // context, then report honestly: the model's context cannot grow
        // further on this machine.
        kv_on_cpu_active_ = params_.kv_on_cpu;
        throw std::runtime_error(
            "context budget exhausted: growing to " + std::to_string(target) + " tokens needs ~" +
            std::to_string(static_cast<int>(est_mib)) +
            " MiB of K/V cache but the CPU KV budget is " +
            std::to_string(cpu_budget_ / (1024 * 1024)) +
            " MiB. This model's context cannot grow further on this machine.");
    }

    // Snapshot the sequence state (logits + embedding + KV) BEFORE freeing
    // the old context, then restore into the new one.
    const size_t state_sz = ctx_ ? llama_state_seq_get_size(ctx_, /*seq_id=*/0) : 0;
    std::vector<uint8_t> state;
    if (ctx_ && state_sz > 0) {
        state.resize(state_sz);
        if (llama_state_seq_get_data(ctx_, state.data(), state_sz, /*seq_id=*/0) != state_sz) {
            state.clear();  // snapshot failed -> the growth just won't preserve
        }
    }

    const float yarn_before = yarn_factor_;
    const int prev = n_ctx_;
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    try {
        create_ctx(target);
    } catch (...) {
        // Same rollback discipline as resize(): the previously-working size
        // was just freed, so recreating it near-certainly succeeds. The
        // snapshot is still valid afterwards, but the epoch changed, so the
        // caller must re-bind (or invalidate) its ResidentMap either way.
        r.spilled = false;
        if (prev > 0) {
            try {
                create_ctx(prev);
            } catch (...) {
                n_ctx_ = 0;
            }
        }
        throw;
    }
    r.n_ctx = n_ctx_;

    if (yarn_factor_ != yarn_before) {
        // RoPE changed (crossed a YaRN bucket): the restored K/V would be
        // scaled differently than it was computed. Honest answer: the token
        // list stands, the KV is cold — one re-prefill, then elasticity
        // continues inside the new bucket.
        return r;
    }
    if (!state.empty() &&
        llama_state_seq_set_data(ctx_, state.data(), state_sz, /*seq_id=*/0) == state_sz) {
        r.preserved = true;
    }
    return r;
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
