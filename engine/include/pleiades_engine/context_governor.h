#pragma once

#include <array>
#include <cstdint>
#include <optional>

#include "llama.h"

namespace pleiades_engine {

// Gear ladder ported verbatim from pleiades/hardware.py:633 (CONTEXT_GEARS).
// Phase 2 semantics: this is the STARTING ladder only — beyond its top the
// governor keeps doubling, so the context has NO fixed ceiling. The ceiling
// is a machine budget (GPU KV until --kv-budget-mib, then host RAM until
// --kv-cpu-budget-mib), never a context number; only an explicit
// --ctx-max pin (n_ctx_max > 0) hard-caps growth.
inline constexpr std::array<int, 6> CONTEXT_GEARS = {4096, 8192, 16384, 32768, 65536, 131072};

// Context-param parity knobs beyond n_ctx (llama_context_params fields;
// see include/llama.h). Defaults here match llama_context_default_params()'s
// own defaults exactly (src/llama-context.cpp), so a default-constructed
// ContextParams reproduces this engine's pre-Phase-2 behavior byte for byte.
//
// n_threads == 0 here means "leave llama_context_default_params()'s own
// GGML_DEFAULT_N_THREADS default alone" (0 is not a valid real thread
// count, so it's safe to use as the sentinel) rather than a real override.
struct ContextParams {
    int n_batch = 2048;
    int n_ubatch = 512;
    int n_threads = 0;
    llama_flash_attn_type flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    ggml_type type_k = GGML_TYPE_F16;
    ggml_type type_v = GGML_TYPE_F16;
    // Phase 2 — KV placement: when true, the K/V cache lives in host RAM
    // (llama_context_params::offload_kqv = false) while compute stays on the
    // GPU. This is the spill lever: the governor flips it automatically when
    // a requested growth would exceed the GPU KV budget, so the context keeps
    // growing (slower decode) instead of truncating or dying.
    bool kv_on_cpu = false;
    // Phase 2 — elastic RoPE: past the model's trained context, engage YaRN
    // automatically (power-of-two factor buckets, logged). Set false to
    // forbid growth beyond n_ctx_train (the pre-Phase-2 behavior).
    bool auto_yarn = true;
};

class ContextGovernor {
public:
    // Outcome of a KV-preserving growth.
    struct GrowResult {
        int n_ctx = 0;
        bool preserved = false;  // sequence state survived (KV re-bound, no re-prefill)
        bool spilled = false;    // this growth moved the K/V cache to host RAM
    };

    ContextGovernor() = default;
    ~ContextGovernor();

    ContextGovernor(const ContextGovernor&) = delete;
    ContextGovernor& operator=(const ContextGovernor&) = delete;

    // Create the first context at n_ctx against `model`. n_ctx_max > 0 is a
    // HARD PIN (never grow past it — what an explicit --ctx/--ctx-max user
    // asked for); 0 means unbounded: growth is governed by the KV budgets,
    // not by a context number. `params` carries the non-n_ctx context knobs
    // and is retained for the lifetime of this governor so every resize()
    // re-creates the context with the SAME settings.
    void create(llama_model* model, int n_ctx, int n_ctx_max = 0, const ContextParams& params = {});

    // Phase 2 — grow to `requested` (clamped) PRESERVING the sequence state:
    // snapshot (llama_state_seq_get_data) -> recreate -> restore. No
    // re-prefill, no epoch-driven cache loss on the Engine side (the caller
    // re-binds its PrefixCache when `preserved` is true). Enforces the KV
    // budgets: growth past the GPU KV budget spills the cache to host RAM;
    // growth past the RAM budget throws. A YaRN factor-bucket change invalidates
    // the cached K/V (RoPE changes), reported as preserved=false — the token
    // list still stands, so the next request cold-decodes correctly.
    GrowResult grow_preserving(int requested);

    // Legacy KV-LOSSY resize (free + recreate) — the explicit /resize
    // endpoint's semantics. Engine-internal growth uses grow_preserving().
    int resize(int requested);

    int clamp_gear(int requested) const;
    std::optional<int> next_gear() const;
    // Smallest gear >= `need` (doubling past the ladder). Throws when a hard
    // pin (n_ctx_max) is set and `need` exceeds it — the honest "prompt
    // exceeds the pinned context" case.
    int next_gear_for(size_t need) const;

    // KV budgets in bytes; 0 disables the check (KV on GPU then relies on
    // allocation failures; KV on CPU is unbounded). The server measures free
    // device VRAM at boot for the auto GPU budget when no --kv-budget-mib is
    // given.
    void set_budgets(size_t kv_budget_bytes, size_t cpu_budget_bytes);

    // Structural K/V cost of ONE context token in bytes, from the model's
    // GGUF geometry (n_layer x n_kv_head x head_dims x element size — the
    // C++ twin of hardware.py::kv_bytes_per_token). Budget checks use it to
    // price a target gear before growing. 0 when unknown.
    size_t kv_bytes_per_token() const;

    llama_context* ctx() const { return ctx_; }
    int n_ctx() const { return n_ctx_; }
    int n_ctx_max() const { return n_ctx_max_; }
    const ContextParams& params() const { return params_; }
    bool kv_on_cpu() const { return kv_on_cpu_active_; }
    float yarn_factor() const { return yarn_factor_; }

    // Monotonic counter incremented every time a fresh llama_context is
    // created (create() and each resize()/grow()). Consumers that cache
    // anything tied to the LIVE context's KV memory (e.g. Engine's prefix
    // cache) compare this against the epoch they last saw and drop that
    // cache when it changes — unless the grow was KV-preserving, in which
    // case the caller re-binds to the new epoch instead.
    uint64_t epoch() const { return epoch_; }

private:
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    int n_ctx_ = 0;
    int n_ctx_max_ = 0;
    uint64_t epoch_ = 0;
    ContextParams params_;
    bool kv_on_cpu_active_ = false;  // mirrors params_.kv_on_cpu (mutable by spill)
    float yarn_factor_ = 1.0f;
    size_t kv_budget_ = 0;
    size_t cpu_budget_ = 0;

    void create_ctx(int n_ctx);
};

}  // namespace pleiades_engine
