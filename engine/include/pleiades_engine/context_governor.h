#pragma once

#include <array>
#include <cstdint>
#include <optional>

#include "llama.h"

namespace pleiades_engine {

// Gear ladder + resize policy ported verbatim from pleiades/hardware.py:633
// (CONTEXT_GEARS) and pleiades/inference/server.py's clamp_gear()/
// next_gear() -- see docs/specs/2026-07-21-native-inference-engine-design.md,
// Phase 1. Keep this in sync with the Python elastic engine; they must agree
// on the same gears so bench/parity comparisons in later phases are valid.
inline constexpr std::array<int, 6> CONTEXT_GEARS = {4096, 8192, 16384, 32768, 65536, 131072};

// Context-param parity knobs beyond n_ctx (llama_context_params fields;
// see include/llama.h). Defaults here match llama_context_default_params()'s
// own defaults exactly (src/llama-context.cpp), so a default-constructed
// ContextParams reproduces this engine's pre-parity-pass behavior byte for
// byte: only n_ctx was ever set explicitly before this struct existed.
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
    // Phase E (elastic context) — KV placement: when true, the K/V cache
    // lives in host RAM (llama_context_params::offload_kqv = false) while
    // compute stays on the GPU. The spill lever: the governor flips it
    // automatically when a requested growth would exceed the GPU KV budget,
    // so the context keeps growing (slower decode) instead of truncating.
    bool kv_on_cpu = false;
    // Phase E — elastic RoPE: past the model's trained context, engage YaRN
    // automatically (power-of-two factor buckets, logged). false forbids
    // growth beyond n_ctx_train (the pre-Phase-E behavior).
    bool auto_yarn = true;
};

// Owns the live llama_context*. Resize = free + recreate. This is safe
// because Pleiades' memory is turn-based via Anamnesis, not KV-cache-based
// (see Phase 0 finding in the design doc) -- a resize does not need to
// preserve KV/session state, exactly like today's Python elastic engine's
// EngineState.resize().
class ContextGovernor {
public:
    ContextGovernor() = default;
    ~ContextGovernor();

    ContextGovernor(const ContextGovernor&) = delete;
    ContextGovernor& operator=(const ContextGovernor&) = delete;

    // Create the first context at n_ctx against `model`. Phase E semantics:
    // n_ctx_max > 0 is a HARD PIN (never grow past it — what an explicit
    // --ctx-max user asked for); 0 means UNBOUNDED — growth is governed by
    // the KV budgets, not by a context number. `params` carries the non-n_ctx
    // context knobs (flash attention, KV cache quantization, n_ubatch/
    // n_batch, thread count) and is retained for the lifetime of this
    // governor so every subsequent resize() re-creates the context with
    // the SAME settings, not silently reverting to library defaults.
    void create(llama_model* model, int n_ctx, int n_ctx_max = 0, const ContextParams& params = {});

    // Free the current context and recreate at `requested` (clamped via
    // clamp_gear), reapplying the ContextParams passed to create(). Returns
    // the actual n_ctx used. If the new context can't be created (routine
    // when growing on a memory-tight box) this throws, but first rolls back
    // by recreating at the previous n_ctx -- the governor stays serviceable
    // at the old size rather than holding a null context. Only if that
    // rollback ALSO fails is there no live context afterward: then ctx() is
    // nullptr and n_ctx() reports 0 (never a stale size), and a later
    // successful resize() recovers.
    int resize(int requested);

    int clamp_gear(int requested) const;
    std::optional<int> next_gear() const;
    // Phase E — smallest gear >= `need`, doubling past the ladder forever
    // (the context has no fixed ceiling unless a hard --ctx-max pin is set).
    // Throws when a pin exists and `need` exceeds it.
    int next_gear_for(size_t need) const;

    // Phase E — KV budgets in bytes; 0 disables the check. The server
    // measures free device VRAM at boot for the auto GPU budget when no
    // --kv-budget-mib is given; the CPU budget defaults to 8192 MiB
    // (--kv-cpu-budget-mib 0 = unlimited).
    void set_budgets(size_t kv_budget_bytes, size_t cpu_budget_bytes);

    // Phase E — structural K/V cost of ONE context token, from the model's
    // GGUF geometry (n_layer x n_kv_head x head_dims x per-element size —
    // the C++ twin of hardware.py::kv_bytes_per_token). Budget checks price
    // a target gear before growing. 0 when unknown.
    size_t kv_bytes_per_token() const;

    // Phase E — outcome of a KV-preserving growth.
    struct GrowResult {
        int n_ctx = 0;
        bool preserved = false;  // sequence state survived (no re-prefill needed)
        bool spilled = false;    // this growth moved the K/V cache to host RAM
    };

    // Phase E — grow to `requested` (clamped) PRESERVING the sequence state:
    // snapshot (llama_state_seq_get_data) -> recreate -> restore. The caller
    // (Engine) re-binds its ResidentMap when `preserved` is true. Enforces
    // the KV budgets: growth past the GPU KV budget spills the cache to host
    // RAM (offload_kqv=false); growth past the RAM budget throws an honest
    // error instead of truncating. A YaRN factor-bucket change invalidates
    // the cached K/V (RoPE changes) and is reported as preserved=false.
    // Rolls back like resize() if the new context cannot be created.
    GrowResult grow_preserving(int requested);

    llama_context* ctx() const { return ctx_; }
    int n_ctx() const { return n_ctx_; }
    int n_ctx_max() const { return n_ctx_max_; }
    const ContextParams& params() const { return params_; }
    bool kv_on_cpu() const { return kv_on_cpu_active_; }
    float yarn_factor() const { return yarn_factor_; }

    // Monotonic counter incremented every time a fresh llama_context is
    // created (create() and each resize()). Consumers that cache anything
    // tied to the LIVE context's KV memory (e.g. Engine's prefix cache)
    // compare this against the epoch they last saw and drop that cache when
    // it changes, because resize() = llama_free + recreate throws the KV
    // away. This is deliberately NOT "did the llama_context* pointer
    // change" -- freed-then-immediately-reallocated memory routinely comes
    // back at the SAME address (see the Phase 4 test-bug writeup in the
    // design doc), so pointer identity is not a reliable "was the KV
    // recreated" signal; a monotonic counter is.
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
