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

    // Create the first context at n_ctx against `model`. n_ctx_max is the
    // elastic ceiling (never resize beyond it); 0 means "no ceiling beyond
    // the model's own trained context". `params` carries the non-n_ctx
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

    llama_context* ctx() const { return ctx_; }
    int n_ctx() const { return n_ctx_; }
    int n_ctx_max() const { return n_ctx_max_; }
    const ContextParams& params() const { return params_; }

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

    void create_ctx(int n_ctx);
};

}  // namespace pleiades_engine
