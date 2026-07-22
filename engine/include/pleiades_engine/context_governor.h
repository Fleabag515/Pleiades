#pragma once

#include <array>
#include <optional>

#include "llama.h"

namespace pleiades_engine {

// Gear ladder + resize policy ported verbatim from pleiades/hardware.py:633
// (CONTEXT_GEARS) and pleiades/inference/server.py's clamp_gear()/
// next_gear() -- see docs/specs/2026-07-21-native-inference-engine-design.md,
// Phase 1. Keep this in sync with the Python elastic engine; they must agree
// on the same gears so bench/parity comparisons in later phases are valid.
inline constexpr std::array<int, 6> CONTEXT_GEARS = {4096, 8192, 16384, 32768, 65536, 131072};

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
    // the model's own trained context".
    void create(llama_model* model, int n_ctx, int n_ctx_max = 0);

    // Free the current context and recreate at `requested` (clamped via
    // clamp_gear). Returns the actual n_ctx used.
    int resize(int requested);

    int clamp_gear(int requested) const;
    std::optional<int> next_gear() const;

    llama_context* ctx() const { return ctx_; }
    int n_ctx() const { return n_ctx_; }
    int n_ctx_max() const { return n_ctx_max_; }

private:
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    int n_ctx_ = 0;
    int n_ctx_max_ = 0;

    void create_ctx(int n_ctx);
};

}  // namespace pleiades_engine
