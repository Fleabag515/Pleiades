#pragma once

#include <functional>
#include <string>

#include "llama.h"
#include "pleiades_engine/context_governor.h"
#include "pleiades_engine/model_manager.h"
#include "pleiades_engine/prefix_cache.h"

namespace pleiades_engine {

struct GenerationResult {
    std::string text;
    int n_prompt_tokens = 0;
    int n_generated_tokens = 0;
    // How many leading prompt tokens were served from the KV prefix cache
    // (Phase 6) instead of being re-decoded. n_prompt_tokens -
    // n_prompt_cached is the number actually pushed through llama_decode
    // this call. 0 on a cold/first request or right after a resize.
    int n_prompt_cached = 0;
    double prompt_seconds = 0.0;
    double generate_seconds = 0.0;
};

// Request-level facade: tokenize a prompt, decode it, then greedily sample
// completion tokens up to n_predict or until end-of-generation. No chat
// template applied here -- callers (e.g. the Phase 3 HTTP shim) format the
// prompt themselves; see pleiades_engine::format_chatml() for the current
// stopgap formatter. No tool calls yet either.
class Engine {
public:
    Engine(ModelManager& models, ContextGovernor& ctx);

    // Non-streaming: runs to completion (n_predict tokens or EOG) and
    // returns the full text + stats in one shot.
    GenerationResult complete(const std::string& prompt, int n_predict = 128);

    // Streaming: same generation, but invokes `on_token(piece)` once per
    // token as it's produced (before decoding the next one) so callers can
    // forward real per-token latency instead of buffering the whole
    // completion (used by Phase 3's SSE /v1/chat/completions handler).
    // Returning false from `on_token` stops generation early (e.g. client
    // disconnected mid-stream). `complete()` is just `generate()` with no
    // callback.
    GenerationResult generate(const std::string& prompt, int n_predict = 128,
                               const std::function<bool(const std::string&)>& on_token = nullptr);

    // Drop the KV prefix cache and hard-clear the live context's KV for
    // sequence 0. Normally unnecessary (generate() self-manages the cache),
    // but exposed for callers that want an explicit fresh start and for
    // tests. Safe to call before the first generate().
    void reset_prefix_cache();

    // Prefix-cache introspection (used by tests and benchmarks).
    const PrefixCache& prefix_cache() const { return prefix_; }

private:
    ModelManager& models_;
    ContextGovernor& ctx_;
    PrefixCache prefix_;
};

}  // namespace pleiades_engine
