#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "llama.h"
#include "pleiades_engine/context_governor.h"
#include "pleiades_engine/model_manager.h"
#include "pleiades_engine/prefix_cache.h"

namespace pleiades_engine {

// Per-request sampling knobs, threaded from the HTTP shim's request body
// (temperature/top_p/top_k/min_p/repeat_penalty/seed/stop -- see
// http_server.cpp::handle_chat()) down into build_sampler_chain() below.
//
// The DEFAULTS here deliberately reproduce the pre-sampling-pass GREEDY
// engine byte for byte: temperature <= 0 selects a pure-argmax greedy
// sampler (exactly llama_sampler_init_greedy(), what engine.cpp built
// unconditionally before this struct existed) and every other field is a
// no-op at its default (top_k 0, top_p 1.0, min_p 0.0, penalties 1.0/0.0).
// So a default-constructed SamplingParams -- what cli_main.cpp, the bench
// binaries, and the deterministic prefix-cache/engine tests all pass -- is
// still deterministic greedy decoding, and their byte-identical assertions
// keep holding. Only the HTTP chat path constructs a SamplingParams with
// real values (its OWN defaults match llama-cpp-python's create_chat_completion
// for parity with pleiades/inference/server.py -- temp 0.2, top_p 0.95,
// top_k 40, min_p 0.05 -- see handle_chat()).
struct SamplingParams {
    float temperature = 0.0f;         // <= 0 => greedy argmax (ignores the truncation knobs below)
    float top_p = 1.0f;               // 1.0 => disabled
    int top_k = 0;                    // <= 0 => disabled
    float min_p = 0.0f;               // 0.0 => disabled
    float typical_p = 1.0f;           // 1.0 => disabled
    float repeat_penalty = 1.0f;      // 1.0 => disabled
    float presence_penalty = 0.0f;    // 0.0 => disabled
    float frequency_penalty = 0.0f;   // 0.0 => disabled
    // Window of most-recent tokens the penalties sampler considers. Matches
    // llama-cpp-python's own last_n_tokens_size default (64) so a request
    // that sets only repeat_penalty behaves the same on both engines.
    int penalty_last_n = 64;
    // LLAMA_DEFAULT_SEED (0xFFFFFFFF) => a fresh random seed per request,
    // matching llama-cpp-python's seed=None default.
    uint32_t seed = LLAMA_DEFAULT_SEED;
    // Stop strings: generation halts (finish_reason "stop") as soon as one
    // appears in the output, and the stop text itself is trimmed from the
    // returned/streamed content -- mirrors llama-cpp-python's `stop`.
    std::vector<std::string> stop;
};

struct GenerationResult {
    std::string text;
    int n_prompt_tokens = 0;
    int n_generated_tokens = 0;
    // How many leading prompt tokens were served from the KV prefix cache
    // (Phase 6) instead of being re-decoded. n_prompt_tokens -
    // n_prompt_cached is the number actually pushed through llama_decode
    // this call. 0 on a cold/first request or right after a resize.
    int n_prompt_cached = 0;
    // True iff generation halted because one of SamplingParams::stop was
    // produced (as opposed to hitting EOG or n_predict). The HTTP shim maps
    // this to finish_reason "stop" either way, but it's exposed for tests.
    bool stopped_on_stop_string = false;
    double prompt_seconds = 0.0;
    double generate_seconds = 0.0;
};

// Request-level facade: tokenize a prompt, decode it, then sample completion
// tokens up to n_predict or until end-of-generation, per the SamplingParams
// (greedy by default -- see that struct). No chat template applied here --
// callers (e.g. the Phase 3 HTTP shim) format the prompt themselves; see
// pleiades_engine::ChatTemplates (chat_template.h) for the real per-model
// jinja templating the HTTP shim renders every request through.
class Engine {
public:
    Engine(ModelManager& models, ContextGovernor& ctx);

    // Non-streaming: runs to completion (n_predict tokens, EOG, or a stop
    // string) and returns the full text + stats in one shot.
    GenerationResult complete(const std::string& prompt, int n_predict = 128,
                               const SamplingParams& sampling = {});

    // Streaming: same generation, but invokes `on_token(piece)` as tokens are
    // produced so callers can forward real per-token latency instead of
    // buffering the whole completion (used by Phase 3's SSE
    // /v1/chat/completions handler). Returning false from `on_token` stops
    // generation early (e.g. client disconnected mid-stream). `complete()` is
    // just `generate()` with no callback.
    //
    // When SamplingParams::stop is EMPTY (the default, and every existing
    // caller), `on_token` fires exactly once per generated token with that
    // token's own piece -- unchanged from before. When stop strings ARE
    // configured, emission is held back at token boundaries that could be the
    // start of a stop string, so `on_token` may instead receive
    // multi-token chunks and never the trailing stop text; the concatenation
    // of every chunk still equals GenerationResult::text.
    GenerationResult generate(const std::string& prompt, int n_predict = 128,
                               const std::function<bool(const std::string&)>& on_token = nullptr,
                               const SamplingParams& sampling = {});

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
