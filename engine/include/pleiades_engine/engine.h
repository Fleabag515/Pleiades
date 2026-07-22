#pragma once

#include <string>

#include "llama.h"
#include "pleiades_engine/context_governor.h"
#include "pleiades_engine/model_manager.h"

namespace pleiades_engine {

struct GenerationResult {
    std::string text;
    int n_prompt_tokens = 0;
    int n_generated_tokens = 0;
    double prompt_seconds = 0.0;
    double generate_seconds = 0.0;
};

// Request-level facade: tokenize a prompt, decode it, then greedily sample
// completion tokens up to n_predict or until end-of-generation. Minimal by
// design for Phase 1 -- no chat template, no streaming, no tool calls; those
// arrive with `common/chat.h` integration in a later phase (see
// docs/specs/2026-07-21-native-inference-engine-design.md).
class Engine {
public:
    Engine(ModelManager& models, ContextGovernor& ctx);

    GenerationResult complete(const std::string& prompt, int n_predict = 128);

private:
    ModelManager& models_;
    ContextGovernor& ctx_;
};

}  // namespace pleiades_engine
