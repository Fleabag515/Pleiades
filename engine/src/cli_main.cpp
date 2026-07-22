// Phase 1 smoke test for the native inference engine skeleton.
//
// Usage: pleiades-engine-cli <model.gguf> "<prompt>" [n_predict]
//
// Loads a GGUF via ModelManager, creates a context via ContextGovernor,
// runs one completion via Engine, and prints the result plus tokens/sec.
// See docs/specs/2026-07-21-native-inference-engine-design.md, Phase 1.
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

#include "llama.h"
#include "pleiades_engine/context_governor.h"
#include "pleiades_engine/engine.h"
#include "pleiades_engine/model_manager.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> \"<prompt>\" [n_predict]\n", argv[0]);
        return 2;
    }
    std::string model_path = argv[1];
    std::string prompt = argv[2];
    int n_predict = argc > 3 ? std::atoi(argv[3]) : 64;

    llama_backend_init();

    try {
        pleiades_engine::ModelManager models;
        models.load(model_path, /*n_gpu_layers=*/0);

        pleiades_engine::ContextGovernor ctx;
        ctx.create(models.model(), /*n_ctx=*/4096);

        pleiades_engine::Engine engine(models, ctx);
        auto result = engine.complete(prompt, n_predict);

        std::printf("--- prompt ---\n%s\n", prompt.c_str());
        std::printf("--- completion ---\n%s\n", result.text.c_str());
        std::printf("--- stats ---\n");
        std::printf("prompt tokens:     %d (%.3fs, %.1f tok/s)\n", result.n_prompt_tokens,
                     result.prompt_seconds,
                     result.prompt_seconds > 0 ? result.n_prompt_tokens / result.prompt_seconds : 0.0);
        std::printf("generated tokens:  %d (%.3fs, %.1f tok/s)\n", result.n_generated_tokens,
                     result.generate_seconds,
                     result.generate_seconds > 0 ? result.n_generated_tokens / result.generate_seconds : 0.0);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        llama_backend_free();
        return 1;
    }

    llama_backend_free();
    return 0;
}
