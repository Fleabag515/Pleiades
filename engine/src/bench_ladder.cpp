// Phase 4: full-gear-ladder sweep. Walks CONTEXT_GEARS sequentially,
// timing each resize step and running a short completion at every rung to
// capture tok/s (prefill+decode) and process memory footprint (VmRSS) as
// n_ctx grows -- complements Phase 2's single-jump benchmark with the
// full-ladder picture the design doc's Phase 4 goal calls for.
//
// Usage: pleiades-engine-bench-ladder <model.gguf> [n_gpu_layers] [start_ctx]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include "llama.h"
#include "pleiades_engine/context_governor.h"
#include "pleiades_engine/engine.h"
#include "pleiades_engine/model_manager.h"

namespace {
double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

long vm_rss_kb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            return std::atol(line.c_str() + 6);
        }
    }
    return -1;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf> [n_gpu_layers] [start_ctx]\n", argv[0]);
        return 2;
    }
    std::string model_path = argv[1];
    int n_gpu_layers = argc > 2 ? std::atoi(argv[2]) : 0;
    int start_ctx = argc > 3 ? std::atoi(argv[3]) : pleiades_engine::CONTEXT_GEARS.front();
    const std::string prompt = "The capital of France is";

    llama_backend_init();

    try {
        auto t0 = std::chrono::steady_clock::now();
        pleiades_engine::ModelManager models;
        models.load(model_path, n_gpu_layers);
        std::printf("model load: %.1f ms, VmRSS=%ld MB\n", ms_since(t0), vm_rss_kb() / 1024);

        pleiades_engine::ContextGovernor ctx;
        int ceiling = llama_model_n_ctx_train(models.model());
        ctx.create(models.model(), start_ctx, ceiling);
        pleiades_engine::Engine engine(models, ctx);

        std::printf("%-10s %-12s %-14s %-12s %-14s %-10s\n", "n_ctx", "resize_ms", "prompt_tok/s", "decode_tok/s",
                    "n_generated", "VmRSS_MB");

        bool first = true;
        for (int gear : pleiades_engine::CONTEXT_GEARS) {
            if (gear < start_ctx || gear > ceiling) continue;
            double resize_ms = 0.0;
            if (!first) {
                auto tr = std::chrono::steady_clock::now();
                ctx.resize(gear);
                resize_ms = ms_since(tr);
            }
            first = false;

            auto r = engine.complete(prompt, 24);
            double prompt_tps = r.prompt_seconds > 0 ? r.n_prompt_tokens / r.prompt_seconds : 0.0;
            double decode_tps = r.generate_seconds > 0 ? r.n_generated_tokens / r.generate_seconds : 0.0;
            std::printf("%-10d %-12.1f %-14.1f %-12.1f %-14d %-10ld\n", ctx.n_ctx(), resize_ms, prompt_tps, decode_tps,
                        r.n_generated_tokens, vm_rss_kb() / 1024);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        llama_backend_free();
        return 1;
    }

    llama_backend_free();
    return 0;
}
